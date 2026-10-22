"""Semantic layer: load a YAML model and compile semantic queries into SQL.

A semantic query is a small JSON object:

    {
      "metrics":    ["revenue", "gross_margin"],
      "dimensions": ["region", "order_date__month"],
      "filters":    [{"field": "order_date", "operator": "between", "value": ["2025-01-01", "2025-12-31"]},
                     {"field": "revenue", "operator": ">", "value": 100000}],
      "order_by":   [{"field": "revenue", "direction": "desc"}],
      "limit":      10
    }

The compiler guarantees that every query uses the governed metric definitions,
joins only the tables it needs, and passes user-supplied values as bound
parameters rather than string-splicing them into SQL.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

GRAINS = ("day", "week", "month", "quarter", "year")
OPERATORS = ("=", "!=", ">", ">=", "<", "<=", "in", "not_in", "between", "contains")
AGGREGATIONS = {
    "sum": "SUM({})",
    "count": "COUNT({})",
    "count_distinct": "COUNT(DISTINCT {})",
    "avg": "AVG({})",
    "min": "MIN({})",
    "max": "MAX({})",
}
QUERY_KEYS = {"metrics", "dimensions", "filters", "order_by", "limit"}
MAX_LIMIT = 10_000
_METRIC_REF = re.compile(r"\{(\w+)\}")


class SemanticError(ValueError):
    """Raised for an invalid model or query. Messages are written to be shown to an LLM."""


@dataclass
class Entity:
    name: str
    table: str
    alias: str
    join: str
    via: str | None = None


@dataclass
class Dimension:
    name: str
    expr: str
    description: str = ""
    type: str = "categorical"
    grains: tuple[str, ...] = ()
    synonyms: tuple[str, ...] = ()


@dataclass
class Metric:
    name: str
    description: str = ""
    agg: str | None = None
    expr: str | None = None
    filter: str | None = None
    derived: str | None = None
    format: str | None = None
    synonyms: tuple[str, ...] = ()

    @property
    def is_derived(self) -> bool:
        return self.derived is not None


@dataclass
class CompiledQuery:
    sql: str
    params: list[Any]
    columns: list[str]
    tables: list[str] = field(default_factory=list)

    def rendered(self) -> str:
        """SQL with parameters inlined as literals. For display only; execute `sql` with `params`."""
        values = iter(self.params)

        def literal(_: re.Match) -> str:
            v = next(values)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return repr(v)
            return "'" + str(v).replace("'", "''") + "'"

        return re.sub(r"\?", literal, self.sql)


def _suggest(name: str, options: list[str]) -> str:
    close = difflib.get_close_matches(name, options, n=3, cutoff=0.5)
    return f" Did you mean: {', '.join(close)}?" if close else f" Valid options: {', '.join(sorted(options))}."


class SemanticModel:
    def __init__(self, spec: dict):
        self.name = spec.get("name", "model")
        self.description = " ".join(str(spec.get("description", "")).split())
        fact = spec["fact"]
        self.fact = Entity("fact", fact["table"], fact["alias"], join="")
        self.entities = {
            name: Entity(name, e["table"], e["alias"], e["join"], e.get("via"))
            for name, e in spec.get("entities", {}).items()
        }
        self.dimensions = {
            name: Dimension(name, d["expr"], d.get("description", ""), d.get("type", "categorical"),
                            tuple(d.get("grains", ())), tuple(d.get("synonyms", ())))
            for name, d in spec.get("dimensions", {}).items()
        }
        self.metrics = {
            name: Metric(name, m.get("description", ""), m.get("agg"), m.get("expr"), m.get("filter"),
                         m.get("derived"), m.get("format"), tuple(m.get("synonyms", ())))
            for name, m in spec.get("metrics", {}).items()
        }
        self._validate_model()

    @classmethod
    def load(cls, path: str | Path) -> "SemanticModel":
        with open(path) as f:
            return cls(yaml.safe_load(f))

    # ------------------------------------------------------------------ model validation

    def _validate_model(self) -> None:
        aliases = {self.fact.alias} | {e.alias for e in self.entities.values()}
        if len(aliases) != len(self.entities) + 1:
            raise SemanticError("Entity aliases must be unique.")
        for e in self.entities.values():
            if e.via and e.via not in self.entities:
                raise SemanticError(f"Entity '{e.name}' joins via unknown entity '{e.via}'.")
        clash = set(self.dimensions) & set(self.metrics)
        if clash:
            raise SemanticError(f"Names used for both a metric and a dimension: {sorted(clash)}")
        for d in self.dimensions.values():
            if d.type == "time" and not set(d.grains) <= set(GRAINS):
                raise SemanticError(f"Dimension '{d.name}' has invalid grains {d.grains}.")
            if "__" in d.name:
                raise SemanticError(f"Dimension names cannot contain '__': {d.name}")
        for m in self.metrics.values():
            if m.is_derived:
                for ref in _METRIC_REF.findall(m.derived):
                    if ref not in self.metrics:
                        raise SemanticError(f"Derived metric '{m.name}' references unknown metric '{ref}'.")
                self._base_metrics(m.name)  # raises on cycles
            elif m.agg not in AGGREGATIONS or not m.expr:
                raise SemanticError(f"Metric '{m.name}' needs an expr and an agg in {list(AGGREGATIONS)}.")

    # ------------------------------------------------------------------ catalog

    @property
    def dimension_names(self) -> list[str]:
        """Every addressable dimension name, including time grains like order_date__month."""
        names = []
        for d in self.dimensions.values():
            names.append(d.name)
            names.extend(f"{d.name}__{g}" for g in d.grains if g != "day")
        return names

    def catalog_text(self, dimension_values: dict[str, list[str]] | None = None) -> str:
        dimension_values = dimension_values or {}
        lines = ["METRICS"]
        for m in self.metrics.values():
            syn = f" (aka {', '.join(m.synonyms)})" if m.synonyms else ""
            lines.append(f"- {m.name}{syn}: {m.description}")
        lines.append("\nDIMENSIONS")
        for d in self.dimensions.values():
            syn = f" (aka {', '.join(d.synonyms)})" if d.synonyms else ""
            extra = ""
            if d.type == "time":
                grains = [g for g in d.grains if g != "day"]
                extra = f" Time dimension; grains: {', '.join(f'{d.name}__{g}' for g in grains)}."
            elif d.name in dimension_values:
                extra = f" Values: {', '.join(map(str, dimension_values[d.name]))}."
            lines.append(f"- {d.name}{syn}: {d.description}{extra}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ resolution helpers

    def _dimension_sql(self, name: str) -> str:
        base, _, grain = name.partition("__")
        dim = self.dimensions.get(base)
        if dim is None:
            raise SemanticError(f"Unknown dimension '{name}'." + _suggest(name, self.dimension_names))
        if not grain:
            return dim.expr
        if dim.type != "time":
            raise SemanticError(f"Dimension '{base}' is not a time dimension, so it has no grain '{grain}'.")
        if grain not in dim.grains:
            raise SemanticError(f"'{base}' supports grains {list(dim.grains)}, not '{grain}'.")
        if grain == "day":
            return dim.expr
        return f"CAST(date_trunc('{grain}', {dim.expr}) AS DATE)"

    def _is_time(self, name: str) -> bool:
        dim = self.dimensions.get(name.partition("__")[0])
        return dim is not None and dim.type == "time"

    def _base_metrics(self, name: str, stack: tuple[str, ...] = ()) -> set[str]:
        if name in stack:
            raise SemanticError(f"Circular metric definition: {' -> '.join(stack + (name,))}")
        m = self.metrics[name]
        if not m.is_derived:
            return {name}
        out: set[str] = set()
        for ref in _METRIC_REF.findall(m.derived):
            out |= self._base_metrics(ref, stack + (name,))
        return out

    def _metric_sql(self, name: str, pushed_filter: str | None) -> str:
        m = self.metrics[name]
        if m.is_derived:
            return "(" + _METRIC_REF.sub(lambda r: self._metric_sql(r.group(1), pushed_filter), m.derived) + ")"
        inner = m.expr
        if m.filter and m.filter != pushed_filter:
            inner = f"CASE WHEN {m.filter} THEN {m.expr} END"
        return AGGREGATIONS[m.agg].format(inner)

    def _check_metric(self, name: str) -> None:
        if name not in self.metrics:
            raise SemanticError(f"Unknown metric '{name}'." + _suggest(name, list(self.metrics)))

    # ------------------------------------------------------------------ compile

    def compile(self, query: dict) -> CompiledQuery:
        if not isinstance(query, dict):
            raise SemanticError("Query must be a JSON object.")
        unknown = set(query) - QUERY_KEYS
        if unknown:
            raise SemanticError(f"Unknown query keys {sorted(unknown)}. Allowed: {sorted(QUERY_KEYS)}.")

        metrics = list(dict.fromkeys(query.get("metrics") or []))
        dimensions = list(dict.fromkeys(query.get("dimensions") or []))
        filters = query.get("filters") or []
        order_by = query.get("order_by") or []
        limit = query.get("limit")

        if not metrics and not dimensions:
            raise SemanticError("A query needs at least one metric or dimension.")
        for m in metrics:
            self._check_metric(m)
        dim_sql = {d: self._dimension_sql(d) for d in dimensions}

        # Ordering by an unselected metric is allowed: we add it to the output.
        order_terms = []
        for o in order_by:
            if not isinstance(o, dict) or "field" not in o:
                raise SemanticError("Each order_by entry must look like {\"field\": ..., \"direction\": \"asc|desc\"}.")
            f, direction = o["field"], str(o.get("direction", "asc")).lower()
            if direction not in ("asc", "desc"):
                raise SemanticError(f"order_by direction must be 'asc' or 'desc', not '{direction}'.")
            if f in self.metrics:
                if f not in metrics:
                    metrics.append(f)
            elif f not in dim_sql:
                if f.partition("__")[0] in self.dimensions:
                    raise SemanticError(f"Cannot order by dimension '{f}' unless it is in 'dimensions'.")
                raise SemanticError(f"Unknown order_by field '{f}'."
                                    + _suggest(f, list(self.metrics) + dimensions))
            order_terms.append(f'"{f}" {direction.upper()}')
        if not order_terms:  # Default: chronological when a time dimension is present.
            order_terms = [f'"{d}" ASC' for d in dimensions if self._is_time(d)]

        where, having, params = [], [], []
        having_metrics = []
        for flt in filters:
            if not isinstance(flt, dict) or not {"field", "operator"} <= set(flt):
                raise SemanticError("Each filter must look like {\"field\": ..., \"operator\": ..., \"value\": ...}.")
            f, op, value = flt["field"], str(flt["operator"]).lower(), flt.get("value")
            if op not in OPERATORS:
                raise SemanticError(f"Unknown operator '{op}'. Use one of {list(OPERATORS)}.")
            if f in self.metrics:
                if op in ("contains",):
                    raise SemanticError("Metric filters only support comparison operators.")
                having_metrics.append(f)
                having.append((f, op, value))
            else:
                expr = self._dimension_sql(f)
                clause, vals = self._predicate(expr, op, value, cast_date=self._is_time(f))
                where.append(clause)
                params.extend(vals)

        # Push a shared metric filter into WHERE when every metric agrees on it.
        all_metric_names = metrics + having_metrics
        bases = set().union(*(self._base_metrics(m) for m in all_metric_names)) if all_metric_names else set()
        base_filters = {self.metrics[b].filter for b in bases}
        pushed = base_filters.pop() if len(base_filters) == 1 else None
        if pushed:
            where.insert(0, pushed)

        having_sql, having_params = [], []
        for f, op, value in having:
            clause, vals = self._predicate(self._metric_sql(f, pushed), op, value, cast_date=False)
            having_sql.append(clause)
            having_params.extend(vals)

        select = [f'{sql} AS "{d}"' for d, sql in dim_sql.items()]
        select += [f'{self._metric_sql(m, pushed)} AS "{m}"' for m in metrics]

        used_sql = " ".join([*dim_sql.values(), *where, *(c for c in having_sql)]
                            + [self._metric_sql(m, None) for m in all_metric_names]
                            + [self.metrics[b].filter or "" for b in bases])
        joins = self._joins_for(used_sql)

        lines = ["SELECT", "  " + ",\n  ".join(select), f"FROM {self.fact.table} AS {self.fact.alias}"]
        lines += [f"JOIN {e.table} AS {e.alias} ON {e.join}" for e in joins]
        if where:
            lines.append("WHERE " + "\n  AND ".join(where))
        if dimensions and metrics:
            lines.append("GROUP BY " + ", ".join(str(i + 1) for i in range(len(dimensions))))
        elif dimensions:
            lines[0] = "SELECT DISTINCT"
        if having_sql:
            lines.append("HAVING " + "\n  AND ".join(having_sql))
        if order_terms:
            lines.append("ORDER BY " + ", ".join(order_terms))
        if limit is not None:
            if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_LIMIT:
                raise SemanticError(f"limit must be an integer between 1 and {MAX_LIMIT}.")
            lines.append(f"LIMIT {limit}")

        return CompiledQuery(
            sql="\n".join(lines),
            params=params + having_params,
            columns=dimensions + metrics,
            tables=[self.fact.table] + [e.table for e in joins],
        )

    def _joins_for(self, sql: str) -> list[Entity]:
        needed = {name for name, e in self.entities.items() if re.search(rf"\b{e.alias}\.", sql)}
        for name in list(needed):  # pull in intermediate entities
            via = self.entities[name].via
            while via:
                needed.add(via)
                via = self.entities[via].via
        ordered: list[Entity] = []
        remaining = [e for n, e in self.entities.items() if n in needed]
        while remaining:  # emit each entity after the one it joins through
            progressed = False
            for e in list(remaining):
                if e.via is None or any(o.name == e.via for o in ordered):
                    ordered.append(e)
                    remaining.remove(e)
                    progressed = True
            if not progressed:
                raise SemanticError("Could not resolve join order.")
        return ordered

    @staticmethod
    def _predicate(expr: str, op: str, value: Any, cast_date: bool) -> tuple[str, list[Any]]:
        def norm(v: Any) -> Any:
            if isinstance(v, (date, datetime)):
                return v.isoformat()
            if isinstance(v, (dict, list)):
                raise SemanticError(f"Filter value {v!r} must be a scalar.")
            if cast_date:
                try:
                    date.fromisoformat(str(v))
                except ValueError:
                    raise SemanticError(f"Time filters need ISO dates like '2025-01-31', got {v!r}.") from None
            return v

        ph = "CAST(? AS DATE)" if cast_date else "?"
        if op in ("in", "not_in"):
            if not isinstance(value, list) or not value:
                raise SemanticError(f"Operator '{op}' needs a non-empty list value.")
            sql_op = "IN" if op == "in" else "NOT IN"
            return f"{expr} {sql_op} ({', '.join([ph] * len(value))})", [norm(v) for v in value]
        if op == "between":
            if not isinstance(value, list) or len(value) != 2:
                raise SemanticError("Operator 'between' needs a two-element list value [low, high] (inclusive).")
            return f"{expr} BETWEEN {ph} AND {ph}", [norm(value[0]), norm(value[1])]
        if value is None or isinstance(value, list):
            raise SemanticError(f"Operator '{op}' needs a single value.")
        if op == "contains":
            return f"{expr} ILIKE '%' || ? || '%'", [str(value)]
        sql_op = "<>" if op == "!=" else op
        return f"{expr} {sql_op} {ph}", [norm(value)]
