"""Claude agents that answer questions about the warehouse.

SemanticAgent  - may only query through the semantic layer (the product).
SQLAgent       - writes raw SQL against the schema (the baseline to beat).

Both run the same tool-use loop, so evals compare the interface, not the plumbing.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import duckdb

from .semantic import OPERATORS, SemanticError, SemanticModel

DEFAULT_MODEL = os.environ.get("TTW_MODEL", "claude-sonnet-5-5")
ROWS_SHOWN_TO_MODEL = 50
CATEGORICAL_VALUES_IN_PROMPT = 25


def to_jsonable(v: Any) -> Any:
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return v


def execute(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> tuple[list[str], list[tuple]]:
    cur = con.execute(sql, params or [])
    return [d[0] for d in cur.description], cur.fetchall()


@dataclass
class QueryRecord:
    tool: str
    input: dict
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    error: str | None = None


@dataclass
class AgentResult:
    question: str
    answer: str
    records: list[QueryRecord]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    turns: int = 0

    @property
    def final(self) -> QueryRecord | None:
        """The last successful data query; this is what evals score."""
        for r in reversed(self.records):
            if r.error is None and r.tool in ("query_semantic_layer", "run_sql"):
                return r
        return None


class BaseAgent:
    tools: list[dict]

    def __init__(self, con: duckdb.DuckDBPyConnection, client: Any = None, model: str = DEFAULT_MODEL,
                 max_turns: int = 8, max_tokens: int = 2048):
        self.con = con
        self.model = model
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        lo, hi = con.execute("SELECT MIN(order_date), MAX(order_date) FROM orders").fetchone()
        self.data_range = (lo.isoformat(), hi.isoformat())

    def system_prompt(self) -> str:
        raise NotImplementedError

    def handle_tool(self, name: str, args: dict) -> tuple[dict, QueryRecord | None]:
        raise NotImplementedError

    def _result_payload(self, sql: str, columns: list[str], rows: list[tuple]) -> dict:
        return {
            "sql": sql,
            "columns": columns,
            "row_count": len(rows),
            "rows": [[to_jsonable(v) for v in r] for r in rows[:ROWS_SHOWN_TO_MODEL]],
            "truncated": len(rows) > ROWS_SHOWN_TO_MODEL,
        }

    def ask(self, question: str) -> AgentResult:
        start = time.perf_counter()
        system = self.system_prompt()
        messages: list[dict] = [{"role": "user", "content": question}]
        result = AgentResult(question=question, answer="", records=[])

        for turn in range(1, self.max_turns + 1):
            response = self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=system,
                tools=self.tools, messages=messages,
            )
            result.turns = turn
            usage = getattr(response, "usage", None)
            if usage is not None:
                result.input_tokens += getattr(usage, "input_tokens", 0) or 0
                result.output_tokens += getattr(usage, "output_tokens", 0) or 0

            messages.append({"role": "assistant", "content": response.content})
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                result.answer = "".join(b.text for b in response.content if b.type == "text").strip()
                break

            tool_results = []
            for block in tool_uses:
                payload, record = self.handle_tool(block.name, dict(block.input or {}))
                if record is not None:
                    result.records.append(record)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(payload, default=str),
                    "is_error": "error" in payload,
                })
            messages.append({"role": "user", "content": tool_results})
        else:
            result.answer = "(Stopped: reached the maximum number of tool calls without a final answer.)"

        result.latency_s = time.perf_counter() - start
        return result


class SemanticAgent(BaseAgent):
    """Answers questions using only the governed semantic layer."""

    def __init__(self, con: duckdb.DuckDBPyConnection, model_def: SemanticModel, **kwargs: Any):
        super().__init__(con, **kwargs)
        self.semantic = model_def
        self.dimension_values = self._low_cardinality_values()
        self.tools = self._tool_specs()

    def _low_cardinality_values(self) -> dict[str, list[str]]:
        """Put small value lists in the prompt so the model rarely needs a lookup call."""
        out = {}
        for d in self.semantic.dimensions.values():
            if d.type == "time":
                continue
            values = self._distinct(d.name, limit=CATEGORICAL_VALUES_IN_PROMPT + 1)
            if len(values) <= CATEGORICAL_VALUES_IN_PROMPT:
                out[d.name] = values
        return out

    def _distinct(self, dimension: str, limit: int, search: str | None = None) -> list[str]:
        q = self.semantic.compile({
            "dimensions": [dimension],
            "filters": [{"field": dimension, "operator": "contains", "value": search}] if search else [],
            "order_by": [{"field": dimension, "direction": "asc"}],
            "limit": limit,
        })
        return [to_jsonable(r[0]) for r in self.con.execute(q.sql, q.params).fetchall()]

    def _tool_specs(self) -> list[dict]:
        metric_names = list(self.semantic.metrics)
        dimension_names = self.semantic.dimension_names
        categorical = [d.name for d in self.semantic.dimensions.values() if d.type != "time"]
        return [
            {
                "name": "query_semantic_layer",
                "description": (
                    "Run a query against the governed semantic layer and get rows back. Choose metrics and "
                    "dimensions from the catalog; the layer writes the SQL. Returns the compiled SQL and up to "
                    f"{ROWS_SHOWN_TO_MODEL} rows. Errors explain what to fix."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "metrics": {"type": "array", "items": {"type": "string", "enum": metric_names}},
                        "dimensions": {"type": "array", "items": {"type": "string", "enum": dimension_names}},
                        "filters": {
                            "type": "array",
                            "description": (
                                "Dimension filters become WHERE; metric filters become HAVING. Dates are ISO "
                                "strings. 'between' takes [low, high] inclusive; 'in'/'not_in' take a list; "
                                "'contains' is a case-insensitive substring match."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "field": {"type": "string", "enum": dimension_names + metric_names},
                                    "operator": {"type": "string", "enum": list(OPERATORS)},
                                    "value": {"description": "A string, number, or list, depending on the operator."},
                                },
                                "required": ["field", "operator", "value"],
                            },
                        },
                        "order_by": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "field": {"type": "string"},
                                    "direction": {"type": "string", "enum": ["asc", "desc"]},
                                },
                                "required": ["field"],
                            },
                        },
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10000},
                    },
                },
            },
            {
                "name": "get_dimension_values",
                "description": "List the distinct values of a categorical dimension, optionally filtered by a substring.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "dimension": {"type": "string", "enum": categorical},
                        "search": {"type": "string"},
                    },
                    "required": ["dimension"],
                },
            },
        ]

    def system_prompt(self) -> str:
        lo, hi = self.data_range
        return f"""You are a data analyst for an {self.semantic.description}

Data covers orders from {lo} to {hi}. Interpret relative dates ("this year", "last quarter") relative to {hi}.

You answer questions by querying a governed semantic layer with the query_semantic_layer tool. You never write SQL. Metrics already encode business rules (for example, revenue counts completed orders only). Use them as defined and do not try to re-derive them from other metrics.

How to build queries:
- Pick metrics and dimensions from the catalog below. Time dimensions take a grain suffix, e.g. order_date__month.
- Filter dates on the base time dimension with ISO dates, e.g. {{"field": "order_date", "operator": "between", "value": ["2025-01-01", "2025-12-31"]}}.
- Use exact dimension values. If unsure of a value, call get_dimension_values first.
- For "top N" or "which X had the most" questions, use order_by with a limit.
- For thresholds on aggregates ("categories with revenue over $1M"), filter on the metric.
- Prefer one query that answers the whole question. If the catalog cannot answer the question, say so plainly instead of approximating.

Answer format: lead with the direct answer and key numbers, keep it brief, and mention a metric definition when it changes how the number should be read (e.g. "revenue excludes returns"). Format fractions like 0.42 as percentages.

CATALOG
{self.semantic.catalog_text(self.dimension_values)}"""

    def handle_tool(self, name: str, args: dict) -> tuple[dict, QueryRecord | None]:
        if name == "get_dimension_values":
            dim = args.get("dimension", "")
            try:
                values = self._distinct(dim, limit=200, search=args.get("search"))
                return {"dimension": dim, "values": values}, None
            except (SemanticError, duckdb.Error) as e:
                return {"error": str(e)}, None

        if name == "query_semantic_layer":
            record = QueryRecord(tool=name, input=args)
            try:
                compiled = self.semantic.compile(args)
                record.sql = compiled.rendered()
                record.columns, record.rows = execute(self.con, compiled.sql, compiled.params)
                return self._result_payload(record.sql, record.columns, record.rows), record
            except (SemanticError, duckdb.Error) as e:
                record.error = str(e)
                return {"error": record.error}, record

        return {"error": f"Unknown tool '{name}'."}, None


class SQLAgent(BaseAgent):
    """Baseline: the model writes raw SQL against the documented schema."""

    tools = [{
        "name": "run_sql",
        "description": f"Run a read-only DuckDB SQL query and get up to {ROWS_SHOWN_TO_MODEL} rows back.",
        "input_schema": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "A single SELECT (or WITH ... SELECT) statement."}},
            "required": ["sql"],
        },
    }]

    def schema_text(self) -> str:
        rows = self.con.execute("""
            SELECT table_name, column_name, data_type FROM information_schema.columns
            WHERE table_schema = 'main' ORDER BY table_name, ordinal_position
        """).fetchall()
        tables: dict[str, list[str]] = {}
        for t, c, dt in rows:
            tables.setdefault(t, []).append(f"{c} {dt}")
        return "\n".join(f"{t}({', '.join(cols)})" for t, cols in tables.items())

    def system_prompt(self) -> str:
        lo, hi = self.data_range
        return f"""You are a data analyst for an online retailer. Answer questions by writing DuckDB SQL with the run_sql tool.

Data covers orders from {lo} to {hi}. Interpret relative dates relative to {hi}.

SCHEMA
{self.schema_text()}

Notes:
- orders.status is one of: completed, returned, cancelled.
- orders.channel is one of: web, mobile, marketplace.
- order_items.discount is a fraction (0.10 = 10% off); unit_price is before discount.
- products.unit_cost is the cost per unit.

Answer format: lead with the direct answer and key numbers, keep it brief. Format fractions as percentages."""

    def handle_tool(self, name: str, args: dict) -> tuple[dict, QueryRecord | None]:
        if name != "run_sql":
            return {"error": f"Unknown tool '{name}'."}, None
        sql = str(args.get("sql", "")).strip().rstrip(";")
        record = QueryRecord(tool=name, input=args, sql=sql)
        if not re.match(r"(?is)^\s*(select|with)\b", sql) or ";" in sql:
            record.error = "Only a single SELECT or WITH statement is allowed."
            return {"error": record.error}, record
        try:
            record.columns, record.rows = execute(self.con, sql)
            return self._result_payload(sql, record.columns, record.rows), record
        except duckdb.Error as e:
            record.error = str(e)
            return {"error": record.error}, record
