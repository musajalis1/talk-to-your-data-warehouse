"""Execution-based evaluation: run each question, compare result sets to gold SQL.

We compare *results*, not SQL text, because many different queries are correct.
Matching rules:
  - Numbers are compared at 6 significant digits; ints and floats are equivalent.
  - Dates and midnight timestamps are equivalent.
  - Column names and order are ignored; the prediction may include extra columns.
  - Row order matters only for `ordered` questions. For `top_n` questions the
    first N predicted rows must match the N gold rows.
"""

from __future__ import annotations

import itertools
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import duckdb
import yaml

from .agent import AgentResult, BaseAgent, execute
from .semantic import SemanticError, SemanticModel

MAX_COLUMNS_FOR_PERMUTATION = 7


def normalize(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float, Decimal)):
        return float(f"{float(v):.6g}")
    if isinstance(v, datetime):
        return v.date().isoformat() if v.time() == datetime.min.time() else v.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    return str(v)


def results_match(gold_rows: list[tuple], pred_rows: list[tuple], *, ordered: bool = False,
                  top_n: bool = False) -> tuple[bool, str]:
    gold = [tuple(normalize(v) for v in r) for r in gold_rows]
    pred = [tuple(normalize(v) for v in r) for r in pred_rows]
    if not gold:
        return (not pred), "gold is empty" if not pred else "expected no rows"
    if not pred:
        return False, "prediction returned no rows"

    g_cols, p_cols = len(gold[0]), len(pred[0])
    if p_cols < g_cols:
        return False, f"prediction has {p_cols} columns, expected at least {g_cols}"
    if top_n:
        if len(pred) < len(gold):
            return False, f"expected at least {len(gold)} rows, got {len(pred)}"
    elif len(pred) != len(gold):
        return False, f"expected {len(gold)} rows, got {len(pred)}"

    if p_cols > MAX_COLUMNS_FOR_PERMUTATION:
        candidates: Any = [tuple(range(g_cols))]
    else:
        candidates = itertools.permutations(range(p_cols), g_cols)
    for cols in candidates:
        projected = [tuple(r[i] for i in cols) for r in pred]
        if top_n and projected[:len(gold)] == gold:
            return True, "match"
        if not top_n and ordered and projected == gold:
            return True, "match"
        if not top_n and not ordered and Counter(projected) == Counter(gold):
            return True, "match"
    return False, "values differ" + (" (or row order differs)" if ordered or top_n else "")


def load_questions(path: str | Path) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["questions"]


@dataclass
class EvalRecord:
    id: str
    mode: str
    question: str
    tags: list[str]
    passed: bool
    reason: str
    sql: str | None = None
    answer: str = ""
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    errors: list[str] = field(default_factory=list)


def evaluate(con: duckdb.DuckDBPyConnection, model: SemanticModel, questions: list[dict], mode: str,
             agent: BaseAgent | None = None, progress: Callable[[EvalRecord], None] | None = None) -> list[EvalRecord]:
    """mode: 'oracle' (compile the reference semantic query) or an agent mode ('semantic', 'sql')."""
    records = []
    for q in questions:
        _, gold_rows = execute(con, q["gold_sql"])
        rec = EvalRecord(id=q["id"], mode=mode, question=q["question"], tags=q.get("tags", []),
                         passed=False, reason="")
        if mode == "oracle":
            start = time.perf_counter()
            try:
                compiled = model.compile(q["semantic_query"])
                rec.sql = compiled.rendered()
                _, pred_rows = execute(con, compiled.sql, compiled.params)
                rec.passed, rec.reason = results_match(gold_rows, pred_rows, ordered=q.get("ordered", False),
                                                       top_n=q.get("top_n", False))
            except (SemanticError, duckdb.Error) as e:
                rec.reason = f"error: {e}"
            rec.latency_s = time.perf_counter() - start
        else:
            if agent is None:
                raise ValueError(f"mode '{mode}' needs an agent")
            try:
                result: AgentResult = agent.ask(q["question"])
            except Exception as e:  # API errors shouldn't kill the whole run
                rec.reason = f"agent error: {type(e).__name__}: {e}"
                records.append(rec)
                if progress:
                    progress(rec)
                continue
            rec.answer = result.answer
            rec.latency_s = result.latency_s
            rec.input_tokens, rec.output_tokens = result.input_tokens, result.output_tokens
            rec.tool_calls = len(result.records)
            rec.errors = [r.error for r in result.records if r.error]
            final = result.final
            if final is None:
                rec.reason = "agent never ran a successful query"
            else:
                rec.sql = final.sql
                rec.passed, rec.reason = results_match(gold_rows, final.rows, ordered=q.get("ordered", False),
                                                       top_n=q.get("top_n", False))
        records.append(rec)
        if progress:
            progress(rec)
    return records


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "n/a"


def write_report(runs: dict[str, list[EvalRecord]], out_dir: str | Path, model_name: str | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    modes = list(runs)

    with (out_dir / f"eval-{stamp}.jsonl").open("w") as f:
        for recs in runs.values():
            for r in recs:
                f.write(json.dumps(asdict(r), default=str) + "\n")

    lines = [f"# Eval report ({stamp})", ""]
    if model_name and any(m != "oracle" for m in modes):
        lines += [f"Model: `{model_name}`", ""]
    lines += ["## Summary", "", "| Mode | Accuracy | Avg latency (s) | Avg tool calls | Input tokens | Output tokens |",
              "|---|---|---|---|---|---|"]
    for mode, recs in runs.items():
        n, ok = len(recs), sum(r.passed for r in recs)
        lat = sum(r.latency_s for r in recs) / n if n else 0
        calls = sum(r.tool_calls for r in recs) / n if n else 0
        lines.append(f"| {mode} | {ok}/{n} ({_pct(ok, n)}) | {lat:.2f} | {calls:.1f} | "
                     f"{sum(r.input_tokens for r in recs):,} | {sum(r.output_tokens for r in recs):,} |")

    tags = sorted({t for recs in runs.values() for r in recs for t in r.tags})
    lines += ["", "## Accuracy by tag", "", "| Tag | " + " | ".join(modes) + " |",
              "|---|" + "---|" * len(modes)]
    for tag in tags:
        cells = []
        for mode in modes:
            tagged = [r for r in runs[mode] if tag in r.tags]
            cells.append(f"{sum(r.passed for r in tagged)}/{len(tagged)}")
        lines.append(f"| {tag} | " + " | ".join(cells) + " |")

    lines += ["", "## Per question", "", "| ID | Question | " + " | ".join(modes) + " |",
              "|---|---|" + "---|" * len(modes)]
    by_id = {mode: {r.id: r for r in recs} for mode, recs in runs.items()}
    for r in next(iter(runs.values())):
        cells = ["pass" if by_id[m][r.id].passed else "**FAIL**" for m in modes if r.id in by_id[m]]
        lines.append(f"| {r.id} | {r.question} | " + " | ".join(cells) + " |")

    failures = [r for recs in runs.values() for r in recs if not r.passed]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            lines += [f"### {r.id} [{r.mode}] {r.question}", "", f"Reason: {r.reason}", ""]
            if r.errors:
                lines += ["Tool errors along the way:", ""] + [f"- {e}" for e in r.errors] + [""]
            if r.sql:
                lines += ["```sql", r.sql, "```", ""]
            if r.answer:
                lines += ["Answer given:", "", "> " + r.answer.replace("\n", "\n> "), ""]

    path = out_dir / f"eval-{stamp}.md"
    path.write_text("\n".join(lines) + "\n")
    return path
