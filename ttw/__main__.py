"""Command line interface.

    python -m ttw generate                 # build data/warehouse.duckdb
    python -m ttw catalog                  # print the semantic catalog
    python -m ttw compile '{"metrics": ["revenue"], "dimensions": ["region"]}'
    python -m ttw ask "What was revenue by region last quarter?"
    python -m ttw ask --baseline "..."     # raw-SQL agent instead
    python -m ttw eval --modes oracle semantic sql
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("TTW_DB", ROOT / "data" / "warehouse.duckdb"))
MODEL_PATH = Path(os.environ.get("TTW_SEMANTIC_MODEL", ROOT / "semantic" / "model.yml"))
QUESTIONS_PATH = ROOT / "evals" / "questions.yml"
RESULTS_DIR = ROOT / "evals" / "results"


def _connect():
    import duckdb
    if not DB_PATH.exists():
        sys.exit(f"No warehouse at {DB_PATH}. Run: python -m ttw generate")
    return duckdb.connect(str(DB_PATH), read_only=True)


def _semantic():
    from .semantic import SemanticModel
    return SemanticModel.load(MODEL_PATH)


def _require_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY to use the agents (oracle eval and compile work without it).")


def cmd_generate(args: argparse.Namespace) -> None:
    from .generate import generate
    counts = generate(DB_PATH, seed=args.seed)
    print(f"Wrote {DB_PATH}: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))


def cmd_catalog(args: argparse.Namespace) -> None:
    print(_semantic().catalog_text())


def cmd_compile(args: argparse.Namespace) -> None:
    from .agent import execute
    compiled = _semantic().compile(json.loads(args.query))
    print(compiled.rendered())
    if args.run:
        cols, rows = execute(_connect(), compiled.sql, compiled.params)
        print("\n" + " | ".join(cols))
        for r in rows[:50]:
            print(" | ".join(str(v) for v in r))


def cmd_ask(args: argparse.Namespace) -> None:
    from .agent import SemanticAgent, SQLAgent
    _require_key()
    con = _connect()
    kwargs = {"model": args.model} if args.model else {}
    agent = SQLAgent(con, **kwargs) if args.baseline else SemanticAgent(con, _semantic(), **kwargs)
    result = agent.ask(args.question)
    print(result.answer)
    if args.show_sql and result.final:
        print("\n--- SQL ---\n" + (result.final.sql or ""))
    print(f"\n({result.latency_s:.1f}s, {len(result.records)} queries, "
          f"{result.input_tokens:,} in / {result.output_tokens:,} out tokens)", file=sys.stderr)


def cmd_eval(args: argparse.Namespace) -> None:
    from .agent import DEFAULT_MODEL, SemanticAgent, SQLAgent
    from .evaluation import evaluate, load_questions, write_report

    questions = load_questions(QUESTIONS_PATH)
    if args.ids:
        questions = [q for q in questions if q["id"] in set(args.ids)]
    con, semantic = _connect(), _semantic()
    model_name = args.model or DEFAULT_MODEL
    runs = {}
    for mode in args.modes:
        agent = None
        if mode == "semantic":
            _require_key()
            agent = SemanticAgent(con, semantic, model=model_name)
        elif mode == "sql":
            _require_key()
            agent = SQLAgent(con, model=model_name)
        print(f"\n== {mode} ==")
        runs[mode] = evaluate(con, semantic, questions, mode, agent=agent, progress=lambda r: print(
            f"  {r.id} {'PASS' if r.passed else 'FAIL'}  {r.question}" + ("" if r.passed else f"  [{r.reason}]")))
        ok = sum(r.passed for r in runs[mode])
        print(f"  -> {ok}/{len(runs[mode])} correct")
    path = write_report(runs, RESULTS_DIR, model_name=model_name)
    print(f"\nReport: {path}")
    if args.fail_under is not None:
        worst = min(sum(r.passed for r in recs) / len(recs) for recs in runs.values())
        if worst < args.fail_under:
            sys.exit(f"Accuracy {worst:.0%} is below --fail-under {args.fail_under:.0%}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ttw", description="Talk to your warehouse")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="Build the synthetic DuckDB warehouse")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("catalog", help="Print the semantic catalog the agent sees")
    p.set_defaults(func=cmd_catalog)

    p = sub.add_parser("compile", help="Compile a semantic query (JSON) to SQL")
    p.add_argument("query")
    p.add_argument("--run", action="store_true", help="Also execute it")
    p.set_defaults(func=cmd_compile)

    p = sub.add_parser("ask", help="Ask a question in plain English")
    p.add_argument("question")
    p.add_argument("--baseline", action="store_true", help="Use the raw-SQL agent")
    p.add_argument("--show-sql", action="store_true")
    p.add_argument("--model")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="Run the evaluation set")
    p.add_argument("--modes", nargs="+", default=["oracle"], choices=["oracle", "semantic", "sql"])
    p.add_argument("--ids", nargs="+", help="Only run these question ids")
    p.add_argument("--model")
    p.add_argument("--fail-under", type=float, help="Exit non-zero if any mode scores below this (0-1). For CI.")
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
