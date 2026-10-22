from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest

from ttw.agent import SemanticAgent, SQLAgent, execute
from ttw.evaluation import evaluate, load_questions, results_match
from ttw.generate import generate
from ttw.semantic import SemanticError, SemanticModel

ROOT = Path(__file__).resolve().parent.parent
REVENUE_2025 = {"metrics": ["revenue"],
                "filters": [{"field": "order_date", "operator": "between", "value": ["2025-01-01", "2025-12-31"]}]}


@pytest.fixture(scope="session")
def con(tmp_path_factory):
    path = tmp_path_factory.mktemp("wh") / "warehouse.duckdb"
    generate(path)
    c = duckdb.connect(str(path), read_only=True)
    yield c
    c.close()


@pytest.fixture(scope="session")
def model():
    return SemanticModel.load(ROOT / "semantic" / "model.yml")


# ---------------------------------------------------------------- compiler

def test_joins_only_needed_tables(model):
    q = model.compile({"metrics": ["order_count"]})
    assert q.tables == ["order_items", "orders"]
    q = model.compile({"metrics": ["revenue"], "dimensions": ["region"]})
    assert q.tables == ["order_items", "orders", "customers"]  # customers joins via orders


def test_shared_metric_filter_is_pushed_to_where(model):
    sql = model.compile({"metrics": ["revenue", "order_count"]}).sql
    assert "WHERE o.status = 'completed'" in sql and "CASE WHEN" not in sql


def test_mixed_metric_filters_use_case_when(model):
    sql = model.compile({"metrics": ["revenue", "return_rate"]}).sql
    assert "WHERE" not in sql
    assert "CASE WHEN o.status = 'returned'" in sql and "CASE WHEN o.status = 'completed'" in sql


def test_derived_metrics_expand(model):
    sql = model.compile({"metrics": ["average_order_value"]}).sql
    assert "SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) / NULLIF(COUNT(DISTINCT o.order_id), 0)" in sql


def test_values_are_bound_parameters_not_spliced(model):
    evil = "West'; DROP TABLE orders; --"
    q = model.compile({"metrics": ["revenue"], "filters": [{"field": "region", "operator": "=", "value": evil}]})
    assert evil not in q.sql and q.params == [evil]


def test_metric_filters_become_having(model):
    q = model.compile({"metrics": ["revenue"], "dimensions": ["category"],
                       "filters": [{"field": "revenue", "operator": ">", "value": 5}]})
    assert "HAVING SUM(" in q.sql and q.params == [5]


def test_order_by_unselected_metric_adds_it(model):
    q = model.compile({"metrics": ["revenue"], "dimensions": ["product"],
                       "order_by": [{"field": "units_sold", "direction": "desc"}], "limit": 3})
    assert q.columns == ["product", "revenue", "units_sold"]


@pytest.mark.parametrize("query, message", [
    ({"metrics": ["revenu"]}, "Did you mean: revenue"),
    ({"dimensions": ["regoin"]}, "Did you mean: region"),
    ({"dimensions": ["region__month"]}, "not a time dimension"),
    ({"dimensions": ["order_date__hour"]}, "supports grains"),
    ({"metrics": ["revenue"], "filters": [{"field": "order_date", "operator": "=", "value": "last year"}]}, "ISO dates"),
    ({"metrics": ["revenue"], "filters": [{"field": "region", "operator": "between", "value": "West"}]}, "two-element"),
    ({"metrics": ["revenue"], "order_by": [{"field": "region"}]}, "unless it is in 'dimensions'"),
    ({"metrics": ["revenue"], "limit": 0}, "limit must be"),
    ({"metrics": ["revenue"], "sql": "SELECT 1"}, "Unknown query keys"),
    ({}, "at least one metric or dimension"),
])
def test_errors_are_helpful(model, query, message):
    with pytest.raises(SemanticError, match=message):
        model.compile(query)


def test_model_validation_rejects_cycles():
    spec = {"fact": {"table": "t", "alias": "t"}, "metrics": {
        "a": {"derived": "{b} + 1"}, "b": {"derived": "{a} + 1"}}}
    with pytest.raises(SemanticError, match="Circular"):
        SemanticModel(spec)


# ---------------------------------------------------------------- scorer

def test_normalization_equivalences():
    ok, _ = results_match([(Decimal("10.50"), date(2025, 1, 1))], [(10.5, datetime(2025, 1, 1))])
    assert ok


def test_column_order_and_extra_columns_ignored():
    ok, _ = results_match([("West", 1.0)], [(1.0, "West", 99)])
    assert ok


def test_ordered_and_top_n():
    gold = [("a", 3), ("b", 2)]
    assert results_match(gold, [("b", 2), ("a", 3)])[0]
    assert not results_match(gold, [("b", 2), ("a", 3)], ordered=True)[0]
    assert results_match([("a", 3)], [("a", 3), ("b", 2)], ordered=True, top_n=True)[0]
    assert not results_match([("a", 3)], [("b", 2), ("a", 3)], ordered=True, top_n=True)[0]


def test_oracle_matches_every_gold_answer(con, model):
    records = evaluate(con, model, load_questions(ROOT / "evals" / "questions.yml"), "oracle")
    failures = [(r.id, r.reason) for r in records if not r.passed]
    assert not failures


def test_scorer_rejects_revenue_that_ignores_the_business_rule(con):
    """A plausible-but-wrong answer (counting returned/cancelled orders) must fail."""
    _, gold = execute(con, """SELECT SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) FROM order_items oi
        JOIN orders o USING (order_id) WHERE o.status = 'completed' AND year(o.order_date) = 2025""")
    _, naive = execute(con, """SELECT SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) FROM order_items oi
        JOIN orders o USING (order_id) WHERE year(o.order_date) = 2025""")
    assert not results_match(gold, naive)[0]


# ---------------------------------------------------------------- agent loop (no network)

class FakeClient:
    """Replays scripted assistant turns and records what the agent sent."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        blocks = self.turns.pop(0)
        content = [SimpleNamespace(**b) for b in blocks]
        return SimpleNamespace(content=content, usage=SimpleNamespace(input_tokens=100, output_tokens=20),
                               stop_reason="tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn")


def tool_use(id_, name, input_):
    return {"type": "tool_use", "id": id_, "name": name, "input": input_}


def text(t):
    return {"type": "text", "text": t}


def test_semantic_agent_recovers_from_error_and_answers(con, model):
    client = FakeClient([
        [tool_use("t1", "query_semantic_layer", {"metrics": ["sales_total"]})],
        [tool_use("t2", "query_semantic_layer", REVENUE_2025)],
        [text("2025 revenue was about $2.7M.")],
    ])
    agent = SemanticAgent(con, model, client=client)
    result = agent.ask("What was revenue in 2025?")

    assert result.answer == "2025 revenue was about $2.7M."
    assert result.turns == 3 and result.input_tokens == 300
    assert result.records[0].error and "Unknown metric" in result.records[0].error
    # messages: [user question, assistant tool_use, user tool_result, ...]
    first_tool_result = client.calls[1]["messages"][2]["content"][0]
    assert first_tool_result["is_error"] is True
    assert result.final is result.records[1]
    _, gold = execute(con, model.compile(REVENUE_2025).sql, model.compile(REVENUE_2025).params)
    assert results_match(gold, result.final.rows)[0]


def test_semantic_agent_prompt_and_tools(con, model):
    agent = SemanticAgent(con, model, client=FakeClient([]))
    prompt = agent.system_prompt()
    assert "2024-01-01" in prompt and "2025-12-31" in prompt
    assert "Values: Midwest, Northeast, South, West." in prompt
    schema = agent.tools[0]["input_schema"]["properties"]
    assert "order_date__month" in schema["dimensions"]["items"]["enum"]


def test_dimension_lookup_tool(con, model):
    agent = SemanticAgent(con, model, client=FakeClient([]))
    payload, _ = agent.handle_tool("get_dimension_values", {"dimension": "product", "search": "shoe"})
    assert payload["values"] == ["Running Shoes"]


def test_sql_agent_is_read_only_and_single_statement(con):
    agent = SQLAgent(con, client=FakeClient([]))
    payload, _ = agent.handle_tool("run_sql", {"sql": "DELETE FROM orders"})
    assert "error" in payload
    payload, _ = agent.handle_tool("run_sql", {"sql": "SELECT 1; DROP TABLE orders"})
    assert "error" in payload
    payload, _ = agent.handle_tool("run_sql", {"sql": "SELECT COUNT(*) AS n FROM orders"})
    assert payload["rows"] == [[26351]]


def test_max_turns_stops_loop(con, model):
    looping = [[tool_use(f"t{i}", "query_semantic_layer", REVENUE_2025)] for i in range(3)]
    agent = SemanticAgent(con, model, client=FakeClient(looping), max_turns=3)
    result = agent.ask("loop forever")
    assert "maximum number of tool calls" in result.answer and result.turns == 3


def test_tool_results_are_valid_json(con, model):
    agent = SemanticAgent(con, model, client=FakeClient([]))
    payload, _ = agent.handle_tool("query_semantic_layer", {"metrics": ["revenue"], "dimensions": ["order_date__month"]})
    json.loads(json.dumps(payload))
    assert payload["row_count"] == 24 and payload["rows"][0][0] == "2024-01-01"
