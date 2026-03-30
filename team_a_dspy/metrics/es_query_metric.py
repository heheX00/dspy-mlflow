from __future__ import annotations

from typing import Any

import dspy

try:
    from team_a_dspy.services.judge_dspy import JudgeDSPY
    from team_a_dspy.services.sandbox_es_client import SandboxESClient
except ModuleNotFoundError:
    from services.judge_dspy import JudgeDSPY
    from services.sandbox_es_client import SandboxESClient


def normalize_query_dsl(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("query_dsl")
    if isinstance(nested, dict):
        return nested
    return payload

class ExecutionAwareESMetric:
    def __init__(self, sandbox_client: SandboxESClient):
        self.sandbox_client = sandbox_client

    def __call__(self, example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
        gold = normalize_query_dsl(getattr(example, "query_dsl", {}))
        candidate = normalize_query_dsl(getattr(pred, "query_dsl", {}))
        if not candidate:
            return 0.0

        result = self.sandbox_client.evaluate_query_dsl(query_dsl=candidate, expected_query_dsl=gold)
        return float(result.get("score", 0.0))


class RelevanceAwareExecutionMetric:
    """Combine sandbox execution quality with judge_relevance on retrieved results."""

    def __init__(self, sandbox_client: SandboxESClient, judge: JudgeDSPY):
        self.sandbox_client = sandbox_client
        self.judge = judge

    def __call__(self, example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
        gold = normalize_query_dsl(getattr(example, "query_dsl", {}))
        candidate = normalize_query_dsl(getattr(pred, "query_dsl", {}))
        nl_query = str(getattr(example, "nl_query", ""))

        if not candidate:
            return 0.0

        execution = self.sandbox_client.evaluate_query_dsl(query_dsl=candidate, expected_query_dsl=gold)

        execution_score = float(execution.get("score", 0.0))
        if not execution.get("is_valid"):
            return execution_score

        try:
            search_response = self.sandbox_client.search(candidate)
            docs = search_response.get("hits", {}).get("hits", [])
            aggregation = self.judge._aggregate_es_documents(docs)
            relevance = self.judge.compute_relevance_score(nl_query=nl_query, aggregation=aggregation)
            relevance_score = max(0.0, min(1.0, float(relevance.get("relevance_score", 0)) / 100.0))
        except Exception:
            relevance_score = 0.0

        return round(0.65 * execution_score + 0.35 * relevance_score, 4)


def metric_exact_query_dsl(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    gold = normalize_query_dsl(getattr(example, "query_dsl", {}))
    candidate = normalize_query_dsl(getattr(pred, "query_dsl", {}))
    return 1.0 if gold == candidate else 0.0
