from __future__ import annotations

import asyncio
from typing import Any

import dspy

from services.sandbox_es_client import SandboxESClient


def normalize_query_dsl(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("query_dsl")
    if isinstance(nested, dict):
        return nested
    return payload


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        return loop.run_until_complete(coro)


class ExecutionAwareESMetric:
    def __init__(self, sandbox_client: SandboxESClient):
        self.sandbox_client = sandbox_client

    def __call__(self, example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
        gold = normalize_query_dsl(getattr(example, "query_dsl", {}))
        candidate = normalize_query_dsl(getattr(pred, "query_dsl", {}))

        if not candidate:
            return 0.0

        result = _run_async(
            self.sandbox_client.evaluate_query_dsl(
                query_dsl=candidate,
                expected_query_dsl=gold,
            )
        )

        if not result.get("is_valid"):
            return float(min(result.get("score", 0.0), 0.20))
        return float(result.get("score", 0.0))


def metric_exact_query_dsl(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    gold = normalize_query_dsl(getattr(example, "query_dsl", {}))
    candidate = normalize_query_dsl(getattr(pred, "query_dsl", {}))
    return 1.0 if gold == candidate else 0.0