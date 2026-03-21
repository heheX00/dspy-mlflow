from __future__ import annotations

from typing import Any, Dict, Tuple

from elasticsearch import Elasticsearch

from team_a_dspy.modules.es_query_pipeline import (
    validate_query_shape,
    validate_read_only_query,
)
from team_a_dspy.utils.es_client import get_es_client
from team_a_dspy.utils.json_utils import extract_first_json_object


class ESSandboxJudge:
    """
    Execution metric for DSPy optimizers.

    Pass conditions:
    1. Query is valid JSON and read-only
    2. Query executes successfully in sandbox
    3. Query returns meaningful / non-empty results
    4. Query satisfies example-specific intent checks
    """

    def __init__(self, es: Elasticsearch | None = None, index_name: str = "gkg") -> None:
        self.es = es or get_es_client()
        self.index_name = index_name

    def _execute(self, query: Dict[str, Any]) -> Dict[str, Any]:
        return self.es.search(index=self.index_name, body=query)

    def _has_meaningful_results(self, response: Dict[str, Any], query: Dict[str, Any]) -> Tuple[bool, str]:
        if query.get("size") == 0:
            aggs = response.get("aggregations") or {}
            if not aggs:
                return False, "Aggregation query returned no aggregations"

            for agg_value in aggs.values():
                buckets = agg_value.get("buckets")
                if isinstance(buckets, list) and len(buckets) > 0:
                    return True, "ok"

            return False, "Aggregation query returned no non-empty buckets"

        total_hits = ((response.get("hits") or {}).get("total") or {}).get("value", 0)
        if total_hits <= 0:
            return False, "Document query returned zero hits"
        return True, "ok"

    def _logical_checks(self, example: Any, response: Dict[str, Any], query: Dict[str, Any]) -> Tuple[bool, str]:
        checks = getattr(example, "expected_checks", {}) or {}

        expected_top_size = checks.get("expected_top_size")
        if expected_top_size is not None:
            if int(query.get("size", 10)) != int(expected_top_size):
                return False, f"Top-level size mismatch: expected {expected_top_size}"

        require_size_zero = checks.get("require_size_zero")
        if require_size_zero is True and query.get("size") != 0:
            return False, "Expected size=0 for ranking/aggregation question"

        required_agg = checks.get("required_agg")
        if required_agg:
            if required_agg not in (response.get("aggregations") or {}):
                return False, f"Missing required aggregation: {required_agg}"

        required_field_fragment = checks.get("required_field_fragment")
        if required_field_fragment:
            query_str = str(query)
            if required_field_fragment not in query_str:
                return False, f"Required field fragment missing: {required_field_fragment}"

        min_bucket_count = checks.get("min_bucket_count")
        agg_name = checks.get("required_agg")
        if min_bucket_count is not None and agg_name:
            buckets = ((response.get("aggregations") or {}).get(agg_name) or {}).get("buckets", [])
            if len(buckets) < int(min_bucket_count):
                return False, f"Too few buckets: got {len(buckets)}, expected >= {min_bucket_count}"

        min_hits = checks.get("min_hits")
        if min_hits is not None:
            total_hits = ((response.get("hits") or {}).get("total") or {}).get("value", 0)
            if total_hits < int(min_hits):
                return False, f"Too few hits: got {total_hits}, expected >= {min_hits}"

        return True, "ok"

    def score_prediction(self, example: Any, prediction: Any) -> tuple[bool, Dict[str, Any]]:
        try:
            raw_output = prediction.querydsl_query if hasattr(prediction, "querydsl_query") else str(prediction)
            query = extract_first_json_object(raw_output)
            validate_read_only_query(query)
            question = getattr(example, "user_question", "")
            validate_query_shape(question, query)
        except Exception as exc:
            return False, {"stage": "parse_or_safety", "error": str(exc)}

        try:
            response = self._execute(query)
        except Exception as exc:
            return False, {"stage": "execution", "error": str(exc), "query": query}

        meaningful, meaningful_reason = self._has_meaningful_results(response, query)
        if not meaningful:
            return False, {
                "stage": "meaningfulness",
                "reason": meaningful_reason,
                "query": query,
            }

        passed, reason = self._logical_checks(example, response, query)
        return passed, {
            "stage": "logical_checks",
            "reason": reason,
            "query": query,
            "response_summary": {
                "took": response.get("took"),
                "timed_out": response.get("timed_out"),
                "hits_total": ((response.get("hits") or {}).get("total") or {}).get("value"),
                "agg_keys": list((response.get("aggregations") or {}).keys()),
            },
        }

    def metric(self, example: Any, prediction: Any, trace: Any = None) -> bool:
        passed, _details = self.score_prediction(example, prediction)
        return passed