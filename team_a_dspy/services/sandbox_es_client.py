from __future__ import annotations

from copy import deepcopy
from typing import Any

from services.config import settings
from services.es_client import ESClient


class SandboxESClient(ESClient):
    SAFE_MAX_SIZE = 100
    SAFE_MAX_AGG_SIZE = 100

    AGG_FIELD_TYPES = {
        "terms",
        "avg",
        "sum",
        "min",
        "max",
        "cardinality",
        "value_count",
        "date_histogram",
        "histogram",
        "stats",
        "extended_stats",
        "percentiles",
        "median_absolute_deviation",
    }

    QUERY_FIELD_TYPES = {
        "term",
        "range",
        "match",
        "match_phrase",
        "wildcard",
        "prefix",
        "regexp",
    }

    BOOL_KEYS = {"must", "should", "filter", "must_not"}

    def __init__(
        self,
        host: str | None = None,
        username: str | None = None,
        password: str | None = None,
        index: str | None = None,
        verify_ssl: bool | None = None,
    ):
        super().__init__(
            host or settings.sandbox_es_host,
            username if username is not None else settings.sandbox_es_username,
            password if password is not None else settings.sandbox_es_password,
            index or settings.sandbox_es_index,
            verify_ssl if verify_ssl is not None else settings.sandbox_es_verify_ssl,
        )
        self._flat_mapping_cache: dict[str, str] | None = None

    def get_flat_mapping(self) -> dict[str, str]:
        if self._flat_mapping_cache is None:
            self._flat_mapping_cache = self.flatten_es_mapping()
        return self._flat_mapping_cache

    async def validate_query_dsl(
        self,
        query_dsl: dict[str, Any],
        expected_query_dsl: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self.evaluate_query_dsl(
            query_dsl=query_dsl,
            expected_query_dsl=expected_query_dsl,
        )

    async def evaluate_query_dsl(
        self,
        query_dsl: dict[str, Any],
        expected_query_dsl: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized = self.normalize_query_dsl(query_dsl)
        if not normalized:
            return self._result(
                is_valid=False,
                score=0.0,
                feedback="Generated query_dsl is empty or not a dictionary.",
                details={"error": "empty_query"},
                safety_score=0.0,
                schema_score=0.0,
                execution_score=0.0,
                task_shape_score=0.0,
                semantic_score=0.0,
                exact_match_score=0.0,
            )

        safety_ok, safety_feedback = self._check_safety(normalized)
        if not safety_ok:
            return self._result(
                is_valid=False,
                score=0.0,
                feedback=safety_feedback,
                details={"error": "unsafe_query"},
                safety_score=0.0,
                schema_score=0.0,
                execution_score=0.0,
                task_shape_score=0.0,
                semantic_score=0.0,
                exact_match_score=0.0,
            )

        schema_score, unknown_fields, referenced_fields = self._score_schema_fields(normalized)
        execution_score, execution_feedback, execution_details = await self._score_execution(normalized)

        task_shape_score = 1.0
        semantic_score = 1.0
        exact_match_score = 0.0

        if expected_query_dsl is not None:
            expected = self.normalize_query_dsl(expected_query_dsl)
            task_shape_score = self._score_task_shape(expected, normalized)
            semantic_score = self._score_semantic_alignment(expected, normalized)
            exact_match_score = 1.0 if expected == normalized else 0.0

        base_score = (
            (0.20 * 1.0)
            + (0.25 * schema_score)
            + (0.25 * execution_score)
            + (0.15 * task_shape_score)
            + (0.15 * semantic_score)
        )

        is_valid = execution_score >= 0.95 and schema_score >= 0.999 and not unknown_fields

        # softer penalty so optimizer sees gradations instead of everything collapsing to 0.2
        if is_valid:
            final_score = round(base_score, 4)
        else:
            final_score = round(max(0.0, min(base_score, 0.75)), 4)

        feedback_parts: list[str] = []
        if unknown_fields:
            feedback_parts.append(f"Unknown fields referenced: {', '.join(sorted(unknown_fields))}.")
        if execution_feedback:
            feedback_parts.append(execution_feedback)
        if not feedback_parts:
            feedback_parts.append("Query validated and executed successfully.")

        return self._result(
            is_valid=is_valid,
            score=final_score,
            feedback=" ".join(feedback_parts),
            details={
                "referenced_fields": sorted(referenced_fields),
                "unknown_fields": sorted(unknown_fields),
                **execution_details,
            },
            safety_score=1.0,
            schema_score=schema_score,
            execution_score=execution_score,
            task_shape_score=task_shape_score,
            semantic_score=semantic_score,
            exact_match_score=exact_match_score,
        )

    @staticmethod
    def normalize_query_dsl(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        nested = payload.get("query_dsl")
        if isinstance(nested, dict):
            return nested
        return payload

    def _result(
        self,
        *,
        is_valid: bool,
        score: float,
        feedback: str,
        details: dict[str, Any],
        safety_score: float,
        schema_score: float,
        execution_score: float,
        task_shape_score: float,
        semantic_score: float,
        exact_match_score: float,
    ) -> dict[str, Any]:
        return {
            "is_valid": bool(is_valid),
            "score": float(score),
            "feedback": feedback,
            "details": details,
            "safety_score": float(safety_score),
            "schema_score": float(schema_score),
            "execution_score": float(execution_score),
            "task_shape_score": float(task_shape_score),
            "semantic_score": float(semantic_score),
            "exact_match_score": float(exact_match_score),
        }

    def _check_safety(self, query_dsl: dict[str, Any]) -> tuple[bool, str]:
        if self._contains_key(query_dsl, "script"):
            return False, "Unsafe query: 'script' is not allowed."

        for size in self._find_numeric_values_for_key(query_dsl, "size"):
            if size > self.SAFE_MAX_SIZE:
                return False, f"Unsafe query: size={size} exceeds SAFE_MAX_SIZE={self.SAFE_MAX_SIZE}."

        for agg_size in self._find_terms_agg_sizes(query_dsl):
            if agg_size > self.SAFE_MAX_AGG_SIZE:
                return False, (
                    f"Unsafe query: aggregation size={agg_size} exceeds "
                    f"SAFE_MAX_AGG_SIZE={self.SAFE_MAX_AGG_SIZE}."
                )

        return True, ""

    def _score_schema_fields(self, query_dsl: dict[str, Any]) -> tuple[float, set[str], set[str]]:
        valid_fields = set(self.get_flat_mapping().keys())
        referenced_fields = self.extract_referenced_fields(query_dsl)

        if not referenced_fields:
            return 0.0, set(), set()

        unknown_fields = {field for field in referenced_fields if field not in valid_fields}
        known_count = len(referenced_fields) - len(unknown_fields)
        score = round(known_count / max(1, len(referenced_fields)), 4)
        return score, unknown_fields, referenced_fields

    async def _score_execution(self, query_dsl: dict[str, Any]) -> tuple[float, str, dict[str, Any]]:
        body = self._cap_query(query_dsl)
        query_part = body.get("query", {})

        if query_part:
            try:
                validate_response = self.es.indices.validate_query(
                    index=self.index,
                    body={"query": query_part},
                    explain=True,
                )
                validate_body = self._response_body(validate_response)
                if validate_body and not bool(validate_body.get("valid", True)):
                    return 0.10, "Elasticsearch validate_query reported the query as invalid.", {
                        "validate_response": validate_body
                    }
            except Exception as exc:
                return 0.10, f"Elasticsearch validate_query failed: {type(exc).__name__}: {exc}", {}

        try:
            search_response = self.es.search(index=self.index, body=body)
            search_body = self._response_body(search_response)
            return 1.0, "", {
                "hits_total": self._extract_total_hits(search_body),
                "aggregation_names": list(search_body.get("aggregations", {}).keys()),
            }
        except Exception as exc:
            return 0.10, f"Elasticsearch search failed: {type(exc).__name__}: {exc}", {}

    def _score_task_shape(self, expected: dict[str, Any], predicted: dict[str, Any]) -> float:
        checks = [
            int(("aggs" in expected) == ("aggs" in predicted)),
            int(("sort" in expected) == ("sort" in predicted)),
            int(("query" in expected) == ("query" in predicted)),
            int(("_source" in expected) == ("_source" in predicted)),
            int(self._contains_query_type(expected, "range") == self._contains_query_type(predicted, "range")),
        ]

        expected_size = expected.get("size")
        predicted_size = predicted.get("size")
        if expected_size is None and predicted_size is None:
            checks.append(1)
        elif isinstance(expected_size, int) and isinstance(predicted_size, int):
            checks.append(int(expected_size == predicted_size))
        else:
            checks.append(0)

        return round(sum(checks) / max(1, len(checks)), 4)

    def _score_semantic_alignment(self, expected: dict[str, Any], predicted: dict[str, Any]) -> float:
        expected_fields = self.extract_referenced_fields(expected)
        predicted_fields = self.extract_referenced_fields(predicted)

        if not expected_fields and not predicted_fields:
            return 1.0
        if not expected_fields or not predicted_fields:
            return 0.0

        overlap = len(expected_fields & predicted_fields)
        union = len(expected_fields | predicted_fields)
        return round(overlap / max(1, union), 4)

    def extract_referenced_fields(self, query_dsl: dict[str, Any]) -> set[str]:
        fields: set[str] = set()

        def visit(node: Any, parent_key: str | None = None) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    # bool containers
                    if key in self.BOOL_KEYS:
                        visit(value, parent_key=key)
                        continue

                    if key == "bool":
                        visit(value, parent_key="bool")
                        continue

                    # query clauses with field name in key
                    if key in self.QUERY_FIELD_TYPES and isinstance(value, dict):
                        for field_name, field_clause in value.items():
                            fields.add(field_name)
                            visit(field_clause, parent_key=key)
                        continue

                    # query-style terms: {"terms": {"field_name": ["a", "b"]}}
                    if key == "terms" and isinstance(value, dict):
                        if "field" in value:
                            # aggregation-style terms
                            field_value = value.get("field")
                            if isinstance(field_value, str):
                                fields.add(field_value)
                            for inner_key, inner_value in value.items():
                                if inner_key != "field":
                                    visit(inner_value, parent_key="terms_agg")
                        else:
                            for field_name, field_clause in value.items():
                                fields.add(field_name)
                                visit(field_clause, parent_key="terms_query")
                        continue

                    # aggs block
                    if key == "aggs" and isinstance(value, dict):
                        for _, agg_body in value.items():
                            visit(agg_body, parent_key="aggs")
                        continue

                    # aggregation operators
                    if key in self.AGG_FIELD_TYPES and isinstance(value, dict):
                        field_value = value.get("field")
                        if isinstance(field_value, str):
                            fields.add(field_value)

                        # date_histogram often uses field/fixed_interval/calendar_interval
                        for inner_key, inner_value in value.items():
                            if inner_key != "field":
                                visit(inner_value, parent_key=key)
                        continue

                    # sort fields
                    if key == "sort":
                        if isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict):
                                    for sort_field in item.keys():
                                        fields.add(sort_field)
                                        visit(item[sort_field], parent_key="sort")
                        elif isinstance(value, dict):
                            for sort_field in value.keys():
                                fields.add(sort_field)
                                visit(value[sort_field], parent_key="sort")
                        continue

                    visit(value, parent_key=key)

            elif isinstance(node, list):
                for item in node:
                    visit(item, parent_key=parent_key)

        visit(query_dsl)
        return {field for field in fields if isinstance(field, str) and field.strip()}

    def _cap_query(self, query_dsl: dict[str, Any]) -> dict[str, Any]:
        body = deepcopy(query_dsl)

        top_size = body.get("size")
        if isinstance(top_size, int):
            body["size"] = min(top_size, self.SAFE_MAX_SIZE)

        aggs = body.get("aggs")
        if isinstance(aggs, dict):
            self._cap_agg_sizes(aggs)

        return body

    def _cap_agg_sizes(self, aggs: dict[str, Any]) -> None:
        for _, agg_body in aggs.items():
            if not isinstance(agg_body, dict):
                continue
            for agg_type, agg_config in agg_body.items():
                if agg_type == "aggs" and isinstance(agg_config, dict):
                    self._cap_agg_sizes(agg_config)
                    continue
                if agg_type == "terms" and isinstance(agg_config, dict):
                    size = agg_config.get("size")
                    if isinstance(size, int):
                        agg_config["size"] = min(size, self.SAFE_MAX_AGG_SIZE)
                if isinstance(agg_config, dict) and "aggs" in agg_config and isinstance(agg_config["aggs"], dict):
                    self._cap_agg_sizes(agg_config["aggs"])

    @staticmethod
    def _contains_key(node: Any, target_key: str) -> bool:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == target_key:
                    return True
                if SandboxESClient._contains_key(value, target_key):
                    return True
        elif isinstance(node, list):
            return any(SandboxESClient._contains_key(item, target_key) for item in node)
        return False

    @staticmethod
    def _find_numeric_values_for_key(node: Any, target_key: str) -> list[int]:
        values: list[int] = []
        if isinstance(node, dict):
            for key, value in node.items():
                if key == target_key and isinstance(value, int):
                    values.append(value)
                values.extend(SandboxESClient._find_numeric_values_for_key(value, target_key))
        elif isinstance(node, list):
            for item in node:
                values.extend(SandboxESClient._find_numeric_values_for_key(item, target_key))
        return values

    @staticmethod
    def _find_terms_agg_sizes(node: Any) -> list[int]:
        sizes: list[int] = []
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "terms" and isinstance(value, dict):
                    size = value.get("size")
                    if isinstance(size, int):
                        sizes.append(size)
                sizes.extend(SandboxESClient._find_terms_agg_sizes(value))
        elif isinstance(node, list):
            for item in node:
                sizes.extend(SandboxESClient._find_terms_agg_sizes(item))
        return sizes

    @staticmethod
    def _contains_query_type(node: Any, query_type: str) -> bool:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == query_type:
                    return True
                if SandboxESClient._contains_query_type(value, query_type):
                    return True
        elif isinstance(node, list):
            return any(SandboxESClient._contains_query_type(item, query_type) for item in node)
        return False

    @staticmethod
    def _response_body(response: Any) -> dict[str, Any]:
        if isinstance(response, dict):
            return response
        if hasattr(response, "body") and isinstance(response.body, dict):
            return response.body
        return {}

    @staticmethod
    def _extract_total_hits(search_body: dict[str, Any]) -> int:
        hits_total = search_body.get("hits", {}).get("total", 0)
        if isinstance(hits_total, dict):
            value = hits_total.get("value", 0)
            return int(value) if isinstance(value, int) else 0
        return int(hits_total) if isinstance(hits_total, int) else 0