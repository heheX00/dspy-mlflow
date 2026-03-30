from __future__ import annotations

from copy import deepcopy
from typing import Any


try:
    from team_a_dspy.services.config import settings
    from team_a_dspy.services.es_client import ESClient
except ModuleNotFoundError:
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
        "terms",
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
        semantic_details: dict[str, Any] = {}

        if expected_query_dsl is not None:
            expected = self.normalize_query_dsl(expected_query_dsl)
            task_shape_score = self._score_task_shape(expected, normalized)
            semantic_score, semantic_details = self._score_semantic_alignment(expected, normalized)
            exact_match_score = 1.0 if expected == normalized else 0.0

        final_score = (
            0.15 * 1.0
            + 0.15 * schema_score
            + 0.20 * execution_score
            + 0.20 * task_shape_score
            + 0.30 * semantic_score
        )

        final_score = round(final_score, 4)

        # strict validity: query must run, use known fields, and be semantically very close
        is_valid = (
            execution_score >= 0.95
            and schema_score >= 0.999
            and not unknown_fields
            and task_shape_score >= 0.80
            and semantic_score >= 0.85
        )

        feedback_parts: list[str] = []
        if unknown_fields:
            feedback_parts.append(f"Unknown fields referenced: {', '.join(sorted(unknown_fields))}.")
        if execution_feedback:
            feedback_parts.append(execution_feedback)

        if expected_query_dsl is not None:
            semantic_messages = semantic_details.get("messages", [])
            if semantic_messages:
                feedback_parts.extend(semantic_messages)

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
                **semantic_details,
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
            int(("track_total_hits" in expected) == ("track_total_hits" in predicted)),
            int(("_source" in expected) == ("_source" in predicted)),
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

    def _score_semantic_alignment(self, expected: dict[str, Any], predicted: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        expected_fields = self.extract_referenced_fields(expected)
        predicted_fields = self.extract_referenced_fields(predicted)

        field_overlap_score = self._jaccard_score(expected_fields, predicted_fields)

        expected_terms = self._extract_terms_constraints(expected)
        predicted_terms = self._extract_terms_constraints(predicted)
        terms_score, term_messages = self._compare_terms_constraints(expected_terms, predicted_terms)

        expected_ranges = self._extract_range_constraints(expected)
        predicted_ranges = self._extract_range_constraints(predicted)
        range_score, range_messages = self._compare_range_constraints(expected_ranges, predicted_ranges)

        expected_aggs = self._extract_agg_signature(expected)
        predicted_aggs = self._extract_agg_signature(predicted)
        agg_score, agg_messages = self._compare_agg_signature(expected_aggs, predicted_aggs)

        expected_sort = self._extract_sort_signature(expected)
        predicted_sort = self._extract_sort_signature(predicted)
        sort_score, sort_messages = self._compare_sort_signature(expected_sort, predicted_sort)

        semantic_score = (
            0.20 * field_overlap_score
            + 0.35 * terms_score
            + 0.20 * range_score
            + 0.15 * agg_score
            + 0.10 * sort_score
        )
        semantic_score = round(semantic_score, 4)

        messages = term_messages + range_messages + agg_messages + sort_messages

        return semantic_score, {
            "expected_fields": sorted(expected_fields),
            "predicted_fields": sorted(predicted_fields),
            "expected_terms": expected_terms,
            "predicted_terms": predicted_terms,
            "expected_ranges": expected_ranges,
            "predicted_ranges": predicted_ranges,
            "expected_aggs": expected_aggs,
            "predicted_aggs": predicted_aggs,
            "expected_sort": expected_sort,
            "predicted_sort": predicted_sort,
            "field_overlap_score": field_overlap_score,
            "terms_score": terms_score,
            "range_score": range_score,
            "agg_score": agg_score,
            "sort_score": sort_score,
            "messages": messages,
        }

    def extract_referenced_fields(self, query_dsl: dict[str, Any]) -> set[str]:
        fields: set[str] = set()

        def visit(node: Any, parent_key: str | None = None) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in self.BOOL_KEYS:
                        visit(value, parent_key=key)
                        continue

                    if key == "bool":
                        visit(value, parent_key="bool")
                        continue

                    if key in {"term", "range", "match", "match_phrase", "wildcard", "prefix", "regexp"} and isinstance(value, dict):
                        for field_name, field_clause in value.items():
                            fields.add(field_name)
                            visit(field_clause, parent_key=key)
                        continue

                    if key == "terms" and isinstance(value, dict):
                        if "field" in value:
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

                    if key == "aggs" and isinstance(value, dict):
                        for _, agg_body in value.items():
                            visit(agg_body, parent_key="aggs")
                        continue

                    if key in self.AGG_FIELD_TYPES and isinstance(value, dict):
                        field_value = value.get("field")
                        if isinstance(field_value, str):
                            fields.add(field_value)

                        for inner_key, inner_value in value.items():
                            if inner_key != "field":
                                visit(inner_value, parent_key=key)
                        continue

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

    def _extract_terms_constraints(self, query_dsl: dict[str, Any]) -> dict[str, list[str]]:
        constraints: dict[str, set[str]] = {}

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "term" and isinstance(value, dict):
                        for field_name, field_value in value.items():
                            constraints.setdefault(field_name, set()).add(str(field_value))
                    elif key == "terms" and isinstance(value, dict):
                        if "field" not in value:
                            for field_name, field_values in value.items():
                                if isinstance(field_values, list):
                                    for item in field_values:
                                        constraints.setdefault(field_name, set()).add(str(item))
                                else:
                                    constraints.setdefault(field_name, set()).add(str(field_values))
                    else:
                        visit(value)
            elif isinstance(node, list):
                for item in node:
                    visit(item)

        visit(query_dsl)
        return {k: sorted(v) for k, v in constraints.items()}

    def _extract_range_constraints(self, query_dsl: dict[str, Any]) -> dict[str, dict[str, Any]]:
        constraints: dict[str, dict[str, Any]] = {}

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "range" and isinstance(value, dict):
                        for field_name, range_body in value.items():
                            if isinstance(range_body, dict):
                                constraints[field_name] = {
                                    k: range_body[k]
                                    for k in sorted(range_body.keys())
                                }
                    else:
                        visit(value)
            elif isinstance(node, list):
                for item in node:
                    visit(item)

        visit(query_dsl)
        return constraints

    def _extract_agg_signature(self, query_dsl: dict[str, Any]) -> list[dict[str, Any]]:
        signatures: list[dict[str, Any]] = []

        aggs = query_dsl.get("aggs", {})
        if not isinstance(aggs, dict):
            return signatures

        def walk_aggs(agg_dict: dict[str, Any]) -> None:
            for agg_name, agg_body in agg_dict.items():
                if not isinstance(agg_body, dict):
                    continue
                for agg_type, agg_cfg in agg_body.items():
                    if agg_type == "aggs" and isinstance(agg_cfg, dict):
                        walk_aggs(agg_cfg)
                        continue
                    if isinstance(agg_cfg, dict):
                        signatures.append(
                            {
                                "name": agg_name,
                                "type": agg_type,
                                "field": agg_cfg.get("field"),
                                "calendar_interval": agg_cfg.get("calendar_interval"),
                                "fixed_interval": agg_cfg.get("fixed_interval"),
                            }
                        )
                        if "aggs" in agg_cfg and isinstance(agg_cfg["aggs"], dict):
                            walk_aggs(agg_cfg["aggs"])

        walk_aggs(aggs)
        return sorted(
            signatures,
            key=lambda x: (str(x.get("type")), str(x.get("field")), str(x.get("calendar_interval"))),
        )

    def _extract_sort_signature(self, query_dsl: dict[str, Any]) -> list[dict[str, Any]]:
        signature: list[dict[str, Any]] = []
        sort = query_dsl.get("sort", [])
        if isinstance(sort, list):
            for item in sort:
                if isinstance(item, dict):
                    for field, cfg in item.items():
                        if isinstance(cfg, dict):
                            signature.append({"field": field, "order": cfg.get("order", "asc")})
                        else:
                            signature.append({"field": field, "order": "asc"})
        elif isinstance(sort, dict):
            for field, cfg in sort.items():
                if isinstance(cfg, dict):
                    signature.append({"field": field, "order": cfg.get("order", "asc")})
                else:
                    signature.append({"field": field, "order": "asc"})
        return signature

    def _compare_terms_constraints(
        self,
        expected: dict[str, list[str]],
        predicted: dict[str, list[str]],
    ) -> tuple[float, list[str]]:
        if not expected and not predicted:
            return 1.0, []
        if expected and not predicted:
            return 0.0, ["Missing expected term/terms constraints."]
        if predicted and not expected:
            return 0.0, ["Predicted unexpected term/terms constraints."]

        scores: list[float] = []
        messages: list[str] = []

        all_fields = sorted(set(expected.keys()) | set(predicted.keys()))
        for field in all_fields:
            exp_values = set(expected.get(field, []))
            pred_values = set(predicted.get(field, []))
            field_score = self._jaccard_score(exp_values, pred_values)
            scores.append(field_score)

            if exp_values != pred_values:
                messages.append(
                    f"Mismatch in constrained values for field '{field}': "
                    f"expected={sorted(exp_values)}, predicted={sorted(pred_values)}."
                )

        return round(sum(scores) / max(1, len(scores)), 4), messages

    def _compare_range_constraints(
        self,
        expected: dict[str, dict[str, Any]],
        predicted: dict[str, dict[str, Any]],
    ) -> tuple[float, list[str]]:
        if not expected and not predicted:
            return 1.0, []
        if expected and not predicted:
            return 0.0, ["Missing expected range constraints."]
        if predicted and not expected:
            return 0.0, ["Predicted unexpected range constraints."]

        scores: list[float] = []
        messages: list[str] = []

        all_fields = sorted(set(expected.keys()) | set(predicted.keys()))
        for field in all_fields:
            exp_range = expected.get(field)
            pred_range = predicted.get(field)

            if exp_range == pred_range:
                scores.append(1.0)
            else:
                scores.append(0.0)
                messages.append(
                    f"Mismatch in range constraint for field '{field}': "
                    f"expected={exp_range}, predicted={pred_range}."
                )

        return round(sum(scores) / max(1, len(scores)), 4), messages

    def _compare_agg_signature(
        self,
        expected: list[dict[str, Any]],
        predicted: list[dict[str, Any]],
    ) -> tuple[float, list[str]]:
        if not expected and not predicted:
            return 1.0, []
        if expected and not predicted:
            return 0.0, ["Missing expected aggregations."]
        if predicted and not expected:
            return 0.0, ["Predicted unexpected aggregations."]

        expected_norm = [
            {
                "type": item.get("type"),
                "field": item.get("field"),
                "calendar_interval": item.get("calendar_interval"),
                "fixed_interval": item.get("fixed_interval"),
            }
            for item in expected
        ]
        predicted_norm = [
            {
                "type": item.get("type"),
                "field": item.get("field"),
                "calendar_interval": item.get("calendar_interval"),
                "fixed_interval": item.get("fixed_interval"),
            }
            for item in predicted
        ]

        score = self._jaccard_score(
            {str(x) for x in expected_norm},
            {str(x) for x in predicted_norm},
        )

        messages: list[str] = []
        if score < 1.0:
            messages.append(
                f"Aggregation signature mismatch: expected={expected_norm}, predicted={predicted_norm}."
            )

        return score, messages

    def _compare_sort_signature(
        self,
        expected: list[dict[str, Any]],
        predicted: list[dict[str, Any]],
    ) -> tuple[float, list[str]]:
        if not expected and not predicted:
            return 1.0, []
        if expected and not predicted:
            return 0.0, ["Missing expected sort clause."]
        if predicted and not expected:
            return 0.0, ["Predicted unexpected sort clause."]

        score = self._jaccard_score(
            {str(x) for x in expected},
            {str(x) for x in predicted},
        )

        messages: list[str] = []
        if score < 1.0:
            messages.append(f"Sort signature mismatch: expected={expected}, predicted={predicted}.")

        return score, messages

    @staticmethod
    def _jaccard_score(a: set[Any], b: set[Any]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return round(len(a & b) / max(1, len(a | b)), 4)

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
