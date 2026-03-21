from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict

import dspy

from team_a_dspy.modules.schema_enricher import QueryAwareSchemaEnricher
from team_a_dspy.signatures.es_query_signatures import (
    AggregationPlanSignature,
    FilterPlanSignature,
    QuerySynthesisSignature,
    SchemaStocktakeSignature,
)
from team_a_dspy.utils.config import settings
from team_a_dspy.utils.json_utils import extract_first_json_object
from team_a_dspy.utils.schema_loader import load_schema_context
from team_a_dspy.utils.schema_retriever import ElasticsearchSchemaRetriever


FORBIDDEN_PATTERNS = [
    r'"script"\s*:',
    r'"update"\s*:',
    r'"delete"\s*:',
    r'"index"\s*:',
    r'"create"\s*:',
    r'"upsert"\s*:',
    r'"runtime_mappings"\s*:',
]

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISO_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$"
)
COMPACT_DATE_RE = re.compile(r"^\d{8}$")
COMPACT_DATETIME_RE = re.compile(r"^\d{14}$")


class QuerySafetyError(ValueError):
    pass


def validate_read_only_query(query: Dict[str, Any]) -> None:
    if not isinstance(query, dict):
        raise QuerySafetyError("Final output must be a JSON object")

    raw = json.dumps(query, ensure_ascii=False)

    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, raw, flags=re.IGNORECASE):
            raise QuerySafetyError(f"Forbidden construct detected: {pattern}")

    forbidden_top_level = {
        "script",
        "runtime_mappings",
        "update",
        "delete",
        "doc",
        "upsert",
    }
    for key in query.keys():
        if key in forbidden_top_level:
            raise QuerySafetyError(f"Forbidden top-level key: {key}")


def _iter_range_clauses(node: Any):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "range" and isinstance(value, dict):
                for field_name, spec in value.items():
                    if isinstance(spec, dict):
                        yield field_name, spec
            yield from _iter_range_clauses(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_range_clauses(item)


def _is_iso_date_like(value: str) -> bool:
    return bool(ISO_DATE_RE.match(value) or ISO_DATETIME_RE.match(value))


def _is_compact_date_like(value: str) -> bool:
    return bool(COMPACT_DATE_RE.match(value) or COMPACT_DATETIME_RE.match(value))


def validate_date_ranges(query: Dict[str, Any]) -> None:
    for field_name, spec in _iter_range_clauses(query):
        if field_name == "V21Date":
            for bound in ("gte", "gt", "lte", "lt"):
                if bound not in spec:
                    continue
                value = spec[bound]

                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    raise QuerySafetyError(
                        f"V21Date range uses numeric {bound}={value}. "
                        "Use ISO date strings for V21Date."
                    )

                if isinstance(value, str):
                    if _is_compact_date_like(value):
                        raise QuerySafetyError(
                            f"V21Date range uses compact date literal '{value}'. "
                            "Use ISO date strings like '2026-03-09' or "
                            "'2026-03-09T00:00:00.000Z'."
                        )
                    if not _is_iso_date_like(value):
                        raise QuerySafetyError(
                            f"V21Date range uses unsupported literal '{value}'. "
                            "Use ISO date or ISO datetime strings."
                        )

        if field_name == "GkgRecordId.Date":
            for bound in ("gte", "gt", "lte", "lt"):
                if bound not in spec:
                    continue
                value = spec[bound]
                if isinstance(value, str) and _is_iso_date_like(value):
                    raise QuerySafetyError(
                        "GkgRecordId.Date range is using ISO date strings. "
                        "Use compact yyyyMMddHHmmss-style string values for GkgRecordId.Date, "
                        "or prefer V21Date for time filtering."
                    )


def validate_query_shape(user_question: str, query: Dict[str, Any]) -> None:
    q = user_question.lower()

    ranking_markers = [
        "top ",
        "most ",
        "highest ",
        "lowest ",
        "safest ",
        "rank ",
        "which country",
        "which countries",
    ]

    if any(marker in q for marker in ranking_markers):
        if query.get("size") != 0:
            raise QuerySafetyError("Ranking/aggregation question should use top-level size=0")
        if "aggs" not in query and "aggregations" not in query:
            raise QuerySafetyError("Ranking/aggregation question is missing aggs")

    if "query" not in query:
        raise QuerySafetyError("Missing top-level query clause")

    validate_date_ranges(query)


class ESQueryDSPyPipeline(dspy.Module):
    """
    Multi-stage DSPy pipeline with query-aware schema enrichment.

    Flow:
    1) Retrieve raw schema dynamically from Elasticsearch mapping
    2) Select candidate fields cheaply
    3) Enrich those fields per query using DSPy
    4) Run existing stocktake/filter/aggregation/query synthesis stages
    """

    def __init__(self) -> None:
        super().__init__()
        self.schema_retriever = ElasticsearchSchemaRetriever(index_name=settings.es_index)
        self.schema_enricher = QueryAwareSchemaEnricher()

        self.stocktake = dspy.ChainOfThought(SchemaStocktakeSignature)
        self.filter_planner = dspy.ChainOfThought(FilterPlanSignature)
        self.agg_planner = dspy.ChainOfThought(AggregationPlanSignature)
        self.query_writer = dspy.ChainOfThought(QuerySynthesisSignature)

    def _resolve_schema_context(
        self,
        user_question: str,
        schema_context: str | None,
    ) -> Dict[str, str]:
        """
        Preserve backward compatibility:
        - if schema_mode=static and schema_context is provided -> use it
        - else use dynamic mapping retrieval + query-aware enrichment
        """
        if settings.schema_mode == "static":
            resolved = schema_context or load_schema_context()
            return {
                "raw_candidate_schema": resolved,
                "effective_schema_context": resolved,
            }

        raw_candidate_schema = self.schema_retriever.build_candidate_schema_context(
            user_question=user_question,
            max_candidates=settings.schema_candidate_field_limit,
        )

        enriched = self.schema_enricher(
            user_question=user_question,
            raw_schema_context=raw_candidate_schema,
            max_fields=settings.schema_enriched_field_limit,
        )

        effective_schema_context = (enriched.enriched_schema or "").strip() or raw_candidate_schema

        return {
            "raw_candidate_schema": raw_candidate_schema,
            "effective_schema_context": effective_schema_context,
        }

    def forward(
        self,
        user_question: str,
        schema_context: str | None = None,
        today_iso_date: str | None = None,
    ) -> dspy.Prediction:
        today_iso_date = today_iso_date or datetime.utcnow().strftime("%Y-%m-%d")

        schema_bundle = self._resolve_schema_context(
            user_question=user_question,
            schema_context=schema_context,
        )
        effective_schema_context = schema_bundle["effective_schema_context"]

        stock = self.stocktake(
            user_question=user_question,
            schema_context=effective_schema_context,
            today_iso_date=today_iso_date,
        )

        filt = self.filter_planner(
            user_question=user_question,
            schema_context=effective_schema_context,
            today_iso_date=today_iso_date,
            relevant_fields=stock.relevant_fields,
            entity_type=stock.entity_type,
            timeframe=stock.timeframe,
            region_scope=stock.region_scope,
            metric_definition=stock.metric_definition,
        )

        agg = self.agg_planner(
            user_question=user_question,
            schema_context=effective_schema_context,
            relevant_fields=stock.relevant_fields,
            entity_type=stock.entity_type,
            metric_definition=stock.metric_definition,
            filter_plan=filt.filter_plan,
            requested_count=filt.requested_count,
            size_strategy=filt.size_strategy,
        )

        final_pred = self.query_writer(
            user_question=user_question,
            schema_context=effective_schema_context,
            today_iso_date=today_iso_date,
            relevant_fields=stock.relevant_fields,
            entity_type=stock.entity_type,
            timeframe=stock.timeframe,
            region_scope=stock.region_scope,
            metric_definition=stock.metric_definition,
            filter_plan=filt.filter_plan,
            requested_count=filt.requested_count,
            size_strategy=filt.size_strategy,
            aggregation_plan=agg.aggregation_plan,
            validation_checklist=agg.validation_checklist,
        )

        query = extract_first_json_object(final_pred.querydsl_query)
        validate_read_only_query(query)
        validate_query_shape(user_question, query)

        return dspy.Prediction(
            querydsl_query=json.dumps(query, ensure_ascii=False, indent=2),
            parsed_query=query,
            raw_candidate_schema=schema_bundle["raw_candidate_schema"],
            effective_schema_context=effective_schema_context,
            stocktake={
                "relevant_fields": stock.relevant_fields,
                "entity_type": stock.entity_type,
                "timeframe": stock.timeframe,
                "region_scope": stock.region_scope,
                "metric_definition": stock.metric_definition,
                "notes": stock.stocktake_notes,
            },
            filter_plan={
                "filter_plan": filt.filter_plan,
                "requested_count": filt.requested_count,
                "size_strategy": filt.size_strategy,
                "notes": filt.filter_notes,
            },
            aggregation_plan={
                "aggregation_plan": agg.aggregation_plan,
                "validation_checklist": agg.validation_checklist,
                "notes": agg.aggregation_notes,
            },
        )