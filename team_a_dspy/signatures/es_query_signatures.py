from __future__ import annotations

import dspy


class SchemaStocktakeSignature(dspy.Signature):
    """
    Understand the schema and user intent before constructing Query DSL.
    """

    user_question = dspy.InputField(desc="Natural language user question")
    schema_context = dspy.InputField(desc="Relevant Elasticsearch schema / mapping notes")
    today_iso_date = dspy.InputField(
        desc="Today's date as ISO format YYYY-MM-DD for relative date normalization"
    )

    relevant_fields = dspy.OutputField(
        desc="Comma-separated exact Elasticsearch fields to use"
    )
    entity_type = dspy.OutputField(
        desc="Primary requested entity type, e.g. people, countries, locations, events, themes, articles"
    )
    timeframe = dspy.OutputField(
        desc=(
            "Normalized time window. "
            "For V21Date, always express dates in ISO format such as YYYY-MM-DD or "
            "YYYY-MM-DDTHH:MM:SS.sssZ. Never use compact YYYYMMDD for V21Date."
        )
    )
    region_scope = dspy.OutputField(
        desc="Geographic scope or region constraint, if any"
    )
    metric_definition = dspy.OutputField(
        desc="How the question should be operationalized into measurable Elasticsearch logic"
    )
    stocktake_notes = dspy.OutputField(
        desc="Short reasoning summary of the schema and task shape"
    )


class FilterPlanSignature(dspy.Signature):
    """
    Convert stocktake into concrete filter logic.
    """

    user_question = dspy.InputField()
    schema_context = dspy.InputField()
    today_iso_date = dspy.InputField()
    relevant_fields = dspy.InputField()
    entity_type = dspy.InputField()
    timeframe = dspy.InputField()
    region_scope = dspy.InputField()
    metric_definition = dspy.InputField()

    filter_plan = dspy.OutputField(
        desc=(
            "Concrete Elasticsearch filter logic using exact fields. "
            "If filtering on V21Date, use ISO date or ISO datetime literals only."
        )
    )
    requested_count = dspy.OutputField(
        desc="Requested result count or top-N count"
    )
    size_strategy = dspy.OutputField(
        desc="Either 'size=0' for ranking/aggregation or 'size>0' for document retrieval"
    )
    filter_notes = dspy.OutputField(
        desc="Short reasoning summary of filters and size strategy"
    )


class AggregationPlanSignature(dspy.Signature):
    """
    Decide aggregation/ranking/measurement strategy.
    """

    user_question = dspy.InputField()
    schema_context = dspy.InputField()
    relevant_fields = dspy.InputField()
    entity_type = dspy.InputField()
    metric_definition = dspy.InputField()
    filter_plan = dspy.InputField()
    requested_count = dspy.InputField()
    size_strategy = dspy.InputField()

    aggregation_plan = dspy.OutputField(
        desc="Precise aggregation/sort/ranking plan, including grouping field and ordering"
    )
    validation_checklist = dspy.OutputField(
        desc=(
            "Checklist of what final DSL must preserve, including correct date field choice "
            "and correct literal format for date ranges."
        )
    )
    aggregation_notes = dspy.OutputField(
        desc="Short reasoning summary of aggregation strategy"
    )


class QuerySynthesisSignature(dspy.Signature):
    """
    Produce final Elasticsearch Query DSL JSON only.
    """

    user_question = dspy.InputField()
    schema_context = dspy.InputField()
    today_iso_date = dspy.InputField()
    relevant_fields = dspy.InputField()
    entity_type = dspy.InputField()
    timeframe = dspy.InputField()
    region_scope = dspy.InputField()
    metric_definition = dspy.InputField()
    filter_plan = dspy.InputField()
    requested_count = dspy.InputField()
    size_strategy = dspy.InputField()
    aggregation_plan = dspy.InputField()
    validation_checklist = dspy.InputField()

    querydsl_query = dspy.OutputField(
        desc=(
            "Return exactly one valid Elasticsearch Query DSL JSON object. "
            "No markdown. No prose. No code fences. Read-only only. "
            "If using V21Date in a range query, use ISO date strings like "
            "'2026-03-09' or ISO datetimes like '2026-03-09T00:00:00.000Z'. "
            "Never use compact forms like '20260309' for V21Date."
        )
    )