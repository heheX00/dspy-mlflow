from __future__ import annotations

import dspy


class QueryAwareSchemaEnrichmentSignature(dspy.Signature):
    """
    Turn raw mapping fields into a concise, query-aware schema context.

    Keep only fields that matter for answering the user's question.
    """

    user_question = dspy.InputField(
        desc="Natural language analytics question from the user."
    )
    raw_schema_context = dspy.InputField(
        desc=(
            "Candidate Elasticsearch schema fields in the form "
            "'Field: <name>\\nType: <type>'."
        )
    )
    max_fields = dspy.InputField(
        desc="Maximum number of fields to keep in the final enriched schema."
    )

    enriched_schema = dspy.OutputField(
        desc=(
            "Return only the most relevant fields. For each field, use exactly:\n"
            "Field: <exact field name>\n"
            "Type: <exact type>\n"
            "Description: <what the field represents>\n"
            "Relevance: <how it is used for this query: filter, aggregation, grouping, date, sorting, etc>\n\n"
            "Rules:\n"
            "- Keep output concise.\n"
            "- Keep exact field names unchanged.\n"
            "- Ignore irrelevant fields.\n"
            "- Prefer 4 to 8 fields unless max_fields requires otherwise.\n"
            "- Do not invent fields.\n"
            "- Do not include prose before or after the field blocks.\n"
            "- If the field is V21Date, explicitly say it is an ISO-8601 date/datetime field.\n"
            "- If the field is V21Date, explicitly say range literals must use ISO strings like "
            "2026-03-19 or 2026-03-19T01:00:00.000Z.\n"
            "- Never describe V21Date as compact YYYYMMDD."
        )
    )


class QueryAwareSchemaEnricher(dspy.Module):
    """
    Lightweight DSPy module that enriches candidate schema fields per query.
    """

    def __init__(self) -> None:
        super().__init__()
        self.enrich = dspy.ChainOfThought(QueryAwareSchemaEnrichmentSignature)

    def forward(
        self,
        user_question: str,
        raw_schema_context: str,
        max_fields: int = 8,
    ) -> dspy.Prediction:
        pred = self.enrich(
            user_question=user_question,
            raw_schema_context=raw_schema_context,
            max_fields=str(max_fields),
        )

        enriched_schema = (pred.enriched_schema or "").strip()
        return dspy.Prediction(enriched_schema=enriched_schema)