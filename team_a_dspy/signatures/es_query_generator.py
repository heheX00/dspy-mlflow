from __future__ import annotations

import dspy

from services.chroma_client import ChromaClient
from services.judge_dspy import JudgeDSPY
from signatures.schema_interpreter import SchemaRetriever


class NLToQuerySignature(dspy.Signature):
    """
    Convert a natural-language request into Elasticsearch Query DSL for the GDELT index.

    Hard rules:
    - Only use fields that appear in es_schema.
    - Never use scripts.
    - For top-N / ranking requests, prefer size=0 with aggregations.
    - For record retrieval requests, return hits with a small size and useful _source fields.
    - For time-bounded requests, include a range on the real date field from es_schema.
    - Do not invent fields such as date, title, text, content, category, country unless they explicitly exist in es_schema.
    - Output only a JSON object in query_dsl.
    """

    nl_query: str = dspy.InputField(desc="Natural-language question.")
    es_schema: str = dspy.InputField(desc="Retrieved schema snippets with field names, types, and usage guidance.")

    query_dsl: dict = dspy.OutputField(desc="Elasticsearch Query DSL JSON object.")


class RefinerSignature(dspy.Signature):
    """
    Repair a previously generated Elasticsearch Query DSL using judge feedback.
    Preserve the original user intent and only use fields that exist in es_schema.
    Output only a JSON object in query_dsl.
    """

    nl_query: str = dspy.InputField(desc="Original natural-language question.")
    es_schema: str = dspy.InputField(desc="Retrieved schema snippets with field names, types, and usage guidance.")
    failed_query_dsl: dict = dspy.InputField(desc="Previous invalid or low-quality Elasticsearch Query DSL.")
    feedback: str = dspy.InputField(desc="Judge feedback describing what is wrong.")

    query_dsl: dict = dspy.OutputField(desc="Repaired Elasticsearch Query DSL JSON object.")


class NLToQueryDSL(dspy.Module):
    def __init__(self, chroma_client: ChromaClient, dspy_judge: JudgeDSPY, max_refine_attempts: int = 3):
        super().__init__()
        self.schema_retriever = SchemaRetriever(chroma_client=chroma_client)
        self.generate_query = dspy.ChainOfThought(NLToQuerySignature)
        self.refiner = dspy.ChainOfThought(RefinerSignature)
        self.dspy_judge = dspy_judge
        self.max_refine_attempts = max_refine_attempts

    def forward(self, nl_query: str) -> dspy.Prediction:
        es_schema = self.schema_retriever(nl_query)
        generated = self.generate_query(nl_query=nl_query, es_schema=es_schema)
        current_query_dsl = generated.query_dsl

        for _ in range(self.max_refine_attempts + 1):
            judge_result = self.dspy_judge.evaluate_query_dsl(
                generated_query_dsl=current_query_dsl
            )
            if judge_result.get("is_valid"):
                return dspy.Prediction(query_dsl=current_query_dsl, es_schema=es_schema, judge_result=judge_result)

            repaired = self.refiner(
                nl_query=nl_query,
                es_schema=es_schema,
                failed_query_dsl=current_query_dsl,
                feedback=judge_result.get("feedback", "Unknown validation failure."),
            )
            current_query_dsl = repaired.query_dsl

        final_judge_result = _run_async(
            self.dspy_judge.evaluate_query_dsl(generated_query_dsl=current_query_dsl)
        )
        return dspy.Prediction(query_dsl=current_query_dsl, es_schema=es_schema, judge_result=final_judge_result)