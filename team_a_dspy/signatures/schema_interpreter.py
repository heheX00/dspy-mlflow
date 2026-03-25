from __future__ import annotations

import re
from typing import Any

import dspy

from services.chroma_client import ChromaClient


class SchemaInterpreter(dspy.Signature):
    """
    Analyze a GDELT field and produce a concise usage-oriented interpretation.
    """

    field_name: str = dspy.InputField(desc="Name of the field in the Elasticsearch mapping.")
    field_type: str = dspy.InputField(desc="Elasticsearch field type.")
    sample_values: str = dspy.InputField(desc="Representative sample values from recent documents.")

    interpretation: str = dspy.OutputField(
        desc="Short explanation of the field meaning, common aliases, and how it should be queried."
    )


class DataAwareSchemaInterpreter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.interpret = dspy.Predict(SchemaInterpreter)

    def forward(self, field_name: str, field_type: str, sample_values: str) -> str:
        prediction = self.interpret(
            field_name=field_name,
            field_type=field_type,
            sample_values=sample_values,
        )
        return str(prediction.interpretation)


class SchemaRetriever:
    """
    Deterministic schema retriever.

    This is intentionally a plain Python class instead of a dspy.Module so that
    optimizer clones do not attempt to deep-copy external Chroma client state.
    """

    STOPWORDS = {
        "the",
        "a",
        "an",
        "of",
        "for",
        "to",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "about",
        "show",
        "find",
        "get",
        "give",
        "list",
        "top",
        "latest",
        "recent",
        "news",
        "articles",
        "article",
        "mentioned",
        "last",
        "this",
        "that",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "what",
        "which",
        "who",
        "how",
        "many",
        "much",
        "overall",
        "over",
        "past",
        "week",
        "weeks",
        "day",
        "days",
        "month",
        "months",
        "year",
        "years",
        "yesterday",
        "today",
    }

    def __init__(self, chroma_client: ChromaClient, k_primary: int = 10, k_fallback: int = 5):
        self.chroma_client = chroma_client
        self.k_primary = k_primary
        self.k_fallback = k_fallback

    def __call__(self, nl_query: str) -> str:
        return self.forward(nl_query)

    def forward(self, nl_query: str) -> str:
        relevant_schema: dict[str, dict[str, Any]] = {}

        primary_results = self.chroma_client.query(query_text=nl_query, k=self.k_primary)
        for item in self.flatten_chroma_results(primary_results):
            relevant_schema[item["field_name"]] = item

        if len(relevant_schema) < 10:
            for token in self._expand_query_terms(nl_query):
                token_results = self.chroma_client.query(query_text=token, k=self.k_fallback)
                for item in self.flatten_chroma_results(token_results):
                    relevant_schema[item["field_name"]] = item
                if len(relevant_schema) >= 15:
                    break

        if not relevant_schema:
            return "No relevant schema information found."

        passages: list[str] = []
        for field_name in sorted(relevant_schema.keys()):
            field = relevant_schema[field_name]
            passages.append(
                "\n".join(
                    [
                        f"Field: {field['field_name']}",
                        f"Type: {field['field_type']}",
                        f"Usage: {field['interpretation']}",
                    ]
                )
            )
        return "\n---\n".join(passages)

    def _expand_query_terms(self, nl_query: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9_.-]+", nl_query.lower())
        base_terms = [token for token in tokens if len(token) > 2 and token not in self.STOPWORDS]

        heuristic_terms: list[str] = []
        joined = " ".join(base_terms)

        if any(word in joined for word in ["country", "countries", "iran", "syria", "saudi", "japan", "ukraine"]):
            heuristic_terms.extend(["country code", "country", "location", "region"])
        if any(word in joined for word in ["person", "people", "leader", "president", "minister"]):
            heuristic_terms.extend(["person", "people", "entity person"])
        if any(word in joined for word in ["organization", "company", "org", "source"]):
            heuristic_terms.extend(["organization", "company", "source", "news source"])
        if any(word in joined for word in ["tone", "sentiment", "negative", "positive", "polarity"]):
            heuristic_terms.extend(["tone", "sentiment", "polarity", "negative score", "positive score"])
        if any(word in joined for word in ["theme", "topic", "military", "conflict", "disaster"]):
            heuristic_terms.extend(["theme", "topic", "enhanced themes"])
        if any(word in joined for word in ["date", "time", "trend", "yesterday", "today", "week", "month"]):
            heuristic_terms.extend(["date", "time", "timestamp", "histogram"])

        seen: set[str] = set()
        expanded: list[str] = []
        for term in base_terms + heuristic_terms:
            if term not in seen:
                seen.add(term)
                expanded.append(term)
        return expanded[:15]

    @staticmethod
    def flatten_chroma_results(raw_results: dict[str, Any]) -> list[dict[str, Any]]:
        flattened_schema: list[dict[str, Any]] = []

        ids = raw_results.get("ids") or []
        documents = raw_results.get("documents") or []
        metadatas = raw_results.get("metadatas") or []
        distances = raw_results.get("distances") or []

        if not ids or not ids[0]:
            return flattened_schema

        row_ids = ids[0]
        row_docs = documents[0] if documents else []
        row_metadatas = metadatas[0] if metadatas else []
        row_distances = distances[0] if distances else [None] * len(row_ids)

        for field_id, doc_string, metadata, distance in zip(row_ids, row_docs, row_metadatas, row_distances):
            meta = metadata if isinstance(metadata, dict) else {}
            flattened_schema.append(
                {
                    "field_name": meta.get("field_name", field_id),
                    "field_type": meta.get("field_type", "unknown"),
                    "distance_score": round(float(distance), 4) if distance is not None else None,
                    "interpretation": str(doc_string),
                }
            )
        return flattened_schema