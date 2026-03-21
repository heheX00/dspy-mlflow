from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from team_a_dspy.utils.config import settings
from team_a_dspy.utils.es_client import get_es_client


@dataclass(frozen=True)
class FieldInfo:
    name: str
    es_type: str


class ElasticsearchSchemaRetriever:
    """
    Retrieve, flatten, normalize, and cache Elasticsearch mappings.

    Keeps raw mapping retrieval cheap by caching flattened fields in memory.
    """

    def __init__(
        self,
        index_name: str | None = None,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        self.index_name = index_name or settings.es_index
        self.cache_ttl_seconds = cache_ttl_seconds or settings.schema_cache_ttl_seconds
        self._lock = threading.Lock()
        self._cached_at: float = 0.0
        self._cached_fields: List[FieldInfo] = []

    def get_flat_fields(self, force_refresh: bool = False) -> List[FieldInfo]:
        now = time.time()

        with self._lock:
            cache_valid = (
                not force_refresh
                and self._cached_fields
                and (now - self._cached_at) < self.cache_ttl_seconds
            )
            if cache_valid:
                return list(self._cached_fields)

            client = get_es_client()
            mapping_response = client.indices.get_mapping(index=self.index_name)
            fields = self._flatten_mapping(mapping_response)
            self._cached_fields = fields
            self._cached_at = now
            return list(self._cached_fields)

    def build_raw_schema_context(
        self,
        fields: Iterable[FieldInfo] | None = None,
        limit: int | None = None,
    ) -> str:
        field_list = list(fields) if fields is not None else self.get_flat_fields()

        if limit is not None:
            field_list = field_list[:limit]

        if not field_list:
            return "No schema fields found."

        chunks: List[str] = []
        for f in field_list:
            chunks.append(f"Field: {f.name}\nType: {f.es_type}")

        return "\n\n".join(chunks)

    def build_candidate_schema_context(
        self,
        user_question: str,
        max_candidates: int | None = None,
    ) -> str:
        max_candidates = max_candidates or settings.schema_candidate_field_limit
        all_fields = self.get_flat_fields()
        ranked = self.rank_fields_for_question(user_question, all_fields)
        selected = [field for field, _score in ranked[:max_candidates]]
        return self.build_raw_schema_context(selected)

    def rank_fields_for_question(
        self,
        user_question: str,
        fields: Iterable[FieldInfo],
    ) -> List[Tuple[FieldInfo, float]]:
        """
        Cheap heuristic ranking to reduce LLM input size before DSPy enrichment.
        """
        q = user_question.lower()
        q_tokens = self._tokenize(q)

        alias_boosts = {
            "people": ["V2Persons.V1Person", "V2Persons.V1Person.keyword"],
            "person": ["V2Persons.V1Person", "V2Persons.V1Person.keyword"],
            "individual": ["V2Persons.V1Person", "V2Persons.V1Person.keyword"],
            "org": ["V2Organizations.V2Organization", "V2Orgs.V1Org", "V2Orgs.V1Org.keyword"],
            "organisation": ["V2Organizations.V2Organization", "V2Orgs.V1Org", "V2Orgs.V1Org.keyword"],
            "organization": ["V2Organizations.V2Organization", "V2Orgs.V1Org", "V2Orgs.V1Org.keyword"],
            "country": ["V2Locations.CountryCode"],
            "iran": ["V2Locations.CountryCode", "V2Locations.FullName", "V2Locations.FullName.keyword"],
            "negative": [
                "V15Tone.Tone",
                "V15Tone.NegativeScore",
                "V15Tone.Polarity",
            ],
            "tone": ["V15Tone.Tone"],
            "sentiment": ["V15Tone.Tone", "V15Tone.NegativeScore", "V15Tone.Polarity"],
            "date": ["V21Date"],
            "week": ["V21Date"],
            "month": ["V21Date"],
            "today": ["V21Date"],
            "yesterday": ["V21Date"],
            "last": ["V21Date"],
            "recent": ["V21Date"],
            "theme": ["V2EnhancedThemes.V2Theme", "V2EnhancedThemes.V2Theme.keyword"],
            "location": ["V2Locations.FullName", "V2Locations.FullName.keyword", "V2Locations.CountryCode"],
        }

        scored: List[Tuple[FieldInfo, float]] = []

        for field in fields:
            score = 0.0
            field_tokens = self._tokenize(field.name)

            overlap = len(q_tokens.intersection(field_tokens))
            score += overlap * 2.0

            if "keyword" in field.es_type:
                score += 0.2

            if field.name == "V21Date":
                if any(t in q for t in ["day", "week", "month", "year", "today", "yesterday", "last", "recent"]):
                    score += 6.0

            if field.name in {"V15Tone.Tone", "V15Tone.NegativeScore", "V15Tone.Polarity"}:
                if any(t in q for t in ["negative", "positive", "tone", "sentiment"]):
                    score += 5.0

            for trigger, boosted_fields in alias_boosts.items():
                if trigger in q and field.name in boosted_fields:
                    score += 7.0

            if any(
                important in field.name
                for important in [
                    "V2Persons",
                    "V2Locations",
                    "V2Orgs",
                    "V2Organizations",
                    "V15Tone",
                    "V21Date",
                    "V2EnhancedThemes",
                ]
            ):
                score += 0.5

            scored.append((field, score))

        scored.sort(key=lambda x: (-x[1], x[0].name))
        return scored

    def _flatten_mapping(self, mapping_response: Dict[str, Any]) -> List[FieldInfo]:
        index_payload = next(iter(mapping_response.values()), {})
        mappings = index_payload.get("mappings", {})
        root_properties = mappings.get("properties", {})

        fields: Dict[str, FieldInfo] = {}
        self._walk_properties(prefix="", properties=root_properties, out=fields)

        return sorted(fields.values(), key=lambda x: x.name)

    def _walk_properties(
        self,
        prefix: str,
        properties: Dict[str, Any],
        out: Dict[str, FieldInfo],
    ) -> None:
        for field_name, node in properties.items():
            full_name = f"{prefix}.{field_name}" if prefix else field_name

            field_type = self._normalize_type(node)
            if field_type:
                out[full_name] = FieldInfo(name=full_name, es_type=field_type)

            sub_properties = node.get("properties", {})
            if sub_properties:
                self._walk_properties(full_name, sub_properties, out)

            for subfield_name, subfield_node in node.get("fields", {}).items():
                subfield_full_name = f"{full_name}.{subfield_name}"
                subfield_type = self._normalize_type(subfield_node)
                if subfield_type:
                    out[subfield_full_name] = FieldInfo(
                        name=subfield_full_name,
                        es_type=subfield_type,
                    )

    def _normalize_type(self, node: Dict[str, Any]) -> str:
        base_type = node.get("type")
        subfield_types = [
            sub.get("type")
            for sub in node.get("fields", {}).values()
            if isinstance(sub, dict) and sub.get("type")
        ]

        types: List[str] = []
        if base_type:
            types.append(base_type)

        for t in subfield_types:
            if t not in types:
                types.append(t)

        return " + ".join(types)

    def _tokenize(self, text: str) -> set[str]:
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = text.replace(".", " ").replace("_", " ")
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return set(tokens)