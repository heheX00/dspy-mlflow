from __future__ import annotations

import ast
import json
from collections import Counter
from typing import Any

import dspy

from services.config import settings
from services.es_client import ESClient
from services.sandbox_es_client import SandboxESClient
from signatures.judge_relevance import RelevanceScorerSignature
from signatures.schema_interpreter import PlanSchemaRetriver


class JudgeDSPY:
    """Judge for syntax/execution validation and semantic relevance scoring."""

    def __init__(self, es_client: ESClient | SandboxESClient):
        self.es_client = es_client
        self.lm = dspy.LM(
            base_url=settings.llm_base_url,
            model=f"openai/{settings.llm_model_name}",
            api_key=settings.llm_api_key,
            temperature=0.0,
        )
        self.schema_planner = dspy.ChainOfThought(PlanSchemaRetriver)

    def evaluate_query_dsl(self, generated_query_dsl: dict, expected_query_dsl: dict | None = None) -> dict:
        evaluator = getattr(self.es_client, "evaluate_query_dsl", None)
        if callable(evaluator):
            return evaluator(query_dsl=generated_query_dsl, expected_query_dsl=expected_query_dsl)
        return self._evaluate_query_dsl_syntax(generated_query_dsl=generated_query_dsl)

    def _evaluate_query_dsl_syntax(self, generated_query_dsl: dict) -> dict:
        validator = getattr(self.es_client, "validate_query_dsl")
        return validator(query_dsl=generated_query_dsl)

    def _extract_query_intent(self, nl_query: str) -> list[str]:
        with dspy.context(lm=self.lm):
            plan = self.schema_planner(nl_query=nl_query)
            return [c.strip() for c in str(plan.search_terms).split(",") if c.strip()]

    def _aggregate_es_documents(self, raw_es_documents: list[dict]) -> dict[str, Any]:
        def _safe_get(dct: dict, *keys: str):
            current = dct
            for key in keys:
                if not isinstance(current, dict):
                    return None
                current = current.get(key)
            return current

        def _try_parse_json(value: str) -> dict | None:
            if not isinstance(value, str):
                return None
            value = value.strip()
            if not value:
                return None
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None

        def _extract_embedded_json(value: str) -> dict | None:
            if not isinstance(value, str):
                return None
            start = value.find("{")
            end = value.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            return _try_parse_json(value[start : end + 1])

        def _is_empty(value: Any) -> bool:
            if value is None:
                return True
            if isinstance(value, str) and not value.strip():
                return True
            if isinstance(value, (list, dict, tuple, set)) and len(value) == 0:
                return True
            return False

        def _merge_missing(base: dict, candidate: dict):
            for key, value in candidate.items():
                if key not in base or _is_empty(base.get(key)):
                    base[key] = value

        def _parse_list_like(value: Any) -> list[str]:
            if value is None:
                return []
            if isinstance(value, list):
                return [str(v).strip() for v in value if str(v).strip()]
            if isinstance(value, dict):
                for nested in value.values():
                    if isinstance(nested, list):
                        return [str(v).strip() for v in nested if str(v).strip()]
                return []
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    return []
                if stripped.startswith("[") and stripped.endswith("]"):
                    try:
                        parsed = json.loads(stripped)
                        if isinstance(parsed, list):
                            return [str(v).strip() for v in parsed if str(v).strip()]
                    except json.JSONDecodeError:
                        try:
                            parsed = ast.literal_eval(stripped)
                            if isinstance(parsed, list):
                                return [str(v).strip() for v in parsed if str(v).strip()]
                        except (ValueError, SyntaxError):
                            pass
                return [stripped]
            return [str(value).strip()] if str(value).strip() else []

        def _normalize_source(raw_doc: dict) -> dict:
            source = raw_doc.get("_source", raw_doc) if isinstance(raw_doc, dict) else {}
            if not isinstance(source, dict):
                return {}

            normalized = dict(source)

            event_original = _safe_get(source, "event", "original")
            parsed_original = _try_parse_json(event_original) if isinstance(event_original, str) else None
            if parsed_original:
                _merge_missing(normalized, parsed_original)

            message = source.get("message")
            parsed_message = _extract_embedded_json(message) if isinstance(message, str) else None
            if parsed_message:
                _merge_missing(normalized, parsed_message)

            return normalized

        theme_counts = Counter()
        person_counts = Counter()
        org_counts = Counter()
        location_counts = Counter()
        country_counts = Counter()
        event_code_counts = Counter()

        tone_sum = 0.0
        positive_sum = 0.0
        negative_sum = 0.0
        polarity_sum = 0.0
        tone_docs = 0
        total_docs = 0

        for raw_doc in raw_es_documents:
            normalized = _normalize_source(raw_doc)
            if not normalized:
                continue

            total_docs += 1

            themes = _parse_list_like(_safe_get(normalized, "V2EnhancedThemes", "V2Theme"))
            persons = _parse_list_like(_safe_get(normalized, "V2Persons", "V1Person"))
            orgs = _parse_list_like(_safe_get(normalized, "V2Orgs", "V1Org"))
            locations = _parse_list_like(_safe_get(normalized, "V2Locations", "FullName"))
            countries = [
                c.upper()
                for c in _parse_list_like(_safe_get(normalized, "V2Locations", "CountryCode"))
                if c and isinstance(c, str)
            ]

            event_codes = []
            for key in ("EventCode", "EventBaseCode", "EventRootCode"):
                event_codes.extend(_parse_list_like(normalized.get(key)))
                event_codes.extend(_parse_list_like(_safe_get(normalized, "event", key)))

            tone = _safe_get(normalized, "V15Tone", "Tone")
            positive = _safe_get(normalized, "V15Tone", "PositiveScore")
            negative = _safe_get(normalized, "V15Tone", "NegativeScore")
            polarity = _safe_get(normalized, "V15Tone", "Polarity")

            theme_counts.update(themes)
            person_counts.update(persons)
            org_counts.update(orgs)
            location_counts.update(locations)
            country_counts.update(countries)
            event_code_counts.update(event_codes)

            if tone is not None:
                try:
                    tone_sum += float(tone)
                    tone_docs += 1
                except (TypeError, ValueError):
                    pass
            if positive is not None:
                try:
                    positive_sum += float(positive)
                except (TypeError, ValueError):
                    pass
            if negative is not None:
                try:
                    negative_sum += float(negative)
                except (TypeError, ValueError):
                    pass
            if polarity is not None:
                try:
                    polarity_sum += float(polarity)
                except (TypeError, ValueError):
                    pass

        denominator = tone_docs if tone_docs else 1

        return {
            "total_documents": total_docs,
            "aggregations": {
                "themes": dict(theme_counts.most_common()),
                "persons": dict(person_counts.most_common()),
                "orgs": dict(org_counts.most_common()),
                "locations": dict(location_counts.most_common()),
                "countries": dict(country_counts.most_common()),
                "event_codes": dict(event_code_counts.most_common()),
                "tone_summary": {
                    "avg_tone": tone_sum / denominator if tone_docs else None,
                    "avg_positive": positive_sum / denominator if tone_docs else None,
                    "avg_negative": negative_sum / denominator if tone_docs else None,
                    "avg_polarity": polarity_sum / denominator if tone_docs else None,
                    "docs_with_tone": tone_docs,
                },
            },
        }

    def compute_relevance_score(self, nl_query: str, aggregation: dict[str, Any]) -> dict[str, Any]:
        relevance_signature = dspy.ChainOfThought(RelevanceScorerSignature)
        with dspy.context(lm=self.lm):
            agg_summary = {
                "total_documents": aggregation.get("total_documents", 0),
                "top_themes": list(aggregation.get("aggregations", {}).get("themes", {}).items())[:5],
                "top_persons": list(aggregation.get("aggregations", {}).get("persons", {}).items())[:5],
                "top_orgs": list(aggregation.get("aggregations", {}).get("orgs", {}).items())[:5],
                "top_locations": list(aggregation.get("aggregations", {}).get("locations", {}).items())[:5],
                "top_countries": list(aggregation.get("aggregations", {}).get("countries", {}).items())[:5],
                "tone_summary": aggregation.get("aggregations", {}).get("tone_summary", {}),
            }
            evaluation = relevance_signature(
                nl_query=nl_query,
                aggregation_summary=json.dumps(agg_summary, indent=2),
            )
            return {
                "relevance_score": int(evaluation.relevance_score),
                "entity_coverage": evaluation.entity_coverage,
                "theme_alignment": evaluation.theme_alignment,
                "tone_alignment": evaluation.tone_alignment,
                "reasoning": evaluation.reasoning,
            }