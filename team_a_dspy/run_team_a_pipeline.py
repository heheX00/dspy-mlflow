from __future__ import annotations

import datetime
import json

from team_a_dspy.metrics.es_sandbox_metric import ESSandboxJudge
from team_a_dspy.modules.es_query_pipeline import ESQueryDSPyPipeline
from team_a_dspy.optimizers.compile_pipeline import configure_lm
from team_a_dspy.utils.config import settings
from team_a_dspy.utils.schema_loader import load_schema_context


def run_once(question: str) -> dict:
    configure_lm()
    today_iso_date = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")

    # Preserve fallback behavior for static mode.
    schema_context = load_schema_context() if settings.schema_mode == "static" else None

    program = ESQueryDSPyPipeline()
    prediction = program(
        user_question=question,
        schema_context=schema_context,
        today_iso_date=today_iso_date,
    )

    judge = ESSandboxJudge(index_name=settings.es_index)

    class DummyExample:
        user_question = question
        expected_checks = {"must_have_query": True}

    passed, details = judge.score_prediction(DummyExample(), prediction)

    return {
        "question": question,
        "passed": passed,
        "raw_candidate_schema": prediction.raw_candidate_schema,
        "effective_schema_context": prediction.effective_schema_context,
        "querydsl_query": prediction.querydsl_query,
        "parsed_query": prediction.parsed_query,
        "stocktake": prediction.stocktake,
        "filter_plan": prediction.filter_plan,
        "aggregation_plan": prediction.aggregation_plan,
        "judge_details": details,
    }


if __name__ == "__main__":
    question = "Top 10 people mentioned in negative news about Iran last week"
    result = run_once(question)
    print(json.dumps(result, ensure_ascii=False, indent=2))