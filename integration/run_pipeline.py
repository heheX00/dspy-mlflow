from __future__ import annotations

import json

import dspy
import mlflow

from team_a_dspy.metrics.es_sandbox_metric import ESSandboxJudge
from team_a_dspy.modules.es_query_pipeline import ESQueryDSPyPipeline
from team_a_dspy.optimizers.compile_pipeline import configure_lm
from team_a_dspy.utils.config import settings
from team_a_dspy.utils.schema_loader import load_schema_context


def run_once(question: str) -> dict:
    configure_lm()
    schema_context = load_schema_context()
    program = ESQueryDSPyPipeline()

    prediction = program(
        user_question=question,
        schema_context=schema_context,
    )

    judge = ESSandboxJudge(index_name=settings.es_index)
    dummy_example = dspy.Example(
        user_question=question,
        schema_context=schema_context,
        expected_checks={"must_have_query": True},
    ).with_inputs("user_question", "schema_context")

    passed, details = judge.score_prediction(dummy_example, prediction)

    return {
        "question": question,
        "passed": passed,
        "querydsl_query": prediction.querydsl_query,
        "parsed_query": prediction.parsed_query,
        "stocktake": prediction.stocktake,
        "filter_plan": prediction.filter_plan,
        "aggregation_plan": prediction.aggregation_plan,
        "judge_details": details,
    }


if __name__ == "__main__":
    question = "Show the top 10 countries by event count in the last 7 days."

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    with mlflow.start_run(run_name="dspy_es_query_dsl_inference"):
        result = run_once(question)

        mlflow.log_param("question", question)
        mlflow.log_param("model", settings.llm_model_name)
        mlflow.log_metric("execution_pass", 1.0 if result["passed"] else 0.0)

        with open("integration/latest_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        mlflow.log_artifact("integration/latest_result.json")

        print(json.dumps(result, ensure_ascii=False, indent=2))