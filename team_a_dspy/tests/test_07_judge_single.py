import pytest
from datetime import datetime

from team_a_dspy.metrics.es_sandbox_metric import ESSandboxJudge
from team_a_dspy.modules.es_query_pipeline import ESQueryDSPyPipeline
from team_a_dspy.utils.schema_loader import load_schema_context


@pytest.mark.integration
def test_judge_single():
    schema = load_schema_context()

    program = ESQueryDSPyPipeline()
    judge = ESSandboxJudge()

    question = "Which 10 people were mentioned most often this month?"

    pred = program(
        user_question=question,
        schema_context=schema,
        today_iso_date=datetime.utcnow().strftime("%Y%m%d"),
    )

    class Example:
        user_question = question
        expected_checks = {
            "require_size_zero": True,
            "required_agg": "top_people",
        }

    passed, _ = judge.score_prediction(Example(), pred)

    assert isinstance(passed, bool)