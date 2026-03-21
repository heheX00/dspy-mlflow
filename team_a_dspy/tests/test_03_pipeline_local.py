from datetime import datetime

from team_a_dspy.modules.es_query_pipeline import ESQueryDSPyPipeline
from team_a_dspy.utils.schema_loader import load_schema_context


def test_pipeline_generates_query():
    schema = load_schema_context()

    program = ESQueryDSPyPipeline()
    pred = program(
        user_question="Which 10 people were mentioned most often this month?",
        schema_context=schema,
        today_iso_date=datetime.utcnow().strftime("%Y%m%d"),
    )

    assert isinstance(pred.querydsl_query, str)
    assert isinstance(pred.parsed_query, dict)
    assert "query" in pred.parsed_query