from team_a_dspy.run_team_a_pipeline import run_once


def test_run_pipeline():
    result = run_once("Which 10 people were mentioned most often this month?")

    assert "querydsl_query" in result
    assert isinstance(result["parsed_query"], dict)