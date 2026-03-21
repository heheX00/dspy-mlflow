from team_a_dspy.utils.json_utils import extract_first_json_object
from team_a_dspy.modules.es_query_pipeline import (
    validate_read_only_query,
    validate_query_shape,
)


def test_json_and_validation():
    raw = '{"size":0,"query":{"bool":{"filter":[]}},"aggs":{"top_people":{"terms":{"field":"V2Persons.V1Person.keyword","size":10}}}}'
    parsed = extract_first_json_object(raw)

    validate_read_only_query(parsed)
    validate_query_shape(
        "Which 10 people were mentioned most often this month?",
        parsed,
    )

    assert parsed["size"] == 0