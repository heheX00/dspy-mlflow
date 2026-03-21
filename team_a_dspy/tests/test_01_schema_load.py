from team_a_dspy.utils.schema_loader import load_schema_context


def test_schema_load():
    schema = load_schema_context()
    assert isinstance(schema, str)
    assert len(schema) > 100