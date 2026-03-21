from team_a_dspy.data.bootstrap_examples import load_examples


def test_dataset_loader():
    examples = load_examples("team_a_dspy/data/examples_seed.jsonl")

    assert len(examples) > 0
    assert hasattr(examples[0], "user_question")
    assert hasattr(examples[0], "querydsl_query")