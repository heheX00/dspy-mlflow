import pytest

from team_a_dspy.optimizers.compile_pipeline import compile_pipeline


@pytest.mark.integration
def test_optimizer_compile():
    program, trainset, devset = compile_pipeline(
        examples_path="team_a_dspy/data/examples_seed.jsonl"
    )

    assert program is not None
    assert len(trainset) > 0
    assert len(devset) > 0