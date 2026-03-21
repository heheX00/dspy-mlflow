import pytest

from team_a_dspy.optimizers.compile_pipeline import configure_lm


@pytest.fixture(scope="session", autouse=True)
def setup_lm():
    """
    Automatically configure DSPy LM once per test session.
    """
    configure_lm()