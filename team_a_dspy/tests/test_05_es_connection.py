import pytest

from team_a_dspy.utils.es_client import get_es_client
from team_a_dspy.utils.config import settings


@pytest.mark.integration
def test_es_connection():
    es = get_es_client()
    info = es.info()
    count = es.count(index=settings.es_index)

    assert "cluster_name" in info
    assert "count" in count