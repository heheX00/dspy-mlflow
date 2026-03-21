import pytest

from team_a_dspy.utils.es_client import get_es_client
from team_a_dspy.utils.config import settings


@pytest.mark.integration
def test_known_query_execution():
    es = get_es_client()

    query = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {
                        "range": {
                            "GkgRecordId.Date": {
                                "gte": 20260301,
                                "lte": 20260319
                            }
                        }
                    }
                ]
            }
        },
        "aggs": {
            "top_people": {
                "terms": {
                    "field": "V2Persons.V1Person.keyword",
                    "size": 10
                }
            }
        }
    }

    resp = es.search(index=settings.es_index, body=query)

    buckets = resp["aggregations"]["top_people"]["buckets"]
    assert isinstance(buckets, list)