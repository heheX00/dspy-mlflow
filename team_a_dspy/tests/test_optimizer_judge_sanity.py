from services.sandbox_es_client import SandboxESClient


def test_extract_referenced_fields_handles_terms_agg():
    client = SandboxESClient(host="http://localhost:9200")

    query = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {"range": {"V21Date": {"gte": "20260101", "lte": "20260131"}}}
                ]
            }
        },
        "aggs": {
            "top_people": {
                "terms": {
                    "field": "V2Persons.V1Person.keyword",
                    "size": 10,
                    "order": {"_count": "desc"},
                }
            }
        },
    }

    fields = client.extract_referenced_fields(query)
    client.close()

    assert "V21Date" in fields
    assert "V2Persons.V1Person.keyword" in fields
    assert "field" not in fields
    assert "size" not in fields
    assert "order" not in fields