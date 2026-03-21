from elasticsearch import Elasticsearch

from team_a_dspy.utils.config import settings


def get_es_client() -> Elasticsearch:
    return Elasticsearch(
        settings.es_host,
        basic_auth=(settings.es_username, settings.es_password),
        verify_certs=settings.es_verify_ssl,
        request_timeout=settings.es_request_timeout_seconds,
    )