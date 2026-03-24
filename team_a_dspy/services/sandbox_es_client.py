from services.es_client import ESClient
from services.config import settings


class SandboxESClient(ESClient):
    """
    Client for interacting with the sandbox Elasticsearch instance.

    Allows optional override of connection parameters so it can be used both:
    - inside Docker (sandbox_elasticsearch)
    - from host machine (localhost)
    """

    def __init__(
        self,
        host: str | None = None,
        username: str | None = None,
        password: str | None = None,
        index: str | None = None,
        verify_ssl: bool | None = None,
    ):
        super().__init__(
            host or settings.sandbox_es_host,
            username if username is not None else settings.sandbox_es_username,
            password if password is not None else settings.sandbox_es_password,
            index or settings.sandbox_es_index,
            verify_ssl if verify_ssl is not None else settings.sandbox_es_verify_ssl,
        )

    def validate_query_dsl(self, query_dsl: dict):
        """
        Validates the generated Query DSL against the sandbox Elasticsearch index.

        For indices.validate_query, only the "query" section should be passed.
        """

        try:
            full_dsl = query_dsl.get("query_dsl", {})
            query_part = full_dsl.get("query", {})

            if not query_part:
                return {
                    "is_valid": False,
                    "feedback": "Missing 'query' field in generated query_dsl.",
                }

            response = self.es.indices.validate_query(
                index=self.index,
                body={"query": query_part},
                explain=True,
            )

            return {
                "is_valid": response.body.get("valid", False),
                "feedback": response.body.get(
                    "explanations",
                    response.body.get("error", "No explanation provided"),
                ),
            }

        except Exception as e:
            return {
                "is_valid": False,
                "feedback": f"Elasticsearch validation error: {type(e).__name__}: {e}",
            }

    def close(self):
        self.es.close()