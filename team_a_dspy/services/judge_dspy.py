from __future__ import annotations

from services.sandbox_es_client import SandboxESClient


class JudgeDSPY:
    def __init__(self, sandbox_es_client: SandboxESClient):
        self.es_client = sandbox_es_client

    async def evaluate_query_dsl(
        self,
        generated_query_dsl: dict,
        expected_query_dsl: dict | None = None,
    ) -> dict:
        return await self.es_client.evaluate_query_dsl(
            query_dsl=generated_query_dsl,
            expected_query_dsl=expected_query_dsl,
        )