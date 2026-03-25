from __future__ import annotations

import asyncio

from services.sandbox_es_client import SandboxESClient


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        return loop.run_until_complete(coro)


class JudgeDSPY:
    def __init__(self, sandbox_es_client: SandboxESClient):
        self.es_client = sandbox_es_client

    def evaluate_query_dsl(
        self,
        generated_query_dsl: dict,
        expected_query_dsl: dict | None = None,
    ) -> dict:
        return _run_async(
            self.es_client.evaluate_query_dsl(
                query_dsl=generated_query_dsl,
                expected_query_dsl=expected_query_dsl,
            )
        )