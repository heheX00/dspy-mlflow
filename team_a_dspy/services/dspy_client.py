from __future__ import annotations

from collections import defaultdict

import dspy

from services.chroma_client import ChromaClient
from services.config import settings
from services.es_client import ESClient, SAMPLE_SIZE_PER_DAY
from services.judge_dspy import JudgeDSPY
from services.sandbox_es_client import SandboxESClient
from signatures.es_query_generator import NLToQueryDSL
from signatures.schema_interpreter import DataAwareSchemaInterpreter

IGNORED_FIELDS = ["@timestamp", "log", "event", "filename", "filename_path"]
NUMBER_OF_DAYS = 7
MAX_SAMPLES_PER_FIELD = min(35, SAMPLE_SIZE_PER_DAY * NUMBER_OF_DAYS)


class DSPYClient:
    def __init__(
        self,
        es_client: ESClient | SandboxESClient,
        chroma_client: ChromaClient,
        judge_dspy: JudgeDSPY,
    ):
        self.es_client = es_client
        self.chroma_client = chroma_client
        self.judge_dspy = judge_dspy

        self.lm = dspy.LM(
            base_url=settings.llm_base_url,
            model=f"openai/{settings.llm_model_name}",
            api_key=settings.llm_api_key,
            temperature=0.0,
        )
        dspy.configure(lm=self.lm)

        self.schema_interpreter = DataAwareSchemaInterpreter()
        self.query_generator = NLToQueryDSL(chroma_client=self.chroma_client, dspy_judge=self.judge_dspy)

    def close(self) -> None:
        self.es_client.close()

    def fetch_samples(self) -> list[dict]:
        return self.es_client.get_last_x_days_samples(days=NUMBER_OF_DAYS)

    @staticmethod
    def flatten_field(doc: dict, field_samples: dict, prefix: str = "") -> None:
        for key, value in doc.items():
            if key in IGNORED_FIELDS:
                continue

            current_field = f"{prefix}{key}"
            if isinstance(value, dict):
                DSPYClient.flatten_field(value, field_samples, prefix=f"{current_field}.")
                continue

            if value is None or str(value).strip() == "":
                continue

            if isinstance(value, list):
                if not value:
                    continue
                preview = value[:3]
                val_str = f"[{', '.join(map(str, preview))}...]" if len(value) > 3 else str(value)
            else:
                val_str = str(value)

            if len(field_samples[current_field]) < MAX_SAMPLES_PER_FIELD:
                field_samples[current_field].add(val_str)

    def interpret_field(self) -> None:
        samples = self.fetch_samples()
        field_samples = defaultdict(set)
        for doc in samples:
            self.flatten_field(doc, field_samples)

        field_types = self.es_client.flatten_es_mapping()
        with dspy.context(lm=self.lm):
            for field_name, sample_values in field_samples.items():
                field_type = field_types.get(field_name, "unknown")
                interpretation = self.schema_interpreter(
                    field_name=field_name,
                    field_type=field_type,
                    sample_values=str(list(sample_values)),
                )
                self.chroma_client.add_documents(
                    {
                        "field_name": field_name,
                        "field_type": field_type,
                        "interpretation": str(interpretation),
                    }
                )

    def startup(self) -> None:
        self.interpret_field()

    def generate_query_dsl(self, query_text: str) -> dict:
        prediction = self.query_generator(nl_query=query_text)
        if isinstance(prediction, dict):
            return prediction
        return prediction.query_dsl