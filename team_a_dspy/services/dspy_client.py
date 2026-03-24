import dspy

from collections import defaultdict
from sentence_transformers.model_card import IGNORED_FIELDS

from services.chroma_client import ChromaClient
from services.es_client import ESClient, SAMPLE_SIZE_PER_DAY
from services.sandbox_es_client import SandboxESClient
from services.judge_dspy import JudgeDSPY
from services.config import settings

from signatures.schema_interpreter import DataAwareSchemaInterpreter
from signatures.es_query_generator import NLToQueryDSL

IGNORED_FIELDS = ["@timestamp", "log", "event", "filename", "filename_path"]
NUMBER_OF_DAYS = 7
MAX_SAMPLES_PER_FIELD = min(35, SAMPLE_SIZE_PER_DAY * NUMBER_OF_DAYS)
DEV = settings.dev

class DSPYClient:
    """
    Initializes and manages connections to Elasticsearch and ChromaDB,
    configures a language model, fetches recent document samples, extracts
    and flattens field data, interprets field schemas using a schema
    interpreter, and stores interpretations in ChromaDB.
    """
    def __init__(
            self,
            es_client: ESClient | SandboxESClient,
            chroma_client: ChromaClient,
            judge_dspy: JudgeDSPY
    ):
        # Initialize service clients
        self.es_client = es_client
        self.chroma_client = chroma_client
        self.judge_dspy = judge_dspy
        
        # Configure the language model with 0 temperature for deterministic outputs.
        # For more crreative LLM outputs (e.g., for schema interpretation), we could consider using a higher temperature.
        # See the use of with dspy.context(lm=self.lm, temperature=0.7) in the interpret_field method for an example of this.
        self.lm = dspy.LM(
            base_url=settings.llm_base_url,
            model=f"openai/{settings.llm_model_name}",
            api_key=settings.llm_api_key,
            temperature=0.0,
        )

        # Load DSPY modules
        self.schema_interpreter = DataAwareSchemaInterpreter()
        self.query_generator = NLToQueryDSL(chroma_client=self.chroma_client, dspy_judge=judge_dspy)

        dspy.configure(lm=self.lm)
    
    def close(self):
        """
        Closes any open connections or resources held by the clients.
        """
        self.es_client.close()

    def fetch_samples(self):
        """
        Fetches samples of documents from Elasticsearch for the last NUMBER_OF_DAYS days.
        """
        samples = self.es_client.get_last_x_days_samples(days=NUMBER_OF_DAYS)
        return samples
    
    @staticmethod
    def flatten_field(doc: dict, field_samples: dict, prefix: str = ""):
        """
        Recursively flattens the fields in a document and collects sample values for each field.
        The field_samples dictionary is updated in-place with the field names as keys and sets of sample values as values.
        """
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
                    
    def interpret_field(self):
        """
        Fetches samples of documents from Elasticsearch, extracts the unique field names and sample values, and uses the schema interpreter to generate interpretations for each field. 
        The interpretations are then stored in ChromaDB.
        """
        samples = self.fetch_samples()
        field_samples = defaultdict(set)
        for doc in samples:
            self.flatten_field(doc, field_samples)
        field_types = self.es_client.flatten_es_mapping()
        with dspy.context(lm=self.lm):
            for field_name, sample_values in field_samples.items():
                field_type = field_types.get(field_name, "unknown")
                interpretation = self.schema_interpreter(field_name=field_name, field_type=field_type, sample_values=list(sample_values))
                self.chroma_client.add_documents({"field_name": field_name, "field_type": field_type, "interpretation": str(interpretation)})

    def startup(self):
        """
        Performs any necessary startup initialization, such as interpreting fields and populating ChromaDB.
        """
        self.interpret_field()     

    def generate_query_dsl(self, query_text: str) -> dict:
        """
        Generates a query DSL based on the input natural language query text using the query generator.
        """
        query_dsl = self.query_generator(nl_query=query_text)
        return query_dsl