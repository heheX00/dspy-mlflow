from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Elasticsearch
    es_host: str
    es_username: str
    es_password: str
    es_index: str = "gkg"
    es_verify_ssl: bool = False
    es_request_timeout_seconds: int = 30

    # LLM
    llm_base_url: str
    llm_model_name: str
    llm_api_key: str = "not-required"
    llm_temperature: float = 0.0
    llm_timeout_seconds: int = 60

    # Existing static schema fallback
    schema_context_path: str = "team_a_dspy/data/schema_context.txt"

    # New schema behavior
    schema_mode: str = "dynamic"  # "dynamic" or "static"
    schema_cache_ttl_seconds: int = 300
    schema_candidate_field_limit: int = 24
    schema_enriched_field_limit: int = 8


settings = Settings()