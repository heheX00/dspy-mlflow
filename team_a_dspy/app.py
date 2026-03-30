from elasticsearch import helpers
from fastapi import Depends, FastAPI, HTTPException, Request, status, BackgroundTasks
from fastapi.concurrency import asynccontextmanager
from pydantic import BaseModel

from services.dspy_client import DSPYClient
from services.es_client import ESClient
from services.chroma_client import ChromaClient
from services.sandbox_es_client import SandboxESClient
from services.config import settings
from services.judge_dspy import JudgeDSPY

import mlflow
import time
from mlflow.genai.scorers import Correctness


def setup_mlflow(*, enable_dspy_autolog: bool = False) -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    if enable_dspy_autolog and settings.mlflow_enable_dspy_autolog:
        mlflow.dspy.autolog()

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_mlflow(enable_dspy_autolog=True)
    # Initialize clients and store them in app state
    es_client = ESClient(
        host=settings.es_host,
        username=settings.es_username,
        password=settings.es_password,
        index=settings.es_index,
        verify_ssl=settings.es_verify_ssl
    )
    chroma_client = ChromaClient(dev=False)
    sandbox_es_client = SandboxESClient()

    # Validate generated DSL against the same ES target used for actual search execution.
    dspy_judge = JudgeDSPY(es_client=sandbox_es_client)
    dpsy_client = DSPYClient(es_client=es_client, chroma_client=chroma_client, judge_dspy=dspy_judge)
    

    app.state.es_client = es_client
    app.state.sandbox_es_client = sandbox_es_client
    app.state.chroma_client = chroma_client
    app.state.dspy_client = dpsy_client
    app.state.dspy_judge = dspy_judge

    yield
    # Cleanup if necessary (e.g., close connections)
    
    dpsy_client.close()

def get_dspy_client(request: Request) -> DSPYClient:
    return request.app.state.dspy_client

def get_dspy_judge(request: Request) -> JudgeDSPY:
    return request.app.state.dspy_judge

def get_es_client(request: Request) -> ESClient:
    return request.app.state.es_client

def get_sandbox_es_client(request: Request) -> SandboxESClient:
    return request.app.state.sandbox_es_client

def require_dev_mode() -> None:
    if not settings.dev:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found"
        )

app = FastAPI(title="GDELT Text-to-Query-DSL", lifespan=lifespan)

class QueryRequest(BaseModel):
    query_text: str

class QueryResponse(BaseModel):
    query_dsl: dict


class MetricEvaluationRequest(BaseModel):
    query_text: str
    query_dsl: dict


@app.post("/generate_query", response_model=QueryResponse, dependencies=[Depends(require_dev_mode)])
async def generate_query(
    query: QueryRequest,
    dspy_client: DSPYClient = Depends(get_dspy_client)
):
    start_time = time.perf_counter()
    with mlflow.start_run(run_name="generate_query"):
        query_dsl = dspy_client.generate_query_dsl(query.query_text)
        mlflow.log_param("query_text", query.query_text)
        mlflow.log_metric("latency_ms", (time.perf_counter() - start_time) * 1000)
        mlflow.log_dict(query_dsl, "generated_query_dsl.json")
    return QueryResponse(query_dsl=query_dsl)

@app.post("/evaluate_query", response_model=dict, dependencies=[Depends(require_dev_mode)])
async def evaluate_query(
    query: QueryResponse,
    dspy_judge: JudgeDSPY = Depends(get_dspy_judge)
):
    start_time = time.perf_counter()
    with mlflow.start_run(run_name="evaluate_query"):
        evaluation_result = dspy_judge._evaluate_query_dsl_syntax(generated_query_dsl=query.query_dsl)
        mlflow.log_metric("latency_ms", (time.perf_counter() - start_time) * 1000)
        mlflow.log_metric("is_valid", 1 if evaluation_result.get("is_valid") else 0)
        mlflow.log_param("feedback", evaluation_result.get("feedback", ""))
    return evaluation_result

@app.post("/search", response_model=dict)
async def search(
    query: QueryRequest,
    dspy_client: DSPYClient = Depends(get_dspy_client),
    es_client: ESClient = Depends(get_es_client)
):
    start_time = time.perf_counter()
    with mlflow.start_run(run_name="search"):
        query_dsl = dspy_client.generate_query_dsl(query.query_text)
        search_results = es_client.search(query_dsl=query_dsl)
        hits = search_results.get("hits", {}).get("hits", [])
        mlflow.log_param("query_text", query.query_text)
        mlflow.log_metric("latency_ms", (time.perf_counter() - start_time) * 1000)
        mlflow.log_dict(query_dsl, "executed_query_dsl.json")
        mlflow.log_metric("hits_count", len(hits))
    return search_results

@app.post("/search_and_aggregate", response_model=dict)
async def search_and_aggregate(
    query: QueryRequest,
    dspy_client: DSPYClient = Depends(get_dspy_client),
    es_client: ESClient = Depends(get_es_client),
    judge_dspy: JudgeDSPY = Depends(get_dspy_judge)
):
    start_time = time.perf_counter()
    with mlflow.start_run(run_name="search_and_aggregate"):
        query_dsl = dspy_client.generate_query_dsl(query.query_text)
        search_results = es_client.search(query_dsl=query_dsl)
        docs = search_results.get("hits", {}).get("hits", [])
        aggregations = judge_dspy._aggregate_es_documents(docs)
        mlflow.log_param("query_text", query.query_text)
        mlflow.log_metric("latency_ms", (time.perf_counter() - start_time) * 1000)
        mlflow.log_dict(query_dsl, "executed_query_dsl.json")
        mlflow.log_dict(aggregations, "aggregations.json")
    return aggregations

@app.post("/evaluate_relevance", response_model=dict)
async def evaluate_relevance(
    query: QueryRequest,
    dspy_client: DSPYClient = Depends(get_dspy_client),
    es_client: ESClient = Depends(get_es_client),
    judge_dspy: JudgeDSPY = Depends(get_dspy_judge)
):
    start_time = time.perf_counter()
    with mlflow.start_run(run_name="evaluate_relevance"):
        query_dsl = dspy_client.generate_query_dsl(query.query_text)
        search_results = es_client.search(query_dsl=query_dsl)
        docs = search_results.get("hits", {}).get("hits", [])
        aggregations = judge_dspy._aggregate_es_documents(docs)
        print(docs)
        print(aggregations)
        relevance_evaluation = judge_dspy.compute_relevance_score(nl_query=query.query_text, aggregation=aggregations)
        mlflow.log_param("query_text", query.query_text)
        mlflow.log_metric("latency_ms", (time.perf_counter() - start_time) * 1000)
        mlflow.log_dict(query_dsl, "executed_query_dsl.json")
        mlflow.log_dict(aggregations, "aggregations.json")
        mlflow.log_metric("relevance_score", relevance_evaluation.get("relevance_score", 0))
    return relevance_evaluation


@app.get("/initialize", dependencies=[Depends(require_dev_mode)])
async def initialize(
    sandbox_es_client: SandboxESClient = Depends(get_sandbox_es_client),
    dspy_client: DSPYClient = Depends(get_dspy_client)
):
    dspy_client.startup()
    sample_docs = dspy_client.fetch_samples()
    def push_to_dev_es(sandbox_es_client: SandboxESClient, docs: list[dict]):
        """
        Pushes the sample documents to the sandbox ES instance.
        """
        if not docs:
            return
        actions = [
            {
                "_index": settings.sandbox_es_index,
                "_id": doc.get("_id"),
                "_source": doc.get("_source"),
            }
            for doc in docs
        ]
        success, failed = helpers.bulk(sandbox_es_client.es, actions)
        print(f"Succeeded: {success}, Failed: {failed}")

    push_to_dev_es(sandbox_es_client, sample_docs)
    return {"status": "initialized"}

@app.get("/load_example", dependencies=[Depends(require_dev_mode)])
async def load_example(
    dspy_client: DSPYClient = Depends(get_dspy_client)
):
    example_query = "Find all events related to natural disasters in 2020."
    query_dsl = dspy_client.generate_query_dsl(example_query)
    return {"query": example_query, "generated_query_dsl": query_dsl}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}