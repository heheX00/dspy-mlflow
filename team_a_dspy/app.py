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
import nest_asyncio
from mlflow.genai.scorers import Correctness

nest_asyncio.apply()

@asynccontextmanager
async def lifespan(app: FastAPI):
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

    dspy_judge = JudgeDSPY(sandbox_es_client=sandbox_es_client)
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

def run_mlflow_eval(dspy_client, query_text: str):
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(experiment_id="0")
    eval_dataset = [{
        "inputs": {"query_text": "Which is the safest country in the Middle East to travel to last week and provide figures?"},
        "expectations": {
            "generated_dsl_query": {
            "size": 0,
            "query": {
                "bool": {
                "must": [
                    {
                    "terms": {
                        "V2Locations.CountryCode.keyword": [
                        "IS",
                        "IZ",
                        "JO",
                        "SA",
                        "AE",
                        "QA",
                        "KW",
                        "OM",
                        "BH",
                        "LB",
                        "SY",
                        "YM",
                        "IR",
                        "TU"
                        ]
                    }
                    },
                    {
                    "range": {
                        "V21Date": {
                        "gte": "now-7d/d",
                        "lte": "now/d"
                        }
                    }
                    }
                ]
                }
            },
            "aggs": {
                "by_country": {
                "terms": {
                    "field": "V2Locations.CountryCode.keyword",
                    "size": 20
                },
                "aggs": {
                    "avg_tone": {
                    "avg": {
                        "field": "V15Tone.Tone"
                    }
                    },
                    "avg_negative_score": {
                    "avg": {
                        "field": "V15Tone.NegativeScore"
                    }
                    },
                    "avg_positive_score": {
                    "avg": {
                        "field": "V15Tone.PositiveScore"
                    }
                    },
                    "conflict_articles": {
                    "filter": {
                        "terms": {
                        "V2EnhancedThemes.V2Theme": [
                            "ARMEDCONFLICT",
                            "CRISISLEX_CRISISLEXREC",
                            "MILITARY"
                        ]
                        }
                    }
                    }
                }
                }
            }
            }
        }
    }]
    mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=dspy_client.generate_query_dsl,
        scorers=[Correctness()],
    )

@app.post("/generate_query", response_model=QueryResponse, dependencies=[Depends(require_dev_mode)])
async def generate_query(
    query: QueryRequest,
    background_tasks: BackgroundTasks,
    dspy_client: DSPYClient = Depends(get_dspy_client)
):
    query_dsl = dspy_client.generate_query_dsl(query.query_text)
    background_tasks.add_task(run_mlflow_eval, dspy_client, query.query_text)
    return QueryResponse(query_dsl=query_dsl)

@app.post("/evaluate_query", response_model=dict, dependencies=[Depends(require_dev_mode)])
async def evaluate_query(
    query: QueryResponse,
    dspy_judge: JudgeDSPY = Depends(get_dspy_judge)
):
    evaluation_result = dspy_judge.evaluate_query_dsl(generated_query_dsl=query.query_dsl)
    return evaluation_result

@app.post("/search", response_model=dict)
async def search(
    query: QueryRequest,
    dspy_client: DSPYClient = Depends(get_dspy_client),
    es_client: ESClient = Depends(get_es_client)
):
    query_dsl = dspy_client.generate_query_dsl(query.query_text)
    search_results = es_client.search(query_dsl=query_dsl)
    return search_results

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