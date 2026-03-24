import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any

import dspy

# Allow running this file directly from anywhere in the repo.
THIS_FILE = Path(__file__).resolve()
TEAM_A_ROOT = THIS_FILE.parents[1]
if str(TEAM_A_ROOT) not in sys.path:
    sys.path.insert(0, str(TEAM_A_ROOT))

from services.chroma_client import ChromaClient
from services.config import settings
from services.judge_dspy import JudgeDSPY
from services.sandbox_es_client import SandboxESClient
from signatures.es_query_generator import NLToQuerySignature
from signatures.schema_interpreter import SchemaRetriever

class OptimizableNLToQueryDSL(dspy.Module):
    """
    Optimization target for first-pass Query DSL generation.

    This stays aligned with the repo's production flow:
    - retrieve schema from ChromaDB
    - generate Query DSL with DSPy ChainOfThought

    It intentionally does not include the async judge/refinement loop because
    BootstrapFewShot expects a synchronous student program for compilation.
    Judge-based validation is handled in the post-compile evaluation phase.
    """

    def __init__(self, chroma_client: ChromaClient):
        super().__init__()
        self.schema_retriever = SchemaRetriever(chroma_client=chroma_client)
        self.generate_query = dspy.ChainOfThought(NLToQuerySignature)

    def forward(self, nl_query: str):
        es_schema = self.schema_retriever(nl_query=nl_query)
        generated_query = self.generate_query(nl_query=nl_query, es_schema=es_schema)
        return dspy.Prediction(query_dsl=generated_query.query_dsl)


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "nl_query" not in row or "expected_query_dsl" not in row:
                raise ValueError(
                    f"Invalid JSONL row at line {line_no} in {path}: "
                    "each row must contain 'nl_query' and 'expected_query_dsl'."
                )
            rows.append(row)
    return rows


def rows_to_examples(rows: list[dict[str, Any]]) -> list[dspy.Example]:
    return [
        dspy.Example(
            nl_query=row["nl_query"],
            expected_query_dsl=row["expected_query_dsl"],
        ).with_inputs("nl_query")
        for row in rows
    ]


def write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_rows(
    rows: list[dict[str, Any]],
    train_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("Dataset is empty. Add at least one JSONL row.")
    if len(rows) < 2:
        raise ValueError("Dataset must contain at least 2 rows to do train/dev split.")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    shuffled = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    split_index = int(len(shuffled) * train_ratio)
    split_index = max(1, min(len(shuffled) - 1, split_index))
    train_rows = shuffled[:split_index]
    dev_rows = shuffled[split_index:]
    return train_rows, dev_rows


def normalize_query_dsl(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("query_dsl")
    if isinstance(nested, dict):
        return nested
    return payload


def metric_exact_query_dsl(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
) -> float:
    expected = normalize_query_dsl(example.expected_query_dsl)
    predicted = normalize_query_dsl(pred.query_dsl)
    return 1.0 if expected == predicted else 0.0


def configure_lm() -> None:
    lm = dspy.LM(
        base_url=settings.llm_base_url,
        model=f"openai/{settings.llm_model_name}",
        api_key=settings.llm_api_key,
        temperature=0.0,
    )
    dspy.configure(lm=lm)


def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def evaluate_program(
    program: dspy.Module,
    eval_rows: list[dict[str, Any]],
    *,
    run_judge_validation: bool,
) -> dict[str, Any]:
    judge = None
    if run_judge_validation:
        judge = JudgeDSPY(
            sandbox_es_client=SandboxESClient(host="http://localhost:9200")
        )

    total = len(eval_rows)
    exact_matches = 0
    valid_queries = 0
    judged_queries = 0
    results: list[dict[str, Any]] = []

    try:
        for idx, row in enumerate(eval_rows, start=1):
            nl_query = row["nl_query"]
            expected = normalize_query_dsl(row["expected_query_dsl"])

            prediction = program(nl_query=nl_query)
            predicted = normalize_query_dsl(prediction.query_dsl)
            exact_match = predicted == expected
            if exact_match:
                exact_matches += 1

            judge_result: dict[str, Any] | None = None
            if judge is not None:
                judged_queries += 1
                try:
                    judge_result = run_async(
                        judge.evaluate_query_dsl({"query_dsl": predicted})
                    )
                    if judge_result.get("is_valid"):
                        valid_queries += 1
                except Exception as exc:
                    judge_result = {
                        "is_valid": False,
                        "feedback": f"Judge validation failed: {type(exc).__name__}: {exc}",
                    }

            results.append(
                {
                    "index": idx,
                    "nl_query": nl_query,
                    "exact_match": exact_match,
                    "expected_query_dsl": expected,
                    "predicted_query_dsl": predicted,
                    "judge_result": judge_result,
                }
            )
    finally:
        if judge is not None:
            try:
                run_async(judge.es_client.close())
            except Exception:
                pass

    summary: dict[str, Any] = {
        "num_examples": total,
        "exact_matches": exact_matches,
        "exact_match_rate": (exact_matches / total) if total else 0.0,
        "results": results,
    }
    if judge is not None:
        summary["judge_validation"] = {
            "num_judged": judged_queries,
            "num_valid": valid_queries,
            "valid_rate": (valid_queries / judged_queries) if judged_queries else 0.0,
        }
    return summary


def print_summary(label: str, summary: dict[str, Any]) -> None:
    print(
        f"[{label}] examples={summary['num_examples']} | "
        f"exact_matches={summary['exact_matches']} | "
        f"exact_match_rate={summary['exact_match_rate']:.3f}"
    )
    judge_validation = summary.get("judge_validation")
    if judge_validation:
        print(
            f"[{label}] judge_valid={judge_validation['num_valid']} | "
            f"judge_valid_rate={judge_validation['valid_rate']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split a JSONL dataset, optimize the Team A DSPy query generator on the "
            "train split, and evaluate the compiled program on the dev split."
        )
    )
    parser.add_argument(
        "--dataset",
        default=str(TEAM_A_ROOT / "data" / "optimizer_devset.jsonl"),
        help="Path to a JSONL dataset with nl_query and expected_query_dsl fields.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of examples to use for the optimizer train split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/dev splits.",
    )
    parser.add_argument(
        "--artifact-output",
        default=str(TEAM_A_ROOT / "optimizers" / "artifacts" / "optimized_query_generator.json"),
        help="Path to save the compiled DSPy program artifact.",
    )
    parser.add_argument(
        "--report-output",
        default=str(TEAM_A_ROOT / "optimizers" / "artifacts" / "optimization_report.json"),
        help="Path to save the train/dev split and evaluation report JSON.",
    )
    parser.add_argument(
        "--save-splits-dir",
        default=str(TEAM_A_ROOT / "data" / "splits"),
        help="Directory to save the reproducible train/dev JSONL split files.",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip sandbox Elasticsearch validation during evaluation.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    artifact_output_path = Path(args.artifact_output).resolve()
    report_output_path = Path(args.report_output).resolve()
    save_splits_dir = Path(args.save_splits_dir).resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    configure_lm()

    rows = load_jsonl_rows(dataset_path)
    train_rows, dev_rows = split_rows(rows, train_ratio=args.train_ratio, seed=args.seed)
    trainset = rows_to_examples(train_rows)

    train_split_path = save_splits_dir / "optimizer_trainset.jsonl"
    dev_split_path = save_splits_dir / "optimizer_devset.jsonl"
    write_jsonl_rows(train_split_path, train_rows)
    write_jsonl_rows(dev_split_path, dev_rows)

    chroma_client = ChromaClient(dev=settings.dev)
    student = OptimizableNLToQueryDSL(chroma_client=chroma_client)

    teleprompter = dspy.BootstrapFewShot(metric=metric_exact_query_dsl)
    optimized_program = teleprompter.compile(student=student, trainset=trainset)

    artifact_output_path.parent.mkdir(parents=True, exist_ok=True)
    optimized_program.save(str(artifact_output_path))

    train_summary = evaluate_program(
        optimized_program,
        train_rows,
        run_judge_validation=not args.skip_judge,
    )
    dev_summary = evaluate_program(
        optimized_program,
        dev_rows,
        run_judge_validation=not args.skip_judge,
    )

    report = {
        "dataset_path": str(dataset_path),
        "artifact_output": str(artifact_output_path),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "split_files": {
            "train": str(train_split_path),
            "dev": str(dev_split_path),
        },
        "optimizer": {
            "type": "dspy.BootstrapFewShot",
            "metric": "metric_exact_query_dsl",
            "student_program": "OptimizableNLToQueryDSL",
        },
        "train_summary": train_summary,
        "dev_summary": dev_summary,
    }

    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    with report_output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Loaded dataset: {dataset_path}")
    print(f"Saved train split: {train_split_path}")
    print(f"Saved dev split: {dev_split_path}")
    print(f"Saved optimized artifact: {artifact_output_path}")
    print(f"Saved report: {report_output_path}")
    print_summary("TRAIN", train_summary)
    print_summary("DEV", dev_summary)


if __name__ == "__main__":
    main()
