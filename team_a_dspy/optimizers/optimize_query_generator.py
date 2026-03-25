from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import dspy

THIS_FILE = Path(__file__).resolve()
TEAM_A_ROOT = THIS_FILE.parents[1]
if str(TEAM_A_ROOT) not in sys.path:
    sys.path.insert(0, str(TEAM_A_ROOT))

from metrics.es_query_metric import ExecutionAwareESMetric, metric_exact_query_dsl, normalize_query_dsl
from services.chroma_client import ChromaClient
from services.config import settings
from services.judge_dspy import JudgeDSPY
from services.sandbox_es_client import SandboxESClient
from signatures.es_query_generator import NLToQuerySignature
from signatures.schema_interpreter import SchemaRetriever


class OptimizableNLToQueryDSL(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_query = dspy.ChainOfThought(NLToQuerySignature)

    def forward(self, nl_query: str, es_schema: str) -> dspy.Prediction:
        generated_query = self.generate_query(nl_query=nl_query, es_schema=es_schema)
        return dspy.Prediction(query_dsl=generated_query.query_dsl)


def configure_lm() -> None:
    lm = dspy.LM(
        base_url=settings.llm_base_url,
        model=f"openai/{settings.llm_model_name}",
        api_key=settings.llm_api_key,
        temperature=0.0,
    )
    dspy.configure(lm=lm)


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


def write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_rows(
    rows: list[dict[str, Any]],
    train_ratio: float,
    seed: int,
    min_dev_size: int = 20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("Dataset is empty. Add at least one JSONL row.")
    if len(rows) < 2:
        raise ValueError("Dataset must contain at least 2 rows to do train/dev split.")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    ratio_dev_size = len(shuffled) - int(len(shuffled) * train_ratio)
    dev_size = max(min_dev_size, ratio_dev_size)

    # keep at least 1 train row
    dev_size = min(dev_size, len(shuffled) - 1)

    split_index = len(shuffled) - dev_size
    train_rows = shuffled[:split_index]
    dev_rows = shuffled[split_index:]

    return train_rows, dev_rows

def build_field_aliases(field_name: str) -> list[str]:
    aliases = [field_name, field_name.lower(), field_name.replace(".", " ").lower()]
    aliases.extend(field_name.split("."))

    lname = field_name.lower()
    if "countrycode" in lname:
        aliases.extend(["country", "country code", "location country"])
    if "person" in lname:
        aliases.extend(["person", "people", "individual"])
    if "organization" in lname or "org" in lname:
        aliases.extend(["organization", "company", "institution"])
    if "theme" in lname:
        aliases.extend(["theme", "topic", "category"])
    if "tone" in lname or "polarity" in lname or "negative" in lname or "positive" in lname:
        aliases.extend(["tone", "sentiment", "polarity", "negative score", "positive score"])
    if "date" in lname or "time" in lname:
        aliases.extend(["date", "time", "timestamp", "day", "week", "month"])
    if "src" in lname or "source" in lname:
        aliases.extend(["source", "publisher", "news source"])

    seen: set[str] = set()
    deduped: list[str] = []
    for alias in aliases:
        cleaned = alias.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped


def ensure_chroma_has_schema(chroma_client: ChromaClient, sandbox_client: SandboxESClient) -> None:
    if chroma_client.count() > 0:
        return

    flat_mapping = sandbox_client.get_flat_mapping()
    docs: list[dict[str, str]] = []
    for field_name, field_type in sorted(flat_mapping.items()):
        aliases = build_field_aliases(field_name)
        interpretation = "\n".join(
            [
                f"Field: {field_name}",
                f"Type: {field_type}",
                f"Aliases: {', '.join(aliases)}",
                f"Usage: Use this field when the user asks about {', '.join(aliases[:6])}.",
            ]
        )
        docs.append(
            {
                "field_name": field_name,
                "field_type": field_type,
                "interpretation": interpretation,
            }
        )
    chroma_client.add_documents(docs)


def enrich_rows_with_schema(rows: list[dict[str, Any]], retriever: SchemaRetriever) -> list[dict[str, Any]]:
    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched["es_schema"] = retriever(row["nl_query"])
        enriched_rows.append(enriched)
    return enriched_rows


def rows_to_examples(rows: list[dict[str, Any]]) -> list[dspy.Example]:
    examples: list[dspy.Example] = []
    for row in rows:
        examples.append(
            dspy.Example(
                nl_query=row["nl_query"],
                es_schema=row["es_schema"],
                query_dsl=normalize_query_dsl(row["expected_query_dsl"]),
            ).with_inputs("nl_query", "es_schema")
        )
    return examples


def extract_fields_from_expected_query(sandbox_client: SandboxESClient, row: dict[str, Any]) -> set[str]:
    expected = normalize_query_dsl(row["expected_query_dsl"])
    return sandbox_client.extract_referenced_fields(expected)


def schema_text_contains_all_fields(schema_text: str, required_fields: set[str]) -> bool:
    if not required_fields:
        return True
    return all(field in schema_text for field in required_fields)


def filter_incompatible_rows(
    rows: list[dict[str, Any]],
    sandbox_client: SandboxESClient,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_fields = set(sandbox_client.get_flat_mapping().keys())

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for row in rows:
        expected_fields = extract_fields_from_expected_query(sandbox_client, row)
        missing_in_mapping = sorted(field for field in expected_fields if field not in valid_fields)
        missing_in_schema = sorted(field for field in expected_fields if field not in row["es_schema"])

        if missing_in_mapping or missing_in_schema:
            bad_row = dict(row)
            bad_row["_drop_reason"] = {
                "missing_in_mapping": missing_in_mapping,
                "missing_in_schema": missing_in_schema,
                "expected_fields": sorted(expected_fields),
            }
            dropped.append(bad_row)
            continue

        kept.append(row)

    return kept, dropped


# def filter_incompatible_rows(
#     rows: list[dict[str, Any]],
#     sandbox_client: SandboxESClient,
# ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
#     """
#     Drop only if expected fields do not exist in sandbox mapping.
#     If expected fields are missing only from retrieved es_schema text,
#     repair the row by appending those field definitions.
#     """
#     flat_mapping = sandbox_client.get_flat_mapping()
#     valid_fields = set(flat_mapping.keys())

#     kept: list[dict[str, Any]] = []
#     dropped: list[dict[str, Any]] = []

#     for row in rows:
#         expected_fields = extract_fields_from_expected_query(sandbox_client, row)
#         schema_text = row.get("es_schema", "") or ""

#         missing_in_mapping = sorted(field for field in expected_fields if field not in valid_fields)
#         missing_in_schema = sorted(field for field in expected_fields if field not in schema_text)

#         # Hard drop only if field truly does not exist in ES mapping
#         if missing_in_mapping:
#             bad_row = dict(row)
#             bad_row["_drop_reason"] = {
#                 "reason": "expected_fields_missing_from_sandbox_mapping",
#                 "missing_in_mapping": missing_in_mapping,
#                 "missing_in_schema": missing_in_schema,
#                 "expected_fields": sorted(expected_fields),
#             }
#             dropped.append(bad_row)
#             continue

#         repaired_row = dict(row)

#         # Soft repair if retriever did not include all needed fields
#         if missing_in_schema:
#             appendix = build_required_schema_appendix(
#                 sandbox_client=sandbox_client,
#                 required_fields=set(missing_in_schema),
#             )
#             repaired_row["es_schema"] = (
#                 schema_text.strip()
#                 + "\n\n### REQUIRED_FIELDS_APPENDIX ###\n"
#                 + appendix
#             )
#             repaired_row["_schema_repair"] = {
#                 "missing_in_schema_before_repair": missing_in_schema,
#                 "expected_fields": sorted(expected_fields),
#             }

#         kept.append(repaired_row)

#     return kept, dropped


# def build_required_schema_appendix(
#     sandbox_client: SandboxESClient,
#     required_fields: set[str],
# ) -> str:
#     flat_mapping = sandbox_client.get_flat_mapping()
#     lines: list[str] = []

#     for field_name in sorted(required_fields):
#         field_type = flat_mapping.get(field_name, "unknown")
#         aliases = build_field_aliases(field_name)
#         lines.append(
#             "\n".join(
#                 [
#                     f"Field: {field_name}",
#                     f"Type: {field_type}",
#                     f"Aliases: {', '.join(aliases)}",
#                     f"Usage: This field is relevant to the current task.",
#                 ]
#             )
#         )

#     return "\n\n".join(lines)


def build_optimizer(metric_callable, optimizer_type: str):
    optimizer_type = optimizer_type.lower().strip()
    if optimizer_type == "mipro":
        mipro_cls = getattr(dspy, "MIPROv2", None)
        if mipro_cls is not None:
            return mipro_cls(metric=metric_callable, auto="light")
        print("WARNING: dspy.MIPROv2 not available; falling back to BootstrapFewShot.")

    return dspy.BootstrapFewShot(
        metric=metric_callable,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
    )


def compile_program(
    student: dspy.Module,
    trainset: list[dspy.Example],
    devset: list[dspy.Example],
    metric_callable,
    optimizer_type: str,
):
    teleprompter = build_optimizer(metric_callable, optimizer_type)
    compile_kwargs = {"student": student, "trainset": trainset}
    if teleprompter.__class__.__name__ == "MIPROv2":
        compile_kwargs["valset"] = devset
        compile_kwargs["requires_permission_to_run"] = False
    return teleprompter, teleprompter.compile(**compile_kwargs)


def evaluate_program(program: dspy.Module, eval_rows: list[dict[str, Any]], judge: JudgeDSPY) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    exact_matches = 0
    valid_queries = 0

    for idx, row in enumerate(eval_rows, start=1):
        expected = normalize_query_dsl(row["expected_query_dsl"])
        prediction = program(nl_query=row["nl_query"], es_schema=row["es_schema"])
        predicted = normalize_query_dsl(prediction.query_dsl)
        exact_match = predicted == expected
        if exact_match:
            exact_matches += 1

        judge_result = judge.evaluate_query_dsl(generated_query_dsl=predicted, expected_query_dsl=expected)
        if judge_result.get("is_valid"):
            valid_queries += 1

        results.append(
            {
                "index": idx,
                "nl_query": row["nl_query"],
                "es_schema": row["es_schema"],
                "exact_match": exact_match,
                "expected_query_dsl": expected,
                "predicted_query_dsl": predicted,
                "judge_result": judge_result,
            }
        )

    total = len(eval_rows)
    summary = {
        "num_examples": total,
        "exact_matches": exact_matches,
        "exact_match_rate": exact_matches / total if total else 0.0,
        "judge_validation": {
            "num_judged": total,
            "num_valid": valid_queries,
            "valid_rate": valid_queries / total if total else 0.0,
        },
        "avg_judge_score": sum(item["judge_result"]["score"] for item in results) / total if total else 0.0,
        "results": results,
    }
    return summary


def print_summary(label: str, summary: dict[str, Any]) -> None:
    print(
        f"[{label}] examples={summary['num_examples']} | "
        f"exact_matches={summary['exact_matches']} | "
        f"exact_match_rate={summary['exact_match_rate']:.3f}"
    )
    judge_validation = summary["judge_validation"]
    print(
        f"[{label}] judge_valid={judge_validation['num_valid']} | "
        f"judge_valid_rate={judge_validation['valid_rate']:.3f} | "
        f"avg_judge_score={summary['avg_judge_score']:.3f}"
    )


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted(make_json_safe(v) for v in obj)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize the Team A DSPy Elasticsearch query generator.")
    parser.add_argument("--dataset", default=str(TEAM_A_ROOT / "data" / "optimizer_fullset.jsonl"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--min-dev-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer-type", choices=["bootstrap", "mipro"], default="mipro")
    parser.add_argument("--metric-type", choices=["execution", "exact"], default="execution")
    parser.add_argument(
        "--artifact-output",
        default=str(TEAM_A_ROOT / "optimizers" / "artifacts" / "optimized_query_generator.json"),
    )
    parser.add_argument(
        "--report-output",
        default=str(TEAM_A_ROOT / "optimizers" / "artifacts" / "optimization_report.json"),
    )
    parser.add_argument(
        "--save-splits-dir",
        default=str(TEAM_A_ROOT / "data" / "splits"),
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    artifact_output_path = Path(args.artifact_output).resolve()
    report_output_path = Path(args.report_output).resolve()
    save_splits_dir = Path(args.save_splits_dir).resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    configure_lm()

    sandbox_client = SandboxESClient(
        host="http://localhost:9200" if settings.dev else None
    )
    judge = JudgeDSPY(sandbox_es_client=sandbox_client)

    rows = load_jsonl_rows(dataset_path)

    chroma_client = ChromaClient(dev=settings.dev)
    ensure_chroma_has_schema(chroma_client, sandbox_client)
    schema_retriever = SchemaRetriever(chroma_client=chroma_client)

    all_rows = enrich_rows_with_schema(rows, schema_retriever)
    kept_rows, dropped_rows = filter_incompatible_rows(all_rows, sandbox_client)
    
    print("Total rows:", len(rows))
    print("Kept rows:", len(kept_rows))
    print("Dropped rows:", len(dropped_rows))


    if len(kept_rows) < 2:
        raise ValueError("Not enough compatible rows remain after filtering.")

    train_rows, dev_rows = split_rows(
        kept_rows,
        train_ratio=args.train_ratio,
        seed=args.seed,
        min_dev_size=args.min_dev_size,
    )

    dropped_train_rows = []
    dropped_dev_rows = dropped_rows

    if not train_rows:
        raise ValueError("All train rows were dropped as incompatible with sandbox mapping/schema.")
    if not dev_rows:
        raise ValueError("All dev rows were dropped as incompatible with sandbox mapping/schema.")

    trainset = rows_to_examples(train_rows)
    devset = rows_to_examples(dev_rows)

    train_split_path = save_splits_dir / "optimizer_trainset.jsonl"
    dev_split_path = save_splits_dir / "optimizer_devset.jsonl"
    dropped_train_path = save_splits_dir / "optimizer_trainset_dropped.jsonl"
    dropped_dev_path = save_splits_dir / "optimizer_devset_dropped.jsonl"

    write_jsonl_rows(train_split_path, train_rows)
    write_jsonl_rows(dev_split_path, dev_rows)
    write_jsonl_rows(dropped_train_path, dropped_train_rows)
    write_jsonl_rows(dropped_dev_path, dropped_dev_rows)

    if args.metric_type == "execution":
        metric_callable = ExecutionAwareESMetric(sandbox_client=sandbox_client)
        metric_name = "ExecutionAwareESMetric"
    else:
        metric_callable = metric_exact_query_dsl
        metric_name = "metric_exact_query_dsl"

    student = OptimizableNLToQueryDSL()
    teleprompter, optimized_program = compile_program(
        student=student,
        trainset=trainset,
        devset=devset,
        metric_callable=metric_callable,
        optimizer_type=args.optimizer_type,
    )

    artifact_output_path.parent.mkdir(parents=True, exist_ok=True)
    optimized_program.save(str(artifact_output_path))

    train_summary = evaluate_program(optimized_program, train_rows, judge)
    dev_summary = evaluate_program(optimized_program, dev_rows, judge)

    report = {
        "dataset_path": str(dataset_path),
        "artifact_output": str(artifact_output_path),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "split_files": {
            "train": str(train_split_path),
            "dev": str(dev_split_path),
            "dropped_train": str(dropped_train_path),
            "dropped_dev": str(dropped_dev_path),
        },
        "optimizer": {
            "type": teleprompter.__class__.__name__,
            "metric": metric_name,
            "student_program": "OptimizableNLToQueryDSL",
        },
        "dataset_filtering": {
            "train_kept": len(train_rows),
            "train_dropped": len(dropped_train_rows),
            "dev_kept": len(dev_rows),
            "dev_dropped": len(dropped_dev_rows),
        },
        "train_summary": train_summary,
        "dev_summary": dev_summary,
    }

    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    with report_output_path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(report), f, indent=2, ensure_ascii=False)
    sandbox_client.close()

    print(f"Loaded dataset: {dataset_path}")
    print(f"Saved train split: {train_split_path}")
    print(f"Saved dev split: {dev_split_path}")
    print(f"Saved dropped train split: {dropped_train_path}")
    print(f"Saved dropped dev split: {dropped_dev_path}")
    print(f"Saved optimized artifact: {artifact_output_path}")
    print(f"Saved report: {report_output_path}")
    print_summary("TRAIN", train_summary)
    print_summary("DEV", dev_summary)


if __name__ == "__main__":
    main()