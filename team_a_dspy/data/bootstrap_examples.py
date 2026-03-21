from __future__ import annotations

import json
from pathlib import Path
from typing import List

import dspy


def load_examples(jsonl_path: str) -> List[dspy.Example]:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Example dataset not found: {jsonl_path}")

    examples: List[dspy.Example] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            ex = dspy.Example(
                user_question=row["user_question"],
                schema_context=row["schema_context"],
                today_yyyymmdd=row.get("today_yyyymmdd", "20260319"),
                querydsl_query=row["querydsl_query"],
                expected_checks=row.get("expected_checks", {}),
            ).with_inputs("user_question", "schema_context", "today_yyyymmdd")
            examples.append(ex)

    return examples


def split_train_dev(examples: List[dspy.Example], dev_ratio: float = 0.2):
    if not examples:
        raise ValueError("No examples loaded")

    split_idx = max(1, int(len(examples) * (1 - dev_ratio)))
    return examples[:split_idx], examples[split_idx:]