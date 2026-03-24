from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dspy


class DSPYOptimiser:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_examples(self) -> list[dspy.Example]:
        path = Path(self.filepath)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        examples: list[dspy.Example] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row: dict[str, Any] = json.loads(line)
                if "nl_query" not in row or "expected_query_dsl" not in row:
                    raise ValueError(
                        f"Invalid row at line {line_no}: expected nl_query and expected_query_dsl"
                    )
                examples.append(
                    dspy.Example(
                        nl_query=row["nl_query"],
                        query_dsl=row["expected_query_dsl"],
                    ).with_inputs("nl_query")
                )
        return examples