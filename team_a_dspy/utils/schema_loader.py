from __future__ import annotations

from pathlib import Path

from team_a_dspy.utils.config import settings


def load_schema_context() -> str:
    path = Path(settings.schema_context_path)
    if not path.exists():
        return (
            "Schema context file not found. "
            "Provide mappings / field notes in team_a_dspy/data/schema_context.txt"
        )
    return path.read_text(encoding="utf-8")