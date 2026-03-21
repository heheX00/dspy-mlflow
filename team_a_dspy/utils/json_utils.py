from __future__ import annotations

import json
import re
from typing import Any, Dict


class InvalidJSONError(ValueError):
    pass


def extract_first_json_object(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        raise InvalidJSONError("Empty model output")

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = cleaned.find("{")
    if start == -1:
        raise InvalidJSONError("No JSON object found")

    depth = 0
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start:i + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise InvalidJSONError(f"Invalid JSON extracted: {exc}") from exc
                if not isinstance(parsed, dict):
                    raise InvalidJSONError("Extracted JSON is not an object")
                return parsed

    raise InvalidJSONError("Unbalanced JSON object")