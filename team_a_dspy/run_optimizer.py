from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    script = repo_root / "optimizers" / "optimize_query_generator.py"

    cmd = [
        sys.executable,
        str(script),
        "--dataset",
        str(repo_root / "data" / "optimizer_fullset.jsonl"),
        "--optimizer-type",
        "mipro",
        "--metric-type",
        "execution",
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()