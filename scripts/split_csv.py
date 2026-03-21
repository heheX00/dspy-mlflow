from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT_CSV = Path("data/output_last_7_days.csv")
OUTPUT_DIR = Path("data/output_last_7_days_parts")
ROWS_PER_FILE = 4_000


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {INPUT_CSV}")
    print(f"Writing split files to: {OUTPUT_DIR}")
    print(f"Rows per file: {ROWS_PER_FILE}")

    total_rows = 0
    file_count = 0

    for chunk_idx, chunk in enumerate(
        pd.read_csv(INPUT_CSV, chunksize=ROWS_PER_FILE, low_memory=False),
        start=1,
    ):
        out_path = OUTPUT_DIR / f"output_last_7_days_part_{chunk_idx:04d}.csv"
        chunk.to_csv(out_path, index=False)
        row_count = len(chunk)

        total_rows += row_count
        file_count += 1

        print(f"Wrote {out_path.name} with {row_count} rows")

    print("\nSplit complete.")
    print(f"Total files created: {file_count}")
    print(f"Total rows written: {total_rows}")


if __name__ == "__main__":
    main()