from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from typing import Any, Dict, Generator

import pandas as pd
from elasticsearch import Elasticsearch, helpers

ES_URL = "http://localhost:9200"
INDEX_NAME = "gkg_sandbox"

# Folder containing split CSV files
CSV_PARTS_DIR = Path("data/output_last_7_days_parts")
CSV_GLOB_PATTERN = "output_last_7_days_part_*.csv"

READ_CHUNK_SIZE = 5000
BULK_CHUNK_SIZE = 1000
REQUEST_TIMEOUT = 120
MAX_RETRIES = 3
MAX_ERROR_PRINTS_PER_CHUNK = 5

# Columns in your CSV that contain Python-dict / Python-list strings
LITERAL_COLUMNS = {
    "GkgRecordId",
    "V15Tone",
    "V21AllNames",
    "V21Quotations",
    "V21SocVideo",
    "V2EnhancedThemes",
    "V2ExtrasXML",
    "V2GCAM",
    "V2Locations",
    "V2Orgs",
    "V2Persons",
    "V2SrcCmnName",
    "V2SrcCollectionId",
    "event",
    "host",
    "log",
}

# Columns to drop from _source
DROP_COLUMNS = {
    "message",
    "tags",
}

# Keep this in sync with create_index.py if you want stricter control
ALLOWED_TOP_LEVEL_FIELDS = {
    "@timestamp",
    "@version",
    "GkgRecordId",
    "RecordId",
    "V15Tone",
    "V21AllNames",
    "V21Date",
    "V21Quotations",
    "V21RelImg",
    "V21ShareImg",
    "V21SocImage",
    "V21SocVideo",
    "V2DocId",
    "V2EnhancedThemes",
    "V2ExtrasXML",
    "V2GCAM",
    "V2Locations",
    "V2Orgs",
    "V2Persons",
    "V2SrcCmnName",
    "V2SrcCollectionId",
    "datatype",
    "event",
    "filename",
    "filename_path",
    "host",
    "log",
    "location",
}


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def maybe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def maybe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def safe_literal_eval(value: Any) -> Any:
    value = clean_value(value)
    if value is None:
        return None
    if isinstance(value, (dict, list, int, float, bool)):
        return value
    if not isinstance(value, str):
        return value

    text = value.strip()
    if text == "":
        return None

    # Try JSON first because event.original is JSON
    parsed_json = safe_json_loads(text)
    if parsed_json is not None:
        return parsed_json

    try:
        return ast.literal_eval(text)
    except Exception:
        return value


def first_valid_geo_point_from_v2locations(v2locations: Any) -> dict[str, float] | None:
    if not isinstance(v2locations, dict):
        return None

    lons = v2locations.get("LocationLongitude")
    lats = v2locations.get("LocationLatitude")

    if not isinstance(lons, list) or not isinstance(lats, list):
        return None

    for lon_raw, lat_raw in zip(lons, lats):
        lon = maybe_float(lon_raw)
        lat = maybe_float(lat_raw)
        if lon is None or lat is None:
            continue
        if -180 <= lon <= 180 and -90 <= lat <= 90:
            return {"lat": lat, "lon": lon}

    return None


def normalize_gkgrecordid(value: Any) -> dict[str, Any] | None:
    obj = safe_literal_eval(value)
    if not isinstance(obj, dict):
        return None

    out: Dict[str, Any] = {}
    if "Translingual" in obj:
        out["Translingual"] = bool(obj["Translingual"])
    if "Date" in obj:
        raw_date = clean_value(obj["Date"])
        if raw_date is not None:
            # Keep as string so ES can parse using yyyyMMddHHmmss format
            out["Date"] = str(int(float(raw_date)))
    if "NumberInBatch" in obj:
        out["NumberInBatch"] = maybe_int(obj["NumberInBatch"])
    return out


def normalize_v15tone(value: Any) -> dict[str, Any] | None:
    obj = safe_literal_eval(value)
    if not isinstance(obj, dict):
        return None

    keys = [
        "ActivityRefDensity",
        "NegativeScore",
        "Polarity",
        "PositiveScore",
        "SelfGroupRefDensity",
        "Tone",
    ]
    out: Dict[str, Any] = {}
    for key in keys:
        if key in obj:
            out[key] = maybe_float(obj[key])
    return out


def normalize_simple_object(value: Any) -> dict[str, Any] | None:
    obj = safe_literal_eval(value)
    if isinstance(obj, dict):
        return obj
    return None


def to_utc_iso_z(value: Any) -> str | None:
    value = clean_value(value)
    if value is None:
        return None

    try:
        dt = pd.to_datetime(value, utc=True)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    except Exception:
        return None


def normalize_v21date(value: Any, row_dict: Dict[str, Any]) -> str | None:
    """
    Prefer the actual CSV V21Date if it is already an ISO date string.
    Fallback to event.original.V21Date if needed.
    Never convert to integer yyyyMMddHHmmss for the indexed V21Date field.
    """
    normalized = to_utc_iso_z(value)
    if normalized is not None:
        return normalized

    event_raw = row_dict.get("event")
    event_obj = safe_literal_eval(event_raw)

    if isinstance(event_obj, dict):
        original_payload = event_obj.get("original")
        if isinstance(original_payload, str):
            original_json = safe_json_loads(original_payload)
            if isinstance(original_json, dict):
                original_v21date = original_json.get("V21Date")
                normalized = to_utc_iso_z(original_v21date)
                if normalized is not None:
                    return normalized

    return None


def row_to_doc(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    doc: Dict[str, Any] = {}

    for key, raw_value in row_dict.items():
        if key in DROP_COLUMNS:
            continue
        if key not in ALLOWED_TOP_LEVEL_FIELDS:
            continue

        value = clean_value(raw_value)
        if value is None:
            continue

        if key == "GkgRecordId":
            norm = normalize_gkgrecordid(value)
            if norm:
                doc[key] = norm

        elif key == "V15Tone":
            norm = normalize_v15tone(value)
            if norm:
                doc[key] = norm

        elif key == "V21Date":
            norm = normalize_v21date(value, row_dict)
            if norm is not None:
                doc[key] = norm

        elif key in {
            "V21AllNames",
            "V21Quotations",
            "V21SocVideo",
            "V2EnhancedThemes",
            "V2ExtrasXML",
            "V2GCAM",
            "V2Locations",
            "V2Orgs",
            "V2Persons",
            "V2SrcCmnName",
            "V2SrcCollectionId",
            "event",
            "host",
            "log",
        }:
            norm = normalize_simple_object(value)
            if norm:
                doc[key] = norm

        elif key == "@version":
            doc[key] = str(value)

        else:
            doc[key] = value

    # Build geo_point for the mapped "location" field
    location_point = first_valid_geo_point_from_v2locations(doc.get("V2Locations"))
    if location_point is not None:
        doc["location"] = location_point

    return doc


def generate_actions(
    df: pd.DataFrame,
    *,
    start_row_number: int,
    file_tag: str,
) -> Generator[Dict[str, Any], None, None]:
    for offset, (_, row) in enumerate(df.iterrows()):
        row_dict = row.to_dict()
        doc = row_to_doc(row_dict)

        # Prefer RecordId as document id; add file_tag fallback to avoid collisions
        record_id = doc.get("RecordId")
        if record_id is not None:
            doc_id = str(record_id)
        else:
            doc_id = f"{file_tag}-row-{start_row_number + offset}"

        yield {
            "_index": INDEX_NAME,
            "_id": doc_id,
            "_source": doc,
        }


def extract_error_reason(result: Dict[str, Any]) -> str:
    try:
        _, info = next(iter(result.items()))
        error = info.get("error")
        if isinstance(error, dict):
            err_type = error.get("type", "unknown_error")
            reason = error.get("reason", str(error))
            return f"{err_type}: {reason}"
        return str(error)
    except Exception:
        return str(result)


def get_csv_files() -> list[Path]:
    if not CSV_PARTS_DIR.exists():
        raise FileNotFoundError(f"CSV parts directory not found: {CSV_PARTS_DIR}")

    files = sorted(CSV_PARTS_DIR.glob(CSV_GLOB_PATTERN))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {CSV_PARTS_DIR} matching pattern {CSV_GLOB_PATTERN}"
        )
    return files


def ingest_one_csv_file(es_bulk: Elasticsearch, csv_path: Path) -> tuple[int, int, int]:
    print(f"\n=== Processing file: {csv_path.name} ===")
    print(f"File size: {csv_path.stat().st_size / (1024 ** 2):.2f} MB")

    total_rows_seen = 0
    total_success = 0
    total_failed = 0
    chunk_number = 0
    file_tag = csv_path.stem

    for chunk in pd.read_csv(
        csv_path,
        chunksize=READ_CHUNK_SIZE,
        low_memory=False,
    ):
        chunk_number += 1
        chunk_start_row = total_rows_seen
        chunk_rows = len(chunk)

        success_count = 0
        fail_count = 0
        printed_errors = 0

        actions = generate_actions(
            chunk,
            start_row_number=chunk_start_row,
            file_tag=file_tag,
        )

        for ok, result in helpers.streaming_bulk(
            client=es_bulk,
            actions=actions,
            chunk_size=BULK_CHUNK_SIZE,
            max_retries=MAX_RETRIES,
            raise_on_error=False,
            raise_on_exception=False,
        ):
            if ok:
                success_count += 1
            else:
                fail_count += 1
                if printed_errors < MAX_ERROR_PRINTS_PER_CHUNK:
                    printed_errors += 1
                    print(
                        f"[{csv_path.name} | chunk {chunk_number} | error sample {printed_errors}] "
                        f"{extract_error_reason(result)}"
                    )

        total_rows_seen += chunk_rows
        total_success += success_count
        total_failed += fail_count

        print(
            f"{csv_path.name} | chunk {chunk_number} done: "
            f"rows_seen={total_rows_seen}, success={success_count}, failed={fail_count}"
        )

        if fail_count == chunk_rows:
            print(f"{csv_path.name} chunk {chunk_number} completely failed. Stopping this file early.")
            break

    return total_rows_seen, total_success, total_failed


def main() -> None:
    csv_files = get_csv_files()

    es = Elasticsearch(ES_URL)
    if not es.ping():
        raise ConnectionError(f"Cannot connect to Elasticsearch at {ES_URL}")

    es_bulk = es.options(request_timeout=REQUEST_TIMEOUT)

    print(f"Connected to Elasticsearch: {ES_URL}")
    print(f"Target index: {INDEX_NAME}")
    print(f"CSV parts directory: {CSV_PARTS_DIR}")
    print(f"Files found: {len(csv_files)}")
    print("Starting multi-file chunked load...")

    grand_total_rows = 0
    grand_total_success = 0
    grand_total_failed = 0
    start_time = time.time()

    for file_idx, csv_path in enumerate(csv_files, start=1):
        print(f"\n##### FILE {file_idx}/{len(csv_files)} #####")

        rows_seen, success, failed = ingest_one_csv_file(es_bulk, csv_path)

        grand_total_rows += rows_seen
        grand_total_success += success
        grand_total_failed += failed

        elapsed = time.time() - start_time
        rate = grand_total_success / elapsed if elapsed > 0 else 0.0

        print(
            f"\nCompleted {csv_path.name}: rows={rows_seen}, success={success}, failed={failed}"
        )
        print(
            f"Running totals: rows={grand_total_rows}, success={grand_total_success}, "
            f"failed={grand_total_failed}, avg_rate={rate:.2f} docs/sec"
        )

    es.indices.refresh(index=INDEX_NAME)

    total_elapsed = time.time() - start_time
    print("\nLoad complete.")
    print(f"Total files processed: {len(csv_files)}")
    print(f"Total rows read: {grand_total_rows}")
    print(f"Total indexed successfully: {grand_total_success}")
    print(f"Total failed: {grand_total_failed}")
    print(f"Total time: {total_elapsed:.2f} seconds")
    if total_elapsed > 0:
        print(f"Overall average rate: {grand_total_success / total_elapsed:.2f} docs/sec")


if __name__ == "__main__":
    main()