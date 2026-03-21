import os
import csv
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch, helpers

# ========================
# CONFIG (from your env)
# ========================
ES_USERNAME = os.getenv("ES_USERNAME", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "changeme")
ES_INDEX = os.getenv("ES_INDEX", "gkg")
ES_VERIFY_SSL = os.getenv("ES_VERIFY_SSL", "false").lower() == "true"

ES_HOST = "https://xxx.xxx.xxx.xxx:xxxx"  # change if needed
TIME_FIELD = "@timestamp"  # ⚠️ CHANGE THIS if your index uses something else (e.g. DATE, DATEADDED)

OUTPUT_FILE = "output_last_7_days.csv"

# ========================
# CONNECT
# ========================
es = Elasticsearch(
    ES_HOST,
    basic_auth=(ES_USERNAME, ES_PASSWORD),
    verify_certs=ES_VERIFY_SSL
)

# ========================
# TIME FILTER (last 7 days)
# ========================
now = datetime.utcnow()
last_7_days = now - timedelta(days=7)

query = {
    "_source": True,
    "query": {
        "range": {
            TIME_FIELD: {
                "gte": last_7_days.isoformat(),
                "lte": now.isoformat()
            }
        }
    }
}

# ========================
# HELPER: normalize values for CSV
# ========================
def normalize(value):
    if isinstance(value, list):
        return "|".join(map(str, value))
    if isinstance(value, dict):
        return str(value)
    return value

# ========================
# PASS 1: collect all fields
# ========================
print("Scanning to collect all fields...")

all_fields = set()
doc_count = 0

for doc in helpers.scan(
    es,
    index=ES_INDEX,
    query=query,
    scroll="2m",
    size=1000
):
    source = doc["_source"]
    all_fields.update(source.keys())
    doc_count += 1

print(f"Found {len(all_fields)} unique fields across {doc_count} documents")

all_fields = sorted(all_fields)

# ========================
# PASS 2: write CSV
# ========================
print("Writing CSV...")

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=all_fields)
    writer.writeheader()

    for doc in helpers.scan(
        es,
        index=ES_INDEX,
        query=query,
        scroll="2m",
        size=1000
    ):
        source = doc["_source"]
        row = {k: normalize(source.get(k, "")) for k in all_fields}
        writer.writerow(row)

print(f"✅ Export completed → {OUTPUT_FILE}")