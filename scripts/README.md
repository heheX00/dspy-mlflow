# Elasticsearch Sandbox Setup (DSPy)

This module sets up a local Elasticsearch sandbox and loads large-scale GDELT-style CSV data for testing DSPy query generation and evaluation.

It consists of:

- `scripts/create_index.py` → creates the Elasticsearch index with mapping
- `scripts/load_csv_to_es.py` → loads CSV data into Elasticsearch (chunked)
- `data/` → contains input CSV files (e.g. `output_last_7_days.csv`)

---

# 📦 Project Structure

```
dspy-mlflow/
├── scripts/
│   ├── create_index.py
│   └── load_csv_to_es.py
├── data/
│   └── output_last_7_days.csv
└── README.md
```

---

# 🚀 Prerequisites

### 1. Docker (Elasticsearch)

Start Elasticsearch sandbox:

```cmd
docker run -d ^
--name es-sandbox ^
-p 9200:9200 ^
-e discovery.type=single-node ^
-e xpack.security.enabled=false ^
-e ES_JAVA_OPTS="-Xms2g -Xmx2g" ^
docker.elastic.co/elasticsearch/elasticsearch:8.12.0
```

Check it is running:

```cmd
curl http://localhost:9200
```

---

### 2. Python Environment

Install required packages:

```cmd
pip install pandas elasticsearch
```

---

# 🧱 Step 1: Create Index

Creates index `gkg_sandbox` with predefined mapping.

```cmd
python scripts\create_index.py
```

### Notes

- If index already exists, it will skip creation
- To reset index:

```cmd
curl -X DELETE http://localhost:9200/gkg_sandbox
python scripts\create_index.py
```

---

# 📥 Step 2: Load CSV Data

Loads CSV data into Elasticsearch using chunked streaming.

```cmd
python scripts\load_csv_to_es.py
```

---

# ⚙️ How It Works

## `load_csv_to_es.py`

Designed for **large datasets (10GB+)**

### Key Features

- Reads CSV in chunks (`5000 rows`)
- Uses `streaming_bulk` for stable ingestion
- Converts flat CSV → nested Elasticsearch documents
- Handles type normalization:
  - `long` fields
  - `float` fields
- Builds `geo_point` (`location`) from lat/lon
- Drops problematic fields:
  - `_id`, `_index`, `_score`
  - `.keyword` export columns
  - parent object conflicts (e.g. `GkgRecordId`)
- Prints progress after each chunk

---

## Example Output

```
Chunk 1 done: 5000 data loaded (chunk rows=5000, success=5000, failed=0, avg_rate=850.21 docs/sec)
Chunk 2 done: 10000 data loaded (chunk rows=5000, success=5000, failed=0, avg_rate=912.44 docs/sec)
```

---

# 🧠 Important Notes

## 1. CSV Format (GDELT-style)

The CSV contains flattened fields like:

```
GkgRecordId.Date
V2Persons.V1Person
V2Locations.CountryCode
V2EnhancedThemes.V2Theme
```

These are converted into nested JSON:

```json
{
  "GkgRecordId": {
    "Date": 20240301
  },
  "V2Persons": {
    "V1Person": "Donald Trump"
  }
}
```

---

## 2. `.keyword` Columns

CSV exports may include:

```
V2Persons.V1Person.keyword
```

These are **ignored** because mapping already defines `.keyword` subfields.

---

## 3. Mapping Conflicts (Critical)

Common error:

```
object mapping for [GkgRecordId] tried to parse field as object but found concrete value
```

This is handled automatically by:

- removing parent scalar fields
- keeping only structured nested fields

---

## 4. Performance Expectations

For ~10GB CSV:

| Environment | Expected Time |
|------------|-------------|
| SSD + 4GB RAM ES | ~30–90 minutes |
| HDD / low RAM | several hours |

---

## 5. Resource Recommendations

Set Elasticsearch heap:

```cmd
-e ES_JAVA_OPTS="-Xms2g -Xmx2g"
```

Minimum:

- Docker memory: **4GB**
- Prefer SSD

---

# 🔍 Verification

Check document count:

```cmd
curl http://localhost:9200/gkg_sandbox/_count
```

Sample query:

```cmd
curl -X GET "http://localhost:9200/gkg_sandbox/_search?pretty" ^
-H "Content-Type: application/json" ^
-d "{\"size\":3,\"query\":{\"match_all\":{}}}"
```

---

# 🧪 Example Aggregation

Top 10 people:

```json
{
  "size": 0,
  "aggs": {
    "top_people": {
      "terms": {
        "field": "V2Persons.V1Person.keyword",
        "size": 10
      }
    }
  }
}
```

---

# 🧩 How This Fits DSPy

This sandbox is used for:

- Query DSL generation testing
- DSPy evaluation ("judge")
- Validation of:
  - syntax correctness
  - semantic correctness
  - ranking accuracy

---

# ⚠️ Troubleshooting

## All rows failing

Check errors printed:

```
document_parsing_exception
```

Usually caused by:

- wrong field types
- nested/object conflicts
- bad numeric values

---

## Slow ingestion

- Increase ES memory
- Reduce `READ_CHUNK_SIZE` (e.g. 2000)
- Ensure SSD

---

## Container crash

- Increase Docker RAM
- Check logs:

```cmd
docker logs es-sandbox
```

---

# ✅ Recommended Workflow

```cmd
docker run ... (start ES)

curl -X DELETE http://localhost:9200/gkg_sandbox
python scripts\create_index.py
python scripts\load_csv_to_es.py
```

---

# 📌 Next Steps

You can extend this with:

- DSPy evaluation pipeline
- Query validation metrics
- Auto schema extraction → `schema_context.txt`
- MLflow tracking integration
