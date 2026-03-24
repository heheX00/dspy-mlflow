# DSPy + MLflow Research Spike

A 2-week research sprint to explore programmatic prompting (DSPy) and ML lifecycle tracking (MLflow), building toward a reproducible Text-to-ESQL pipeline.

## Sprint Objectives

1. **Programmatic Prompting** — Understand the shift from manual prompt engineering to metric-driven, algorithmic optimization (DSPy).
2. **ML Lifecycle** — Recognize why saving configurations, datasets, and generated artifacts is critical for reproducibility (MLflow).
3. **Pipeline Integration** — Pass data and models between DSPy (orchestration) and MLflow (tracking).
4. **Knowledge Handoff** — Produce documentation and a presentation for the core engineering team.

## Project Structure

```
dspy-mlflow/
├── team_a_dspy/          # Team A: DSPy exploration
│   ├── signatures/       # DSPy signature definitions
│   ├── metrics/          # Evaluation metrics
│   ├── optimizers/       # Optimizer experiments
│   └── data/             # Dummy/sample datasets
├── team_b_mlflow/        # Team B: MLflow exploration
│   ├── tracking/         # MLflow tracking scripts
│   └── artifacts/        # Saved artifacts
├── integration/          # Joint: Combined DSPy + MLflow pipeline
├── docs/                 # Knowledge transfer documentation
│   ├── dspy_guide.md     # Team A: DSPy concepts & how-to
│   ├── mlflow_guide.md   # Team B: MLflow dashboard & artifacts
│   └── presentation/     # Final presentation materials
└── experiments/          # Phase 3: Broad experimentation runs
```

## Timeline

| Phase | Days | Focus |
|-------|------|-------|
| 1. Foundations | Days 1–3 | Learn tools in isolation, build "Hello World" |
| 2. Integration | Days 4–6 | Connect DSPy output to MLflow tracking |
| 3. Experimentation | Days 7–8 | Run 3–4 distinct experiments, populate dashboard |
| 4. Knowledge Transfer | Days 9–10 | Documentation, presentation, handoff |

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### MLflow Server (Team B)

```bash
mlflow server --host 127.0.0.1 --port 5000
```

### Run the Pipeline (Post-Integration)

```bash
python integration/run_pipeline.py
```

## DSPy Optimizer Workflow

This repository now includes an optimizer script for the Team A query generator:

- Script: `team_a_dspy/optimizers/optimize_query_generator.py`
- Dataset template: `team_a_dspy/data/optimizer_trainset.example.jsonl`

Steps:

1. Copy `team_a_dspy/data/optimizer_trainset.example.jsonl` to `team_a_dspy/data/optimizer_trainset.jsonl`.
2. Add more rows in JSONL format with:
	 - `nl_query`
	 - `expected_query_dsl`
3. Run optimization:

```bash
python team_a_dspy/optimizers/optimize_query_generator.py \
	--trainset team_a_dspy/data/optimizer_trainset.jsonl \
	--output team_a_dspy/optimizers/artifacts/optimized_query_generator.json
```

The script uses `dspy.BootstrapFewShot` with an exact-DSL metric and saves a compiled DSPy artifact.

## Teams

- **Team A (DSPy)**: Signatures, metrics, optimizers
- **Team B (MLflow)**: Tracking server, logging, artifact management
