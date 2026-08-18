# Live Demo Instructions

This runbook demonstrates the complete credit-card fraud MLOps pipeline in execution order. It is written for Windows PowerShell and assumes the repository has already been cloned.

## What the Demo Proves

The demonstration covers:

1. Environment and source-code validation
2. DVC dataset/model versioning
3. Two-model training and comparison
4. MLflow experiment tracking and model registration
5. Dockerized FastAPI deployment
6. Kafka transaction ingestion
7. Spark streaming inference
8. Prometheus metrics and Grafana dashboards
9. Airflow retraining orchestration
10. GitHub Actions CI/CD

## Pipeline Sequence

```text
DVC Dataset
    ↓
Validation and Preprocessing
    ↓
Logistic Regression + Random Forest
    ↓
MLflow Tracking, Registry, and Champion Alias
    ↓
Versioned Model Artifact
    ├──→ FastAPI → Prometheus → Grafana
    └──→ Spark ← Kafka ← Transaction Producer
                     ↓
              Prediction and Invalid Topics

Airflow orchestrates validation, retraining, comparison, registration,
promotion, and drift-baseline generation.
```

## Before the Live Session

Confirm the following before presenting:

- Docker Desktop is installed and its Linux engine is running.
- Conda or Miniconda is installed.
- `creditcard.csv` exists at the repository root.
- The presenting Google account has access to `DA5402W_DVC_Remote` and is listed as an OAuth test user.
- Ports `3000`, `5000`, `8000`, `8080`, `8082`, `9090`, and `9092` are available.
- The repository is connected to GitHub.

Open PowerShell in the project directory:

```powershell
cd "C:\Users\creak\Downloads\DA5402W MLOps Lab\Project"
```

If demonstrating from a different machine, replace that path with the cloned repository path.

## Terminal Layout

Use four terminals during the streaming portion:

| Terminal | Purpose |
|---|---|
| Terminal 1 | Setup, validation, training, API, Airflow, and monitoring commands |
| Terminal 2 | Long-running Spark streaming job |
| Terminal 3 | Kafka transaction producer |
| Terminal 4 | Kafka prediction consumer |

Browser tabs should be prepared for:

- `http://localhost:5000` — MLflow
- `http://localhost:8000/docs` — FastAPI Swagger
- `http://localhost:8082` — Airflow
- `http://localhost:9090/targets` — Prometheus targets
- `http://localhost:3000` — Grafana

## Step 1 — Activate and Verify the Environment

In Terminal 1:

```powershell
conda activate MLDL
$env:PYTHONPATH="src;."

python --version
python -m pip install -r requirements.txt
```

For a new environment:

```powershell
conda create -n MLDL python=3.11 -y
conda activate MLDL
python -m pip install -r requirements.txt
$env:PYTHONPATH="src;."
```

Explain that the dependency files are pinned and separate runtime requirements are provided for the API, Airflow, and Spark images.

## Step 2 — Run Automated Quality Checks

```powershell
pytest
ruff check src api tests scripts airflow/dags
docker compose config --quiet
```

Expected evidence:

```text
14 passed
All checks passed!
```

`docker compose config --quiet` should complete without a configuration error. A Docker credential warning does not invalidate the Compose structure.

## Step 3 — Demonstrate DVC Versioning

Configure the repository's team OAuth client in Git-ignored local DVC configuration:

```powershell
$oauth = Get-Content "config\dvc-google-oauth-client.json" -Raw | ConvertFrom-Json
dvc remote modify --local gdrive gdrive_client_id $oauth.installed.client_id
dvc remote modify --local gdrive gdrive_client_secret $oauth.installed.client_secret
```

The first cloud command on a new machine opens Google authorization. Sign in with an account that has both Drive-folder access and OAuth test-user access.

Confirm the tracked dataset metadata and shared remote:

```powershell
Get-Content creditcard.csv.dvc
dvc remote list
dvc dag
dvc status
dvc status --cloud
dvc pull
```

Expected status:

```text
Data and pipelines are up to date.
Cache and remote 'gdrive' are in sync.
Everything is up to date.
```

To reproduce changed or missing outputs:

```powershell
$env:PYTHONPATH="src"
dvc repro
```

Explain that:

- `creditcard.csv` is excluded from Git.
- `creditcard.csv.dvc` stores the dataset hash and size.
- `.dvc/config` points to the shared Google Drive remote without storing a user authorization token.
- `config/dvc-google-oauth-client.json` identifies the assignment's OAuth application; every user authorizes independently.
- `dvc.yaml` defines the training pipeline.
- `dvc.lock` records the exact dependency, parameter, metric, and output state.
- `params.yaml` controls model settings and promotion gates.

## Step 4 — Start MLflow

Start Docker Desktop before running this command:

```powershell
docker compose up --build -d mlflow
```

Verify health:

```powershell
Invoke-RestMethod http://localhost:5000/health
```

Open:

```text
http://localhost:5000
```

## Step 5 — Train, Compare, Register, and Promote Models

```powershell
$env:PYTHONPATH="src"
$env:MLFLOW_TRACKING_URI="http://localhost:5000"

python -m fraud_mlops.training.train
```

Point out the following in the output:

- `logistic_regression` and `random_forest` candidate results
- Validation and held-out test metrics
- Candidate-specific thresholds
- Selected model and selection rationale
- MLflow run IDs
- Registered model version
- Promotion decision

Open MLflow and show:

1. Experiment `credit-card-fraud-detection`
2. The two candidate runs
3. Parameters and metrics for each model
4. Registered model `credit-card-fraud-detector`
5. Alias `champion`

Latest verified full-data test results:

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.950 | 0.3152 | 0.8878 | 0.4652 | 0.9736 | 0.7120 |
| Random Forest | 0.170 | 0.6056 | 0.8776 | 0.7167 | 0.9738 | 0.8253 |

Explain that thresholds are selected only from validation data and final metrics are reported from the untouched test split.

## Step 6 — Start the Complete Docker Stack

```powershell
docker compose up --build -d
docker compose ps
```

Wait for the required services to become healthy. The local URLs are:

| Service | URL |
|---|---|
| FastAPI Swagger | `http://localhost:8000/docs` |
| MLflow | `http://localhost:5000` |
| Prometheus | `http://localhost:9090` |
| Grafana | `http://localhost:3000` |
| Spark master | `http://localhost:8080` |
| Airflow | `http://localhost:8082` |

Airflow and Grafana use `admin` / `admin` for this local demonstration.

If a service is still starting:

```powershell
docker compose ps
docker compose logs --tail 100 <service-name>
```

## Step 7 — Demonstrate the FastAPI Service

Check readiness and model metadata:

```powershell
Invoke-RestMethod http://localhost:8000/health |
  ConvertTo-Json -Depth 5

Invoke-RestMethod http://localhost:8000/model-info |
  ConvertTo-Json -Depth 5
```

Expected model information includes:

- Model name and type
- Model version
- Classification threshold
- Feature count
- MLflow run ID
- Dataset fingerprint
- Training timestamp

Send a transaction from the dataset:

```powershell
$env:PYTHONPATH="src"

python -c "import pandas as pd, requests; from fraud_mlops.config import FEATURE_COLUMNS; row=pd.read_csv('creditcard.csv', nrows=1).iloc[0]; payload={c:float(row[c]) for c in FEATURE_COLUMNS}; print(requests.post('http://localhost:8000/predict', json=payload, timeout=30).json())"
```

Expected response fields:

```text
fraud_probability
predicted_class
threshold
model_name
model_type
model_version
request_id
```

Also show the interactive endpoint documentation at `http://localhost:8000/docs`.

## Step 8 — Start Spark Streaming Inference

In Terminal 2:

```powershell
cd "C:\Users\creak\Downloads\DA5402W MLOps Lab\Project"

docker compose exec -e PYTHONPATH=/app/src spark-master /opt/spark/bin/spark-submit `
  --conf spark.jars.ivy=/tmp/.ivy2 `
  --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.2 `
  /app/src/fraud_mlops/streaming/spark_streaming.py `
  --bootstrap-servers kafka:29092 `
  --topic credit-card-transactions `
  --output-topic fraud-predictions `
  --invalid-topic fraud-invalid-transactions `
  --model-path /app/models/fraud_model.joblib
```

Wait until Spark reports that its streaming queries are active. Spark must be listening before publishing the demo transactions because it uses the latest Kafka offset by default.

Explain that Spark:

1. Reads JSON transactions from Kafka.
2. Applies the expected 30-feature schema.
3. Rejects incomplete or malformed records.
4. Loads the versioned model artifact.
5. Calculates probability and class.
6. Publishes valid results to `fraud-predictions`.
7. Publishes invalid records to `fraud-invalid-transactions`.

## Step 9 — Publish Transactions to Kafka

In Terminal 3:

```powershell
cd "C:\Users\creak\Downloads\DA5402W MLOps Lab\Project"
conda activate MLDL
$env:PYTHONPATH="src"

python -m fraud_mlops.streaming.kafka_producer `
  --bootstrap-servers localhost:9092 `
  --topic credit-card-transactions `
  --limit 100 `
  --delay-seconds 0.05
```

Expected message:

```text
Published 100 transactions to credit-card-transactions.
```

Return to Terminal 2 briefly and show Spark processing the micro-batches.

## Step 10 — Consume Prediction Results

In Terminal 4:

```powershell
cd "C:\Users\creak\Downloads\DA5402W MLOps Lab\Project"

docker compose exec kafka kafka-console-consumer `
  --bootstrap-server kafka:29092 `
  --topic fraud-predictions `
  --from-beginning `
  --max-messages 5
```

Each output record should include the transaction features plus:

```text
fraud_probability
predicted_class
model_version
processed_at
```

Optional invalid-record inspection:

```powershell
docker compose exec kafka kafka-console-consumer `
  --bootstrap-server kafka:29092 `
  --topic fraud-invalid-transactions `
  --from-beginning
```

Press `Ctrl+C` when an unbounded consumer or Spark stream needs to be stopped.

## Step 11 — Demonstrate Prometheus and Grafana

Show Prometheus targets:

```text
http://localhost:9090/targets
```

The `fraud-api` target should be `UP`.

Inspect raw application metrics:

```powershell
Invoke-WebRequest http://localhost:8000/metrics |
  Select-Object -ExpandProperty Content
```

Generate 100 API predictions so the rolling drift monitor calculates a window:

```powershell
$env:PYTHONPATH="src"

python -c "import pandas as pd, requests; from fraud_mlops.config import FEATURE_COLUMNS; df=pd.read_csv('creditcard.csv', nrows=100); payloads=[{c:float(row[c]) for c in FEATURE_COLUMNS} for _,row in df.iterrows()]; responses=[requests.post('http://localhost:8000/predict', json=p, timeout=30) for p in payloads]; print('successful', sum(r.ok for r in responses))"
```

Inspect drift state:

```powershell
Invoke-RestMethod http://localhost:8000/drift |
  ConvertTo-Json -Depth 6
```

Open Grafana at `http://localhost:3000` and show:

- API request rate
- Prediction latency
- Predictions by class
- API error rate
- Model-loaded status
- Feature drift scores
- Drift flags

The Prometheus datasource and dashboard are provisioned automatically.

## Step 12 — Demonstrate Airflow Retraining

Trigger the workflow from Terminal 1:

```powershell
docker compose exec airflow-scheduler airflow dags trigger credit_card_fraud_retraining
```

Inspect runs:

```powershell
docker compose exec airflow-scheduler airflow dags list-runs `
  --dag-id credit_card_fraud_retraining
```

Open `http://localhost:8082`, sign in, and show the graph:

```text
validate_dataset
    ↓
train_and_compare
    ↓
evaluate_selection
    ↓
record_promotion
    ↓
verify_drift_baseline
```

Explain that the DAG validates the full dataset, compares both candidates, applies the promotion gates, records the MLflow registry decision, and verifies that training refreshed the drift reference statistics.

After a successful retraining run, reload the local model artifact in the API:

```powershell
docker compose restart api
Invoke-RestMethod http://localhost:8000/model-info |
  ConvertTo-Json -Depth 5
```

## Step 13 — Demonstrate GitHub Actions CI/CD

Open the repository’s **Actions** tab on GitHub and select the latest `MLOps CI/CD` run.

Show that the workflow performs:

1. Dependency installation
2. Ruff static checks
3. Pytest with coverage
4. DVC graph validation
5. Synthetic two-model training
6. API Docker-image build
7. GitHub Container Registry publication on pushes

The workflow runs automatically for pull requests, pushes to `main`, and `v*` tags.

## Step 14 — Final Architecture Explanation

Conclude with the following lifecycle:

1. Git versions source and configuration.
2. DVC versions the dataset and generated model pipeline.
3. Airflow orchestrates training and retraining.
4. Logistic Regression and Random Forest provide a meaningful model comparison.
5. MLflow records experiments and identifies the champion model.
6. FastAPI exposes traceable online predictions.
7. Kafka simulates real transaction arrival.
8. Spark validates and scores the stream.
9. Prometheus captures operational and model metrics.
10. Grafana visualizes service behavior and drift.
11. GitHub Actions verifies changes and publishes the API image.

## Shutdown

Stop containers while retaining persistent volumes:

```powershell
docker compose down
```

Only when intentionally resetting all Compose-managed service state:

```powershell
docker compose down -v
```

## Quick Recovery Guide

| Problem | Check or command |
|---|---|
| Docker command cannot connect | Start Docker Desktop and wait for the Linux engine |
| Python cannot import `fraud_mlops` | `$env:PYTHONPATH="src;."` |
| Dataset not found | Configure the included OAuth client, confirm Drive-folder and OAuth test-user access, then run `dvc pull` |
| Google says the DVC app is blocked | Confirm the `gdrive_client_id` and `gdrive_client_secret` from `config\dvc-google-oauth-client.json` were added with `dvc remote modify --local` |
| API reports degraded health | Run training and then `docker compose restart api` |
| MLflow logging fails | Set `$env:MLFLOW_TRACKING_URI="http://localhost:5000"` and check `docker compose logs mlflow` |
| Spark shows no transactions | Start Spark first, then run the Kafka producer again |
| Kafka prediction topic is empty | Check Terminal 2 for Spark errors and verify the model file exists |
| Prometheus target is down | Run `docker compose ps` and confirm the API is healthy |
| Grafana has no data | Generate API predictions and wait for the Prometheus scrape interval |
| Airflow task fails | Inspect `docker compose logs --tail 200 airflow-scheduler` |

## Compact Command Checklist

```powershell
conda activate MLDL
$env:PYTHONPATH="src;."
pytest
ruff check src api tests scripts airflow/dags
dvc status
dvc remote list
dvc status --cloud
docker compose up --build -d mlflow
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
python -m fraud_mlops.training.train
docker compose up --build -d
Invoke-RestMethod http://localhost:8000/model-info
docker compose exec airflow-scheduler airflow dags trigger credit_card_fraud_retraining
docker compose ps
```

The Spark job, Kafka producer, and Kafka consumer commands must still be run in their separate terminals as described above.
