# Environment Recreation Guide

This guide explains how teammates can recreate the local environment used for the credit card fraud MLOps pipeline.

## Recommended Setup

Use:

- Windows PowerShell
- Anaconda or Miniconda
- Docker Desktop
- Git

The verified local environment was named:

```text
MLDL
```

Teammates can use the same name or choose their own. The commands below use `MLDL` for consistency.

## 1. Clone The Repository

```powershell
git clone https://github.com/reachshujath/DA5402W_Project.git
cd DA5402W_Project
```

## 2. Create The Conda Environment

Create a new environment with Python 3.12:

```powershell
conda create -n MLDL python=3.12 -y
conda activate MLDL
```

Install the Python dependencies:

```powershell
pip install -r requirements.txt
```

If Airflow causes dependency resolution issues on a teammate machine, use the Docker-based Airflow services from `docker-compose.yml` instead of installing/running Airflow locally.

## 3. Verify The Environment

Check the Python executable:

```powershell
python -c "import sys; print(sys.executable)"
```

The output should include:

```text
envs\MLDL
```

Check required packages:

```powershell
python -c "import pandas, sklearn, joblib, fastapi, uvicorn, mlflow, prometheus_client, confluent_kafka, pyspark; print('ok')"
```

Expected output:

```text
ok
```

## 4. Place The Dataset

The dataset is not committed to GitHub because it is large.

Download the Kaggle Credit Card Fraud Detection dataset and place the file at the repository root:

```text
creditcard.csv
```

The final path should look like:

```text
DA5402W_Project\creditcard.csv
```

## 5. Set `PYTHONPATH`

In every new PowerShell terminal where you run project Python modules, set:

```powershell
$env:PYTHONPATH="src;."
```

For commands that only need the `fraud_mlops` package, this is also enough:

```powershell
$env:PYTHONPATH="src"
```

Command Prompt uses different syntax:

```cmd
set PYTHONPATH=src;.
```

PowerShell is recommended for this project because the runbook commands use PowerShell syntax.

## 6. Run Tests

```powershell
$env:PYTHONPATH="src;."
pytest
```

Expected:

```text
14 passed
```

Warnings are acceptable.

## 7. Train The Model

```powershell
$env:PYTHONPATH="src"
python -m fraud_mlops.training.train
```

This generates:

- `models/fraud_model.joblib`
- `reports/metrics.json`
- `reports/reference_stats.json`
- `data/processed/x_train.csv`
- `data/processed/x_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`

For a faster smoke test:

```powershell
$env:PYTHONPATH="src"
python -m fraud_mlops.training.train --sample-rows 5000
```

## 8. Run The API

```powershell
$env:PYTHONPATH="src;."
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open Swagger:

```text
http://127.0.0.1:8000/docs
```

Verify:

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `GET /metrics`

## 9. Start Docker Services

Docker Desktop must be running.

Start the main infrastructure:

```powershell
docker compose up -d zookeeper kafka prometheus grafana spark-master spark-worker
```

Check status:

```powershell
docker compose ps
```

## 10. Kafka Verification

Publish sample transactions:

```powershell
$env:PYTHONPATH="src"
python -m fraud_mlops.streaming.kafka_producer --limit 100 --delay-seconds 0.05
```

Read messages:

```powershell
docker compose exec kafka kafka-console-consumer `
  --bootstrap-server kafka:29092 `
  --topic credit-card-transactions `
  --from-beginning `
  --max-messages 5
```

Expected: five JSON transaction records.

## 11. Spark Verification

Run the Spark streaming consumer:

```powershell
docker compose exec -e PYTHONPATH=/app/src spark-master /opt/spark/bin/spark-submit `
  --conf spark.jars.ivy=/tmp/.ivy2 `
  --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.2 `
  /app/src/fraud_mlops/streaming/spark_streaming.py `
  --bootstrap-servers kafka:29092 `
  --topic credit-card-transactions `
  --checkpoint /tmp/fraud-stream-checkpoint-v2
```

If the streaming job shows empty batches, publish more Kafka messages from another terminal:

```powershell
$env:PYTHONPATH="src"
python -m fraud_mlops.streaming.kafka_producer --limit 20 --delay-seconds 0.05
```

Expected: Spark prints parsed transaction rows.

## 12. MLflow

Start MLflow:

```powershell
mlflow ui --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

Expected experiment:

```text
credit-card-fraud-detection
```

Expected registered model:

```text
credit-card-fraud-detector
```

## 13. Prometheus And Grafana

Prometheus is configured to scrape the API service inside Docker Compose:

```text
api:8000
```

Open Prometheus targets:

```text
http://localhost:9090/targets
```

Expected:

```text
fraud-api UP
```

Open Grafana:

```text
http://localhost:3000
```

Default login:

```text
Username: admin
Password: admin
```

The Prometheus data source and fraud dashboard are provisioned automatically from `monitoring/grafana/`.

Generate predictions through Swagger to populate the dashboard.

## 14. Airflow

Start Airflow:

```powershell
docker compose up -d airflow-postgres airflow-init airflow-webserver airflow-scheduler
```

Open:

```text
http://localhost:8082
```

Login:

```text
Username: admin
Password: admin
```

Trigger:

```text
credit_card_fraud_retraining
```

Expected successful tasks:

- `validate_dataset`
- `train_and_compare`
- `evaluate_selection`
- `record_promotion`
- `verify_drift_baseline`

## 15. DVC

DVC is initialized and `creditcard.csv.dvc` versions the dataset metadata. A shared remote is not configured by default.

After installing DVC, restore from a team remote or place the dataset at the root and reproduce the pipeline:

```powershell
$env:PYTHONPATH="src"
dvc repro
```

Do not commit the raw dataset directly to Git.

## Troubleshooting

| Issue | Fix |
|---|---|
| `fraud_mlops` cannot be imported | Set `$env:PYTHONPATH="src;."` in the current terminal |
| Command fails with `$env:PYTHONPATH` syntax error | You are likely in Command Prompt; switch to PowerShell or use `set PYTHONPATH=src;.` |
| `kafka.vendor.six.moves` error | Ensure `confluent-kafka` is installed and the producer imports `confluent_kafka.Producer` |
| Prometheus target is `DOWN` | Confirm the Compose `api` service is healthy and Prometheus is scraping `api:8000` |
| Spark Kafka connector mismatch | Use connector `org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.2` with the current Spark image |
| Spark cannot write Ivy cache | Include `--conf spark.jars.ivy=/tmp/.ivy2` |
| Airflow training fails with a missing Python package | Rebuild the custom image with `docker compose build airflow-init airflow-webserver airflow-scheduler` |

## Optional: Export Your Own Exact Environment

If one teammate has a working Conda environment and wants to share a closer snapshot:

```powershell
conda activate MLDL
conda env export --from-history > environment.yml
```

Other teammates can recreate it with:

```powershell
conda env create -f environment.yml
conda activate MLDL
```

For a more exact but less portable export:

```powershell
conda env export > environment-full.yml
```
