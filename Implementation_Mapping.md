# Evaluation Criteria Implementation Mapping

This file maps the course requirements to concrete repository evidence after completion.

| Criterion | Implementation evidence |
|---|---|
| Git and GitHub | Git repository, GitHub remote, staged implementation history, GitHub Actions workflow |
| Automated Airflow pipeline | Five-stage `credit_card_fraud_retraining` DAG covering validation through registry and drift baseline |
| Additional data engineering tool | Kafka ingestion plus Spark schema validation, inference, prediction topic, and invalid-message topic |
| Data processing | Schema/missing/type/range validation, stratified splitting, scaling, PCA-feature passthrough, balanced estimators |
| Two model comparison | Balanced Logistic Regression and Random Forest evaluated with validation-only threshold selection |
| Proper metrics | Precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, and explicit promotion gates |
| MLflow | Candidate runs, parameters, metrics, evaluation artifacts, model registry, and `champion` alias |
| Dataset/model versioning | Initialized `.dvc/`, `creditcard.csv.dvc`, parameterized `dvc.yaml`, and generated `dvc.lock` |
| Deployment | FastAPI endpoints, traceable prediction response, purpose-built API image, complete Compose deployment |
| Monitoring/logging | Structured logs, Prometheus operational/model/drift metrics, provisioned Grafana dashboard |
| CI/CD | GitHub Actions lint, tests/coverage, synthetic training, DVC validation, image build, and GHCR publication |
| Documentation | README architecture diagram, rationale, setup, API, Docker, DVC, streaming, Airflow, monitoring, and CI commands |

## Verification Commands

```powershell
$env:PYTHONPATH="src;."
pytest
ruff check src api tests scripts airflow/dags
dvc dag
dvc repro
docker compose config --quiet
```

Runtime integration evidence is produced by starting Compose, publishing Kafka transactions, reading `fraud-predictions`, triggering the Airflow DAG, and viewing the provisioned Grafana dashboard.
