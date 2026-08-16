# MLOps Project Implementation Plan

## Project

Real-Time Credit Card Fraud Detection using an End-to-End MLOps Pipeline

## Source Materials Used

- `Project_Proposal.md`
- `creditcard.csv`

## Dataset Summary

The project uses the credit card fraud detection dataset described in the proposal.

- Rows: 284,807 transactions
- Columns: 31
- Features: `Time`, `V1` to `V28`, `Amount`
- Target: `Class`
- Legitimate transactions: 284,315
- Fraud transactions: 492
- Fraud rate: approximately 0.173%
- Missing values: none detected
- Challenge: severe class imbalance

Because the dataset is anonymized and PCA-transformed, the implementation should focus on reliable pipeline behavior, reproducibility, monitoring, and deployment rather than complex feature engineering.

## Implementation Strategy

Build the project in layers, starting with a working local baseline and then adding MLOps components around it. This reduces integration risk and makes each phase demonstrable.

## Phase 1: Repository and Environment Setup

### Goals

- Create a clean project structure.
- Make the environment reproducible.
- Prepare local development and Docker-based execution.

### Tasks

- Create the repository layout:
  - `data/raw/`
  - `data/processed/`
  - `src/`
  - `src/training/`
  - `src/inference/`
  - `src/streaming/`
  - `src/validation/`
  - `api/`
  - `airflow/dags/`
  - `monitoring/prometheus/`
  - `monitoring/grafana/`
  - `tests/`
  - `models/`
  - `notebooks/`
- Add Python dependency management with `requirements.txt` or `pyproject.toml`.
- Add `.gitignore`.
- Initialize DVC for dataset and model versioning.
- Store `creditcard.csv` under DVC control instead of committing it directly.
- Add a basic `README.md` with setup and run instructions.

### Deliverables

- Reproducible project skeleton
- Tracked dataset metadata through DVC
- Local Python environment ready for development

## Phase 2: Data Validation and Preprocessing

### Goals

- Validate incoming transaction data before training or inference.
- Define a stable schema for batch and streaming pipelines.

### Tasks

- Define the expected schema:
  - `Time`: float
  - `V1` to `V28`: float
  - `Amount`: float
  - `Class`: integer, used only for training/evaluation
- Implement validation checks:
  - Required columns exist
  - No missing values
  - Numeric columns have valid numeric types
  - `Class` contains only `0` and `1`
  - `Amount` is non-negative
- Create preprocessing logic:
  - Split features and target
  - Scale `Amount`
  - Optionally scale `Time`
  - Preserve `V1` to `V28` as already transformed features
- Save processed train/test splits.
- Use stratified splitting because of class imbalance.

### Deliverables

- Data validation module
- Preprocessing pipeline
- Reproducible train/test split

## Phase 3: Baseline Model Training

### Goals

- Train a simple but defensible fraud classifier.
- Prioritize recall, precision, F1-score, PR-AUC, and ROC-AUC over accuracy.

### Recommended Models

Start with:

- Logistic Regression with `class_weight="balanced"`
- Random Forest or XGBoost as an optional second model

### Tasks

- Train baseline Logistic Regression.
- Evaluate with:
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - PR-AUC
  - Confusion matrix
- Tune the classification threshold instead of relying only on `0.5`.
- Save the trained preprocessing pipeline and model together.
- Add unit tests for preprocessing and prediction shape.

### Deliverables

- Baseline model artifact
- Evaluation report
- Threshold selection notes
- Training script

## Phase 4: MLflow Experiment Tracking and Model Registry

### Goals

- Track training runs and register the best model.
- Make model selection reproducible.

### Tasks

- Run local MLflow tracking server.
- Log:
  - Dataset version
  - Parameters
  - Metrics
  - Confusion matrix artifact
  - Model artifact
  - Classification threshold
- Register the best model in MLflow Model Registry.
- Define model stages:
  - `Staging`
  - `Production`
  - `Archived`

### Deliverables

- MLflow experiment
- Registered fraud detection model
- Reproducible model training record

## Phase 5: FastAPI Prediction Service

### Goals

- Serve the registered model through a REST API.
- Expose metrics for monitoring.

### API Endpoints

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `GET /metrics`

### Tasks

- Load the production model from MLflow or a local model path.
- Validate incoming JSON requests.
- Return:
  - Fraud probability
  - Predicted class
  - Model version
  - Threshold used
- Add Prometheus-compatible metrics:
  - Request count
  - Error count
  - Prediction latency
  - Fraud prediction count
  - Legitimate prediction count

### Deliverables

- FastAPI inference service
- API schema
- Prediction tests
- Dockerfile for API service

## Phase 6: Kafka-Based Streaming Simulation

### Goals

- Simulate real-time credit card transactions from the historical dataset.
- Stream transaction records into Kafka.

### Tasks

- Create a Kafka producer that reads `creditcard.csv`.
- Publish records one by one to a topic such as `credit-card-transactions`.
- Exclude `Class` from inference messages, but optionally publish it to a separate evaluation stream for simulation.
- Add configurable stream speed:
  - Fixed delay mode
  - Burst mode
  - Replay mode
- Use JSON as the message format.

### Deliverables

- Kafka producer script
- Kafka topic configuration
- Streaming replay documentation

## Phase 7: Spark Structured Streaming Processing

### Goals

- Consume transactions from Kafka.
- Parse, validate, and route transactions for prediction.

### Tasks

- Configure Spark Structured Streaming to consume Kafka messages.
- Apply the transaction schema.
- Validate records.
- Send valid records to the FastAPI prediction endpoint or run local batch inference inside Spark.
- Write prediction results to:
  - Console for demo
  - Local file sink for audit
  - Optional Kafka output topic such as `fraud-predictions`

### Deliverables

- Spark streaming job
- Prediction output stream
- Invalid-record handling

## Phase 8: Airflow Training and Retraining Workflow

### Goals

- Orchestrate the ML lifecycle.
- Demonstrate automated retraining.

### DAG Tasks

1. Validate dataset
2. Preprocess dataset
3. Train model
4. Evaluate model
5. Compare against current production model
6. Register model if performance passes threshold
7. Optionally promote model to staging

### Promotion Criteria

Use explicit rules such as:

- New model PR-AUC must be greater than or equal to current production PR-AUC.
- Recall must not fall below a chosen minimum.
- Precision must not fall below a chosen minimum.

### Deliverables

- Airflow DAG
- Automated retraining workflow
- Model promotion rules

## Phase 9: Monitoring with Prometheus and Grafana

### Goals

- Monitor the prediction API and model behavior.
- Demonstrate operational visibility.

### Metrics to Track

- API request rate
- API error rate
- API latency p50/p95/p99
- Total predictions
- Fraud prediction rate
- Average fraud probability
- Input feature summary statistics for `Amount` and selected PCA features
- Drift indicators

### Grafana Dashboard Panels

- API health
- Request throughput
- Latency
- Error rate
- Fraud vs legitimate predictions
- Fraud probability distribution
- Data drift summary

### Deliverables

- Prometheus configuration
- Grafana dashboard JSON
- Monitoring screenshots for final report

## Phase 10: Drift Detection

### Goals

- Add basic model monitoring beyond infrastructure metrics.

### Tasks

- Define a reference dataset from the training split.
- Compare live input windows against reference statistics.
- Start with simple checks:
  - Amount distribution shift
  - Prediction fraud-rate shift
  - Feature mean/std shift for selected features
- Optionally use Evidently AI for richer reports.
- Emit drift metrics to Prometheus or save periodic drift reports.

### Deliverables

- Drift detection module
- Drift report or dashboard metric
- Alert condition for large drift

## Phase 11: Docker Compose Integration

### Goals

- Run the full system locally with one command.

### Services

- Kafka
- Zookeeper or KRaft Kafka mode
- Spark
- MLflow
- FastAPI
- Airflow webserver
- Airflow scheduler
- Prometheus
- Grafana

### Tasks

- Create `docker-compose.yml`.
- Add service health checks.
- Mount data and model volumes.
- Document startup order.
- Provide demo commands for:
  - Training
  - Starting services
  - Streaming transactions
  - Viewing dashboards

### Deliverables

- Dockerized local MLOps stack
- End-to-end demo workflow

## Phase 12: CI/CD Pipeline

### Goals

- Automate quality checks and build steps.

### GitHub Actions CI/CD Stages

1. Lint
2. Test
3. Build Docker images
4. Run training smoke test
5. Package artifacts
6. Deploy locally or to target environment

### Tasks

- Add unit tests for:
  - Data validation
  - Preprocessing
  - Model loading
  - API prediction response
- Add integration smoke tests for API startup.
- Build Docker images in CI.
- Optionally publish images to a container registry.

### Deliverables

- `.gitlab-ci.yml`
- Automated tests
- Docker image build pipeline

## Suggested Timeline

| Week | Focus | Output |
|---|---|---|
| 1 | Repo setup, DVC, data validation | Reproducible project skeleton |
| 2 | Preprocessing and baseline model | Trained baseline model and metrics |
| 3 | MLflow tracking and registry | Registered model |
| 4 | FastAPI serving | Working prediction API |
| 5 | Kafka producer and Spark streaming | Real-time simulation pipeline |
| 6 | Airflow retraining DAG | Automated training workflow |
| 7 | Prometheus, Grafana, drift checks | Monitoring dashboard |
| 8 | Docker Compose, CI/CD, documentation | Final integrated demo |

## Minimum Viable Demo Path

If time becomes limited, prioritize this order:

1. Train and register a model with MLflow.
2. Serve predictions with FastAPI.
3. Simulate streaming with a Kafka producer.
4. Add Prometheus metrics and Grafana dashboard.
5. Add Airflow retraining DAG.
6. Add Spark only after the API, Kafka, and MLflow flow is stable.

This still demonstrates the main MLOps lifecycle while avoiding late-stage integration failure.

## Final Demonstration Script

1. Start services with Docker Compose.
2. Open MLflow and show tracked experiments.
3. Show the registered production model.
4. Start the FastAPI service.
5. Send one manual prediction request.
6. Start the Kafka producer to replay transactions.
7. Run the streaming consumer or Spark job.
8. Show prediction outputs.
9. Open Prometheus metrics.
10. Open Grafana dashboard.
11. Trigger or show the Airflow retraining DAG.
12. Explain how DVC and CI/CD support reproducibility.

## Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Severe class imbalance makes accuracy misleading | Use PR-AUC, recall, precision, F1-score, and threshold tuning |
| Too many tools increase integration complexity | Build a working baseline first, then add one MLOps tool at a time |
| Spark, Kafka, and Airflow may be heavy locally | Use Docker Compose and keep each service independently testable |
| Dataset has anonymized PCA features | Focus on pipeline reliability and operational monitoring |
| Fraud labels are not available in real-time production | Use labels only in training and simulated offline evaluation |

## Success Criteria

- Dataset is versioned and reproducible.
- Model training is automated and tracked.
- Best model is registered in MLflow.
- FastAPI serves predictions from a versioned model.
- Kafka simulates transaction streaming.
- Spark or a streaming consumer processes transactions.
- Prometheus collects API and prediction metrics.
- Grafana visualizes service and model behavior.
- Airflow runs the retraining workflow.
- CI/CD validates code and builds containers.
- The full system can be demonstrated locally.
