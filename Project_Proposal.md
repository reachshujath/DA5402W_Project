# Project Proposal

# Real-Time Credit Card Fraud Detection using an End-to-End MLOps Pipeline

## Course
**MLOps**

---

# 1. Introduction

Machine Learning models are increasingly deployed in production environments where they must continuously process new data, remain reproducible, be monitored for performance degradation, and support automated retraining and deployment. Traditional software engineering practices alone are insufficient for managing the lifecycle of machine learning systems, giving rise to the field of Machine Learning Operations (MLOps).

This project proposes the development of an end-to-end MLOps pipeline for real-time credit card fraud detection. Rather than focusing on designing a complex machine learning algorithm, the emphasis of this project will be on implementing a production-grade MLOps workflow using modern data engineering, model management, deployment, and monitoring tools.

---

# 2. Project Objective

The objective of this project is to demonstrate the practical application of MLOps principles by integrating various tools commonly used in industrial machine learning pipelines.

The project will showcase:

- Real-time data ingestion
- Distributed data processing
- Automated model training
- Experiment tracking
- Model versioning
- Model deployment
- Continuous monitoring
- CI/CD automation
- Dataset and model reproducibility

The machine learning model itself will be intentionally simple (e.g., Logistic Regression or XGBoost) so that the primary focus remains on the MLOps lifecycle.

---

# 3. Problem Statement

Financial institutions process millions of credit card transactions every day. Detecting fraudulent transactions in real time is critical to minimizing financial losses.

This project will simulate a real-world fraud detection system where incoming transactions are streamed through a data pipeline, processed, classified using a trained machine learning model, and monitored continuously for performance and drift.

---

# 4. Dataset

The project will use the publicly available **Credit Card Fraud Detection Dataset** published by the Machine Learning Group (MLG) of Université Libre de Bruxelles.

**Dataset Link**

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Dataset Characteristics

- 284,807 credit card transactions
- Binary classification
  - Class 0 → Legitimate transaction
  - Class 1 → Fraudulent transaction
- Highly imbalanced dataset
- PCA-transformed features
- Suitable for demonstrating streaming inference and automated retraining

---

# 5. Proposed System Architecture

```
                           Credit Card Dataset
                                    │
                                    ▼
                       Kafka Producer (Streaming)
                                    │
                                    ▼
                             Apache Kafka
                                    │
                                    ▼
                      Apache Spark Structured Streaming
                                    │
                                    ▼
                      Feature Processing & Validation
                                    │
                                    ▼
                        Machine Learning Prediction
                                    │
                                    ▼
                           FastAPI Prediction API
                                    │
                ┌───────────────────┴───────────────────┐
                │                                       │
                ▼                                       ▼
          MLflow Tracking                    Prometheus Metrics
                │                                       │
                ▼                                       ▼
         Model Registry                      Grafana Dashboard
                │
                ▼
         Apache Airflow DAG
                │
                ▼
      Automated Retraining Pipeline
```

---

# 6. Technology Stack

| Component | Tool |
|------------|------|
| Programming Language | Python |
| Data Streaming | Apache Kafka |
| Distributed Processing | Apache Spark Structured Streaming |
| Workflow Orchestration | Apache Airflow |
| Machine Learning | Scikit-learn / XGBoost |
| Experiment Tracking | MLflow |
| API Serving | FastAPI |
| Containerization | Docker & Docker Compose |
| Version Control | Git & GitHub |
| Data & Model Versioning | DVC |
| Monitoring | Prometheus |
| Visualization | Grafana |
| CI/CD | GitHub Actions |

---

# 7. Project Workflow

## Phase 1 – Data Ingestion

- Load the historical dataset.
- Simulate live transaction streams using a Kafka Producer.
- Publish transactions to a Kafka topic.

---

## Phase 2 – Stream Processing

Apache Spark Structured Streaming will:

- Consume transactions from Kafka
- Parse incoming messages
- Perform feature preprocessing
- Send transactions for prediction

---

## Phase 3 – Model Development

A baseline machine learning model (e.g., Logistic Regression or XGBoost) will be trained using the historical dataset.

Model performance metrics such as Precision, Recall, F1-score, and ROC-AUC will be evaluated.

---

## Phase 4 – Experiment Tracking

MLflow will record:

- Hyperparameters
- Training metrics
- Model artifacts
- Model versions

The best-performing model will be registered in the MLflow Model Registry.

---

## Phase 5 – Model Serving

The trained model will be deployed using FastAPI.

The API will expose endpoints for fraud prediction, allowing external applications to perform real-time inference.

---

## Phase 6 – Workflow Automation

Apache Airflow will orchestrate automated workflows including:

- Dataset validation
- Model retraining
- Model evaluation
- Model registration
- Deployment

---

## Phase 7 – Monitoring

Prometheus and Grafana will be used to monitor:

- API latency
- Request throughput
- Prediction statistics
- Resource utilization
- Model performance over time

Basic data drift detection will also be incorporated to demonstrate model monitoring.

---

## Phase 8 – Version Control and CI/CD

Git will manage source code while DVC will version datasets and trained models.

GitHub Actions workflows will automate:

- Code validation
- Unit testing
- Docker image creation
- Automated deployment

---

# 8. Expected Deliverables

The project will deliver:

- Complete MLOps pipeline implementation
- Real-time streaming data ingestion
- Automated training workflow
- Model registry using MLflow
- REST API for prediction
- Monitoring dashboard
- Dockerized deployment
- CI/CD pipeline
- Source code repository
- Project documentation

---

# 9. Expected Learning Outcomes

Upon completion of the project, the following competencies will be demonstrated:

- Understanding of the complete MLOps lifecycle
- Building reproducible ML workflows
- Implementing real-time streaming architectures
- Managing machine learning experiments
- Deploying production-ready ML services
- Monitoring deployed models
- Automating ML pipelines using orchestration tools
- Applying CI/CD principles to machine learning systems

---

# 10. Scope of the Project

The project focuses on demonstrating modern MLOps practices rather than developing a highly optimized fraud detection algorithm.

The emphasis will therefore be placed on:

- Pipeline automation
- Scalability
- Reproducibility
- Deployment
- Monitoring
- Operational excellence

The machine learning model serves as a representative workload for illustrating these concepts.

---

# 11. Future Enhancements

Potential future improvements include:

- Online learning and incremental model updates
- Feature Store integration
- Kubernetes deployment
- Auto-scaling inference services
- Advanced drift detection
- Explainable AI using SHAP
- Canary deployments
- Multi-model A/B testing

---

# 12. Conclusion

This project demonstrates an end-to-end production-oriented MLOps pipeline using a real-world fraud detection use case. By integrating tools such as Apache Kafka, Apache Spark, Apache Airflow, MLflow, FastAPI, Docker, Prometheus, Grafana, GitHub Actions, and DVC, the project illustrates the complete lifecycle of deploying and managing machine learning systems in production.

The proposed implementation prioritizes operational robustness, reproducibility, automation, and scalability while keeping the underlying machine learning model intentionally simple. This approach aligns with the primary objective of the course: to gain hands-on experience with modern MLOps practices and tooling.
