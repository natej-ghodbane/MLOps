# 🚀 MLOps Churn Prediction Platform

An **end-to-end MLOps project** for **Customer Churn Prediction** combining  
**Machine Learning, MLflow, FastAPI, Streamlit, Docker, Jenkins CI/CD, and Monitoring with ELK Stack**.

This project demonstrates a **production-style MLOps architecture** with:
- Offline CLI training
- Online API retraining
- Experiment tracking with MLflow
- Interactive Streamlit dashboard
- **System & application monitoring with Elasticsearch and Kibana (Excellence)**
- Full Dockerized deployment

---

## 📂 Project Structure

```
├── ci-cd/
│   └── Jenkinsfile              # CI/CD pipeline
├── data/
│   ├── churn-bigml-80.csv
│   └── churn-bigml-20.csv
├── docker/
│   ├── docker-compose.yml       # Multi-service stack (API, MLflow, ELK, UI)
│   ├── Dockerfile.api           # FastAPI service
│   ├── Dockerfile.mlflow        # MLflow tracking server
│   └── Dockerfile.streamlit     # Streamlit UI
├── models/                      # Saved models & preprocessors
├── src/
│   ├── api/                     # FastAPI backend
│   │   └── api.py
│   ├── ml/                      # ML pipeline
│   │   └── model_pipeline.py
│   ├── monitoring/              # Elasticsearch logging
│   │   └── es_logger.py
│   └── ui/                      # Streamlit interface
│       └── streamlit_app.py
├── mlruns/                      # Local MLflow runs (CLI)
├── Makefile
├── requirements.txt
└── README.md
```

---

## 🧠 Key Features

### ✅ Machine Learning
- XGBoost classifier
- Feature engineering
- Evaluation with Accuracy & ROC-AUC

### ✅ MLOps & Experiment Tracking
- **MLflow (Docker)** for API retraining and experiments
- Model parameters, metrics, and artifacts tracking

### ✅ APIs
- **FastAPI prediction endpoint**
- **/train-all** → full pipeline (load → prepare → train → evaluate)
- **/retrain** → hyperparameter retraining

### ✅ Monitoring & Observability
- **Elasticsearch** for centralized logs and metrics
- **Kibana dashboards** for visualization
- Monitoring of:
  - Model metrics (accuracy, ROC-AUC)
  - API events (train, predict, retrain)
  - **System metrics: CPU, memory, disk usage**

### ✅ UI / UX
- Professional **Streamlit dashboard**
- Churn probability visualization
- API interaction

### ✅ DevOps
- Fully Dockerized stack
- Jenkins CI/CD pipeline
- Reproducible environment

---

## 📊 Monitoring Stack (ELK)

| Component | URL |
|--------|-----|
| Elasticsearch | http://localhost:9200 |
| Kibana | http://localhost:5601 |

Example monitored metrics:
- `cpu_percent`
- `memory_percent`
- `disk_percent`
- `metrics.accuracy`
- `metrics.roc_auc`

---

## 🐳 Docker Stack

```bash
make docker-build
make docker-up
make docker-down
make docker-logs
```

---

## 🎨 Streamlit Dashboard

👉 **http://localhost:8501**

---

## 📈 MLflow UI

👉 **http://localhost:5000**

---

## 👤 Author

**Natej Ghodbane**  
Engineering Student – MLOps & Data Science  

---

## 📜 License

Educational & academic use.
