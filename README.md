# 🚀 MLOps Churn Prediction Platform

An **end-to-end MLOps project** for **Customer Churn Prediction** combining  
**Machine Learning, MLflow, FastAPI, Streamlit, Docker, and Jenkins CI/CD**.

This project demonstrates a **production-style MLOps architecture** with:
- Offline CLI training
- Online API retraining
- Experiment tracking with MLflow
- Interactive Streamlit dashboard
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
│   ├── docker-compose.yml       # Multi-service stack
│   ├── Dockerfile.api           # FastAPI service
│   ├── Dockerfile.mlflow        # MLflow tracking server
│   └── Dockerfile.streamlit     # Streamlit UI
├── models/                      # Saved models & preprocessors
├── src/
│   ├── api/                     # FastAPI backend
│   │   └── api.py
│   ├── ml/                      # ML pipeline (no MLflow inside)
│   │   ├── main.py              # CLI orchestration
│   │   └── model_pipeline.py    # Train / evaluate logic
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
- Feature engineering + SMOTEENN balancing
- Evaluation with Accuracy & ROC-AUC

### ✅ MLOps & Experiment Tracking
- **MLflow (Docker)** for API retraining
- **Local MLflow (`mlruns/`)** for CLI experiments
- Clean separation of concerns (no MLflow in ML code)

### ✅ APIs
- **FastAPI prediction endpoint**
- **/retrain** → hyperparameter retraining
- **/train-all** → full pipeline (load → prepare → train → evaluate)

### ✅ UI / UX
- Professional **Streamlit dashboard**
- Human-readable inputs
- Churn probability visualization

### ✅ DevOps
- Fully Dockerized stack
- Jenkins CI/CD pipeline
- Linting, formatting, reproducibility

---

## ⚙️ Makefile Commands

### 🔧 Local development
```bash
make setup        # Create venv & install dependencies
make lint         # flake8 linting
make format       # black formatting
make test         # run tests
```

### 🧪 ML pipeline (CLI)
```bash
make load
make prepare
make train
make evaluate
make all           # Full pipeline
```

### 🐳 Docker stack
```bash
make docker-build
make docker-up
make docker-down
make docker-logs
```

---

## 🔌 FastAPI Backend

### Prediction
```
POST /predict
```

Example payload:
```json
{
  "Total charge": 110,
  "Customer service calls": 2,
  "Total intl calls": 3,
  "International plan": 0,
  "Number vmail messages": 5,
  "CScalls Rate": 0.01,
  "Area code_408": 1,
  "Area code_415": 0,
  "Area code_510": 0,
  "State_TX": 1,
  "State_SC": 0,
  "State_MT": 0,
  "State_IL": 0,
  "Total intl charge": 2.7
}
```

### Retraining
```
POST /retrain
POST /train-all
```

Both endpoints log experiments to **MLflow (Docker)**.

---

## 🎨 Streamlit Dashboard

Accessible at:

👉 **http://localhost:8501**

Features:
- 🔮 Churn prediction
- 📊 Probability gauge
- 🔧 Model retraining
- 🧠 API integration

---

## 📊 MLflow

| Usage | Tracking URI |
|-----|--------------|
| CLI training | `file:./mlruns` |
| API retraining | `http://mlflow:5000` |

MLflow UI:

👉 **http://localhost:5000**

---

## 🔁 Jenkins CI/CD Pipeline

Stages:
1. Checkout
2. Install dependencies
3. Lint & format
4. Run ML pipeline
5. Build Docker images
6. Deploy stack


---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

---

## 👤 Author

**Natej Ghodbane**  
Engineering Student – MLOps & Data Science  

---

## 📜 License

Educational & academic use.