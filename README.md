# 🚀 MLOps Churn Prediction Pipeline

This project implements a **full end-to-end MLOps workflow** for a churn prediction model using **FastAPI**, **Streamlit**, **XGBoost**, **Docker**, and **Jenkins CI/CD**.

---

## 📂 Project Structure

```
├── ci-cd/
│   └── Jenkinsfile
├── data/
│   ├── churn-bigml-80.csv
│   └── churn-bigml-20.csv
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.streamlit
│   └── docker-compose.yml
├── models/                
├── src/
│   ├── api/               ← FastAPI backend
│   ├── ml/                ← Training pipeline
│   └── ui/                ← Streamlit interface
├── Makefile
├── requirements.txt
└── README.md
```

---

## 🧠 Features

✔ Fully automated ML pipeline (load → prepare → train → evaluate)  
✔ Model artefact saving (model + scaler + encoders)  
✔ REST API using FastAPI
✔ User interface using Streamlit  
✔ Dockerized multi‑service deployment  
✔ Jenkins CI/CD:  
&nbsp;&nbsp;&nbsp;&nbsp;• lint, format, security checks  
&nbsp;&nbsp;&nbsp;&nbsp;• training pipeline  
&nbsp;&nbsp;&nbsp;&nbsp;• build & push Docker images  
&nbsp;&nbsp;&nbsp;&nbsp;• deploy using Docker Compose  

---

## 🛠 Makefile Commands

### **Local development**
```
make setup        # create venv + install dependencies
make lint         # run flake8
make format       # format code using black
make security     # run bandit security scan
make test         # run pytest
```

### **ML pipeline**
```
make load
make prepare
make train
make evaluate
```

### **Docker workflow**
```
make docker-build     # build API + UI images
make docker-push      # push to Docker Hub
make docker-redeploy  # restart full stack with docker compose
```

---

## ⚙️ FastAPI Backend

Endpoint example:

```
POST /predict
{
  "Total charge": 110,
  "Area code_415": 0,
  "Area code_408": 1,
  "Customer service calls": 2,
  "Area code_510": 0,
  "Total intl calls": 3,
  "International plan": 0,
  "Number vmail messages": 5,
  "State_SC": 0,
  "State_TX": 1,
  "State_MT": 0,
  "Total intl charge": 2.7,
  "State_IL": 0,
  "CScalls Rate": 0.01
}
```

The API loads:
- `churn_model.pkl`
- `churn_scaler.pkl`
- `churn_encoder_state.pkl`
- `churn_encoder_area.pkl`

---

## 🎨 Streamlit Web UI

Runs at:

👉 **http://localhost:8501**

It communicates with FastAPI internally using:

```
API_URL=http://api:8000
```

---

## 🐳 Docker Deployment

### **Build & Run manually**
```
docker compose -f docker/docker-compose.yml up --build
```

### **Services**
| Service | Port | Description |
|---------|------|-------------|
| `api`   | 8000 | FastAPI backend |
| `ui`    | 8501 | Streamlit interface |

---

## 🔁 Jenkins CI/CD Pipeline

Pipeline stages:

1. **Checkout**
2. **Setup virtualenv**
3. **Lint / Format / Security**
4. **Load → Prepare → Train → Evaluate**
5. **Unit tests**
6. **Docker build (API + UI)**
7. **Docker Hub push**
8. **Docker Compose deployment**

After each commit → Jenkins pulls → rebuilds the ML stack automatically.

---

## 📦 Requirements

Install with:

```
pip install -r requirements.txt
```

---

## 📜 License

This project is for educational purposes under the MLOps coursework.

---

## 👤 Author

**Natej Ghodbane**   

---

