# ================================
# VARIABLES
# ================================
PYTHON=python3
ENV_NAME=venv
REQUIREMENTS=requirements.txt
FILES = src/ml/main.py src/ml/model_pipeline.py src/api/api.py

DOCKER_IMAGE_API = noticc/churn-api
DOCKER_IMAGE_UI  = noticc/churn-ui

MAIN_SCRIPT = src/ml/main.py
STREAMLIT_SCRIPT = src/ui/streamlit_app.py
API_SCRIPT = src/api/api.py

# ================================
# 1. SETUP ENVIRONMENT
# ================================
setup:
	@echo "Création de l'environnement virtuel et installation des dépendances..."
	@if [ ! -d "$(ENV_NAME)" ]; then \
		$(PYTHON) -m venv $(ENV_NAME) || (echo "Installe python3-venv : sudo apt install python3-venv" && exit 1); \
	else \
		echo "Environnement déjà existant ($(ENV_NAME))"; \
	fi
	@echo "Installation des dépendances..."
	@. $(ENV_NAME)/bin/activate && pip install --upgrade pip && pip install -r $(REQUIREMENTS)

# ================================
# 2. CODE QUALITY (CI)
# ================================
lint:
	@echo "Running flake8 on project files..."
	@. $(ENV_NAME)/bin/activate && flake8 $(FILES)

format:
	@echo "Running black on project files..."
	@. $(ENV_NAME)/bin/activate && black $(FILES)

security:
	@echo "Running bandit security analysis..."
	@. $(ENV_NAME)/bin/activate && bandit -r $(FILES)

# ================================
# 3. LOAD RAW CSV FILES
# ================================
load:
	@echo "Chargement des CSV bruts..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) $(MAIN_SCRIPT) load

# ================================
# 4. DATA PREPARATION
# ================================
prepare:
	@echo "Préparation des données..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) $(MAIN_SCRIPT) prepare

# ================================
# 5. TRAINING
# ================================
train:
	@echo "Entraînement du modèle..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) $(MAIN_SCRIPT) train

# ================================
# 6. EVALUATION
# ================================
evaluate:
	@echo "Évaluation du modèle..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) $(MAIN_SCRIPT) evaluate

# ================================
# 7. TESTS
# ================================
test:
	@echo "Exécution des tests unitaires..."
	@. $(ENV_NAME)/bin/activate && pytest -q || true

# ================================
# 8. LOCAL DEVELOPMENT SERVERS
# ================================
api:
	@echo "Lancement du serveur FastAPI..."
	@. $(ENV_NAME)/bin/activate && uvicorn src.api.api:app --reload --host 0.0.0.0 --port 8000

streamlit:
	@echo "Lancement de Streamlit..."
	@. $(ENV_NAME)/bin/activate && streamlit run $(STREAMLIT_SCRIPT)

# ================================
# 9. NOTEBOOK SERVER
# ================================
.PHONY: notebook
notebook:
	@echo "Démarrage de Jupyter Notebook..."
	@. $(ENV_NAME)/bin/activate && jupyter notebook

# ================================
# 10. FULL PIPELINE
# ================================
all:
	@echo "Pipeline complet (load + prepare + train + evaluate)..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) $(MAIN_SCRIPT) all

# ================================
# 11. CLEAN
# ================================
clean:
	@echo "Nettoyage des fichiers temporaires..."
	@rm -f *.pkl
	@rm -rf __pycache__
	@find . -type d -name "__pycache__" -exec rm -rf {} +

# ================================
# 12. DOCKER — MULTI-CONTAINERS
# ================================
docker-build-api:
	docker build -t $(DOCKER_IMAGE_API):latest -f docker/Dockerfile.api .

docker-build-ui:
	docker build -t $(DOCKER_IMAGE_UI):latest -f docker/Dockerfile.streamlit .

docker-build:
	make docker-build-api
	make docker-build-ui

docker-push:
	docker push $(DOCKER_IMAGE_API):latest
	docker push $(DOCKER_IMAGE_UI):latest

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

docker-logs:
	docker compose -f docker/docker-compose.yml logs -f

docker-redeploy:
	@echo "Redeploying Docker infrastructure..."
	-docker rm -f churn_api churn_ui 2>/dev/null || true
	docker compose -f docker/docker-compose.yml down
	docker compose -f docker/docker-compose.yml up -d --build
