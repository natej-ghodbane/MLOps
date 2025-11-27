# ================================
# VARIABLES
# ================================
PYTHON=python3
ENV_NAME=venv
REQUIREMENTS=requirements.txt
FILES = main.py model_pipeline.py

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
	@. $(ENV_NAME)/bin/activate && $(PYTHON) main.py load

# ================================
# 4. DATA PREPARATION
# ================================
prepare:
	@echo "Préparation des données..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) main.py prepare

# ================================
# 5. TRAINING
# ================================
train:
	@echo "Entraînement du modèle..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) main.py train

# ================================
# 6. EVALUATION
# ================================
evaluate:
	@echo "Évaluation du modèle..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) main.py evaluate

# ================================
# 7. TESTS
# ================================
test:
	@echo "Exécution des tests unitaires..."
	@. $(ENV_NAME)/bin/activate && pytest -q || true

# ================================
# 8. DEPLOYMENT (FASTAPI)
# ================================
FastAPI:
	@echo "Lancement du serveur FastAPI..."
	@. $(ENV_NAME)/bin/activate && uvicorn api:app --reload --host 0.0.0.0 --port 8000

streamlit:
	@echo "Lancement du serveur FastAPI..."
	@. $(ENV_NAME)/bin/activate && streamlit run streamlit_app.py
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
	@. $(ENV_NAME)/bin/activate && $(PYTHON) main.py all

# ================================
# 11. CLEAN
# ================================
clean:
	@echo "Nettoyage des fichiers temporaires..."
	@rm -f *.pkl
	@rm -rf __pycache__
