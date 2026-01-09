# Utilisation de Python 3.9 pour correspondre à ton environnement local
FROM python:3.9-slim

WORKDIR /app

# 1. Installation des dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Préparation du fichier requirements
COPY requirements.txt .

# Commande pour désactiver la ligne GitHub qui pose problème (portfolio)
RUN sed -i 's/^git+https:\/\/github.com/# &/' requirements.txt

# 3. Installation des bibliothèques standards
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Création de la structure des dossiers
RUN mkdir -p models models_fasttext torchTestClassifiers/data/resultat

# 5. Copie de l'intégralité de ton projet dans le conteneur
# (Le .dockerignore filtrera ce qui est inutile comme l'env virtuel)
COPY . .

# 6. Installation de ton code comme un module Python
# Si tu n'as pas de setup.py, cette ligne n'échouera pas car on installe le dossier courant
RUN pip install -e . || echo "Installation locale ignorée (pas de setup.py)"

# 7. Copie finale du script de prédiction au bon endroit
COPY scripts/predict_jobs_mlflow.py ./predict.py

# 8. Configuration de la connexion vers ton MLflow Windows
ENV MLFLOW_TRACKING_URI="http://host.docker.internal:5000"

# 9. Commande de lancement
CMD ["python", "predict.py"]