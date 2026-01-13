FROM python:3.10-slim

WORKDIR /app

# 1. Installation des dépendances système (nécessaires pour compiler FastText)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Copie et installation des bibliothèques Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copie des modèles (poids du réseau et vecteurs de mots)
COPY models/citp_classifier_model.pth ./models/
COPY models_fasttext/cc.fr.300.bin ./models_fasttext/

# 4. Copie de ton script (On prend ton fichier et on le nomme predict.py dans le Docker)
COPY scripts/predict_jobs_mlflow.py ./predict.py

# 5. Commande de lancement
# Note : Ton script a besoin que le serveur MLflow soit accessible. 
# Si tu lances le docker localement, il faudra utiliser l'IP de ton PC.
CMD ["python", "predict.py"]