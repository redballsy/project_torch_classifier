import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import fasttext
import numpy as np
import mlflow
import mlflow.pytorch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

# ============================================
# BRIDGE DE COMPATIBILITÉ NUMPY
# ============================================
if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np
    try:
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
    except AttributeError:
        pass

warnings.filterwarnings('ignore')

# CONFIGURATION MLFLOW DYNAMIQUE
# On vérifie si on est sur GitHub Actions
IS_GITHUB = os.getenv('GITHUB_ACTIONS') == 'true'

if not IS_GITHUB:
    mlflow.set_tracking_uri("http://localhost:5000")
    print("🏠 Mode Local : Tracking vers MLflow Localhost")
else:
    print("🤖 Mode GitHub : Sauvegarde locale uniquement (Artifacts)")

mlflow.set_experiment("CITP_Classification_Project")

# ============================================
# Configuration Dynamique des Chemins
# ============================================
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_SCRIPT_DIR)

FASTTEXT_MODEL_PATH = os.path.join(BASE_DIR, "models_fasttext", "cc.fr.300.bin")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "torchTestClassifiers", "data", "entrainer", "entrainer2_propre.xlsx")
SAVED_MODEL_PATH = os.path.join(BASE_DIR, "models", "citp_classifier_model.pth")

os.makedirs(os.path.dirname(SAVED_MODEL_PATH), exist_ok=True)

# [Classes CITPDataset et CITPClassifier restent identiques...]

def train_main():
    # Sur GitHub, on ne lance pas de start_run vers localhost pour éviter les erreurs de connexion
    # On utilise MLflow en mode "local" qui écrit dans un dossier ./mlruns
    with mlflow.start_run(run_name="Training_CITP_Torch"):
        print(f"🧠 Chargement du modèle FastText...")
        ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
        
        df = pd.read_excel(TRAIN_DATA_PATH).dropna(subset=['code', 'nomenclature'])
        le = LabelEncoder()
        df['code_str'] = df['code'].astype(str)
        le.fit(df['code_str'])
        
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['code_str'])
        train_loader = DataLoader(CITPDataset(train_df, ft_model, le), batch_size=32, shuffle=True)
        
        num_classes = len(le.classes_)
        model = CITPClassifier(300, num_classes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        mlflow.log_param("num_classes", num_classes)

        print(f"🚀 Début de l'entraînement...")
        for epoch in range(50):
            model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(batch['embedding'])
                loss = criterion(outputs, batch['label'])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss/len(train_loader)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            
        # --- SAUVEGARDE ---
        state = {
            'model_state_dict': model.state_dict(),
            'label_encoder': le,
            'input_dim': 300,
            'num_classes': num_classes
        }
        torch.save(state, SAVED_MODEL_PATH)

        # Sur GitHub, on log le modèle dans les artifacts de l'expérience
        # On ne tente l'enregistrement Registry (registered_model_name) QUE si on est en local
        if not IS_GITHUB:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name="Job_Classifier_CITP"
            )
        else:
            mlflow.pytorch.log_model(pytorch_model=model, artifact_path="model")
        
        print(f"✅ Entraînement terminé. Modèle dispo dans : {SAVED_MODEL_PATH}")

if __name__ == "__main__":
    train_main()