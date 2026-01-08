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

# ============================================
# 1. Classe Dataset
# ============================================
class CITPDataset(Dataset):
    def __init__(self, dataframe, ft_model, label_encoder):
        self.embeddings = []
        labels_str = dataframe['code'].astype(str).tolist()
        self.labels = label_encoder.transform(labels_str)
        
        print(f"Vectorisation de {len(dataframe)} lignes...")
        for text in dataframe['nomenclature']:
            clean_text = str(text).lower().strip().replace("\n", " ")
            vector = ft_model.get_sentence_vector(clean_text)
            self.embeddings.append(vector)

    def __len__(self): return len(self.embeddings)
    
    def __getitem__(self, idx):
        return {
            'embedding': torch.FloatTensor(self.embeddings[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ============================================
# 2. Architecture du Classifieur
# ============================================
class CITPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CITPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

# ============================================
# 3. Fonction d'entraînement
# ============================================
def train_main():
    with mlflow.start_run(run_name="Training_CITP_Torch"):
        print(f"🧠 Chargement du modèle FastText...")
        ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
        
        df = pd.read_excel(TRAIN_DATA_PATH).dropna(subset=['code', 'nomenclature'])
        df['code_str'] = df['code'].astype(str)

        # --- AJOUT DU FILTRAGE POUR LA STRATIFICATION ---
        counts = df['code_str'].value_counts()
        valid_classes = counts[counts >= 2].index
        
        if len(valid_classes) < len(counts):
            print(f"⚠️ Suppression de {len(counts) - len(valid_classes)} classes orphelines (n=1)")
            df = df[df['code_str'].isin(valid_classes)].reset_index(drop=True)

        le = LabelEncoder()
        le.fit(df['code_str'])
        
        # Split stratifié sécurisé
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