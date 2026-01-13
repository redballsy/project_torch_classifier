import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import fasttext
import pickle
import mlflow
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ============================================
# 1. CONFIGURATION ET CHEMINS DYNAMIQUES
# ============================================
# Path(__file__) trouve l'emplacement actuel du script. 
# .parent.parent remonte à la racine du projet (project_torch_classifier)
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "torchTestClassifiers" / "data" / "entrainer" / "entrainer2_propre.xlsx"
FASTTEXT_MODEL_PATH = BASE_DIR / "models_fasttext" / "cc.fr.300.bin"
MODEL_SAVE_PATH = BASE_DIR / "models" / "citp_classifier_model.pth"
LE_SAVE_PATH = BASE_DIR / "models" / "label_encoder.pkl"

# Création du dossier models s'il n'existe pas
os.makedirs(BASE_DIR / "models", exist_ok=True)

# Configuration MLflow (Correction du chemin pour GitHub Actions)
mlflow_path = str(BASE_DIR).replace('\\', '/')
mlflow.set_tracking_uri(f"file:///{mlflow_path}/mlruns")
mlflow.set_experiment(f"CITP_Train_Run")

# ============================================
# 2. ARCHITECTURE
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
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.network(x)

# ============================================
# 3. LOGIQUE D'ENTRAÎNEMENT
# ============================================
def run_train():
    print(f"📂 Dossier racine détecté : {BASE_DIR}")
    
    # Vérification de l'existence du fichier d'entraînement
    if not os.path.exists(TRAIN_FILE):
        print(f"❌ Erreur : Fichier introuvable -> {TRAIN_FILE}")
        return

    print("⏳ Chargement des ressources (FastText, Excel)...")
    ft_model = fasttext.load_model(str(FASTTEXT_MODEL_PATH))
    
    df = pd.read_excel(TRAIN_FILE)
    texts = df['nomenclature'].astype(str).tolist()
    labels = df['code'].astype(str).tolist()

    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)
    
    with open(LE_SAVE_PATH, 'wb') as f:
        pickle.dump(le, f)

    print(f"⏳ Vectorisation de {len(texts)} lignes...")
    X_vecs = [ft_model.get_sentence_vector(t.lower().strip()) for t in texts]
    X_tensor = torch.FloatTensor(X_vecs)
    y_tensor = torch.LongTensor(y_encoded)

    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

    model = CITPClassifier(300, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    with mlflow.start_run(run_name=f"Train_{datetime.now().strftime('%m%d_%H%M')}"):
        mlflow.log_param("num_samples", len(df))
        mlflow.log_param("num_classes", num_classes)
        
        for epoch in range(15):
            model.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            correct = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    preds = torch.argmax(model(bx), dim=1)
                    correct += (preds == by).sum().item()
            
            val_acc = correct / len(X_val)
            avg_loss = total_loss / len(train_loader)
            
            mlflow.log_metric("loss", avg_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            
            print(f"Epoch {epoch+1}/15 | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        state = {
            'model_state_dict': model.state_dict(),
            'input_dim': 300,
            'num_classes': num_classes,
            'label_encoder': le
        }
        torch.save(state, str(MODEL_SAVE_PATH))
        mlflow.pytorch.log_model(model, "model")
        print(f"✅ Terminé ! Modèle sauvé dans {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    run_train()