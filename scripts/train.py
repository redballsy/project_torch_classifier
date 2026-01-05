# train.py - Version avec correction Oversampling
import os
import pandas as pd
import torch
import torch.nn as nn
import fasttext
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# ============================================
# Configuration des chemins locaux
# ============================================
FASTTEXT_MODEL_PATH = r"C:\Users\Sy Savane Idriss\project_torch_classifier\modelsfastext\cc.fr.300.bin"
BASE_DIR = r"C:\Users\Sy Savane Idriss\project_torch_classifier"
DATA_DIR = os.path.join(BASE_DIR, "torchTestClassifiers", "data", "entrainer")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "entrainer2_propre.xlsx")
SAVED_MODEL_PATH = os.path.join(BASE_DIR, "models", "citp_classifier_model.pth")

os.makedirs(os.path.dirname(SAVED_MODEL_PATH), exist_ok=True)

# ============================================
# 1. Classe Dataset
# ============================================

class CITPDataset(Dataset):
    def __init__(self, dataframe, ft_model, label_encoder):
        self.embeddings = []
        self.labels = label_encoder.transform(dataframe['code'].astype(str))
        
        print(f"Calcul des vecteurs pour {len(dataframe)} lignes...")
        for text in dataframe['nomenclature']:
            vector = ft_model.get_sentence_vector(str(text).lower().strip())
            self.embeddings.append(vector)

    def __len__(self):
        return len(self.embeddings)
    
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
# 3. Script d'entraînement avec OVERSAMPLING
# ============================================

def train_main():
    print(f"🧠 Chargement du modèle FastText local...")
    if not os.path.exists(FASTTEXT_MODEL_PATH):
        print(f"❌ Erreur : Fichier introuvable à {FASTTEXT_MODEL_PATH}")
        return
    ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
    
    print(f"📖 Chargement des données Excel...")
    df = pd.read_excel(TRAIN_DATA_PATH)
    
    # --- ÉTAPE OVERSAMPLING POUR CLASSES UNIQUES ---
    # On compte les répétitions de chaque code
    counts = df['code'].value_counts()
    # On identifie les codes qui n'apparaissent qu'une seule fois
    rare_classes = counts[counts < 2].index
    
    if len(rare_classes) > 0:
        print(f"⚠️ {len(rare_classes)} classes rares détectées. Application de l'oversampling...")
        # On duplique ces lignes une fois pour qu'elles aient au moins 2 membres
        df_rare = df[df['code'].isin(rare_classes)]
        df = pd.concat([df, df_rare], ignore_index=True)
    
    # Encodage des labels avant le split pour assurer la cohérence
    le = LabelEncoder()
    df['code_str'] = df['code'].astype(str)
    le.fit(df['code_str'])
    
    # Séparation avec stratification (maintenant possible grâce à l'oversampling)
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['code_str']
    )
    
    # Préparation des DataLoaders
    train_dataset = CITPDataset(train_df, ft_model, le)
    val_dataset = CITPDataset(val_df, ft_model, le)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Configuration du modèle
    num_classes = len(le.classes_)
    model = CITPClassifier(300, num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"🚀 Entraînement sur {len(train_df)} exemples...")
    
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
            
        if (epoch + 1) % 10 == 0:
            print(f"Époque {epoch+1}/50 | Loss: {total_loss/len(train_loader):.4f}")

    # Sauvegarde
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': le,
        'input_dim': 300,
        'num_classes': num_classes
    }, SAVED_MODEL_PATH)
    
    print(f"✅ Succès ! Modèle sauvegardé dans : {SAVED_MODEL_PATH}")

if __name__ == "__main__":
    train_main()