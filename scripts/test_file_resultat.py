# scripts/test_file_output.py
import os
import pandas as pd
import torch
import torch.nn as nn
import fasttext
from tqdm import tqdm

# ============================================
# CONFIGURATION DES CHEMINS (CORRIGÉE)
# ============================================
# On utilise r"" pour éviter l'erreur 'unicodeescape'
import os

# 1. Trouve le dossier où est le script (scripts/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. Remonte à la racine du projet (project_torch_classifier/)
BASE_DIR = os.path.dirname(CURRENT_DIR)

# 2. Configuration des chemins RELATIFS
# On part de la racine du projet pour descendre dans les dossiers
FASTTEXT_MODEL_PATH = os.path.join(BASE_DIR, "models_fasttext", "cc.fr.300.bin")
SAVED_MODEL_PATH = os.path.join(BASE_DIR, "models", "citp_classifier_model.pth")

# Dossier de données
DATA_DIR = os.path.join(BASE_DIR, "torchTestClassifiers", "data")

# Chemin du fichier Excel (Entrée)
INPUT_FILE = os.path.join(DATA_DIR, "entrainer", "propre.xlsx")

# Chemin du fichier résultat (Sortie)
# On s'assure que le dossier 'resultat' existe
RESULT_DIR = os.path.join(DATA_DIR, "resultat")
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

OUTPUT_FILE = os.path.join(RESULT_DIR, "propre_predit.xlsx")
# ============================================
# ARCHITECTURE DU MODÈLE
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

def run_prediction():
    print("⏳ Chargement du modèle...")
    
    # 1. Chargement sécurisé
    checkpoint = torch.load(SAVED_MODEL_PATH, weights_only=False)
    le = checkpoint['label_encoder']
    num_classes = checkpoint['num_classes']
    
    ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
    
    model = CITPClassifier(300, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Lecture du fichier
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Erreur : Le fichier est introuvable ici : {INPUT_FILE}")
        return

    df = pd.read_excel(INPUT_FILE)
    
    # On vérifie si la colonne existe (en minuscule pour être sûr)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'nomenclature' not in df.columns:
        print(f"❌ Erreur : Colonne 'nomenclature' absente. Colonnes trouvées : {list(df.columns)}")
        return

    # 3. Prédictions
    print(f"🚀 Prédiction de {len(df)} lignes...")
    codes_predis = []
    confiances = []

    with torch.no_grad():
        for text in tqdm(df['nomenclature']):
            vec = torch.FloatTensor(ft_model.get_sentence_vector(str(text).lower())).unsqueeze(0)
            output = model(vec)
            probs = torch.softmax(output, dim=1)
            conf, idx = torch.max(probs, 1)
            
            codes_predis.append(le.inverse_transform([idx.item()])[0])
            confiances.append(f"{conf.item()*100:.2f}%")

    # 4. Sauvegarde
    df['code_citp_predit'] = codes_predis
    df['confiance'] = confiances
    
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"✅ Terminé ! Fichier créé : {OUTPUT_FILE}")

if __name__ == "__main__":
    run_prediction()