import os
import pandas as pd
import torch
import fasttext
import mlflow
import mlflow.pytorch
from tqdm import tqdm

# ============================================
# CONFIGURATION MLFLOW & CHEMINS
# ============================================
mlflow.set_tracking_uri("http://localhost:5000")
# On utilise l'alias @champion défini dans l'UI MLflow 3.1.4
MODEL_URI = "models:/Job_Classifier_CITP@champion"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

FASTTEXT_MODEL_PATH = os.path.join(BASE_DIR, "models_fasttext", "cc.fr.300.bin")
DATA_DIR = os.path.join(BASE_DIR, "torchTestClassifiers", "data")
INPUT_FILE = os.path.join(DATA_DIR, "entrainer", "propre.xlsx")
RESULT_DIR = os.path.join(DATA_DIR, "resultat")
OUTPUT_FILE = os.path.join(RESULT_DIR, "propre_predit_mlflow.xlsx")

os.makedirs(RESULT_DIR, exist_ok=True)

def run_prediction():
    print(f"⏳ Connexion à MLflow et chargement du modèle @champion...")
    
    try:
        # 1. Chargement du modèle et du label encoder depuis MLflow
        # MLflow récupère l'objet complet (architecture + poids)
        model = mlflow.pytorch.load_model(MODEL_URI)
        model.eval()
        
        # On récupère le LabelEncoder que nous avions logué dans le dictionnaire de sauvegarde
        # Note : On charge le dictionnaire local pour récupérer le LabelEncoder (LE)
        # car MLflow logue l'objet NN.Module. 
        # Si tu as suivi l'étape upload_to_mlflow, le LE est dans ton .pth local.
        SAVED_MODEL_PATH = os.path.join(BASE_DIR, "models", "citp_classifier_model.pth")
        checkpoint = torch.load(SAVED_MODEL_PATH, weights_only=False)
        le = checkpoint['label_encoder']
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return

    print("📝 Chargement de FastText...")
    ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

    # 2. Lecture du fichier
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Erreur : Fichier introuvable : {INPUT_FILE}")
        return

    df = pd.read_excel(INPUT_FILE)
    df.columns = [c.strip().lower() for c in df.columns]
    
    if 'nomenclature' not in df.columns:
        print(f"❌ Erreur : Colonne 'nomenclature' absente.")
        return

    # 3. Prédictions
    print(f"🚀 Prédiction de {len(df)} lignes avec le modèle MLflow...")
    codes_predis = []
    confiances = []

    with torch.no_grad():
        for text in tqdm(df['nomenclature']):
            # Vectorisation
            vec = torch.FloatTensor(ft_model.get_sentence_vector(str(text).lower())).unsqueeze(0)
            
            # Passage dans le modèle chargé via MLflow
            output = model(vec)
            probs = torch.softmax(output, dim=1)
            conf, idx = torch.max(probs, 1)
            
            # Décodage du label
            codes_predis.append(le.inverse_transform([idx.item()])[0])
            confiances.append(f"{conf.item()*100:.2f}%")

    # 4. Sauvegarde
    df['code_citp_predit'] = codes_predis
    df['confiance_ia'] = confiances
    
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"✅ Terminé ! Résultats sauvegardés dans : {OUTPUT_FILE}")

if __name__ == "__main__":
    run_prediction()