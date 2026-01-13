import os
import pandas as pd
import torch
import torch.nn as nn
import fasttext
import pickle
import mlflow
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# ============================================
# 1. CONFIGURATION DYNAMIQUE
# ============================================
# Détection automatique du dossier racine (project_torch_classifier)
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "citp_classifier_model.pth"
FASTTEXT_PATH = BASE_DIR / "models_fasttext" / "cc.fr.300.bin"
INPUT_FILE = BASE_DIR / "torchTestClassifiers" / "data" / "entrainer" / "topredict_nettoye.xlsx"
CODE_UNIQ_FILE = BASE_DIR / "torchTestClassifiers" / "data" / "entrainer" / "codeuniq.xlsx"
RESULT_DIR = BASE_DIR / "torchTestClassifiers" / "data" / "resultat"

# Création du dossier de résultats
os.makedirs(RESULT_DIR, exist_ok=True)

# Configuration MLflow (compatible Linux/Windows)
mlflow_path = str(BASE_DIR).replace('\\', '/')
mlflow.set_tracking_uri(f"file:///{mlflow_path}/mlruns")
mlflow.set_experiment("CITP_Inference_Logs")

# ============================================
# 2. ARCHITECTURE
# ============================================
class CITPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.network(x)

# ============================================
# 3. LOGIQUE DE PRÉDICTION
# ============================================
def run_prediction():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Erreur : Modèle introuvable -> {MODEL_PATH}")
        return

    with mlflow.start_run(run_name=f"Inference_{datetime.now().strftime('%H%M')}"):
        
        print("🚀 Chargement des ressources...")
        checkpoint = torch.load(str(MODEL_PATH), map_location='cpu', weights_only=False)
        model = CITPClassifier(checkpoint['input_dim'], checkpoint['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        le = checkpoint['label_encoder']

        ft = fasttext.load_model(str(FASTTEXT_PATH))
        
        # Chargement des fichiers
        df = pd.read_excel(INPUT_FILE)
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        df_codes = pd.read_excel(CODE_UNIQ_FILE)
        df_codes.columns = [str(c).strip().lower() for c in df_codes.columns]

        col_jointure_droite = 'code_citp' if 'code_citp' in df_codes.columns else df_codes.columns[0]

        results = []
        print(f"🔮 Prédiction sur {len(df)} lignes...")
        
        with torch.no_grad():
            for _, row in tqdm(df.iterrows(), total=len(df)):
                texte = str(row['nomenclature']).lower().strip()
                vec = torch.tensor(ft.get_sentence_vector(texte)).unsqueeze(0)
                
                outputs = model(vec)
                probs = torch.softmax(outputs, dim=1)
                conf, idx = torch.max(probs, dim=1)
                
                metier_predit = le.inverse_transform([idx.item()])[0]
                results.append({
                    'nomenclature': row['nomenclature'],
                    'code_citp_predit': str(metier_predit),
                    'confiance_pourcent': round(conf.item() * 100, 2)
                })

        df_res = pd.DataFrame(results)

        # Force le type string pour éviter l'erreur de merge (Object vs Int64)
        df_res['code_citp_predit'] = df_res['code_citp_predit'].astype(str)
        df_codes[col_jointure_droite] = df_codes[col_jointure_droite].astype(str)

        # Fusion et calcul moyenne
        df_res = pd.merge(df_res, df_codes, left_on='code_citp_predit', right_on=col_jointure_droite, how='left')
        avg_conf = df_res['confiance_pourcent'].mean()
        df_res['pourcentage_global_moyen'] = round(avg_conf, 2)
        
        mlflow.log_metric("avg_confidence", avg_conf)

        # Sauvegarde
        timestamp = datetime.now().strftime("%H%M%S")
        output_filename = f"resultats_complets_{timestamp}.xlsx"
        output_path = RESULT_DIR / output_filename
        df_res.to_excel(output_path, index=False)

        mlflow.log_artifact(str(output_path))
        
        print(f"\n✅ Terminé !")
        print(f"📊 Confiance moyenne : {avg_conf:.2f}%")
        print(f"📁 Fichier généré : {output_path}")

if __name__ == "__main__":
    run_prediction()