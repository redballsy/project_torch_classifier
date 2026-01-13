import os
import pandas as pd
import torch
import torch.nn as nn
import fasttext
import pickle
import mlflow
from tqdm import tqdm
from datetime import datetime

# ============================================
# 1. CONFIGURATION
# ============================================
BASE_DIR = r"C:\Users\Sy Savane Idriss\project_torch_classifier"
MODEL_PATH = os.path.join(BASE_DIR, "models", "citp_classifier_model.pth")
FASTTEXT_PATH = os.path.join(BASE_DIR, "models_fasttext", "cc.fr.300.bin")
INPUT_FILE = os.path.join(BASE_DIR, "torchTestClassifiers", "data", "entrainer", "topredict_nettoye.xlsx")
CODE_UNIQ_FILE = os.path.join(BASE_DIR, "torchTestClassifiers", "data", "entrainer", "codeuniq.xlsx")
RESULT_DIR = os.path.join(BASE_DIR, "torchTestClassifiers", "data", "resultat")
os.makedirs(RESULT_DIR, exist_ok=True)

mlflow_local_path = BASE_DIR.replace('\\', '/')
mlflow.set_tracking_uri(f"file:///{mlflow_local_path}/mlruns")
mlflow.set_experiment("CITP_Inference_Logs")

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
# 2. LOGIQUE DE PRÉDICTION
# ============================================
def run_prediction():
    with mlflow.start_run(run_name=f"Inference_{datetime.now().strftime('%H%M')}"):
        
        print("🚀 Chargement des ressources...")
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        model = CITPClassifier(checkpoint['input_dim'], checkpoint['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        le = checkpoint['label_encoder']

        ft = fasttext.load_model(FASTTEXT_PATH)
        
        # Chargement
        df = pd.read_excel(INPUT_FILE)
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        df_codes = pd.read_excel(CODE_UNIQ_FILE)
        df_codes.columns = [str(c).strip().lower() for c in df_codes.columns]

        # Détection de la colonne de jointure dans codeuniq
        # On cherche 'code_citp' ou on prend la 1ère colonne
        col_jointure_droite = 'code_citp' if 'code_citp' in df_codes.columns else df_codes.columns[0]

        results = []
        print(f"🔮 Prédiction en cours sur {len(df)} lignes...")
        
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

        # --- RÉSOLUTION DU CONFLIT DE TYPE (Object vs Int64) ---
        # On force les deux colonnes en TEXTE (string) avant le merge
        df_res['code_citp_predit'] = df_res['code_citp_predit'].astype(str)
        df_codes[col_jointure_droite] = df_codes[col_jointure_droite].astype(str)

        # Fusion (Merge)
        df_res = pd.merge(df_res, df_codes, left_on='code_citp_predit', right_on=col_jointure_droite, how='left')

        # Calcul du pourcentage global moyen
        avg_conf = df_res['confiance_pourcent'].mean()
        df_res['pourcentage_global_moyen'] = round(avg_conf, 2)
        
        mlflow.log_metric("avg_confidence", avg_conf)

        # Sauvegarde
        timestamp = datetime.now().strftime("%H%M%S")
        output_path = os.path.join(RESULT_DIR, f"resultats_complets_{timestamp}.xlsx")
        df_res.to_excel(output_path, index=False)

        mlflow.log_artifact(output_path)
        
        print(f"\n✅ Terminé !")
        print(f"📊 Confiance moyenne : {avg_conf:.2f}%")
        print(f"📁 Fichier : {output_path}")

if __name__ == "__main__":
    run_prediction()