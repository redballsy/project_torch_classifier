import torch
import pandas as pd
import os
from torchTestClassifiers.models.model import CITPClassifier
from torchTestClassifiers.data.datamodule import CITPDataModule

# --- CONFIGURATION DES CHEMINS ---
PATH_MODEL_FT = r"C:\Users\Sy Savane Idriss\project_torch_classifier\models\fine_tuned_model.ckpt"

# ATTENTION : On utilise le fichier CNPS pour le vocabulaire car c'est celui utilisé pour le Fine-Tuning
PATH_TRAIN_DATA = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\JOB1_Codifications_Professions CNPS_vf.xlsx"
# On garde le fichier ISCO uniquement pour traduire l'index de sortie en Code (ex: 1 -> "2141")
PATH_ISCO_REF = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\pretrain\CITP_08.xlsx"

PATH_NEW_FILE = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\tester_Codifications_Professions CNPS_vf.xlsx"
OUTPUT_FILE = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\nouveau_vf.xlsx"

def fill_empty_codes():
    print("⏳ Étape 1 : Chargement des dictionnaires...")
    # 1. Vocabulaire du modèle fine-tuné (Pour comprendre les mots en entrée)
    dm_train = CITPDataModule(data_path=PATH_TRAIN_DATA)
    dm_train.setup()
    
    # 2. Map des labels ISCO (Pour traduire la sortie en code 4 chiffres)
    dm_ref = CITPDataModule(data_path=PATH_ISCO_REF)
    dm_ref.setup()
    inv_label_map = {v: k for k, v in dm_ref.label_map.items()}

    print("🧠 Étape 2 : Chargement du modèle fine-tuné...")
    model = CITPClassifier.load_from_checkpoint(PATH_MODEL_FT, map_location="cpu")
    model.eval()
    model.freeze()

    print(f"📁 Étape 3 : Lecture du fichier cible...")
    df = pd.read_excel(PATH_NEW_FILE)
    col_source = 'nomenclature' if 'nomenclature' in df.columns else 'libelle'

    print(f"🚀 Prédiction en cours...")
    codes_predits = []
    confiances = []

    for text in df[col_source]:
        tokens = str(text).lower().split()
        
        # UTILISATION DU VOCABULAIRE CNPS (dm_train)
        input_ids = [dm_train.vocab.get(word, 1) for word in tokens]
        
        # Padding standard
        if len(input_ids) < 10:
            input_ids += [0] * (10 - len(input_ids))
        else:
            input_ids = input_ids[:10]

        input_tensor = torch.tensor([input_ids])
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            
        # Traduction de l'index de sortie en Code ISCO via la map de référence
        final_code = inv_label_map.get(pred_idx.item(), "INCONNU")
        
        codes_predits.append(final_code)
        confiances.append(f"{conf.item():.2%}")

    df['code'] = codes_predits
    df['score_confiance'] = confiances

    df.to_excel(OUTPUT_FILE, index=False)
    print(f"\n✅ TERMINÉ ! Fichier généré : {OUTPUT_FILE}")

if __name__ == "__main__":
    fill_empty_codes()