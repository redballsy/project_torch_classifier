import os
import pandas as pd
import torch
import fasttext
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "models:/Job_Classifier_CITP@champion")

# ============================================
# CHEMINS
# ============================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

FASTTEXT_MODEL_PATH = os.path.join(BASE_DIR, "models_fasttext", "cc.fr.300.bin")
SAVED_MODEL_PATH = os.path.join(BASE_DIR, "models", "citp_classifier_model.pth")

DATA_DIR = os.path.join(BASE_DIR, "torchTestClassifiers", "data")
INPUT_FILE = os.path.join(DATA_DIR, "entrainer", "propre.xlsx")
TRUE_CODES_FILE = os.path.join(DATA_DIR, "entrainer", "codeuniq.xlsx")
RESULT_DIR = os.path.join(DATA_DIR, "resultat")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = os.path.join(RESULT_DIR, f"propre_predit_{timestamp}.xlsx")
OUTPUT_FILE_SIMPLE = os.path.join(RESULT_DIR, "propre_predit.xlsx")

os.makedirs(RESULT_DIR, exist_ok=True)

def run_prediction():
    print("=" * 60)
    print("🚀 PRÉDICTION AVEC ÉVALUATION")
    print("=" * 60)
    
    # Vérification fichiers
    if not os.path.exists(INPUT_FILE):
        print(f"❌ {INPUT_FILE} introuvable")
        return
    
    if not os.path.exists(TRUE_CODES_FILE):
        print(f"❌ {TRUE_CODES_FILE} introuvable")
        return

    # Chargement modèle
    try:
        print("⏳ Chargement modèle MLflow...")
        model = mlflow.pytorch.load_model(MODEL_URI)
        model.eval()
        
        checkpoint = torch.load(SAVED_MODEL_PATH, weights_only=False)
        le = checkpoint['label_encoder']
        print("✅ Modèle chargé")
        
    except Exception as e:
        print(f"❌ Erreur modèle: {e}")
        return

    # FastText
    try:
        ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
        print("✅ FastText chargé")
    except Exception as e:
        print(f"❌ Erreur FastText: {e}")
        return

    # Lecture données
    print("\n📖 LECTURE DONNÉES...")
    try:
        df_input = pd.read_excel(INPUT_FILE)
        df_input.columns = [str(c).strip().lower() for c in df_input.columns]
        
        if 'nomenclature' not in df_input.columns:
            print("❌ 'nomenclature' manquante")
            return
        
        df_true = pd.read_excel(TRUE_CODES_FILE)
        df_true.columns = [str(c).strip().lower() for c in df_true.columns]
        
        if 'code_citp' not in df_true.columns:
            print("❌ 'code_citp' manquante")
            return
        
        # Ajustement taille
        if len(df_input) != len(df_true):
            print(f"⚠️  Ajustement taille: {len(df_input)} vs {len(df_true)}")
            min_rows = min(len(df_input), len(df_true))
            df_input = df_input.head(min_rows).copy()
            df_true = df_true.head(min_rows).copy()
            
        print(f"✅ {len(df_input)} lignes à traiter")
        
    except Exception as e:
        print(f"❌ Erreur lecture: {e}")
        return

    # PRÉDICTIONS
    print(f"\n🚀 PRÉDICTIONS...")
    
    codes_predits = []
    confiances = []
    true_codes_list = []
    true_codes_binary_list = []
    true_results_fasletrue = []
    true_results_binary = []

    try:
        with torch.no_grad():
            for idx, text in enumerate(tqdm(df_input['nomenclature'], desc="Prédictions")):
                # Prédiction
                text_str = str(text).strip() if pd.notna(text) else ""
                vec = torch.FloatTensor(ft_model.get_sentence_vector(text_str.lower())).unsqueeze(0)
                output = model(vec)
                probs = torch.softmax(output, dim=1)
                conf, pred_idx = torch.max(probs, 1)
                
                code_predis = le.inverse_transform([pred_idx.item()])[0]
                codes_predits.append(code_predis)
                confiances.append(f"{conf.item()*100:.2f}%")
                
                # Vrai code
                vrai_code = "#N/A"
                if idx < len(df_true):
                    code_val = df_true.iloc[idx]['code_citp']
                    if pd.notna(code_val):
                        vrai_code = str(code_val).strip()
                true_codes_list.append(vrai_code)
                
                # Format 0000
                if vrai_code != "#N/A" and vrai_code:
                    try:
                        code_num = int(float(vrai_code))
                        true_code_binary = f"{code_num:04d}"
                    except:
                        true_code_binary = str(vrai_code).zfill(4)
                else:
                    true_code_binary = "#N/A"
                true_codes_binary_list.append(true_code_binary)
                
                # TRUE/FALSE
                if vrai_code != "#N/A" and vrai_code and true_code_binary != "#N/A":
                    try:
                        code_pred_num = int(float(str(code_predis)))
                        code_pred_binary = f"{code_pred_num:04d}"
                    except:
                        code_pred_binary = str(code_predis).zfill(4)
                    
                    is_correct = (true_code_binary == code_pred_binary)
                    true_results_fasletrue.append(is_correct)
                else:
                    true_results_fasletrue.append("#N/A")
                
                # 1/0
                if vrai_code != "#N/A" and vrai_code and true_code_binary != "#N/A":
                    if true_results_fasletrue[-1] == True:
                        true_results_binary.append(1)
                    elif true_results_fasletrue[-1] == False:
                        true_results_binary.append(0)
                    else:
                        true_results_binary.append("#N/A")
                else:
                    true_results_binary.append("#N/A")
        
        print("✅ Prédictions terminées")
        
    except Exception as e:
        print(f"❌ Erreur prédictions: {e}")
        return

    # CRÉATION DATAFRAME
    print(f"\n📊 ASSEMBLAGE RÉSULTATS...")
    
    df_output = pd.DataFrame({
        'nomenclature': df_input['nomenclature'],
        'code_citp_predit': codes_predits,
        'confiance': confiances,
        'True_code_citp_predit_fasletrue': true_codes_list,
        'True code_citp_predit_binary': true_codes_binary_list,
        'True_result_citp_predit_fasletrue': true_results_fasletrue,
        'True_result_citp_predit_binary': true_results_binary
    })
    
    # CALCUL PRÉCISION
    print(f"\n📈 CALCUL PRÉCISION...")
    
    valid_results = [r for r in true_results_binary if r in [0, 1]]
    
    if valid_results:
        nb_correct = sum(valid_results)
        nb_total_valid = len(valid_results)
        accuracy = (nb_correct / nb_total_valid) * 100
        
        print(f"   ✓ Correctes: {nb_correct}/{nb_total_valid}")
        print(f"   📊 Précision: {accuracy:.2f}%")
        
        accuracy_text = f"{accuracy:.2f}%"
        accuracy_value = accuracy
    else:
        accuracy_text = "0.00%"
        accuracy_value = 0
        print("   ⚠️  Pas d'évaluation")
    
    # SAUVEGARDE EXCEL - AVEC COLONNE H CORRECTE
    print(f"\n💾 SAUVEGARDE EXCEL...")
    
    try:
        # Créer un DataFrame pour l'export
        # On ajoute d'abord toutes les colonnes sauf H
        export_df = df_output.copy()
        
        # Méthode SÛRE pour avoir H2 avec valeur et autres vides
        # On va créer la colonne avec None (vraiment vide dans Excel)
        overall_col = [None] * len(export_df)  # None = cellule vraiment vide
        overall_col[0] = accuracy_text  # H2 seulement
        
        export_df['Overall_percentage_of_true'] = overall_col
        
        # Sauvegarde fichier avec timestamp
        export_df.to_excel(OUTPUT_FILE, index=False)
        print(f"✅ Fichier principal: {os.path.basename(OUTPUT_FILE)}")
        
        # Essayer fichier simple
        try:
            if os.path.exists(OUTPUT_FILE_SIMPLE):
                try:
                    os.remove(OUTPUT_FILE_SIMPLE)
                except:
                    pass  # Ignorer si on ne peut pas supprimer
            
            export_df.to_excel(OUTPUT_FILE_SIMPLE, index=False)
            print(f"✅ Fichier simple: {os.path.basename(OUTPUT_FILE_SIMPLE)}")
        except PermissionError:
            print(f"⚠️  {os.path.basename(OUTPUT_FILE_SIMPLE)} - fichier probablement ouvert dans Excel")
        except Exception as e:
            print(f"⚠️  Erreur fichier simple: {e}")
            
    except Exception as e:
        print(f"❌ Erreur sauvegarde: {e}")
        return
    
    # STATISTIQUES
    print("\n" + "=" * 70)
    print("📊 STATISTIQUES")
    print("=" * 70)
    
    nb_total = len(df_output)
    nb_true = len([r for r in true_results_fasletrue if r == True])
    nb_false = len([r for r in true_results_fasletrue if r == False])
    nb_na = len([r for r in true_results_fasletrue if r == "#N/A"])
    nb_correct_1 = sum([r for r in true_results_binary if r == 1])
    nb_incorrect_0 = sum([r for r in true_results_binary if r == 0])
    
    print(f"Total lignes:              {nb_total}")
    print(f"TRUE (correctes):          {nb_true}")
    print(f"FALSE (incorrectes):       {nb_false}")
    print(f"#N/A (non évaluées):       {nb_na}")
    print(f"")
    print(f"Colonne G (1/0):")
    print(f"  1 (bonnes):              {nb_correct_1}")
    print(f"  0 (mauvaises):           {nb_incorrect_0}")
    print(f"")
    print(f"PRÉCISION (H2):            {accuracy_text}")
    print("=" * 70)
    
    # MLFLOW
    print(f"\n📤 ENVOI MLFLOW...")
    try:
        with mlflow.start_run(run_name=f"Prediction_{timestamp}"):
            mlflow.log_artifact(OUTPUT_FILE)
            
            mlflow.log_metric("accuracy", accuracy_value)
            mlflow.log_metric("total_rows", nb_total)
            mlflow.log_metric("correct", nb_true)
            mlflow.log_metric("incorrect", nb_false)
            mlflow.log_metric("not_evaluated", nb_na)
            mlflow.log_metric("binary_correct", nb_correct_1)
            mlflow.log_metric("binary_incorrect", nb_incorrect_0)
            
            mlflow.log_param("timestamp", timestamp)
            mlflow.log_param("input_rows", nb_total)
            mlflow.log_param("evaluated_rows", nb_true + nb_false)
            
            print(f"✅ MLflow: Fichier et métriques envoyés")
            print(f"🌐 Accès: {MLFLOW_URI}")
            
    except Exception as e:
        print(f"⚠️  Erreur MLflow: {e}")
        print("Les résultats locaux sont disponibles")
    
    print("\n" + "=" * 60)
    print("🎉 TERMINÉ !")
    print("=" * 60)
    print(f"📁 Fichier: {OUTPUT_FILE}")
    print(f"📊 Précision: {accuracy_text}")
    print(f"📝 Colonne H: H2={accuracy_text}, H3-H{len(export_df)+1}=vides")
    print("=" * 60)

if __name__ == "__main__":
    run_prediction()