import mlflow
import mlflow.pytorch
import torch
import os
from sklearn.preprocessing import LabelEncoder

# 1. Configuration locale
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("CITP_Classification_Project")

MODEL_PATH = "models/citp_classifier_model.pth"

def upload_existing_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Erreur : Le fichier {MODEL_PATH} est introuvable.")
        return

    print("📦 Lecture du modèle téléchargé...")
    # On charge le dictionnaire sauvegardé par GitHub
    checkpoint = torch.load(MODEL_PATH)
    
    # On recrée l'objet modèle pour que MLflow puisse le "comprendre"
    # Note : Assure-toi que la classe CITPClassifier est bien définie ou importée ici
    from train import CITPClassifier 
    
    model = CITPClassifier(input_dim=300, num_classes=checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])

    with mlflow.start_run(run_name="Import_GitHub_Artifact"):
        # Log des paramètres
        mlflow.log_param("num_classes", checkpoint['num_classes'])
        mlflow.log_param("source", "github_actions_artifact")
        
        # LOG DU MODÈLE (C'est cette ligne qui crée l'artifact)
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="Job_Classifier_CITP"
        )
        
        print("✅ Succès ! Le modèle est maintenant dans MLflow avec ses Artifacts.")
        print("🔗 Rafraîchis http://localhost:5000 et regarde le run 'Import_GitHub_Artifact'")

if __name__ == "__main__":
    upload_existing_model()