# scripts/predict.py
import torch
import fasttext
import os
import torch.nn as nn

# 1. On trouve où se situe le script actuel dynamiquement
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. On construit le chemin à partir de là
chemin_fichier = os.path.join(BASE_DIR, "data", "entrainer", "CNPS_Code_NC.xlsx")
FASTTEXT_MODEL_PATH = os.path.join(BASE_DIR, "modelsfastext", "cc.fr.300.bin")
SAVED_MODEL_PATH = os.path.join(BASE_DIR, "models", "citp_classifier_model.pth")

# Architecture (doit être identique à celle de l'entraînement)
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

def predict():
    # 1. Charger les ressources
    checkpoint = torch.load(SAVED_MODEL_PATH)
    le = checkpoint['label_encoder']
    
    ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
    
    model = CITPClassifier(300, checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("\n--- Système de Classification CITP Prêt ---")
    
    while True:
        metier = input("\nEntrez un intitulé de métier (ou 'q' pour quitter) : ")
        if metier.lower() == 'q': break
        
        # Transformation en vecteur
        with torch.no_grad():
            vector = torch.FloatTensor(ft_model.get_sentence_vector(metier.lower())).unsqueeze(0)
            output = model(vector)
            
            # Récupération de la probabilité
            probs = torch.nn.functional.softmax(output, dim=1)
            confiance, pred_idx = torch.max(probs, 1)
            
            code_predi = le.inverse_transform([pred_idx.item()])[0]
            
            print(f"✅ Code CITP prédit : {code_predi} (Confiance : {confiance.item()*100:.2f}%)")

if __name__ == "__main__":
    predict()