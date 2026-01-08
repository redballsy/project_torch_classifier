"""
SCRIPT D'INFÉRENCE POUR CLASSIFICATION ISCO
Utilisation: python inference.py "texte du métier"
"""

import torch
import pickle
import sys
import numpy as np

# Ajouter le chemin pour importer votre modèle
import sys
sys.path.append(r"C:\Users\Sy Savane Idriss\project_torch_classifier")

try:
    from torchTestClassifiers.models.model import CITPClassifier
    MODEL_TYPE = "CITPClassifier"
except:
    print("⚠️ CITPClassifier non trouvé, chargement générique...")
    import torch.nn as nn
    MODEL_TYPE = "Generic"

class ISCOPredictor:
    def __init__(self, model_path, mappings_path):
        """Initialiser le prédicteur"""
        print("🔧 Chargement du modèle...")
        
        # Charger les mappings
        with open(mappings_path, 'rb') as f:
            self.mappings = pickle.load(f)
        
        self.vocab = self.mappings['vocab']
        self.idx_to_label = self.mappings['idx_to_label']
        self.label_to_idx = self.mappings['label_to_idx']
        self.max_len = 30
        
        # Charger le modèle
        try:
            if MODEL_TYPE == "CITPClassifier":
                self.model = CITPClassifier.load_from_checkpoint(
                    model_path,
                    num_classes=self.mappings['num_classes'],
                    vocab_size=self.mappings['vocab_size']
                )
            else:
                # Modèle générique de secours
                class GenericModel(nn.Module):
                    def __init__(self, vocab_size, num_classes):
                        super().__init__()
                        self.embedding = nn.Embedding(vocab_size, 300, padding_idx=0)
                        self.lstm = nn.LSTM(300, 128, batch_first=True, bidirectional=True)
                        self.fc = nn.Linear(256, num_classes)
                    
                    def forward(self, x):
                        x = self.embedding(x)
                        x, _ = self.lstm(x)
                        x = x[:, -1, :]
                        return self.fc(x)
                
                self.model = GenericModel(
                    self.mappings['vocab_size'],
                    self.mappings['num_classes']
                )
                checkpoint = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(checkpoint['state_dict'])
            
            self.model.eval()
            print("✅ Modèle chargé avec succès")
            print(f"   - {self.mappings['num_classes']} codes ISCO")
            print(f"   - Vocabulaire de {len(self.vocab)} mots")
            
        except Exception as e:
            print(f"❌ Erreur de chargement: {e}")
            raise
    
    def preprocess(self, text):
        """Préparer le texte pour la prédiction"""
        text = str(text).lower().strip()
        
        # Tokenization simple
        tokens = text.split()[:self.max_len]
        token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Padding
        if len(token_ids) < self.max_len:
            token_ids = token_ids + [self.vocab['<PAD>']] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        
        return torch.tensor([token_ids], dtype=torch.long)
    
    def predict(self, text, top_k=5):
        """Faire une prédiction"""
        # Préparation
        input_tensor = self.preprocess(text)
        
        # Prédiction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(self.idx_to_label)))
        
        # Formater les résultats
        results = []
        for i in range(top_indices.shape[1]):
            idx = top_indices[0][i].item()
            code = self.idx_to_label[idx]
            prob = top_probs[0][i].item() * 100
            
            results.append({
                'code': code,
                'confidence': prob,
                'rank': i + 1
            })
        
        return results
    
    def predict_batch(self, texts, top_k=3):
        """Prédire plusieurs textes"""
        predictions = []
        for text in texts:
            result = self.predict(text, top_k)
            predictions.append({
                'text': text,
                'predictions': result
            })
        return predictions

def main():
    """Fonction principale"""
    # Chemins des fichiers
    MODEL_PATH = "models/model_cnps_clean.ckpt"
    MAPPINGS_PATH = "models/model_cnps_clean_mappings.pkl"
    
    # Vérifier les arguments
    if len(sys.argv) < 2:
        print("Usage: python inference.py "texte du métier"")
        print("Exemple: python inference.py "ingénieur en informatique"")
        print("
Ou tester avec des exemples prédéfinis:")
        test_texts = [
            "ingénieur informatique",
            "technicien de maintenance",
            "secrétaire administratif",
            "comptable général",
            "ouvrier agricole",
            "médecin généraliste",
            "enseignant primaire",
            "cuisinier restaurant",
            "chauffeur poids lourd",
            "agent de sécurité"
        ]
        
        # Créer et tester le modèle
        try:
            predictor = ISCOPredictor(MODEL_PATH, MAPPINGS_PATH)
            print(f"
{'='*60}")
            print("TESTS AUTOMATIQUES SUR 10 EXEMPLES:")
            print('='*60)
            
            for text in test_texts:
                predictions = predictor.predict(text, top_k=3)
                print(f"
🔍 "{text}"")
                for pred in predictions:
                    print(f"   {pred['rank']}. {pred['code']} ({pred['confidence']:.1f}%)")
        
        except Exception as e:
            print(f"Erreur: {e}")
            print("
Assurez-vous que:")
            print("1. Le modèle est entraîné (fichier .ckpt existe)")
            print("2. Les mappings sont sauvegardés (.pkl existe)")
        
        return
    
    # Prédiction pour l'argument fourni
    text_to_predict = " ".join(sys.argv[1:])
    
    try:
        # Charger et prédire
        predictor = ISCOPredictor(MODEL_PATH, MAPPINGS_PATH)
        
        print(f"
{'='*60}")
        print(f"PRÉDICTION POUR: "{text_to_predict}"")
        print('='*60)
        
        predictions = predictor.predict(text_to_predict, top_k=5)
        
        for pred in predictions:
            confidence_bar = "█" * int(pred['confidence'] / 5)
            print(f"
{pred['rank']}. Code ISCO: {pred['code']}")
            print(f"   Confiance: {pred['confidence']:.1f}% [{confidence_bar:<20}]")
        
        print(f"
{'='*60}")
        print(f"Meilleure prédiction: {predictions[0]['code']} ({predictions[0]['confidence']:.1f}% de confiance)")
        
    except FileNotFoundError as e:
        print(f"❌ Fichier non trouvé: {e}")
        print("
📋 Pour entraîner le modèle d'abord:")
        print("   python scripts/training.py")
    
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    main()
