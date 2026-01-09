import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import fasttext
import numpy as np
import mlflow
import mlflow.pytorch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score
import warnings

# ============================================
# BRIDGE DE COMPATIBILITÉ NUMPY
# ============================================
if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np
    try:
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
    except AttributeError:
        pass

warnings.filterwarnings('ignore')

# CONFIGURATION MLFLOW DYNAMIQUE
IS_GITHUB = os.getenv('GITHUB_ACTIONS') == 'true'

if not IS_GITHUB:
    mlflow.set_tracking_uri("http://localhost:5000")
    print("🏠 Mode Local : Tracking vers MLflow Localhost")
else:
    print("🤖 Mode GitHub : Sauvegarde locale uniquement (Artifacts)")

mlflow.set_experiment("CITP_Classification_Project")

# ============================================
# Configuration Dynamique des Chemins
# ============================================
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_SCRIPT_DIR)

FASTTEXT_MODEL_PATH = os.path.join(BASE_DIR, "models_fasttext", "cc.fr.300.bin")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "torchTestClassifiers", "data", "entrainer", "entrainer2_propre.xlsx")
SAVED_MODEL_PATH = os.path.join(BASE_DIR, "models", "citp_classifier_model.pth")

os.makedirs(os.path.dirname(SAVED_MODEL_PATH), exist_ok=True)

# ============================================
# 1. Classe Dataset
# ============================================
class CITPDataset(Dataset):
    def __init__(self, dataframe, ft_model, label_encoder):
        self.embeddings = []
        labels_str = dataframe['code'].astype(str).tolist()
        self.labels = label_encoder.transform(labels_str)
        
        print(f"Vectorisation de {len(dataframe)} lignes...")
        for i, text in enumerate(dataframe['nomenclature']):
            if i % 10000 == 0 and i > 0:
                print(f"  {i}/{len(dataframe)}...")
            clean_text = str(text).lower().strip().replace("\n", " ")
            vector = ft_model.get_sentence_vector(clean_text)
            self.embeddings.append(vector)

    def __len__(self): return len(self.embeddings)
    
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
# 3. Fonction pour calculer les métriques
# ============================================
def calculate_metrics(all_labels, all_preds):
    """Calcule les métriques de performance"""
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    return accuracy, recall

# ============================================
# 4. Fonction d'entraînement avec validation et métriques
# ============================================
def train_main():
    with mlflow.start_run(run_name="Training_CITP_Torch"):
        print(f"🧠 Chargement du modèle FastText...")
        ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
        
        df = pd.read_excel(TRAIN_DATA_PATH).dropna(subset=['code', 'nomenclature'])
        df['code_str'] = df['code'].astype(str)

        # Simple oversampling pour classes avec 1 exemple
        print(f"\n🔍 Analyse des classes...")
        class_counts = df['code_str'].value_counts()
        orphan_classes = class_counts[class_counts == 1].index.tolist()
        
        if orphan_classes:
            print(f"  {len(orphan_classes)} classes avec 1 seul exemple")
            print("  🌀 Duplication des échantillons uniques...")
            
            # Dupliquer les échantillons uniques
            duplicated_rows = []
            for class_name in orphan_classes:
                sample = df[df['code_str'] == class_name].iloc[0]
                duplicated_rows.append({
                    'code': sample['code'],
                    'nomenclature': sample['nomenclature'],
                    'code_str': sample['code_str']
                })
            
            # Ajouter au DataFrame
            if duplicated_rows:
                duplicated_df = pd.DataFrame(duplicated_rows)
                df = pd.concat([df, duplicated_df], ignore_index=True)
                print(f"  ✅ {len(duplicated_rows)} échantillons ajoutés")
        
        le = LabelEncoder()
        le.fit(df['code_str'])
        
        # Split stratifié
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['code_str'])
        
        # Création des DataLoaders
        train_loader = DataLoader(CITPDataset(train_df, ft_model, le), batch_size=64, shuffle=True)
        val_loader = DataLoader(CITPDataset(val_df, ft_model, le), batch_size=64, shuffle=False)
        
        num_classes = len(le.classes_)
        model = CITPClassifier(300, num_classes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # ============================================
        # LOGGING DES HYPERPARAMÈTRES
        # ============================================
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("num_layers", 3)
        mlflow.log_param("layer1_neurons", 512)
        mlflow.log_param("layer2_neurons", 256)
        mlflow.log_param("layer3_neurons", num_classes)
        mlflow.log_param("input_dim", 300)
        mlflow.log_param("dropout_rate", 0.3)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("val_samples", len(val_df))
        mlflow.log_param("epochs", 20)  # Réduit pour être plus rapide
        mlflow.log_param("optimizer", "AdamW")

        print(f"\n🚀 Début de l'entraînement (20 epochs)...")
        print(f"📊 Données d'entraînement: {len(train_df)}")
        print(f"📊 Données de validation: {len(val_df)}")
        
        train_losses = []
        train_accuracies = []
        train_recalls = []
        
        val_losses = []
        val_accuracies = []
        val_recalls = []
        
        for epoch in range(20):
            # ===== TRAINING =====
            model.train()
            total_loss = 0
            train_preds = []
            train_labels = []
            
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(batch['embedding'])
                loss = criterion(outputs, batch['label'])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                train_preds.extend(preds.numpy())
                train_labels.extend(batch['label'].numpy())
            
            avg_train_loss = total_loss / len(train_loader)
            train_acc, train_rec = calculate_metrics(train_labels, train_preds)
            
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_acc)
            train_recalls.append(train_rec)
            
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("train_recall", train_rec, step=epoch)
            
            # ===== VALIDATION =====
            model.eval()
            val_total_loss = 0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(batch['embedding'])
                    loss = criterion(outputs, batch['label'])
                    val_total_loss += loss.item()
                    
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.numpy())
                    val_labels.extend(batch['label'].numpy())
            
            avg_val_loss = val_total_loss / len(val_loader)
            val_acc, val_rec = calculate_metrics(val_labels, val_preds)
            
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_acc)
            val_recalls.append(val_rec)
            
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("val_recall", val_rec, step=epoch)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"\n📊 Epoch {epoch+1}/20")
                print(f"   Training - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, Recall: {train_rec:.4f}")
                print(f"   Validation - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, Recall: {val_rec:.4f}")
        
        # ============================================
        # SAUVEGARDE
        # ============================================
        state = {
            'model_state_dict': model.state_dict(),
            'label_encoder': le,
            'input_dim': 300,
            'num_classes': num_classes,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'train_recalls': train_recalls,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'val_recalls': val_recalls
        }
        torch.save(state, SAVED_MODEL_PATH)
        
        # ============================================
        # AFFICHAGE RÉCAPITULATIF
        # ============================================
        print("\n" + "="*50)
        print("📋 RÉCAPITULATIF")
        print("="*50)
        print(f"\n📊 MÉTRIQUES FINALES:")
        print(f"{'':<12} {'Training':<10} {'Validation':<10}")
        print(f"{'-'*35}")
        print(f"{'Loss':<12} {train_losses[-1]:<10.4f} {val_losses[-1]:<10.4f}")
        print(f"{'Accuracy':<12} {train_accuracies[-1]:<10.4f} {val_accuracies[-1]:<10.4f}")
        print(f"{'Recall':<12} {train_recalls[-1]:<10.4f} {val_recalls[-1]:<10.4f}")
        
        print(f"\n⚙️  HYPERPARAMÈTRES:")
        print(f"  • Batch Size: 64")
        print(f"  • Learning Rate: 0.001")
        print(f"  • Nombre de couches: 3")
        print(f"  • Neurones: 512 → 256 → {num_classes}")
        print(f"  • Dropout: 0.3")
        
        print(f"\n📈 DONNÉES:")
        print(f"  • Échantillons d'entraînement: {len(train_df)}")
        print(f"  • Échantillons de validation: {len(val_df)}")
        print(f"  • Nombre de classes: {num_classes}")
        print("="*50 + "\n")

        if not IS_GITHUB:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name="Job_Classifier_CITP"
            )
        else:
            mlflow.pytorch.log_model(pytorch_model=model, artifact_path="model")
        
        print(f"✅ Entraînement terminé. Modèle sauvegardé: {SAVED_MODEL_PATH}")

if __name__ == "__main__":
    train_main()