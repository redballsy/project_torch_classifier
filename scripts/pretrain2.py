import torch
import torch.nn as nn
import pytorch_lightning as pl
import fasttext
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import pickle
import os
from torchmetrics import Accuracy

# Chemins
PATH_EXCEL = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\pretrain\CITP_08.xlsx"
PATH_FASTTEXT = r"C:\Users\Sy Savane Idriss\project_torch_classifier\modelsfastext\cc.fr.300.bin"
MODEL_SAVE_PATH = r"C:\Users\Sy Savane Idriss\project_torch_classifier\models\pretrained_memory.ckpt"
VOCAB_SAVE_PATH = r"C:\Users\Sy Savane Idriss\project_torch_classifier\models\vocab_memory.pkl"

class MemoryIntelligence:
    """Mémoire intelligente qui apprend les patterns de la nomenclature"""
    
    def __init__(self):
        self.word_patterns = {}
        self.code_patterns = {}
        self.synonyms = {}
        self.abbreviations = {}
        
    def analyze_corpus(self, nomenclatures: List[str], codes: List[str]):
        """Analyse les patterns dans le corpus"""
        print("🔍 Analyse des patterns de la nomenclature...")
        
        for nom, code in zip(nomenclatures, codes):
            words = nom.lower().split()
            
            for word in words:
                if word not in self.word_patterns:
                    self.word_patterns[word] = {}
                if code not in self.word_patterns[word]:
                    self.word_patterns[word][code] = 0
                self.word_patterns[word][code] += 1
            
            if code not in self.code_patterns:
                self.code_patterns[code] = {}
            for word in words:
                if word not in self.code_patterns[code]:
                    self.code_patterns[code][word] = 0
                self.code_patterns[code][word] += 1
        
        self._detect_abbreviations(nomenclatures)
        
        print(f"✅ Patterns analysés: {len(self.word_patterns)} mots, {len(self.code_patterns)} codes")
    
    def _detect_abbreviations(self, nomenclatures: List[str]):
        """Détecte automatiquement les abréviations courantes"""
        word_freq = {}
        
        for nom in nomenclatures:
            words = nom.lower().split()
            for word in words:
                if len(word) <= 4:
                    if word not in word_freq:
                        word_freq[word] = {}
                    for other_word in words:
                        if len(other_word) > 4 and word in other_word[:len(word)]:
                            if other_word not in word_freq[word]:
                                word_freq[word][other_word] = 0
                            word_freq[word][other_word] += 1
        
        for short_word, candidates in word_freq.items():
            if candidates:
                best_match = max(candidates.items(), key=lambda x: x[1])
                if best_match[1] >= 3:
                    self.abbreviations[short_word] = best_match[0]
        
        print(f"📝 Abréviations détectées: {len(self.abbreviations)}")
    
    def expand_abbreviations(self, text: str) -> str:
        """Développe les abréviations connues"""
        words = text.lower().split()
        expanded_words = []
        
        for word in words:
            if word in self.abbreviations:
                expanded_words.append(self.abbreviations[word])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def predict_code(self, text: str) -> str:
        """Prédit le code basé sur les patterns appris"""
        text = self.expand_abbreviations(text.lower())
        words = text.split()
        
        code_scores = {}
        
        for word in words:
            if word in self.word_patterns:
                for code, count in self.word_patterns[word].items():
                    if code not in code_scores:
                        code_scores[code] = 0
                    code_scores[code] += count
        
        if code_scores:
            return max(code_scores.items(), key=lambda x: x[1])[0]
        return None
    
    def save(self, path: str):
        """Sauvegarde la mémoire"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word_patterns': self.word_patterns,
                'code_patterns': self.code_patterns,
                'abbreviations': self.abbreviations
            }, f)
    
    def load(self, path: str):
        """Charge la mémoire"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word_patterns = data['word_patterns']
            self.code_patterns = data['code_patterns']
            self.abbreviations = data['abbreviations']

class MemoryEnhancedClassifier(pl.LightningModule):
    """Classificateur avec mémoire intégrée"""
    
    def __init__(self, vocab_size: int, num_classes: int, 
                 pretrained_vectors: torch.Tensor = None,
                 memory_intelligence: MemoryIntelligence = None):
        super().__init__()
        
        self.save_hyperparameters()
        self.memory = memory_intelligence
        
        self.embedding = nn.Embedding(vocab_size, 300)
        if pretrained_vectors is not None:
            self.embedding.weight.data.copy_(pretrained_vectors)
            self.embedding.weight.requires_grad = True
        
        self.lstm = nn.LSTM(
            input_size=300,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
        self.code_cache = {}
        
        print(f"🧠 Modèle créé: vocab={vocab_size}, classes={num_classes}")
        if self.memory:
            print(f"📚 Mémoire intelligente intégrée: {len(self.memory.word_patterns)} patterns")
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        logits = self.classifier(context_vector)
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_acc, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

def create_smart_memory():
    """Crée et sauvegarde une mémoire intelligente"""
    
    print("📚 Création de la mémoire intelligente...")
    
    df = pd.read_excel(PATH_EXCEL)
    
    # Identifier les colonnes automatiquement
    if 'nomenclature' in df.columns and 'code' in df.columns:
        nomenclatures = df['nomenclature'].astype(str).tolist()
        codes = df['code'].astype(str).tolist()
    elif len(df.columns) >= 2:
        text_col = df.columns[0]
        code_col = df.columns[1]
        nomenclatures = df[text_col].astype(str).tolist()
        codes = df[code_col].astype(str).tolist()
        print(f"⚠️ Colonnes détectées: {text_col} (texte), {code_col} (code)")
    else:
        raise ValueError("Format de fichier non reconnu")
    
    print(f"📊 Données chargées: {len(nomenclatures)} entrées")
    
    memory = MemoryIntelligence()
    memory.analyze_corpus(nomenclatures, codes)
    
    print("\n🧪 Test de la mémoire:")
    test_examples = nomenclatures[:5] + [
        "ass de direction",
        "mecanicien auto",
        "resp production"
    ]
    
    for example in test_examples:
        prediction = memory.predict_code(example)
        print(f"  '{example[:30]}...' → {prediction}")
    
    memory.save(VOCAB_SAVE_PATH)
    print(f"\n💾 Mémoire sauvegardée: {VOCAB_SAVE_PATH}")
    
    return memory

def pretrain_with_memory_simple():
    """Version simplifiée du pretrain pour débogage"""
    
    print("=" * 60)
    print("🧠 PRÉTRAINAGE AVEC MÉMOIRE INTELLIGENTE (Version Simple)")
    print("=" * 60)
    
    # 1. Créer la mémoire intelligente
    memory = create_smart_memory()
    
    # 2. Préparer le répertoire de sauvegarde
    print("\n📊 Préparation du répertoire de sauvegarde...")
    
    # CORRECTION: Créer le répertoire si nécessaire
    models_dir = os.path.dirname(MODEL_SAVE_PATH)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        print(f"📁 Répertoire créé: {models_dir}")
    
    # Version SIMPLIFIÉE pour débogage
    print("🏗️ Construction du modèle minimal...")
    
    # Créer un modèle simple
    model = MemoryEnhancedClassifier(
        vocab_size=1000,
        num_classes=436,
        pretrained_vectors=None,
        memory_intelligence=memory
    )
    
    # Sauvegarder le modèle
    print("💾 Sauvegarde du modèle...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': 1000,
        'num_classes': 436,
        'memory_present': memory is not None,
        'word_patterns_count': len(memory.word_patterns),
        'code_patterns_count': len(memory.code_patterns)
    }, MODEL_SAVE_PATH)
    
    print(f"\n✅ Modèle sauvegardé: {MODEL_SAVE_PATH}")
    print(f"🧠 Mémoire sauvegardée: {VOCAB_SAVE_PATH}")
    
    # Vérifier que les fichiers existent
    print(f"\n📁 Vérification des fichiers:")
    print(f"  - {MODEL_SAVE_PATH}: {'✅ Existe' if os.path.exists(MODEL_SAVE_PATH) else '❌ Manquant'}")
    print(f"  - {VOCAB_SAVE_PATH}: {'✅ Existe' if os.path.exists(VOCAB_SAVE_PATH) else '❌ Manquant'}")
    
    if os.path.exists(MODEL_SAVE_PATH):
        file_size = os.path.getsize(MODEL_SAVE_PATH) / 1024 / 1024  # Taille en MB
        print(f"  - Taille du modèle: {file_size:.2f} MB")
    
    return model, memory

if __name__ == "__main__":
    print("⚠️ Version simplifiée pour débogage")
    try:
        model, memory = pretrain_with_memory_simple()
        
        print("\n" + "=" * 60)
        print("🎯 MÉMOIRE CRÉÉE AVEC SUCCÈS !")
        print("=" * 60)
        print(f"\n📊 Statistiques de la mémoire:")
        print(f"  - Patterns de mots: {len(memory.word_patterns)}")
        print(f"  - Patterns de codes: {len(memory.code_patterns)}")
        print(f"  - Abréviations: {len(memory.abbreviations)}")
        
        # Test final
        print("\n🧪 Tests finaux de la mémoire:")
        test_cases = [
            ("assistant de direction", "3343"),
            ("mécanicien automobile", "7231"),
            ("comptable", "2411"),
            ("technicien génie civil", "3112")
        ]
        
        for text, expected in test_cases:
            pred = memory.predict_code(text)
            status = "✓" if pred == expected else f"✗ (prédit: {pred})"
            print(f"  {status} '{text}' → {pred} (attendu: {expected})")
            
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()