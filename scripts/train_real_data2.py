import torch
import torch.nn as nn
import pytorch_lightning as pl
import fasttext
import os
import pandas as pd
import pickle
from torchTestClassifiers.models.model import CITPClassifier
from torchTestClassifiers.data.datamodule import CITPDataModule

# Chemins
PATH_PRETRAIN_DATA = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\pretrain\CITP_08.xlsx"
PATH_REAL_DATA = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\fichier_A_entrainer.xlsx"
PATH_PRETRAINED_MODEL = r"C:\Users\Sy Savane Idriss\project_torch_classifier\models\pretrained_memory.ckpt"
PATH_MEMORY = r"C:\Users\Sy Savane Idriss\project_torch_classifier\models\vocab_memory.pkl"
PATH_FASTTEXT_FR = r"C:\Users\Sy Savane Idriss\project_torch_classifier\modelsfastext\cc.fr.300.bin"
SAVE_PATH = r"C:\Users\Sy Savane Idriss\project_torch_classifier\models\fine_tuned_model.ckpt"

class EnhancedCITPClassifier(CITPClassifier):
    """Version améliorée avec validation accuracy"""
    
    def __init__(self, num_classes, vocab_size, pretrained_vectors):
        # Extraire la dimension d'embedding du tensor pretrained_vectors
        embedding_dim = pretrained_vectors.size(1)
        super().__init__(num_classes, vocab_size, embedding_dim)
        
        # Charger les poids pré-entraînés dans l'embedding
        self.embedding.weight.data.copy_(pretrained_vectors)
    
    def validation_step(self, batch, batch_idx):
        """Étape de validation avec calcul d'accuracy"""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # Calculer l'accuracy manuellement
        preds = torch.argmax(y_hat, dim=1)
        accuracy = (preds == y).float().mean()
        
        # Logguer les métriques
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

def fine_tune_simple():
    """Version simplifiée du fine-tuning"""
    
    print("=" * 70)
    print("🎯 FINE-TUNING SIMPLIFIÉ")
    print("=" * 70)
    
    # 1. Vérifier les fichiers
    print("🔍 Vérification des fichiers...")
    
    if not os.path.exists(PATH_PRETRAINED_MODEL):
        print(f"❌ Modèle pré-entraîné non trouvé: {PATH_PRETRAINED_MODEL}")
        print("➡️ Exécutez d'abord: python scripts/pretrain2.py")
        return None
    
    # 2. Charger les données CNPS
    print("\n📊 Chargement des données CNPS...")
    dm = CITPDataModule(data_path=PATH_REAL_DATA, batch_size=32)
    dm.setup()
    
    # 3. Charger FastText pour les embeddings
    print("🔤 Chargement FastText...")
    ft_fr = fasttext.load_model(PATH_FASTTEXT_FR)
    
    # 4. Créer les embeddings avec FastText uniquement
    print(f"✨ Création des embeddings ({len(dm.vocab)} mots)...")
    new_vectors = torch.zeros(len(dm.vocab), 300)
    
    for word, i in dm.vocab.items():
        try:
            vec = ft_fr.get_word_vector(word)
            new_vectors[i] = torch.from_numpy(vec)
        except:
            new_vectors[i] = torch.randn(300)
    
    # 5. Créer le modèle amélioré
    print(f"🏗️ Création du modèle ({dm.num_classes} classes)...")
    
    # Vérifier les dimensions
    print(f"  Vocab size: {len(dm.vocab)}")
    print(f"  Embedding dimension: {new_vectors.size(1)}")
    print(f"  Num classes: {dm.num_classes}")
    
    model = EnhancedCITPClassifier(
        num_classes=dm.num_classes,
        vocab_size=len(dm.vocab),
        pretrained_vectors=new_vectors
    )
    
    # 6. Configuration de l'entraînement
    print("\n⚙️ Configuration de l'entraînement...")
    
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.dirname(SAVE_PATH),
        filename="model-{epoch:02d}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=2,
        verbose=True
    )
    
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        verbose=True
    )
    
    progress_bar = pl.callbacks.TQDMProgressBar(refresh_rate=10)
    
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback, progress_bar],
        log_every_n_steps=5,
        enable_progress_bar=True,
        enable_model_summary=True,
        num_sanity_val_steps=0
    )
    
    # 7. Entraînement
    print("\n🎯 Début de l'entraînement...")
    trainer.fit(model, datamodule=dm)
    
    # 8. Afficher les meilleures métriques
    print("\n📈 MEILLEURES MÉTRIQUES :")
    print(f"Best validation accuracy: {checkpoint_callback.best_model_score:.3f}")
    
    # 9. Sauvegarde
    print("\n💾 Sauvegarde...")
    trainer.save_checkpoint(SAVE_PATH)
    
    # Sauvegarde supplémentaire
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': dm.vocab,
        'num_classes': dm.num_classes,
        'val_accuracy': checkpoint_callback.best_model_score
    }, SAVE_PATH.replace('.ckpt', '.pt'))
    
    print(f"\n✅ Fine-tuning terminé !")
    print(f"📁 Modèle: {SAVE_PATH}")
    
    return model, checkpoint_callback.best_model_score

if __name__ == "__main__":
    print("⚠️ Version simplifiée - sans transfert de prétrain")
    model, best_acc = fine_tune_simple()
    
    if model:
        print("\n" + "=" * 70)
        print("✨ ENTRAÎNEMENT RÉUSSI !")
        print("=" * 70)
        
        print(f"\n📊 RÉSULTATS FINAUX:")
        print(f"  - Classes: {model.num_classes if hasattr(model, 'num_classes') else 'N/A'}")
        print(f"  - Vocabulaire: {len(model.vocab) if hasattr(model, 'vocab') else len(dm.vocab)}")
        print(f"  - Best validation accuracy: {best_acc:.3f}")
        
        print("\n🧪 Test rapide du modèle...")
        dm = CITPDataModule(data_path=PATH_REAL_DATA, batch_size=32)
        dm.setup()
        
        val_loader = dm.val_dataloader()
        batch = next(iter(val_loader))
        x, y = batch
        
        with torch.no_grad():
            model.eval()
            y_hat = model(x)
            preds = torch.argmax(y_hat, dim=1)
            accuracy = (preds == y).float().mean()
            print(f"  - Test batch accuracy: {accuracy:.3f}")
        
    else:
        print("\n❌ L'entraînement a échoué.")