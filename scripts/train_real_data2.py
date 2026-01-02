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
PATH_REAL_DATA = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\CNPS_Nettoyer2_Final.xlsx"
PATH_PRETRAINED_MODEL = r"C:\Users\Sy Savane Idriss\project_torch_classifier\models\pretrained_memory.ckpt"
PATH_MEMORY = r"C:\Users\Sy Savane Idriss\project_torch_classifier\models\vocab_memory.pkl"
PATH_FASTTEXT_FR = r"C:\Users\Sy Savane Idriss\project_torch_classifier\modelsfastext\cc.fr.300.bin"
SAVE_PATH = r"C:\Users\Sy Savane Idriss\project_torch_classifier\models\fine_tuned_model.ckpt"

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
    
    # 5. Créer le modèle
    print(f"🏗️ Création du modèle ({dm.num_classes} classes)...")
    model = CITPClassifier(
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
        save_top_k=2
    )
    
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )
    
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=5,
        enable_progress_bar=True
    )
    
    # 7. Entraînement
    print("\n🎯 Début de l'entraînement...")
    trainer.fit(model, datamodule=dm)
    
    # 8. Sauvegarde
    print("\n💾 Sauvegarde...")
    trainer.save_checkpoint(SAVE_PATH)
    
    # Sauvegarde supplémentaire
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': dm.vocab,
        'num_classes': dm.num_classes
    }, SAVE_PATH.replace('.ckpt', '.pt'))
    
    print(f"\n✅ Fine-tuning terminé !")
    print(f"📁 Modèle: {SAVE_PATH}")
    
    return model

if __name__ == "__main__":
    print("⚠️ Version simplifiée - sans transfert de prétrain")
    model = fine_tune_simple()
    
    if model:
        print("\n" + "=" * 70)
        print("✨ ENTRAÎNEMENT RÉUSSI !")
        print("=" * 70)
        
        print(f"\n📊 Modèle final:")
        print(f"  - Classes: {model.num_classes if hasattr(model, 'num_classes') else 'N/A'}")
        print(f"  - Vocabulaire: {len(model.vocab) if hasattr(model, 'vocab') else len(dm.vocab)}")
    else:
        print("\n❌ L'entraînement a échoué.")