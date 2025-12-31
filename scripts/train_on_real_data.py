import torch
import torch.nn as nn
import pytorch_lightning as pl
import fasttext
import os
import pandas as pd
from torchmetrics import Accuracy
from torchTestClassifiers.models.model import CITPClassifier
from torchTestClassifiers.data.datamodule import CITPDataModule

# --- CONFIGURATION DES CHEMINS ---
PATH_PRETRAIN_DATA = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\pretrain\CITP_08.xlsx"
PATH_REAL_DATA = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\JOB1_Codifications_Professions CNPS_vf.xlsx"
PATH_PRETRAINED_MODEL = r"C:\Users\Sy Savane Idriss\project_torch_classifier\models\pretrained_memory.ckpt"
PATH_FASTTEXT_FR = r"C:\Users\Sy Savane Idriss\project_torch_classifier\modelsfastext\cc.fr.300.bin"
SAVE_PATH = r"C:\Users\Sy Savane Idriss\project_torch_classifier\models\fine_tuned_model.ckpt"

def fine_tune_with_memory():
    # 1. Charger les données réelles (CNPS)
    print("📊 Chargement des données CNPS...")
    # Augmente num_workers si ton CPU le permet (ex: 4 ou 7) pour aller plus vite
    dm = CITPDataModule(data_path=PATH_REAL_DATA, batch_size=32)
    dm.setup()
    
    # Charger le vocabulaire de référence (ISCO) pour le transfert
    print("📖 Chargement du vocabulaire de référence (ISCO)...")
    dm_ref = CITPDataModule(data_path=PATH_PRETRAIN_DATA)
    dm_ref.setup()
    vocab_pretrain = dm_ref.vocab

    # 2. Charger le modèle pré-entraîné
    print("🧠 Chargement du modèle pré-entraîné...")
    pretrained_model = CITPClassifier.load_from_checkpoint(
        PATH_PRETRAINED_MODEL,
        map_location="cpu"
    )
    
    # 3. Préparation des embeddings hybrides
    print("✨ Initialisation des embeddings hybrides (Pretrain + FastText)...")
    ft_fr = fasttext.load_model(PATH_FASTTEXT_FR)
    new_vectors = torch.zeros(len(dm.vocab), 300)
    pretrained_embeddings = pretrained_model.embedding.weight.data
    
    for word, i in dm.vocab.items():
        if word in vocab_pretrain:
            idx_pretrained = vocab_pretrain[word]
            if idx_pretrained < pretrained_embeddings.size(0):
                new_vectors[i] = pretrained_embeddings[idx_pretrained]
        else:
            # Nouveau mot inconnu du pretraining -> FastText
            new_vectors[i] = torch.from_numpy(ft_fr.get_word_vector(word))

    # 4. Créer le nouveau modèle adapté aux 389 classes
    print(f"🏗️ Création du modèle pour {dm.num_classes} classes...")
    model = CITPClassifier(
        num_classes=dm.num_classes,
        vocab_size=len(dm.vocab),
        pretrained_vectors=new_vectors
    )
    
    # 5. TRANSFERT DES POIDS (Intelligence de compréhension)
    print("🔄 Transfert des connaissances (Filtrage des couches)...")
    state_dict = model.state_dict()
    pretrained_dict = pretrained_model.state_dict()
    
    for name, param in pretrained_dict.items():
        if name in state_dict:
            # On ignore les couches de sortie (tailles différentes) et l'embedding (déjà fait)
            if "classifier" in name or "fc" in name or "embedding" in name:
                continue
            
            # Copie si les dimensions sont identiques (LSTM, Linéaires cachées)
            if param.shape == state_dict[name].shape:
                state_dict[name].copy_(param)

    model.load_state_dict(state_dict)
    print("✅ Transfert des couches de compréhension effectué !")

    # 6. Configuration de l'entraînement (Correction du monitor 'val_acc')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.dirname(SAVE_PATH),
        filename="best_fine_tuned",
        monitor="val_acc", # Nom exact utilisé dans les logs
        mode="max",
        save_top_k=1
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", 
        patience=10, 
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    # 7. Lancement du Fine-Tuning
    print("\n🎯 Début du Fine-Tuning sur données réelles...")
    trainer.fit(model, datamodule=dm)
    
    # Sauvegarde finale du checkpoint complet
    trainer.save_checkpoint(SAVE_PATH)
    print(f"\n💾 Modèle final sauvegardé dans : {SAVE_PATH}")

if __name__ == "__main__":
    fine_tune_with_memory()