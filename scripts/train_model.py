import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchTestClassifiers.models.model import CITPClassifier
from torchTestClassifiers.data.datamodule import CITPDataModule
import torch
import fasttext
import os

PATH_EXCEL = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\citp_six_positions_nomenclature.xlsx"
PATH_FASTTEXT = r"C:\Users\Sy Savane Idriss\project_torch_classifier\modelsfastext\cc.fr.300.bin"

def train():
    # 1. Données
    dm = CITPDataModule(data_path=PATH_EXCEL, batch_size=32)
    dm.setup()
    
    # 2. FastText
    print("⏳ Chargement du modèle FastText...")
    ft_model = fasttext.load_model(PATH_FASTTEXT)
    
    vocab = dm.vocab 
    embedding_dim = 300
    pretrained_vectors = torch.zeros(len(vocab), embedding_dim)
    
    for word, i in vocab.items():
        pretrained_vectors[i] = torch.from_numpy(ft_model.get_word_vector(word))
    
    print(f"✅ Vocabulaire aligné : {len(vocab)} mots.")

    # 3. Modèle
    model = CITPClassifier(
        num_classes=dm.num_classes,
        vocab_size=len(vocab),
        pretrained_vectors=pretrained_vectors
    )

    # 4. Tuning des Callbacks
    # On laisse 15 époques de chance au modèle pour s'améliorer
    early_stop = EarlyStopping(monitor="val_loss", patience=15, mode="min", verbose=True)
    
    checkpoint = ModelCheckpoint(
        monitor="val_acc", 
        mode="max", 
        save_top_k=1, 
        filename="best_citp_model"
    )

    # 5. Trainer (Configuré pour ton CPU)
    trainer = pl.Trainer(
        max_epochs=200,              # On augmente le maximum, EarlyStopping gérera la fin
        callbacks=[early_stop, checkpoint],
        log_every_n_steps=5,
        accelerator="cpu",
        devices=1
    )

    print("🚀 Lancement de l'entraînement amélioré...")
    trainer.fit(model, datamodule=dm)
    print(f"✅ Terminé ! Meilleur modèle : {checkpoint.best_model_path}")

if __name__ == "__main__":
    train()