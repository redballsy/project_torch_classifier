import pytorch_lightning as pl
from torchTestClassifiers.models.model import CITPClassifier
from torchTestClassifiers.data.datamodule import CITPDataModule
import torch
import fasttext

# Chemins
PATH_EXCEL = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\pretrain\CITP_08.xlsx"
PATH_FASTTEXT = r"C:\Users\Sy Savane Idriss\project_torch_classifier\modelsfastext\cc.fr.300.bin"

def pretrain():
    # 1. Charger les données officielles
    dm = CITPDataModule(data_path=PATH_EXCEL, batch_size=16)
    dm.setup()
    
    # 2. Préparer les vecteurs FastText
    ft_model = fasttext.load_model(PATH_FASTTEXT)
    vocab = dm.vocab
    pretrained_vectors = torch.zeros(len(vocab), 300)
    for word, i in vocab.items():
        pretrained_vectors[i] = torch.from_numpy(ft_model.get_word_vector(word))
    
    # 3. Créer le modèle
    model = CITPClassifier(
        num_classes=dm.num_classes,
        vocab_size=len(vocab),
        pretrained_vectors=pretrained_vectors
    )

    # 4. Configurer un entraînement agressif pour la mémorisation
    # On veut que le modèle colle parfaitement aux données (Overfitting voulu ici)
    trainer = pl.Trainer(
        max_epochs=100, 
        accelerator="cpu",
        devices=1,
        enable_checkpointing=True
    )

    print("🧠 Mémorisation de la nomenclature officielle...")
    trainer.fit(model, datamodule=dm)
    
    # 5. Sauvegarder le cerveau "pré-entraîné"
    trainer.save_checkpoint("models/pretrained_memory.ckpt")
    print("✅ Nomenclature mémorisée dans 'models/pretrained_memory.ckpt'")

if __name__ == "__main__":
    pretrain()