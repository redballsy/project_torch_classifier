import pytorch_lightning as pl
from torchTestClassifiers.models.model import CITPClassifier
from torchTestClassifiers.data.datamodule import CITPDataModule
import torch
import fasttext
import os

PATH_REAL_DATA = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\file_manually_coded.xlsx"
PATH_PRETRAINED_MODEL = r"C:\Users\Sy Savane Idriss\project_torch_classifier\models\pretrained_memory.ckpt"
PATH_FASTTEXT = r"C:\Users\Sy Savane Idriss\project_torch_classifier\modelsfastext\cc.fr.300.bin"

def fine_tune():
    # 1. Charger les données réelles
    dm = CITPDataModule(data_path=PATH_REAL_DATA, batch_size=32)
    dm.setup()
    
    # 2. Préparer FastText
    ft_model = fasttext.load_model(PATH_FASTTEXT)
    new_vectors = torch.zeros(len(dm.vocab), 300)
    for word, i in dm.vocab.items():
        new_vectors[i] = torch.from_numpy(ft_model.get_word_vector(word))
    
    # 3. Créer le modèle (Vierge au départ)
    model = CITPClassifier(num_classes=dm.num_classes, vocab_size=len(dm.vocab), pretrained_vectors=new_vectors)

    # 4. INJECTER LE SAVOIR DU PRETRAIN
    print("🧠 Récupération de l'intelligence du Pretrain...")
    checkpoint = torch.load(PATH_PRETRAINED_MODEL, map_location="cpu")
    state_dict = checkpoint['state_dict']
    
    # On garde UNIQUEMENT la logique (fc_hidden)
    # On enlève ce qui concerne les 864 classes pour ne pas avoir d'erreur
    keys_to_keep = {k: v for k, v in state_dict.items() if "fc_hidden" in k}
    
    # On injecte cette logique dans notre nouveau modèle
    model.load_state_dict(keys_to_keep, strict=False)
    print("✅ Logique métier injectée ! Le modèle ne repart pas de zéro.")

    # 5. Entraînement sur les données réelles
    trainer = pl.Trainer(max_epochs=50, accelerator="cpu", devices=1)
    trainer.fit(model, datamodule=dm)
    
    trainer.save_checkpoint("models/final_model_real.ckpt")

if __name__ == "__main__":
    fine_tune()