import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from collections import Counter

class CITPDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=10):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 1. Transformer le texte en liste de chiffres selon le vocabulaire
        tokens = str(self.texts[idx]).lower().split()
        # On remplace chaque mot par son ID, ou 0 si le mot est inconnu
        encoded = [self.vocab.get(token, 0) for token in tokens]
        
        # 2. Ajuster la longueur (Padding) pour que tous les exemples fassent la même taille
        if len(encoded) < self.max_length:
            encoded += [0] * (self.max_length - len(encoded))
        else:
            encoded = encoded[:self.max_length]
            
        return torch.tensor(encoded), torch.tensor(self.labels[idx])

class CITPDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=32):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.vocab = {"<PAD>": 0, "<UNK>": 1} # Vocabulaire de base

    def setup(self, stage=None):
        df = pd.read_excel(self.data_path)
        texts = df['nomenclature'].astype(str).tolist()
        
        # --- Création du vocabulaire ---
        words = []
        for t in texts:
            words.extend(t.lower().split())
        
        counts = Counter(words)
        for i, (word, count) in enumerate(counts.items(), start=2):
            self.vocab[word] = i
        
        self.vocab_size = len(self.vocab)
        # -------------------------------

        self.label_map = {label: i for i, label in enumerate(df['code'].unique())}
        self.num_classes = len(self.label_map)
        labels = [self.label_map[l] for l in df['code']]
        
        dataset = CITPDataset(texts, labels, self.vocab)
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_ds, self.val_ds = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)