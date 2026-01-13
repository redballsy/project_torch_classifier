# --- Patch SSL pour HuggingFace / requests ---
import os, certifi
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModel, Trainer, TrainingArguments,
                          get_linear_schedule_with_warmup)
from sentence_transformers import SentenceTransformer
import evaluate
import joblib

# -----------------------------
# CONFIG
# -----------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\entrainer2_propre.xlsx"
LLM_NAME = "camembert-base"
SBERT_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FREEZE_LAYERS = 6
WARMUP_RATIO = 0.1

torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------
# 1. Charger dataset
# -----------------------------
df = pd.read_excel(DATA_PATH)
df = df.dropna(subset=["nomenclature", "code"]).copy()

encoder = LabelEncoder()
df["label_enc"] = encoder.fit_transform(df["code"])

counts = df["label_enc"].value_counts()
rare_classes = counts[counts < 2].index

print("Classes rares supprimées (moins de 2 occurrences) :")
for lbl in rare_classes:
    code_original = encoder.inverse_transform([lbl])[0]
    occ = counts[lbl]
    print(f" - Code: {code_original}, Label encodé: {lbl}, Occurrences: {occ}")

df = df[~df["label_enc"].isin(rare_classes)].copy()

print("\nDistribution des classes après suppression :")
print(df["label_enc"].value_counts())

df["label"] = df["label_enc"].apply(lambda x: str(encoder.inverse_transform([x])[0]))

train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])
print("\nTaille train:", len(train_df), " | Taille val:", len(val_df))

# -----------------------------
# 2. SBERT embeddings
# -----------------------------
print("\nCalcul des embeddings SBERT...")
sbert = SentenceTransformer(SBERT_NAME, device=DEVICE)
train_sbert = sbert.encode(train_df["nomenclature"].tolist(), convert_to_numpy=True, batch_size=64, show_progress_bar=True)
val_sbert = sbert.encode(val_df["nomenclature"].tolist(), convert_to_numpy=True, batch_size=64, show_progress_bar=True)
sbert_dim = train_sbert.shape[1]

# -----------------------------
# 3. Tokenisation LLM
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)

def tokenize_batch(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=64)

train_tok = tokenize_batch(train_df["nomenclature"].tolist())
val_tok = tokenize_batch(val_df["nomenclature"].tolist())

# -----------------------------
# 4. Construire datasets HF
# -----------------------------
train_dataset = Dataset.from_dict({
    "input_ids": train_tok["input_ids"],
    "attention_mask": train_tok["attention_mask"],
    "labels": train_df["label"].tolist(),
    "sbert": [emb.tolist() for emb in train_sbert]
})

val_dataset = Dataset.from_dict({
    "input_ids": val_tok["input_ids"],
    "attention_mask": val_tok["attention_mask"],
    "labels": val_df["label"].tolist(),
    "sbert": [emb.tolist() for emb in val_sbert]
})

# -----------------------------
# 5. Fusion Model avec freeze partiel
# -----------------------------
class FusionClassifier(nn.Module):
    def __init__(self, llm_name, num_labels, sbert_dim, freeze_layers=0, dropout=0.2):
        super().__init__()
        self.llm = AutoModel.from_pretrained(llm_name)
        llm_hidden = self.llm.config.hidden_size
        fusion_dim = llm_hidden + sbert_dim

        # Détection des couches encoder (CamemBERT = roberta)
        if hasattr(self.llm, "roberta"):
            encoder_layers = self.llm.roberta.encoder.layer
        elif hasattr(self.llm, "bert"):
            encoder_layers = self.llm.bert.encoder.layer
        else:
            encoder_layers = None

        # Freeze partiel
        if encoder_layers is not None and freeze_layers > 0:
            for i in range(min(freeze_layers, len(encoder_layers))):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = False
            print(f"Gel des {min(freeze_layers, len(encoder_layers))} premières couches du LLM.")

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_labels)
        )

    def forward(self, input_ids=None, attention_mask=None, sbert=None, labels=None):
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS]
        sbert_tensor = torch.tensor(sbert, dtype=torch.float32, device=cls_emb.device)
        fused = torch.cat([cls_emb, sbert_tensor], dim=1)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

# -----------------------------
# 6. Collator
# -----------------------------
label_list = sorted(df["label"].unique())
label2id = {lbl: i for i, lbl in enumerate(label_list)}
id2label = {i: lbl for lbl, i in label2id.items()}

def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long)
    labels = torch.tensor([label2id[item["labels"]] for item in batch], dtype=torch.long)
    sbert = np.array([item["sbert"] for item in batch], dtype=np.float32)
    return {"input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sbert": sbert}

# -----------------------------
# 7. Metrics
# -----------------------------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"]
    }

# -----------------------------
# 8. Training + Scheduler
# -----------------------------
num_labels = len(label_list)
model = FusionClassifier(LLM_NAME, num_labels=num_labels, sbert_dim=sbert_dim, freeze_layers=FREEZE_LAYERS).to(DEVICE)

training_args = TrainingArguments(
    output_dir="./outputs_fusion",
    num_train_epochs=6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    report_to=[]
)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=training_args.learning_rate,
    weight_decay=training_args.weight_decay
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None)
)

# Scheduler (linear warmup)
total_train_steps = int(len(train_dataset) / training_args.per_device_train_batch_size) * training_args.num_train_epochs
num_warmup_steps = int(WARMUP_RATIO * total_train_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=total_train_steps
)
trainer.optimizers = (optimizer, scheduler)

trainer.train()

metrics = trainer.evaluate()
print("Final metrics:", metrics)

# -----------------------------
# 9. Sauvegarde
# -----------------------------
trainer.save_model("./outputs_fusion/best_model")
joblib.dump(label2id, "./outputs_fusion/label2id.joblib")
joblib.dump(id2label, "./outputs_fusion/id2label.joblib")

# -----------------------------
# 10. Inference
# -----------------------------
def predict_codes(nomenclatures):
    sbert_emb = sbert.encode(nomenclatures, convert_to_numpy=True, batch_size=64)
    tok = tokenize_batch(nomenclatures)
    inputs = {
        "input_ids": torch.tensor(tok["input_ids"], dtype=torch.long, device=DEVICE),
        "attention_mask": torch.tensor(tok["attention_mask"], dtype=torch.long, device=DEVICE),
        "sbert": sbert_emb
    }
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
    return [id2label[p] for p in preds]

# Exemple de test
examples = ["comptabilite", "avocats", "recouvrement"]
print("Predictions:", predict_codes(examples))