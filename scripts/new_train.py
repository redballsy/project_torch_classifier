import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import fasttext
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# 1. Load datasets
# -----------------------------
corpus = pd.read_excel(
    r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\pretrain\corpus_variante_par_code.xlsx"
)
citp = pd.read_excel(
    r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\pretrain\CITP_08.xlsx"
)
train_data = pd.read_excel(
    r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\entrainer2_propre.xlsx"
)

# Convert code columns to string
for df in [corpus, citp, train_data]:
    df["code"] = df["code"].astype(str)

# -----------------------------
# 2. Load FastText + SBERT
# -----------------------------
ft_model = fasttext.load_model( r"C:\Users\Sy Savane Idriss\project_torch_classifier\models_fasttext\cc.fr.300.bin"
) # French FastText model
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# 3. Hybrid embeddings
# -----------------------------
def hybrid_embedding(text):
    ft_vec = ft_model.get_sentence_vector(text)
    sbert_vec = sbert.encode(text)
    return np.concatenate([ft_vec, sbert_vec])

# Generate embeddings for training and corpus
train_data["embedding"] = train_data["nomenclature"].apply(hybrid_embedding)
corpus["embedding"] = corpus["nomenclature"].apply(hybrid_embedding)

# -----------------------------
# 4. Prepare supervised dataset
# -----------------------------
X = np.stack(train_data["embedding"].values)
y = train_data["code"].values

encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y_enc, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)

# Split 70/30
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# -----------------------------
# 5. Define classifier
# -----------------------------
class HybridClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

input_dim = X.shape[1]
num_classes = len(encoder.classes_)
model_nn = HybridClassifier(input_dim, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_nn.parameters(), lr=1e-3)

# -----------------------------
# 6. Training loop
# -----------------------------
epochs = 10
for epoch in range(epochs):
    # --- Train ---
    model_nn.train()
    train_loss, train_preds, train_labels = 0, [], []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model_nn(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_preds.extend(outputs.argmax(1).tolist())
        train_labels.extend(yb.tolist())
    train_acc = accuracy_score(train_labels, train_preds)

    # --- Validation ---
    model_nn.eval()
    val_loss, val_preds, val_labels = 0, [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            outputs = model_nn(xb)
            loss = criterion(outputs, yb)
            val_loss += loss.item()
            val_preds.extend(outputs.argmax(1).tolist())
            val_labels.extend(yb.tolist())
    val_acc = accuracy_score(val_labels, val_preds)

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")

print("\n=== Classification Report (Validation) ===")
print(classification_report(val_labels, val_preds, target_names=encoder.classes_))

# -----------------------------
# 7. Prediction with fallback + mode flag
# -----------------------------
def predict_job(title, top_k=3, threshold=0.6):
    model_nn.eval()
    with torch.no_grad():
        emb = hybrid_embedding(title)
        emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
        logits = model_nn(emb_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        top_indices = probs.argsort()[-top_k:][::-1]
        best_prob = probs[top_indices[0]]
        
        if best_prob >= threshold:
            # Classifier confident → return codes + official nomenclature
            results = []
            for i in top_indices:
                code = encoder.classes_[i]
                if code in citp["code"].values:
                    label = citp.loc[citp["code"] == code, "nomenclature"].iloc[0]
                else:
                    label = "Unknown"
                results.append((f"{label} ({code})", round(probs[i], 3)))
            return {"mode": "classifier", "results": results}
        else:
            # Fallback: semantic search in corpus
            query_emb = torch.tensor(emb, dtype=torch.float32)
            scores = []
            for _, row in corpus.iterrows():
                score = util.cos_sim(query_emb, torch.tensor(row["embedding"], dtype=torch.float32)).item()
                scores.append((row["nomenclature"], row["code"], round(score, 3)))
            scores = sorted(scores, key=lambda x: x[2], reverse=True)[:top_k]
            return {"mode": "semantic_search", "results": scores}

# -----------------------------
# 8. Save trained model
# -----------------------------
save_path = r"C:\Users\Sy Savane Idriss\project_torch_classifier\models\citp_classifier_model.pth"
torch.save(model_nn.state_dict(), save_path)
print(f"✅ Model saved to {save_path}")

# -----------------------------
# 9. Example usage
# -----------------------------
print("\n=== Example Prediction ===")
print(predict_job("réparateur de dent", top_k=3))