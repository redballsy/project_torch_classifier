import os, certifi
import pandas as pd
import numpy as np
import joblib
import fasttext
from sentence_transformers import SentenceTransformer

# ✅ Force correct certificate bundle for Hugging Face downloads
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

# -----------------------------
# 1. Paths
# -----------------------------
PATH_TRAIN = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\entrainer\entrainer2_propre.xlsx"
PATH_CORPUS = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\pretrain\corpus_variante_par_code.xlsx"
PATH_CITP   = r"C:\Users\Sy Savane Idriss\project_torch_classifier\torchTestClassifiers\data\pretrain\CITP_08.xlsx"

# -----------------------------
# 2. Load datasets
# -----------------------------
train_df  = pd.read_excel(PATH_TRAIN)
corpus_df = pd.read_excel(PATH_CORPUS)
citp_df   = pd.read_excel(PATH_CITP)

# Ensure codes are strings and clean text fields
for df in [train_df, corpus_df, citp_df]:
    df["code"] = df["code"].astype(str)
    df["nomenclature"] = df["nomenclature"].astype(str).str.replace("\n", " ").str.strip()

# -----------------------------
# 3. Load FastText + SBERT
# -----------------------------
ft_model = fasttext.load_model(
    r"C:\Users\Sy Savane Idriss\project_torch_classifier\models_fasttext\cc.fr.300.bin"
)

# ⚡ Use a smaller multilingual model to avoid huge downloads
sbert = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

# -----------------------------
# 4. Hybrid embedding function
# -----------------------------
def hybrid_embedding(text):
    if not isinstance(text, str):
        text = str(text)
    clean_text = text.replace("\n", " ").strip()
    ft_vec = ft_model.get_sentence_vector(clean_text)
    sbert_vec = sbert.encode(clean_text, convert_to_numpy=True)
    return np.concatenate([ft_vec, sbert_vec])

# -----------------------------
# 5. Compute embeddings
# -----------------------------
print("⚡ Computing train embeddings...")
train_embeddings  = np.stack(train_df["nomenclature"].apply(hybrid_embedding).values)

print("⚡ Computing corpus embeddings...")
corpus_embeddings = np.stack(corpus_df["nomenclature"].apply(hybrid_embedding).values)

# -----------------------------
# 6. Save cached embeddings + metadata
# -----------------------------
np.save("train_embeddings.npy", train_embeddings)
np.save("corpus_embeddings.npy", corpus_embeddings)
joblib.dump(train_df["code"].values, "train_labels.pkl")
joblib.dump(corpus_df, "corpus_df.pkl")
joblib.dump(citp_df, "citp_df.pkl")

print("✅ Embeddings cached successfully")