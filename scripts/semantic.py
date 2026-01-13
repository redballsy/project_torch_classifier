import os, certifi
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()
import os
import torch
import joblib
import fasttext
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# 1. SETUP & MODELS
# ============================================
BASE_DIR = r"C:\Users\Sy Savane Idriss\project_torch_classifier"
FT_MODEL_PATH = os.path.join(BASE_DIR, "models_fasttext", "cc.fr.300.bin")

print("⚡ Loading Embedding Models...")
ft_model = fasttext.load_model(FT_MODEL_PATH)
sbert = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

# ============================================
# 2. HYBRID ENCODING FUNCTION
# ============================================
def get_hybrid_vector(text):
    clean_text = str(text).replace("\n", " ").strip().lower()
    # FastText (300 dim)
    ft_vec = ft_model.get_sentence_vector(clean_text)
    # SBERT (384 dim)
    sbert_vec = sbert.encode(clean_text, convert_to_numpy=True)
    # Hybrid (684 dim)
    return np.concatenate([ft_vec, sbert_vec]).reshape(1, -1)

# ============================================
# 3. SEARCH ENGINE
# ============================================
def semantic_search(query_text, corpus_embeddings, corpus_metadata, top_k=5):
    """
    Finds the most similar entries in the corpus for a given query.
    """
    # 1. Encode the query
    query_vec = get_hybrid_vector(query_text)
    
    # 2. Calculate Cosine Similarity against the whole corpus
    similarities = cosine_similarity(query_vec, corpus_embeddings).flatten()
    
    # 3. Get top K indices
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "text": corpus_metadata.iloc[idx]['nomenclature'],
            "code": corpus_metadata.iloc[idx]['code'],
            "score": f"{similarities[idx]:.4f}"
        })
    return results

# ============================================
# 4. EXECUTION
# ============================================
if __name__ == "__main__":
    # Load your pre-computed corpus embeddings (from your cache script)
    # This file should have shape (N, 684)
    corpus_embeddings = np.load("corpus_embeddings.npy")
    corpus_df = joblib.load("corpus_df.pkl") # Contains 'nomenclature' and 'code'

    query = "Ingénieur en intelligence artificielle"
    matches = semantic_search(query, corpus_embeddings, corpus_df)

    print(f"\n🔍 Query: '{query}'")
    print("-" * 50)
    for i, res in enumerate(matches):
        print(f"{i+1}. [{res['score']}] Code: {res['code']} | {res['text']}")