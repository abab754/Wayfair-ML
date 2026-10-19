from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed"

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class DenseRetriever:
    def __init__(self, model_name=DEFAULT_MODEL):
        df = pd.read_parquet(PROCESSED_DIR / "products.parquet")
        self.product_ids = df["product_id"].tolist()
        texts = df["search_text"].fillna("").tolist()

        # Load the sentence-transformer model
        self.model = SentenceTransformer(model_name)

        # Encode all products into vectors
        print(f"Encoding {len(texts)} products with {model_name}...")
        embeddings = self.model.encode(
            texts, show_progress_bar=True, batch_size=128, convert_to_numpy=True
        )
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)

        # Build FAISS index and free the raw array
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized = cosine
        self.index.add(embeddings)
        del embeddings

    def search(self, query, k=10):
        # Encode query and normalize
        q_vec = self.model.encode([query])
        q_vec = np.array(q_vec, dtype="float32")
        faiss.normalize_L2(q_vec)

        # Search the index
        _, indices = self.index.search(q_vec, k)
        return [self.product_ids[i] for i in indices[0]]

    def get_scores(self, query, product_ids):
        """Return cosine similarity scores for specific product IDs."""
        q_vec = self.model.encode([query])
        q_vec = np.array(q_vec, dtype="float32")
        faiss.normalize_L2(q_vec)

        pid_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}
        valid_pids = [pid for pid in product_ids if pid in pid_to_idx]
        indices = [pid_to_idx[pid] for pid in valid_pids]
        # Reconstruct vectors and compute dot product (= cosine since normalized)
        vecs = np.array([self.index.reconstruct(i) for i in indices], dtype="float32")
        scores = (q_vec @ vecs.T).flatten()
        return {pid: float(scores[j]) for j, pid in enumerate(valid_pids)}
