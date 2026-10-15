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
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=256)
        embeddings = np.array(embeddings, dtype="float32")

        # Normalize for cosine similarity (inner product on normalized vectors = cosine)
        faiss.normalize_L2(embeddings)

        # Build FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized = cosine
        self.index.add(embeddings)

    def search(self, query, k=10):
        # Encode query and normalize
        q_vec = self.model.encode([query])
        q_vec = np.array(q_vec, dtype="float32")
        faiss.normalize_L2(q_vec)

        # Search the index
        _, indices = self.index.search(q_vec, k)
        return [self.product_ids[i] for i in indices[0]]
