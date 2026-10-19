from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed"


class BM25Retriever:
    def __init__(self):
        df = pd.read_parquet(PROCESSED_DIR / "products.parquet")
        self.product_ids = df["product_id"].tolist()

        # Tokenize: lowercase and split on whitespace
        tokenized = [str(text).lower().split() for text in df["search_text"]]
        self.index = BM25Okapi(tokenized)

    def search(self, query, k=10):
        tokens = query.lower().split()
        scores = self.index.get_scores(tokens)
        # Get top-k indices sorted by score descending
        top_indices = scores.argsort()[::-1][:k]
        return [self.product_ids[i] for i in top_indices]

    def get_scores(self, query, product_ids):
        """Return BM25 scores for specific product IDs."""
        tokens = query.lower().split()
        all_scores = self.index.get_scores(tokens)
        pid_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}
        return {pid: float(all_scores[pid_to_idx[pid]]) for pid in product_ids if pid in pid_to_idx}
