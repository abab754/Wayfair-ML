"""Build feature vectors for (query, product) pairs for the LTR reranker."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed"

FEATURE_NAMES = [
    "bm25_score",
    "dense_score",
    "query_token_overlap",
    "product_name_overlap",
    "category_match",
    "average_rating",
    "review_count",
    "rating_count",
    "search_text_len",
    "description_len",
]


class FeatureBuilder:
    def __init__(self, bm25_retriever, dense_retriever):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever

        # Load product data as DataFrame, indexed by product_id for fast lookup
        df = pd.read_parquet(PROCESSED_DIR / "products.parquet")
        df = df.set_index("product_id")
        # Pre-fill NaNs to avoid repeated checks
        df["average_rating"] = df["average_rating"].fillna(0.0)
        df["review_count"] = df["review_count"].fillna(0.0)
        df["rating_count"] = df["rating_count"].fillna(0.0)
        df["product_name"] = df["product_name"].fillna("")
        df["product_class"] = df["product_class"].fillna("")
        df["product_description"] = df["product_description"].fillna("")
        df["search_text"] = df["search_text"].fillna("")
        self.df = df

    def _product_features(self, pid, query_tokens):
        """Compute static + text-overlap features for one product."""
        try:
            row = self.df.loc[pid]
        except KeyError:
            return {name: 0.0 for name in FEATURE_NAMES if name not in ("bm25_score", "dense_score")}

        search_tokens = set(str(row["search_text"]).lower().split())
        name_tokens = set(str(row["product_name"]).lower().split())
        product_class_lower = str(row["product_class"]).lower()

        if query_tokens:
            token_overlap = len(query_tokens & search_tokens) / len(query_tokens)
            name_overlap = len(query_tokens & name_tokens) / len(query_tokens)
        else:
            token_overlap = 0.0
            name_overlap = 0.0

        cat_match = 1.0 if any(t in product_class_lower for t in query_tokens) else 0.0

        return {
            "query_token_overlap": token_overlap,
            "product_name_overlap": name_overlap,
            "category_match": cat_match,
            "average_rating": float(row["average_rating"]),
            "review_count": float(row["review_count"]),
            "rating_count": float(row["rating_count"]),
            "search_text_len": len(str(row["search_text"]).split()),
            "description_len": len(str(row["product_description"]).split()),
        }

    def build_features(self, query, candidate_ids):
        """Build feature matrix for a query and its candidate product IDs.

        Returns a list of dicts, one per candidate, in the same order as candidate_ids.
        """
        query_tokens = set(query.lower().split())

        # Batch-get retriever scores for all candidates at once
        bm25_scores = self.bm25.get_scores(query, candidate_ids)
        dense_scores = self.dense.get_scores(query, candidate_ids)

        features = []
        for pid in candidate_ids:
            feat = self._product_features(pid, query_tokens)
            feat["bm25_score"] = bm25_scores.get(pid, 0.0)
            feat["dense_score"] = dense_scores.get(pid, 0.0)
            features.append(feat)

        return features
