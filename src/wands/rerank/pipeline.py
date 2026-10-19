"""Reranking pipeline that wraps retrieval + LambdaMART for eval compatibility."""

from pathlib import Path

import lightgbm as lgb
import numpy as np

from wands.rerank.features import FEATURE_NAMES, FeatureBuilder
from wands.retrieval.hybrid import HybridRetriever

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models" / "ltr"


class LTRPipeline:
    """Retriever-compatible wrapper: hybrid retrieval → feature extraction → LambdaMART rerank."""

    def __init__(self, bm25, dense, model_path=None):
        self.hybrid = HybridRetriever(bm25=bm25, dense=dense)
        self.feature_builder = FeatureBuilder(bm25, dense)
        model_path = model_path or str(MODEL_DIR / "lambdamart.txt")
        self.model = lgb.Booster(model_file=model_path)

    def search(self, query, k=10):
        # Step 1: Retrieve candidates via hybrid
        candidates = self.hybrid.search(query, k=100)

        # Step 2: Build features
        features = self.feature_builder.build_features(query, candidates)
        X = np.array([[f[name] for name in FEATURE_NAMES] for f in features], dtype="float32")

        # Step 3: Score with LambdaMART and rerank
        scores = self.model.predict(X)
        ranked_indices = np.argsort(-scores)

        return [candidates[i] for i in ranked_indices[:k]]
