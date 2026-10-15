from wands.retrieval.bm25 import BM25Retriever
from wands.retrieval.dense import DenseRetriever

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class HybridRetriever:
    def __init__(self, rrf_k=100, dense_model=DEFAULT_MODEL):
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(model_name=dense_model)
        self.rrf_k = rrf_k

    def search(self, query, k=10):
        # Pull large candidate sets from both retrievers
        bm25_results = self.bm25.search(query, k=100)
        dense_results = self.dense.search(query, k=100)

        # Compute RRF scores — rank is 1-indexed
        scores = {}
        for rank, pid in enumerate(bm25_results, start=1):
            scores[pid] = scores.get(pid, 0) + 1 / (self.rrf_k + rank)
        for rank, pid in enumerate(dense_results, start=1):
            scores[pid] = scores.get(pid, 0) + 1 / (self.rrf_k + rank)

        # Sort by fused score descending, return top k
        ranked = sorted(scores, key=scores.get, reverse=True)
        return ranked[:k]
