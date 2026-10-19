"""Run LTR pipeline evaluation."""
import json
import sys
import os

# Suppress progress bars
os.environ["TQDM_DISABLE"] = "1"

from wands.retrieval.bm25 import BM25Retriever
from wands.retrieval.dense import DenseRetriever
from wands.rerank.pipeline import LTRPipeline
from wands.eval.runner import evaluate, save_results

print("Loading BM25...", flush=True)
bm25 = BM25Retriever()

print("Loading Dense...", flush=True)
dense = DenseRetriever()

print("Building LTR pipeline...", flush=True)
pipeline = LTRPipeline(bm25=bm25, dense=dense)

print("Running eval on val split (72 queries)...", flush=True)
results = evaluate(pipeline, split="val")
results["model"] = "hybrid_ltr_reranker"

print("\n=== RESULTS ===", flush=True)
print(json.dumps(results["overall"], indent=2), flush=True)
print(f"\nLatency p50={results['latency']['p50']:.3f}s p95={results['latency']['p95']:.3f}s", flush=True)

print("\nBy query class:", flush=True)
for qc, metrics in sorted(results["by_query_class"].items()):
    print(f"  {qc}: NDCG@10={metrics['ndcg@10']:.4f} MRR={metrics['mrr']:.4f} R@100={metrics['recall@100']:.4f} (n={metrics['count']})", flush=True)

save_results(results, "sprint3_ltr_reranker")
print("\nDone!", flush=True)
