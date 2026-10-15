# WANDS Hard-Query Search Engine

Search relevance system built on [Wayfair's WANDS dataset](https://github.com/wayfair/WANDS), targeting hard queries: long-tail, vague, style-based, and price-constrained searches.

## Results (val split, 72 queries)

| Model | NDCG@10 | MRR | Recall@100 | p50 Latency | p95 Latency |
|-------|---------|-----|------------|-------------|-------------|
| BM25 (baseline) | 0.6098 | 0.7777 | 0.3066 | 22ms | 47ms |
| Dense zero-shot (MiniLM-L6-v2) | 0.5943 | 0.7997 | 0.2735 | 7ms | 839ms |
| Dense fine-tuned (TripletLoss) | 0.2865 | 0.5323 | 0.1359 | 9ms | 24ms |
| **Hybrid RRF (BM25 + dense)** | **0.6709** | **0.8379** | **0.3152** | - | - |

### Key findings

- **BM25 is a strong baseline** for this dataset — keyword matching works well for product-type queries.
- **Dense zero-shot** edges out BM25 on MRR (better at surfacing the first relevant result) but lags on NDCG and recall overall.
- **Fine-tuning degraded the dense model.** With only 336 training queries, the bi-encoder overfits under contrastive learning. Documented as honest result.
- **Hybrid RRF is the best retriever.** +10.0% NDCG@10 over BM25, +7.7% MRR. Reciprocal Rank Fusion combines BM25's keyword precision with dense retrieval's semantic understanding. Tuned k=100 on val (swept k=10–150).
- **BM25 weakest categories:** Bookcases (0.0), Cabinet/Drawer Pulls (0.0), Candle Holders (0.0), Desks (0.0), Vanities (0.0) — queries where lexical overlap is poor.
- **Dense zero-shot strongest improvements over BM25:** Bookcases (0.0 -> 0.33), Desks (0.0 -> 0.18), Beds (0.61 -> 0.81) — semantic matching helps when exact keywords don't appear.

## Dataset

- **Source:** [Wayfair WANDS](https://github.com/wayfair/WANDS)
- ~42,994 products, 480 queries, ~233K human relevance judgments
- Graded relevance: Exact=2, Partial=1, Irrelevant=0
- Split by query_id (seed=42): 336 train / 72 val / 72 test

## Architecture

```
Query -> BM25 Retriever -----> [Hybrid RRF Fusion] -> Reranker -> Results
     \-> Dense Retriever --/
```

## Technical decisions

### Sprint 0
- **Experiment tracking:** JSON files in `results/` over MLflow — simpler for solo project, version-controllable, no server needed.
- **Package manager:** uv with Python 3.12

### Sprint 1 — Data pipeline + eval harness + BM25 baseline
- **Data processing:** PySpark job cleans raw WANDS TSVs, builds `search_text` field (name + class + category hierarchy + features), writes parquet.
- **BigQuery:** Loaded processed data for SQL analysis (label distribution, query classes, products per category, hardest queries, rating/review stats).
- **Eval harness:** Custom NDCG@k, MRR, Recall@k implementations with 11 unit tests on hand-computed examples. Runner evaluates any retriever with `search(query, k) -> list[product_id]`, outputs per-query-class slices and p50/p95 latency.
- **BM25:** `rank_bm25.BM25Okapi` over tokenized `search_text`. Whitespace tokenization, lowercased.
- **Train/val/test split:** By query_id to prevent label leakage. 70/15/15 ratio, fixed seed=42, saved to `splits.json`.

### Sprint 2 — Dense + hybrid retrieval
- **Dense retriever:** `all-MiniLM-L6-v2` sentence-transformer + FAISS `IndexFlatIP` with L2-normalized embeddings (cosine similarity via inner product).
- **Fine-tuning attempt:** TripletLoss with hard negatives mined from BM25. ~14.5K triplets, 2 epochs. Model degraded due to insufficient query diversity.
- **Hybrid RRF:** Reciprocal Rank Fusion combining top-100 from each retriever. Tuned k parameter on val (swept 10–150, best k=100). Products found by both retrievers get boosted scores.

## Project structure

```
src/wands/
  data/       # PySpark processing, splits, BigQuery loader
  eval/       # metrics (NDCG, MRR, Recall), eval runner
  retrieval/  # bm25.py, dense.py, hybrid.py
  rerank/     # (Sprint 3)
  llm/        # (Sprint 4)
  serving/    # (Sprint 5)
tests/        # unit tests for metrics
results/      # experiment result JSONs
sql/          # BigQuery analysis queries
notebooks/    # EDA (Sprint 1 BM25 failure analysis)
docs/         # decisions, progress notes
```

## Quick start

```bash
make setup        # create venv and install deps
make data         # download WANDS + run processing pipeline
make eval MODEL=bm25   # run eval harness
make test         # pytest
make lint         # ruff linter
```
