# Sprint Progress

## Sprint 0 — Setup

Scaffolded repo structure, pyproject.toml, Makefile, .gitignore, and CI config. Chose JSON over MLflow for experiment tracking. Set up uv with Python 3.12. `make setup && make test && make lint` all pass.

## Sprint 1 — Data pipeline + eval harness + BM25 baseline

Built PySpark data processing job that cleans raw WANDS TSVs and produces parquet files with a combined `search_text` field (name + class + category hierarchy + features). Loaded data into BigQuery and wrote 5 analysis queries covering label distribution, query classes, products per category, hardest queries, and rating/review stats.

Implemented custom evaluation metrics (NDCG@k, MRR, Recall@k) with 11 unit tests on hand-computed examples. Built an eval runner that takes any retriever with a `search(query, k)` interface and produces per-query-class metric breakdowns with latency tracking.

BM25 baseline using `rank_bm25.BM25Okapi`: NDCG@10=0.6098, MRR=0.7777, Recall@100=0.3066 on 72 val queries. EDA notebook identified failure categories: Bookcases, Cabinet/Drawer Pulls, Candle Holders, Desks, Vanities all scored NDCG@10=0.0 — queries where product text lacks the exact keywords users search for.

**Surprise:** BM25 is remarkably strong on WANDS. Most queries are product-type queries (e.g., "dining chairs") where keyword matching works well. The hard cases are semantic/style queries where lexical overlap breaks down.

## Sprint 2 — Dense + hybrid retrieval

Evaluated zero-shot `all-MiniLM-L6-v2` with FAISS: NDCG@10=0.5943, MRR=0.7997. Dense retrieval has higher MRR than BM25 (better at surfacing the first relevant result) but lower NDCG@10 and Recall@100 overall. It improved several BM25-failure categories (Bookcases 0.0->0.33, Desks 0.0->0.18).

Attempted fine-tuning with TripletLoss and BM25-mined hard negatives (~14.5K triplets, 2 epochs). Model degraded significantly (NDCG@10=0.2865). Root cause: only 336 training queries provide insufficient diversity for contrastive learning. Documented as honest result.

Hybrid retrieval via Reciprocal Rank Fusion (RRF) combining BM25 + zero-shot dense: NDCG@10=0.6709 (+10.0% over BM25), MRR=0.8379, Recall@100=0.3152. Swept RRF k parameter (10–150) on val; k=100 was optimal. Built full ablation table sliced by query class across all 4 models.

**Surprise:** The complementarity of BM25 and dense retrieval is striking. Categories where BM25 scored 0.0 (Bookcases, Desks, Vanities) all improved with hybrid. The biggest single-category gain was Beds: 0.61 → 1.00. A few regressions exist (Sofas, Area Rugs) worth investigating in Sprint 3.

## Sprint 3 — Learning-to-rank reranker

Added `get_scores()` methods to BM25 and Dense retrievers so features can be extracted per candidate. Built a 10-feature LambdaMART reranker (LightGBM lambdarank) with: bm25_score, dense_score, query_token_overlap, product_name_overlap, category_match, average_rating, review_count, rating_count, search_text_len, description_len.

Feature generation was split into 3 subprocess stages to avoid OOM on a 18GB local machine: (1) BM25 candidates + scores, (2) dense scores, (3) text/metadata features. Stages 1-3 were run on a GCP e2-highmem-2 VM (16GB RAM, ~$0.10 total cost). Training produced 56K train rows and 12K val rows across 72 val queries.

Results on val (72 queries):

| Model | NDCG@10 | MRR | Recall@100 |
|---|---|---|---|
| BM25 | 0.6098 | 0.7777 | 0.3066 |
| Hybrid RRF | 0.6709 | 0.8379 | 0.3152 |
| **Hybrid + LTR** | **0.7233** | **0.8417** | **0.8017** |

Feature importance (gain): product_name_overlap (5548) >> dense_score (2399) >> query_token_overlap (1692) >> bm25_score (1633) >> category_match (945) >> description_len (601) >> search_text_len (476) >> review_count (395) >> rating_count (310) >> average_rating (167).

**Surprise:** product_name_overlap dominates feature importance — whether query terms appear in the product name matters far more than raw retriever scores. Recall@100 jumped from 0.32 to 0.80 because the LTR pipeline uses a union of BM25+dense top-100 candidates (~170 per query), vastly expanding the candidate pool. Latency not measured locally due to OOM; will measure when deploying (Sprint 5).
