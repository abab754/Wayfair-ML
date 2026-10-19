# CLAUDE.md — WANDS Hard-Query Search Engine

## Project goal

Build a production-style **search relevance system on Wayfair's public WANDS dataset** that measurably improves results on *hard* queries: long-tail, vague, style-based, and price-constrained searches like "something cozy for a small reading nook under $300".

The project exists to demonstrate the full ML lifecycle for a Wayfair Machine Learning Scientist application: data pipeline → baseline → modeling → LLM layer → deployment → measured business impact → write-up.

**The single most important rule:** every change is judged against the eval harness. If it doesn't move NDCG (or it trades off latency, cost, or quality), we say so in numbers.

## Dataset

- Source: https://github.com/wayfair/WANDS (clone or download into `data/raw/`; never commit raw data)
- Files (tab-separated):
  - `product.csv`: ~43K products (name, class, category hierarchy, description, features, ratings)
  - `query.csv`: 480 real customer queries with a query class
  - `label.csv`: ~233K human (query, product) judgments: `Exact`, `Partial`, `Irrelevant`
- Graded relevance for metrics: `Exact = 2`, `Partial = 1`, `Irrelevant = 0`. Use this everywhere; it's defined once in `src/wands/eval/metrics.py`.

## Tech stack

| Area | Tools |
|---|---|
| Language | Python 3.11 |
| Data | PySpark, pandas, BigQuery (SQL) |
| Retrieval | `rank_bm25` or Pyserini (BM25), sentence-transformers (dense), FAISS |
| Modeling | PyTorch, LightGBM (LambdaMART ranker) |
| LLM | An LLM API for query parsing and grounded explanations |
| Serving | FastAPI, Docker, GCP Cloud Run |
| Quality | pytest, ruff, mypy (light), GitHub Actions |
| Tracking | MLflow or plain JSON results in `results/` (pick one in Sprint 0, keep it) |

## Repo structure

```
.
├── CLAUDE.md
├── README.md                 # results table first, then how to run
├── pyproject.toml
├── Makefile                  # common commands (see below)
├── Dockerfile
├── .github/workflows/ci.yml
├── configs/                  # YAML configs per experiment
├── data/
│   ├── raw/                  # WANDS files (gitignored)
│   └── processed/            # parquet outputs (gitignored)
├── notebooks/                # exploration only; nothing imported from here
├── results/                  # metric JSONs + ablation tables (committed)
├── src/wands/
│   ├── data/                 # loading, cleaning, Spark jobs, BigQuery loaders
│   ├── eval/                 # metrics, splits, eval runner, slicing by query type
│   ├── retrieval/            # bm25.py, dense.py, hybrid.py
│   ├── rerank/               # features.py, ltr.py, cross_encoder.py
│   ├── llm/                  # query_parser.py, explainer.py, prompts/
│   ├── serving/              # FastAPI app, search pipeline orchestration
│   └── utils/
├── tests/
└── docs/
    └── writeup.md            # the final technical write-up
```

## Commands

Keep these working at all times. Add to the Makefile as sprints progress.

```
make setup        # install deps (uv or pip)
make data         # download WANDS + run processing pipeline
make eval MODEL=bm25          # run eval harness for a retriever/pipeline
make test         # pytest
make lint         # ruff + mypy
make serve        # run FastAPI locally
make docker       # build + run container
```

## Working rules (for Claude and for me)

1. **Never touch the test split after Sprint 1.** Queries are split once, by query id, into train/val/test with a fixed seed and saved to `data/processed/splits.json`. Tune on val; report test only for final numbers.
2. **No label leakage.** Training pairs for the retriever and ranker come only from train-split queries.
3. **Every experiment writes a result file** to `results/<sprint>_<name>.json` with: NDCG@10, MRR, Recall@100, per-query-class breakdown, p50/p95 latency, and config used.
4. **Baselines stay runnable.** BM25 must always reproduce its Sprint 1 number.
5. **LLM outputs must be grounded.** Explanations may only use the product's own attributes. Track hallucination rate.
6. **Small, reviewable changes.** One concern per PR. Tests come with new logic in `src/`.
7. **Notebooks are scratch.** Anything reused moves into `src/wands/`.
8. **Secrets go in env vars** (`.env` gitignored, `.env.example` committed). Never hardcode API keys or GCP project IDs.
9. When unsure about a design choice, write the tradeoff in a short note in `docs/decisions.md` rather than silently picking.
10. Don't just write the code without asking. I want to be writing majority of the logic/code and I want to fully understand what's going on at each step. Provide me some resources when I ask what's going on
11. Let's test everything we worked on after each sprint -> commit -> push -> PR with clean description

## Sprints

Each sprint ends with: passing CI, an updated `results/` file, and a one-paragraph entry in `docs/progress.md` (what changed, the numbers, and what surprised me).

### Sprint 0 — Setup (1–2 days)
- [ ] Repo scaffold matching the structure above; `pyproject.toml`, Makefile, `.gitignore`
- [ ] ruff + pytest + GitHub Actions running on push
- [ ] Download WANDS into `data/raw/`; load all three files and sanity-check counts
- [ ] Pick experiment tracking (MLflow vs JSON) and record it in `docs/decisions.md`

**Done when:** `make setup && make test && make lint` pass in CI on an empty-but-real project.

### Sprint 1 — Data pipeline + eval harness + BM25 baseline (week 1–2)
- [ ] PySpark job: clean products, build a `search_text` field (name + class + category + features), write parquet
- [ ] Load processed tables into BigQuery; write 3–5 analysis queries in `sql/` (label distribution, query classes, products per category)
- [ ] Fixed train/val/test query split (`splits.json`)
- [ ] `eval/metrics.py`: NDCG@k, MRR, Recall@k with unit tests on hand-computed examples
- [ ] `eval/runner.py`: takes any retriever implementing `search(query, k) -> list[product_id]`, outputs the standard result JSON including per-query-class slices
- [ ] BM25 baseline over `search_text`
- [ ] EDA notebook: which query classes BM25 fails on, with 10 concrete failure examples

**Done when:** `make eval MODEL=bm25` produces a reproducible result file, and I can name the query types where BM25 is weakest.

### Sprint 2 — Dense + hybrid retrieval (week 3)
- [ ] Off-the-shelf sentence-transformer + FAISS index; evaluate zero-shot
- [ ] Fine-tune the bi-encoder in PyTorch on train-split pairs (Exact/Partial as positives) with hard negatives mined from BM25
- [ ] Hybrid retrieval: reciprocal rank fusion of BM25 + dense; tune the fusion on val
- [ ] Ablation table: BM25 vs zero-shot dense vs fine-tuned dense vs hybrid, sliced by query class

**Done when:** hybrid beats BM25 on val NDCG@10, and the ablation table is in `results/`.

### Sprint 3 — Learning-to-rank reranker (week 4)
- [ ] Feature builder: BM25 score, dense score, category match, attribute overlap, price, rating, review count, text length
- [ ] LightGBM LambdaMART ranker trained on train-split candidates from the hybrid retriever
- [ ] (Optional) cross-encoder reranker; compare quality vs latency
- [ ] Feature importance plot + short interpretation

**Done when:** retriever + reranker beats hybrid alone on val, with latency measured for each stage.

### Sprint 4 — LLM query understanding + grounded explanations (week 5)
- [ ] `llm/query_parser.py`: turns a raw query into structured constraints (price range, room, style, color, material, product type) using JSON output with a schema
- [ ] Use parsed constraints as filters/boosts in the pipeline
- [ ] Router: only send queries to the LLM parser when they look "hard" (length, price words, style words); measure the effect on simple queries
- [ ] `llm/explainer.py`: one-line "why this matches" per top result, built only from product attributes
- [ ] Hallucination check: automated test that flags claims not supported by product fields; report the rate on 100 sampled explanations
- [ ] Cache LLM calls; record cost per 1,000 queries

**Done when:** LLM layer improves NDCG on hard query slices without hurting easy ones (or the router fixes it), and cost + hallucination rate are reported.

### Sprint 5 — Ship it (week 6)
- [ ] FastAPI `/search` endpoint running the full pipeline; `/health` endpoint
- [ ] Dockerfile (slim image, model artifacts loaded at startup)
- [ ] Deploy to GCP Cloud Run; document the deploy steps in README
- [ ] CI eval gate: GitHub Actions runs eval on a small fixed query subset and fails if NDCG@10 drops more than a set threshold
- [ ] Simple search UI (Streamlit or a minimal HTML page) showing results + explanations
- [ ] Load test: p50/p95 latency for the deployed service

**Done when:** a public demo URL works, and CI blocks a deliberately worse model.

### Sprint 6 — Analysis, business framing, write-up (week 7–8)
- [ ] Final test-set numbers for every pipeline stage (run once)
- [ ] Error analysis: 20 remaining failures, grouped into causes
- [ ] Business framing: translate NDCG lift on long-tail queries into estimated impact with clearly stated assumptions
- [ ] `docs/writeup.md`: problem, data, method, ablations, latency/cost, failures, next steps (written for a technical reader, with a non-technical summary at top)
- [ ] README rewritten: results table first, demo link, architecture diagram, how to run
- [ ] 2-minute demo video

**Done when:** someone at Wayfair could read the README in 60 seconds and understand what improved, by how much, and where it still fails.

## Current sprint

**Sprint 3 complete.** Sprints 0–3 done. Next: Sprint 4 (LLM query understanding).
