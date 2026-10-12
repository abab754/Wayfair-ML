# Design Decisions

## Sprint 0

### Experiment tracking: JSON over MLflow

We use plain JSON files in `results/` instead of MLflow.

- **Why:** Solo project, simpler to version-control and diff. No server to run.
- **Tradeoff:** No built-in experiment comparison UI. If we want one later, we can add a notebook or small script to read the JSON files.
- **Format:** Each experiment writes to `results/<sprint>_<name>.json` with NDCG@10, MRR, Recall@100, per-query-class breakdown, latency, and config.
