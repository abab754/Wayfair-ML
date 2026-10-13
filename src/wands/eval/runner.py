import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from wands.eval.metrics import RELEVANCE, mrr, ndcg_at_k, recall_at_k

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"


def load_eval_data(split):
    with open(f"{PROCESSED_DIR}/splits.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    query_df = pd.read_parquet(f"{PROCESSED_DIR}/queries.parquet")
    label_df = pd.read_parquet(f"{PROCESSED_DIR}/labels.parquet")

    query_ids = data[split]
    filtered_query_df = query_df[query_df["query_id"].isin(query_ids)]
    filtered_label_df = label_df[label_df["query_id"].isin(query_ids)]

    return (filtered_query_df, filtered_label_df)


def build_relevance(labels_df, query_id):
    res = {}
    filtered_df = labels_df.loc[labels_df["query_id"] == query_id]
    for index, row in filtered_df.iterrows():
        res[row["product_id"]] = RELEVANCE[row["label"]]

    return res


def evaluate(retriever, split="val", k=10, recall_k=100):
    q_df, l_df = load_eval_data(split)

    per_query = []
    latencies = []

    for _, row in q_df.iterrows():
        query_id = row["query_id"]
        query_text = row["query"]
        query_class = row["query_class"] if pd.notna(row["query_class"]) else "Unknown"
        relevance = build_relevance(l_df, query_id)

        start = time.time()
        predicted_ids = retriever.search(query_text, k=recall_k)
        elapsed = time.time() - start
        latencies.append(elapsed)

        per_query.append(
            {
                "query_id": int(query_id),
                "query_class": query_class,
                "ndcg@10": ndcg_at_k(predicted_ids, relevance, k),
                "mrr": mrr(predicted_ids, relevance),
                "recall@100": recall_at_k(predicted_ids, relevance, recall_k),
            }
        )

    # Overall averages
    overall = {
        "ndcg@10": sum(q["ndcg@10"] for q in per_query) / len(per_query),
        "mrr": sum(q["mrr"] for q in per_query) / len(per_query),
        "recall@100": sum(q["recall@100"] for q in per_query) / len(per_query),
    }

    # Per-query-class averages
    by_class = defaultdict(list)
    for q in per_query:
        by_class[q["query_class"]].append(q)

    by_query_class = {}
    for qc, queries in sorted(by_class.items()):
        by_query_class[qc] = {
            "ndcg@10": sum(q["ndcg@10"] for q in queries) / len(queries),
            "mrr": sum(q["mrr"] for q in queries) / len(queries),
            "recall@100": sum(q["recall@100"] for q in queries) / len(queries),
            "count": len(queries),
        }

    latencies.sort()
    n = len(latencies)

    return {
        "split": split,
        "num_queries": len(per_query),
        "overall": overall,
        "by_query_class": by_query_class,
        "latency": {
            "p50": latencies[n // 2],
            "p95": latencies[int(n * 0.95)],
        },
    }


def save_results(results, name):
    out_path = RESULTS_DIR / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")
