"""Train and evaluate a LambdaMART learning-to-rank reranker.

Feature generation is split into stages to avoid OOM:
  Stage 1 (candidates): BM25 retriever → hybrid candidates + BM25 scores → parquet
  Stage 2 (dense):      Dense retriever → dense scores for saved candidates → parquet
  Stage 3 (combine):    Product metadata → text features → merge all into final features

Usage:
  python -m wands.rerank.ltr stage1     # candidates + BM25 scores
  python -m wands.rerank.ltr stage2     # dense scores
  python -m wands.rerank.ltr stage3     # text/metadata features + combine
  python -m wands.rerank.ltr train      # train LambdaMART on cached features
  python -m wands.rerank.ltr generate   # run stages 1-3 sequentially
  python -m wands.rerank.ltr            # generate + train
"""

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from wands.eval.metrics import RELEVANCE
from wands.rerank.features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "ltr"
CACHE_DIR = ROOT / "data" / "processed" / "ltr_cache"


def load_split_data(split):
    """Load queries and labels for a split."""
    with open(PROCESSED_DIR / "splits.json") as f:
        splits = json.load(f)

    query_ids = set(splits[split])
    queries_df = pd.read_parquet(PROCESSED_DIR / "queries.parquet")
    labels_df = pd.read_parquet(PROCESSED_DIR / "labels.parquet")

    queries_df = queries_df[queries_df["query_id"].isin(query_ids)]
    labels_df = labels_df[labels_df["query_id"].isin(query_ids)]

    return queries_df, labels_df


def stage1_candidates():
    """Load BM25 only. Get hybrid candidates + BM25 scores, save to disk."""
    from wands.retrieval.bm25 import BM25Retriever

    print("Loading BM25 retriever...")
    bm25 = BM25Retriever()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        queries_df, labels_df = load_split_data(split)
        n_q = len(queries_df)
        print(f"\nStage 1 ({split}): {n_q} queries — retrieving candidates + BM25 scores")

        rows = []
        for i, (_, q_row) in enumerate(queries_df.iterrows()):
            query_id = q_row["query_id"]
            query_text = q_row["query"]

            # BM25 top-100 candidates (hybrid needs dense too, but we use BM25-only
            # candidates here and will add dense candidates in stage 2)
            candidates = bm25.search(query_text, k=100)
            bm25_scores = bm25.get_scores(query_text, candidates)

            # Build relevance lookup
            q_labels = labels_df[labels_df["query_id"] == query_id]
            relevance = {}
            for _, l_row in q_labels.iterrows():
                relevance[l_row["product_id"]] = RELEVANCE[l_row["label"]]

            for pid in candidates:
                rows.append(
                    {
                        "query_id": int(query_id),
                        "query": query_text,
                        "product_id": int(pid),
                        "bm25_score": bm25_scores.get(pid, 0.0),
                        "label": relevance.get(pid, 0),
                    }
                )

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(queries_df)} queries done")

        df = pd.DataFrame(rows)
        out_path = CACHE_DIR / f"{split}_stage1.parquet"
        df.to_parquet(out_path)
        print(f"  Saved {len(df)} rows to {out_path}")

    print("\nStage 1 complete. BM25 memory can now be freed.")


def stage2_dense():
    """Load Dense retriever only. Compute dense scores for stage1 candidates."""
    from wands.retrieval.dense import DenseRetriever

    print("Loading Dense retriever...")
    dense = DenseRetriever()

    for split in ["train", "val"]:
        stage1_df = pd.read_parquet(CACHE_DIR / f"{split}_stage1.parquet")
        queries_df, labels_df = load_split_data(split)
        print(f"\nStage 2 ({split}): computing dense scores")

        # Group by query
        all_rows = []
        for i, (query_id, group) in enumerate(stage1_df.groupby("query_id", sort=False)):
            query_text = group["query"].iloc[0]
            bm25_candidates = group["product_id"].tolist()

            # Get dense top-100 to merge (RRF-style union)
            dense_candidates = dense.search(query_text, k=100)

            # Union of candidates
            all_candidates = list(dict.fromkeys(bm25_candidates + dense_candidates))

            # Dense scores for all candidates
            dense_scores = dense.get_scores(query_text, all_candidates)

            # BM25 scores from stage1
            bm25_lookup = dict(zip(group["product_id"], group["bm25_score"]))

            # Labels from stage1 + any new candidates
            label_lookup = dict(zip(group["product_id"], group["label"]))
            # For dense-only candidates, look up labels
            q_labels = labels_df[labels_df["query_id"] == query_id]
            for _, l_row in q_labels.iterrows():
                pid = l_row["product_id"]
                if pid not in label_lookup:
                    label_lookup[pid] = RELEVANCE[l_row["label"]]

            for pid in all_candidates:
                all_rows.append(
                    {
                        "query_id": int(query_id),
                        "query": query_text,
                        "product_id": int(pid),
                        "bm25_score": bm25_lookup.get(pid, 0.0),
                        "dense_score": dense_scores.get(pid, 0.0),
                        "label": label_lookup.get(pid, 0),
                    }
                )

            if (i + 1) % 50 == 0:
                print(f"  {i + 1} queries done")

        df = pd.DataFrame(all_rows)
        out_path = CACHE_DIR / f"{split}_stage2.parquet"
        df.to_parquet(out_path)
        print(f"  Saved {len(df)} rows to {out_path}")

    print("\nStage 2 complete. Dense memory can now be freed.")


def stage3_features():
    """Load product metadata only. Compute text/metadata features and merge with scores."""
    print("Loading product data...")
    products_df = pd.read_parquet(PROCESSED_DIR / "products.parquet")
    products_df = products_df.set_index("product_id")
    products_df["average_rating"] = products_df["average_rating"].fillna(0.0)
    products_df["review_count"] = products_df["review_count"].fillna(0.0)
    products_df["rating_count"] = products_df["rating_count"].fillna(0.0)
    products_df["product_name"] = products_df["product_name"].fillna("")
    products_df["product_class"] = products_df["product_class"].fillna("")
    products_df["product_description"] = products_df["product_description"].fillna("")
    products_df["search_text"] = products_df["search_text"].fillna("")

    for split in ["train", "val"]:
        stage2_df = pd.read_parquet(CACHE_DIR / f"{split}_stage2.parquet")
        print(f"\nStage 3 ({split}): computing text/metadata features for {len(stage2_df)} rows")

        feature_rows = []
        for _, row in stage2_df.iterrows():
            pid = row["product_id"]
            query_text = row["query"]
            query_tokens = set(query_text.lower().split())

            feat = {
                "query_id": row["query_id"],
                "product_id": pid,
                "label": row["label"],
                "bm25_score": row["bm25_score"],
                "dense_score": row["dense_score"],
            }

            try:
                p = products_df.loc[pid]
                search_tokens = set(str(p["search_text"]).lower().split())
                name_tokens = set(str(p["product_name"]).lower().split())
                product_class_lower = str(p["product_class"]).lower()

                if query_tokens:
                    n_qt = len(query_tokens)
                    feat["query_token_overlap"] = len(query_tokens & search_tokens) / n_qt
                    feat["product_name_overlap"] = len(query_tokens & name_tokens) / n_qt
                else:
                    feat["query_token_overlap"] = 0.0
                    feat["product_name_overlap"] = 0.0

                feat["category_match"] = (
                    1.0 if any(t in product_class_lower for t in query_tokens) else 0.0
                )
                feat["average_rating"] = float(p["average_rating"])
                feat["review_count"] = float(p["review_count"])
                feat["rating_count"] = float(p["rating_count"])
                feat["search_text_len"] = len(str(p["search_text"]).split())
                feat["description_len"] = len(str(p["product_description"]).split())
            except KeyError:
                feat.update({name: 0.0 for name in FEATURE_NAMES if name not in feat})

            feature_rows.append(feat)

        df = pd.DataFrame(feature_rows)
        out_path = CACHE_DIR / f"{split}_features.parquet"
        df.to_parquet(out_path)
        print(f"  Saved {len(df)} rows to {out_path}")
        pos = (df["label"] > 0).sum()
        print(f"  {pos} positive labels, {len(df) - pos} negative/unlabeled")

    print("\nStage 3 complete. Feature files ready for training.")


def generate_features():
    """Run all three stages sequentially as separate processes."""
    import subprocess
    import sys

    python = sys.executable
    module = "wands.rerank.ltr"

    for stage in ["stage1", "stage2", "stage3"]:
        print(f"\n{'=' * 60}")
        print(f"Running {stage} in subprocess...")
        print(f"{'=' * 60}")
        result = subprocess.run(
            [python, "-m", module, stage],
            cwd=ROOT,
            env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
        )
        if result.returncode != 0:
            print(f"\n{stage} failed with exit code {result.returncode}")
            return False

    print("\nAll stages complete!")
    return True


def train():
    """Load cached features and train LambdaMART."""
    print("Loading cached features...")
    train_df = pd.read_parquet(CACHE_DIR / "train_features.parquet")
    val_df = pd.read_parquet(CACHE_DIR / "val_features.parquet")

    X_train = train_df[FEATURE_NAMES].values
    y_train = train_df["label"].values
    groups_train = train_df.groupby("query_id", sort=False).size().tolist()

    X_val = val_df[FEATURE_NAMES].values
    y_val = val_df["label"].values
    groups_val = val_df.groupby("query_id", sort=False).size().tolist()

    print(f"Train: {X_train.shape[0]} rows, {sum(y_train > 0)} positive")
    print(f"Val: {X_val.shape[0]} rows, {sum(y_val > 0)} positive")

    train_set = lgb.Dataset(X_train, label=y_train, group=groups_train, feature_name=FEATURE_NAMES)
    val_set = lgb.Dataset(X_val, label=y_val, group=groups_val, feature_name=FEATURE_NAMES)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [10],
        "n_estimators": 200,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "min_child_samples": 20,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_set,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[lgb.log_evaluation(period=25)],
    )

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "lambdamart.txt"
    model.save_model(str(model_path))
    print(f"\nModel saved to {model_path}")

    # Print feature importance
    importance = model.feature_importance(importance_type="gain")
    print("\nFeature importance (gain):")
    for name, imp in sorted(zip(FEATURE_NAMES, importance), key=lambda x: x[1], reverse=True):
        print(f"  {name:>25}: {imp:.1f}")


def main():
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else ""

    if cmd == "stage1":
        stage1_candidates()
    elif cmd == "stage2":
        stage2_dense()
    elif cmd == "stage3":
        stage3_features()
    elif cmd == "train":
        train()
    elif cmd == "generate":
        generate_features()
    else:
        if generate_features():
            train()


if __name__ == "__main__":
    main()
