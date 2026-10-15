"""Fine-tune a bi-encoder on WANDS train-split pairs with hard negatives from BM25."""

import json
import random
from pathlib import Path

import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader

from wands.retrieval.bm25 import BM25Retriever

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "dense_finetuned"

BASE_MODEL = "all-MiniLM-L6-v2"
SEED = 42
EPOCHS = 2
BATCH_SIZE = 32
MAX_POSITIVES_PER_QUERY = 50


def load_split_data(split):
    """Load queries, labels, and products filtered to a split."""
    with open(PROCESSED_DIR / "splits.json") as f:
        splits = json.load(f)

    query_ids = set(splits[split])
    queries_df = pd.read_parquet(PROCESSED_DIR / "queries.parquet")
    labels_df = pd.read_parquet(PROCESSED_DIR / "labels.parquet")
    products_df = pd.read_parquet(PROCESSED_DIR / "products.parquet")

    queries_df = queries_df[queries_df["query_id"].isin(query_ids)]
    labels_df = labels_df[labels_df["query_id"].isin(query_ids)]

    return queries_df, labels_df, products_df


def build_product_text_map(products_df):
    """Map product_id -> search_text."""
    return dict(zip(products_df["product_id"], products_df["search_text"].fillna("")))


def build_triplets(queries_df, labels_df, product_texts, bm25_retriever, max_negatives=5):
    """Build (query, positive, negative) triplets with hard negatives from BM25.

    For each query:
    - Positives = products labeled Exact or Partial
    - Hard negatives = BM25 top results that are labeled Irrelevant
    - Fallback negatives = random Irrelevant products for that query
    """
    random.seed(SEED)
    triplets = []

    for _, q_row in queries_df.iterrows():
        query_id = q_row["query_id"]
        query_text = q_row["query"]

        q_labels = labels_df[labels_df["query_id"] == query_id]
        positives = q_labels[q_labels["label"].isin(["Exact", "Partial"])]["product_id"].tolist()
        irrelevants = q_labels[q_labels["label"] == "Irrelevant"]["product_id"].tolist()

        if not positives or not irrelevants:
            continue

        # Get hard negatives: BM25 top results that are actually Irrelevant
        bm25_results = bm25_retriever.search(query_text, k=100)
        hard_negatives = [pid for pid in bm25_results if pid in set(irrelevants)]

        # Use hard negatives if available, otherwise random irrelevants
        negatives = hard_negatives[:max_negatives] if hard_negatives else []
        if len(negatives) < max_negatives:
            remaining = [pid for pid in irrelevants if pid not in set(negatives)]
            random.shuffle(remaining)
            negatives.extend(remaining[: max_negatives - len(negatives)])

        # Cap positives to keep training size manageable
        if len(positives) > MAX_POSITIVES_PER_QUERY:
            random.shuffle(positives)
            positives = positives[:MAX_POSITIVES_PER_QUERY]

        # Create triplets: pair each positive with one random negative
        for pos_id in positives:
            pos_text = product_texts.get(pos_id, "")
            neg_id = random.choice(negatives)
            neg_text = product_texts.get(neg_id, "")
            triplets.append(InputExample(texts=[query_text, pos_text, neg_text]))

    random.shuffle(triplets)
    return triplets


def main():
    print("Loading data...")
    train_queries, train_labels, products_df = load_split_data("train")
    val_queries, val_labels, _ = load_split_data("val")
    product_texts = build_product_text_map(products_df)

    print("Building BM25 index for hard negative mining...")
    bm25 = BM25Retriever()

    print("Building training triplets...")
    train_triplets = build_triplets(train_queries, train_labels, product_texts, bm25)
    print(f"Created {len(train_triplets)} training triplets")

    # Build a small validation set for monitoring
    val_triplets = build_triplets(val_queries, val_labels, product_texts, bm25, max_negatives=2)
    print(f"Created {len(val_triplets)} validation triplets")

    # Set up evaluator from val triplets
    val_anchors = [t.texts[0] for t in val_triplets[:500]]
    val_positives = [t.texts[1] for t in val_triplets[:500]]
    val_negatives = [t.texts[2] for t in val_triplets[:500]]
    evaluator = TripletEvaluator(val_anchors, val_positives, val_negatives, name="val")

    # Load model and train
    model = SentenceTransformer(BASE_MODEL)
    train_dataloader = DataLoader(train_triplets, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.TripletLoss(model=model)

    print(f"Training for {EPOCHS} epochs, {len(train_dataloader)} batches per epoch...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=EPOCHS,
        warmup_steps=int(len(train_dataloader) * 0.1),
        output_path=str(MODEL_DIR),
        show_progress_bar=True,
    )

    print(f"Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
