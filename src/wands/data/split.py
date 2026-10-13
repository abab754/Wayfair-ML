import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15 (remainder)


def main():
    query_df = pd.read_csv(RAW_DIR / "query.csv", sep="\t")
    query_ids = sorted(query_df["query_id"].unique().tolist())

    random.seed(SEED)
    random.shuffle(query_ids)

    n = len(query_ids)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    splits = {
        "train": sorted(query_ids[:train_end]),
        "val": sorted(query_ids[train_end:val_end]),
        "test": sorted(query_ids[val_end:]),
    }

    out_path = PROCESSED_DIR / "splits.json"
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(
        f"Split {n} queries: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
