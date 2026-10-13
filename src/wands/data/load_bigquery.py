import os
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed"

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "search-engine-511420")
DATASET_ID = "wands"


def main():
    client = bigquery.Client(project=PROJECT_ID)

    # Create dataset if it doesn't exist
    dataset_ref = bigquery.DatasetReference(PROJECT_ID, DATASET_ID)
    client.create_dataset(bigquery.Dataset(dataset_ref), exists_ok=True)
    print(f"Dataset {DATASET_ID} ready.")

    tables = ["products", "queries", "labels"]
    for name in tables:
        table_id = f"{PROJECT_ID}.{DATASET_ID}.{name}"
        # Spark writes parquet as a directory of part files — pandas handles this
        df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")

        job_config = bigquery.LoadJobConfig(
            autodetect=True,
            write_disposition="WRITE_TRUNCATE",
        )
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()  # wait for completion

        table = client.get_table(table_id)
        print(f"Loaded {table.num_rows} rows into {table_id}")


if __name__ == "__main__":
    main()
