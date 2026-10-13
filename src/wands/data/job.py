from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = str(ROOT / "data" / "raw")
PROCESSED_DIR = str(ROOT / "data" / "processed")


def main():
    # Step 1: Initialize SparkSession (The entrypoint to Spark)
    spark = SparkSession.builder.appName("WANDS Search Engine").getOrCreate()

    # Set log level to avoid spamming the console
    spark.sparkContext.setLogLevel("WARN")

    try:
        print("Starting PySpark job...")

        # Step 2: EXTRACT (Read raw CSV data)
        # In production, replace this path with your S3, HDFS, or GCS path
        product_df = spark.read.csv(
            f"{RAW_DIR}/product.csv",
            header=True,
            inferSchema=True,
            sep="\t",
        )
        query_df = spark.read.csv(
            f"{RAW_DIR}/query.csv",
            header=True,
            inferSchema=True,
            sep="\t",
        )
        label_df = spark.read.csv(
            f"{RAW_DIR}/label.csv",
            header=True,
            inferSchema=True,
            sep="\t",
        )
        print(
            f"Loaded: {product_df.count()} products, "
            f"{query_df.count()} queries, {label_df.count()} labels"
        )

        # Step 3: TRANSFORM — clean text columns and build search_text
        text_cols = ["product_name", "product_class", "category hierarchy", "product_features"]
        for c in text_cols:
            product_df = product_df.withColumn(
                c, F.when(F.trim(F.col(c)) == "", None).otherwise(F.trim(F.col(c)))
            )

        product_df = product_df.withColumn(
            "search_text",
            F.concat_ws(" ", *[F.col(c) for c in text_cols]),
        )

        # Step 4: LOAD — write parquet
        product_df.write.mode("overwrite").parquet(f"{PROCESSED_DIR}/products.parquet")
        query_df.write.mode("overwrite").parquet(f"{PROCESSED_DIR}/queries.parquet")
        label_df.write.mode("overwrite").parquet(f"{PROCESSED_DIR}/labels.parquet")

        print(f"Wrote processed data to {PROCESSED_DIR}")

    except Exception as e:
        print(f"Job failed with error: {str(e)}")
        raise

    finally:
        # Step 5: Stop the session to release cluster resources
        spark.stop()


if __name__ == "__main__":
    main()
