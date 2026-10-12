import pandas as pd

DATA = "../../../data/raw"

query_df = pd.read_csv(f"{DATA}/query.csv", sep='\t')
dimensions = query_df.shape

print(f"Dimensions (Rows, Columns): {dimensions}")
print(f"Number of rows: {query_df.shape[0]}")
print(f"Number of columns: {query_df.shape[1]}")

# get products
product_df = pd.read_csv(f"{DATA}/product.csv", sep='\t')
dimensions = product_df.shape

print(f"Dimensions (Rows, Columns): {dimensions}")
print(f"Number of rows: {product_df.shape[0]}")
print(f"Number of columns: {product_df.shape[1]}")

# get labels
label_df = pd.read_csv(f"{DATA}/label.csv", sep='\t')
dimensions = label_df.shape

print(f"Dimensions (Rows, Columns): {dimensions}")
print(f"Number of rows: {label_df.shape[0]}")
print(f"Number of columns: {label_df.shape[1]}")