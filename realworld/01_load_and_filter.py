"""
01_load_and_filter.py
=====================
Load the AnirbanSaha/health-misinformation-reddit-dataset from HuggingFace,
filter to texts with >= 150 characters, and save to data/filtered_misinfo.csv.

Run:
    python 01_load_and_filter.py
"""

import os
import pandas as pd
from datasets import load_dataset

os.makedirs("data", exist_ok=True)
os.makedirs("results/real/responses", exist_ok=True)
os.makedirs("results/real/judgments", exist_ok=True)

print("Loading dataset from HuggingFace...")
ds = load_dataset("AnirbanSaha/health-misinformation-reddit-dataset")
df = ds["train"].to_pandas()

print(f"Total rows loaded: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Filter: keep only texts with >= 150 characters
df["text_len"] = df["text"].str.len()
df_filtered = df[df["text_len"] >= 150].reset_index(drop=True)
df_filtered["doc_id"] = [f"misinfo_{i:04d}" for i in range(len(df_filtered))]

print(f"\nRows with 150+ characters: {len(df_filtered)}")
print(f"\nText length distribution:")
print(df_filtered["text_len"].describe())
print(f"\nSample texts (first 3):")
for _, row in df_filtered.head(3).iterrows():
    print(f"  [{row['text_len']} chars] {row['text'][:120]}...")

df_filtered[["doc_id", "text", "text_len"]].to_csv(
    "data/filtered_misinfo.csv", index=False
)
print(f"\nSaved {len(df_filtered)} texts -> data/filtered_misinfo.csv")
