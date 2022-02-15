import os

import pandas as pd

from sklearn.model_selection import train_test_split

from faq.config import DATA_DIR


def split_dataset_by_source(source):
    filtered_df = df[df.source == source]
    filtered_df.to_json(
        os.path.join(DATA_DIR, "by_source", f"{source}_cloud_faq_dataset.jsonl"),
        orient="records",
        lines=True,
    )

    train_X, val_X = train_test_split(filtered_df, random_state=42)
    train_X.to_json(
        os.path.join(DATA_DIR, "by_source", f"{source}_train_cloud_faq_dataset.jsonl"),
        orient="records",
        lines=True,
    )
    val_X.to_json(
        os.path.join(DATA_DIR, "by_source", f"{source}_val_cloud_faq_dataset.jsonl"),
        orient="records",
        lines=True,
    )


filename = "cloud_faq_dataset.jsonl"
path = os.path.join(DATA_DIR, filename)
df = pd.read_json(path, lines=True)
os.makedirs(os.path.join(DATA_DIR, "by_source"))
for source in df.source.value_counts().index:
    split_dataset_by_source(source)

