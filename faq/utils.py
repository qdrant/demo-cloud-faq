import os

import pandas as pd
from sklearn.model_selection import train_test_split

from faq.config import DATA_DIR


if __name__ == '__main__':
    path = os.path.join(DATA_DIR, "cloud_faq_dataset.jsonl")
    df = pd.read_json(path, lines=True)
    train_X, val_X = train_test_split(df, stratify=df.source, random_state=42)

    train_X.to_json(
        os.path.join(DATA_DIR, "train_cloud_faq_dataset.jsonl"),
        orient="records",
        lines=True,
    )
    val_X.to_json(
        os.path.join(DATA_DIR, "val_cloud_faq_dataset.jsonl"), orient="records", lines=True
    )
