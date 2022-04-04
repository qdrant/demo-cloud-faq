import os

import pandas as pd

from sklearn.model_selection import train_test_split

from faq.config import DATA_DIR

path = os.path.join(DATA_DIR, "cloud_faq_dataset.jsonl")
df = pd.read_json(path, lines=True)
y = df.source
train_X, val_X = train_test_split(df, stratify=df.source, random_state=42)
print(df.head())
train_X.to_json(
    os.path.join(DATA_DIR, "train_cloud_faq_dataset.jsonl"),
    orient="records",
    lines=True,
)
val_X.to_json(
    os.path.join(DATA_DIR, "val_cloud_faq_dataset.jsonl"), orient="records", lines=True
)
