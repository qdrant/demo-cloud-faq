import os

import pandas as pd

from sklearn.model_selection import train_test_split

from faq.config import DATA_DIR


filename = "cloud_faq_dataset.jsonl"
path = os.path.join(DATA_DIR, filename)
df = pd.read_json(path, lines=True)

source = "aws"
filtered_df = df[df.source == source]
filtered_df.to_json(
    os.path.join(DATA_DIR, f"{source}_cloud_faq_dataset.jsonl"),
    orient="records",
    lines=True,
)


train_X, val_X = train_test_split(filtered_df, random_state=42)
train_X.to_json(
    os.path.join(DATA_DIR, f"{source}_train_cloud_faq_dataset.jsonl"),
    orient="records",
    lines=True,
)
val_X.to_json(
    os.path.join(DATA_DIR, f"{source}_val_cloud_faq_dataset.jsonl"),
    orient="records",
    lines=True,
)
