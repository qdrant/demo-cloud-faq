#!/bin/bash

mkdir data
wget -O data/cloud_faq_dataset.jsonl https://storage.googleapis.com/demo-cloud-faq/dataset/cloud_faq_dataset.jsonl
python3 -m faq.train_val_split