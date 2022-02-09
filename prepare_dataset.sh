#!/bin/bash


mc cp qdrant/demo-cloud-faq/dataset/cloud_faq_dataset.jsonl  data/cloud_faq_dataset.jsonl


python -m faq.datasets.train_val_split