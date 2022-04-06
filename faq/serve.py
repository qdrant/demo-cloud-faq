import os
import json
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset

from quaterion_models.model import MetricModel
from quaterion.distances import Distance

from faq.config import DATA_DIR, ROOT_DIR


class ServeFAQDataset(Dataset):
    """Dataset class to process .jsonl files with FAQ from popular cloud providers."""

    def __init__(self, dataset_path):
        self.dataset: List[str] = self.read_dataset(dataset_path)

    def __getitem__(self, index) -> str:
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def read_dataset(dataset_path) -> List[str]:
        """Read jsonl-file into memory

        Expected line format:

        {
            "question": str,
            "answer": str,
            ...,
        }

        Args:
            dataset_path: path to dataset file

        Returns:
            List[str]
        """
        with open(dataset_path, "r") as fd:
            sentences = []
            for json_line in fd:
                line = json.loads(json_line)
                sentences.append(line['question'])
                sentences.append(line['answer'])
            return sentences


if __name__ == "__main__":
    loaded_model = MetricModel.load(os.path.join(ROOT_DIR, "servable"))

    path = os.path.join(DATA_DIR, "val_part.jsonl")
    dataset = ServeFAQDataset(path)
    dataloader = DataLoader(dataset, collate_fn=loaded_model.get_collate_fn())

    embeddings = torch.Tensor()
    loaded_model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            embeddings = torch.cat([embeddings, loaded_model(batch)])

    distance = Distance.get_by_name(Distance.COSINE)
    distance_matrix = distance.distance_matrix(embeddings)
    distance_matrix[torch.eye(distance_matrix.shape[0], dtype=torch.bool)] = 1.0
    nearest = torch.argsort(distance_matrix, dim=-1)[:, 0]
    for i in range(len(dataset)):
        print("Anchor: ", dataset[i])
        print("Nearest: ", dataset[nearest[i]], end='\n\n')
