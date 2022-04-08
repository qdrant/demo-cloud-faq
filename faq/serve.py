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
            return [json.loads(json_line)["answer"] for json_line in fd]


if __name__ == "__main__":
    loaded_model = MetricModel.load(os.path.join(ROOT_DIR, "servable"))
    path = os.path.join(DATA_DIR, "val_cloud_faq_dataset.jsonl")
    dataset = ServeFAQDataset(path)
    dataloader = DataLoader(dataset)

    questions = [
        "what is amazon workspaces?",
        # A: amazon workspaces is a managed, secure cloud desktop service
        "what is the pricing of aws lambda functions powered by aws graviton2 processors?",
        # A: aws lambda functions powered by aws graviton2 processors are 20% cheaper compared to
        # x86-based lambda functions
        "can i run a cluster or job for a long time?",
        # yes, you can run a cluster for as long as is required
        "if i continue to use my own memcached clients with my elasticache cluster – will i be able to get this feature?"
        # you can specify any currently supported version (minor and/or major) when creating a new
        # cluster
    ]

    answer_embeddings = torch.Tensor()
    for batch in dataloader:
        answer_embeddings = torch.cat(
            [
                answer_embeddings,
                loaded_model.encode(batch, to_numpy=False),
            ]
        )

    distance = Distance.get_by_name(Distance.COSINE)
    question_embeddings = loaded_model.encode(questions, to_numpy=False)
    # WARNING: results will be incorrect until fixing of cosine distance matrix in quaterion
    question_answers_distances = distance.distance_matrix(
        question_embeddings, answer_embeddings
    )
    answers_indices = question_answers_distances.min(dim=1)[1]
    for question_index, answer_index in enumerate(answers_indices):
        print("Question: ", questions[question_index])
        print("Answer: ", dataset[answer_index])
