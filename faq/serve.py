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
        "are mysql database cluster connections encrypted?",
        # A: connections between a database cluster and an application are always encrypted using
        # ssl
        "what is azure content moderator?",
        # A: azure content moderator is an ai service that lets you handle content that is
        # potentially offensive, risky, or otherwise undesirable
        "what is the pricing of aws lambda functions powered by aws graviton2 processors?"
        # A: aws lambda functions powered by aws graviton2 processors are 20% cheaper compared to
        # x86-based lambda functions
    ]

    answer_embeddings = torch.Tensor()
    for batch in dataloader:
        answer_embeddings = torch.cat(
            [answer_embeddings, loaded_model.encode(batch, to_numpy=False)]
        )

    distance = Distance.get_by_name(Distance.COSINE)
    question_embeddings = loaded_model.encode(questions, to_numpy=False)
    question_answers_distances = distance.distance_matrix(
        question_embeddings, answer_embeddings
    )
    answers_indices = question_answers_distances.min(dim=1)[1]
    for question_index, answer_index in enumerate(answers_indices):
        print("Question: ", questions[question_index])
        print("Answer: ", dataset[answer_index])
