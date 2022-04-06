import json
from typing import List, Dict

from torch.utils.data import Dataset
from quaterion.dataset.similarity_samples import SimilarityPairSample


class FAQDataset(Dataset):
    """Dataset class to process .jsonl files with FAQ from popular cloud providers."""

    def __init__(self, dataset_path):
        self.dataset: List[Dict[str, str]] = self.read_dataset(dataset_path)

    def __getitem__(self, index) -> SimilarityPairSample:
        line = self.dataset[index]
        question = line["question"]
        subgroup = hash(question)
        return SimilarityPairSample(
            obj_a=line["question"], obj_b=line["answer"], score=1, subgroup=subgroup
        )

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def read_dataset(dataset_path) -> List[Dict[str, str]]:
        """Read jsonl-file into memory

        Expected line format:

        {
            "source": str,
            "filename": str,
            "question": str,
            "answer": str
        }

        Args:
            dataset_path: path to dataset file

        Returns:
            List[Dict[str, str]]
        """
        with open(dataset_path, "r") as fd:
            return [json.loads(json_line) for json_line in fd]
