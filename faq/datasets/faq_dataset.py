import json
import random

from collections import defaultdict

from torch.utils.data import IterableDataset
from quaterion.dataset.similarity_data_loader import SimilarityPairSample


class FAQDataset(IterableDataset):
    def __init__(self, dataset_path):
        self.raw_dataset = self.read_dataset(dataset_path)
        self._shuffled_values = list(
            line for lines in self.raw_dataset.values() for line in lines
        )
        self.size = len(self._shuffled_values)

        self.start_index = None
        self.end_index = None
        random.shuffle(self._shuffled_values)

    @staticmethod
    def read_dataset(dataset_path):
        raw_dataset = defaultdict(list)
        with open(dataset_path, "r") as fd:
            for json_line in fd:
                line = json.loads(json_line)
                raw_dataset[line["source"]].append(line)
        return raw_dataset

    @staticmethod
    def emit_negative_sample(question, negative_samples, subgroup):
        for negative_sample in negative_samples:
            yield SimilarityPairSample(
                obj_a=question,
                obj_b=negative_sample["answer"],
                score=0,
                subgroup=subgroup,
            )

    def __iter__(self):
        if self.start_index is not None and self.end_index:
            print("start ind ", self.start_index, "end ind ", self.end_index)

        if self.start_index is not None:
            shuffled_values = self._shuffled_values[
                self.start_index : self.end_index
            ]
        else:
            shuffled_values = self._shuffled_values

        for ind, line in enumerate(shuffled_values):
            subgroup = 0
            if not hasattr(self, "worker_id"):
                negative_indices = list(range(len(shuffled_values)))
                negative_indices.remove(ind)
                shuffled_negative_indices = random.sample(
                    negative_indices, len(negative_indices)
                )
                negative_samples = (
                    shuffled_values[index]
                    for index in shuffled_negative_indices
                )

                question = line["question"]
                yield SimilarityPairSample(
                    obj_a=question,
                    obj_b=line["answer"],
                    score=1,
                    subgroup=subgroup,
                )

                yield from self.emit_negative_sample(
                    question, negative_samples, subgroup
                )
            else:
                question = line["question"]
                yield SimilarityPairSample(
                    obj_a=question, obj_b=line["answer"], score=1, subgroup=0
                )

    def __getitem__(self, index) -> SimilarityPairSample:
        raise NotImplementedError("That's a lazy reader")
