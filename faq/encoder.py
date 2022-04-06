import os

from typing import List

from quaterion_models.types import TensorInterchange, CollateFnType
from torch import Tensor, nn
from quaterion_models.encoders import Encoder
from sentence_transformers.models import Transformer, Pooling
from quaterion.dataset.similarity_samples import SimilarityPairSample


class FAQEncoder(Encoder):
    def __init__(self, transformer, pooling):
        super().__init__()
        self.transformer = transformer
        self.pooling = pooling
        self.encoder = nn.Sequential(self.transformer, self.pooling)

    def trainable(self) -> bool:
        """Defines if encoder is trainable.

        This flag affects caching and checkpoint saving of the encoder.
        """
        return False

    @property
    def embedding_size(self) -> int:
        return self.transformer.get_word_embedding_dimension()

    def text_collate(self, batch: List[SimilarityPairSample]):
        return self.transformer.tokenize(batch)

    def get_collate_fn(self) -> CollateFnType:
        return self.text_collate

    def forward(self, batch: TensorInterchange) -> Tensor:
        return self.encoder(batch)["sentence_embedding"]

    @staticmethod
    def _transformer_path(path: str):
        return os.path.join(path, "transformer")

    @staticmethod
    def _pooling_path(path: str):
        return os.path.join(path, "pooling")

    def save(self, output_path: str):
        transformer_path = self._transformer_path(output_path)
        os.makedirs(transformer_path, exist_ok=True)
        pooling_path = self._pooling_path(output_path)
        os.makedirs(pooling_path, exist_ok=True)
        self.transformer.save(self._transformer_path(output_path))
        self.pooling.save(self._pooling_path(output_path))

    @classmethod
    def load(cls, input_path: str) -> Encoder:
        transformer = Transformer.load(cls._transformer_path(input_path))
        pooling = Pooling.load(cls._pooling_path(input_path))
        return cls(transformer=transformer, pooling=pooling)
