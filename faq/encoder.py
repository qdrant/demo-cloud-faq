import os
from torch import Tensor, nn
from sentence_transformers.models import Transformer, Pooling
from quaterion_models.types import TensorInterchange, CollateFnType
from quaterion_models.encoders import Encoder


class FAQEncoder(Encoder):
    def __init__(self, transformer, pooling):
        super().__init__()
        self.transformer = transformer
        self.pooling = pooling
        self.encoder = nn.Sequential(self.transformer, self.pooling)

    @property
    def trainable(self) -> bool:
        return False

    @property
    def embedding_size(self) -> int:
        return self.transformer.get_word_embedding_dimension()

    def forward(self, batch: TensorInterchange) -> Tensor:
        return self.encoder(batch)["sentence_embedding"]

    def get_collate_fn(self) -> CollateFnType:
        return self.transformer.tokenize

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
        self.transformer.save(transformer_path)
        self.pooling.save(pooling_path)

    @classmethod
    def load(cls, input_path: str) -> Encoder:
        transformer = Transformer.load(cls._transformer_path(input_path))
        pooling = Pooling.load(cls._pooling_path(input_path))
        return cls(transformer=transformer, pooling=pooling)
