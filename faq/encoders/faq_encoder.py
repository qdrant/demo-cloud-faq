import os

from typing import List, Any

from torch import nn, Tensor
from sentence_transformers.models import Transformer, Pooling
from quaterion_models.encoders import Encoder
from quaterion_models.types import TensorInterchange, CollateFnType


class FAQEncoder(Encoder):
    def __init__(
        self,
        transformer: Transformer,
        pooling: Pooling,
        trainable: bool = False,
    ):
        super().__init__()
        self._trainable = trainable
        self.transformer = transformer
        self.pooling = pooling
        self.encoder = nn.Sequential(self.transformer, self.pooling)

    def device(self):
        return next(self.parameters()).is_cuda

    def disable_gradients_if_required(self):
        """
        Disables gradients of the model if it is declared as not trainable

        :return:
        """

        if not self.trainable():
            for _, weights in self.named_parameters():
                weights.requires_grad = False

    def trainable(self) -> bool:
        """
        Defines if encoder is trainable. This flag affects caching and
        checkpoint saving of the encoder.
        :return: bool
        """
        return self._trainable

    def embedding_size(self) -> int:
        """
        :return: Size of resulting embedding
        """
        return self.transformer.get_word_embedding_dimension()

    @classmethod
    def extract_texts(cls, batch: List[Any]) -> List[str]:
        return batch

    def texts_collate(self, batch: List[str]) -> TensorInterchange:
        """
        Transforms a batch from a list of texts to a batch of tensors for the
        model

        :param batch:
            a batch from texts
        :return:
            a batch of tensors for the model
        """
        sentence_features = self.transformer.tokenize(
            self.extract_texts(batch)
        )
        return sentence_features

    def get_collate_fn(self) -> CollateFnType:
        return self.texts_collate

    def forward(self, batch: TensorInterchange) -> Tensor:
        return self.encoder.forward(batch)["sentence_embedding"]

    @classmethod
    def _transformer_save_path(cls, path: str):
        return os.path.join(path, "transformer")

    @classmethod
    def _pooling_save_path(cls, path: str):
        return os.path.join(path, "pooling")

    def save(self, output_path: str):
        transformer_path = self._transformer_save_path(output_path)
        os.mkdir(transformer_path)
        self.transformer.save(transformer_path)

        pooling_path = self._pooling_save_path(output_path)
        os.mkdir(pooling_path)
        self.pooling.save(pooling_path)

    @classmethod
    def load(cls, input_path: str) -> "Encoder":
        transformer_path = cls._transformer_save_path(input_path)
        transformer = Transformer.load(transformer_path)

        pooling_path = cls._pooling_save_path(input_path)
        pooling = Pooling.load(pooling_path)

        return cls(transformer=transformer, pooling=pooling)
