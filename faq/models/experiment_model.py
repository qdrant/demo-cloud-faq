import json

from typing import Union, Dict, Any, Optional

import torch

from torch.optim import Adam
from torchmetrics import (
    MeanMetric,
    MetricCollection,
)
from pytorch_lightning.utilities.types import (
    TRAIN_DATALOADERS,
    EVAL_DATALOADERS,
)
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
from quaterion.utils.enums import TrainStage
from quaterion.loss.similarity_loss import SimilarityLoss
from quaterion.train.trainable_model import TrainableModel
from quaterion.train.cache import CacheConfig, CacheType
from quaterion.loss import MultipleNegativesRankingLoss
from quaterion_models.heads.encoder_head import EncoderHead
from quaterion_models.encoders import Encoder

from faq.encoders.faq_encoder import FAQEncoder
from faq.utils.metrics import (
    retrieval_reciprocal_rank_2d,
    retrieval_precision_2d,
)
from faq.models.fixed_contrastive_loss import FixedContrastiveLoss
from faq.utils.utils import wrong_prediction_indices


class ExperimentModel(TrainableModel):
    def __init__(self, pretrained_name="all-MiniLM-L6-v2", lr=10e-2, loss_fn="mnr"):
        self._pretrained_name = pretrained_name
        self.lr = lr
        self._loss_fn = loss_fn

        super().__init__()

        self.metric = MetricCollection(
            {
                "rrk": MeanMetric(compute_on_step=False),
                "rp@1": MeanMetric(compute_on_step=False),
            }
        )

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        pre_trained_model = SentenceTransformer(self._pretrained_name)
        transformer: Transformer = pre_trained_model[0]
        pooling: Pooling = pre_trained_model[1]
        encoder = FAQEncoder(transformer, pooling)
        return encoder

    def configure_caches(self) -> CacheConfig:
        return CacheConfig(CacheType.AUTO, batch_size=1024)

    def configure_head(self, input_embedding_size: int) -> EncoderHead:
        raise NotImplementedError()

    def configure_loss(self) -> SimilarityLoss:
        return (
            MultipleNegativesRankingLoss(symmetric=True)
            if self._loss_fn == "mnr"
            else FixedContrastiveLoss(margin=1)
        )

    def process_results(
        self,
        embeddings: torch.Tensor,
        targets: Dict[str, Any],
        batch_idx,
        stage: TrainStage,
        **kwargs,
    ):
        """
        Define any additional evaluations of embeddings here.

        :param embeddings: Tensor of batch embeddings, shape: [batch_size x embedding_size]
        :param targets: Output of batch target collate
        :param batch_idx: ID of the processing batch
        :param stage: Train, validation or test stage
        :return: None
        """
        embeddings_count = int(embeddings.shape[0])

        distance_matrix = self.loss.distance_metric(embeddings, embeddings, matrix=True)
        distance_matrix[torch.eye(embeddings_count, dtype=torch.bool)] = 1.0
        predicted_similarity = 1.0 - distance_matrix

        pairs = targets["pairs"]
        labels = targets["labels"]

        target = torch.zeros_like(distance_matrix)
        target[pairs[:, 0], pairs[:, 1]] = labels
        target[pairs[:, 1], pairs[:, 0]] = labels

        rrk = retrieval_reciprocal_rank_2d(predicted_similarity, target)
        rp_at_one = retrieval_precision_2d(predicted_similarity, target)

        self.metric["rrk"](rrk.mean())
        self.metric["rp@1"](rp_at_one.mean())
        self.log(
            f"{stage}_metric",
            self.metric.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        features, targets = batch
        embeddings = self.model(features)

        embeddings_count = int(embeddings.shape[0])

        distance_matrix = self.loss.distance_metric(embeddings, embeddings, matrix=True)
        distance_matrix[torch.eye(embeddings_count, dtype=torch.bool)] = 1.0
        predicted_similarity = 1.0 - distance_matrix
        res = wrong_prediction_indices(predicted_similarity)

        prefix = "valid" if dataloader_idx else "train"
        with open(f"{prefix}_wrong_predictions.jsonl", "w") as f:
            for i in range(res[0].shape[0]):
                json.dump(
                    {
                        "anchor": res[0][i].item(),
                        "wrong": res[1][i].item(),
                        "right": res[2][i].item(),
                    },
                    f,
                )
                f.write("\n")
        return 1

    def on_train_epoch_start(self) -> None:
        self.metric.reset()

    def on_validation_epoch_start(self) -> None:
        """
        Lightning has an odd order of callbacks.
        https://github.com/PyTorchLightning/pytorch-lightning/issues/9811
        To use the same metric object for both training and validation
        stages, we need to reset metric before validation starts its
        computation
        """
        self.metric.reset()

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)

    # region anchors
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/10667
    def train_dataloader(self, *args, **kwargs) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    # endregion
