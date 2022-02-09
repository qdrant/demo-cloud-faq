from typing import Union, Dict, Any

import torch
from quaterion.eval.metrics import (
    retrieval_reciprocal_rank_2d,
    retrieval_precision_2d_at_one,
)

from torch.optim import Adam
from torchmetrics import (
    MeanMetric,
    MetricCollection,
    RetrievalMRR,
    RetrievalPrecision,
)
from torchmetrics.utilities.data import get_group_indexes
from torchmetrics.functional import (
    retrieval_reciprocal_rank,
    retrieval_precision,
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
from quaterion.train.encoders import CacheConfig, CacheType
from quaterion.loss.contrastive_loss import ContrastiveLoss
from quaterion_models.heads.encoder_head import EncoderHead
from quaterion_models.encoders import Encoder

from faq.encoders.faq_encoder import FAQEncoder
from faq.heads.skip_connection import SkipConnectionHead


class SkipConnectionModel(TrainableModel):
    def __init__(self, pretrained_name="all-MiniLM-L6-v2", lr=10e-5):
        self._pretrained_name = pretrained_name
        self.lr = lr
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
        return CacheConfig(CacheType.AUTO, num_workers=4)

    def configure_head(self, input_embedding_size: int) -> EncoderHead:
        return SkipConnectionHead(
            input_embedding_size=input_embedding_size,
            output_embedding_size=input_embedding_size,
        )

    def configure_loss(self) -> SimilarityLoss:
        return ContrastiveLoss(margin=0.8)

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

        distance_matrix = self.loss.distance_metric(
            embeddings, embeddings, matrix=True
        )
        distance_matrix[torch.eye(embeddings_count, dtype=torch.bool)] = 1.0
        preds = (
            1.0
            - distance_matrix[: embeddings_count // 2, embeddings_count // 2 :]
        )
        labels = torch.zeros(preds.shape, device=preds.device)
        labels[torch.eye(*labels.shape).bool()] = True
        # indices = torch.arange(0, preds.shape[0]).view(preds.shape[0], -1).repeat(1, preds.shape[1])
        rrk = retrieval_reciprocal_rank_2d(preds, labels)
        rp_at_one = retrieval_precision_2d_at_one(preds, labels)
        # rrk_metric = RetrievalMRR()
        # rp_at_one_metric = RetrievalPrecision(k=1)
        # rrk = rrk_metric(preds, labels, indexes=indices)
        # rp_at_one = rp_at_one_metric(preds, labels, indexes=indices)
        self.metric["rrk"](rrk.mean())
        self.metric["rp@1"](rp_at_one.mean())
        self.log(
            f"{stage}_metric",
            self.metric.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

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
