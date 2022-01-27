from typing import Union, Dict, Any

import torch

from torch.optim import Adam
from torchmetrics import MeanMetric
from torchmetrics.utilities.data import get_group_indexes
from torchmetrics.functional import retrieval_reciprocal_rank
from pytorch_lightning.utilities.types import (
    TRAIN_DATALOADERS,
    EVAL_DATALOADERS,
)
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import util
from quaterion.utils.enums import TrainStage
from quaterion.loss.similarity_loss import SimilarityLoss
from quaterion.train.trainable_model import TrainableModel
from quaterion.train.encoders import CacheConfig, CacheType
from quaterion.loss.contrastive_loss import ContrastiveLoss
from quaterion_models.heads.gated_head import GatedHead
from quaterion_models.heads.encoder_head import EncoderHead
from quaterion_models.encoders import Encoder

from encoders.faq_encoder import FAQEncoder


class GatedModel(TrainableModel):
    def __init__(self, pretrained_name="all-MiniLM-L6-v2", lr=10e-5):
        self._pretrained_name = pretrained_name
        self.lr = lr
        super().__init__()

        self.metric = MeanMetric(compute_on_step=False)
        self.overlapping = 3

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        pre_trained_model = SentenceTransformer(self._pretrained_name)
        transformer: Transformer = pre_trained_model[0]
        pooling: Pooling = pre_trained_model[1]
        encoder = FAQEncoder(transformer, pooling)
        return encoder

    def configure_caches(self) -> CacheConfig:
        return CacheConfig(CacheType.AUTO, num_workers=4)

    def configure_head(self, input_embedding_size: int) -> EncoderHead:
        return GatedHead(input_embedding_size=input_embedding_size)

    def configure_loss(self) -> SimilarityLoss:
        return ContrastiveLoss()

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
        pairs = targets["pairs"]
        subgroups = targets["subgroups"]
        labels = targets["labels"]
        rep_anchor = embeddings[pairs[:, 0]]
        rep_other = embeddings[pairs[:, 1]]
        distances = self.loss.distance_metric(rep_anchor, rep_other)
        pairs_num = len(pairs)
        for group in get_group_indexes(indexes=subgroups[:pairs_num]):
            mini_preds = 1.0 - distances[group]
            mini_target = labels[group] > 0
            self.metric(retrieval_reciprocal_rank(mini_preds, mini_target))
        self.log(f"{stage}_metric", self.metric.compute())

    def training_step(self, batch, batch_idx, **kwargs) -> torch.Tensor:
        stage = TrainStage.TRAIN
        features, targets = batch
        embeddings = self.model(features)
        pos_embeddings = embeddings[:1]
        neg_embeddings = embeddings[embeddings.shape[0] // 2 :]
        cosine_scores = util.cos_sim(pos_embeddings, neg_embeddings).view(-1)

        sort_order = cosine_scores.argsort(descending=True)
        batch_size = int(sort_order[0].item()) + self.overlapping
        if batch_size > cosine_scores.shape[0]:
            batch_size = cosine_scores.shape[0]

        indices = []
        for ind, sort_ind in enumerate(sort_order):
            if ind <= batch_size:
                indices.append(ind)

        emb = [embeddings[embeddings.shape[0] // 2 + ind] for ind in indices]
        flatten_batch = torch.concat(emb)
        batch = flatten_batch.view(len(emb), *pos_embeddings[0].shape)
        anchors = pos_embeddings.repeat(batch.shape[0], 1)
        hard_embeddings = torch.concat([batch, anchors])
        hard_targets = {
            "pairs": torch.LongTensor(
                [[i, i + len(batch)] for i in range(len(batch))]
            ),
            "labels": torch.Tensor(
                [targets["labels"][ind] for ind in indices]
            ),
            "subgroups": torch.Tensor(
                [targets["subgroups"][ind] for ind in indices] * 2
            ),
        }
        loss = self.loss(embeddings=hard_embeddings, **hard_targets)
        self.log(f"{stage}_loss", loss)
        self.process_results(
            embeddings=hard_embeddings,
            targets=hard_targets,
            batch_idx=batch_idx,
            stage=stage,
            **kwargs,
        )
        return loss

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
