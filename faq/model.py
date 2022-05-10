from typing import Union, Dict, Optional

from quaterion.eval.attached_metric import AttachedMetric
from torch.optim import Adam
from quaterion import TrainableModel
from quaterion.train.cache import CacheConfig, CacheType
from quaterion.loss import MultipleNegativesRankingLoss, SimilarityLoss
from sentence_transformers import SentenceTransformer
from quaterion.eval.pair import RetrievalPrecision, RetrievalReciprocalRank
from sentence_transformers.models import Transformer, Pooling
from quaterion_models.encoders import Encoder
from quaterion_models.heads import EncoderHead
from quaterion_models.heads.skip_connection_head import SkipConnectionHead
from faq.encoder import FAQEncoder


class FAQModel(TrainableModel):
    def __init__(self, lr=10e-5, *args, **kwargs):
        self.lr = lr
        super().__init__(*args, **kwargs)

    def configure_metrics(self):
        return [
            AttachedMetric(
                "RetrievalPrecision",
                RetrievalPrecision(k=1),
                prog_bar=True,
                on_epoch=True,
            ),
            AttachedMetric(
                "RetrievalReciprocalRank",
                RetrievalReciprocalRank(),
                prog_bar=True,
                on_epoch=True
            ),
        ]

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)

    def configure_loss(self) -> SimilarityLoss:
        return MultipleNegativesRankingLoss(symmetric=True)

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        pre_trained_model = SentenceTransformer("all-MiniLM-L6-v2")
        transformer: Transformer = pre_trained_model[0]
        pooling: Pooling = pre_trained_model[1]
        encoder = FAQEncoder(transformer, pooling)
        return encoder

    def configure_head(self, input_embedding_size: int) -> EncoderHead:
        return SkipConnectionHead(input_embedding_size)

    def configure_caches(self) -> Optional[CacheConfig]:
        return CacheConfig(CacheType.AUTO, batch_size=1024)
