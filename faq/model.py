from typing import Union, Dict, Optional, Any

from torch import Tensor
from torch.optim import AdamW
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling

from quaterion import TrainableModel
from quaterion.utils.enums import TrainStage
from quaterion.train.cache import CacheConfig, CacheType
from quaterion.loss import MultipleNegativesRankingLoss, SimilarityLoss
from quaterion.eval.pair import RetrievalPrecision, RetrievalReciprocalRank
from quaterion_models.encoders import Encoder
from quaterion_models.heads import EncoderHead
from quaterion_models.heads.skip_connection_head import SkipConnectionHead

from faq.encoder import FAQEncoder


class FAQModel(TrainableModel):
    def __init__(self, pretrained_name="all-MiniLM-L6-v2", lr=10e-2, *args, **kwargs):
        self._pretrained_name = pretrained_name
        self.lr = lr

        super().__init__(*args, **kwargs)

        self.retrieval_precision = RetrievalPrecision(k=1)
        self.retrieval_reciprocal_rank = RetrievalReciprocalRank()

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.lr)

    def configure_loss(self) -> SimilarityLoss:
        return MultipleNegativesRankingLoss(symmetric=True)

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        pre_trained_model = SentenceTransformer(self._pretrained_name)
        transformer: Transformer = pre_trained_model[0]
        pooling: Pooling = pre_trained_model[1]
        encoder = FAQEncoder(transformer, pooling)
        return encoder

    def configure_head(self, input_embedding_size: int) -> EncoderHead:
        return SkipConnectionHead(input_embedding_size)

    def configure_caches(self) -> Optional[CacheConfig]:
        return CacheConfig(CacheType.AUTO, batch_size=64)

    def process_results(
        self,
        embeddings: Tensor,
        targets: Dict[str, Any],
        batch_idx: int,
        stage: TrainStage,
        **kwargs,
    ):
        self.retrieval_reciprocal_rank.update(embeddings, **targets)

        self.log(
            f"{stage}.rrk",
            self.retrieval_reciprocal_rank.compute().mean(),
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )
        self.retrieval_reciprocal_rank.reset()

        self.retrieval_precision.update(embeddings, **targets)
        self.log(
            f"{stage}.rp@1",
            self.retrieval_precision.compute().mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.retrieval_precision.reset()