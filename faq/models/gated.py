from quaterion_models.heads.encoder_head import EncoderHead
from quaterion_models.heads.gated_head import GatedHead

from torchmetrics import (
    MeanMetric,
    MetricCollection,
)

from faq.models.experiment_model import ExperimentModel


class GatedModel(ExperimentModel):
    def __init__(self, pretrained_name="all-MiniLM-L6-v2", lr=10e-2):
        self._pretrained_name = pretrained_name
        self.lr = lr
        print(f"models lr: {self.lr}")
        super().__init__()

        self.metric = MetricCollection(
            {
                "rrk": MeanMetric(compute_on_step=False),
                "rp@1": MeanMetric(compute_on_step=False),
            }
        )

    def configure_head(self, input_embedding_size: int) -> EncoderHead:
        return GatedHead(input_embedding_size=input_embedding_size)
