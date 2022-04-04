from quaterion_models.heads.encoder_head import EncoderHead
from quaterion_models.heads.gated_head import GatedHead

from faq.models.experiment_model import ExperimentModel


class GatedModel(ExperimentModel):
    def __init__(self, pretrained_name="all-MiniLM-L6-v2", lr=10e-2, loss_fn="mnr"):
        super().__init__(pretrained_name, lr, loss_fn)

    def configure_head(self, input_embedding_size: int) -> EncoderHead:
        return GatedHead(input_embedding_size=input_embedding_size)
