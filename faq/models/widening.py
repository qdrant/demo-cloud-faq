from quaterion_models.heads import EncoderHead, WideningHead


from faq.models.experiment_model import ExperimentModel


class WideningModel(ExperimentModel):
    def __init__(self, pretrained_name="all-MiniLM-L6-v2", lr=10e-2, loss_fn="mnr"):
        super().__init__(pretrained_name, lr, loss_fn)

    def configure_head(self, input_embedding_size: int) -> EncoderHead:
        return WideningHead(input_embedding_size=input_embedding_size)
