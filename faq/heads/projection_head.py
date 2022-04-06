import torch

from quaterion_models.heads import EncoderHead


class ProjectionHead(EncoderHead):

    def __init__(self, input_embedding_size: int, output_embedding_size):
        super().__init__(input_embedding_size)
        self.output_embeddings_size = output_embedding_size
        self.fc = torch.nn.Linear(input_embedding_size, output_embedding_size)

    @property
    def output_size(self) -> int:
        return self.output_embedding_size

    def forward(self, input_vectors: torch.Tensor) -> torch.Tensor:
        """
        :param input_vectors: shape: (batch_size * vector_size)
        :return: (batch_size * vector_size)
        """
        return self.fc(input_vectors)
