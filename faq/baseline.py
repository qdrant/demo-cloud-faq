import os
import json
from typing import List, Dict

import torch
from sentence_transformers import SentenceTransformer

from quaterion.distances import Distance
from quaterion.eval.pair import retrieval_precision, retrieval_reciprocal_rank

from faq.config import DATA_DIR


def load_data(file_path):
    questions = []
    answers = []
    with open(file_path, "r") as f:
        for j_line in f:
            line = json.loads(j_line)
            questions.append(line["question"])
            answers.append(line["answer"])
    return questions + answers


def process(texts: List[str]) -> Dict[str, float]:
    """

    Args:
        texts: list of questions and answers, first half contains questions, second - answers

    Returns:
        metrics
    """
    embeddings = model.encode(texts, convert_to_tensor=True)
    distance_matrix = Distance.get_by_name(Distance.COSINE).distance_matrix(
        embeddings
    )
    distance_matrix[
        torch.eye(embeddings.shape[0], dtype=torch.bool)
    ] = 1.0  # max cosine distance is 1.0.
    # Set max distance between element and itself to ignore

    num_of_pairs = distance_matrix.shape[0] // 2
    ones_vec = torch.tensor([1]).repeat(num_of_pairs).to(embeddings.device)

    question_to_answer_labels = torch.diag_embed(ones_vec, num_of_pairs).to(
        embeddings.device
    )  # matrix: shape (embeddings_num, embeddings_num) -  with shifted by `num_of_pairs` diagonal
    # to the right.
    # For example:
    # If a question's index is 5, then answer's index (where we just set 1 ) is `num_of_pairs` + 5
    answer_to_question_labels = torch.diag_embed(ones_vec, -num_of_pairs).to(
        embeddings.device
    )  # matrix: shape (embeddings_num, embeddings_num) -  with shifted by `num_of_pairs` diagonal
    # to the bottom.
    # For example:
    # If an answer's index is `num_of_pairs` + 5, then question's index (where we just set 1 ) is 5
    labels = question_to_answer_labels + answer_to_question_labels

    metrics = {
        "rrk": retrieval_precision(distance_matrix, labels, k=1).mean().item(),
        "rp@1": retrieval_reciprocal_rank(distance_matrix, labels).mean().item(),
    }

    return metrics


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")  # load model

    if torch.cuda.is_available():
        model.cuda()

    filename = "val_cloud_faq_dataset.jsonl"
    filepath = os.path.join(DATA_DIR, filename)
    data = load_data(filepath)
    print(process(data))

