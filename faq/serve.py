import os
import json

import torch
from quaterion_models.model import SimilarityModel
from quaterion.distances import Distance

from faq.config import DATA_DIR, ROOT_DIR


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SimilarityModel.load(os.path.join(ROOT_DIR, "servable"))
    model.to(device)
    dataset_path = os.path.join(DATA_DIR, "val_cloud_faq_dataset.jsonl")

    with open(dataset_path) as fd:
        answers = [json.loads(json_line)["answer"] for json_line in fd]

    # everything is ready, let's encode our answers
    answer_embeddings = model.encode(answers, to_numpy=False)

    # Some prepared questions and answers to ensure that our model works as intended
    questions = [
        "what is the pricing of aws lambda functions powered by aws graviton2 processors?",
        "can i run a cluster or job for a long time?",
        "what is the dell open manage system administrator suite (omsa)?",
        "what are the differences between the event streams standard and event streams enterprise plans?",
    ]
    ground_truth_answers = [
        "aws lambda functions powered by aws graviton2 processors are 20% cheaper compared to x86-based lambda functions",
        "yes, you can run a cluster for as long as is required",
        "omsa enables you to perform certain hardware configuration tasks and to monitor the hardware directly via the operating system",
        "to find out more information about the different event streams plans, see choosing your plan",
    ]

    # encode our questions and find the closest to them answer embeddings
    question_embeddings = model.encode(questions, to_numpy=False)
    distance = Distance.get_by_name(Distance.COSINE)
    question_answers_distances = distance.distance_matrix(
        question_embeddings, answer_embeddings
    )
    answers_indices = question_answers_distances.min(dim=1)[1]
    for q_ind, a_ind in enumerate(answers_indices):
        print("Q:", questions[q_ind])
        print("A:", answers[a_ind], end="\n\n")
        assert (
            answers[a_ind] == ground_truth_answers[q_ind]
        ), f"<{answers[a_ind]}> != <{ground_truth_answers[q_ind]}>"
