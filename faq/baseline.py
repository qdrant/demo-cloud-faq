import os
import json

import torch

from quaterion.loss import ContrastiveLoss
from sentence_transformers import SentenceTransformer

from faq.config import DATA_DIR
from faq.utils.utils import wrong_prediction_indices


def load_sentences(filename):
    sentences = []
    others = []
    with open(filename, "r") as f:
        for j_line in f:
            line = json.loads(j_line)
            sentences.append(line["question"])
            others.append(line["answer"])
    sentences.extend(others)
    return sentences


def map_wrong_answers(sentences, indices_filename, res_filename):
    i = 0
    with open(indices_filename, "r") as f:
        with open(res_filename, "w") as w:
            for j_line in f:
                i += 1
                line = json.loads(j_line)
                anchor = int(line["anchor"])
                wrong = int(line["wrong"])
                right = int(line["right"])
                mapped_line = {
                    "anchor": sentences[anchor],
                    "wrong": sentences[wrong],
                    "right": sentences[right],
                }
                json.dump(mapped_line, w)
                w.write("\n")
    return i


model = SentenceTransformer("all-MiniLM-L6-v2")  # load model
filename = os.path.join(DATA_DIR, "aws_cloud_faq_dataset.jsonl")
sentences_ = load_sentences(filename)  # load sentences, first half contains
# all first elements, second half contains all second elements
embeddings = model.encode(sentences_, convert_to_tensor=True)
distance_matrix = ContrastiveLoss().distance_metric(
    embeddings, embeddings, matrix=True
)  # compute quadratic matrix of distances
distance_matrix[torch.eye(embeddings.shape[0], dtype=torch.bool)] = 2.0  # max
# cosine distance is 2, set max distance between element and itself to avoid
# paying attention to it
predicted_similarity = 2.0 - distance_matrix
mutually = False  # True if we want to compare all elements with each other,
# False if we want to compare only first elements with seconds
res = wrong_prediction_indices(predicted_similarity, False)  # a tuple of
# source element's index, wrongly mapped element's index and right element
# index
indices_filename = "aws_raw_wrong_predictions.jsonl"

with open(indices_filename, "w") as f:  # store indices, may be unnecessary
    for i in range(res[0].shape[0]):
        json.dump(
            {
                "anchor": res[0][i].item(),
                "wrong": res[1][i].item(),
                "right": res[2][i].item(),
            },
            f,
        )
        f.write("\n")

res_filename = "aws_raw_wrong_sentences.jsonl"
lines_written = map_wrong_answers(sentences_, indices_filename, res_filename)

print(
    f"lines_written: {lines_written}, num of pairs: {len(sentences_) // 2}, "
    f"p@1: {1 - lines_written / (len(sentences_) // 2)}"
)
