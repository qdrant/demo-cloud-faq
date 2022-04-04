import os
import json

import torch
from sentence_transformers import SentenceTransformer

from quaterion.distances import Distance
from quaterion.eval.pair.retrieval_precision import retrieval_precision
from quaterion.eval.pair.retrieval_reciprocal_rank import retrieval_reciprocal_rank

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
if torch.cuda.is_available():
    model.cuda()
METRICS = {}


def process(filename):
    filename = os.path.join(DATA_DIR, filename)
    sentences_ = load_sentences(filename)  # load sentences, first half contains
    # all first elements, second half contains all second elements
    embeddings = model.encode(sentences_, convert_to_tensor=True)
    # compute quadratic matrix of similarities
    predicted_similarity = Distance.get_by_name(Distance.COSINE).similarity_matrix(
        embeddings
    )

    predicted_similarity[
        torch.eye(embeddings.shape[0], dtype=torch.bool)
    ] = 0.0  # set min similarity between element and itself to avoid paying attention to it

    num_of_pairs = predicted_similarity.shape[0] // 2
    ones_vec = torch.tensor([1]).repeat(num_of_pairs).to(embeddings.device)

    labels = torch.diag_embed(ones_vec, num_of_pairs).to(
        embeddings.device
    )  # matrix with shifted by num_of_pairs diagonal to the right. Second elements of pairs are on
    # that diagonal.
    labels[torch.diag_embed(ones_vec, -num_of_pairs) > 0] = 1  # Fill diagonal
    # with offset of num_of_pairs to the bottom. First elements of pairs are on
    # those indices.
    anchors_rows = predicted_similarity[:num_of_pairs]
    anchors_columns = anchors_rows[:, num_of_pairs:]
    others_rows = predicted_similarity[num_of_pairs:]
    others_columns = others_rows[:, :num_of_pairs]
    fetched_predicted_similarity = torch.cat([anchors_columns, others_columns])

    anchors_labels_rows = labels[:num_of_pairs]
    anchors_labels_columns = anchors_labels_rows[:, num_of_pairs:]
    others_labels_rows = labels[num_of_pairs:]
    others_labels_columns = others_labels_rows[:, :num_of_pairs]
    fetched_labels = torch.cat([anchors_labels_columns, others_labels_columns])

    metrics = {
        "rrk": retrieval_reciprocal_rank(fetched_predicted_similarity, fetched_labels).mean().item(),
        "rp@1": retrieval_precision(fetched_predicted_similarity, fetched_labels, 1).mean().item()
    }
    METRICS.update({os.path.basename(filename): metrics})
    print(metrics)
    mutually = False  # True if we want to compare all elements with each other,
    # False if we want to compare only first elements with seconds
    res = wrong_prediction_indices(predicted_similarity, mutually)  # a tuple of
    # source element's index, wrongly mapped element's index and right element
    # index

    base_name = os.path.basename(filename)
    wo_ext = os.path.splitext(base_name)[0]
    dir_wrong = os.path.join(DATA_DIR, "wrong", "baseline")
    os.makedirs(dir_wrong, exist_ok=True)
    indices_filename = os.path.join(dir_wrong, f"{wo_ext}_wrong_pred.jsonl")
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

    res_filename = os.path.join(dir_wrong, f"{wo_ext}_wrong_sent.jsonl")
    lines_written = map_wrong_answers(sentences_, indices_filename, res_filename)

    print(
        f"{filename}: lines_written: {lines_written}, num of pairs: {len(sentences_) // 2}, "
        f"p@1: {(1 - (lines_written / (len(sentences_) // 2)))}"
    )


for filename in (
    # "yandex_cloud_train_cloud_faq_dataset.jsonl",
    # "yandex_cloud_val_cloud_faq_dataset.jsonl",
    # "aws_train_cloud_faq_dataset.jsonl",
    # "aws_val_cloud_faq_dataset.jsonl",
    # "azure_train_cloud_faq_dataset.jsonl",
    # "azure_val_cloud_faq_dataset.jsonl",
    # "gcp_train_cloud_faq_dataset.jsonl",
    # "gcp_val_cloud_faq_dataset.jsonl",
    # "hetzner_train_cloud_faq_dataset.jsonl",
    # "hetzner_val_cloud_faq_dataset.jsonl",
    # "ibm_train_cloud_faq_dataset.jsonl",
    # "ibm_val_cloud_faq_dataset.jsonl",
    "train_cloud_faq_dataset.jsonl",
    "val_cloud_faq_dataset.jsonl"
):
    process(filename)


res_path = os.path.join(DATA_DIR, "results", "baseline")
os.makedirs(res_path, exist_ok=True)
with open(os.path.join(res_path, "full_ds_baseline.json"), "w") as fd:
    json.dump(METRICS, fd, indent=2)

