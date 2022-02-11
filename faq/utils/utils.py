import torch

from torch.utils.data import get_worker_info


def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    dataset.worker_id = worker_id
    dataset.total_workers = worker_info.num_workers

    dataset_size = worker_info.dataset.size
    batch_size = dataset_size // worker_info.num_workers
    dataset.start_index = batch_size * worker_id
    dataset.end_index = dataset.start_index + batch_size
    if dataset.worker_id == dataset.total_workers - 1:
        dataset.end_index = dataset_size


def wrong_prediction_indices(preds, mutually=False):
    num_of_pairs = preds.shape[0] // 2
    ones_vec = torch.tensor([1]).repeat(num_of_pairs)

    labels = torch.diag_embed(ones_vec, num_of_pairs)  # matrix with shifted
    # by num_of_pairs diagonal to the right. Second elements of pairs are on
    # that diagonal.
    labels[torch.diag_embed(ones_vec, -num_of_pairs) > 0] = 1  # Fill diagonal
    # with offset of num_of_pairs to the bottom. First elements of pairs are on
    # those indices.

    if not mutually:
        preds = preds[:num_of_pairs, num_of_pairs:]  # rows with first elements
        # and columns with second elements

    indices = preds.topk(1)[1].squeeze()  # indices of elements with max
    # probability
    first_half = torch.arange(0, num_of_pairs)  # indices of first elements

    if not mutually:
        incorrect_first_elem_pairs = indices[:num_of_pairs] != torch.arange(
            0, num_of_pairs
        )

        anchors = first_half[incorrect_first_elem_pairs]  # indices of first
        # elements for which we found wrong pairs
        others = (
            indices[:num_of_pairs][incorrect_first_elem_pairs] + num_of_pairs
        )  # indices of second elements which were wrongly mapped
        right = first_half[incorrect_first_elem_pairs] + num_of_pairs
        # indices of correct second elements
    else:
        second_half = torch.arange(num_of_pairs, num_of_pairs * 2)  # indices
        # of second elements
        incorrect_first_elem_pairs = indices[:num_of_pairs] != torch.arange(
            num_of_pairs, num_of_pairs * 2
        )  # mask of incorrectly mapped first elements
        incorrect_second_elem_pairs = indices[num_of_pairs:] != torch.arange(
            0, num_of_pairs
        )  # mask of incorrectly mapped second elements

        anchors = torch.cat(
            [
                first_half[incorrect_first_elem_pairs],
                second_half[incorrect_second_elem_pairs],
            ]
        )
        others = torch.cat(
            [
                indices[:num_of_pairs][incorrect_first_elem_pairs],
                indices[num_of_pairs:][incorrect_second_elem_pairs],
            ]
        )
        right = torch.cat(
            [
                first_half[incorrect_first_elem_pairs] + num_of_pairs,
                second_half[incorrect_second_elem_pairs] - num_of_pairs,
            ]
        )

    return anchors, others, right


if __name__ == "__main__":

    preds = torch.tensor(
        [
            [0.0000, 0.0874, -0.1275, 0.4034, 0.01, 0.0759, -0.0441, 0.4855],
            [0.9874, 0.0000, 0.3433, 0.0778, 0.2135, 0.9168, 0.1264, 0.1650],
            [
                -0.1275,
                0.3433,
                0.0000,
                -0.0047,
                -0.0482,
                0.2804,
                0.4316,
                -0.0085,
            ],
            [0.4034, 0.0778, -0.0047, 0.0000, 0.0581, 0.0426, 0.0119, 0.6140],
            [0.5641, 0.2135, -0.0482, 0.0581, 0.0000, 0.2038, -0.0381, 0.2507],
            [0.0759, 0.9168, 0.92, 0.0426, 0.2038, 0.0000, 0.1595, 0.1017],
            [
                -0.0441,
                0.1264,
                0.4316,
                0.0119,
                -0.0381,
                0.1595,
                0.0000,
                -0.0504,
            ],
            [0.4855, 0.1650, -0.0085, 0.6140, 0.2507, 0.1017, -0.0504, 0.0000],
        ]
    )
    wrong_prediction_indices(preds, True)
