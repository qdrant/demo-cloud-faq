import pytest

import torch

from faq.utils.metrics import retrieval_reciprocal_rank_2d, retrieval_precision_2d


@pytest.mark.parametrize(
    "rrk_preds, rrk_targets, expected",
    [
        (
            torch.Tensor([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]]),
            torch.Tensor([[0, 0, 1], [0, 0, 1]]),
            torch.Tensor([0.5, 1]),
        ),
    ],
)
def test_retrieval_reciprocal_rank_2d(rrk_preds, rrk_targets, expected):
    real_metric = retrieval_reciprocal_rank_2d(rrk_preds, rrk_targets)
    assert torch.equal(real_metric, expected)


@pytest.mark.parametrize(
    "rp_preds, rp_targets, expected",
    [
        (
            torch.Tensor([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]]),
            torch.Tensor([[0, 0, 1], [0, 0, 1]]),
            torch.Tensor([0.0, 1.0]),
        ),
    ],
)
def test_retrieval_precision_2d(rp_preds, rp_targets, expected):
    real_metric = retrieval_precision_2d(rp_preds, rp_targets)
    assert torch.equal(real_metric, expected)


@pytest.mark.parametrize(
    "rp_preds, rp_targets, expected",
    [
        (
            torch.Tensor([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]]),
            torch.Tensor([[0., 0., 1.], [0., 0., 0.9]]),
            torch.Tensor([0.0, 0.9]),
        ),
    ],
)
def test_retrieval_precision_2d_float_targer(rp_preds, rp_targets, expected):
    real_metric = retrieval_precision_2d(rp_preds, rp_targets)
    assert torch.equal(real_metric, expected)
