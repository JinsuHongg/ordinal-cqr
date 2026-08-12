import torch
from torch import nn

from ordinal_cqr.explainability.poshoc_uc import APSWrapper, COPOCWrapper, _aps_scores, _exact_augmented_quantile
from ordinal_cqr.models.backbone import (
    COPOCUnimodalHead,
    ResNet18BinomialCls,
    ResNet18COPOC,
    is_unimodal_probabilities,
)


def _contiguous(mask: torch.Tensor) -> bool:
    active = mask.nonzero(as_tuple=False).flatten()
    return bool(len(active) and torch.equal(active, torch.arange(active[0], active[-1] + 1)))


def test_eq5_head_has_expected_transform_and_gradients():
    eta = torch.tensor([[1.0, -2.0, 3.0, -4.0]], requires_grad=True)
    logits = COPOCUnimodalHead()(eta)
    assert torch.equal(logits, torch.tensor([[-1.0, -3.0, -6.0, -10.0]]))
    probabilities = logits.softmax(dim=1)
    assert probabilities.shape == (1, 4)
    assert torch.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(1))
    assert is_unimodal_probabilities(probabilities).all()
    probabilities[:, 0].sum().backward()
    assert eta.grad is not None and torch.isfinite(eta.grad).all()


def test_copoc_backbone_emits_k_logits_and_is_unimodal():
    model = ResNet18COPOC(in_channels=3, time_steps=1, num_classes=5)
    assert model.latent.out_features == 5
    logits = model(torch.randn(2, 3, 1, 64, 64))
    assert logits.shape == (2, 5)
    assert is_unimodal_probabilities(logits.softmax(dim=1)).all()


def test_eq5_head_is_not_the_legacy_binomial_parameterization():
    eta = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0]])
    copoc_probs = COPOCUnimodalHead()(eta).softmax(dim=1)
    # Every legacy output has logits log C(4,k) + k z for one scalar z.
    legacy = ResNet18BinomialCls(num_classes=5)
    for z in torch.linspace(-5, 5, 101):
        binomial_probs = (legacy.log_comb + legacy.k_vals * z).softmax(dim=0)
        assert not torch.allclose(copoc_probs[0], binomial_probs, atol=1e-5)


class _FixedCOPOC(nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, x):
        return self.logits[: x.shape[0]]


def test_copoc_uses_pooled_true_label_aps_and_contiguous_inference_sets():
    logits = COPOCUnimodalHead()(torch.tensor([[0.1, 0.3, -0.2, 0.4, -0.1]]))
    model = _FixedCOPOC(logits)
    wrapper = COPOCWrapper(model, num_classes=5, alpha=0.5)
    x = torch.zeros(1, 3, 1, 4, 4)
    y = torch.tensor([2])
    wrapper.calibrate([(x, y)])
    probs = logits.softmax(dim=1)
    expected = _aps_scores(probs)[0, y[0]]
    assert torch.allclose(wrapper.q_hat, expected)
    first = wrapper.predict_step((x, torch.tensor([0])), 0)["prediction_set"]
    second = wrapper.predict_step((x, torch.tensor([4])), 0)["prediction_set"]
    assert torch.equal(first, second)  # inference does not use test labels
    assert _contiguous(first[0])


def test_copoc_aps_tie_rule_is_deterministic_and_contiguous():
    # A flat strict-unimodal tie plateau is resolved by stable class-index order.
    logits = torch.zeros(1, 5)
    wrapper = COPOCWrapper(_FixedCOPOC(logits), num_classes=5, alpha=0.5)
    wrapper.q_hat.fill_(0.45)
    prediction_set = wrapper.predict_step((torch.zeros(1, 3, 1, 4, 4), torch.tensor([3])), 0)["prediction_set"]
    assert prediction_set.tolist() == [[True, True, True, False, False]]
    assert _contiguous(prediction_set[0])


def test_aps_uses_exact_quantile_and_inverts_its_saved_score():
    scores = torch.tensor([0.2, 0.6, 0.9])
    assert torch.allclose(_exact_augmented_quantile(scores, alpha=0.5), torch.tensor(0.6))
    assert torch.isinf(_exact_augmented_quantile(scores[:1], alpha=0.1))

    logits = torch.log(torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]]))
    wrapper = APSWrapper(_FixedCOPOC(logits), num_classes=3, alpha=0.5)
    x = torch.zeros(2, 3, 1, 4, 4)
    z = torch.tensor([42.0, 63.0])
    y = torch.tensor([0, 1])
    wrapper.calibrate([(x, z, y)])
    assert torch.allclose(wrapper.q_hat, torch.tensor(0.7))
    output = wrapper.predict_step((x, z, y), 0)
    prediction_set = output["prediction_set"]
    assert prediction_set.tolist() == [[True, False, False], [False, True, False]]
    assert output["target"].tolist() == [0, 1]
