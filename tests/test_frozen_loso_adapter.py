"""Frozen adapter invariants without importing LISA's SciPy dependency."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from scripts.exp07_frozen_calibration.frozen_loso import Adapter, parameter_digest


class SubjectLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.randn(3, 2, 4))

    def forward(self, x, subjects):
        return torch.einsum("bkj,bjt->bkt", self.A[subjects], x)


class Spatial(nn.Module):
    """Same relevant call order and return convention as upstream SpatialModule."""
    def __init__(self):
        super().__init__()
        self.self_attention = nn.Conv1d(3, 4, 1, bias=False)
        self.unmixing_layer = nn.Conv1d(4, 4, 1)
        self.subject_layer = SubjectLayer()
        self.temporal_filter = nn.Conv1d(2, 2, 3, padding=1, groups=2, bias=False)
        self.tf_gate = nn.Parameter(torch.tensor([-0.7, 0.9]))
        self.activation = nn.GELU()

    def after_branches(self, x):
        gate = self.tf_gate.sigmoid()[None, :, None]
        return self.activation(x + gate * (self.temporal_filter(x) - x))

    def forward(self, xs, get_in_out_temporal=False):
        x, subjects = xs
        x = self.subject_layer(self.unmixing_layer(self.self_attention(x)), subjects)
        if get_in_out_temporal:
            return x, self.temporal_filter(x)
        return self.after_branches(x)


class TinyLISA(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_module = Spatial()
        self.tail = nn.Sequential(nn.Conv1d(2, 3, 3, padding=1), nn.BatchNorm1d(3),
                                  nn.GELU(), nn.Conv1d(3, 4, 1), nn.BatchNorm1d(4))
        self.register_buffer("unchanging_marker", torch.tensor([7.0]))
        self.tail[1].running_mean.copy_(torch.tensor([1.2, -0.8, 0.3]))
        self.tail[1].running_var.copy_(torch.tensor([0.8, 1.4, 2.1]))

    def forward(self, xs):
        z = self.spatial_module(xs)
        return z, self.tail(z)


def make_adapter(model=None, batch_size=2):
    if model is None:
        torch.manual_seed(4)
        model = TinyLISA()
    saved = {"fit_subjects": [1, 3], "holdout": 2,
             "criterion_state_dict": {"temperature": torch.tensor(-1.2)}}
    args = SimpleNamespace(device="cpu", batch_size=batch_size, coords_dir=Path("unused"))
    return Adapter(args, saved, model=model)


def inputs():
    return torch.randn(5, 3, 11, generator=torch.Generator().manual_seed(33)) + 0.9


def test_external_donor_matches_original_source_forward():
    adapter = make_adapter()
    original = deepcopy(adapter.model)
    # deepcopy preserves hook closures, so remove the copied adapter hook only.
    original.spatial_module.subject_layer._forward_hooks.clear()
    x = inputs()
    assert adapter.source_subjects == [1, 3]
    torch.testing.assert_close(adapter.mean_weights,
                               original.spatial_module.subject_layer.A[[0, 2]].mean(0))
    with torch.inference_mode():
        for donor, subject in enumerate(adapter.source_subjects):
            adapter.reset(donor=donor)
            expected = original((x, torch.full((len(x),), subject - 1, dtype=torch.long)))[1]
            # Shared-W einsum and per-row-W einsum may choose different kernels.
            torch.testing.assert_close(adapter.predict(x), expected, rtol=1e-6, atol=1e-7)
    assert adapter.close()["neural_parameters_unchanged"]


def test_mapping_precedes_temporal_filter_gate_and_activation():
    adapter = make_adapter()
    x = inputs()
    gain = 2.3
    adapter.reset(gain=gain)
    before = adapter.branches(x).clone()
    mapping = {"matrix": np.array([[0.0, 1.0], [-1.0, 0.0]]),
               "bias": np.array([0.8, -0.3])}
    matrix = torch.tensor(mapping["matrix"], dtype=x.dtype)
    bias = torch.tensor(mapping["bias"], dtype=x.dtype)
    expected_branches = torch.einsum("ij,njt->nit", matrix, before) + bias[None, :, None]
    with torch.inference_mode():
        expected_prediction = adapter.model.tail(adapter.model.spatial_module.after_branches(expected_branches))
    adapter.reset(gain=gain, mapping=mapping)
    torch.testing.assert_close(adapter.branches(x), expected_branches)
    torch.testing.assert_close(adapter.predict(x), expected_prediction)
    adapter.close()


@pytest.mark.parametrize("blend", [0.0, 0.5, 1.0])
def test_adabn_moment_mixture_and_immutable_neural_parameters(blend):
    adapter = make_adapter(batch_size=2)
    x = inputs()
    params = parameter_digest(adapter.model)
    first_bn = adapter.model.tail[1]
    with torch.inference_mode():
        bn_input = adapter.model.tail[0](adapter.model.spatial_module.after_branches(adapter.branches(x)))
    mean = bn_input.double().mean((0, 2))
    variance = bn_input.double().var((0, 2), unbiased=False)
    source_mean, source_var = first_bn.running_mean.double().clone(), first_bn.running_var.double().clone()
    expected_mean = (1 - blend) * source_mean + blend * mean
    # Mixture variance includes the variance between source and target means.
    expected_var = ((1 - blend) * source_var + blend * variance
                    + blend * (1 - blend) * (source_mean - mean).square())
    report = adapter.adabn(x, blend)
    torch.testing.assert_close(first_bn.running_mean.double(), expected_mean, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(first_bn.running_var.double(), expected_var, rtol=1e-6, atol=1e-7)
    assert report["tail.1"]["observations"] == len(x) * x.shape[2]
    assert len(report) == 2
    assert parameter_digest(adapter.model) == params
    assert all(not m.training for m in adapter.model.modules())
    assert all(p.grad is None and not p.requires_grad for p in adapter.model.parameters())
    assert all(int(m.num_batches_tracked) == 0 for m in adapter.model.modules() if isinstance(m, nn.BatchNorm1d))
    audit = adapter.close()
    assert audit["buffers_restored"]
    for name, value in adapter.model.named_buffers():
        torch.testing.assert_close(value, adapter.source_buffers[name], rtol=0, atol=0)


def test_adabn_chunk_invariance_and_reset_between_repeats():
    left = make_adapter(batch_size=1)
    right = make_adapter(batch_size=4)
    x = inputs()
    baseline = left.predict(x).clone()
    left.adabn(x, 0.5)
    right.adabn(x, 0.5)
    for (_, a), (_, b) in zip(left.model.named_buffers(), right.model.named_buffers()):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(left.predict(x), right.predict(x), rtol=1e-5, atol=1e-6)
    left.reset()
    torch.testing.assert_close(left.predict(x), baseline, rtol=0, atol=0)
    left.close()
    right.close()
