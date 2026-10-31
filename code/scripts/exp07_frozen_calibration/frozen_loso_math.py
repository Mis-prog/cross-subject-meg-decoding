"""Numerical calibration helpers independent of LISA, SciPy, and training code.

Arrays use (windows, channels, time). All fits use float64 and return a fixed
affine transformation; none of these functions optimize neural network weights.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math

import numpy as np


def _windows(x, name="x"):
    value = np.asarray(x, dtype=np.float64)
    if value.ndim != 3 or any(size == 0 for size in value.shape):
        raise ValueError(f"{name} must be nonempty (windows, channels, time)")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def effective_scaler(params: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Return the effective offset and scale of new or legacy scaler parameters.

    Legacy preparation applies RobustScaler followed by StandardScaler. For
    nondegenerate channels this is mathematically the ordinary population mean
    and standard deviation. Existing legacy parameters are used exactly here.
    """
    if "offset" in params and "scale" in params:
        offset = np.asarray(params["offset"], dtype=np.float64)
        scale = np.asarray(params["scale"], dtype=np.float64)
    else:
        rc = np.asarray(params["robust_center"], dtype=np.float64)
        rs = np.asarray(params["robust_scale"], dtype=np.float64)
        sm = np.asarray(params["standard_mean"], dtype=np.float64)
        ss = np.asarray(params["standard_scale"], dtype=np.float64)
        if not (rc.shape == rs.shape == sm.shape == ss.shape):
            raise ValueError("Legacy scaler vectors must have matching shapes")
        offset, scale = rc + sm * rs, rs * ss
    if offset.ndim != 1 or not len(offset) or offset.shape != scale.shape:
        raise ValueError("Scaler offset and scale must be matching nonempty vectors")
    if not np.isfinite(offset).all() or not np.isfinite(scale).all():
        raise ValueError("Scaler contains non-finite values")
    if (scale < 0).any():
        raise ValueError("Scaler scales cannot be negative")
    return offset.copy(), np.where(scale == 0, 1.0, scale)


def fit_standardizer(x) -> dict:
    """Fit channel mean/std using only the supplied calibration windows.

    Uses population variance (ddof=0). Constant channels have scale 1. Values
    must not be clipped before fitting; apply_standardizer clips afterwards.
    """
    value = _windows(x)
    offset = value.mean(axis=(0, 2))
    scale = value.std(axis=(0, 2), ddof=0)
    # A floating-point mean of identical nonbinary values can differ by one ULP,
    # leaving a spurious positive variance. Detect truly constant channels too.
    constant = value.max(axis=(0, 2)) == value.min(axis=(0, 2))
    scale[(scale == 0) | constant] = 1.0
    return {"offset": offset.tolist(), "scale": scale.tolist()}


def apply_standardizer(x, params: Mapping) -> np.ndarray:
    """Apply a fixed channel transform, clipping to [-20, 20] as in exp06."""
    value = _windows(x)
    offset, scale = effective_scaler(params)
    if value.shape[1] != offset.size:
        raise ValueError("Scaler channel count does not match input")
    result = (value - offset[None, :, None]) / scale[None, :, None]
    return np.clip(result, -20, 20).astype(np.float32)


def population_standardizer(scalers: Mapping, source_subjects: Sequence[int]) -> dict:
    """Median effective channel offsets/scales from explicit source subjects.

    Dataset scaler keys are ``sub-01``, ``sub-02``, etc. An explicit source list
    prevents the held-out subject's full-training-split scaler from leaking in.
    """
    subjects = [int(subject) for subject in source_subjects]
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("Source subjects must be nonempty and unique")
    effective = [effective_scaler(scalers[f"sub-{subject:02d}"]) for subject in subjects]
    offsets = np.stack([pair[0] for pair in effective])
    scales = np.stack([pair[1] for pair in effective])
    return {"offset": np.median(offsets, axis=0).tolist(),
            "scale": np.median(scales, axis=0).tolist()}


def _number(row: Mapping, key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key} in window {row.get('window_id')}")
    return value


def sample_blocks(rows: Sequence[Mapping], subject: int, budget: int,
                  repeats: int, seed: int, window_seconds: float = 3) -> list[list[int]]:
    """Sample complete contiguous blocks from one subject's train split only.

    A block never crosses audio file, task, session, missing audio window, or raw
    MEG discontinuity. The raw start column, when present, is in 1000 Hz samples.
    All possible block starts have equal probability. Repeats use distinct starts
    when available, otherwise sampling with replacement is necessary. The seed
    depends on subject and caller seed, not on a method or calibration budget.
    A zero-second budget yields one empty block, independent of repeats.
    """
    if not isinstance(repeats, (int, np.integer)) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not math.isfinite(float(window_seconds)) or window_seconds <= 0:
        raise ValueError("window_seconds must be positive and finite")
    if not math.isfinite(float(budget)) or budget < 0:
        raise ValueError("budget must be nonnegative and finite")
    count = float(budget) / float(window_seconds)
    if not math.isclose(count, round(count), rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"budget {budget} must be divisible by window_seconds {window_seconds}")
    count = int(round(count))
    if count == 0:
        return [[]]
    if int(subject) < 0 or int(seed) < 0:
        raise ValueError("subject and seed must be nonnegative")
    groups = defaultdict(list)
    ids = set()
    for row in rows:
        if int(row["subject"]) != int(subject) or row["split"] != "train":
            continue
        window_id = int(row["window_id"])
        if window_id in ids:
            raise ValueError(f"Duplicate window_id {window_id} for subject {subject}")
        ids.add(window_id)
        groups[(str(row.get("session", "")), str(row.get("task", "")),
                str(row["audio_file"]))].append(row)

    candidates, longest = [], 0
    raw_key = "meg_start_sample_1000hz"

    def append_run(run):
        nonlocal longest
        longest = max(longest, len(run))
        for start in range(len(run) - count + 1):
            candidates.append([int(row["window_id"]) for row in run[start:start + count]])

    for key in sorted(groups):
        group = sorted(groups[key], key=lambda r: (_number(r, "audio_start_seconds"), int(r["window_id"])))
        run = []
        for row in group:
            adjacent = True
            if run:
                previous = run[-1]
                delta = _number(row, "audio_start_seconds") - _number(previous, "audio_start_seconds")
                adjacent = math.isclose(delta, window_seconds, rel_tol=0, abs_tol=1e-7)
                present = row.get(raw_key) not in (None, "")
                previous_present = previous.get(raw_key) not in (None, "")
                if present != previous_present:
                    adjacent = False
                elif present:
                    raw_delta = _number(row, raw_key) - _number(previous, raw_key)
                    adjacent = adjacent and math.isclose(raw_delta, window_seconds * 1000,
                                                         rel_tol=0, abs_tol=1e-7)
            if not adjacent:
                append_run(run)
                run = []
            run.append(row)
        append_run(run)
    if not candidates:
        raise ValueError(f"Subject {subject}: no contiguous {budget:g}s train block; "
                         f"longest available is {longest * window_seconds:g}s. "
                         "Blocks cannot cross audio fragments, tasks, sessions, or recording gaps.")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(subject)]))
    selected = rng.choice(len(candidates), size=repeats, replace=len(candidates) < repeats)
    return [candidates[int(index)] for index in selected]


def fit_branch_map(x, y, kind: str) -> dict:
    """Fit target ``x`` to paired source reference ``y`` before temporal filters.

    Each matched time sample is an observation. All fits include a translation.
    ``scalar`` and ``diagonal`` are centered ordinary least squares; coefficients
    are unconstrained and may be negative. ``procrustes`` fits an orthogonal
    matrix in O(K), allowing reflection, without scale. Unidentifiable scalar or
    diagonal coefficients use 1; entirely zero cross-covariance uses identity.
    Output convention is ``output[n] = matrix @ x[n] + bias[:, None]``.
    """
    target, reference = _windows(x, "x"), _windows(y, "y")
    if target.shape != reference.shape:
        raise ValueError("Target and reference arrays must have identical paired shapes")
    if kind not in {"scalar", "diagonal", "procrustes"}:
        raise ValueError(f"Unknown branch map kind: {kind}")
    channels = target.shape[1]
    a = target.transpose(1, 0, 2).reshape(channels, -1)
    b = reference.transpose(1, 0, 2).reshape(channels, -1)
    mean_a, mean_b = a.mean(axis=1), b.mean(axis=1)
    constant_a = np.ptp(a, axis=1) == 0
    constant_b = np.ptp(b, axis=1) == 0
    a, b = a - mean_a[:, None], b - mean_b[:, None]
    a[constant_a] = 0
    b[constant_b] = 0
    if kind == "scalar":
        energy = np.einsum("ij,ij->", a, a)
        gain = np.einsum("ij,ij->", a, b) / energy if energy > 0 else 1.0
        matrix = np.eye(channels) * gain
    elif kind == "diagonal":
        energy = np.einsum("ij,ij->i", a, a)
        numerator = np.einsum("ij,ij->i", a, b)
        gains = np.divide(numerator, energy, out=np.ones(channels), where=energy > 0)
        matrix = np.diag(gains)
    else:
        cross = b @ a.T
        if not np.any(cross):
            matrix = np.eye(channels)
        else:
            u, _, vh = np.linalg.svd(cross, full_matrices=False)
            matrix = u @ vh
    bias = mean_b - matrix @ mean_a
    if not np.isfinite(matrix).all() or not np.isfinite(bias).all():
        raise ValueError("Branch-map fit produced non-finite parameters")
    return {"kind": kind, "matrix": matrix, "bias": bias}


def apply_branch_map(x, mapping: Mapping) -> np.ndarray:
    """Apply a fitted affine branch map; calculations/output use float64."""
    value = _windows(x)
    matrix = np.asarray(mapping["matrix"], dtype=np.float64)
    bias = np.asarray(mapping["bias"], dtype=np.float64)
    channels = value.shape[1]
    if matrix.shape != (channels, channels) or bias.shape != (channels,):
        raise ValueError("Branch map shape does not match channel count")
    if not np.isfinite(matrix).all() or not np.isfinite(bias).all():
        raise ValueError("Branch map contains non-finite parameters")
    return np.einsum("ij,njt->nit", matrix, value) + bias[None, :, None]


def source_norm_gain(source_ws) -> float:
    """RMS source Frobenius norm divided by Frobenius norm of mean source W_s."""
    weights = _windows(source_ws, "source_ws")
    mean_norm = np.linalg.norm(weights.mean(axis=0))
    if mean_norm == 0:
        raise ValueError("Mean source W_s is zero; norm correction is undefined")
    rms = np.sqrt(np.square(weights).sum(axis=(1, 2)).mean())
    gain = float(rms / mean_norm)
    if not math.isfinite(gain):
        raise ValueError("Source norm correction is non-finite")
    return gain
