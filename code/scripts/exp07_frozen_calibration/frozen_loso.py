"""Frozen LISA calibration. No optimizer, backward pass, or checkpoint mutation.

Run through run_frozen_loso.py; see experiments/exp07_frozen_calibration/README.md.
Imports of LISA are deliberately lazy, so --help and numerical tests work offline.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "tune", "choose", "evaluate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--data-dir", type=Path, default=ROOT / "data/pilot/loso")
        sub.add_argument("--runs-dir", type=Path, default=ROOT / "data/pilot/loso_runs")
        sub.add_argument("--output-dir", type=Path, default=ROOT / "experiments/exp07_frozen_calibration/results")
        sub.add_argument("--protocol", type=Path, default=ROOT / "experiments/exp07_frozen_calibration/protocol.json")
        sub.add_argument("--coords-dir", type=Path, default=ROOT / "code/LISA/data/preprocessed")
        sub.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        sub.add_argument("--batch-size", type=int, default=64)
        if command in ("tune", "evaluate"):
            sub.add_argument("--holdout", type=int, required=True)
            sub.add_argument("--seed", type=int, default=0)
    return parser


# Help/argument errors must not require GPU libraries or OpenMP shared memory.
_CLI_ARGS = make_parser().parse_args() if __name__ == "__main__" else None

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fqw-mpl")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F


# Support direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exp07_frozen_calibration.frozen_loso_math import (
    apply_standardizer, effective_scaler, fit_branch_map, fit_standardizer,
    population_standardizer, sample_blocks, source_norm_gain,
)

PREPARED = ("meg_unscaled.npy", "meg_windows.npy", "audio_embeddings.npy",
            "meg_windows.csv", "audio_windows.csv", "scalers.json",
            "channels.json", "fragments.json", "protocol.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def digest_json(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def code_digest():
    files = [Path(__file__), Path(__file__).with_name("frozen_loso_math.py")]
    return digest_json({p.name: sha256(p) for p in files})


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_path(args, holdout, seed):
    return args.runs_dir / f"shared_h{holdout:02d}_s{seed}.pt"


def validate_checkpoint(saved, holdout, seed, protocol, original_sha):
    if saved.get("kind") != "shared" or saved.get("ea_transforms") is not None:
        raise ValueError("Only original shared checkpoints without EA are supported")
    if saved.get("holdout") != holdout or saved.get("seed") != seed:
        raise ValueError("Checkpoint holdout/seed does not match its requested job")
    expected = set(protocol["subjects"]) - {holdout}
    fitted = saved.get("fit_subjects", [])
    if len(fitted) != len(expected) or set(fitted) != expected:
        raise ValueError("Checkpoint fit_subjects must exclude exactly the held-out subject")
    if saved.get("protocol_sha256") != original_sha:
        raise ValueError("Checkpoint and prepared data have different original protocols")
    config = saved["config"]
    if config.get("feature_reduction") != "none" or config.get("time_reduction") != "none":
        raise ValueError("This series requires the original unreduced audio embedding space")
    if not config.get("use_subject_layer") or config.get("use_spatial_attention") != "3D":
        raise ValueError("Expected the original 3D attention + subject-layer architecture")
    weights = saved["model_state_dict"]["spatial_module.subject_layer.A"]
    if tuple(weights.shape) != (27, 25, 270) or not torch.isfinite(weights).all():
        raise ValueError("Unexpected/non-finite subject matrices; expected (27,25,270)")


def validate_protocol(protocol):
    # Deliberately narrow: changing the scientific design requires a new series.
    if (protocol.get("subjects") != list(range(1, 28))
            or protocol.get("dev_subjects") != [1, 2, 3]
            or protocol.get("seeds") != [0, 1, 2]
            or protocol.get("budgets_seconds") != [0, 30, 120]
            or protocol.get("repeats") != 3 or protocol.get("window_seconds") != 3
            or protocol.get("tune_seed") != 0
            or protocol.get("gain_rules") != ["1", "2", "4", "source_norm"]
            or protocol.get("bn_blends") != [0.25, 0.5, 1.0]):
        raise ValueError("Protocol differs from the supported frozen_loso_v1 design")


class Data:
    """Read-only mmap access. Target samples are indexed explicitly by split."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.rows = read_rows(self.directory / "meg_windows.csv")
        self.audio_rows = read_rows(self.directory / "audio_windows.csv")
        self.scalers = read_json(self.directory / "scalers.json")
        self.original_protocol = read_json(self.directory / "protocol.json")
        self.scaled = np.load(self.directory / "meg_windows.npy", mmap_mode="r")
        self.raw = np.load(self.directory / "meg_unscaled.npy", mmap_mode="r")
        self.audio = np.load(self.directory / "audio_embeddings.npy", mmap_mode="r")
        self.by_subject_audio = {}
        self.validate()

    def validate(self):
        if (self.scaled.shape != (len(self.rows), 208, 300)
                or self.raw.shape != self.scaled.shape
                or self.audio.shape != (len(self.audio_rows), 149, 768)):
            raise ValueError("Unexpected array shapes: need MEG[N,208,300], audio[A,149,768]")
        if any(a.dtype != np.float32 for a in (self.raw, self.scaled, self.audio)):
            raise ValueError("Prepared arrays must be float32")
        for i, row in enumerate(self.audio_rows):
            if int(row["audio_id"]) != i:
                raise ValueError("audio_id must index audio_embeddings.npy in row order")
            if self.original_protocol["splits"][row["audio_file"]] != row["split"]:
                raise ValueError("Audio table differs from original split protocol")
        for i, row in enumerate(self.rows):
            s, a = int(row["subject"]), int(row["audio_id"])
            if int(row["window_id"]) != i or int(row["model_subject_index"]) != s - 1:
                raise ValueError("Window/subject indices are not the original LOSO layout")
            audio = self.audio_rows[a]
            if any(row[key] != audio[key] for key in ("audio_file", "split")):
                raise ValueError("MEG/audio split or filename mismatch")
            if float(row["audio_start_seconds"]) != float(audio["audio_start_seconds"]):
                raise ValueError("MEG/audio time mismatch")
            if (s, a) in self.by_subject_audio:
                raise ValueError("Duplicate subject/audio row; template matching would be ambiguous")
            self.by_subject_audio[s, a] = i
        if set(int(r["subject"]) for r in self.rows) != set(range(1, 28)):
            raise ValueError("Full 27-subject prepared arrays are required (not mini_loso)")
        for s in range(1, 28):
            for split, expected in (("train", 294), ("validation", 76), ("test", 99)):
                if len(self.ids(s, split)) != expected:
                    raise ValueError(f"sub-{s:02d}: unexpected {split} count")
            if any((s, a) not in self.by_subject_audio for a in range(len(self.audio_rows))):
                raise ValueError(f"sub-{s:02d}: missing shared audio windows")
            scaler = self.scalers[f"sub-{s:02d}"]
            offset, scale = effective_scaler(scaler)
            if offset.shape != (208,) or scale.shape != (208,):
                raise ValueError(f"sub-{s:02d}: wrong source scaler dimensions")
            # Full exp06 scalers omit IDs; their train-only provenance is bound by
            # the preparation receipt's exact file hash, verified in check().
            if "fit_window_ids" in scaler and set(scaler["fit_window_ids"]) != set(self.ids(s, "train")):
                raise ValueError(f"sub-{s:02d}: source scaler not fitted exactly on train windows")

    def ids(self, subject, split):
        return [int(r["window_id"]) for r in self.rows
                if int(r["subject"]) == subject and r["split"] == split]

    def audio_ids(self, ids):
        return [int(self.rows[i]["audio_id"]) for i in ids]

    def bank_ids(self, split):
        return [int(r["audio_id"]) for r in self.audio_rows if r["split"] == split]

    def matched_source_ids(self, source, target_ids):
        # Do not sort: tensor time/order must match the target block exactly.
        return [self.by_subject_audio[source, a] for a in self.audio_ids(target_ids)]


def file_record(path, previous=None):
    path = Path(path).resolve()
    stat = path.stat()
    signature = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    # First check hashes all bytes. Resume uses unchanged size + nanosecond mtime.
    if previous and all(previous.get(k) == v for k, v in signature.items()):
        return previous
    return {**signature, "sha256": sha256(path)}


def upstream_state():
    directory = ROOT / "code/LISA"
    revision = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain",
                                     "--untracked-files=no"], text=True).strip()
    if dirty:
        raise ValueError("Tracked LISA files have local changes; use the pinned upstream checkout")
    return revision


def environment(args):
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "cuda": torch.version.cuda, "batch_size": args.batch_size}


def manifest_identity(manifest):
    # Paths and mtimes can change when transferring files; content identity cannot.
    return digest_json({"files": {k: v["sha256"] for k, v in manifest["files"].items()},
                        "protocol_sha256": manifest["protocol_sha256"],
                        "code_sha256": manifest["code_sha256"],
                        "upstream_commit": manifest["upstream_commit"]})


def check(args, protocol):
    paths = {f"data/{name}": args.data_dir / name for name in PREPARED}
    report_path = ROOT / protocol["preparation_report"]
    paths["preparation_report"] = report_path
    for name in ("sensor_xyz.npy", "coords208_xy_scaled.npy"):
        paths[f"coords/{name}"] = args.coords_dir / "coords" / name
    for subject in protocol["subjects"]:
        seeds = [protocol["tune_seed"]] if subject in protocol["dev_subjects"] else protocol["seeds"]
        for seed in seeds:
            paths[f"checkpoint/{subject}/{seed}"] = checkpoint_path(args, subject, seed)
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required input files are missing:\n" + "\n".join(missing)
                                + "\nCopy the exp06 prepared arrays/checkpoints; no training is started.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install the GPU environment or explicitly use --device cpu.")
    original_sha = sha256(args.data_dir / "protocol.json")
    report = read_json(report_path)
    if report["protocol_sha256"] != original_sha:
        raise ValueError("Preparation report and data protocol hashes disagree")
    revision = upstream_state()
    if revision != protocol["upstream_commit"] or revision != report["upstream_commit"]:
        raise ValueError("LISA checkout differs from the checkpoint preparation revision")
    path = args.output_dir / "manifest.json"
    previous = read_json(path).get("files", {}) if path.exists() else {}
    files = {}
    for key, item in paths.items():
        files[key] = file_record(item, previous.get(key))
        if key.startswith("data/") and files[key]["sha256"] != report["prepared_sha256"][item.name]:
            raise ValueError(f"Prepared artifact differs from exp06 preparation receipt: {item.name}")
    data = Data(args.data_dir)
    for subject in protocol["subjects"]:
        sample_blocks(data.rows, subject, max(protocol["budgets_seconds"]), protocol["repeats"], subject)
        seeds = [0] if subject in protocol["dev_subjects"] else protocol["seeds"]
        for seed in seeds:
            saved = torch.load(checkpoint_path(args, subject, seed), map_location="cpu", weights_only=False)
            validate_checkpoint(saved, subject, seed, protocol, original_sha)
    manifest = {"schema_version": 1, "files": files, "protocol_sha256": sha256(args.protocol),
                "code_sha256": code_digest(), "upstream_commit": revision}
    manifest["fingerprint"] = manifest_identity(manifest)
    # A tiny real forward pass detects dependency/config/device issues before the queue.
    adapter = Adapter(args, torch.load(checkpoint_path(args, 1, 0), map_location="cpu", weights_only=False))
    x = torch.zeros(2, 208, 300, device=adapter.device)
    with torch.inference_mode():
        output = adapter.predict(x)
    if tuple(output.shape) != (2, 768, 149) or not torch.isfinite(output).all():
        raise ValueError("Checkpoint smoke inference produced invalid embeddings")
    adapter.close()
    manifest["checked_environment"] = environment(args)
    atomic_json(path, manifest)
    print(json.dumps({"status": "passed", "checkpoints": 75, "meg_windows": len(data.rows),
                      "audio_windows": len(data.audio_rows), "manifest": str(path)}), flush=True)


def load_manifest(args):
    path = args.output_dir / "manifest.json"
    if not path.exists():
        raise ValueError("Run the check stage first; manifest.json is missing")
    manifest = read_json(path)
    if (manifest["protocol_sha256"] != sha256(args.protocol)
            or manifest["code_sha256"] != code_digest()
            or manifest["fingerprint"] != manifest_identity(manifest)):
        raise ValueError("Code/protocol/manifest changed: rerun check, use a new output directory for a new series")
    for record in manifest["files"].values():
        stat = Path(record["path"]).stat()
        if stat.st_size != record["size"] or stat.st_mtime_ns != record["mtime_ns"]:
            raise ValueError(f"Input changed since check: {record['path']}; rerun check")
    for name in PREPARED:
        if str((args.data_dir / name).resolve()) != manifest["files"][f"data/{name}"]["path"]:
            raise ValueError("--data-dir differs from the checked manifest")
    for name in ("sensor_xyz.npy", "coords208_xy_scaled.npy"):
        if str((args.coords_dir / "coords" / name).resolve()) != manifest["files"][f"coords/{name}"]["path"]:
            raise ValueError("--coords-dir differs from the checked manifest")
    if upstream_state() != manifest["upstream_commit"]:
        raise ValueError("Upstream checkout changed since check")
    return manifest


class Adapter:
    """External subject projection and branch map; original network stays immutable."""

    def __init__(self, args, saved, model=None):
        self.device = torch.device(args.device)
        self.batch_size = args.batch_size
        if model is None:
            from lisa.model.load_model import load_from_config
            config = copy.deepcopy(saved["config"])
            config["dirprocess"] = str(args.coords_dir)
            model, _ = load_from_config(config, None, self.device)
            model.load_state_dict(saved["model_state_dict"], strict=True)
        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.source_subjects = sorted(saved["fit_subjects"])
        self.holdout = saved["holdout"]
        self.source_weights = self.model.spatial_module.subject_layer.A.detach()[
            torch.tensor([s - 1 for s in self.source_subjects], device=self.device)].clone()
        self.mean_weights = self.source_weights.mean(0)
        self.weight = self.mean_weights
        self.mapping = None
        self.hook = self.model.spatial_module.subject_layer.register_forward_hook(self._replace)
        self.source_buffers = {name: b.detach().clone() for name, b in self.model.named_buffers()}
        self.parameter_hash = parameter_digest(self.model)
        self.temperature = float(saved["criterion_state_dict"]["temperature"].exp())
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid checkpoint temperature")

    def _replace(self, module, inputs, output):
        result = torch.einsum("kj,njt->nkt", self.weight, inputs[0])
        if self.mapping is not None:
            matrix, bias = self.mapping
            result = torch.einsum("ij,njt->nit", matrix, result) + bias[None, :, None]
        return result

    @torch.no_grad()
    def reset(self, gain=1.0, donor=None, mapping=None):
        for name, buffer in self.model.named_buffers():
            buffer.copy_(self.source_buffers[name])
        self.weight = self.mean_weights * float(gain) if donor is None else self.source_weights[donor]
        self.mapping = None if mapping is None else (
            torch.as_tensor(mapping["matrix"], dtype=torch.float32, device=self.device),
            torch.as_tensor(mapping["bias"], dtype=torch.float32, device=self.device))
        self.model.eval()

    @torch.inference_mode()
    def predict(self, x):
        subject = torch.full((len(x),), self.holdout - 1, dtype=torch.long, device=self.device)
        return self.model((x, subject))[1]

    @torch.inference_mode()
    def branches(self, x):
        subject = torch.full((len(x),), self.holdout - 1, dtype=torch.long, device=self.device)
        return self.model.spatial_module((x, subject), get_in_out_temporal=True)[0]

    @torch.inference_mode()
    def scores(self, x, candidates):
        result = []
        for start in range(0, len(x), self.batch_size):
            predicted = self.predict(x[start:start + self.batch_size])
            result.append(F.normalize(predicted.flatten(1), dim=1) @ candidates.T)
        output = torch.cat(result)
        if not torch.isfinite(output).all():
            raise ValueError("Non-finite retrieval scores")
        return output

    @torch.inference_mode()
    def adabn(self, x, blend):
        """Sequential input moments: downstream layers see already-calibrated upstream BN."""
        if not 0 <= blend <= 1 or len(x) == 0:
            raise ValueError("AdaBN needs nonempty calibration data and blend in [0,1]")
        saved = {}
        modules = [(n, m) for n, m in self.model.named_modules()
                   if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        for name, module in modules:
            if not module.track_running_stats:
                raise ValueError("AdaBN requires BatchNorm with stored running statistics")
            moments = {"n": 0, "sum": None, "squares": None}

            def collect(layer, inputs):
                values = inputs[0].detach().double().movedim(1, 0).flatten(1)
                part, squares = values.sum(1), values.square().sum(1)
                moments["sum"] = part if moments["sum"] is None else moments["sum"] + part
                moments["squares"] = squares if moments["squares"] is None else moments["squares"] + squares
                moments["n"] += values.shape[1]

            hook = module.register_forward_pre_hook(collect)
            try:
                for start in range(0, len(x), self.batch_size):
                    self.predict(x[start:start + self.batch_size])
            finally:
                hook.remove()
            mu = moments["sum"] / moments["n"]
            var = (moments["squares"] / moments["n"] - mu.square()).clamp_min(0)
            source_mu = self.source_buffers[f"{name}.running_mean"].double()
            source_var = self.source_buffers[f"{name}.running_var"].double()
            mixed_mu = (1 - blend) * source_mu + blend * mu
            mixed_var = ((1 - blend) * (source_var + source_mu.square())
                         + blend * (var + mu.square()) - mixed_mu.square()).clamp_min(0)
            if not torch.isfinite(mixed_mu).all() or not torch.isfinite(mixed_var).all():
                raise ValueError("Non-finite AdaBN moments")
            module.running_mean.copy_(mixed_mu)
            module.running_var.copy_(mixed_var)
            saved[name] = {"mean": mixed_mu.cpu().tolist(), "variance": mixed_var.cpu().tolist(),
                           "observations": moments["n"]}
        return saved

    def close(self):
        if parameter_digest(self.model) != self.parameter_hash:
            raise RuntimeError("Neural parameters changed during frozen calibration")
        allowed = {n for n in self.source_buffers if n.endswith(("running_mean", "running_var"))}
        for name, buffer in self.model.named_buffers():
            if name not in allowed and not torch.equal(buffer, self.source_buffers[name]):
                raise RuntimeError(f"Unexpected mutation of buffer {name}")
        self.reset()
        self.hook.remove()
        return {"neural_parameters_unchanged": True, "parameter_sha256": self.parameter_hash,
                "allowed_buffer_changes": sorted(allowed), "buffers_restored": True}


def gain_value(rule, adapter):
    return (source_norm_gain(adapter.source_weights.cpu().numpy())
            if rule == "source_norm" else float(rule))


def metrics(scores, audio_ids, bank_ids, temperature):
    target_np = np.searchsorted(bank_ids, audio_ids)
    if (target_np >= len(bank_ids)).any() or not np.array_equal(np.asarray(bank_ids)[target_np], audio_ids):
        raise ValueError("Query label missing from the split-specific candidate gallery")
    target = torch.as_tensor(target_np, device=scores.device)
    ranks = (scores > scores.gather(1, target[:, None])).sum(1) + 1
    return {"loss": F.cross_entropy(scores / temperature, target).item(),
            "top1": (ranks == 1).float().mean().item(),
            "top10": (ranks <= 10).float().mean().item(),
            "mean_rank": ranks.float().mean().item(), "queries": len(target), "candidates": len(bank_ids)}


def synchronized_time(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def candidates(data, split, device):
    ids = data.bank_ids(split)
    values = torch.as_tensor(np.array(data.audio[ids]), device=device).transpose(1, 2)
    return ids, F.normalize(values.flatten(1), dim=1)


def receipt_path(args, stage, holdout, seed):
    return args.output_dir / stage / f"h{holdout:02d}_s{seed}.json"


def job_identity(args, manifest, tuning_sha=None):
    return {"stage": args.command, "holdout": args.holdout, "seed": args.seed,
            "manifest_fingerprint": manifest["fingerprint"], "environment": environment(args),
            "tuning_sha256": tuning_sha}


def load_tuning(args, manifest):
    path = args.output_dir / "tuning.json"
    tuning = read_json(path)
    if (tuning.get("status") != "complete" or tuning.get("manifest_fingerprint") != manifest["fingerprint"]
            or tuning.get("protocol_sha256") != sha256(args.protocol)):
        raise ValueError("Tuning does not match this checked series")
    for relpath, checksum in tuning["tuning_receipts"].items():
        if sha256(args.output_dir / relpath) != checksum:
            raise ValueError("A dev receipt changed after hyperparameters were chosen")
    return tuning, sha256(path)


def try_resume(path, fingerprint):
    if not path.exists():
        return False
    old = read_json(path)
    if old.get("status") != "complete" or old.get("fingerprint") != fingerprint:
        raise ValueError(f"Existing output has different provenance: {path}. Use a new --output-dir.")
    integrity = old.get("receipt_sha256")
    if integrity != digest_json({k: v for k, v in old.items() if k != "receipt_sha256"}):
        raise ValueError(f"Receipt integrity check failed: {path}")
    print(json.dumps({"status": "already_complete", "receipt": str(path)}), flush=True)
    return True


def finish_receipt(path, receipt):
    receipt["receipt_sha256"] = digest_json(receipt)
    atomic_json(path, receipt)


def run_job(args, protocol):
    dev = args.holdout in protocol["dev_subjects"]
    if args.holdout not in protocol["subjects"] or args.seed not in protocol["seeds"]:
        raise ValueError("Unknown holdout/seed")
    if args.command == "tune" and (not dev or args.seed != 0):
        raise ValueError("Tune is restricted to dev subjects 1-3, seed 0")
    if args.command == "evaluate" and dev:
        raise ValueError("Dev subjects cannot enter the test comparison")
    manifest = load_manifest(args)
    tuning, tuning_sha = load_tuning(args, manifest) if args.command == "evaluate" else (None, None)
    identity = job_identity(args, manifest, tuning_sha)
    fingerprint = digest_json(identity)
    destination = receipt_path(args, args.command, args.holdout, args.seed)
    if try_resume(destination, fingerprint):
        return
    checkpoint = checkpoint_path(args, args.holdout, args.seed)
    expected = manifest["files"][f"checkpoint/{args.holdout}/{args.seed}"]
    if str(checkpoint.resolve()) != expected["path"]:
        raise ValueError("--runs-dir differs from the checked manifest")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint(saved, args.holdout, args.seed, protocol,
                        manifest["files"]["data/protocol.json"]["sha256"])
    data = Data(args.data_dir)
    adapter = Adapter(args, saved)
    device = adapter.device
    split = "validation" if args.command == "tune" else "test"
    eval_ids = data.ids(args.holdout, split)
    eval_audio = data.audio_ids(eval_ids)
    eval_bank_ids, eval_bank = candidates(data, split, device)
    train_bank_ids, train_bank = candidates(data, "train", device)
    blocks = sample_blocks(data.rows, args.holdout, max(protocol["budgets_seconds"]),
                           protocol["repeats"], args.seed * 1000 + args.holdout)
    records, calibrations = [], []
    started = synchronized_time(device)
    source_norm = source_norm_gain(adapter.source_weights.cpu().numpy())
    for budget in protocol["budgets_seconds"]:
        budget_blocks = [[]] if budget == 0 else [b[:budget // 3] for b in blocks]
        for repeat, block in enumerate(budget_blocks):
            cal_start = synchronized_time(device)
            scaler = (fit_standardizer(data.raw[block]) if block
                      else population_standardizer(data.scalers, adapter.source_subjects))
            scaler_seconds = synchronized_time(device) - cal_start
            x_eval = torch.as_tensor(apply_standardizer(data.raw[eval_ids], scaler), device=device)
            x_cal = (torch.as_tensor(apply_standardizer(data.raw[block], scaler), device=device)
                     if block else None)
            calibration = {"budget_seconds": budget, "repeat": repeat, "window_ids": block,
                           "audio_ids": data.audio_ids(block), "scaler": scaler,
                           "unique_meg_seconds": len(set(block)) * 3,
                           "audio_file": data.rows[block[0]]["audio_file"] if block else None,
                           "audio_start_seconds": float(data.rows[block[0]]["audio_start_seconds"]) if block else None,
                           "elapsed_recording_seconds": (int(data.rows[block[-1]]["meg_stop_sample_1000hz"])
                                                         - int(data.rows[block[0]]["meg_start_sample_1000hz"])) / 1000 if block else 0,
                           "maps": {}, "bn": {}}
            calibrations.append(calibration)

            def record(method, setting, calibration_seconds, scores=None, inference_seconds=None, **extra):
                if scores is None:
                    begin = synchronized_time(device)
                    scores = adapter.scores(x_eval, eval_bank)
                    inference_seconds = synchronized_time(device) - begin
                result = {"method": method, "setting": str(setting), "budget_seconds": budget,
                          "repeat": repeat, "split": split, "calibration_seconds": calibration_seconds,
                          "inference_seconds": inference_seconds,
                          **metrics(scores, eval_audio, eval_bank_ids, adapter.temperature), **extra}
                stories = {}
                for story in ("lw1", "easy_money"):
                    indices = [i for i, w in enumerate(eval_ids)
                               if data.rows[w]["audio_file"].startswith(story + "_")]
                    if indices:
                        stories[story] = metrics(scores[indices], [eval_audio[i] for i in indices],
                                                 eval_bank_ids, adapter.temperature)
                result["by_story_same_gallery"] = stories
                records.append(result)
                print(json.dumps({"holdout": args.holdout, "seed": args.seed, "budget": budget,
                                  "repeat": repeat, "method": method, "setting": str(setting),
                                  "top10": round(result["top10"], 4)}), flush=True)

            if args.command == "tune":
                for rule in protocol["gain_rules"]:
                    gain = gain_value(rule, adapter)
                    adapter.reset(gain=gain)
                    record("gain_mean_ws", rule, scaler_seconds, gain_rule=rule, gain=gain)
                    if block:
                        for blend in protocol["bn_blends"]:
                            adapter.reset(gain=gain)
                            begin = synchronized_time(device)
                            bn = adapter.adabn(x_cal, blend)
                            duration = synchronized_time(device) - begin
                            setting = f"{rule}:{blend:g}"
                            calibration["bn"][setting] = bn
                            record("adabn", setting, scaler_seconds + duration,
                                   gain_rule=rule, gain=gain, bn_blend=blend)
                continue

            rule, blend = tuning["gain_rule"], tuning["bn_blend"]
            gain = gain_value(rule, adapter)
            adapter.reset()
            record("mean_ws", "1", scaler_seconds, gain=1.0)
            adapter.reset(gain=gain)
            record("gain_mean_ws", rule, scaler_seconds, gain_rule=rule, gain=gain)
            # Ensemble averages source prediction similarities, never source weights.
            begin = synchronized_time(device)
            ensemble = None
            for donor in range(len(adapter.source_subjects)):
                adapter.reset(donor=donor)
                donor_scores = adapter.scores(x_eval, eval_bank)
                ensemble = donor_scores.clone() if ensemble is None else ensemble + donor_scores
            ensemble /= len(adapter.source_subjects)
            record("ensemble", "uniform_cosine", scaler_seconds, scores=ensemble,
                   inference_seconds=synchronized_time(device) - begin,
                   donors=adapter.source_subjects)
            if not block:
                continue

            begin = synchronized_time(device)
            selection = []
            for donor, subject in enumerate(adapter.source_subjects):
                adapter.reset(donor=donor)
                score = metrics(adapter.scores(x_cal, train_bank), data.audio_ids(block),
                                train_bank_ids, adapter.temperature)
                selection.append({"subject": subject, "loss": score["loss"]})
            best = min(range(len(selection)), key=lambda i: selection[i]["loss"])
            donor_seconds = synchronized_time(device) - begin
            adapter.reset(donor=best)
            calibration["donor_scores"] = selection
            calibration["selected_subject"] = adapter.source_subjects[best]
            record("select_ws", "calibration_train_gallery_loss", scaler_seconds + donor_seconds,
                   selected_subject=adapter.source_subjects[best])

            adapter.reset(gain=gain)
            begin = synchronized_time(device)
            calibration["bn"][f"{rule}:{blend:g}"] = adapter.adabn(x_cal, blend)
            bn_seconds = synchronized_time(device) - begin
            record("adabn", f"{rule}:{blend:g}", scaler_seconds + bn_seconds,
                   gain_rule=rule, gain=gain, bn_blend=blend)

            # Same checkpoint, own fitted Ws, known shared train audio, no target in template.
            begin = synchronized_time(device)
            reference = None
            with torch.inference_mode():
                for donor, subject in enumerate(adapter.source_subjects):
                    adapter.reset(donor=donor)
                    source_ids = data.matched_source_ids(subject, block)
                    if any(data.rows[i]["split"] != "train" for i in source_ids):
                        raise ValueError("Reference template attempted to use non-train windows")
                    source_x = torch.as_tensor(np.array(data.scaled[source_ids]), device=device)
                    branches = adapter.branches(source_x).double()
                    reference = branches.clone() if reference is None else reference + branches
                reference = (reference / len(adapter.source_subjects)).cpu().numpy()
                adapter.reset(gain=gain)
                target = adapter.branches(x_cal).cpu().numpy()
            template_seconds = synchronized_time(device) - begin
            calibration["template_source_subjects"] = adapter.source_subjects
            calibration["template_seconds"] = template_seconds
            for kind in ("scalar", "diagonal", "procrustes"):
                begin = synchronized_time(device)
                mapping = fit_branch_map(target, reference, kind)
                map_seconds = synchronized_time(device) - begin
                calibration["maps"][kind] = jsonable(mapping)
                adapter.reset(gain=gain, mapping=mapping)
                record(f"align_{kind}", "centered", scaler_seconds + template_seconds + map_seconds,
                       gain_rule=rule, gain=gain)
    audit = adapter.close()
    receipt = {"schema_version": 1, "status": "complete", **identity, "fingerprint": fingerprint,
               "protocol_sha256": sha256(args.protocol), "checkpoint_sha256": expected["sha256"],
               "source_subjects": adapter.source_subjects, "source_norm_gain": source_norm,
               "calibrations": calibrations, "records": records, "audit": audit,
               "seconds": synchronized_time(device) - started,
               "peak_gpu_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None}
    expected_rows = 100 if args.command == "tune" else 51
    if len(records) != expected_rows:
        raise RuntimeError(f"Incomplete job: {len(records)} records, expected {expected_rows}")
    finish_receipt(destination, receipt)
    print(json.dumps({"status": "complete", "receipt": str(destination), "records": len(records)}), flush=True)


def balanced_loss(receipts, method, predicate):
    per_subject = []
    for receipt in receipts:
        grouped = {}
        for row in receipt["records"]:
            if row["method"] == method and predicate(row):
                grouped.setdefault(row["budget_seconds"], []).append(row["loss"])
        if not grouped:
            raise ValueError("Missing tuning candidate")
        per_subject.append(float(np.mean([np.mean(values) for values in grouped.values()])))
    return float(np.mean(per_subject))


def choose(args, protocol):
    manifest = load_manifest(args)
    receipts, checksums = [], {}
    for subject in protocol["dev_subjects"]:
        path = receipt_path(args, "tune", subject, 0)
        item = read_json(path)
        if (item.get("status") != "complete" or item.get("stage") != "tune"
                or item.get("holdout") != subject or item.get("seed") != 0
                or item.get("manifest_fingerprint") != manifest["fingerprint"]
                or len(item.get("records", [])) != 100
                or item.get("receipt_sha256") != digest_json({k: v for k, v in item.items() if k != "receipt_sha256"})):
            raise ValueError(f"Incomplete or inconsistent dev tuning: {path}")
        receipts.append(item)
        checksums[str(path.relative_to(args.output_dir))] = sha256(path)
    gain_scores = {rule: balanced_loss(receipts, "gain_mean_ws", lambda r: r["gain_rule"] == rule)
                   for rule in protocol["gain_rules"]}
    gain = min(protocol["gain_rules"], key=lambda rule: gain_scores[rule])
    bn_scores = {str(blend): balanced_loss(receipts, "adabn",
                                         lambda r: r["gain_rule"] == gain and r["bn_blend"] == blend)
                 for blend in protocol["bn_blends"]}
    blend = min(protocol["bn_blends"], key=lambda value: bn_scores[str(value)])
    result = {"schema_version": 1, "status": "complete", "protocol_sha256": sha256(args.protocol),
              "manifest_fingerprint": manifest["fingerprint"], "tuning_receipts": checksums,
              "gain_rule": gain, "bn_blend": blend, "gain_validation_loss": gain_scores,
              "bn_validation_loss": bn_scores, "selection_split": "validation",
              "subjects": protocol["dev_subjects"], "seed": 0}
    path = args.output_dir / "tuning.json"
    if path.exists() and read_json(path) != result:
        raise ValueError("Existing tuning.json differs: use a new output directory for a new series")
    atomic_json(path, result)
    print(json.dumps({"status": "chosen", "gain_rule": gain, "bn_blend": blend, "path": str(path)}), flush=True)


def main(argv=None):
    parser = make_parser()
    args = _CLI_ARGS if argv is None and _CLI_ARGS is not None else parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    for name in ("data_dir", "runs_dir", "output_dir", "protocol", "coords_dir"):
        setattr(args, name, getattr(args, name).resolve())
    protocol = read_json(args.protocol)
    validate_protocol(protocol)
    torch.set_num_threads(max(1, int(os.environ.get("FQW_NUM_THREADS", "2"))))
    torch.manual_seed(getattr(args, "seed", 0))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "check":
        check(args, protocol)
    elif args.command == "choose":
        choose(args, protocol)
    else:
        run_job(args, protocol)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
