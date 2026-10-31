"""Summarize frozen calibration receipts using people, not seeds, as independent units."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
METHODS = ("mean_ws", "gain_mean_ws", "ensemble", "select_ws", "adabn",
           "align_scalar", "align_diagonal", "align_procrustes")
ZERO_METHODS = METHODS[:3]
METRICS = ("top1", "top10", "mean_rank", "loss")
BASELINE = "gain_mean_ws"


def expected_record_keys(budgets: list[int], repeats: int) -> set[tuple[str, int, int]]:
    return {(method, budget, repeat) for budget in budgets
            for method in (ZERO_METHODS if budget == 0 else METHODS)
            for repeat in ([0] if budget == 0 else range(repeats))}


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False)
    try:
        with handle:
            handle.write(content)
        os.replace(handle.name, path)
    finally:
        if os.path.exists(handle.name):
            os.unlink(handle.name)


def csv_text(rows: list[dict], columns: list[str]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def confidence_interval(values: np.ndarray, draws: int, seed: int) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    bounds = np.quantile(means, [0.025, 0.975])
    return float(bounds[0]), float(bounds[1])


def validate_receipt(receipt: dict, path: Path, holdout: int, seed: int,
                     expected: set[tuple[str, int, int]], protocol_hash: str,
                     tuning_hash: str | None = None) -> list[dict]:
    if (receipt.get("schema_version") != 1 or receipt.get("status") != "complete"
            or receipt.get("stage") != "evaluate" or receipt.get("holdout") != holdout
            or receipt.get("seed") != seed):
        raise ValueError(f"Invalid or unfinished evaluation receipt: {path}")
    if not receipt.get("fingerprint"):
        raise ValueError(f"Missing provenance fingerprint: {path}")
    # Every worker receipt carries this hash; reject accidental mixtures of protocols.
    if receipt.get("protocol_sha256") != protocol_hash:
        raise ValueError(f"Protocol hash mismatch: {path}")
    actual_tuning_hash = receipt.get("tuning_sha256")
    if (not isinstance(actual_tuning_hash, str) or len(actual_tuning_hash) != 64
            or any(character not in "0123456789abcdef" for character in actual_tuning_hash)):
        raise ValueError(f"Missing or invalid tuning_sha256: {path}")
    if tuning_hash is not None and actual_tuning_hash != tuning_hash:
        raise ValueError(f"Tuning hash mismatch: {path}")
    rows = receipt.get("records")
    if not isinstance(rows, list):
        raise ValueError(f"Receipt records must be a list: {path}")
    seen = set()
    result = []
    for row in rows:
        key = (row.get("method"), row.get("budget_seconds"), row.get("repeat"))
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected or duplicate result {key}: {path}")
        seen.add(key)
        for metric in METRICS:
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid {metric} in {key}: {path}")
        if not 0 <= row["top1"] <= row["top10"] <= 1:
            raise ValueError(f"Top-k values must be fractions in [0, 1]: {path}")
        for metric in ("queries", "candidates"):
            if not isinstance(row.get(metric), int) or row[metric] < 1:
                raise ValueError(f"Invalid {metric}: {path}")
        if not 1 <= row["mean_rank"] <= row["candidates"]:
            raise ValueError(f"Mean rank outside the candidate bank: {path}")
        result.append({**row, "holdout": holdout, "seed": seed})
    if seen != expected:
        raise ValueError(f"A complete receipt has {len(expected - seen)} missing records: {path}")
    contents = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    integrity = hashlib.sha256(json.dumps(contents, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if receipt.get("receipt_sha256") != integrity:
        raise ValueError(f"Receipt integrity check failed: {path}")
    return result


def aggregate(records: list[dict], holdouts: list[int], seeds: list[int], budgets: list[int],
              repeats: int, draws: int = 10000, bootstrap_seed: int = 0) -> tuple[list[dict], list[dict], list[dict], list[int]]:
    """Average repeats, then seeds within subject; only use complete requested subjects."""
    groups = defaultdict(list)
    for row in records:
        groups[(row["holdout"], row["method"], row["budget_seconds"], row["seed"])].append(row)
    per_subject = []
    excluded = set()
    for person in holdouts:
        for budget in budgets:
            n_repeats = 1 if budget == 0 else repeats
            for method in (ZERO_METHODS if budget == 0 else METHODS):
                seed_groups = [groups[(person, method, budget, seed)] for seed in seeds]
                if any(len(group) != n_repeats for group in seed_groups):
                    excluded.add(person)
                    continue
                row = {"holdout": person, "method": method, "budget_seconds": budget,
                       "seeds": len(seeds), "repeats_per_seed": n_repeats}
                for metric in (*METRICS, "calibration_seconds", "inference_seconds"):
                    if all(metric in record for group in seed_groups for record in group):
                        row[metric] = float(np.mean([np.mean([record[metric] for record in group]) for group in seed_groups]))
                per_subject.append(row)
    by_method = defaultdict(list)
    for row in per_subject:
        by_method[(row["method"], row["budget_seconds"])].append(row)
    summaries = []
    for (method, budget), rows in sorted(by_method.items(), key=lambda entry: (entry[0][1], entry[0][0])):
        for metric in (*METRICS, "calibration_seconds", "inference_seconds"):
            if not all(metric in row for row in rows):
                continue
            values = np.asarray([row[metric] for row in rows], dtype=float)
            low, high = confidence_interval(values, draws, bootstrap_seed)
            # For ranks/loss/time a larger value is worse; Top-k goes in the opposite direction.
            ordered = np.sort(values)
            worst = ordered[:math.ceil(len(values) / 4)] if metric.startswith("top") else ordered[-math.ceil(len(values) / 4):]
            summaries.append({"method": method, "budget_seconds": budget, "metric": metric,
                              "subjects": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
                              "ci95_low": low, "ci95_high": high, "worst_quartile_mean": float(worst.mean())})
    baseline = {(row["holdout"], row["budget_seconds"]): row for row in per_subject if row["method"] == BASELINE}
    comparisons = []
    for (method, budget), rows in sorted(by_method.items(), key=lambda entry: (entry[0][1], entry[0][0])):
        if method == BASELINE:
            continue
        paired = [(row, baseline[(row["holdout"], budget)]) for row in rows if (row["holdout"], budget) in baseline]
        for metric in METRICS:
            if not paired:
                continue
            delta = np.asarray([row[metric] - base[metric] for row, base in paired])
            low, high = confidence_interval(delta, draws, bootstrap_seed)
            improvement = delta if metric.startswith("top") else -delta
            comparisons.append({"method": method, "baseline": BASELINE, "budget_seconds": budget,
                                "metric": metric, "subjects": len(delta), "mean_difference": float(delta.mean()),
                                "median_difference": float(np.median(delta)), "ci95_low": low, "ci95_high": high,
                                "improved_fraction": float(np.mean(improvement > 0)),
                                "tied_fraction": float(np.mean(improvement == 0))})
    return per_subject, summaries, comparisons, sorted(excluded)


def summarize(output_dir: Path, protocol_path: Path, holdouts: list[int] | None = None,
              seeds: list[int] | None = None, draws: int = 10000, bootstrap_seed: int = 0) -> dict:
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    dev = protocol.get("dev_subjects", [1, 2, 3])
    canonical_holdouts = [p for p in protocol.get("subjects", list(range(1, 28))) if p not in dev]
    canonical_seeds = protocol.get("seeds", [0, 1, 2])
    selected_holdouts = sorted(set(holdouts if holdouts is not None else canonical_holdouts))
    selected_seeds = sorted(set(seeds if seeds is not None else canonical_seeds))
    if not selected_holdouts or not set(selected_holdouts).issubset(canonical_holdouts):
        raise ValueError("Selected holdouts must be non-dev subjects in the protocol")
    if not selected_seeds or not set(selected_seeds).issubset(canonical_seeds):
        raise ValueError("Selected seeds must belong to the protocol")
    if draws < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    budgets = protocol["budgets_seconds"]
    repeats = protocol["repeats"]
    expected = expected_record_keys(budgets, repeats)
    tuning_path = output_dir / "tuning.json"
    tuning_hash = hashlib.sha256(tuning_path.read_bytes()).hexdigest() if tuning_path.exists() else None
    records, missing = [], []
    completed = 0
    for person in selected_holdouts:
        for seed in selected_seeds:
            path = output_dir / "evaluate" / f"h{person:02d}_s{seed}.json"
            if not path.exists():
                missing.append(str(path))
                continue
            receipt = json.loads(path.read_text())
            records.extend(validate_receipt(receipt, path, person, seed, expected, protocol_hash, tuning_hash))
            if tuning_hash is None:
                tuning_hash = receipt["tuning_sha256"]
            completed += 1
    per_subject, summaries, comparisons, excluded = aggregate(records, selected_holdouts, selected_seeds, budgets,
                                                             repeats, draws, bootstrap_seed)
    subset = selected_holdouts != sorted(canonical_holdouts) or selected_seeds != sorted(canonical_seeds)
    report = {"schema_version": 1, "analysis": "exploratory", "primary_metric": "top10",
              "unit_of_analysis": "subject; mean over repeats, then seeds",
              "protocol_sha256": protocol_hash, "tuning_sha256": tuning_hash,
              "tuning_file_verified": tuning_path.exists(), "requested_holdouts": selected_holdouts,
              "requested_seeds": selected_seeds, "expected_jobs": len(selected_holdouts) * len(selected_seeds),
              "completed_jobs": completed, "canonical_jobs": len(canonical_holdouts) * len(canonical_seeds),
              "partial": bool(missing) or subset, "requested_subset_complete": not missing,
              "missing_receipts": missing, "excluded_incomplete_subjects": excluded,
              "record_count": len(records), "records_per_job": len(expected),
              "bootstrap": {"draws": draws, "seed": bootstrap_seed, "interval": "95% percentile, subjects resampled",
                            "multiplicity_adjustment": "none; descriptive intervals, not confirmatory tests"},
              "summary": summaries, "paired_vs_gain_mean_ws": comparisons}
    raw_columns = ["holdout", "seed", "method", "setting", "budget_seconds", "repeat", *METRICS,
                   "queries", "candidates", "calibration_seconds", "inference_seconds"]
    atomic_text(output_dir / "records.csv", csv_text(records, raw_columns))
    atomic_text(output_dir / "per_subject.csv", csv_text(per_subject, ["holdout", "method", "budget_seconds", "seeds",
                                                                    "repeats_per_seed", *METRICS, "calibration_seconds", "inference_seconds"]))
    atomic_text(output_dir / "summary.csv", csv_text(summaries, ["method", "budget_seconds", "metric", "subjects", "mean",
                                                              "median", "ci95_low", "ci95_high", "worst_quartile_mean"]))
    atomic_text(output_dir / "paired_vs_gain.csv", csv_text(comparisons, ["method", "baseline", "budget_seconds", "metric",
                                                                      "subjects", "mean_difference", "median_difference",
                                                                      "ci95_low", "ci95_high", "improved_fraction", "tied_fraction"]))
    state = "ЧАСТИЧНЫЙ РЕЗУЛЬТАТ" if report["partial"] else "СЕРИЯ ЗАВЕРШЕНА"
    lines = [f"# Frozen LISA: {state}", "", f"Готово {completed}/{report['expected_jobs']} запрошенных задач; полная серия — {report['canonical_jobs']}.",
             "", "Сначала усреднены повторы и сиды внутри человека. Независимая единица — человек. "
             "95% интервалы — bootstrap по людям; они описательные, без поправки на множественные сравнения. "
             "Test уже использовался в exp06: этот анализ исследовательский.", "",
             "| Метод | Бюджет, с | Людей | Top-10, % | 95% ДИ, % | Медиана, % | Худшая четверть, % |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        if row["metric"] != "top10":
            continue
        interval = "—" if row["ci95_low"] is None else f"{100 * row['ci95_low']:.1f}–{100 * row['ci95_high']:.1f}"
        lines.append(f"| {row['method']} | {row['budget_seconds']} | {row['subjects']} | {100 * row['mean']:.1f} | "
                     f"{interval} | {100 * row['median']:.1f} | {100 * row['worst_quartile_mean']:.1f} |")
    if missing:
        lines += ["", f"Не хватает {len(missing)} receipts. Люди без всех запрошенных сидов исключены из сводных оценок: {excluded}."]
    if subset:
        lines += ["", "Запрошено подмножество полной серии. Его нельзя выдавать за результат по 24 людям × 3 сида."]
    if not summaries:
        lines += ["", "Пока нет человека с полным набором запрошенных сидов; статистическая сводка пуста."]
    atomic_text(output_dir / "summary.md", "\n".join(lines) + "\n")
    atomic_text(output_dir / "summary.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, default=ROOT / "experiments/exp07_frozen_calibration/results")
    cli.add_argument("--protocol", type=Path, default=ROOT / "experiments/exp07_frozen_calibration/protocol.json")
    cli.add_argument("--holdouts", type=int, nargs="+")
    cli.add_argument("--seeds", type=int, nargs="+")
    cli.add_argument("--bootstrap-draws", type=int, default=10000)
    cli.add_argument("--bootstrap-seed", type=int, default=0)
    args = cli.parse_args(argv)
    try:
        report = summarize(args.output_dir, args.protocol, args.holdouts, args.seeds,
                           args.bootstrap_draws, args.bootstrap_seed)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        cli.error(str(exc))
    print(f"Receipts: {report['completed_jobs']}/{report['expected_jobs']}; "
          f"{'partial' if report['partial'] else 'complete'}. Summary: {args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
