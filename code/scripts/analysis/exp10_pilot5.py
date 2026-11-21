"""Verify and plot exp10's exploratory five-epoch pilot.

The unit of aggregation is a person, with equal weight for each of three
held-out people. Validation uses 108 candidates; test uses 1005. This report
does not substitute for the primary experiment with up to 50 epochs.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIANTS = ("individual", "shared_only", "switch")
TASKS = ("individual_f0_s42", "shared_only_f0_s42", "switch_p0p9_f0_s42")
LABELS = ("Среднее личных", "Только общая", "Переключение 0,9")
COLORS = ("#6c757d", "#0072b2", "#d55e00")


def verify(root):
    """Recompute macro metrics from window ranks and verify provenance."""
    summary = pd.read_csv(root / "summary/summary.csv")
    reference_manifest = None
    checked = []
    for variant, task in zip(VARIANTS, TASKS):
        run, evaluation = root / "runs" / task, root / "eval" / task
        manifest = json.loads((run / "manifest.json").read_text())
        done = json.loads((evaluation / "done.json").read_text())
        history = pd.read_csv(run / "metrics.csv")
        assert len(history) == manifest["settings"]["nepoch"] == 5
        assert manifest["task"]["seed"] == 42
        assert manifest["task"]["test_subjects"] == [11, 20, 22]
        assert done["test_candidates"] == 1005
        assert manifest["protocol"]["data"]["validation_candidates"] == 108
        groups = [set(manifest["task"][key]) for key in ("train_subjects", "val_subjects", "test_subjects")]
        assert [len(group) for group in groups] == [21, 3, 3]
        assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
        digest = hashlib.sha256((run / "best.pt").read_bytes()).hexdigest()
        assert digest == done["checkpoint_sha256"]
        if reference_manifest is None:
            reference_manifest = manifest
        else:
            for field in ("source", "settings", "protocol"):
                assert manifest[field] == reference_manifest[field], field
        windows = pd.read_csv(evaluation / "windows.csv.gz")
        assert len(windows) == 5025
        assert not windows.duplicated(["subject_id", "session_id", "wav_index"]).any()
        assert windows["rank"].between(0, 1004).all()
        assert windows.groupby("subject_id").size().to_dict() == {11: 2010, 20: 1005, 22: 2010}
        windows = windows.assign(top1=windows["rank"].eq(0), top10=windows["rank"].lt(10))
        people = windows.groupby("subject_id")[["top1", "top10"]].mean()
        for metric in ("top1", "top10"):
            actual = float(people[metric].mean())
            expected = float(summary.loc[(summary.variant == variant) & (summary.metric == metric), "mean"].iloc[0])
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        checked.append(dict(task=task, checkpoint_sha256=digest, windows=len(windows), epochs=5))
    return dict(status="verified; exploratory n=3", aggregation="equal-weight mean over people", runs=checked)


def plot(root, target):
    people = pd.read_csv(root / "summary/per_subject.csv")
    summary = pd.read_csv(root / "summary/summary.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, (test, validation) = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1, 1.05]})
    fig.subplots_adjust(top=.78, bottom=.22, wspace=.28)
    fig.suptitle("exp10 · предварительный пилот: бюджет 5 эпох", x=.08, ha="left", y=.97, fontsize=19, weight="bold")
    fig.text(.08, .90, "1 фолд · 3 новых человека · seed 42 · авторская ICA · без размеченной калибровки", fontsize=11, color="#444444")
    means = [float(summary.loc[(summary.variant == v) & (summary.metric == "top10"), "mean"].iloc[0]) * 100 for v in VARIANTS]
    test.bar(range(3), means, color=COLORS, alpha=.32, width=.62, zorder=2)
    for subject, marker in zip((11, 20, 22), ("o", "s", "^")):
        values = [float(people.loc[(people.variant == v) & (people.subject_id == subject), "top10"].iloc[0]) * 100 for v in VARIANTS]
        test.plot(range(3), values, color="#555555", alpha=.65, marker=marker, linewidth=1, markersize=5, label=f"Человек {subject}", zorder=4)
    for x, mean in enumerate(means):
        test.scatter(x, mean, color=COLORS[x], marker="D", s=65, zorder=5)
        test.annotate(f"{mean:.2f}%", (x, mean), xytext=(0, 10), textcoords="offset points", ha="center", weight="bold", color=COLORS[x], bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=1.5))
    test.axhline(100 * 10 / 1005, color="#999999", linestyle="--", linewidth=1)
    test.set_xticks(range(3), ["Среднее\nличных", "Только\nобщая", "Переключение\n0,9 / 0,1"])
    test.set_title("Тест · 1005 кандидатов", loc="left", weight="bold", pad=14)
    test.set_ylabel("Top-10, %")
    test.legend(loc="upper left", fontsize=9, frameon=False)
    for task, label, color in zip(TASKS, LABELS, COLORS):
        history = pd.read_csv(root / "runs" / task / "metrics.csv")
        best = json.loads((root / "runs" / task / "best.json").read_text())
        validation.plot(history.epoch + 1, history.val_top10 * 100, marker="o", color=color, linewidth=2, label=label)
        point = history.loc[history.epoch == best["epoch"]].iloc[0]
        validation.scatter(int(point.epoch) + 1, point.val_top10 * 100, s=100, facecolors="none", edgecolors=color, linewidths=1.7, zorder=5)
    validation.axhline(100 * 10 / 108, color="#999999", linestyle="--", linewidth=1)
    validation.set_title("Валидация · 108 кандидатов", loc="left", weight="bold", pad=14)
    validation.set_xlabel("Эпоха")
    validation.set_xticks(range(1, 6))
    validation.legend(loc="upper left", fontsize=9, frameon=False)
    for ax in (test, validation):
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2, zorder=0)
    fig.text(.08, .09, "Столбцы/ромбы — среднее, линии — люди. Пунктир — случайный уровень (1,0% / 9,3%); кольцо — выбранная эпоха.", fontsize=9, color="#444444")
    fig.text(.08, .045, "Пилот n=3: различия между условиями незначимы (p после Холма = 0,75). Основная серия продолжается.", fontsize=10, color="#444444")
    target.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(target / f"pilot5_results.{suffix}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root)
    plot(args.root, args.output)
    (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
