"""Разбиение, запуск и парная оценка двух вариантов LISA в exp12."""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
PROTOCOL = EXPERIMENT_DIR / "protocol.json"
OUT = REPO_ROOT / "outputs" / EXPERIMENT_DIR.name

sys.path.insert(0, str(REPO_ROOT / "code"))

from scripts.exp08_09_full_lisa.full_lisa import CANON, job_lock
from scripts.exp10_shared_matrix.run_experiment import (
    evaluate_task,
    read_json,
    sha256,
    train_task,
    write_json,
)


def tasks(protocol):
    """Сформировать пару задач с одинаковыми людьми для каждого фолда."""
    groups = protocol["split"]["folds"]
    subjects = set(protocol["data"]["subjects"])
    members = [subject for group in groups for subject in group]
    if (len(groups) < 3 or any(not group for group in groups)
            or len(members) != len(subjects) or set(members) != subjects
            or any(type(subject) is not int or not 1 <= subject <= 27 for subject in members)):
        raise ValueError("Тестовые группы должны разбивать участников на непересекающиеся выборки")
    if sorted(protocol["variants"]) != ["individual", "switch"]:
        raise ValueError("Для exp12 нужны варианты individual и switch")
    seeds = protocol["seeds"]
    probabilities = protocol["routing"]["p_personal"]
    if len(seeds) != 1 or type(seeds[0]) is not int or seeds[0] < 0:
        raise ValueError("Для exp12 нужен один неотрицательный целый сид")
    if len(probabilities) != 1 or not 0 < probabilities[0] < 1:
        raise ValueError("Для exp12 нужна одна вероятность личной матрицы между 0 и 1")

    seed, probability = seeds[0], probabilities[0]
    switch_label = f"switch_p{str(probability).replace('.', 'p')}"
    result = {}
    for fold, test in enumerate(groups):
        validation = groups[(fold + 1) % len(groups)]
        split = {
            "fold": fold,
            "seed": seed,
            "train_subjects": sorted(subjects - set(test) - set(validation)),
            "val_subjects": sorted(validation),
            "test_subjects": sorted(test),
        }
        result[f"individual_f{fold}_s{seed}"] = dict(split, variant="individual", p_personal=1)
        result[f"{switch_label}_f{fold}_s{seed}"] = dict(split, variant="switch", p_personal=probability)
    return result


def read_evaluation(out, name, spec, protocol):
    """Проверить происхождение чекпойнта и соответствие тестовых ID фолду."""
    run = out / "runs" / name
    evaluation = out / "eval" / name
    manifest = read_json(run / "manifest.json")
    if manifest["protocol"] != protocol or manifest["task"] != spec:
        raise ValueError(f"{name}: протокол или разбиение отличаются от сохранённого запуска")
    if read_json(evaluation / "done.json")["checkpoint_sha256"] != sha256(run / "best.pt"):
        raise ValueError(f"{name}: тестовая оценка относится к другому чекпойнту")
    scores = pd.read_csv(evaluation / "subjects.csv")
    if sorted(scores["subject_id"]) != spec["test_subjects"]:
        raise ValueError(f"{name}: для каждого тестового участника нужна ровно одна строка")
    return scores, (manifest["settings"], manifest["source"])


def summarize(protocol, out):
    """Сравнить Test Top-10 одного и того же человека в обоих вариантах."""
    from scipy import stats

    out = Path(out)
    all_tasks = tasks(protocol)
    tables, completed = [], []
    reference = None
    for fold in range(len(protocol["split"]["folds"])):
        pair = [(name, spec) for name, spec in all_tasks.items() if spec["fold"] == fold]
        if any(not (out / "eval" / name / "done.json").is_file() for name, _ in pair):
            continue
        for name, spec in pair:
            scores, identity = read_evaluation(out, name, spec, protocol)
            if reference is None:
                reference = identity
            if identity != reference:
                raise ValueError("Для парного сравнения нужны одинаковые настройки и входные данные")
            tables.append(scores.assign(fold=fold, seed=spec["seed"], variant=spec["variant"],
                                        p_personal=spec["p_personal"]))
        completed.append([fold, protocol["seeds"][0]])

    if not completed:
        raise ValueError("Пока нет фолда с тестовой оценкой обоих вариантов")
    per_run = pd.concat(tables, ignore_index=True)
    metrics = ["top1", "top10", "mean_rank"]
    per_subject = per_run[["variant", "subject_id", *metrics]].sort_values(["variant", "subject_id"])
    wide = per_subject.pivot(index="subject_id", columns="variant", values="top10")
    delta = wide["switch"] - wide["individual"]
    if delta.isna().any():
        raise ValueError("У всех тестовых участников должны быть результаты обеих моделей")
    p_value = float(stats.wilcoxon(delta).pvalue) if (delta != 0).any() else 1.0
    comparison = pd.DataFrame([dict(comparison="switch-individual", subjects=len(delta),
                                    top10_delta=float(delta.mean()), p=p_value, p_holm=p_value)])
    summary = []
    for variant, rows in per_subject.groupby("variant"):
        for metric in metrics:
            values = rows[metric]
            half_width = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)) if len(values) > 1 else None
            summary.append(dict(variant=variant, metric=metric, subjects=len(values), mean=float(values.mean()),
                                ci95_half=half_width))

    target = out / "summary"
    target.mkdir(exist_ok=True)
    per_run.to_csv(target / "per_seed.csv", index=False)
    per_subject.to_csv(target / "per_subject.csv", index=False)
    pd.DataFrame(summary).to_csv(target / "summary.csv", index=False)
    comparison.to_csv(target / "comparisons.csv", index=False)
    planned = len(protocol["split"]["folds"])
    write_json(target / "coverage.json", dict(completed_fold_seeds=completed, planned_fold_seeds=planned,
                                              complete=len(completed) == planned))
    print(comparison.to_string(index=False))
    return per_subject


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=["tasks", "train", "eval", "summarize"])
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--canon", type=Path, default=CANON, help="каталог preprocessed")
    parser.add_argument("--out", type=Path, default=OUT, help="каталог результатов")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task", help="точное имя задачи из команды tasks")
    selection.add_argument("--all", action="store_true", help="оба варианта во всех выбранных фолдах")
    parser.add_argument("--fold", type=int, help="показать или запустить задачи одного фолда")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nepoch", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--dry-run", action="store_true", help="показать задачи без чтения данных")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    all_tasks = tasks(protocol)

    if args.fold is not None and not 0 <= args.fold < len(protocol["split"]["folds"]):
        parser.error("--fold вне диапазона")
    if args.task and args.fold is not None:
        parser.error("--task задаёт фолд в имени задачи и используется отдельно от --fold")
    if args.command == "summarize":
        summarize(protocol, args.out)
        return
    if args.task:
        if args.task not in all_tasks:
            parser.error("неизвестная задача, проверь список через tasks")
        selected = {args.task: all_tasks[args.task]}
    else:
        if args.command != "tasks" and not args.all:
            parser.error("для train и eval укажи --task или --all")
        selected = {name: spec for name, spec in all_tasks.items()
                    if args.fold is None or spec["fold"] == args.fold}
    if args.command == "tasks" or args.dry_run:
        for name, spec in selected.items():
            print(name, json.dumps(spec))
        return

    # Флаги запуска переопределяют настройки обучения из протокола.
    minimums = {"nepoch": 1, "batch_size": 2, "eval_batch_size": 1, "workers": 0}
    settings = {}
    for name, minimum in minimums.items():
        override = getattr(args, name)
        settings[name] = protocol["training"][name] if override is None else override
        if settings[name] < minimum:
            parser.error(f"--{name.replace('_', '-')} должен быть не меньше {minimum}")
    if protocol["training"]["early_stopping_patience"] < 1:
        parser.error("early_stopping_patience в протоколе должен быть положительным")
    if args.threads is not None:
        if args.threads < 1:
            parser.error("--threads должен быть положительным")
        import torch

        torch.set_num_threads(args.threads)

    canon, out = args.canon.resolve(), args.out.resolve()
    if args.command == "eval":
        missing = [name for name in selected if not (out / "runs" / name / "done.json").is_file()]
        if missing:
            parser.error(f"сначала заверши обучение: {', '.join(missing)}")
    for name, spec in selected.items():
        with job_lock(f"{args.command}_{name}", out):
            if args.command == "train":
                train_task(name, spec, protocol, canon, out, args.device, settings)
            else:
                evaluate_task(name, spec, protocol, canon, out, args.device, settings["eval_batch_size"])


if __name__ == "__main__":
    main()
