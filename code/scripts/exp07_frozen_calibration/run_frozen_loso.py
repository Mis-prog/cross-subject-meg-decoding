"""Schedule frozen LISA calibration jobs; never train, download, or rent a GPU.

Run from any directory. A completed worker checks its own receipt and input hashes;
the scheduler deliberately submits it again instead of trusting file existence.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import TextIO


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = ROOT / "experiments/exp07_frozen_calibration/protocol.json"


@dataclass(frozen=True)
class Job:
    name: str
    command: list[str]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", choices=("check", "tune", "evaluate", "all"), default="check")
    result.add_argument("--workers", type=int, default=1, help="Concurrent GPU processes; start with 1.")
    result.add_argument("--holdouts", type=int, nargs="+", help="Evaluation subset only; dev tuning always uses all dev people.")
    result.add_argument("--seeds", type=int, nargs="+", help="Evaluation subset only; dev tuning always uses seed 0.")
    result.add_argument("--data-dir", type=Path, default=ROOT / "data/pilot/loso")
    result.add_argument("--runs-dir", type=Path, default=ROOT / "data/pilot/loso_runs")
    result.add_argument("--output-dir", type=Path, default=ROOT / "experiments/exp07_frozen_calibration/results")
    result.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    result.add_argument("--coords-dir", type=Path, default=ROOT / "code/LISA/data/preprocessed")
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--dry-run", action="store_true", help="Print the plan without reading data, starting jobs, or writing files.")
    return result


def build_plan(args: argparse.Namespace) -> list[list[Job]]:
    """Return dependency stages; jobs inside a stage can run concurrently."""
    if args.workers < 1 or args.batch_size < 1:
        raise ValueError("--workers and --batch-size must be positive")
    if args.protocol.exists():
        protocol = json.loads(args.protocol.read_text())
    elif args.dry_run:
        protocol = {}
    else:
        raise FileNotFoundError(f"Protocol not found: {args.protocol}")
    subjects = protocol.get("subjects", list(range(1, 28)))
    dev = protocol.get("dev_subjects", [1, 2, 3])
    canonical_seeds = protocol.get("seeds", [0, 1, 2])
    evaluated = [person for person in subjects if person not in dev]
    holdouts = sorted(set(args.holdouts if args.holdouts is not None else evaluated))
    seeds = sorted(set(args.seeds if args.seeds is not None else canonical_seeds))
    if not holdouts or not set(holdouts).issubset(evaluated):
        raise ValueError(f"--holdouts must be evaluation subjects: {evaluated}")
    if not seeds or not set(seeds).issubset(canonical_seeds):
        raise ValueError(f"--seeds must belong to {canonical_seeds}")

    common = []
    for option in ("data_dir", "runs_dir", "output_dir", "protocol", "coords_dir"):
        common += ["--" + option.replace("_", "-"), str(getattr(args, option).resolve())]
    common += ["--device", args.device, "--batch-size", str(args.batch_size)]

    def worker(name: str, verb: str, *extra: str) -> Job:
        return Job(name, [sys.executable, str(SCRIPT_DIR / "frozen_loso.py"), verb, *common, *extra])

    stages = [[worker("check", "check")]]
    if args.stage in ("tune", "all"):
        stages.append([worker(f"tune_h{h:02d}_s0", "tune", "--holdout", str(h), "--seed", "0") for h in dev])
        stages.append([worker("choose", "choose")])
    if args.stage in ("evaluate", "all"):
        stages.append([
            worker(f"evaluate_h{h:02d}_s{seed}", "evaluate", "--holdout", str(h), "--seed", str(seed))
            for seed in seeds for h in holdouts
        ])
        stages.append([Job("summary", [sys.executable, str(SCRIPT_DIR / "summarize_frozen_loso.py"),
                                     "--output-dir", str(args.output_dir.resolve()),
                                     "--protocol", str(args.protocol.resolve()),
                                     "--holdouts", *map(str, holdouts), "--seeds", *map(str, seeds)])])
    return stages


def stop_processes(running: dict[str, tuple[subprocess.Popen, Job, TextIO, float]]) -> None:
    """Stop entire child process groups, then reap children and close every log."""
    for process, _, _, _ in running.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5
    for process, _, log, _ in running.values():
        try:
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        finally:
            log.close()
    running.clear()


def run_stage(jobs: list[Job], workers: int, logs_dir: Path) -> bool:
    pending = list(jobs)
    running: dict[str, tuple[subprocess.Popen, Job, TextIO, float]] = {}
    succeeded = False
    try:
        while pending or running:
            for name, (process, job, log, started) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del running[name]
                elapsed = time.monotonic() - started
                print(f"{name}: {'ok' if code == 0 else f'FAILED (exit {code})'}, {elapsed:.1f} s", flush=True)
                if code != 0:
                    print(f"Read {logs_dir / (job.name + '.log')}. Dependent stages will not run.", file=sys.stderr)
                    return False
            while pending and len(running) < workers:
                job = pending.pop(0)
                log_path = logs_dir / (job.name + ".log")
                log = log_path.open("a", buffering=1)
                log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {shlex.join(job.command)}\n")
                try:
                    process = subprocess.Popen(job.command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                running[job.name] = (process, job, log, time.monotonic())
                print(f"Started {job.name}; log: {log_path}", flush=True)
            if running:
                time.sleep(0.2)
        succeeded = True
        return True
    finally:
        if not succeeded:
            stop_processes(running)


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    try:
        stages = build_plan(args)
    except (OSError, ValueError) as exc:
        cli.error(str(exc))
    if args.dry_run:
        for index, stage in enumerate(stages, 1):
            print(f"Stage {index}: {len(stage)} job(s)")
            for job in stage:
                print(shlex.join(job.command))
        return 0
    logs = args.output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        for stage in stages:
            if not run_stage(stage, args.workers, logs):
                return 1
    except KeyboardInterrupt:
        print("Interrupted; children stopped. Re-run the same command to resume verified jobs.", file=sys.stderr)
        return 130
    except OSError as exc:
        print(f"Cannot start a job: {exc}", file=sys.stderr)
        return 1
    print("Requested stages completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
