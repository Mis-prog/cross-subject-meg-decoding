# Repository Guidelines

## Collaboration & Learning Preferences

- Пользователь хочет восстановить и углубить знание PyTorch и по умолчанию пишет код сам.
- При работе над кодом сначала объясняйте идею простыми словами, формы тензоров и нужные операции; помогайте подсказками, небольшими примерами и разбором ошибок. Не выдавайте полную реализацию и не меняйте код за пользователя без его просьбы.
- Проверяйте написанный пользователем код и объясняйте причины предложенных исправлений, чтобы поддерживать обучение.
- Если пользователь сообщает, что не успевает, или просит взять реализацию на себя, помогайте написанием и изменением кода в рамках запроса. Не требуйте повторного подтверждения уже данной просьбы.

## Project Structure & Module Organization

This workspace studies cross-subject MEG speech decoding and LISA calibration.

- `code/scripts/`: experiment packages (`exp00_*`, `exp03_*`–`exp09_*`); see its README for ownership and commands. Full-pipeline server helpers live in `exp08_09_full_lisa/vast/`.
- `code/LISA/`: ignored, separate upstream checkout; use the commit pinned in `README.md` and preserve upstream sources.
- `tests/`: regression tests for numerical helpers, pipelines, adapters, and experiment runners.
- `experiments/expNN_*/`: a report in `README.md`, an optional `protocol.json`, and small artifacts in `results/`. Shared exp08/exp09 material lives in `exp08_09_full_lisa/`.
- `code/notebooks/`: exploratory work; `docs/`: research, meetings, presentations, and thesis text under `thesis/`.

Each experiment README must include `Гипотеза`, `Задачи`, and `Скрипты` sections with repository-relative script paths. Distinguish planned implementations from available code and hypotheses from confirmed results.

## Build, Test, and Development Commands

Run from the repository root with Python 3.11:

- `uv sync --frozen`: install locked dependencies; requires the pinned `code/LISA/` checkout, installed as an editable dependency.
- `.venv/bin/python -m pytest tests`: run regression tests, including unittest cases.
- `.venv/bin/python -m unittest discover -s tests -p 'test_frozen_loso_math.py'`: run numerical helper tests independently of LISA.
- `.venv/bin/python code/scripts/exp00_lisa_readiness/smoke_lisa.py --output /tmp/fqw_lisa_smoke.json`: check model computation on synthetic CPU inputs without downloading MEG data.

The workspace has no package build. For the commission deck, run `npm ci`, `npm run dev`, or `npm run build` inside `docs/presentations/2026-10-09_commission/` to install, preview, or build Slidev.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions/files, and `PascalCase` classes. Follow adjacent code, document array shapes and scientific assumptions, and keep numerical helpers separate from training orchestration. No root formatter or linter configuration is present. Name experiment folders `expNN_descriptive_name/` and meeting notes `YYYY-MM-DD_topic.md`.

## Testing Guidelines

Use pytest and existing unittest patterns; name files `test_*.py` and functions `test_*`. Cover scientific invariants, split integrity, provenance, and invalid inputs when changing experiment logic. Some tests require upstream assets or local checkpoints and may skip. No coverage threshold is configured. Synthetic smoke results validate execution only; they do not establish decoding quality.

## Commit & Pull Request Guidelines

History uses short, descriptive Russian action phrases such as “Добавить …” and “Сохранить …”; follow that style. PRs should explain the change, identify affected experiments, and report validation commands/results. For experiment changes, record protocol, seeds, splits, calibration budget, and candidate-gallery size. Include screenshots for presentation changes and link issues when applicable.

## Data, Configuration & Agent Instructions

Keep `.env`, neurodata, checkpoints, and training logs out of Git; follow `.gitignore`. Use `.env.example` for configuration guidance.

Use Potpie for material project context, not trivial edits. When available, check `potpie --json source list` and `potpie --json graph status` once per session; record only durable learnings.
