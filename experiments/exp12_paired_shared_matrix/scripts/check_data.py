"""Run the exp12 preflight on the canonical LISA dataset."""

import argparse
import sys
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd
from lisa.utils.constants import N_SUBJECTS, TEST_STORIES, TRAIN_STORIES

from check_coords import check_coords


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "code"))

from scripts.exp08_09_full_lisa import full_lisa


DATA_FILES = (
    f"dataframe/{full_lisa.DF_TRAIN}",
    f"dataframe/{full_lisa.DF_TEST}",
    "audio/extract_features_train4.npy",
    "audio/extract_features_test4.npy",
    "meg/meg27_sr100.npz",
)


def check_author_split(canon: Path) -> list[str]:
    """Require the authors' audio split and all participants in both tables."""
    problems = []
    expected_subjects = set(range(1, N_SUBJECTS + 1))
    columns = ("subject_id", "story_id", "sound_id")
    for name, stories in ((full_lisa.DF_TRAIN, TRAIN_STORIES), (full_lisa.DF_TEST, TEST_STORIES)):
        rows = pd.read_csv(canon / "dataframe" / name, usecols=columns, dtype=dict.fromkeys(columns, "int64"))
        expected_sounds = {
            (story, sound)
            for story, (_, first, last) in stories.items()
            for sound in range(first, last + 1)
        }
        actual_sounds = set(rows[["story_id", "sound_id"]].itertuples(index=False, name=None))
        if actual_sounds != expected_sounds:
            problems.append(f"{name}: отличаются фрагменты от авторского разбиения. "
                            f"Отсутствуют {sorted(expected_sounds - actual_sounds)}. "
                            f"Лишние {sorted(actual_sounds - expected_sounds)}.")
        if set(rows["subject_id"]) != expected_subjects:
            problems.append(f"{name}: ожидаются участники с ID от 1 до {N_SUBJECTS}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка готовых данных для exp12.")
    parser.add_argument("--canon", type=Path, required=True, help="каталог preprocessed")
    parser.add_argument("--reference-coords", type=Path, help="эталонный каталог coords для сверки SHA-256")
    args = parser.parse_args()

    problems = check_coords(args.canon, args.reference_coords)
    missing = [name for name in DATA_FILES if not (args.canon / name).is_file()]
    problems.extend(f"Файл отсутствует: {args.canon / name}" for name in missing)
    try:
        if not missing:
            problems.extend(check_author_split(args.canon))
        if problems:
            for problem in problems:
                print(problem, file=sys.stderr)
            return 1
        report = full_lisa.check(canon=args.canon)
    except (OSError, ValueError, EOFError, BadZipFile, AttributeError, TypeError) as error:
        print(f"Ошибка чтения подготовленных данных: {error}", file=sys.stderr)
        return 1
    return int(bool(report["problems"]))


if __name__ == "__main__":
    raise SystemExit(main())
