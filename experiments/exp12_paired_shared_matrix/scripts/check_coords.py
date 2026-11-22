"""Check the coordinate arrays used by LISA."""

import argparse
import hashlib
import sys
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


COORDINATE_SHAPES = {
    "sensor_xyz.npy": (208, 3),
    "coords208_xy_scaled.npy": (208, 2),
}


def check_coords(canon: Path, reference_coords: Path | None = None) -> list[str]:
    """Return problems with the arrays or their optional byte-for-byte reference."""
    problems = []
    for name, shape in COORDINATE_SHAPES.items():
        path = canon / "coords" / name
        try:
            array = np.load(path, allow_pickle=False)
            if not isinstance(array, np.ndarray):
                array.close()
                problems.append(f"{path}: ожидается массив NPY")
                continue
            if array.shape != shape:
                problems.append(f"{path}: форма {array.shape}, ожидалась {shape}")
                continue
            if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
                problems.append(f"{path}: ожидаются конечные числовые координаты")
                continue
        except (OSError, ValueError, EOFError, BadZipFile) as error:
            problems.append(f"{path}: {error}")
            continue

        if reference_coords is not None:
            reference = reference_coords / name
            try:
                actual_hash = hashlib.sha256(path.read_bytes()).digest()
                reference_hash = hashlib.sha256(reference.read_bytes()).digest()
            except OSError as error:
                problems.append(str(error))
                continue
            if actual_hash != reference_hash:
                problems.append(f"{path}: SHA-256 отличается от {reference}")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка координат 208 датчиков МЭГ.")
    parser.add_argument("--canon", type=Path, required=True, help="каталог preprocessed")
    parser.add_argument("--reference-coords", type=Path, help="эталонный каталог coords для сверки SHA-256")
    args = parser.parse_args()

    problems = check_coords(args.canon, args.reference_coords)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print("OK: sensor_xyz.npy (208, 3), coords208_xy_scaled.npy (208, 2)")
    if args.reference_coords is not None:
        print("OK: оба файла совпадают с эталоном по SHA-256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
