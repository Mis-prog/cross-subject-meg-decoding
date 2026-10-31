"""Fetch ses-0, task-0/task-2 recordings and ICA for many subjects (LOSO series), verifying hashes.

MEG-MASC raw data is split over the main OSF node and three child nodes (Part 2-4);
subjects are looked up in all of them. Stimuli and sub-01 come from the baseline manifest.
"""
import argparse
import concurrent.futures
import json

import sys
from pathlib import Path

# Support direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exp05_mini_loso.download_mini_loso import ICA_NODE, child_url, keep, list_folder
from scripts.exp03_real_data_pilot.download_pilot import DEST, download
from scripts.exp03_real_data_pilot.inspect_pilot_files import walk

RAW_NODES = ('ag3kj', 'h2tzn', 'u5327', 'dr4wy')


def raw_subject_folders():
    folders = {}
    for node in RAW_NODES:
        for name, item in list_folder(f'https://api.osf.io/v2/nodes/{node}/files/osfstorage/').items():
            if name.startswith('sub-'):
                folders.setdefault(name, item)
    return folders


def subject_files(subject, raw_root, ica_root):
    sessions = list_folder(child_url(raw_root[subject]))
    if 'ses-0' not in sessions:
        return []
    files = walk(child_url(sessions['ses-0']), f'raw/{subject}/ses-0')
    files += walk(child_url(ica_root[subject]), f'ica/{subject}', skip=('ses-1',))
    return [f for f in files if keep(f['path'])]


def complete(subject, files):
    paths = {f['path'] for f in files}
    return all(f'raw/{subject}/ses-0/meg/{subject}_ses-0_task-{t}_meg.con' in paths and
               f'ica/{subject}/ses-0/meg/{subject}_ses-0_task-{t}_meg-ica.fif' in paths for t in (0, 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', nargs='*', help='default: every subject with raw data and ICA')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    raw_root = raw_subject_folders()
    ica_root = list_folder(child_url(list_folder(ICA_NODE)['clean']))
    subjects = args.subjects or sorted(set(raw_root) & set(ica_root))
    selected, available, skipped = [], [], []
    for subject in subjects:
        files = subject_files(subject, raw_root, ica_root)
        if not complete(subject, files):
            print(f'{subject}: incomplete ses-0 task-0/2 data, skipped', flush=True)
            skipped.append(subject)
            continue
        print(f'{subject}: {len(files)} files, {sum(f["size"] for f in files) / 1e9:.3f} GB', flush=True)
        available.append(subject)
        selected += files
    # Stimuli and dataset-level files are shared; take them from the baseline manifest.
    selected += [i for i in json.loads((DEST / 'baseline_downloads.json').read_text())
                 if not i['path'].startswith(('raw/sub-', 'ica/sub-'))]
    selected = list({f['path']: f for f in selected}.values())
    print('Available:', available, 'skipped:', skipped, flush=True)
    print('Total:', len(selected), 'files,', f'{sum(f["size"] for f in selected) / 1e9:.2f} GB', flush=True)
    if args.dry_run:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        verified = list(pool.map(download, sorted(selected, key=lambda i: i['path'])))
    (DEST / 'loso_downloads.json').write_text(json.dumps(
        {'subjects': available, 'skipped': skipped, 'files': verified}, indent=2) + '\n')


if __name__ == '__main__':
    main()
