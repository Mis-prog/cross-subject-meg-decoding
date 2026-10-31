"""Fetch ses-0, task-0/task-2 recordings and ICA for extra subjects (mini-LOSO pilot), verifying hashes."""
import argparse
import concurrent.futures
import json

import requests

import sys
from pathlib import Path

# Support direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exp03_real_data_pilot.download_pilot import DEST, download
from scripts.exp03_real_data_pilot.inspect_pilot_files import walk

RAW_ROOT = 'https://api.osf.io/v2/nodes/ag3kj/files/osfstorage/'
ICA_NODE = 'https://api.osf.io/v2/nodes/3wrft/files/osfstorage/'
TASKS = (0, 2)


def list_folder(url):
    items = []
    while url:
        response = requests.get(url, params={'page[size]': 100}, timeout=45)
        response.raise_for_status()
        payload = response.json()
        items += payload['data']
        url = payload.get('links', {}).get('next')
    return {i['attributes']['name']: i for i in items}


def child_url(item):
    return item['relationships']['files']['links']['related']['href']


def subject_inventory(subject, raw_root, ica_root):
    raw_sessions = list_folder(child_url(raw_root[subject]))
    files = walk(child_url(raw_sessions['ses-0']), f'raw/{subject}/ses-0')
    files += walk(child_url(ica_root[subject]), f'ica/{subject}', skip=('ses-1',))
    return files


def keep(path):
    if '_task-' not in path:
        return True
    return any(f'_task-{task}_' in path for task in TASKS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', nargs='+', default=['sub-02', 'sub-03', 'sub-04'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    raw_root = list_folder(RAW_ROOT)
    ica_root = list_folder(child_url(list_folder(ICA_NODE)['clean']))
    selected = []
    for subject in args.subjects:
        if subject not in raw_root or subject not in ica_root:
            raise ValueError(f'{subject}: no raw or ICA folder on OSF')
        files = [f for f in subject_inventory(subject, raw_root, ica_root) if keep(f['path'])]
        size = sum(f['size'] for f in files)
        print(f'{subject}: {len(files)} files, {size / 1e9:.3f} GB', flush=True)
        selected += files
    total = sum(f['size'] for f in selected)
    print('Total:', len(selected), 'files,', f'{total / 1e9:.3f} GB', flush=True)
    if args.dry_run:
        return

    (DEST / 'mini_loso_inventory.json').write_text(json.dumps(selected, indent=2) + '\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        verified = list(pool.map(download, sorted(selected, key=lambda i: i['path'])))
    (DEST / 'mini_loso_downloads.json').write_text(json.dumps(verified, indent=2) + '\n')


if __name__ == '__main__':
    main()
