"""Fetch the fixed sub-01/ses-0, task-0/task-2 subset, verifying source hashes."""
import concurrent.futures
import json

import sys
from pathlib import Path

# Support direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exp03_real_data_pilot.download_pilot import DEST, download


def main():
    inventory = json.loads((DEST / 'inventory.json').read_text())
    inventory += json.loads((DEST / 'downloads.json').read_text())
    selected = {}
    for item in inventory:
        path = item['path']
        if path.startswith(('raw/sub-01/ses-0/', 'ica/sub-01/ses-0/')):
            keep = '_task-' not in path or any(f'_task-{task}_' in path for task in (0, 2))
        elif path.startswith('raw/stimuli/audio/'):
            keep = path.split('/')[-1].startswith(('lw1_', 'easy_money_'))
        elif path.startswith('raw/stimuli/text_with_wordlists/'):
            keep = path.split('/')[-1].startswith(('lw1_', 'easy_money_'))
        else:
            keep = path in ('raw/dataset_description.json', 'raw/participants.tsv', 'raw/participants.json')
        if keep:
            selected[path] = item
    print('Selected', len(selected), 'files;', sum(i['size'] for i in selected.values()), 'source bytes', flush=True)
    # Reuse existing files; each is verified again, and pilot manifest is preserved.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        verified = list(pool.map(download, sorted(selected.values(), key=lambda i: i['path'])))
    (DEST / 'baseline_downloads.json').write_text(json.dumps(verified, indent=2) + '\n')


if __name__ == '__main__':
    main()
