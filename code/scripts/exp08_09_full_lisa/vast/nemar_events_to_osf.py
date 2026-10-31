"""Rewrite NEMAR events.tsv files to the original OSF MEG-MASC format that the LISA code parses.

NEMAR (nm000229) splits the event description into BIDS columns (trial_type='sound', story,
sound_id, ...). On OSF, trial_type holds the whole description as a Python dict literal, and LISA
reads it with ast.literal_eval to find each sound onset. This script rebuilds that dict from the
columns and keeps only onset/duration/trial_type/sample: with NEMAR's extra 'value' column,
mne_bids appends '/<value>' to some descriptions and they no longer parse. Onsets, samples and MEG
data are untouched. Originals are kept as *_events.tsv.nemar.

  python nemar_events_to_osf.py [--root code/LISA/data/MASC-MEG]
"""
import argparse
import ast
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / 'LISA/data/MASC-MEG'
FIELDS = ('story', 'story_uid', 'sound_id', 'kind', 'start', 'sound', 'phoneme', 'word', 'sequence_id',
          'condition', 'word_index', 'speech_rate', 'voice', 'pronounced')


def value(text):
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, (int, float)) else text
    except (ValueError, SyntaxError):
        return text


def convert(path):
    backup = path.with_name(path.name + '.nemar')
    if not backup.exists():
        path.replace(backup)
    with backup.open(encoding='utf-8-sig') as f:  # some NEMAR files start with a BOM
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
    columns = ['onset', 'duration', 'trial_type', 'sample']
    for row in rows:
        if not row['trial_type'].startswith('{'):  # some files already carry the OSF dict
            row['trial_type'] = repr({k: value(row[k]) for k in FIELDS if k in row and row[k] != 'n/a'})
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, columns, delimiter='\t', lineterminator='\n', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=str(ROOT))
    args = parser.parse_args()
    files = sorted(Path(args.root).glob('sub-*/ses-*/meg/*_events.tsv'))
    total = sum(convert(p) for p in files)
    print(f'{len(files)} events files, {total} rows rewritten')


if __name__ == '__main__':
    main()
