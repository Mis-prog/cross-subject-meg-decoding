"""Download only sub-01/ses-0/task-0 and the first lw1 audio stimulus from OSF."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / 'data/pilot'


def download(item):
    relative = Path(item['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe destination')
    output = DEST / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size == item['size']:
        print('Already present:', relative, flush=True)
    else:
        tmp = output.with_name(output.name + '.part')
        with requests.get(item['download'], stream=True, timeout=(30, 90)) as response:
            response.raise_for_status()
            with tmp.open('wb') as handle:
                for block in response.iter_content(1024 * 1024):
                    handle.write(block)
        if tmp.stat().st_size != item['size']:
            raise ValueError(f'Size mismatch: {relative}')
        tmp.replace(output)
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with output.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            sha256.update(block)
            md5.update(block)
    hashes = {'sha256': sha256.hexdigest(), 'md5': md5.hexdigest()}
    for name, expected in item.get('hashes', {}).items():
        if name in hashes and expected != hashes[name]:
            raise ValueError(f'OSF {name} mismatch: {relative}')
    print('Verified:', relative, output.stat().st_size, 'bytes', flush=True)
    return dict(item, local_sha256=sha256.hexdigest())


def main():
    inventory = json.loads((DEST / 'inventory.json').read_text())
    selected = []
    for item in inventory:
        path = item['path']
        if path.startswith('raw/sub-01/ses-0/'):
            keep = '_task-0_' in path or '_task-' not in path
        elif path.startswith('ica/'):
            keep = '_task-0_' in path
        else:
            keep = path in ('raw/stimuli/audio/lw1_0.wav',
                            'raw/stimuli/text_with_wordlists/lw1_produced_0.txt')
        if keep:
            selected.append(item)
    response = requests.get('https://api.osf.io/v2/nodes/ag3kj/files/osfstorage/',
                            params={'page[size]': 100}, timeout=45)
    response.raise_for_status()
    for item in response.json()['data']:
        a = item['attributes']
        if a['name'] in ('dataset_description.json', 'participants.tsv', 'participants.json'):
            selected.append({'path': 'raw/' + a['name'], 'size': a['size'],
                             'download': item['links']['download'],
                             'metadata': item['links']['self'],
                             'hashes': a.get('extra', {}).get('hashes', {})})
    print('Selected', len(selected), 'files;', sum(x['size'] for x in selected), 'bytes', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(download, selected))
    (DEST / 'downloads.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
