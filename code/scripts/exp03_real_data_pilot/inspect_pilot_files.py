"""List public OSF files for subject 01/session 0 and stimuli, without downloading data."""
import concurrent.futures
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[3]


def walk(url, prefix, skip=()):
    files = []
    while url:
        response = requests.get(url, params={'page[size]': 100}, timeout=45)
        response.raise_for_status()
        payload = response.json()
        for item in payload['data']:
            attr = item['attributes']
            name = attr['name']
            if name in skip:
                continue
            path = prefix + '/' + name
            if attr['kind'] == 'folder':
                child = item['relationships']['files']['links']['related']['href']
                files.extend(walk(child, path, skip))
            else:
                files.append({'path': path, 'size': attr.get('size'),
                              'download': item['links']['download'],
                              'metadata': item['links']['self'],
                              'hashes': attr.get('extra', {}).get('hashes', {})})
        url = payload.get('links', {}).get('next')
    return files


if __name__ == '__main__':
    jobs = [
        ('https://api.osf.io/v2/nodes/ag3kj/files/osfstorage/63e66d65cb544b00fd9e6239/', 'raw/sub-01/ses-0', ()),
        ('https://api.osf.io/v2/nodes/3wrft/files/osfstorage/697383c989eff1b138bb80ab/', 'ica/sub-01', ('ses-1',)),
        ('https://api.osf.io/v2/nodes/ag3kj/files/osfstorage/63e614ffcb544b00a09e5e7a/', 'raw/stimuli', ()),
    ]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(walk, *job) for job in jobs]
        for future in futures:
            results.extend(future.result())
    output = ROOT / 'data/pilot/inventory.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    for item in results:
        print(item['size'], item['path'])
