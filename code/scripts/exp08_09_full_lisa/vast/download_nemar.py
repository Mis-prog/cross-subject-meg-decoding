"""Download MEG-MASC from the NEMAR mirror (nm000229) into the layout the LISA scripts expect.

Used while OSF is unavailable. Files go to code/LISA/data/MASC-MEG/<BIDS path>; every file is checked
against the manifest (sha256 for annexed files, git blob sha1 for the rest). Re-running skips files
that already verify, so an interrupted download can simply be restarted.

  python download_nemar.py --workers 16
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MANIFEST = 'https://data.nemar.org/nm000229/v1.0.1/manifest.json'
DEST = Path(__file__).resolve().parents[3] / 'LISA/data/MASC-MEG'
CHUNK = 1 << 22


def digest(path, algorithm, size):
    h = hashlib.sha256() if algorithm == 'sha256' else hashlib.sha1(f'blob {size}\0'.encode())
    with path.open('rb') as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def verified(path, entry):
    return (path.exists() and path.stat().st_size == entry['size']
            and digest(path, entry['checksum_algorithm'], entry['size']) == entry['checksum'])


def fetch(entry, dest, retries=5):
    path = dest / entry['path']
    if verified(path, entry):
        return entry['path'], 0, 'ok'
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    for attempt in range(retries):
        try:
            have = part.stat().st_size if part.exists() else 0
            if have > entry['size']:
                part.unlink()
                have = 0
            request = urllib.request.Request(entry['url'], headers={'Range': f'bytes={have}-'} if have else {})
            with urllib.request.urlopen(request, timeout=120) as response:
                if have and response.status != 206:  # server ignored the range: start over
                    have = 0
                with part.open('ab' if have else 'wb') as f:
                    while block := response.read(CHUNK):
                        f.write(block)
            if part.stat().st_size == entry['size'] and \
                    digest(part, entry['checksum_algorithm'], entry['size']) == entry['checksum']:
                part.replace(path)
                return entry['path'], entry['size'], 'downloaded'
            part.unlink()
        except Exception as exc:  # noqa: BLE001 — retried below
            last = exc
        time.sleep(5 * (attempt + 1))
    return entry['path'], 0, f'FAILED ({last if "last" in locals() else "checksum mismatch"})'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--dest', default=str(DEST))
    args = parser.parse_args()
    dest = Path(args.dest)
    with urllib.request.urlopen(MANIFEST, timeout=120) as response:
        manifest = json.load(response)
    total = sum(e['size'] for e in manifest)
    print(f'{len(manifest)} files, {total / 1e9:.1f} GB -> {dest}', flush=True)
    done_bytes, failed, start = 0, [], time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch, e, dest) for e in sorted(manifest, key=lambda e: -e['size'])]
        for i, future in enumerate(as_completed(futures), 1):
            path, size, status = future.result()
            done_bytes += size
            if status.startswith('FAILED'):
                failed.append(path)
            if status != 'ok' or i % 100 == 0:
                rate = done_bytes / max(time.time() - start, 1) / 1e6
                print(f'[{i}/{len(manifest)}] {status} {path} ({rate:.0f} MB/s)', flush=True)
    (dest / 'nemar_download.json').write_text(json.dumps(
        {'manifest': MANIFEST, 'files': len(manifest), 'bytes': total, 'failed': failed}, indent=2))
    print('FAILED:', failed if failed else 'none', flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
