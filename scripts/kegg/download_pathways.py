"""Download KEGG human pathways (by default, those in
data/pathway/hsa_ids_paper.txt) to data/pathway/hsa.tsv and data/pathway/txt/.
"""
import argparse
import time
import urllib.request
from pathlib import Path

BASE_URL = 'https://rest.kegg.jp'
INTERVAL = 0.4  # stay under the 3 requests/second limit


def fetch(path):
    with urllib.request.urlopen(f'{BASE_URL}/{path}') as res:
        text = res.read().decode('utf-8')
    time.sleep(INTERVAL)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true',
                        help='download all current human pathways')
    args = parser.parse_args()

    out_dir = Path('data/pathway')
    txt_dir = out_dir / 'txt'
    txt_dir.mkdir(parents=True, exist_ok=True)

    lines = [line for line in fetch('list/pathway/hsa').splitlines() if line]
    if not args.all:
        paper_ids = set((out_dir / 'hsa_ids_paper.txt').read_text().split())
        lines = [line for line in lines if line.split('\t')[0] in paper_ids]
        missing = paper_ids - {line.split('\t')[0] for line in lines}
        if missing:
            print(f'Not found in current KEGG: {sorted(missing)}')
    (out_dir / 'hsa.tsv').write_text('\n'.join(lines) + '\n')

    ids = [line.split('\t')[0] for line in lines]
    for i, hsa in enumerate(ids, 1):
        txt_path = txt_dir / f'{hsa}.txt'
        if txt_path.exists():
            continue
        print(f'[{i}/{len(ids)}] {hsa}')
        txt_path.write_text(fetch(f'get/{hsa}'))


if __name__ == '__main__':
    main()
