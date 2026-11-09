"""Collect only the official assets specified by the project handover document."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / '00_下载记录'
MANIFEST = RECORDS / '下载清单.json'
LOCK = threading.Lock()


def metadata(url, name):
    req = urllib.request.Request(url, headers={'User-Agent': 'SeismicDataCollection/1.0'})
    with urllib.request.urlopen(req, timeout=90) as response:
        value = json.load(response)
    (RECORDS / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    return value


def prepare():
    tasks = []

    def add(dataset, path, url, size=None, checksum=None, kind='data'):
        tasks.append(dict(dataset=dataset, path=path, url=url, expected_bytes=size,
                          checksum=checksum, kind=kind, status='pending'))

    for recid, dataset, folder in [
        (13926822, 'CRACKS', '03_CRACKS_F3_v2/原始压缩包'),
        (4536561, 'RGT', '05_DeepRGT/原始分卷'),
        (5044922, 'RGT_code', '05_DeepRGT/作者代码'),
    ]:
        record = metadata(f'https://zenodo.org/api/records/{recid}', f'Zenodo_{recid}.json')
        for f in record['files']:
            filename = f['key'].split('/')[-1]
            add(dataset, f'{folder}/{filename}', f['links']['self'], f['size'], f['checksum'])

    record = metadata('https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/YBYGBK', 'Thebe_Dataverse.json')
    for item in record['data']['latestVersion']['files']:
        f = item['dataFile']
        directory = item.get('directoryLabel', '')
        if f['filename'].endswith('.npz') or directory in ['license', 'code', 'docs', 'data/raw']:
            add('Thebe', f"02_Thebe/{directory}/{f['filename']}",
                f"https://dataverse.harvard.edu/api/access/datafile/{f['id']}",
                f['filesize'], f"{f['checksum']['type'].lower()}:{f['checksum']['value']}")

    repos = [
        ('xinwucwp/faultSeg', '01_FaultSeg3D/作者代码'),
        ('anyuzoey/CNNforFaultInterpretation', '02_Thebe/作者代码'),
        ('olivesgatech/CRACKS', '03_CRACKS_F3_v2/作者代码'),
        ('ajbugge/HorizonTracker', '04_HorizonTracker/作者示例与代码'),
        ('douyimin/FaultSegV3', '06_FaultSegV3_CIG_Bench/作者代码'),
        ('olivesgatech/large-bench-geo', '07_统一比较框架/作者代码'),
    ]
    for repo, folder in repos:
        info = metadata(f'https://api.github.com/repos/{repo}', repo.replace('/', '_') + '_repo.json')
        commit = metadata(f"https://api.github.com/repos/{repo}/commits/{info['default_branch']}", repo.replace('/', '_') + '_commit.json')
        sha = commit['sha']
        name = repo.split('/')[-1]
        add('repositories', f'{folder}/{name}_{sha[:12]}.zip',
            f'https://codeload.github.com/{repo}/zip/{sha}', kind='repository')

    ms = metadata('https://www.modelscope.cn/api/v1/datasets/douyimin/CIG-Bench-Dataset/repo/tree?Revision=master&Root=ClassicDatasets', 'ModelScope_ClassicDatasets.json')
    f = next(f for f in ms['Data']['Files'] if f['Name'] == 'FaultSegV1.zip')
    url = 'https://www.modelscope.cn/api/v1/datasets/douyimin/CIG-Bench-Dataset/repo?Revision=master&FilePath=ClassicDatasets%2FFaultSegV1.zip'
    add('FaultSegV1_mirror', '06_FaultSegV3_CIG_Bench/原始压缩包/FaultSegV1.zip', url, f['Size'], 'sha256:' + f['Sha256'])
    MANIFEST.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'files': len(tasks), 'known_bytes': sum(t['expected_bytes'] or 0 for t in tasks)}, ensure_ascii=False), flush=True)


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run():
    tasks = json.loads(MANIFEST.read_text(encoding='utf-8'))

    def save():
        temp = MANIFEST.with_suffix('.tmp')
        temp.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temp, MANIFEST)

    def download(t):
        path = ROOT / t['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + '.part')
        with LOCK:
            if t['status'] == 'verified' and path.exists() and path.stat().st_size == t.get('actual_bytes'):
                return
            t.update(status='downloading', started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
            save()
        log = RECORDS / (hashlib.sha256(t['path'].encode()).hexdigest()[:12] + '.log')
        try:
            if not path.exists():
                cmd = ['curl.exe', '--location', '--fail', '--retry', '5', '--retry-delay', '5',
                       '--connect-timeout', '30', '--speed-limit', '1024', '--speed-time', '120',
                       '--max-time', '7200', '--continue-at', '-', '--output', str(partial), t['url']]
                with log.open('ab') as output:
                    result = subprocess.run(cmd, stdout=output, stderr=output)
                if result.returncode:
                    raise RuntimeError(f'curl exit {result.returncode}; see {log.name}')
                partial.replace(path)
            size = path.stat().st_size
            if t['expected_bytes'] is not None and size != t['expected_bytes']:
                raise RuntimeError(f"Size mismatch: {size} != {t['expected_bytes']}")
            algorithm, expected = t['checksum'].split(':', 1) if t['checksum'] else ('sha256', None)
            actual = digest(path, algorithm)
            if expected and actual.lower() != expected.lower():
                raise RuntimeError(f'{algorithm} checksum mismatch')
            with LOCK:
                t.update(status='verified', actual_bytes=size, verified_checksum=f'{algorithm}:{actual}',
                         finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
                t.pop('error', None)
                save()
            print(f"OK {t['path']} ({size:,} bytes)", flush=True)
        except Exception as error:
            if path.exists() and ('mismatch' in str(error).lower()):
                quarantine=path.with_name(path.name+'.invalid_'+str(int(time.time())))
                path.replace(quarantine)
            with LOCK:
                t.update(status='failed', error=str(error))
                save()
            print(f"FAILED {t['path']}: {error}", flush=True)

    # Bounded streams: 8 for large Zenodo volumes, 3 for Thebe.
    groups = {}
    for task in tasks:
        groups.setdefault(task['dataset'], []).append(task)
    rgt = sorted(groups.pop('RGT'), key=lambda t: t['path'])
    for i in range(8):
        groups[f'RGT_{i}'] = rgt[i::8]
    thebe = groups.pop('Thebe')
    for i in range(3):
        groups[f'Thebe_{i}'] = thebe[i::3]
    def group(items):
        for task in sorted(items, key=lambda t: t['expected_bytes'] or 0):
            download(task)
    with concurrent.futures.ThreadPoolExecutor(max_workers=14) as pool:
        list(pool.map(group, groups.values()))
    print('All transfer attempts finished.', flush=True)


def status():
    tasks = json.loads(MANIFEST.read_text(encoding='utf-8'))
    for dataset in dict.fromkeys(t['dataset'] for t in tasks):
        items = [t for t in tasks if t['dataset'] == dataset]
        counts = {state: sum(t['status'] == state for t in items) for state in ['verified', 'downloading', 'pending', 'failed']}
        partial = sum((ROOT / t['path']).with_name(Path(t['path']).name + '.part').stat().st_size
                      for t in items if (ROOT / t['path']).with_name(Path(t['path']).name + '.part').exists())
        print(dataset, counts, 'verified_GB=', round(sum(t.get('actual_bytes', 0) for t in items if t['status']=='verified')/1e9, 3), 'partial_GB=', round(partial/1e9,3))
    for t in tasks:
        if t['status'] == 'failed':
            print('FAILED', t['path'], t.get('error'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'status'])
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'status': status}[args.action]()
