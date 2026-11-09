import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import time
import concurrent.futures
import threading

LOCK = threading.Lock()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / '下载工具依赖'))
import gdown

def run():
    recovery_path=HERE/'FaultSeg3D镜像恢复记录.json'
    recovery=json.loads(recovery_path.read_text(encoding='utf-8')) if recovery_path.exists() else {}
    recovered={str((ROOT/f['relative_path']).resolve()):f for f in recovery.get('files',[])}
    for folder_id, name in [
        ('1FcykAxpqiy2NpLP1icdatrrSQgLRXLP8', '训练与验证数据'),
        ('1q8sAoLJgbhYHRubzyqMi9KkTeZWXWtNd', '预训练模型与预测示例'),
    ]:
        target = ROOT / '01_FaultSeg3D' / name
        manifest = HERE / f'GoogleDrive_{folder_id}.json'
        try:
            files = gdown.download_folder(id=folder_id, output=str(target), skip_download=True,
                                          quiet=True, use_cookies=False, timeout=(30, 90))
            rows = [f._asdict() for f in files]
            for row in rows:
                item=recovered.get(str(Path(row['local_path']).resolve()))
                if item and item['acquisition']!='original_download':
                    row.update(acquisition=item['acquisition'],source_url=item['source_url'])
                if recovery and row['path'].replace('\\','/')=='train/fault.zip':
                    original=ROOT/'01_FaultSeg3D/训练与验证数据/train/fault.zip'
                    if original.exists():
                        row.update(local_path=str(original),acquisition='original_google_drive_archive',
                                   source_url=f"https://drive.google.com/file/d/{row['id']}/view")
                    else:
                        row.update(local_path=str(ROOT/recovery['generated_label_archive']),
                                   acquisition='locally_repacked_mirror_fault_labels',
                                   source_url='https://modelscope.cn/datasets/douyimin/CIG-Bench-Dataset')
            manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'{name}: {len(rows)} files discovered', flush=True)
            def download(pair):
                row, f = pair
                path = Path(row['local_path'])
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        result = gdown.download(id=f.id, output=str(path), quiet=True, use_cookies=False,
                                                resume=True, timeout=(30,120), retries=2)
                        if not result:
                            raise RuntimeError('No file returned')
                    except Exception as error:
                        row.update(status='failed', error=str(error))
                        with LOCK:
                            manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
                        print(f'FAILED {path.name}: {error}', flush=True)
                        return
                h = hashlib.sha256()
                with path.open('rb') as source:
                    for block in iter(lambda: source.read(8*1024*1024), b''):
                        h.update(block)
                row.update(status='downloaded', bytes=path.stat().st_size, sha256=h.hexdigest())
                with LOCK:
                    manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
                print(f'OK {name}/{path.name} {row["bytes"]}', flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(download, zip(rows, files)))
        except Exception as error:
            manifest.write_text(json.dumps({'status':'failed', 'error':str(error)}, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'FAILED {name}: {error}', flush=True)

if __name__ == '__main__':
    run()
