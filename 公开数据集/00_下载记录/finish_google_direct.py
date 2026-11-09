"""Use Google's public file-download endpoint when its legacy uc route fails."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
LOCK=threading.Lock()


def run():
    groups=[];tasks=[]
    for folder_id in ['1q8sAoLJgbhYHRubzyqMi9KkTeZWXWtNd','1FcykAxpqiy2NpLP1icdatrrSQgLRXLP8']:
        manifest=HERE/f'GoogleDrive_{folder_id}.json'
        items=json.loads(manifest.read_text(encoding='utf-8'));groups.append((manifest,items))
        for row in items:
            if row.get('status')=='failed' or row.get('acquisition','').startswith(('modelscope','locally_repacked')):
                if row['path'].replace('\\','/')=='train/fault.zip':
                    row['local_path']=str(ROOT/'01_FaultSeg3D/训练与验证数据/train/fault.zip')
                tasks.append((manifest,items,row))

    def transfer(task):
        manifest,items,row=task
        path=Path(row['local_path']);path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_name(path.name+'.direct.part')
        url=f"https://drive.usercontent.google.com/download?id={row['id']}&export=download&authuser=0&confirm=t"
        try:
            result=subprocess.run(['curl.exe','-sS','-L','-f','--retry','3','--max-time','600','-o',str(temp),url],capture_output=True)
            if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace')[-300:])
            with temp.open('rb') as source:header=source.read(100)
            if path.suffix=='.dat' and temp.stat().st_size!=8388608:raise ValueError('Unexpected DAT size')
            if path.suffix=='.hdf5' and not header.startswith(b'\x89HDF\r\n\x1a\n'):raise ValueError('Not an HDF5 file')
            if path.suffix=='.json':json.loads(temp.read_text(encoding='utf-8'))
            if path.suffix=='.zip':
                with zipfile.ZipFile(temp) as z:
                    bad=z.testzip()
                    if bad:raise ValueError('ZIP CRC failure '+bad)
            h=hashlib.sha256(temp.read_bytes()).hexdigest()
            agreement=None
            if path.exists() and path.suffix=='.dat':
                agreement=hashlib.sha256(path.read_bytes()).hexdigest()==h
                if not agreement:raise ValueError('Original/mirror DAT mismatch')
            temp.replace(path)
            with LOCK:
                row.update(status='downloaded',acquisition='original_google_drive_public_download',source_url=url,
                           sha256=h,bytes=path.stat().st_size)
                if agreement is not None:row['mirror_byte_agreement']=agreement
                row.pop('error',None)
                manifest.write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8')
            print('OK',row['path'],flush=True)
        except Exception as error:
            with LOCK:
                row['direct_download_error']=str(error)
                manifest.write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8')
            print('FAILED',row['path'],error,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(transfer,tasks))
    recovery_path=HERE/'FaultSeg3D镜像恢复记录.json';recovery=json.loads(recovery_path.read_text(encoding='utf-8'))
    lookup={str(Path(row['local_path']).resolve()):row for _,items in groups for row in items}
    for row in recovery['files']:
        source=lookup.get(str((ROOT/row['relative_path']).resolve()))
        if source and source.get('acquisition')=='original_google_drive_public_download':
            row.update(acquisition=source['acquisition'],source_url=source['source_url'])
    archive=ROOT/'01_FaultSeg3D/训练与验证数据/train/fault.zip'
    count=0
    recovery_lookup={str((ROOT/row['relative_path']).absolute()):row for row in recovery['files']}
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if not name.endswith('.dat'):continue
                raw=z.read(name);index=Path(name).name
                target=ROOT/'01_FaultSeg3D/训练与验证数据/train/fault'/index
                if target.read_bytes()!=raw:raise ValueError('Original/mirror fault mismatch '+index)
                row=recovery_lookup.get(str(target.absolute()))
                if row:
                    row.update(acquisition='original_google_drive_archive',source_url='https://drive.google.com/drive/folders/1FcykAxpqiy2NpLP1icdatrrSQgLRXLP8')
                count+=1
    recovery.update(original_training_seismic_files_byte_compared=sum(r.get('mirror_byte_agreement') is True or (r.get('status')=='downloaded' and r['path'].replace('\\','/').startswith('train/seis/')) for _,items in groups for r in items),
                    original_training_fault_files_byte_compared=count)
    recovery_path.write_text(json.dumps(recovery,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Original fault labels compared:',count,flush=True)

if __name__=='__main__':run()
