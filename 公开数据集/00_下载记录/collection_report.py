"""Build a human-readable inventory and verify downloaded archives and pairing."""
import csv
import gzip
import io
import json
from pathlib import Path
import re
import tarfile
import time
import zipfile
import shutil

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / '00_下载记录'


def load(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def snapshot():
    tasks = load(RECORDS / '下载清单.json', [])
    rows = []
    for t in tasks:
        path = ROOT / t['path']
        partial = path.with_name(path.name + '.part')
        rows.append({'数据集':t['dataset'], '相对路径':t['path'], '状态':t['status'],
                     '已存字节':path.stat().st_size if path.exists() else partial.stat().st_size if partial.exists() else 0,
                     '预计字节':t.get('expected_bytes') or '', '来源':t['url'],
                     '校验':t.get('verified_checksum', ''), '错误':t.get('error','')})
    google_done = True
    for id in ['1FcykAxpqiy2NpLP1icdatrrSQgLRXLP8','1q8sAoLJgbhYHRubzyqMi9KkTeZWXWtNd']:
        items = load(RECORDS / f'GoogleDrive_{id}.json')
        if not isinstance(items, list):
            google_done = False
            continue
        for t in items:
            path=Path(t['local_path'])
            state=t.get('status','pending')
            google_done &= state == 'downloaded'
            rows.append({'数据集':'FaultSeg3D_GoogleDrive', '相对路径':str(path.relative_to(ROOT)),
                         '状态':state, '已存字节':path.stat().st_size if path.exists() else 0,
                         '预计字节':'', '来源':t.get('source_url',f"https://drive.google.com/file/d/{t['id']}/view"),
                         '校验':('sha256:'+t['sha256']) if t.get('sha256') else '', '错误':t.get('error','')})
    with (RECORDS / '文件清单.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=['数据集','相对路径','状态','已存字节','预计字节','来源','校验','错误'])
        writer.writeheader();writer.writerows(rows)
    lines=['# 公开地震数据集下载进度','',f"更新时间：{time.strftime('%Y-%m-%d %H:%M:%S')}（本机时间）",'',
           '| 数据集 | 完成文件 | 文件总数 | 已存 GB | 状态 |','|---|---:|---:|---:|---|']
    for dataset in dict.fromkeys(r['数据集'] for r in rows):
        group=[r for r in rows if r['数据集']==dataset]
        count=sum(r['状态'] in ['verified','downloaded'] for r in group)
        state='完成' if count==len(group) else '部分失败，等待重试' if any(r['状态']=='failed' for r in group) else '下载中'
        lines.append(f"| {dataset} | {count} | {len(group)} | {sum(r['已存字节'] for r in group)/1e9:.3f} | {state} |")
    complete=bool(tasks) and all(t['status']=='verified' for t in tasks) and google_done
    lines += ['',f"总状态：{'全部下载完成' if complete else '尚未全部完成'}。",'',
              '`.part` 为未完成文件，不作为可用数据。官方提供 MD5/SHA-256 的文件均与官方校验值核对；Google Drive 文件记录本地 SHA-256，并另做压缩包/数据大小检查。',
              '', '来源、精确文件路径及逐文件状态见 `文件清单.csv`。整理和完整性核验结果见 `核验结果.json`（核验完成后生成）。']
    (ROOT/'下载进度.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return complete


def extract_small():
    tasks=load(RECORDS/'下载清单.json',[])
    candidates=[ROOT/t['path'] for t in tasks if t['status']=='verified' and
                (t.get('kind')=='repository' or t['dataset'] in ['CRACKS','RGT_code'])]
    for p in (ROOT/'01_FaultSeg3D').rglob('fault.zip'):
        if 'train' in p.parts:candidates.append(p)
    result=[]
    for p in candidates:
        out=p.parent/'已解压'
        marker=out/(p.stem+'.完成.json')
        if marker.exists():
            result.append(load(marker));continue
        with zipfile.ZipFile(p) as z:
            bad=z.testzip()
            if bad:raise ValueError(f'CRC error {p}: {bad}')
            if sum(i.file_size for i in z.infolist())>5*1024**3:
                raise ValueError(f'Unexpected archive size: {p}')
            out.mkdir(exist_ok=True)
            for item in z.infolist():
                target=(out/item.filename).resolve()
                if not target.is_relative_to(out.resolve()):
                    raise ValueError(f'Unsafe archive path {item.filename}')
            z.extractall(out)
            r={'archive':str(p.relative_to(ROOT)), 'files':sum(not i.is_dir() for i in z.infolist()),
               'uncompressed_bytes':sum(i.file_size for i in z.infolist()),'crc_verified':True}
        marker.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
        result.append(r)
    return result


class JoinedFiles(io.RawIOBase):
    def __init__(self, paths):
        self.paths=iter(paths);self.current=None
    def readable(self):return True
    def read(self,size=-1):
        chunks=[];remaining=size
        while remaining!=0:
            if self.current is None:
                try:self.current=next(self.paths).open('rb')
                except StopIteration:break
            b=self.current.read(remaining)
            if b:
                chunks.append(b)
                if remaining>0:remaining-=len(b)
            else:self.current.close();self.current=None
        return b''.join(chunks)
    def close(self):
        if self.current:self.current.close()
        super().close()


def verify_rgt():
    folder=ROOT/'05_DeepRGT/原始分卷'
    result={}
    for prefix in ['seis','rgt']:
        paths=sorted(folder.glob(prefix+'.zip[0-9][0-9]'))
        files=[]
        nested=[]
        with JoinedFiles(paths) as joined,gzip.GzipFile(fileobj=joined) as gz:
            with tarfile.open(fileobj=gz,mode='r|') as archive:
                for member in archive:
                    if not member.isfile():continue
                    if member.name.endswith('.zip'):
                        # The misleadingly named .zipNN volumes are a split tar.gz
                        # containing an inner ZIP, not split ZIP records.
                        temp=folder/(prefix+'.核验临时.zip')
                        try:
                            with archive.extractfile(member) as source,temp.open('wb') as dest:
                                shutil.copyfileobj(source,dest,length=8*1024*1024)
                            with zipfile.ZipFile(temp) as z:
                                bad=z.testzip()
                                if bad:raise ValueError(f'{prefix} inner ZIP CRC error {bad}')
                                files.extend({'name':i.filename,'bytes':i.file_size} for i in z.infolist() if not i.is_dir())
                            nested.append({'name':member.name,'zip_crc_verified':True})
                        finally:
                            if temp.exists():temp.unlink()
                    else:files.append({'name':member.name,'bytes':member.size})
            # Exhaust the gzip trailer to check its stream checksum.
            while gz.read(8*1024*1024):pass
        result[prefix]={'parts':len(paths),'gzip_crc_verified':True,'nested_archives':nested,'files':files}
    with zipfile.ZipFile(folder/'fault.zip') as z:
        bad=z.testzip()
        if bad:raise ValueError(f'RGT fault CRC error: {bad}')
        result['fault']={'zip_crc_verified':True,'files':[{'name':i.filename,'bytes':i.file_size} for i in z.infolist() if not i.is_dir()]}
    indices={k:{Path(f['name']).stem for f in v['files'] if f['name'].endswith('.dat')} for k,v in result.items()}
    if not (indices['seis']==indices['rgt']==indices['fault']):
        raise ValueError('RGT paired IDs differ')
    result['paired_volume_count']=len(indices['seis'])
    (RECORDS/'RGT完整性核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return {k:v for k,v in result.items() if k=='paired_volume_count'}


def verify_thebe():
    import numpy as np
    arrays={}
    for p in (ROOT/'02_Thebe/data/npzfiles').rglob('*.npz'):
        with zipfile.ZipFile(p) as z:
            values=[]
            for name in z.namelist():
                if name.endswith('.npy'):
                    with z.open(name) as f:
                        ver=np.lib.format.read_magic(f)
                        reader=np.lib.format.read_array_header_1_0 if ver==(1,0) else np.lib.format.read_array_header_2_0
                        shape,fortran,dtype=reader(f)
                    values.append(dict(key=name,shape=list(shape),dtype=str(dtype),fortran=fortran))
        arrays[p.stem]=values
    for stem,values in arrays.items():
        if stem.startswith('seis'):
            fault=arrays.get('fault'+stem[4:])
            if fault is None or [v['shape'] for v in values]!=[v['shape'] for v in fault]:
                raise ValueError(f'Thebe pair missing/shape mismatch {stem}')
    return arrays


def finish():
    result={'time':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    try:
        result['small_archives']=extract_small()
        result['thebe_arrays']=verify_thebe()
        if snapshot():result['rgt']=verify_rgt()
        result['status']='verified' if snapshot() else 'partial_downloads_in_progress'
    except Exception as error:
        result.update(status='verification_failed',error=str(error))
    (RECORDS/'核验结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='finish':print(json.dumps(finish(),ensure_ascii=True)[:1000])
    else:print(snapshot())
