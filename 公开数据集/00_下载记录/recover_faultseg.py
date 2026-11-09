"""Restore the original axis order from the project-linked float32 V1 mirror."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
MIRROR=ROOT/'06_FaultSegV3_CIG_Bench/原始压缩包/FaultSegV1.zip'
TARGET=ROOT/'01_FaultSeg3D/训练与验证数据'


def run():
    source_url='https://modelscope.cn/datasets/douyimin/CIG-Bench-Dataset'
    rows=[];compared=0
    author_validation={}
    author_repo=next((ROOT/'01_FaultSeg3D/作者代码').glob('faultSeg_*.zip'))
    with zipfile.ZipFile(author_repo) as z:
        for name in z.namelist():
            if '/data/validation/' in name and name.endswith('.dat'):
                author_validation[name.split('/data/',1)[1]]=z.read(name)
    with zipfile.ZipFile(MIRROR) as z:
        names=sorted(n for n in z.namelist() if n.endswith('.npz'))
        if len(names)!=220:raise ValueError(f'Expected 220 V1 pairs, found {len(names)}')
        for name in names:
            split='train' if '/train/' in name else 'validation'
            index=int(Path(name).stem)
            with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as pair:
                for key in ['seis','fault']:
                    a=pair[key]
                    if a.shape!=(128,128,128) or a.dtype!=np.float32:
                        raise ValueError(f'Unexpected array {name}:{key}: {a.shape} {a.dtype}')
                    # Mirror is (t,h,w), original .dat is serialized (h,w,t).
                    raw=np.ascontiguousarray(a.transpose(1,2,0),dtype='<f4').tobytes()
                    p=TARGET/split/key/f'{index}.dat'
                    if split=='validation':
                        raw=author_validation[f'validation/{key}/{index}.dat']
                        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
                        rows.append(dict(relative_path=str(p.relative_to(ROOT)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                                         acquisition='original_author_repository',source_url='https://github.com/xinwucwp/faultSeg'))
                        continue
                    if p.exists():
                        if p.read_bytes()!=raw:raise ValueError(f'Original/mirror mismatch: {p}')
                        compared+=1;origin='original_download'
                    else:
                        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
                        origin='modelscope_float32_mirror_axis_restored'
                    rows.append(dict(relative_path=str(p.relative_to(ROOT)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                                     acquisition=origin,source_url=source_url,mirror_member=name,mirror_key=key))
    # Preserve the distinction between the publisher archive and a local replacement.
    archive=TARGET/'train/fault_from_ModelScope.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted((TARGET/'train/fault').glob('*.dat')):z.write(p,'fault/'+p.name)
    report=dict(status='complete',paired_train_volumes=200,paired_validation_volumes=20,
                original_files_byte_compared=compared,axis_conversion='mirror (t,h,w) -> original serialized (h,w,t), little-endian float32',
                mirror_sha256='76cd54cf2061daaef97e8f3dc8d193a57f33ff06ba1d04939f3ca477bdc6d557',
                generated_label_archive=str(archive.relative_to(ROOT)),files=rows)
    (HERE/'FaultSeg3D镜像恢复记录.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='files'},ensure_ascii=True),flush=True)

if __name__=='__main__':run()
