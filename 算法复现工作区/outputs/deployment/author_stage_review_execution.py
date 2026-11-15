
import os,sys,json
from pathlib import Path
root=Path('/home/zjm/faultseg_project/workspace');os.chdir(root);sys.path.insert(0,str(root))
import numpy as np
from fault_workflow.common import load_config,WORKSPACE,sha256,write_json,timestamp,raw_to_model
from fault_workflow.quality import score_region
from fault_workflow.diagnostics import proximity_diagnosis
from fault_workflow.figures import three_views
config=load_config(WORKSPACE/'configs/baseline_balanced.json')
folder=config['output_root']/'review';folder.mkdir(exist_ok=True)
base=config['output_root']/'baseline/faultseg_balanced_v2/author/thebe'
report=json.loads((base/'evaluation_report.json').read_text())
path=base/'probability_raw.npy'
if report['status']!='passed' or sha256(path)!=report['probability_sha256']:raise ValueError('Incomplete author prediction')
prob=np.load(path,mmap_mode='r')
cache=config['cache_root']/'baseline_thebe_validation'
arrays={k:np.load(cache/f'{k}_raw.npy',mmap_mode='r') for k in ('seismic','labels','states')}
protocol=json.loads((root/'configs/quality_review.json').read_text())
rows=[]
for region in protocol['thebe_regions']:
    roi=tuple(slice(s,s+n) for s,n in zip(region['start_raw'],region['shape_raw']))
    score=score_region(arrays['states'][roi],arrays['labels'][roi],prob[roi],protocol)
    rows.append({'region':region,'scores':score,'proximity_diagnosis_only':proximity_diagnosis(arrays['states'][roi],prob[roi])})
    start=region['start_raw']
    three_views(folder/f"{region['name']}_author_full_validation.png",raw_to_model(arrays['seismic'][roi]),
                raw_to_model(arrays['labels'][roi]),'作者模型：完整Thebe验证区的固定开发核心；尚未验收',
                prediction=raw_to_model(prob[roi]),offsets=(start[2],start[1],start[0]+900))
write_json(folder/'author_stage_review.json',{'status':'completed_author_diagnosis_only','finished':timestamp(),'evaluation_report_sha256':sha256(base/'evaluation_report.json'),'cases':rows,'test_used':False,'recognition_quality_accepted':False})
print([(r['region']['name'],r['scores']['at_fixed_threshold']['reference_positive_coverage']) for r in rows])
