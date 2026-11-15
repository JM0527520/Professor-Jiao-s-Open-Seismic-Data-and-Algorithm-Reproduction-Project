
import os,sys,json
from pathlib import Path
root=Path('/home/zjm/faultseg_project/workspace');os.chdir(root);sys.path.insert(0,str(root))
import numpy as np
from fault_workflow.common import load_config,WORKSPACE,write_json,sha256,timestamp,raw_to_model,model_to_raw,standardize
from fault_workflow.baseline import configure_tensorflow,require_audit,require_preparation,get_policy
from fault_workflow.evaluation import _evaluation_model
from fault_workflow.padding_diagnosis import support_standardize
from fault_workflow.policy import ReferenceAccumulator
from fault_workflow.figures import three_views
config=load_config(WORKSPACE/'configs/baseline_balanced.json')
config['tensorflow_gpu_memory_limit_mb']=1152
require_audit(config);require_preparation(config)
tf=configure_tensorflow(config,'gpu');predict,model=_evaluation_model(config,tf,'author')
cache=config['cache_root']/'baseline_thebe_validation'
arrays={k:np.load(cache/f'{k}_raw.npy',mmap_mode='r') for k in ('seismic','labels','states')}
folder=config['output_root']/'review/padding_probe';folder.mkdir(exist_ok=True)
regions=[{'name':'shallow_left','start_raw':[36,128,256]},
         {'name':'shallow_middle','start_raw':[36,1024,384]},
         {'name':'shallow_right','start_raw':[36,2304,512]}]
rows=[]
for region in regions:
    start=region['start_raw'];roi=tuple(slice(s,s+128) for s in start)
    raw=arrays['seismic'][roi];labels=arrays['labels'][roi];states=arrays['states'][roi]
    x=raw_to_model(raw);support=raw_to_model(states!=0)
    control,author_stats=standardize(x);candidate,candidate_stats=support_standardize(x,support)
    outputs={}
    for name,data,stats in [('author',control,author_stats),('support',candidate,candidate_stats)]:
        probability=model_to_raw(predict(data)) if not stats['constant'] else np.zeros(raw.shape,dtype=np.float32)
        path=folder/f"{region['name']}_{name}.npy";np.save(path,probability)
        accumulator=ReferenceAccumulator([.5]);accumulator.update(states,labels,probability)
        outputs[name]={'normalization':stats,'score':accumulator.report(),'probability_sha256':sha256(path)}
        three_views(folder/f"{region['name']}_{name}.png",raw_to_model(raw),raw_to_model(labels),
                    f"固定补零诊断窗口：{region['name']} / {name}；单窗口实验，未采用",
                    prediction=raw_to_model(probability),offsets=(start[2],start[1],900+start[0]))
    rows.append({'region':region,'support_fraction':float(support.mean()),'outputs':outputs})
write_json(folder/'padding_probe.json',{'status':'passed_diagnostic_execution_only','finished':timestamp(),
     'model':model,'protocol':{'regions':regions,'shape_raw':[128,128,128],
       'selection':'fixed shallow coordinates after observing author full-volume padding artifacts; no label-based selection',
       'support':'original full-trace state cache; never recompute from a cropped window',
       'factor':'normalization only: support-only mean/std and neutral zero padding in normalized space',
       'inference':'single 128-cubed window, no overlap; does not measure full-volume seams',
       'threshold':.5,'source_training_changed':False,'test_used':False},
     'cases':rows,'promoted':False,'recognition_quality_accepted':False,
     'code_sha256':{p:sha256(root/p) for p in ['fault_workflow/padding_diagnosis.py','outputs/deployment/padding_probe_execution.py']}})
print([(r['region']['name'],r['support_fraction']) for r in rows],flush=True)
