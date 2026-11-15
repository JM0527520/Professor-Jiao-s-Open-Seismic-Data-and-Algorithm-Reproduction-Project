
import json,hashlib,time
from pathlib import Path
import numpy as np
root=Path('/home/zjm/faultseg_project/workspace')
import os,sys
os.chdir(root);sys.path.insert(0,str(root))
from fault_workflow.common import load_config,WORKSPACE,write_json,timestamp
from fault_workflow.data import read_synthetic
config=load_config(WORKSPACE/'configs/baseline_balanced.json')
window=np.hanning(128).astype(np.float64)
frequency=np.fft.rfftfreq(128)[1:]
def spectrum(trace):
    value=np.asarray(trace,dtype=np.float64)
    if value.shape!=(128,) or not np.isfinite(value).all():
        raise ValueError('Bad diagnostic trace window')
    power=np.abs(np.fft.rfft((value-value.mean())*window))[1:]**2
    total=power.sum()
    return power/total if total>0 else None
sine=np.sin(2*np.pi*np.arange(128)*8/128)
np.testing.assert_allclose(spectrum(sine),spectrum(sine*12+3),rtol=1e-12,atol=1e-12)
assert frequency[int(np.argmax(spectrum(sine)))]==8/128
source=[]
for index in range(200):
    raw=read_synthetic(config['data_root'],'train','seis',index)
    for a in range(16,128,32):
        for b in range(16,128,32):
            p=spectrum(raw[a,b,:])
            if p is not None: source.append(p)
cache=config['cache_root']/'baseline_thebe_validation'
marker=json.loads((cache/'ready.json').read_text())
seismic=np.load(cache/'seismic_raw.npy',mmap_mode='r')
states=np.load(cache/'states_raw.npy',mmap_mode='r')
target=[]
for section in range(200):
    for lateral in range(128,seismic.shape[1],256):
        trace=seismic[section,lateral]
        support=states[section,lateral]!=0
        for start in range(0,seismic.shape[2]-127,128):
            if support[start:start+128].all():
                p=spectrum(trace[start:start+128])
                if p is not None:target.append(p)
def summary(rows):
    power=np.asarray(rows)
    centroid=power@frequency
    return {'windows':len(rows),'mean_normalized_power':power.mean(0).tolist(),
            'spectral_centroid_cycles_per_sample_quantiles':np.quantile(centroid,[.1,.5,.9]).tolist(),
            'peak_of_mean_spectrum_cycles_per_sample':float(frequency[np.argmax(power.mean(0))]),
            'mean_power_below_0_125_cycles_per_sample':float(power[:,frequency<.125].sum(1).mean())}
result={'status':'passed_diagnostic_execution_only','finished':timestamp(),
        'protocol':{'length':128,'source':'all 200 training volumes; raw lateral axes 16,48,80,112; full vertical traces',
                    'target':'all 200 validation sections; lateral=128+256k; vertical disjoint windows start=128k; require full amplitude support',
                    'processing':'per-trace demean; symmetric Hann; rFFT; exclude DC; normalize each nonconstant window by its power',
                    'labels_used_for_selection':False,'test_used':False},
        'frequency_cycles_per_sample':frequency.tolist(),'source':summary(source),'thebe':summary(target),
        'cache_sha256_from_accepted_marker':marker['sha256'],
        'interpretation':'Descriptive trace-spectrum diagnosis only. Wavelet, geology and noise are mixed; physical sample spacing is unverified. This does not establish a resampling factor or recognition accuracy.',
        'training_or_predictions_changed':False}
path=root/'outputs/server_balanced_v2/review/signal_spectrum_diagnosis.json'
write_json(path,result)
print(json.dumps({k:{n:v for n,v in result[k].items() if n!='mean_normalized_power'} for k in ('source','thebe')},indent=2))
