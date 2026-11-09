"""Keep transfers, retry and final verification running independently of the chat."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import collection_report

STATE=HERE/'下载任务状态.json'
env=os.environ.copy()
env['PYTHONUTF8']='1'

def save(state, **extra):
    STATE.write_text(json.dumps(dict(state=state,pid=os.getpid(),updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**extra),ensure_ascii=False,indent=2),encoding='utf-8')

try:
    for attempt in range(1,4):
        save('downloading',attempt=attempt)
        with (HERE/'官方数据下载.log').open('a',encoding='utf-8') as a,(HERE/'GoogleDrive下载.log').open('a',encoding='utf-8') as b:
            processes=[subprocess.Popen([sys.executable,str(HERE/'collect_datasets.py'),'run'],stdout=a,stderr=a,env=env),
                       subprocess.Popen([sys.executable,str(HERE/'collect_google_drive.py')],stdout=b,stderr=b,env=env)]
            while any(p.poll() is None for p in processes):
                collection_report.snapshot()
                time.sleep(30)
        if collection_report.snapshot():break
        if attempt<3:time.sleep(30)
    save('verifying')
    result=collection_report.finish()
    complete=collection_report.snapshot()
    save('complete' if complete and result['status']=='verified' else 'needs_attention',verification=result['status'])
except Exception as error:
    save('needs_attention',error=str(error))
    raise
