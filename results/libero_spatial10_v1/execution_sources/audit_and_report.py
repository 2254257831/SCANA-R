"""Read-only numerical audits and a local, automatically updated study report."""
from pathlib import Path
import json,time,hashlib,csv,argparse,statistics,collections,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(p,default=None):
    try:return json.loads((ROOT/p).read_text(encoding='utf8'))
    except (FileNotFoundError,json.JSONDecodeError):return default
def write(p,obj):
    p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8')
def csvwrite(path,rows):
    if not rows:return
    with (ROOT/path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def audit_collection(folder):
    attempts=json.loads((folder/'attempts.json').read_text());checked=0
    for p in sorted((folder/'accepted').glob('*.npz')):
        d=dict(np.load(p)); ix=int(d['attempt_index'][0]);r=attempts[ix]
        trace=dict(np.load(folder/'traces'/f"task{r['task']}_{r['episode']}_branch{r['branch']}_scale{r['scale']}.npz"))
        if 'observed_proprio' in trace:
            observed=trace['observed_proprio']
        else:
            proof=json.loads((folder/'observation_audit'/(p.stem+'.json')).read_text())
            assert proof['accepted_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
            tp=folder/'traces'/f"task{r['task']}_{r['episode']}_branch{r['branch']}_scale{r['scale']}.npz"
            assert proof['trace_sha256']==hashlib.sha256(tp.read_bytes()).hexdigest()
            observed=np.load(folder/'observation_audit'/(p.stem+'.npz'))['observed_proprio']
        assert r['terminal_success'] and trace['rewards'][-1]>0
        for j,t in enumerate(d['time']):
            assert np.array_equal(d['actions'][j],trace['actions'][t:t+8].astype('float32'))
            assert np.array_equal(d['proprio'][j],observed[t])
            checked+=1
    return {'library':folder.name,'verified_rows':checked,'assertions':'executed reference-tail actions, actual returned pre-action sensor observations, accepted terminal success','sensor_note':'robosuite sensors may sample before the last physics integration substep; not equated to raw final simulator qpos'}
