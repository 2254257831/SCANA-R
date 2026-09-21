from experiment import *
from recovery import REC
from scana_experiments.recovery_augmentation import RecoveryConfig,draw_pulse,apply_pulse,recovery_windows
import pandas as pd

rows=[]
record_cache={t:records(t)[0] for t in ['transfer_cube','insertion']}
cal_cache={t:dict(np.load(WORK/'calibration'/f'{t}.npz')) for t in record_cache}
for p in (REC/'data').rglob('*_audit.json'):
    a=json.loads(p.read_text(encoding='utf-8'));f=p.parent/a['attempts'][-1]['file'];r=dict(np.load(f));chunk=dict(np.load(p.parent/f'{a["branch"]}.npz'))
    source=dict(np.load(OLD/'demonstrations'/a['task']/f'{a["source"]}.npz'))
    times=chunk['time']
    train=record_cache[a['task']];source_order=[int(r['seed']) for r in train].index(a['source'])
    rng=np.random.default_rng(510000+source_order*100+a['branch']);grid=np.array_split(np.arange(16,369,8),12);endpoint=int(rng.choice(grid[a['branch']]))
    cal=cal_cache[a['task']];cfg=RecoveryConfig(amplitude=a['amplitude'])
    pulse=draw_pulse(cal['error'],cal['time'],endpoint,rng,cfg,a['mode'])
    np.testing.assert_array_equal(pulse*a['accepted_scale'],chunk['noise'])
    np.testing.assert_array_equal(apply_pulse(source['action'],pulse,endpoint,a['accepted_scale'],cfg),r['action'])
    cx,cy,ct=recovery_windows(r['input'],r['action'],source['action'],endpoint,cfg)
    np.testing.assert_array_equal(cx,chunk['input']);np.testing.assert_array_equal(cy,chunk['action']);np.testing.assert_array_equal(ct,chunk['time'])
    for i,t in enumerate(times):
        np.testing.assert_array_equal(chunk['input'][i],r['input'][t])
        np.testing.assert_array_equal(chunk['action'][i],r['action'][t:t+16].reshape(-1))
        np.testing.assert_array_equal(chunk['action'][i],source['action'][t:t+16].reshape(-1))
    np.testing.assert_array_equal(r['action'][:,GRIP],source['action'][:,GRIP])
    assert r['rewards'][-1]==4
    rows.append(dict(task=a['task'],mode=a['mode'],amplitude=a['amplitude'],source=a['source'],branch=a['branch'],scale=a['accepted_scale'],attempts=len(a['attempts']),post_pulse_success=int(r['rewards'][a['endpoint']:].max()==4),terminal_success=int(r['rewards'][-1]==4),label_action_match=True,observations_match=True))
savecsv(REC/'branch_audit.csv',rows)
df=pd.DataFrame(rows)
summary=df.groupby(['task','mode','amplitude']).agg(branches=('branch','size'),sources=('source','nunique'),attempts=('attempts','sum'),full_scale=('scale',lambda x:int((x==1).sum())),fallback=('scale',lambda x:int((x==0).sum())),post_pulse_success=('post_pulse_success','sum'),terminal_success=('terminal_success','sum')).reset_index()
summary.to_csv(REC/'branch_summary.csv',index=False,encoding='utf-8-sig')
print(summary.to_string(index=False))
if (REC/'per_episode.csv').exists():
    p=pd.read_csv(REC/'per_episode.csv').groupby(['task','method']).success.agg(['sum','count','mean']).reset_index()
    p.to_csv(REC/'summary.csv',index=False,encoding='utf-8-sig');print(p.to_string(index=False))
