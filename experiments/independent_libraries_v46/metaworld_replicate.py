"""Three new recovery libraries on each fixed single-arm task; original evidence is read-only."""
from pathlib import Path
import os,sys,json,time,hashlib,argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
W=Path(os.environ.get('SCANA_V46_OUTPUT',REPO/'artifacts/independent_libraries_v46')).resolve();W.mkdir(parents=True,exist_ok=True)
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
sys.path[:0]=[str(HERE/'mw_runtime'),str(REPO/'src'),str(REPO/'experiments/metaworld_extension')]
import run as ex
import numpy as np
import pandas as pd
import torch
from scana_experiments.recovery_augmentation import RecoveryConfig,draw_pulse,apply_pulse,recovery_windows
torch.set_num_threads(1)
OUT=W/'metaworld_evidence';OLD=ex.OUT
BANKS=[101,211,307];SEEDS=[17,29,43];METHODS=['Clean repeat','Gaussian recovery','SCANA-R'];LAYOUTS=list(range(98000,98020))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):ex.dump(p,x)
def base(t,b):return OUT/t/str(b)
def records(task):
    result=[]
    for split,n in [('train',40),('development',10)]:
        rows=[]
        for p in sorted((OLD/'references'/task/split).glob('*.npz'),key=lambda p:int(p.stem)):
            r=ex.load_npz(p)
            if r['success'][-1]:r['seed']=int(p.stem);rows.append(r)
            if len(rows)==n:break
        assert len(rows)==n
        result.append(rows)
    return result
def freeze():
    p=dict(tasks=list(ex.TASKS),banks=BANKS,policy_seeds=SEEDS,methods=METHODS,test_layouts=LAYOUTS,
      sources='Same original 40 terminal-success training and 10 development sources per task; no original-data resampling.',
      randomization='Independently permuted episode folds, auxiliary initializations, pulse RNG and endpoint draws per bank; policy streams paired across methods.',
      collection='6 branches/source; 8-step pulse; terminal success acceptance; scales1,.5,.25,0; four actual-observation windows/branch; protected gripper.',
      amplitude='Use original development-selected task/family amplitudes without retuning.',
      fit='Original 22-256-256-64 policy; 1200 AdamW updates; 2048+2048 draws; minimum clean development MSE every100 updates.',
      evaluation='250 steps; chunk16, execute8; official ever-success. 6 fixed tasks x3 libraries x3 policy seeds x20 new layouts x3 methods =3240 episodes.',
      primary='Macro-average SCANA-R minus Gaussian and Clean repeat, conditional on six tasks; paired hierarchical library/seed/shared-layout bootstrap; taskwise all reported.',
      limitation='Only three independent library replicates per task, conditional on original sources; no claim of demonstration-population uncertainty.',
      script_sha256=sha(__file__),adapter_sha256=sha(HERE/'mw_runtime/bridge.py'))
    f=OUT/'protocol_frozen.json';OUT.mkdir(exist_ok=True)
    if f.exists():assert {k:v for k,v in json.loads(f.read_text()).items() if k!='frozen_utc'}==p
    else:dump(f,dict(**p,frozen_utc=datetime.now(timezone.utc).isoformat()))
def fit(x,y,vx,vy,norm,seed,path):
    if path.exists():return ex.load_model(path)
    st=time.process_time();m,ck=ex.fit(x,y,vx,vy,norm,seed,path)
    ck['cpu_seconds']=time.process_time()-st;torch.save(ck,path)
    return m,ck
def prepare(job):
    task,bank=job;b=base(task,bank)
    if (b/'prepared.json').exists():return str(b)
    st=time.perf_counter();tr,va=records(task);x,y,ep,tt=ex.windows(tr);vx,vy,_,_=ex.windows(va);norm=ex.norms(x,y)
    cp=b/'calibration.npz'
    if cp.exists():cal=ex.load_npz(cp)
    else:
        ids=np.asarray([r['seed'] for r in tr]);np.random.default_rng(bank+610000).shuffle(ids)
        errors=np.zeros_like(y);folds=np.zeros(len(x),int)
        for fold in range(5):
            mask=np.isin(ep,ids[fold*8:(fold+1)*8]);folds[mask]=fold
            m,ck=fit(x[~mask],y[~mask],x[~mask],y[~mask],ex.norms(x[~mask],y[~mask]),bank*100+fold,b/'auxiliary'/f'fold{fold}.pt')
            errors[mask]=ex.predict(m,ck,x[mask])-y[mask]
        cal=dict(error=errors.reshape(-1,16,4),time=tt,episode=ep,fold=folds);ex.save_npz(cp,**cal)
    amps=json.loads((OLD/'development'/f'{task}_choice.json').read_text())['amplitudes']
    sets={'Clean repeat':(x,y)};stats=[]
    for method,mode in [('SCANA-R','calibrated'),('Gaussian recovery','gaussian')]:
        rrlist=[];cfg=RecoveryConfig(amplitude=amps[mode],gripper_channels=(3,))
        for si,r in enumerate(tr):
            env=None
            try:
                for branch in range(6):
                    p=b/'recovery'/method/str(r['seed'])/f'{branch}.npz'
                    if p.exists():rr=ex.load_npz(p)
                    else:
                        if env is None:env=ex.make_env(task,r['seed'])
                        rng=np.random.default_rng(bank*100000+si*100+branch)
                        end=int(rng.choice(np.array_split(np.arange(16,ex.HORIZON-31,8),6)[branch]))
                        pulse=draw_pulse(cal['error'],cal['time'],end,rng,cfg,mode);attempts=[];start=time.process_time()
                        for j,scale in enumerate(cfg.backtracking):
                            action=np.clip(apply_pulse(r['action'],pulse,end,scale,cfg),-1,1).astype(np.float32)
                            ar=ex.execute(env,action);attempts.append(ar['success'])
                            if ar['success'][-1]:break
                        assert ar['success'][-1] and np.array_equal(action[:,3],r['action'][:,3])
                        xx,yy,times=recovery_windows(ar['input'],action,r['action'],end,cfg)
                        rr=dict(input=xx.astype(np.float32),action=yy.astype(np.float32),time=times,source=r['seed'],endpoint=end,scale=scale,
                            pulse=pulse,accepted_input=ar['input'],accepted_action=ar['action'],attempt_success=np.stack(attempts),cpu_seconds=time.process_time()-start)
                        ex.save_npz(p,**rr)
                    rrlist.append(rr);stats.append(dict(method=method,source=r['seed'],branch=branch,scale=float(rr['scale']),attempts=len(rr['attempt_success']),cpu_seconds=float(rr['cpu_seconds'])))
            finally:
                if env is not None:env.close()
        sets[method]=(np.concatenate([r['input'] for r in rrlist]),np.concatenate([r['action'] for r in rrlist]))
        print('MW library',task,bank,method,'done',round(time.perf_counter()-st),'s',flush=True)
    pd.DataFrame(stats).to_csv(b/'collection_cost.csv',index=False)
    for seed in SEEDS:
        effective=bank*1000+seed
        for method,(ax,ay) in sets.items():
            xx,yy=ex.dataset(x,y,ax,ay,effective)
            fit(xx,yy,vx,vy,norm,effective,b/'policies'/str(seed)/(method+'.pt'))
    dump(b/'prepared.json',dict(seconds=time.perf_counter()-st,amplitudes=amps));return str(b)
def evaluate(job):
    task,bank,seed,method=job;b=base(task,bank);path=b/'policies'/str(seed)/(method+'.pt');m,ck=ex.load_model(path);rows=[]
    for layout in LAYOUTS:
        p=b/'test'/str(seed)/method/f'{layout}.npz'
        if p.exists():rr=ex.load_npz(p)
        else:rr=ex.rollout(task,layout,m,ck);ex.save_npz(p,**rr)
        rows.append(dict(task=task,bank=bank,seed=seed,method=method,layout=layout,success=int(rr['success'].max()),terminal_success=int(rr['success'][-1]),checkpoint_sha256=sha(path),rollout_sha256=sha(p)))
    return rows
def main():
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=6);args=a.parse_args();freeze();start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for b in pool.map(prepare,[(t,b) for t in ex.TASKS for b in BANKS]):print('MW prepared',b,flush=True)
        rows=[]
        for rr in pool.map(evaluate,[(t,b,s,m) for t in ex.TASKS for b in BANKS for s in SEEDS for m in METHODS]):
            rows.extend(rr);pd.DataFrame(rows).to_csv(OUT/'per_episode.partial.csv',index=False)
            print('MW evaluation',len(rows),'/3240',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'per_episode.csv',index=False)
    dump(OUT/'completion.json',dict(episodes=len(rows),seconds=time.perf_counter()-start,protocol_sha256=sha(OUT/'protocol_frozen.json')))
if __name__=='__main__':main()
