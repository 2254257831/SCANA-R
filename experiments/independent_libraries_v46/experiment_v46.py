"""Prospective replication, component controls and failure interventions.

Uses the frozen ACT physics and source demonstrations. All methods and layout
IDs are fixed before evaluation; historical files are read-only.
"""
from pathlib import Path
import sys,os,json,time,hashlib,argparse,copy
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
W=Path(os.environ.get('SCANA_V46_OUTPUT',REPO/'artifacts/independent_libraries_v46')).resolve();W.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(HERE/'runtime'),str(REPO/'src'),str(REPO/'experiments/scana_r')]
import tempfile
os.environ.setdefault('SCANA_ASSET_CACHE',str(Path(tempfile.gettempdir())/'scana_v46_assets'))
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import experiment as ex
import numpy as np
import torch
import pandas as pd
from scana_experiments.recovery_augmentation import RecoveryConfig,draw_pulse,apply_pulse,recovery_windows,terminally_successful
torch.set_num_threads(1)
OUT=W/'evidence';OUT.mkdir(exist_ok=True)
TASKS=['transfer_cube','insertion'];BANKS=[101,211,307,401,503];SEEDS=[17,29,43]
METHODS=['Clean repeat','SCANA-R','Gaussian recovery','No cross-fitting','Global-time errors','No intermediate backtracking','Gaussian CPU budget']
FAMILIES=METHODS[1:6]
LAYOUTS=list(range(97000,97020))
CONDITIONS=['nominal','observation_noise','noise_joints','noise_objects','noise_low_support','noise_supported','replan_1','replan_1_ensemble','replan_16']
DIAG_LAYOUTS=list(range(96000,96020));OLDSEEDS=[7,11,23,31,47]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def safe_name(s):return s.replace(' ','_')
def bankdir(task,bank):return OUT/'replication'/task/str(bank)
def records(task):return ex.records(task)
def cpu():return time.process_time()

def freeze():
    p=dict(version=1,frozen_utc=datetime.now(timezone.utc).isoformat(),tasks=TASKS,banks=BANKS,policy_seeds=SEEDS,
      methods=METHODS,test_layouts=LAYOUTS,source_episodes='same original 40 training and 10 development episodes; no new source demonstrations',
      library_randomization='independent fold assignment, auxiliary initializations, pulse RNG and endpoint draws for each bank',
      training='1200 updates per auxiliary/final policy; 2048 original +2048 method rows with replacement; min clean development MSE every 100 updates; all random streams paired by bank/seed',
      amplitude={'transfer_cube':.03,'insertion':.08},branches_per_source=12,
      ablations={'No cross-fitting':'one all-training-episode auxiliary model, same 1200 updates',
         'Global-time errors':'sample any error chunk in the same task, remove temporal neighborhood only',
         'No intermediate backtracking':'try scales [1,0] instead of [1,.5,.25,0]; identity retains source coverage'},
      compute_control='Gaussian CPU budget receives the SCANA-R total active CPU time (five auxiliary fits + collection simulation + one final fit), less its own Gaussian collection time, as its final-training allowance; minimum 1200 updates and ceiling 30000; any overrun/unused budget reported; no auxiliary work charged twice per method/seed',
      primary='SCANA-R minus Gaussian recovery at nominal cadence, separately on two tasks; all results reported; paired bank, within-bank seed, shared-layout bootstrap 95% and two-task 97.5% intervals',
      diagnostics=dict(frozen_methods=['Calibrated recovery','Gaussian recovery'],seeds=OLDSEEDS,layouts=DIAG_LAYOUTS,conditions=CONDITIONS,
        low_support='input channels with test noise standard deviation greater than original-training standard deviation; mask fixed from checkpoint before test',
        ensemble='predict every step, average all available predictions of the present action over the last 16 chunks with weights exp(-age/4); no tuning',
        metrics='episode and terminal success, each ordinal reward threshold reached, action first differences, clean-vs-noisy prediction discrepancy at identical states',
        interpretation='retrospectively motivated mechanisms tested on new layouts; channel interventions identify sensitivity, not universally proven causes'),
      scope='state-input simulation; independent libraries conditional on same source demos; no claims about images, semantic task planning or real robots',
      script_sha256=sha(__file__))
    dest=OUT/'protocol_frozen.json'
    if dest.exists():
        old=json.loads(dest.read_text(encoding='utf-8'))
        assert {k:v for k,v in old.items() if k!='frozen_utc'}=={k:v for k,v in p.items() if k!='frozen_utc'},'Protocol changed; use a new study directory.'
    else:dump(dest,p)
    return p

def fit(x,y,vx,vy,xs,ys,seed,path,steps=1200,budget=None):
    path=Path(path)
    if path.exists():return ex.load_policy(path)
    start=cpu();wall=time.perf_counter()
    torch.manual_seed(seed+90000);m=ex.ChunkPolicy(x.shape[1],224);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    tx=torch.tensor(xs.transform(x));ty=torch.tensor(ys.transform(y));vv=torch.tensor(xs.transform(vx));vt=torch.tensor(ys.transform(vy))
    rng=np.random.default_rng(seed+160000);best=float('inf');bst=None;trace=[]
    limit=steps if budget is None else 30000
    for step in range(limit):
        idx=rng.integers(len(tx),size=128);loss=((m(tx[idx])-ty[idx])**2).mean();opt.zero_grad();loss.backward();opt.step()
        if (step+1)%100==0:
            m.eval()
            with torch.no_grad():err=float(((m(vv)-vt)**2).mean())
            trace.append(dict(step=step+1,selection_mse=err,cpu_seconds=cpu()-start))
            if err<best:best=err;bst={k:v.detach().clone() for k,v in m.state_dict().items()};beststep=step+1
            m.train()
            if budget is not None and step+1>=steps and cpu()-start>=budget:break
    ck=dict(model=bst,input_mean=xs.mean,input_std=xs.std,output_mean=ys.mean,output_std=ys.std,seed=seed,best_step=beststep,
        actual_steps=step+1,selection_mse=best,trace=trace,cpu_seconds=cpu()-start,wall_seconds=time.perf_counter()-wall,cpu_allowance=budget)
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(ck,path)
    dump(path.with_suffix('.json'),{k:v for k,v in ck.items() if k not in ['model','input_mean','input_std','output_mean','output_std']})
    m.load_state_dict(bst);m.eval();return m,ck

def calibration(job):
    task,bank=job;dest=bankdir(task,bank)/'calibration.npz'
    if dest.exists():return str(dest)
    tr,va=records(task);x,y,ep,tt=ex.windows(tr);rng=np.random.default_rng(bank+610000)
    ids=np.array([int(r['seed']) for r in tr]);rng.shuffle(ids)
    errors=np.zeros_like(y);folds=np.zeros(len(x),int);cost=0
    for fold in range(5):
        hold=np.isin(ep,ids[fold*8:(fold+1)*8]);folds[hold]=fold
        xs,ys=ex.Standardizer(x[~hold]),ex.Standardizer(y[~hold])
        m,ck=fit(x[~hold],y[~hold],x[~hold],y[~hold],xs,ys,bank*100+fold,bankdir(task,bank)/'auxiliary'/f'fold{fold}.pt')
        errors[hold]=ex.predict(m,ck,x[hold])-y[hold];cost+=ck['cpu_seconds']
    xs,ys=ex.Standardizer(x),ex.Standardizer(y)
    m,ck=fit(x,y,x,y,xs,ys,bank*100+19,bankdir(task,bank)/'auxiliary'/'in_sample.pt')
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,error=errors.reshape(-1,16,14),insample_error=(ex.predict(m,ck,x)-y).reshape(-1,16,14),time=tt,episode=ep,fold=folds,
        crossfit_cpu=cost,insample_cpu=ck['cpu_seconds'])
    return str(dest)

def collect_branch(job):
    task,bank,method,i,k,source=job;folder=bankdir(task,bank)/'branches'/safe_name(method)/str(source);dest=folder/f'{k}.npz'
    if dest.exists():return str(dest)
    r=dict(np.load(ex.OLD/'demonstrations'/task/f'{source}.npz'));cal=dict(np.load(bankdir(task,bank)/'calibration.npz'))
    rng=np.random.default_rng(510000+bank*10000+i*100+k)
    grid=np.array_split(np.arange(16,369,8),12);t=int(rng.choice(grid[k]));cfg=RecoveryConfig(amplitude=.03 if task=='transfer_cube' else .08)
    error=cal['insample_error'] if method=='No cross-fitting' else cal['error']
    if method=='Global-time errors':
        cfg=RecoveryConfig(amplitude=cfg.amplitude,phase_radius=10000)
    e=draw_pulse(error,cal['time'],t,rng,cfg,'gaussian' if method=='Gaussian recovery' else 'calibrated')
    scales=[1.,0.] if method=='No intermediate backtracking' else [1.,.5,.25,0.]
    rewards=[];tried=[];seconds=0;accepted=None
    for s in scales:
        a=apply_pulse(r['action'],e,t,s,cfg);start=cpu();rr=ex.replay(task,r,a);seconds+=cpu()-start
        tried.append(s);rewards.append(rr['rewards'])
        if terminally_successful(rr['rewards']):accepted=rr;break
    assert accepted is not None,'Verified original terminal replay failed'
    xx,yy,times=recovery_windows(accepted['input'],accepted['action'],r['action'],t,cfg)
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,input=xx,action=yy,time=times,source=source,endpoint=t,noise=e,scale=s,attempt_scales=tried,
        attempt_rewards=np.asarray(rewards,dtype=np.int8),simulation_cpu=seconds,
        accepted_input=accepted['input'].astype(np.float32),accepted_action=accepted['action'],reference_input=r['input'][times])
    return str(dest)

def merge_library(job):
    task,bank,method=job;dest=bankdir(task,bank)/'libraries'/(safe_name(method)+'.npz')
    if dest.exists():return str(dest)
    paths=sorted((bankdir(task,bank)/'branches'/safe_name(method)).glob('*/*.npz'))
    assert len(paths)==480,(method,len(paths))
    rs=[dict(np.load(p)) for p in paths];x=np.concatenate([r['input'] for r in rs]);y=np.concatenate([r['action'] for r in rs])
    scales=np.array([float(r['scale']) for r in rs]);refs=np.concatenate([r['reference_input'] for r in rs]);tr,_=records(task)
    rawx,_,_,_=ex.windows(tr);xs=ex.Standardizer(rawx)
    shift=np.linalg.norm((x-refs)/xs.std,axis=1)
    summary=dict(branches=len(rs),windows=len(x),identity=int(sum(scales==0)),full=int(sum(scales==1)),mean_scale=float(scales.mean()),
        attempts=sum(len(r['attempt_scales']) for r in rs),simulation_steps=sum(len(r['attempt_scales'])*400 for r in rs),simulation_cpu=sum(float(r['simulation_cpu']) for r in rs),
        state_shift_median=float(np.median(shift)),state_shift_p90=float(np.quantile(shift,.9)),nonzero_windows=int(sum(shift>1e-6)))
    dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,input=x,action=y)
    dump(dest.with_suffix('.json'),summary);return str(dest)

def fit_final(job):
    task,bank,seed,method=job;d=bankdir(task,bank);dest=d/'policies'/str(seed)/(safe_name(method)+'.pt')
    if dest.exists():return str(dest)
    tr,va=records(task);x,y,_,_=ex.windows(tr);vx,vy,_,_=ex.windows(va);xs,ys=ex.Standardizer(x),ex.Standardizer(y)
    mode='Gaussian recovery' if method=='Gaussian CPU budget' else method
    if mode=='Clean repeat':ax,ay=x,y
    else:
        lib=np.load(d/'libraries'/(safe_name(mode)+'.npz'));ax,ay=lib['input'],lib['action']
    stream=bank*1000+seed;rng=np.random.default_rng(stream+80000);bi=rng.integers(len(x),size=2048);u=rng.random(2048);ai=np.minimum((u*len(ax)).astype(int),len(ax)-1)
    budget=None
    if method=='Gaussian CPU budget':
        aux=float(np.load(d/'calibration.npz')['crossfit_cpu'])
        sc=json.loads((d/'policies'/str(seed)/'SCANA-R.json').read_text())
        c=json.loads((d/'libraries'/'SCANA-R.json').read_text());g=json.loads((d/'libraries'/'Gaussian_recovery.json').read_text())
        budget=max(.001,aux+sc['cpu_seconds']+c['simulation_cpu']-g['simulation_cpu'])
    fit(np.concatenate([x[bi],ax[ai]]),np.concatenate([y[bi],ay[ai]]),vx,vy,xs,ys,stream,dest,budget=budget)
    return str(dest)

def evaluate_one(job):
    task,bank,seed,method,layout=job;d=bankdir(task,bank);dest=d/'test'/str(seed)/safe_name(method)/f'{layout}.npz';ckpath=d/'policies'/str(seed)/(safe_name(method)+'.pt')
    if not dest.exists():
        m,ck=ex.load_policy(ckpath);start=cpu();rr=ex.rollout(task,layout,m,ck);rr['evaluation_cpu']=cpu()-start
        dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**rr)
    rr=dict(np.load(dest));return dict(task=task,bank=bank,seed=seed,method=method,layout=layout,success=int(rr['success']),terminal_success=int(rr['reward'][-1]==4),
        max_reward=float(rr['max_reward']),checkpoint_sha256=sha(ckpath),rollout_sha256=sha(dest))

def diagnostic(job):
    task,method,seed,layout,condition=job;dest=OUT/'failure_interventions'/task/str(seed)/safe_name(method)/condition/f'{layout}.npz'
    ckpath=ex.WORK/'independent_test/policies'/task/str(seed)/(method+'.pt')
    if not dest.exists():
        model,ck=ex.load_policy(ckpath);ex.sim_env.BOX_POSE[0]=ex.initial_pose(task,layout);env=ex.sim_env.make_sim_env('sim_'+task);obs=env.reset().observation
        rng=np.random.default_rng(layout+730000);rng.normal(size=(8,14)) # same stream convention as archived stress tests
        width=len(ck['input_mean']);sigma=np.r_[np.full(14,.01),np.full(width-15,.005),0.];sigma[[6,13]]=0
        perturb=rng.normal(size=(400,width))*sigma;low=(sigma>ck['input_std'])
        mask=np.ones(width,dtype=bool)
        if condition=='noise_joints':mask[14:]=False
        elif condition=='noise_objects':mask[:14]=False
        elif condition=='noise_low_support':mask=low
        elif condition=='noise_supported':mask=~low
        noisy=condition=='observation_noise' or condition.startswith('noise_')
        cadence=1 if condition.startswith('replan_1') and condition!='replan_16' else (16 if condition=='replan_16' else 8)
        xx=[];clean=[];aa=[];rr=[];deviation=[];history=[]
        try:
            for t in range(400):
                base=ex.causal_observation(obs,t);x=base+perturb[t]*mask if noisy else base.copy()
                clean.append(base);xx.append(x)
                if t%cadence==0:
                    chunk=ex.predict(model,ck,x[None]).reshape(16,14)
                    if noisy:
                        p0=ex.predict(model,ck,base[None]).reshape(16,14);deviation.append(float(np.sqrt(np.mean((chunk[:,ex.JOINTS]-p0[:,ex.JOINTS])**2))))
                    if condition=='replan_1_ensemble':history.append((t,chunk));history=history[-16:]
                if condition=='replan_1_ensemble':
                    age=np.array([t-s for s,_ in history]);weight=np.exp(-age/4);action=np.average(np.array([p[t-s] for s,p in history]),axis=0,weights=weight)
                else:action=chunk[t%cadence]
                aa.append(action.copy());ts=env.step(action);obs=ts.observation;rr.append(float(ts.reward))
        finally:env.close()
        dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,input=np.asarray(xx,dtype=np.float32),clean_input=np.asarray(clean),action=np.asarray(aa),reward=np.asarray(rr,dtype=np.int8),
            low_support_mask=low,prediction_rms=deviation,standardized_noise_sigma=sigma/ck['input_std'])
    z=dict(np.load(dest));r=z['reward'];a=z['action'];row=dict(task=task,method=method,seed=seed,layout=layout,condition=condition,
        success=int(max(r)==4),terminal_success=int(r[-1]==4),action_step_rms=float(np.sqrt(np.mean(np.diff(a[:,ex.JOINTS],axis=0)**2))),
        prediction_noise_rms=float(np.mean(z['prediction_rms'])) if len(z['prediction_rms']) else 0.,checkpoint_sha256=sha(ckpath),rollout_sha256=sha(dest))
    for stage in [1,2,3,4]:
        hit=np.flatnonzero(r>=stage);row[f'reach_reward_{stage}']=int(len(hit)>0);row[f'first_reward_{stage}']=int(hit[0]) if len(hit) else -1
    return row

def progress(pool,fn,jobs,label):
    result=[];t=time.perf_counter()
    for i,r in enumerate(pool.map(fn,jobs,chunksize=1),1):
        result.append(r)
        if i%100==0 or i==len(jobs):print(label,i,'/',len(jobs),'seconds',round(time.perf_counter()-t),flush=True)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['calibrate','collect','fit','test','diagnose','all']);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args();freeze()
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.stage in ['calibrate','all']:progress(pool,calibration,[(t,b) for t in TASKS for b in BANKS],'calibration')
        if args.stage in ['collect','all']:
            jobs=[(t,b,m,i,k,int(r['seed'])) for t in TASKS for b in BANKS for m in FAMILIES for i,r in enumerate(records(t)[0]) for k in range(12)]
            progress(pool,collect_branch,jobs,'collection');progress(pool,merge_library,[(t,b,m) for t in TASKS for b in BANKS for m in FAMILIES],'merge')
        if args.stage in ['fit','all']:
            progress(pool,fit_final,[(t,b,s,m) for t in TASKS for b in BANKS for s in SEEDS for m in METHODS[:-1]],'final fit')
            progress(pool,fit_final,[(t,b,s,METHODS[-1]) for t in TASKS for b in BANKS for s in SEEDS],'CPU budget fit')
        if args.stage in ['test','all']:
            rows=progress(pool,evaluate_one,[(t,b,s,m,l) for t in TASKS for b in BANKS for s in SEEDS for m in METHODS for l in LAYOUTS],'independent tests')
            pd.DataFrame(rows).to_csv(OUT/'replication_per_episode.csv',index=False)
        if args.stage in ['diagnose','all']:
            rows=progress(pool,diagnostic,[(t,m,s,l,c) for t in TASKS for m in ['Calibrated recovery','Gaussian recovery'] for s in OLDSEEDS for l in DIAG_LAYOUTS for c in CONDITIONS],'failure interventions')
            pd.DataFrame(rows).to_csv(OUT/'failure_interventions.csv',index=False)
    dump(OUT/f'completion_{args.stage}.json',dict(stage=args.stage,wall_seconds=time.perf_counter()-start,protocol_sha256=sha(OUT/'protocol_frozen.json'),complete=True))

if __name__=='__main__':main()
