"""Prospective six-task extension: prepare -> freeze -> evaluate -> analyze.

Uses official Meta-World environments with a disclosed 250-step horizon, current
state input and a common expert demonstration endpoint-hold rule. The original
ACT algorithm implementation and evidence are not changed.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import argparse, hashlib, json, sys, time, warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.spatial import cKDTree
from bridge import TASKS, HORIZON, make_env, current_input, reference, execute

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from scana_experiments.recovery_augmentation import RecoveryConfig,draw_pulse,apply_pulse,recovery_windows
OUT=ROOT/'artifacts/metaworld_extension_v1'
SEEDS=[7,11,23,31,47]
METHODS=['Clean repeat','Gaussian recovery','Frozen observations','SCANA-R','Demonstration kNN']
AMPS=[.10,.25]
BRANCHES=6
LAYOUTS=list(range(30000,30020))
torch.set_num_threads(1)
warnings.filterwarnings('ignore',message=r'Constant\(s\) may be too high.*',category=UserWarning)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
def load_npz(path):
    with np.load(path) as f:return {k:f[k] for k in f}
def save_npz(path,**data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**data);temp.replace(path)

class Policy(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(22,256),nn.LayerNorm(256),nn.SiLU(),
            nn.Linear(256,256),nn.LayerNorm(256),nn.SiLU(),nn.Linear(256,64))
    def forward(self,x):return self.net(x)

def norms(x,y):
    return dict(input_mean=x.mean(0),input_std=np.maximum(x.std(0),1e-5),
                output_mean=y.mean(0),output_std=np.maximum(y.std(0),1e-5))

def predict(model,ck,x):
    a=torch.as_tensor((np.asarray(x,dtype=np.float32)-ck['input_mean'])/ck['input_std'])
    with torch.no_grad():p=model(a).numpy()
    return p*ck['output_std']+ck['output_mean']

def load_model(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    m=Policy();m.load_state_dict(ck['model']);m.eval();return m,ck

def fit(x,y,vx,vy,norm,seed,path):
    if path.exists():return load_model(path)
    torch.manual_seed(seed+90000);m=Policy();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    xx=torch.as_tensor((x-norm['input_mean'])/norm['input_std'])
    yy=torch.as_tensor((y-norm['output_mean'])/norm['output_std'])
    xv=torch.as_tensor((vx-norm['input_mean'])/norm['input_std'])
    yv=torch.as_tensor((vy-norm['output_mean'])/norm['output_std'])
    rng=np.random.default_rng(seed+160000);best=float('inf');trace=[];start=time.monotonic()
    for step in range(1200):
        idx=rng.integers(len(xx),size=128);loss=((m(xx[idx])-yy[idx])**2).mean()
        opt.zero_grad();loss.backward();opt.step()
        if (step+1)%100==0:
            m.eval()
            with torch.no_grad():value=float(((m(xv)-yv)**2).mean())
            trace.append(dict(step=step+1,mse=value))
            if value<best:
                best=value;best_step=step+1;weights={k:v.detach().clone() for k,v in m.state_dict().items()}
            m.train()
    ck=dict(**norm,model=weights,seed=seed,best_step=best_step,selection_mse=best,trace=trace,
            training_seconds=time.monotonic()-start,input_contract='current 18D state + current goal xyz + elapsed t/250',
            output_contract='16x4 bounded Cartesian delta/gripper command chunk')
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(ck,path)
    dump(path.with_suffix('.json'),{k:v for k,v in ck.items() if k not in ['model',*norm.keys()]})
    m.load_state_dict(weights);m.eval();return m,ck

def collect(task):
    sets=[];audit=[]
    for split,count,base in [('train',40,10000),('development',10,20000)]:
        records=[]
        for seed in range(base,base+1000):
            path=OUT/'references'/task/split/f'{seed}.npz'
            if path.exists():r=load_npz(path)
            else:r=reference(task,seed);save_npz(path,**r)
            ok=bool(r['success'][-1]);audit.append(dict(task=task,split=split,seed=seed,
                ever_success=int(r['success'].max()),terminal_success=int(ok),sha256=sha(path)))
            if ok:r['seed']=seed;records.append(r)
            if len(records)==count:break
        if len(records)!=count:raise RuntimeError(f'{task}: insufficient successful references')
        sets.append(records)
    vectors=[tuple(np.round(r['initial_rand_vec'],10)) for rs in sets for r in rs]
    assert len(set(vectors))==len(vectors),'Duplicate training/development initial layouts'
    dump(OUT/'references'/task/'manifest.json',audit)
    return sets

def windows(records):
    x=[];y=[];ep=[];tt=[]
    for r in records:
        for t in range(0,HORIZON-15,8):
            x.append(r['input'][t]);y.append(r['action'][t:t+16].reshape(-1));ep.append(r['seed']);tt.append(t)
    return np.asarray(x,dtype=np.float32),np.asarray(y,dtype=np.float32),np.asarray(ep),np.asarray(tt)

def crossfit(task,records,x,y,ep,tt):
    path=OUT/'calibration'/f'{task}.npz'
    if path.exists():return load_npz(path)
    errors=np.zeros_like(y)
    for fold in range(5):
        ids=[r['seed'] for r in records[8*fold:8*(fold+1)]]
        held=np.isin(ep,ids);norm=norms(x[~held],y[~held])
        model,ck=fit(x[~held],y[~held],x[~held],y[~held],norm,130+fold,OUT/'calibration'/task/f'fold{fold}.pt')
        errors[held]=predict(model,ck,x[held])-y[held]
    save_npz(path,error=errors.reshape(-1,16,4),time=tt,episode=ep)
    return load_npz(path)

def recovery_set(task,records,cal,mode,amp):
    base=OUT/'recovery'/task/f'{mode}_{amp:.2f}'
    sets=[];stats=[]
    for source_index,r in enumerate(records):
        env=None
        try:
            for branch in range(BRANCHES):
                path=base/str(r['seed'])/f'{branch}.npz'
                if path.exists():rr=load_npz(path)
                else:
                    if env is None:env=make_env(task,r['seed'])
                    rng=np.random.default_rng(510000+source_index*100+branch)
                    grid=np.array_split(np.arange(16,HORIZON-31,8),BRANCHES)
                    end=int(rng.choice(grid[branch]));cfg=RecoveryConfig(amplitude=amp,gripper_channels=(3,))
                    pulse=draw_pulse(cal['error'],cal['time'],end,rng,cfg,mode)
                    attempts=[];accepted=None
                    for j,scale in enumerate(cfg.backtracking):
                        action=np.clip(apply_pulse(r['action'],pulse,end,scale,cfg),-1,1).astype(np.float32)
                        attempt=execute(env,action)
                        ap=base/str(r['seed'])/f'{branch}_attempt{j}.npz';save_npz(ap,**attempt)
                        ok=bool(attempt['success'][-1]);attempts.append(dict(scale=scale,terminal_success=int(ok),file=ap.name,sha256=sha(ap)))
                        if ok:accepted=attempt;break
                    if accepted is None:raise RuntimeError(f'Nominal replay failed {task} {r["seed"]}')
                    xx,yy,times=recovery_windows(accepted['input'],accepted['action'],r['action'],end,cfg)
                    assert np.array_equal(accepted['action'][:,3],r['action'][:,3]),'Gripper changed'
                    actual=accepted['action'][end-8:end,:3]-r['action'][end-8:end,:3]
                    rr=dict(input=xx.astype(np.float32),action=yy.astype(np.float32),time=times,
                            source=r['seed'],endpoint=end,scale=scale,actual_pulse_rms=float(np.sqrt(np.mean(actual**2))))
                    save_npz(path,**rr);dump(path.with_suffix('.json'),dict(attempts=attempts,accepted_scale=scale,source=r['seed'],branch=branch))
                sets.append(rr)
                audit=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
                stats.append(dict(task=task,mode=mode,amplitude=amp,source=r['seed'],branch=branch,
                    attempts=len(audit['attempts']),accepted_scale=float(rr['scale']),actual_pulse_rms=float(rr['actual_pulse_rms']),windows=len(rr['input'])))
        finally:
            if env is not None:env.close()
    pd.DataFrame(stats).to_csv(base/'cost.csv',index=False)
    origin={r['seed']:r for r in records}
    frozen=np.concatenate([origin[int(r['source'])]['input'][r['time']] for r in sets])
    return np.concatenate([r['input'] for r in sets]),np.concatenate([r['action'] for r in sets]),frozen

def dataset(x,y,ax,ay,seed):
    rng=np.random.default_rng(seed+80000);bi=rng.integers(len(x),size=2048);u=rng.random(2048)
    ai=np.minimum((u*len(ax)).astype(int),len(ax)-1)
    return np.concatenate([x[bi],ax[ai]]),np.concatenate([y[bi],ay[ai]])

def rollout(task,layout,model=None,ck=None,knn=None,scenario='nominal',env=None):
    own=env is None
    if own:env=make_env(task,layout)
    obs,_=env.reset();xs=[];actions=[];rewards=[];success=[];raw=[]
    rng=np.random.default_rng(layout+770000);pulse=rng.normal(size=(8,4));pulse[:,3]=0
    pulse*=.25/np.sqrt(np.mean(pulse[:,:3]**2));pulse=np.clip(pulse,-.75,.75)
    try:
        for t in range(HORIZON):
            x=current_input(obs,t);xs.append(x);raw.append(obs.copy())
            if t%8==0:
                if knn is None:chunk=predict(model,ck,x[None]).reshape(16,4)
                else:
                    tree,lib,n=knn;dist,idx=tree.query((x-n['input_mean'])/n['input_std'],k=5)
                    weight=1/np.maximum(dist,1e-6);weight/=weight.sum()
                    chunk=(lib[idx]*weight[:,None]).sum(0).reshape(16,4)
            action=chunk[t%8].copy()
            if scenario=='pulse' and 64<=t<72:action+=pulse[t-64]
            action=np.clip(action,-1,1).astype(np.float32);actions.append(action.copy())
            obs,reward,_,_,info=env.step(action);rewards.append(float(reward));success.append(int(info['success']))
    finally:
        if own:env.close()
    return dict(input=np.asarray(xs),raw_obs=np.asarray(raw),action=np.asarray(actions),
                reward=np.asarray(rewards),success=np.asarray(success,dtype=np.int8))

def prepare_task(task):
    start=time.monotonic();tr,dev=collect(task);x,y,ep,tt=windows(tr);vx,vy,_,_=windows(dev);norm=norms(x,y)
    print(task,'references ready',len(x),'training windows',flush=True)
    cal=crossfit(task,tr,x,y,ep,tt);sets={'Clean repeat':(x,y)};frozen={}
    for mode in ['calibrated','gaussian']:
        for amp in AMPS:
            ax,ay,fx=recovery_set(task,tr,cal,mode,amp);name=f'{mode}_{amp:.2f}';sets[name]=(ax,ay);frozen[name]=(fx,ay)
            print(task,name,'recovery windows',len(ax),flush=True)
    devrows=[]
    for seed in [7,11]:
        for name,(ax,ay) in sets.items():
            xx,yy=dataset(x,y,ax,ay,seed);path=OUT/'development/policies'/task/str(seed)/(name+'.pt')
            model,ck=fit(xx,yy,vx,vy,norm,seed,path)
            for r in dev:
                op=OUT/'development/rollouts'/task/str(seed)/name/f'{r["seed"]}.npz'
                if op.exists():rr=load_npz(op)
                else:rr=rollout(task,r['seed'],model,ck);save_npz(op,**rr)
                devrows.append(dict(task=task,method=name,seed=seed,layout=r['seed'],success=int(rr['success'].max()),terminal_success=int(rr['success'][-1])))
    df=pd.DataFrame(devrows);df.to_csv(OUT/'development'/f'{task}.csv',index=False)
    chosen={}
    for mode in ['calibrated','gaussian']:
        chosen[mode]=max(AMPS,key=lambda a:(df[df.method==f'{mode}_{a:.2f}'].success.mean(),-a))
    dump(OUT/'development'/f'{task}_choice.json',dict(amplitudes=chosen,criterion='Mean ever-success over two seeds x ten development layouts; tie chooses lower amplitude.'))
    finalsets={'Clean repeat':sets['Clean repeat'],'Gaussian recovery':sets[f'gaussian_{chosen["gaussian"]:.2f}'],
               'SCANA-R':sets[f'calibrated_{chosen["calibrated"]:.2f}'],
               'Frozen observations':frozen[f'calibrated_{chosen["calibrated"]:.2f}']}
    for seed in SEEDS:
        for name,(ax,ay) in finalsets.items():
            xx,yy=dataset(x,y,ax,ay,seed)
            fit(xx,yy,vx,vy,norm,seed,OUT/'policies'/task/str(seed)/(name+'.pt'))
        xx,yy=dataset(x,y,x,y,seed)
        save_npz(OUT/'policies'/task/str(seed)/'Demonstration kNN.npz',input=xx,action=yy,**norm)
    dump(OUT/'prepared'/f'{task}.json',dict(task=task,seconds=time.monotonic()-start,train_windows=len(x),development_windows=len(vx),amplitudes=chosen))
    print(task,'PREPARED in',round(time.monotonic()-start),'seconds',flush=True)
    return task

def evaluate_group(job):
    task,method,seed=job
    if method=='Demonstration kNN':
        path=OUT/'policies'/task/str(seed)/(method+'.npz');ck=load_npz(path);model=None
        knn=(cKDTree((ck['input']-ck['input_mean'])/ck['input_std']),ck['action'],ck)
    else:path=OUT/'policies'/task/str(seed)/(method+'.pt');model,ck=load_model(path);knn=None
    rows=[]
    for layout in LAYOUTS:
        env=make_env(task,layout)
        try:
            for scenario in ['nominal','pulse']:
                op=OUT/'test/rollouts'/task/str(seed)/method/scenario/f'{layout}.npz'
                if op.exists():r=load_npz(op)
                else:r=rollout(task,layout,model,ck,knn,scenario,env);save_npz(op,**r)
                rows.append(dict(task=task,method=method,seed=seed,layout=layout,scenario=scenario,
                    success=int(r['success'].max()),terminal_success=int(r['success'][-1]),
                    post_pulse_success=int(r['success'][72:].max()),
                    first_success=next((i for i,v in enumerate(r['success']) if v),HORIZON),
                    checkpoint_sha256=sha(path),rollout_sha256=sha(op)))
        finally:env.close()
    return rows

def protocol():
    return dict(version=1,tasks=list(TASKS),metaworld='3.1.1',mujoco='3.3.0',gymnasium='1.1.1',
        horizon=HORIZON,scope='Six independent task-specific state policies; custom imitation protocol, not the official MT10/ML10 benchmark.',
        demonstration='Official task expert; clip actions to [-1,1]; after first success hold xyz command at zero and retain last gripper command. This stopping rule is used only for demonstration collection, never policy evaluation.',
        train='First 40 terminal-success reference episodes scanning seeds 10000..10999 per task; preserve failures.',
        development='First 10 terminal-success reference episodes scanning seeds 20000..20999 per task; preserve failures.',
        input='22D: current observation indices 0:18 + goal xyz 36:39 + t/250. No previous/future states, action-derived features or deployment success signal.',
        action='4D Cartesian xyz command and gripper in [-1,1]; 16-step prediction, execute first 8, reobserve.',
        network='22-256-LayerNorm-SiLU-256-LayerNorm-SiLU-64 MLP; AdamW lr=0.001 wd=0.0001; 1200 updates, batch128; best development MSE every100 steps.',
        calibration='Five source-episode folds; each auxiliary fits32 sources; choose fitting-fold training MSE checkpoint; each error computed only for held-out sources.',
        recovery=dict(branches_per_source=BRANCHES,pulse_steps=8,phase_radius=8,gripper_channels=[3],amplitudes=AMPS,
            bound='3x amplitude; also clip all executed commands to official [-1,1] actuator interface.',backtracking=[1,.5,.25,0],
            acceptance='Official success on the final step; record post-pulse actual observations and executed reference chunks.'),
        selection='Calibrated and correlated Gaussian amplitudes selected separately on development only, seeds7/11; mean ever-success, ties choose lower amplitude.',
        budget='4096 draws: 2048 original + 2048 method; same original indices, auxiliary index quantiles, initialization and minibatches. Frozen observations reuses SCANA-R source times and executed labels. kNN uses 4096 clean draws, 5 neighbors, inverse-distance mean in train-standardized current-state space.',
        training_seeds=SEEDS,test_layouts=LAYOUTS,test_scenarios=['nominal','pulse'],
        test_pulse='8 steps starting64; Cartesian-command RMS0.25, cap0.75; gripper unchanged; shared deterministic noise across methods and policy seeds.',
        primary='Macro-average SCANA-R minus Clean repeat ever-success over six fixed tasks, nominal condition. Taskwise and Gaussian contrasts always reported.',
        inference='Paired two-way bootstrap over policy seeds and layouts,5000 replicates. Macro-average conditional on six tasks and fixed recovery libraries. Six taskwise comparisons use 99.1667% marginal intervals for Bonferroni family control; raw95% intervals also reported.',
        limitations='Five trained seeds share a recovery bank per task/family/amplitude. State-based simulation only; no claims of language planning, visual VLA or real robots.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','freeze','evaluate']);ap.add_argument('--workers',type=int,default=6);a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);p=protocol();design=OUT/'design.json'
    if design.exists():assert json.loads(design.read_text())==p,'Design changed; create a new study version.'
    else:dump(design,p)
    if a.stage=='prepare':
        with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(prepare_task,list(TASKS)))
    elif a.stage=='freeze':
        for t in TASKS:assert (OUT/'prepared'/f'{t}.json').exists()
        frozen=OUT/'protocol_frozen.json'
        if frozen.exists():raise RuntimeError('Already frozen; do not refreeze after test access.')
        checkpoint_files=list((OUT/'policies').rglob('*.pt'))+list((OUT/'policies').rglob('*.npz'))
        p.update(frozen_utc=datetime.now(timezone.utc).isoformat(),
            code_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},
            checkpoints={f.relative_to(OUT).as_posix():sha(f) for f in checkpoint_files},
            choices={t:json.loads((OUT/'development'/f'{t}_choice.json').read_text()) for t in TASKS})
        dump(frozen,p);print('Frozen',len(checkpoint_files),'models/libraries before test access.',flush=True)
    else:
        frozen=OUT/'protocol_frozen.json';p=json.loads(frozen.read_text())
        for f,h in p['code_sha256'].items():assert sha(Path(__file__).parent/f)==h
        for f,h in p['checkpoints'].items():assert sha(OUT/f)==h
        rows=[];jobs=[(t,m,s) for t in TASKS for m in METHODS for s in SEEDS];start=time.monotonic()
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for rr in pool.map(evaluate_group,jobs):
                rows.extend(rr);pd.DataFrame(rows).to_csv(OUT/'per_episode.partial.csv',index=False)
                print('Meta-World test',len(rows),'/6000; elapsed',round(time.monotonic()-start),'s',flush=True)
        pd.DataFrame(rows).to_csv(OUT/'per_episode.csv',index=False)
        dump(OUT/'completion.json',dict(rollouts=len(rows),seconds=time.monotonic()-start,protocol_sha256=sha(frozen)))

if __name__=='__main__':main()
