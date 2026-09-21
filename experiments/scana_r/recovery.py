"""Second development iteration: verified short-pulse recovery branches.

Perturb eight actions, then execute the nominal reference for the entire tail.
Only observations after the pulse are paired with nominal action chunks. Every
accepted branch has actually executed those labels and achieved task success.
"""
from experiment import *
from scana_experiments.recovery_augmentation import RecoveryConfig,draw_pulse,apply_pulse,terminally_successful,recovery_windows
REC=WORK/'recovery_verified_development'
LEGACY=WORK/'recovery_development'
BRANCHES=12

def branch_one(args):
    task,mode,amp,i,k,source=args
    folder=REC/'data'/task/f'{mode}_{amp:.3f}'/str(source);folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{k}.npz'
    if path.exists():return str(path)
    r=dict(np.load(OLD/'demonstrations'/task/f'{source}.npz'))
    cal=dict(np.load(WORK/'calibration'/f'{task}.npz'))
    rng=np.random.default_rng(510000+i*100+k)
    # Stratify pulse endpoints over the original trajectory; never use test poses.
    grid=np.array_split(np.arange(16,369,8),BRANCHES);t=int(rng.choice(grid[k]))
    cfg=RecoveryConfig(amplitude=amp)
    e=draw_pulse(cal['error'],cal['time'],t,rng,cfg,mode)
    attempts=[];accepted=None
    for attempt,shrink in enumerate([1.,.5,.25,0.]):
        a=apply_pulse(r['action'],e,t,shrink,cfg)
        legacy=LEGACY/'data'/task/f'{mode}_{amp:.3f}'/str(source)/f'{k}_attempt{attempt}.npz'
        rr=None
        if legacy.exists():
            try:
                candidate=dict(np.load(legacy))
                if candidate['action'].shape==a.shape and np.array_equal(candidate['action'],a):rr=candidate
            except Exception:pass  # Interrupted legacy files are regenerated deterministically.
        if rr is None:rr=replay(task,r,a)
        ap=folder/f'{k}_attempt{attempt}.npz';np.savez_compressed(ap,**rr)
        terminal_success=int(terminally_successful(rr['rewards']))
        attempts.append(dict(attempt=attempt,scale=shrink,success=int(rr['success']),terminal_success=terminal_success,file=ap.name))
        if terminal_success:accepted=rr;break
    assert accepted is not None
    xx,yy,times=recovery_windows(accepted['input'],accepted['action'],r['action'],t,cfg)
    tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,input=xx,action=yy,source=source,time=times,endpoint=t,noise=e*shrink,scale=shrink,success=accepted['success'])
    tmp.replace(path);dump(folder/f'{k}_audit.json',dict(task=task,source=source,branch=k,mode=mode,amplitude=amp,endpoint=t,accepted_scale=shrink,attempts=attempts))
    return str(path)

def recovery_set(task,mode,amp,pool):
    tr,_=records(task);crossfit(task)
    jobs=[(task,mode,amp,i,k,int(r['seed'])) for i,r in enumerate(tr) for k in range(BRANCHES)]
    files=list(pool.map(branch_one,jobs));rs=[dict(np.load(p)) for p in files]
    x=np.concatenate([r['input'] for r in rs]);y=np.concatenate([r['action'] for r in rs]);scales=[float(r['scale']) for r in rs]
    print(task,mode,amp,'branches',len(rs),'nonzero',sum(s>0 for s in scales),'full',sum(s==1 for s in scales),'windows',len(x),flush=True)
    dump(REC/'data'/task/f'{mode}_{amp:.3f}'/'summary.json',dict(branches=len(rs),windows=len(x),nonzero=sum(s>0 for s in scales),full=sum(s==1 for s in scales),mean_scale=float(np.mean(scales)),source_coverage=len(set(int(r['source']) for r in rs))))
    return x,y

def development(amps,seeds):
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for task in ['transfer_cube','insertion']:
            tr,va=records(task);x,y,_,_=windows(tr);vx,vy,_,_=windows(va);xs,ys=Standardizer(x),Standardizer(y)
            sets={'Clean repeat':(x,y)}
            for mode in ['calibrated','gaussian']:
                for amp in amps:sets[f'{mode}_{amp:.3f}']=recovery_set(task,mode,amp,pool)
            for seed in seeds:
                rng=np.random.default_rng(seed+80000);bi=rng.integers(len(x),size=2048);u=rng.random(2048)
                for name,(ax,ay) in sets.items():
                    ai=np.minimum((u*len(ax)).astype(int),len(ax)-1)
                    path=REC/'policies'/task/str(seed)/(name+'.pt')
                    if name=='Clean repeat':path=WORK/'development/policies'/task/str(seed)/(name+'.pt')
                    fit(np.concatenate([x[bi],ax[ai]]),np.concatenate([y[bi],ay[ai]]),vx,vy,xs,ys,seed,path)
                    jobs=[]
                    for r in va:
                        layout=int(r['seed']);op=REC/'rollouts'/task/str(seed)/name/f'{layout}.npz';op.parent.mkdir(parents=True,exist_ok=True)
                        jobs.append((task,layout,str(path),str(op)))
                    for op in pool.map(rollout_one,jobs):
                        rr=dict(np.load(op));rows.append(dict(task=task,seed=seed,method=name,layout=int(Path(op).stem),success=int(rr['success']),max_reward=float(rr['max_reward']),checkpoint_sha256=sha(path)))
                    savecsv(REC/'per_episode.csv',rows)
                    print(task,seed,name,'development success',sum(r['success'] for r in rows if r['task']==task and r['seed']==seed and r['method']==name),'/10',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--amps',type=float,nargs='+',default=[.03,.08]);ap.add_argument('--seeds',type=int,nargs='+',default=[7,11,23]);args=ap.parse_args()
    dump(REC/'protocol.json',dict(stage='development iteration 3; strict terminal-success repair; no independent tests accessed',pulses=12,pulse_steps=8,recovery_labels='actual executed nominal action chunks after pulse; final reward must equal four',amplitudes=args.amps,seeds=args.seeds,backtracking=[1,.5,.25,0],budget='same 4096 training draws / 1200 steps, source coverage preserved by identity fallback',calibration='five-fold source-episode crossfitted prediction error; same task, within eight steps; includes bias',selection='ten original selection layouts per task; prospective test still sealed'))
    development(args.amps,args.seeds)
