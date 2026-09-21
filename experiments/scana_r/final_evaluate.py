"""Freeze from development outcomes, then evaluate once on fresh layouts."""
from experiment import *
from recovery import REC,BRANCHES
import pandas as pd
import shutil, datetime

FINAL=WORK/'independent_test'
METHODS=['Clean repeat','Calibrated recovery','Gaussian recovery','Frozen context','Old SCANA','Old kNN','Old deterministic','Clip only']
TEST_SEEDS=[7,11,23,31,47]
LAYOUTS=list(range(92000,92040))

def choose_and_freeze():
    path=FINAL/'protocol_frozen.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    df=pd.read_csv(REC/'per_episode.csv')
    assert len(df)==300,'Development incomplete: 2 tasks x 5 methods x 3 seeds x 10 layouts'
    choice={}
    for task in ['transfer_cube','insertion']:
        choice[task]={}
        for family in ['calibrated','gaussian']:
            rows=[]
            for amp in [.03,.08]:
                d=df[(df.task==task)&(df.method==f'{family}_{amp:.3f}')]
                assert len(d)==30
                rows.append((float(d.success.mean()),-amp,amp))
            score,_,amp=max(rows);choice[task][family]=dict(amplitude=amp,development_success=score)
    protocol=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage='one independent test after development',tasks=['transfer_cube','insertion'],methods=METHODS,seeds=TEST_SEEDS,layouts=LAYOUTS,choices=choice,
        choice_rule='Within each task and noise family, highest success over three development seeds x ten selection layouts; ties use smaller amplitude. Frozen-context ablation uses selected calibrated amplitude. Earlier continuous-noise iteration failed and is archived.',
        train='40 source episodes per task; same as v35; 12 eight-step pulse branches each; thereafter execute reference actions; require terminal reward 4; backtrack pulse scale [1,.5,.25,0]; all attempts retained in audit; 1908 actual recovery windows per method/task',
        protocol='Current qpos + current object xyz + elapsed clock; 16-action MLP; execute first 8; 400 steps, official ACT reward; no deployment access to reference actions or future states',
        budget='4096 resampled windows, exactly 2048 clean + 2048 method; same index quantiles, initialization and minibatch stream; 1200 optimization steps; checkpoint chosen by selection clean MSE',
        original_controls='Old generator-based SCANA/kNN/deterministic and clip-only checkpoints are reused from v35; raw action units restored exactly; no new training/layout evidence is supplied to them',
        primary='Calibrated recovery vs Clean repeat separately on both tasks; practical improvement >=.05 on each; report paired two-way seed/layout bootstrap intervals, not iid episode intervals',
        secondary='Calibrated recovery vs tuned generic Gaussian recovery; do not claim calibration-specific gain if difference is unsupported. Frozen context uses identical source times and nominal labels but original source observations.',
        acceptance='Both tasks exceed clean by >=5 percentage points; primary paired 95% empirical intervals exclude zero; task-specific calibrated-vs-Gaussian comparisons reported regardless of direction',
        multiplicity='Also report 97.5% paired intervals for the family of two primary comparisons; no adaptive stopping or changing algorithms after test access',
        files={'experiment.py':sha(SOURCE_DIR/'experiment.py'),'recovery.py':sha(SOURCE_DIR/'recovery.py'),'final_evaluate.py':sha(SOURCE_DIR/'final_evaluate.py'),'recovery_augmentation.py':sha(ROOT/'src/scana_experiments/recovery_augmentation.py'),'replay_augmentation.py':sha(ROOT/'src/scana_experiments/replay_augmentation.py'),'development_per_episode.csv':sha(REC/'per_episode.csv')},
        limits='Privileged-state simulation pilot, five independent policy trainings per method/task; not GR00T, vision-language, real robot or proof of novelty. Acceptance is practical simulation feasibility only.')
    dump(path,protocol);return protocol

def convert_old(task,seed,name,path):
    if path.exists():return
    old=OLD/'policies'/task/str(seed)/(name+'.pt')
    ck=torch.load(old,map_location='cpu',weights_only=False)
    norms=np.load(OLD/f'training_{task}.npz');scale=np.tile(norms['scale'],16);center=np.tile(norms['center'],16)
    ck['output_mean']=ck['output_mean']/50*scale+center;ck['output_std']=ck['output_std']/50*scale
    ck['original_checkpoint']=str(old);ck['original_sha256']=sha(old);ck['conversion']='normalized label unit to raw joint coordinates; no weight change'
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(ck,path)

def main(main_only=False):
    p=choose_and_freeze();rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for task in p['tasks']:
            tr,va=records(task);x,y,_,_=windows(tr);vx,vy,_,_=windows(va);xs,ys=Standardizer(x),Standardizer(y)
            sets={'Clean repeat':(x,y)}
            for mode,label in [('calibrated','Calibrated recovery'),('gaussian','Gaussian recovery')]:
                amp=p['choices'][task][mode]['amplitude']
                ar=[]
                for r in tr:
                    for branch in range(BRANCHES):ar.append(dict(np.load(REC/'data'/task/f'{mode}_{amp:.3f}'/str(int(r['seed']))/f'{branch}.npz')))
                ax=np.concatenate([r['input'] for r in ar]);ay=np.concatenate([r['action'] for r in ar]);sets[label]=(ax,ay)
                if mode=='calibrated':
                    origin={int(r['seed']):r for r in tr}
                    fx=np.concatenate([origin[int(r['source'])]['input'][r['time']] for r in ar]);sets['Frozen context']=(fx,ay)
            for seed in TEST_SEEDS:
                rng=np.random.default_rng(seed+80000);bi=rng.integers(len(x),size=2048);u=rng.random(2048)
                for name in (METHODS[:4] if main_only else METHODS):
                    path=FINAL/'policies'/task/str(seed)/(name+'.pt')
                    if name in sets:
                        ax,ay=sets[name];ai=np.minimum((u*len(ax)).astype(int),len(ax)-1)
                        xx=np.concatenate([x[bi],ax[ai]]);yy=np.concatenate([y[bi],ay[ai]])
                        reuse=None
                        if name=='Clean repeat':reuse=WORK/'development/policies'/task/str(seed)/'Clean repeat.pt'
                        elif name in ['Calibrated recovery','Gaussian recovery']:
                            family='calibrated' if name=='Calibrated recovery' else 'gaussian';amp=p['choices'][task][family]['amplitude']
                            reuse=REC/'policies'/task/str(seed)/f'{family}_{amp:.3f}.pt'
                        if reuse is not None and reuse.exists() and not path.exists():
                            path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(reuse,path)
                        fit(xx,yy,vx,vy,xs,ys,seed,path)
                    else:
                        oldname={'Old SCANA':'SCANA','Old kNN':'Conditional_kNN','Old deterministic':'Deterministic_MLP','Clip only':'Clip_only'}[name]
                        convert_old(task,seed,oldname,path)
                    jobs=[]
                    for layout in LAYOUTS:
                        op=FINAL/'rollouts'/task/str(seed)/name/f'{layout}.npz';op.parent.mkdir(parents=True,exist_ok=True)
                        jobs.append((task,layout,str(path),str(op)))
                    for op in pool.map(rollout_one,jobs):
                        r=dict(np.load(op));layout=int(Path(op).stem)
                        rows.append(dict(task=task,seed=seed,method=name,layout=layout,success=int(r['success']),terminal_success=int(r['reward'][-1]==4),max_reward=float(r['max_reward']),total_reward=float(r['total_reward']),checkpoint_sha256=sha(path)))
                    savecsv(FINAL/'per_episode.csv',rows)
                    print(task,seed,name,sum(r['success'] for r in rows if r['task']==task and r['seed']==seed and r['method']==name),'/40',flush=True)
    dump(FINAL/'completion.json',dict(complete=True,rollouts=len(rows),protocol_sha256=sha(FINAL/'protocol_frozen.json')))

def summarize():
    df=pd.read_csv(FINAL/'per_episode.csv');assert len(df)==3200
    rng=np.random.default_rng(20260919);si=rng.integers(5,size=(5000,5));li=rng.integers(40,size=(5000,40))
    summary=[];contrasts=[]
    for task in ['transfer_cube','insertion']:
        arrays={}
        for method in METHODS:
            a=df[(df.task==task)&(df.method==method)].pivot(index='seed',columns='layout',values='success').loc[TEST_SEEDS,LAYOUTS].to_numpy()
            arrays[method]=a;b=a[si[:,:,None],li[:,None,:]].mean((1,2));lo,hi=np.quantile(b,[.025,.975])
            summary.append(dict(task=task,method=method,successes=int(a.sum()),rollouts=a.size,mean=float(a.mean()),seed_sd=float(a.mean(1).std(ddof=1)),bootstrap_low=float(lo),bootstrap_high=float(hi)))
        for ref in METHODS:
            if ref=='Calibrated recovery':continue
            a=arrays['Calibrated recovery']-arrays[ref];b=a[si[:,:,None],li[:,None,:]].mean((1,2));lo,hi=np.quantile(b,[.025,.975]);al,ah=np.quantile(b,[.0125,.9875])
            contrasts.append(dict(task=task,method='Calibrated recovery',reference=ref,difference=float(a.mean()),paired_low=float(lo),paired_high=float(hi),family_adjusted_low=float(al),family_adjusted_high=float(ah)))
    savecsv(FINAL/'summary.csv',summary);savecsv(FINAL/'paired_contrasts.csv',contrasts)
    primary=[r for r in contrasts if r['reference']=='Clean repeat']
    dump(FINAL/'acceptance.json',dict(practical_simulation_feasibility=all(r['difference']>=.05 and r['paired_low']>0 for r in primary),both_adjusted_intervals_positive=all(r['family_adjusted_low']>0 for r in primary),primary=primary,calibration_comparison=[r for r in contrasts if r['reference']=='Gaussian recovery'],warning='Five trained seeds crossed with forty layouts per task. No independence claim for all 200 rollouts; no general VLA or real robot claim.'))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':
    main();summarize()
