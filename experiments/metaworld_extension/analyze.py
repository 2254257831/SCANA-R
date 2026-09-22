"""Recompute all extension tables, paired intervals and completeness checks."""
from pathlib import Path
import argparse, itertools, json, shutil
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
SEEDS=[7,11,23,31,47]

def analyze(name, expected_tasks, methods, scenarios, layout_start, ours):
    source=ROOT/'artifacts'/name; dest=ROOT/'results'/name
    path=source/'per_episode.csv'
    if not path.exists():return None
    d=pd.read_csv(path);keys=['task','method','seed','layout','scenario']
    expected=set(itertools.product(expected_tasks,methods,SEEDS,range(layout_start,layout_start+20),scenarios))
    actual=set(d[keys].itertuples(index=False,name=None))
    assert actual==expected and len(d)==len(expected),'Incomplete or duplicated evaluation grid'
    dest.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest/path.name)
    shutil.copy2(source/'protocol_frozen.json',dest/'protocol_frozen.json')
    rng=np.random.default_rng(20260922);si=rng.integers(5,size=(5000,5));li=rng.integers(20,size=(5000,20))
    def boot(a):return a[si[:,:,None],li[:,None,:]].mean((1,2))
    def interval(v,alpha=.05):return [float(x) for x in np.quantile(v,[alpha/2,1-alpha/2])]
    summary=[];contrasts=[];seed_rows=[]
    for condition in scenarios:
        part=d[d.scenario==condition]
        for metric in ['success','terminal_success']+(['post_pulse_success'] if condition.startswith('pulse') else []):
            matrices={}
            for task in expected_tasks:
                for method in methods:
                    a=part[(part.task==task)&(part.method==method)].pivot(index='seed',columns='layout',values=metric).loc[SEEDS,list(range(layout_start,layout_start+20))].to_numpy(dtype=float)
                    matrices[(task,method)]=a
            for method in methods:matrices[('Macro average',method)]=np.mean([matrices[t,method] for t in expected_tasks],axis=0)
            for task in list(expected_tasks)+['Macro average']:
                for method in methods:
                    a=matrices[task,method];lo,hi=interval(boot(a))
                    summary.append(dict(task=task,scenario=condition,metric=metric,method=method,mean=a.mean(),
                        seed_sd=a.mean(1).std(ddof=1),low=lo,high=hi,n_policy_seeds=5,layouts_per_task=20,
                        successes=int(a.sum()) if task!='Macro average' else '',rollouts=100 if task!='Macro average' else len(expected_tasks)*100))
                    for seed,v in zip(SEEDS,a.mean(1)):seed_rows.append(dict(task=task,scenario=condition,metric=metric,method=method,seed=seed,mean=v))
                    if method!=ours:
                        diff=matrices[task,ours]-a;b=boot(diff);lo,hi=interval(b);al,ah=interval(b,.05/len(expected_tasks))
                        contrasts.append(dict(task=task,scenario=condition,metric=metric,reference=method,
                            difference=diff.mean(),paired_low=lo,paired_high=hi,task_family_low=al,task_family_high=ah))
    pd.DataFrame(summary).to_csv(dest/'summary.csv',index=False)
    pd.DataFrame(contrasts).to_csv(dest/'paired_contrasts.csv',index=False)
    pd.DataFrame(seed_rows).to_csv(dest/'per_seed.csv',index=False)
    audit=dict(unique_rollouts=len(d),complete_grid=True,duplicate_rows=0,
        bootstrap_replicates=5000,bootstrap_seed=20260922,
        note='Resample policy seeds and layout seeds as paired blocks; macro averages condition on the fixed task set and recovery banks. Per-task family interval alpha=.05/number of tasks. Stress scenarios are descriptive secondary analyses, not all multiplicity-adjusted confirmatory claims.')
    (dest/'statistics_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(name,len(d),'unique rollouts',flush=True)
    return dest

def main():
    act=analyze('act_extension_v1',['transfer_cube','insertion'],
        ['Clean repeat','Gaussian recovery','Frozen context','Calibrated recovery'],
        ['nominal','replan_1','replan_4','replan_16','pulse_early','pulse_late','observation_noise'],94000,'Calibrated recovery')
    meta=analyze('metaworld_extension_v1',['reach-v3','push-v3','pick-place-v3','door-open-v3','drawer-open-v3','button-press-v3'],
        ['Clean repeat','Gaussian recovery','Frozen observations','SCANA-R','Demonstration kNN'],['nominal','pulse'],30000,'SCANA-R')
    if meta:
        costs=pd.concat([pd.read_csv(p) for p in (ROOT/'artifacts/metaworld_extension_v1/recovery').rglob('cost.csv')],ignore_index=True)
        costs.to_csv(meta/'branch_costs.csv',index=False)
        rows=[]
        for (task,mode,amp),d in costs.groupby(['task','mode','amplitude']):
            choice=json.loads((ROOT/'artifacts/metaworld_extension_v1/development'/f'{task}_choice.json').read_text())['amplitudes'][mode]
            if amp!=choice:continue
            rows.append(dict(task=task,method='SCANA-R' if mode=='calibrated' else 'Gaussian recovery',amplitude=amp,
                accepted_branches=len(d),attempts=int(d.attempts.sum()),windows=int(d.windows.sum()),
                zero_scale=int((d.accepted_scale==0).sum()),full_scale=int((d.accepted_scale==1).sum()),
                mean_scale=float(d.accepted_scale.mean()),mean_executed_pulse_rms=float(d.actual_pulse_rms.mean()),
                near_identity_pulses=int((d.actual_pulse_rms<1e-6).sum())))
        pd.DataFrame(rows).to_csv(meta/'collection_costs.csv',index=False)
        summary=pd.read_csv(meta/'summary.csv');print(summary[(summary.scenario=='nominal')&(summary.metric=='success')].pivot(index='task',columns='method',values='mean').to_string(),flush=True)

if __name__=='__main__':main()
