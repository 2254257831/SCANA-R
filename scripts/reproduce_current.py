"""Recompute the 11,040 current-study evaluations without simulation archives.

Preserves the frozen analyses' grids, RNG seeds and hierarchical resampling.
CPU times are recorded measurements, not timing measurements of this script.
"""
from pathlib import Path
import argparse, ast, json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/independent_libraries_v46'

def validate_grid(frame, dimensions, outcome='success'):
    keys=list(dimensions)
    expected=pd.MultiIndex.from_product(dimensions.values(),names=keys)
    actual=pd.MultiIndex.from_frame(frame[keys])
    if actual.has_duplicates or len(actual)!=len(expected) or len(expected.difference(actual)) or len(actual.difference(expected)):
        raise ValueError('Incomplete, duplicate or unexpected experimental grid: '+', '.join(keys))
    if outcome and not frame[outcome].isin([0,1]).all():
        raise ValueError('Success must be binary and nonmissing')

def array(frame,task,method,p):
    idx=pd.MultiIndex.from_product([p['banks'],p['policy_seeds'],p['test_layouts']],names=['bank','seed','layout'])
    sub=frame[(frame.task==task)&(frame.method==method)].set_index(['bank','seed','layout'])
    return sub.reindex(idx).success.to_numpy().reshape(len(p['banks']),len(p['policy_seeds']),len(p['test_layouts']))

def run(output):
    output.mkdir(parents=True,exist_ok=True)
    generated=[]
    def save(name,rows):
        pd.DataFrame(rows).to_csv(output/name,index=False);generated.append(name)
    p=json.loads((DATA/'protocols/act_protocol_frozen.json').read_text())
    a=pd.read_csv(DATA/'per_episode/act_library.csv')
    dims=dict(task=p['tasks'],bank=p['banks'],seed=p['policy_seeds'],method=p['methods'],layout=p['test_layouts'])
    validate_grid(a,dims)
    rng=np.random.default_rng(2026092901)
    bb=rng.integers(5,size=(10000,5));ss=rng.integers(3,size=(10000,5,3));ll=rng.integers(20,size=(10000,20))
    def boot(z):return z[bb[:,:,None,None],ss[:,:,:,None],ll[:,None,None,:]].mean((1,2,3))
    summary=[];contrasts=[];banks=[]
    for task in p['tasks']:
        arrays={m:array(a,task,m,p) for m in p['methods']}
        for method,z in arrays.items():
            lo,hi=np.quantile(boot(z),[.025,.975])
            summary.append(dict(task=task,method=method,successes=int(z.sum()),episodes=z.size,mean=z.mean(),low=lo,high=hi,bank_sd=z.mean((1,2)).std(ddof=1)))
            banks.extend(dict(task=task,bank=bank,method=method,success=z[j].mean()) for j,bank in enumerate(p['banks']))
        for ref in [m for m in p['methods'] if m!='SCANA-R']:
            diff=arrays['SCANA-R']-arrays[ref];v=boot(diff);lo,hi=np.quantile(v,[.025,.975]);fl,fh=np.quantile(v,[.0125,.9875])
            contrasts.append(dict(task=task,method='SCANA-R',reference=ref,difference=diff.mean(),low=lo,high=hi,family_low=fl,family_high=fh,positive_banks=int(sum(diff.mean((1,2))>0)),bank_differences=diff.mean((1,2)).tolist()))
    save('replication_summary.csv',summary);save('replication_contrasts.csv',contrasts);save('bank_success.csv',banks)
    costs=pd.read_csv(DATA/'analysis/cost_per_run.csv')
    validate_grid(costs,{k:v for k,v in dims.items() if k!='layout'},outcome=None)
    np.testing.assert_allclose(costs.total_cpu,costs[['auxiliary_cpu','collection_cpu','final_training_cpu']].sum(axis=1),rtol=1e-12)
    save('cost_summary.csv',costs.groupby(['task','method']).mean(numeric_only=True).reset_index())
    diag=p['diagnostics'];f=pd.read_csv(DATA/'per_episode/failure_interventions.csv')
    validate_grid(f,dict(task=p['tasks'],method=diag['frozen_methods'],seed=diag['seeds'],condition=diag['conditions'],layout=diag['layouts']))
    metrics=['success','terminal_success','action_step_rms','prediction_noise_rms']+[f'reach_reward_{s}' for s in range(1,5)]
    save('failure_summary.csv',f.groupby(['task','method','condition'])[metrics].mean().reset_index())
    rng=np.random.default_rng(2026092902);ss2=rng.integers(5,size=(5000,5));ll2=rng.integers(20,size=(5000,20));rows=[]
    for (task,method),group in f.groupby(['task','method']):
        mats={c:g.pivot(index='seed',columns='layout',values='success').loc[diag['seeds'],diag['layouts']].to_numpy() for c,g in group.groupby('condition')}
        for treatment,control in [('noise_supported','observation_noise'),('noise_joints','nominal'),('noise_objects','nominal'),('noise_low_support','nominal'),('replan_1_ensemble','replan_1'),('replan_1','nominal')]:
            diff=mats[treatment]-mats[control];v=diff[ss2[:,:,None],ll2[:,None,:]].mean((1,2));lo,hi=np.quantile(v,[.025,.975])
            rows.append(dict(task=task,method=method,treatment=treatment,control=control,difference=diff.mean(),low=lo,high=hi))
    save('failure_contrasts.csv',rows)
    p=json.loads((DATA/'protocols/metaworld_protocol_frozen.json').read_text());a=pd.read_csv(DATA/'per_episode/metaworld_library.csv')
    validate_grid(a,dict(task=p['tasks'],bank=p['banks'],seed=p['policy_seeds'],method=p['methods'],layout=p['test_layouts']))
    rng=np.random.default_rng(2026092903);arrays={};boots={};summary=[];contrasts=[];banks=[]
    for task in p['tasks']:
        bb=rng.integers(3,size=(10000,3));ss=rng.integers(3,size=(10000,3,3));ll=rng.integers(20,size=(10000,20))
        for method in p['methods']:
            z=array(a,task,method,p);arrays[task,method]=z;v=boot(z);boots[task,method]=v;lo,hi=np.quantile(v,[.025,.975])
            summary.append(dict(task=task,method=method,successes=int(z.sum()),episodes=z.size,mean=z.mean(),low=lo,high=hi,bank_sd=z.mean((1,2)).std(ddof=1)))
            banks.extend(dict(task=task,bank=bank,method=method,success=z[j].mean()) for j,bank in enumerate(p['banks']))
        for ref in ['Clean repeat','Gaussian recovery']:
            diff=arrays[task,'SCANA-R']-arrays[task,ref];v=boots[task,'SCANA-R']-boots[task,ref]
            lo,hi=np.quantile(v,[.025,.975]);fl,fh=np.quantile(v,[.05/12,1-.05/12])
            contrasts.append(dict(task=task,reference=ref,difference=diff.mean(),low=lo,high=hi,family_low=fl,family_high=fh))
    for method in p['methods']:
        z=np.stack([arrays[t,method] for t in p['tasks']]);v=np.mean([boots[t,method] for t in p['tasks']],axis=0);lo,hi=np.quantile(v,[.025,.975])
        summary.append(dict(task='Macro-average',method=method,successes=int(z.sum()),episodes=z.size,mean=z.mean(),low=lo,high=hi,bank_sd=np.nan))
    for ref in ['Clean repeat','Gaussian recovery']:
        point=np.mean([arrays[t,'SCANA-R'].mean()-arrays[t,ref].mean() for t in p['tasks']]);v=np.mean([boots[t,'SCANA-R']-boots[t,ref] for t in p['tasks']],axis=0)
        lo,hi=np.quantile(v,[.025,.975]);fl,fh=np.quantile(v,[.0125,.9875]);contrasts.append(dict(task='Macro-average',reference=ref,difference=point,low=lo,high=hi,family_low=fl,family_high=fh))
    save('metaworld_replication_summary.csv',summary);save('metaworld_replication_contrasts.csv',contrasts);save('metaworld_bank_success.csv',banks)
    for name in generated:
        expected=pd.read_csv(DATA/'analysis'/name);actual=pd.read_csv(output/name)
        if 'bank_differences' in expected:
            for x,y in zip(expected.pop('bank_differences'),actual.pop('bank_differences')):np.testing.assert_allclose(ast.literal_eval(x),ast.literal_eval(y),atol=1e-12)
        pd.testing.assert_frame_equal(expected,actual,rtol=1e-12,atol=1e-12)
    report=dict(evaluations=11040,act=4200,metaworld=3240,interventions=3600,matched_statistics=generated,all_matched=True,source_demonstrations_fixed=True,cost_inputs='Recorded cost_per_run.csv; raw per-fit metadata in the separate evidence archives.')
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'outputs/recomputed-current');run(ap.parse_args().output)
