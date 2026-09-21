"""Recompute crossed seed/layout bootstrap statistics without simulation."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ORDER=['Clean repeat','Calibrated recovery','Gaussian recovery','Frozen context','Old SCANA','Old kNN','Old deterministic','Clip only']
SEEDS=[7,11,23,31,47];LAYOUTS=list(range(92000,92040))

def summarize(source,output):
    df=pd.read_csv(source)
    if df.duplicated(['task','seed','method','layout']).any():raise ValueError('Duplicate evaluation records')
    if not set(df.success).issubset({0,1}):raise ValueError('Success must be binary')
    if set(df.task)!={'transfer_cube','insertion'}:raise ValueError('Expected two tasks')
    methods=[m for m in ORDER if m in set(df.method)]
    if set(df.method)!=set(methods) or not set(ORDER[:4]).issubset(methods):raise ValueError('Missing current methods or unknown method')
    rng=np.random.default_rng(20260919);si=rng.integers(5,size=(5000,5));li=rng.integers(40,size=(5000,40))
    summaries=[];contrasts=[]
    for task in ['transfer_cube','insertion']:
        arrays={}
        for method in methods:
            part=df[(df.task==task)&(df.method==method)]
            if len(part)!=200 or set(part.seed)!=set(SEEDS) or set(part.layout)!=set(LAYOUTS):raise ValueError('Incomplete crossed evaluation grid')
            a=part.pivot(index='seed',columns='layout',values='success').loc[SEEDS,LAYOUTS].to_numpy();arrays[method]=a
            b=a[si[:,:,None],li[:,None,:]].mean((1,2));lo,hi=np.quantile(b,[.025,.975])
            summaries.append(dict(task=task,method=method,successes=int(a.sum()),rollouts=a.size,mean=float(a.mean()),seed_sd=float(a.mean(1).std(ddof=1)),bootstrap_low=float(lo),bootstrap_high=float(hi)))
        for ref in methods:
            if ref=='Calibrated recovery':continue
            a=arrays['Calibrated recovery']-arrays[ref];b=a[si[:,:,None],li[:,None,:]].mean((1,2));lo,hi=np.quantile(b,[.025,.975]);al,ah=np.quantile(b,[.0125,.9875])
            contrasts.append(dict(task=task,method='Calibrated recovery',reference=ref,difference=float(a.mean()),paired_low=float(lo),paired_high=float(hi),family_adjusted_low=float(al),family_adjusted_high=float(ah)))
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(summaries).to_csv(output/'summary.csv',index=False)
    pd.DataFrame(contrasts).to_csv(output/'paired_contrasts.csv',index=False)
    (output/'scope.json').write_text(json.dumps(dict(rows=len(df),methods=methods,bootstrap_replicates=5000,resampling='paired two-way seed and layout',scope='Reanalysis of existing trials, not new experiments'),indent=2),encoding='utf-8')
    print(pd.DataFrame(summaries).to_string(index=False))
    return summaries,contrasts

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=ROOT/'results/independent_test/per_episode.csv')
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/recomputed')
    a=ap.parse_args();summarize(a.input,a.output)
