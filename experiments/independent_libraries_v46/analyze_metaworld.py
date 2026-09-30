from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
W=Path(os.environ.get('SCANA_V46_OUTPUT',Path(__file__).resolve().parents[2]/'artifacts/independent_libraries_v46')).resolve();D=W/'metaworld_evidence';O=W/'analysis'
def main():
    p=json.loads((D/'protocol_frozen.json').read_text());a=pd.read_csv(D/'per_episode.csv')
    assert len(a)==3240 and not a.duplicated(['task','bank','seed','method','layout']).any()
    rng=np.random.default_rng(2026092903);N=10000;tasks=p['tasks'];methods=p['methods'];summ=[];contr=[];arrays={}
    boot={};bankrows=[]
    for task in tasks:
        bb=rng.integers(3,size=(N,3));ss=rng.integers(3,size=(N,3,3));ll=rng.integers(20,size=(N,20))
        idx=pd.MultiIndex.from_product([p['banks'],p['policy_seeds'],p['test_layouts']],names=['bank','seed','layout'])
        for method in methods:
            sub=a[(a.task==task)&(a.method==method)].set_index(['bank','seed','layout']).reindex(idx)
            z=sub.success.to_numpy().reshape(3,3,20);assert np.isfinite(z).all();arrays[task,method]=z
            v=z[bb[:,:,None,None],ss[:,:,:,None],ll[:,None,None,:]].mean((1,2,3));boot[task,method]=v
            lo,hi=np.quantile(v,[.025,.975]);summ.append(dict(task=task,method=method,successes=int(z.sum()),episodes=z.size,mean=z.mean(),low=lo,high=hi,bank_sd=z.mean((1,2)).std(ddof=1)))
            bankrows.extend(dict(task=task,bank=b,method=method,success=z[j].mean()) for j,b in enumerate(p['banks']))
        for ref in ['Clean repeat','Gaussian recovery']:
            diff=arrays[task,'SCANA-R']-arrays[task,ref];v=boot[task,'SCANA-R']-boot[task,ref]
            lo,hi=np.quantile(v,[.025,.975]);fl,fh=np.quantile(v,[.05/12,1-.05/12])
            contr.append(dict(task=task,reference=ref,difference=diff.mean(),low=lo,high=hi,family_low=fl,family_high=fh))
    for method in methods:
        z=np.stack([arrays[t,method] for t in tasks]);v=np.stack([boot[t,method] for t in tasks]).mean(0);lo,hi=np.quantile(v,[.025,.975])
        summ.append(dict(task='Macro-average',method=method,successes=int(z.sum()),episodes=z.size,mean=z.mean(),low=lo,high=hi,bank_sd=np.nan))
    for ref in ['Clean repeat','Gaussian recovery']:
        point=np.mean([arrays[t,'SCANA-R'].mean()-arrays[t,ref].mean() for t in tasks]);v=np.mean([boot[t,'SCANA-R']-boot[t,ref] for t in tasks],axis=0)
        lo,hi=np.quantile(v,[.025,.975]);fl,fh=np.quantile(v,[.0125,.9875])
        contr.append(dict(task='Macro-average',reference=ref,difference=point,low=lo,high=hi,family_low=fl,family_high=fh))
    pd.DataFrame(summ).to_csv(O/'metaworld_replication_summary.csv',index=False);pd.DataFrame(contr).to_csv(O/'metaworld_replication_contrasts.csv',index=False)
    pd.DataFrame(bankrows).to_csv(O/'metaworld_bank_success.csv',index=False)
    (O/'metaworld_statistics_audit.json').write_text(json.dumps(dict(episodes=len(a),libraries=18,auxiliary_models=90,final_policies=162,bootstrap=N,
      sampling='Task-specific library and within-library seed resampling, common layouts within each task; macro mean over six fixed tasks.',
      primary_macro_comparisons=2,macro_family_interval=97.5,taskwise_family_interval=99.1666667,source_demonstrations_fixed=True),indent=2))
    print(pd.DataFrame(summ).to_string(index=False));print(pd.DataFrame(contr).to_string(index=False))
if __name__=='__main__':main()
