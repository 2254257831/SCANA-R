from pathlib import Path
import sys, json, csv, hashlib, copy, time
ROOT=Path(__file__).resolve().parents[2]
runtime_paths=[ROOT/'tools/_translation_runtime', ROOT/'tools/_public_runtime'] if sys.version_info[:2]==(3,12) else []
for p in runtime_paths+[ROOT/'code', ROOT/'tools']:
    sys.path.insert(0,str(p))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
torch.set_num_threads(4)
from scana_experiments.chunks import load_chunks_npz, chunks_to_arrays, make_chunks, cap_chunks
from scana_experiments.config import load_config
from scana_experiments.models import build_calibrated_pairs, train_noise_generator, sample_generator, load_checkpoint, _chunk_descriptor
from scana_experiments.constraints import estimate_constraints, batch_map
from scana_experiments.metrics import estimate_rbf_gamma, range_violation_rate
from scana_experiments.audited_metrics import direction_metrics, equal_group_weights, weighted_mmd
W=Path(__file__).resolve().parent
OUT=W/'evidence'; OUT.mkdir(exist_ok=True)
SEEDS=[7,11,23,31,47]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def savecsv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def key(c):return c.content_id or f'{c.source_path}|{c.episode_index}'
def split(chunks,seed):
    keys=sorted(set(key(c) for c in chunks));rng=np.random.default_rng(seed)
    hold=set(rng.choice(keys,size=max(1,round(len(keys)*.2)),replace=False))
    return [c for c in chunks if key(c) not in hold],[c for c in chunks if key(c) in hold]
def desc(a,c):return np.concatenate([a.mean(1)/180,a.std(1)/90,c],axis=1)
def knn(train,a,c,seed,k=5):
    r=desc(train.action,train.condition);q=desc(a,c)
    ix=np.argsort(((q[:,None]-r[None])**2).sum(2),axis=1)[:,:k]
    pick=np.random.default_rng(seed).integers(k,size=(4,len(a)))
    return train.target_delta[ix[np.arange(len(a))[None],pick]]
def zero_z(model,scalers,a,c,cfg):
    with torch.no_grad():
        z=torch.zeros((len(a),cfg.latent_dim))
        y=model(torch.tensor(scalers['action'].transform(a.reshape(len(a),-1))),torch.tensor(scalers['condition'].transform(c)),z).numpy()
    return scalers['delta'].inverse(y).reshape(a.shape)
def model_get(cfg,train,val,name,seed,folder,reuse=None):
    cc=copy.deepcopy(cfg);cc.random_seed=seed;cc.latent_dim=0 if name=='Deterministic MLP' else 16
    folder.mkdir(parents=True,exist_ok=True);target=folder/'noise_generator.pt'
    start=time.perf_counter()
    if target.exists() or (reuse is not None and reuse.exists()):
        p=target if target.exists() else reuse
        model,scalers,ck=load_checkpoint(p,map_location='cpu');rep=ck['report']; elapsed=None
    else:
        print(f'Train {folder} {name}',flush=True)
        model,scalers,report=train_noise_generator(train,val,cc,folder,mmd_weight=0.0 if name=='No MMD' else .1)
        rep=report.__dict__;p=target;elapsed=time.perf_counter()-start
    model.eval()
    dump(folder/'resource.json',dict(model=name,seed=seed,seconds=elapsed,parameters=sum(x.numel() for x in model.parameters()),checkpoint=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size,report=rep))
    return model,scalers,cc
def candidates(train,a,c,models,seed):
    rng=np.random.default_rng(seed+50000);n,t,d=a.shape
    iid=rng.standard_normal((4,n,t,d)).astype(np.float32)
    mean,std=train.target_delta.mean(0),train.target_delta.std(0)
    ar=np.empty_like(iid);ar[:,:,0]=iid[:,:,0]
    for i in range(1,t):ar[:,:,i]=.9*ar[:,:,i-1]+np.sqrt(1-.9**2)*iid[:,:,i]
    out={'Clean DT':np.zeros_like(iid),'Clip only':np.zeros_like(iid),
         'Moment Gaussian':mean+iid*std,'AR Gaussian':mean+ar*std,
         'Empirical chunks':train.target_delta[rng.integers(len(train.action),size=(4,n))],
         'Conditional kNN':knn(train,a,c,seed+51000)}
    for name,(m,s,cc) in models.items():
        out[name]=sample_generator(m,s,a,c,cc,copies=4,seed=seed+60000).reshape(4,n,t,d)
    if 'SCANA' in models:
        m,s,cc=models['SCANA'];raw=out['SCANA']
        out['SCANA z=0']=np.repeat(zero_z(m,s,a,c,cc)[None],4,axis=0)
        out['SCANA mean']=np.repeat(raw.mean(0)[None],4,axis=0)
        perm=rng.permutation(n)
        out['Shuffled condition']=sample_generator(m,s,a,c[perm],cc,copies=4,seed=seed+60000).reshape(4,n,t,d)
    return out
def evaluate(samples,test,train,con,gamma,seed,dt,real):
    rows=[];episode_rows=[]
    weights={'query':np.ones(len(test.action))/len(test.action),
             'anchor':equal_group_weights(test.matched_dt_index),
             'episode':equal_group_weights(np.array([key(real[i]) for i in test.real_index]))}
    std=np.maximum(train.target_delta.reshape(len(train.action),-1).std(0),1e-6)
    for name,ss in samples.items():
        for k,delta in enumerate(ss):
            if name=='Clean DT': a=test.action.copy(); final=np.zeros_like(a)
            else:a,final,_=batch_map(test.action,delta,con)
            dr,cos=direction_metrics(test.action,a)
            mse=(((delta-test.target_delta).reshape(len(delta),-1)/std)**2).mean(1)
            rough=np.linalg.norm(np.diff(a,n=2,axis=1),axis=2).mean(1)
            viol=np.mean((test.action+delta<con.action_low)|(test.action+delta>con.action_high),axis=(1,2))
            for group,w in weights.items():
                r=dict(seed=seed,method=name,copy=k,weighting=group,n=len(a),post_mmd2=weighted_mmd(final,test.target_delta,gamma,w),roughness=float(w@rough),mse=float(w@mse),pre_violation=float(w@viol))
                r.update(dr);rows.append(r)
            for ep in sorted(set(key(real[i]) for i in test.real_index)):
                mask=np.array([key(real[i])==ep for i in test.real_index])
                episode_rows.append(dict(seed=seed,method=name,copy=k,episode=ep,n=int(mask.sum()),mse=float(mse[mask].mean()),roughness=float(rough[mask].mean()),pre_violation=float(viol[mask].mean())))
    return rows,episode_rows
def aggregate(rows,groups):
    import pandas as pd
    df=pd.DataFrame(rows)
    metrics=[c for c in df if c not in groups+['copy','seed'] and pd.api.types.is_numeric_dtype(df[c])]
    first=df.groupby(groups+['seed'])[metrics].mean().reset_index()
    ag=first.groupby(groups)[metrics].agg(['mean','std']).reset_index()
    ag.columns=['_'.join(filter(None,map(str,c))) for c in ag.columns.to_flat_index()]
    return ag.to_dict('records')
