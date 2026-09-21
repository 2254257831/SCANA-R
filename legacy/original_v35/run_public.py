from common import *
import scana_public_lerobot_reproduce_20260714 as pub
from scana_public_policy_physics_closure_20260714 import ChunkPolicy, Standardizer, features, fit_transition_model

def prepare(task):
    dc,hc=pub.TASKS[task]
    de=pub.load_repository(dc,'digital_twin',task);he=pub.load_repository(hc,'real',task)
    d0,dt=pub.split_episodes(de,2026);h0,ht=pub.split_episodes(he,2027)
    dr,dv=pub.split_episodes(d0,91001);hr,hv=pub.split_episodes(h0,91002)
    center,scale=pub.fit_action_normalizer(dr+hr)
    cfg=pub.cfg_for(OUT,7)
    groups=[]
    for i,eps in enumerate([dr,dv,dt,hr,hv,ht]):
        cs=make_chunks(pub.normalize_episodes(eps,center,scale),cfg)
        groups.append(cap_chunks(cs,1600 if i in [0,3] else 500,2026+i))
    ids=[set(e.content_id for e in eps) for eps in [dr,dv,dt,hr,hv,ht]]
    assert all(not ids[i]&ids[j] for i in range(6) for j in range(i))
    return cfg,groups,center,scale

def train_policy(x,y,evx,seed,path,xs,ys):
    torch.manual_seed(seed);m=ChunkPolicy(x.shape[1],y.shape[1]);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    xx=torch.tensor(xs.transform(x));yy=torch.tensor(ys.transform(y));rng=np.random.default_rng(seed+70000)
    steps=300
    for _ in range(steps):
        idx=rng.integers(len(x),size=128);loss=((m(xx[idx])-yy[idx])**2).mean()
        opt.zero_grad();loss.backward();opt.step()
    m.eval()
    with torch.no_grad():p=ys.inverse(m(torch.tensor(xs.transform(evx))).numpy())
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=m.state_dict(),input_mean=xs.mean,input_std=xs.std,output_mean=ys.mean,output_std=ys.std,seed=seed,steps=steps,input_contract='current robot qpos only, 14 dimensions'),path)
    return p

def main():
    allrows=[];policyrows=[];episode_rows=[];trans=[];protocol={}
    for task in pub.TASKS:
        cfg,g,center,scale=prepare(task);dr,dv,dt,hr,hv,ht=g;slug=task.lower().replace(' ','_')
        train=build_calibrated_pairs(dr,hr,cfg);val=build_calibrated_pairs(dv,hv,cfg);test=build_calibrated_pairs(dt,ht,cfg)
        con=estimate_constraints(dr+hr,cfg,train.target_delta);gamma=estimate_rbf_gamma(train.target_delta,seed=91003)
        da,ha,ea=[chunks_to_arrays(c) for c in [dr,hr,ht]]
        # Fit all normalization only on the 32+32 training episodes.
        xs=Standardizer(np.concatenate([features(da),features(ha)]));ys=Standardizer(np.concatenate([da['action'],ha['action']]).reshape(len(dr)+len(hr),-1))
        action_gamma=estimate_rbf_gamma(np.concatenate([da['action'],ha['action']]),seed=91003)
        protocol[task]=dict(seeds=SEEDS,episodes=[len(set(map(key,c))) for c in g],windows=list(map(len,g)),pairs=[len(train.action),len(val.action),len(test.action)],gamma=gamma,action_gamma=action_gamma,center=center.tolist(),scale=scale.tolist(),identities=[sorted(set(map(key,c))) for c in g],config=cfg.__dict__,prediction_input='q_t only; excludes future states, actions, trajectory conditions and episode length',prediction_output='16x14 action chunk',predictor='256-LN-SiLU,256-LN-SiLU,224; AdamW lr.001,wd.0001;300 steps;batch128',budget='4096 sampled rows with replacement; mixed groups exactly 2048 human +2048 scripted; same draw indices, init and minibatches across methods')
        dump(OUT/'public_protocol.json',protocol)
        np.savez_compressed(OUT/f'public_data_{slug}.npz',**{p+'_'+k:v for p,a in [('dt',da),('human',ha),('eval',ea)] for k,v in a.items() if v.dtype.kind not in ['O']},center=center,scale=scale)
        # Independent transition model calibration: unseen DT and human episodes.
        tw=fit_transition_model(np.concatenate([da['state'],ha['state']]),np.concatenate([da['action'],ha['action']]))
        for domain,cs in [('scripted',dt),('human',ht)]:
            ar=chunks_to_arrays(cs);xx=np.concatenate([ar['state'][:,:-1],ar['action'][:,:-1],np.ones((*ar['state'][:,:-1].shape[:2],1))],2)
            err=(xx@tw-ar['state'][:,1:])**2
            trans.append(dict(task=task,domain=domain,heldout_episodes=len(set(map(key,cs))),rmse=float(np.sqrt(err.mean())),scope='held-out prediction of unchanged recorded transitions; no counterfactual validity'))
        for seed in SEEDS:
            models={name:model_get(cfg,train,val,name,seed,OUT/'public_models'/slug/str(seed)/folder) for name,folder in [('SCANA','scana'),('Deterministic MLP','deterministic')]}
            samples=candidates(train,test.action,test.condition,models,seed)
            rr,ee=evaluate(samples,test,train,con,gamma,seed,dt,ht)
            allrows += [dict(task=task,**r) for r in rr];episode_rows +=[dict(task=task,**r) for r in ee]
            ss=candidates(train,da['action'],da['condition'],models,seed)
            labels={'Mixed clean':np.repeat(da['action'][None],4,axis=0)}
            for name in ['Clip only','Empirical chunks','Conditional kNN','Deterministic MLP','SCANA']:
                labels[name]=np.stack([batch_map(da['action'],d,con)[0] for d in ss[name]])
            labels['SCANA K=1']=np.repeat(labels['SCANA'][:1],4,axis=0)
            labels['SCANA z=0']=np.stack([batch_map(da['action'],d,con)[0] for d in ss['SCANA z=0']])
            rng=np.random.default_rng(seed+80000)
            di=rng.integers(len(dr),size=2048);hi=rng.integers(len(hr),size=2048);ci=rng.integers(4,size=2048)
            np.savez_compressed(OUT/'public_models'/slug/str(seed)/'draws.npz',dt=di,human=hi,copy=ci)
            for name,la in labels.items():
                xx=np.concatenate([features(da)[di],features(ha)[hi]])
                yy=np.concatenate([la[ci,di],ha['action'][hi]]).reshape(4096,-1)
                checkpoint=OUT/'public_policy'/slug/str(seed)/(name.replace(' ','_').replace('=','')+'.pt')
                pr=train_policy(xx,yy,features(ea),seed+90000,checkpoint,xs,ys).reshape(ea['action'].shape)
                sq=(pr-ea['action'])**2
                epvals=[float(np.mean(sq[ea['episode_index']==ep])) for ep in np.unique(ea['episode_index'])]
                policyrows.append(dict(task=task,seed=seed,method=name,rmse=float(np.sqrt(sq.mean())),episode_equal_rmse=float(np.sqrt(np.mean(epvals))),train_sampled_rows=4096,unique_dt=int(len(np.unique(di))),unique_human=int(len(np.unique(hi))),heldout_episodes=len(epvals),mmd2=weighted_mmd(pr,ea['action'],action_gamma),model_sha256=sha(checkpoint)))
                print(f'Public {task} seed {seed}: {name} RMSE {np.sqrt(sq.mean()):.4f}',flush=True)
            savecsv(OUT/'public_per_copy.csv',allrows);savecsv(OUT/'public_summary.csv',aggregate(allrows,['task','method','weighting']))
            savecsv(OUT/'public_policy_per_seed.csv',policyrows);savecsv(OUT/'public_policy_summary.csv',aggregate(policyrows,['task','method']))
            savecsv(OUT/'public_per_episode.csv',episode_rows);savecsv(OUT/'public_transition_heldout.csv',trans)
    dump(OUT/'public_complete.json',dict(complete=True,seeds=SEEDS))
if __name__=='__main__':main()
