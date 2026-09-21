from sim_bridge import *
from run_public import prepare,ChunkPolicy,Standardizer
from scana_experiments.chunks import chunk_condition
import importlib.metadata
SIM=W/'simulation';DATA=SIM/'demonstrations';DATA.mkdir(exist_ok=True)
PROTOCOL=dict(tasks=['transfer_cube','insertion'],model_seeds=SEEDS,train_success_episodes=40,selection_success_episodes=10,
    demonstration_seeds='61000 onward transfer;71000 onward insertion; keep successful replay only; record all collection outcomes',
    test_layout_seeds=list(range(81000,81020)),horizon=400,Hz=50,chunk=16,stride=8,execute=8,
    methods=['Clean repeat','Clip only','Empirical chunks','Conditional kNN','Deterministic MLP','SCANA','SCANA K=1'],
    budget='4096 sampled rows, 50% original and 50% method labels; common base indices, initialization, minibatch and copy indices; K=1/K=4 have equal optimization budget',
    input='current 14 robot positions, current simulator object xyz positions, elapsed step/400; privileged-state policy; no images, language, future state, action-derived condition or future episode length',
    output='16x14 normalized absolute joint-position action chunk; execute first 8 then reobserve',
    success='official ACT max episode reward ==4; all test layouts included; no selection by test success',
    policy='MLP 256-LN-SiLU,256-LN-SiLU,224; AdamW lr0.001 wd0.0001;1200 steps,batch128; checkpoint selected every100 steps by clean selection-episode MSE',
    primary_comparison='SCANA versus Clean repeat, kNN and deterministic MLP. Report all methods; no alpha or hyperparameter search using test outcomes.',
    statistic='5 policy seeds crossed with 20 shared layouts per task; report seed spread and two-way seed/layout bootstrap; 100 rollouts are not 100 independent policy trainings',
    interpretation='new simulation pilot, not GR00T or real robot; no prospective power claim',
    mujoco=importlib.metadata.version('mujoco'),dm_control=importlib.metadata.version('dm-control'),torch=torch.__version__,numpy=np.__version__)
dump(SIM/'protocol_frozen.json',PROTOCOL)

def dataset(task):
    records=[];audit=[];offset=61000 if task=='transfer_cube' else 71000
    folder=DATA/task;folder.mkdir(exist_ok=True)
    for seed in range(offset,offset+1000):
        p=folder/f'{seed}.npz'
        if p.exists():
            with np.load(p) as f:r={k:f[k] for k in f}
        else:r=collect(task,seed);np.savez_compressed(p,**r)
        audit.append(dict(seed=seed,success=int(r['success']),max_reward=float(r['rewards'].max()),file=p.name,sha256=sha(p)))
        if int(r['success']):records.append(r)
        if len(records)==50:break
    if len(records)<50:raise RuntimeError('Could not collect required successful demonstrations')
    dump(folder/'collection_manifest.json',audit)
    print(task,'collected',len(records),'successes out of',len(audit),flush=True)
    return records[:40],records[40:]

def windows(records,center,scale):
    x=[];a=[];c=[];ep=[]
    for r in records:
        action=50*(r['action']-center)/scale;state=50*(r['state']-center)/scale
        for start in range(0,385,8):
            x.append(r['input'][start]);a.append(action[start:start+16]);c.append(chunk_condition(action[start:start+16],state[start:start+16],start,400));ep.append(int(r['seed']))
    return np.asarray(x),np.asarray(a),np.asarray(c),np.asarray(ep)

def fit(x,y,vx,vy,xs,ys,seed,path):
    torch.manual_seed(seed);m=ChunkPolicy(x.shape[1],224);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    tx=torch.tensor(xs.transform(x));ty=torch.tensor(ys.transform(y));vv=torch.tensor(xs.transform(vx));vt=torch.tensor(ys.transform(vy))
    rng=np.random.default_rng(seed+70000);best=float('inf');bst=None;trace=[];t0=time.perf_counter()
    for step in range(1200):
        idx=rng.integers(len(tx),size=128);loss=((m(tx[idx])-ty[idx])**2).mean();opt.zero_grad();loss.backward();opt.step()
        if (step+1)%100==0:
            m.eval()
            with torch.no_grad():err=float(((m(vv)-vt)**2).mean())
            trace.append(dict(step=step+1,selection_mse=err))
            if err<best:best=err;bst={k:v.detach().clone() for k,v in m.state_dict().items()};beststep=step+1
            m.train()
    m.load_state_dict(bst);m.eval();path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=bst,input_mean=xs.mean,input_std=xs.std,output_mean=ys.mean,output_std=ys.std,seed=seed,best_step=beststep,selection_mse=best,trace=trace),path)
    dump(path.with_suffix('.json'),dict(sha256=sha(path),seconds=time.perf_counter()-t0,parameters=sum(p.numel() for p in m.parameters()),selection_mse=best,best_step=beststep,trace=trace))
    return m

def rollout(task,layout,m,xs,ys,center,scale):
    sim_env.BOX_POSE[0]=initial_pose(task,layout);env=sim_env.make_sim_env('sim_'+task);ts=env.reset();rewards=[];actions=[];states=[];inputs=[];q=[]
    for step in range(400):
        if step%8==0:
            obs=causal_observation(ts.observation,step);inputs.append(obs)
            with torch.no_grad():chunk=ys.inverse(m(torch.tensor(xs.transform(obs[None]))).numpy()).reshape(16,14)
            q=(chunk/50*scale+center)[:8]
        act=q[step%8];states.append(ts.observation['qpos'].copy());actions.append(act)
        ts=env.step(act);rewards.append(float(ts.reward))
    env.close()
    return dict(success=int(max(rewards)==4),max_reward=max(rewards),total_reward=sum(rewards),action=np.asarray(actions),state=np.asarray(states),reward=np.asarray(rewards),input=np.asarray(inputs))

def main():
    rows=[]
    for task in PROTOCOL['tasks']:
        pubtask='Transfer Cube' if task=='transfer_cube' else 'Insertion'
        cfg,g,center,scale=prepare(pubtask);dr,dv,dt,hr,hv,ht=g
        train=build_calibrated_pairs(dr,hr,cfg);val=build_calibrated_pairs(dv,hv,cfg);con=estimate_constraints(dr+hr,cfg,train.target_delta)
        tr,va=dataset(task);x,a,c,ep=windows(tr,center,scale);vx,va0,_,_=windows(va,center,scale)
        xs,ys=Standardizer(x),Standardizer(a.reshape(len(a),-1));vy=va0.reshape(len(va0),-1)
        np.savez_compressed(SIM/f'training_{task}.npz',input=x,action=a,condition=c,episode=ep,center=center,scale=scale)
        for seed in SEEDS:
            models={name:model_get(cfg,train,val,name,seed,OUT/'public_models'/task/str(seed)/slug) for name,slug in [('SCANA','scana'),('Deterministic MLP','deterministic')]}
            ss=candidates(train,a,c,models,seed)
            labs={'Clean repeat':np.repeat(a[None],4,axis=0)}
            for name in ['Clip only','Empirical chunks','Conditional kNN','Deterministic MLP','SCANA']:
                labs[name]=np.stack([batch_map(a,d,con)[0] for d in ss[name]])
            labs['SCANA K=1']=np.repeat(labs['SCANA'][:1],4,axis=0)
            rng=np.random.default_rng(seed+80000);baseidx=rng.integers(len(x),size=2048);augidx=rng.integers(len(x),size=2048);ci=rng.integers(4,size=2048)
            folder=SIM/'policies'/task/str(seed);folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/'draws.npz',base=baseidx,augmented=augidx,copy=ci)
            for name,labels in labs.items():
                path=folder/(name.replace(' ','_').replace('=','')+'.pt')
                if path.exists():
                    ck=torch.load(path,map_location='cpu',weights_only=False);m=ChunkPolicy(x.shape[1],224);m.load_state_dict(ck['model']);m.eval()
                else:
                    xx=np.concatenate([x[baseidx],x[augidx]]);yy=np.concatenate([a[baseidx],labels[ci,augidx]]).reshape(4096,-1)
                    m=fit(xx,yy,vx,vy,xs,ys,seed+90000,path)
                successes=[]
                for layout in PROTOCOL['test_layout_seeds']:
                    record=SIM/'rollouts'/task/str(seed)/path.stem/f'{layout}.npz';record.parent.mkdir(parents=True,exist_ok=True)
                    if record.exists():
                        with np.load(record) as f:r={k:f[k] for k in f}
                    else:r=rollout(task,layout,m,xs,ys,center,scale);np.savez_compressed(record,**r)
                    row=dict(task=task,seed=seed,method=name,layout=layout,success=int(r['success']),max_reward=float(r['max_reward']),total_reward=float(r['total_reward']),checkpoint_sha256=sha(path))
                    rows.append(row);successes.append(row['success'])
                savecsv(SIM/'closed_loop_per_episode.csv',rows)
                print(task,seed,name,'success',sum(successes),'/20',flush=True)
    dump(SIM/'completion.json',dict(complete=True,rollouts=len(rows),tasks=2,seeds=SEEDS))
if __name__=='__main__':main()
