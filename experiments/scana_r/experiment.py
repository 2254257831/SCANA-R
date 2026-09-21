"""Development experiments. All rollouts here use selection layouts, never test.

Portable release of the frozen experiment. See README.md for pinned dependencies.
"""
from pathlib import Path
import sys, argparse, json, time, hashlib
from concurrent.futures import ProcessPoolExecutor
from sim_bridge import *
from policy_network import ChunkPolicy, Standardizer
JOINTS=np.array([0,1,2,3,4,5,7,8,9,10,11,12])
GRIP=np.array([6,13])

def records(task):
    rows=[]
    for p in sorted((OLD/'demonstrations'/task).glob('*.npz')):
        with np.load(p) as f:r={k:f[k] for k in f}
        if int(r['success']):rows.append(r)
    assert len(rows)==50
    return rows[:40],rows[40:]

def windows(rs):
    x=[];y=[];ep=[];tt=[]
    for r in rs:
        for t in range(0,385,8):
            x.append(r['input'][t]);y.append(r['action'][t:t+16].reshape(-1));ep.append(int(r['seed']));tt.append(t)
    return np.asarray(x,dtype=np.float32),np.asarray(y,dtype=np.float32),np.asarray(ep),np.asarray(tt)

def predict(m,ck,x):
    with torch.no_grad():p=m(torch.tensor((np.asarray(x,dtype=np.float32)-ck['input_mean'])/ck['input_std'])).numpy()
    return p*ck['output_std']+ck['output_mean']

def load_policy(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);m=ChunkPolicy(len(ck['input_mean']),224)
    m.load_state_dict(ck['model']);m.eval()
    return m,ck

def fit(x,y,vx,vy,xs,ys,seed,path,steps=1200):
    if path.exists():return load_policy(path)
    torch.manual_seed(seed+90000);m=ChunkPolicy(x.shape[1],224);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    tx=torch.tensor(xs.transform(x));ty=torch.tensor(ys.transform(y));vv=torch.tensor(xs.transform(vx));vt=torch.tensor(ys.transform(vy))
    rng=np.random.default_rng(seed+160000);best=float('inf');bst=None;trace=[];t0=time.perf_counter()
    for step in range(steps):
        idx=rng.integers(len(tx),size=128);loss=((m(tx[idx])-ty[idx])**2).mean();opt.zero_grad();loss.backward();opt.step()
        if (step+1)%100==0:
            m.eval()
            with torch.no_grad():err=float(((m(vv)-vt)**2).mean())
            trace.append(dict(step=step+1,selection_mse=err))
            if err<best:best=err;bst={k:v.detach().clone() for k,v in m.state_dict().items()};beststep=step+1
            m.train()
    ck=dict(model=bst,input_mean=xs.mean,input_std=xs.std,output_mean=ys.mean,output_std=ys.std,seed=seed,best_step=beststep,selection_mse=best,trace=trace,seconds=time.perf_counter()-t0,output='raw absolute joint positions; grippers normalized')
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(ck,path);m.load_state_dict(bst);m.eval()
    dump(path.with_suffix('.json'),{k:v for k,v in ck.items() if k not in ['model','input_mean','input_std','output_mean','output_std']})
    return m,ck

def replay(task,r,action):
    sim_env.BOX_POSE[0]=r['pose'];env=sim_env.make_sim_env('sim_'+task);ts=env.reset();xs=[];ss=[];rr=[]
    for t,a in enumerate(action):
        xs.append(causal_observation(ts.observation,t));ss.append(ts.observation['qpos'].copy())
        ts=env.step(a);rr.append(float(ts.reward))
    env.close()
    return dict(action=np.asarray(action,dtype=np.float32),input=np.asarray(xs),state=np.asarray(ss),rewards=np.asarray(rr),pose=r['pose'],seed=r['seed'],success=int(max(rr)==4))

def rollout(task,layout,m,ck):
    sim_env.BOX_POSE[0]=initial_pose(task,layout);env=sim_env.make_sim_env('sim_'+task);ts=env.reset();rewards=[];actions=[];states=[];inputs=[]
    for step in range(400):
        if step%8==0:
            obs=causal_observation(ts.observation,step);inputs.append(obs);q=predict(m,ck,obs[None]).reshape(16,14)[:8]
        act=q[step%8];states.append(ts.observation['qpos'].copy());actions.append(act);ts=env.step(act);rewards.append(float(ts.reward))
    env.close()
    return dict(success=int(max(rewards)==4),max_reward=max(rewards),total_reward=sum(rewards),action=np.asarray(actions),state=np.asarray(states),reward=np.asarray(rewards),input=np.asarray(inputs))

def crossfit(task):
    path=WORK/'calibration'/f'{task}.npz'
    if path.exists():return dict(np.load(path))
    tr,va=records(task);x,y,ep,tt=windows(tr);err=np.zeros_like(y)
    for fold in range(5):
        hold=np.isin(ep,[int(r['seed']) for r in tr[fold*8:(fold+1)*8]])
        # Select minimum fitting-fold training MSE; held-out source episodes are excluded.
        xs,ys=Standardizer(x[~hold]),Standardizer(y[~hold])
        m,ck=fit(x[~hold],y[~hold],x[~hold],y[~hold],xs,ys,130+fold,WORK/'calibration'/task/f'fold{fold}.pt')
        err[hold]=predict(m,ck,x[hold])-y[hold]
    path.parent.mkdir(exist_ok=True);np.savez_compressed(path,error=err.reshape(-1,16,14),time=tt,episode=ep)
    return dict(np.load(path))




def rollout_one(args):
    task,layout,policy_path,output_path=args
    path=Path(output_path)
    if not path.exists():
        m,ck=load_policy(Path(policy_path));rr=rollout(task,layout,m,ck)
        tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**rr);tmp.replace(path)
    return str(path)


