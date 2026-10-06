"""Compact causal language-conditioned visuomotor policy for the complete Spatial suite."""
from runtime import *
import numpy as np,torch,time,random,argparse,cv2,hashlib
from torch import nn

torch.set_num_threads(4)
torch.backends.cudnn.benchmark=False
CFG=json.loads((ROOT/'protocol.json').read_text())

class Policy(nn.Module):
    def __init__(self,vocab,H=8):
        super().__init__()
        self.camera=nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.GroupNorm(4,32),nn.SiLU(),
             nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU(),
             nn.Conv2d(64,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU(),
             nn.Conv2d(64,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU(),nn.Flatten(),nn.Linear(64*6*6,128),nn.SiLU())
        self.words=nn.Embedding(vocab+1,32,padding_idx=0)
        self.head=nn.Sequential(nn.Linear(256+15+32,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,H*7))
        self.H=H
    def forward(self,images,prop,tokens):
        B=images.shape[0];v=self.camera(images.reshape(B*2,3,96,96).float()/255.-.5).reshape(B,-1)
        mask=(tokens!=0).unsqueeze(-1)
        lang=(self.words(tokens)*mask).sum(1)/mask.sum(1).clamp_min(1)
        return self.head(torch.cat([v,prop,lang],1)).reshape(B,self.H,7)

class Data:
    def __init__(self):
        self.images=np.load(ROOT/'cache/images.npy',mmap_mode='r')
        self.arr=dict(np.load(ROOT/'cache/labels.npz'))
        self.meta=json.loads((ROOT/'cache/metadata.json').read_text())
        self.tok=np.zeros((len(self.arr['task']),32),np.int64)
        for tid,t in self.meta['token_ids'].items():self.tok[self.arr['task']==int(tid),:len(t)]=t
    def norm(self,ids):
        p=self.arr['proprio'][ids];a=self.arr['actions'][ids]
        return {'pmean':p.mean(0),'pstd':np.maximum(p.std(0),.01),'amean':a.mean((0,1)),'astd':np.maximum(a.std((0,1)),.05)}
    def batch(self,ids,norm):
        return (torch.as_tensor(np.asarray(self.images[ids]),device='cuda'),
                torch.as_tensor((self.arr['proprio'][ids]-norm['pmean'])/norm['pstd'],device='cuda'),
                torch.as_tensor(self.tok[ids],device='cuda'),
                torch.as_tensor((self.arr['actions'][ids]-norm['amean'])/norm['astd'],device='cuda'))

def recovery_data(name,meta):
    d=object.__new__(Data);d.images=np.load(ROOT/'recovery'/name/'images.npy',mmap_mode='r')
    d.arr=dict(np.load(ROOT/'recovery'/name/'labels.npz'));d.meta=meta
    d.tok=np.zeros((len(d.arr['task']),32),np.int64)
    for tid,t in meta['token_ids'].items():d.tok[d.arr['task']==int(tid),:len(t)]=t
    return d

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def train(name,steps,seed=7,fold=None,recovery=None,time_budget=None):
    seed_all(seed);d=Data();trainmask=d.arr['train'].copy()
    if fold is not None:trainmask &=d.arr['fold']!=fold
    ids=np.flatnonzero(trainmask);norm=d.norm(ids)
    dev=np.flatnonzero(~d.arr['train'])
    rng=np.random.default_rng(seed);dev=rng.choice(dev,min(512,len(dev)),replace=False)
    model=Policy(len(d.meta['vocabulary'])).cuda();opt=torch.optim.AdamW(model.parameters(),lr=CFG['learning_rate'],weight_decay=1e-4)
    rd=recovery_data(recovery,d.meta) if recovery else None
    out=ROOT/'models'/name;out.mkdir(parents=True,exist_ok=True)
    losslog=[];torch.cuda.synchronize();start=time.perf_counter();last=start
    max_steps=steps if time_budget is None else 1000000
    for it in range(1,max_steps+1):
        ix=rng.choice(ids,CFG['batch_size'] if rd is None else CFG['batch_size']//2,replace=True)
        x,p,t,y=d.batch(ix,norm);model.train();pred=model(x,p,t);loss=nn.functional.mse_loss(pred,y)
        if rd is not None:
            ri=rng.integers(len(rd.arr['task']),size=CFG['batch_size']//2);rx,rp,rt,ry=rd.batch(ri,norm)
            loss=(loss+nn.functional.mse_loss(model(rx,rp,rt),ry))*.5
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
        finished=(it==max_steps or (time_budget is not None and time.perf_counter()-start>=time_budget))
        if it%250==0 or it==1 or finished:
            model.eval();val=[]
            with torch.no_grad():
                for vi in np.array_split(dev,8):
                    vx,vp,vt,vy=d.batch(vi,norm);val.append(float(nn.functional.mse_loss(model(vx,vp,vt),vy)))
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            row={'step':it,'training_loss':float(loss),'dev_action_MSE':float(np.mean(val)),'elapsed_seconds':elapsed}
            losslog.append(row);print(name,row,flush=True)
            write_json(f'models/{name}/training.json',losslog)
            ckpt={'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'norm':norm,'steps':it,'seed':seed,'fold_excluded':fold,'vocab':d.meta['vocabulary'],'tokens':d.meta['token_ids'],'elapsed_seconds':elapsed,'train_episodes':sorted(set(d.arr['episode'][ids].tolist())),'protocol':CFG,'recovery':recovery,'time_budget':time_budget}
            torch.save(ckpt,out/'latest.pt.tmp'); (out/'latest.pt.tmp').replace(out/'latest.pt')
        if finished:break
    return out/'latest.pt'

def load(path):
    ck=torch.load(path,map_location='cpu');m=Policy(len(ck['vocab'])).cuda();m.load_state_dict(ck['state_dict']);m.eval();return m,ck

@torch.no_grad()
def infer(model,ck,obs,tid):
    frames=[cv2.resize(obs[k],(96,96),interpolation=cv2.INTER_AREA).transpose(2,0,1) for k in ['agentview_image','robot0_eye_in_hand_image']]
    x=torch.as_tensor(np.array(frames)[None],device='cuda');n=ck['norm'];p=torch.as_tensor(((proprio(obs)-n['pmean'])/n['pstd'])[None],device='cuda')
    tokens=np.zeros((1,32),np.int64);t=ck['tokens'][str(tid)];tokens[0,:len(t)]=t
    pred=model(x,p,torch.as_tensor(tokens,device='cuda'))[0].cpu().numpy()
    return np.clip(pred*n['astd']+n['amean'],-1,1)

def evaluate(path,split='dev',count=None):
    import imageio.v2 as imageio
    model,ck=load(path);indices=CFG['development_initial_states'] if split=='dev' else CFG['test_initial_states']
    if count:indices=indices[:count]
    name=Path(path).parent.name;out=ROOT/'evaluation'/name;out.mkdir(parents=True,exist_ok=True)
    existing=out/f'{split}_results.json'
    rows=json.loads(existing.read_text()) if existing.exists() else []
    keys={(x['task'],x['init_state']) for x in rows}
    assert len(keys)==len(rows), 'duplicate evaluation records'
    assert all(x['policy_seed']==ck['seed'] and x['train_steps']==ck['steps'] for x in rows)
    for tid in CFG['tasks']:
        states=task_suite().get_task_init_states(tid)
        for ni,i in enumerate(indices):
            if (tid,i) in keys:
                assert (out/f'{split}_task{tid}_init{i}.npz').exists()
                continue
            evaluation_seed=100000+tid*100+i
            np.random.seed(evaluation_seed);random.seed(evaluation_seed)
            env=make_env(tid,size=128);env.seed(evaluation_seed);env.reset();obs=env.set_init_state(states[i])
            for _ in range(CFG['settling_steps']):obs,_,_,_=env.step(np.zeros(7))
            initial_prop=proprio(obs).copy()
            initial_rgb_sha256=hashlib.sha256(obs['agentview_image'].tobytes()+obs['robot0_eye_in_hand_image'].tobytes()).hexdigest()
            fixed_body_pos=env.sim.model.body_pos.copy();fixed_body_quat=env.sim.model.body_quat.copy()
            success=False;trace=[];actions=[];frames=[];rewards=[];success_flags=[];t0=time.perf_counter()
            for t in range(CFG['max_control_steps']):
                if t%CFG['execute_steps']==0:chunk=infer(model,ck,obs,tid)
                a=chunk[t%CFG['execute_steps']]
                trace.append(env.sim.get_state().flatten());actions.append(a)
                if ni==0 and t%2==0:frames.append(obs['agentview_image'][::-1])
                obs,r,done,info=env.step(a)
                rewards.append(float(r));success_flags.append(bool(env.check_success()))
                if success_flags[-1]:success=True;break
            row={'task':tid,'init_state':i,'split':split,'success':success,'steps':t+1,'seconds':time.perf_counter()-t0,'policy_seed':ck['seed'],'train_steps':ck['steps']}
            rows.append(row);print(name,row,flush=True)
            np.savez_compressed(out/f'{split}_task{tid}_init{i}.npz',states=trace,actions=actions,rewards=rewards,
                                success_flags=success_flags,final_state=env.sim.get_state().flatten(),
                                model_body_pos=fixed_body_pos,model_body_quat=fixed_body_quat,initial_proprio=initial_prop,initial_camera_sha256=initial_rgb_sha256)
            if frames:imageio.mimwrite(out/f'{split}_task{tid}_init{i}.mp4',frames,fps=10,macro_block_size=1)
            write_json(f'evaluation/{name}/{split}_results.json',rows)
            env.close()
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['train','evaluate']);ap.add_argument('--name',default='clean_warmup_seed7');ap.add_argument('--steps',type=int,default=5000);ap.add_argument('--seed',type=int,default=7);ap.add_argument('--fold',type=int);ap.add_argument('--count',type=int);ap.add_argument('--split',default='dev');ap.add_argument('--recovery');ap.add_argument('--time-budget',type=float);a=ap.parse_args()
    if a.stage=='train':train(a.name,a.steps,a.seed,a.fold,a.recovery,a.time_budget)
    else:evaluate(ROOT/'models'/a.name/'latest.pt',a.split,a.count)
