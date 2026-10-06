"""Audit timing and replay official actions without forcing subsequent states."""
from runtime import *
import numpy as np,h5py,time,hashlib,argparse,imageio.v2 as imageio

def audit():
    rows=[];splits={}
    for tid in json.loads((ROOT/'protocol.json').read_text())['tasks']:
        with h5py.File(data_file(tid),'r') as h:
            keys=sorted(h['data'],key=lambda x:int(x.split('_')[1]))
            order=np.random.default_rng(20261002+tid).permutation(len(keys))
            splits[str(tid)]={'train':[keys[i] for i in order[:40]],'dev':[keys[i] for i in order[40:]]}
            for key in keys:
                g=h['data'][key];st=g['states'][:];q=g['obs/joint_states'][:]
                rows.append({'task':tid,'episode':key,'length':len(q),
                             'same_index_joint_max':float(np.max(np.abs(q-st[:,1:8]))),
                             'shifted_joint_max':float(np.max(np.abs(q[:-1]-st[1:,1:8]))),
                             'image_convention':str(h['data'].attrs.get('macros_image_convention','unknown'))})
    write_json('protocol/source_splits.json',splits)
    write_json('preflight/timing_audit.json',{'rows':rows,'alignment':'obs[t-1] paired with actions[t:t+H]; discard t=0. Independently verify against replay before training.',
                                            'max_shifted_error':max(x['shifted_joint_max'] for x in rows)})
    print('timing audit',len(rows),'episodes; maximum shifted joint error',max(x['shifted_joint_max'] for x in rows),flush=True)

def replay(n=5,task_ids=tuple(range(10))):
    splits=json.loads((ROOT/'protocol/source_splits.json').read_text())
    rows=[]
    for tid in task_ids:
        env=make_env(tid,size=128)
        with h5py.File(data_file(tid),'r') as h:
            for idx,key in enumerate(splits[str(tid)]['train'][:n]):
                g=h['data'][key];actions=g['actions'][:];states=g['states'][:]
                t0=time.perf_counter();obs=reset_demo(env,g)
                initial_error=float(np.max(np.abs(env.sim.get_state().flatten()-states[0])))
                errors=[];rewards=[];frames=[];qerrors=[];image_errors=[]
                trace=[]
                for j,a in enumerate(actions):
                    if idx==0 and j%2==0:frames.append(obs['agentview_image'][::-1])
                    trace.append(env.sim.get_state().flatten())
                    obs,r,done,_=env.step(a);rewards.append(float(r))
                    qerrors.append(float(np.max(np.abs(obs['robot0_joint_pos']-g['obs/joint_states'][j]))))
                    if j+1<len(states):errors.append(float(np.linalg.norm(env.sim.get_state().flatten()-states[j+1])))
                    if j in [0,1,10]:image_errors.append(float(np.mean(np.abs(obs['agentview_image'].astype(float)-g['obs/agentview_rgb'][j].astype(float)))))
                row={'task':tid,'episode':key,'steps':len(actions),'terminal_success':bool(env.check_success()),'ever_success':bool(max(rewards)>0),
                     'initial_state_max_error':initial_error,'state_l2_max':max(errors),'joint_error_max':max(qerrors),
                     'pixel_MAE_first_steps':image_errors,'seconds':time.perf_counter()-t0}
                rows.append(row);print(row,flush=True)
                out=ROOT/'preflight/traces';out.mkdir(exist_ok=True)
                np.savez_compressed(out/f'task{tid}_{key}.npz',states=np.asarray(trace),actions=actions,rewards=rewards)
                if frames:imageio.mimwrite(out/f'task{tid}_{key}.mp4',frames,fps=10,macro_block_size=1)
                write_json('preflight/replay_results.json',rows)
        env.close()
    print('terminal success',sum(x['terminal_success'] for x in rows),'/',len(rows),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=5);ap.add_argument('--tasks',nargs='+',type=int,default=list(range(10)));args=ap.parse_args()
    audit();replay(args.n,args.tasks)
