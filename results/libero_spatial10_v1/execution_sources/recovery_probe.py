"""Training-source recovery acceptance gate at the pilot-fixed amplitude, with no learned-policy claims."""
from runtime import *
import numpy as np,h5py,time,argparse

def normalized_pulse(rng,amp):
    e=rng.normal(size=(8,7))
    for t in range(1,8):e[t]=.9*e[t-1]+np.sqrt(1-.9**2)*e[t]
    e[:,6]=0
    for g in [slice(0,3),slice(3,6)]:e[:,g]*=amp/max(np.sqrt(np.mean(e[:,g]**2)),1e-8)
    return np.clip(e,-3*amp,3*amp)

def rollout(env,g,actions):
    obs=reset_demo(env,g);trace=[];rewards=[]
    for a in actions:
        trace.append(env.sim.get_state().flatten());obs,r,_,_=env.step(a);rewards.append(float(r))
    return bool(env.check_success()),np.asarray(trace),np.asarray(rewards)

def main(n):
    cfg=json.loads((ROOT/'protocol.json').read_text());splits=json.loads((ROOT/'protocol/source_splits.json').read_text());rows=[];base_rows=[]
    out=ROOT/'preflight/recovery_probe_v2';out.mkdir(exist_ok=True)
    for tid in cfg['tasks']:
        env=make_env(tid,images=False,size=128)
        with h5py.File(data_file(tid),'r') as h:
            for ei,key in enumerate(splits[str(tid)]['train'][:n]):
                g=h['data'][key];reference=g['actions'][:]
                ok,baseline,_=rollout(env,g,reference);base_rows.append({'task':tid,'episode':key,'success':ok})
                for pi,phase in enumerate(cfg['pulse_time_fractions']):
                    endpoint=int(np.clip(round(len(reference)*phase),8,len(reference)-8))
                    for amp in cfg['amplitude_candidates']:
                        seed=101+tid*10000+ei*100+pi
                        pulse=normalized_pulse(np.random.default_rng(seed),amp)
                        for scale in cfg['backtracking']:
                            actions=reference.copy();actions[endpoint-8:endpoint]=np.clip(actions[endpoint-8:endpoint]+scale*pulse,-1,1)
                            start=time.perf_counter();ok,tr,rewards=rollout(env,g,actions)
                            row={'task':tid,'episode':key,'phase':phase,'endpoint':endpoint,'amplitude':amp,'scale':scale,'seed':seed,
                                 'terminal_success':ok,'ever_success':bool(np.max(rewards)>0),'identity':scale==0,
                                 'effective_action_rms':float(np.sqrt(np.mean((actions[endpoint-8:endpoint,:6]-reference[endpoint-8:endpoint,:6])**2))),
                                 'pulse_end_state_l2_vs_baseline':float(np.linalg.norm(tr[endpoint]-baseline[endpoint])),
                                 'steps':len(actions),'seconds':time.perf_counter()-start}
                            rows.append(row)
                            np.savez_compressed(out/f'task{tid}_{key}_phase{pi}_amp{amp}_scale{scale}.npz',states=tr,actions=actions,rewards=rewards)
                            write_json('preflight/recovery_probe_v2/attempts.json',rows)
                            print(row,flush=True)
                            if ok:break
        env.close()
    write_json('preflight/recovery_probe_v2/baselines.json',base_rows)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=2);main(ap.parse_args().n)
