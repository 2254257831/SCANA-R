"""Replay legacy recovery records to verify actual pre-action sensor observations.

This is an integrity audit, excluded from collection/training cost. Simulator
qpos and standard robosuite sensor output are distinct quantities; do not force
fresh sensor samples or change the benchmark's observation pipeline.
"""
from runtime import *
import numpy as np, h5py, cv2, hashlib, argparse, time

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def verify(library,limit=None):
    folder=ROOT/'recovery'/library
    attempts=json.loads((folder/'attempts.json').read_text())
    checked=0;env=None;last_task=None;h=None
    out=folder/'observation_audit';out.mkdir(exist_ok=True)
    for p in sorted((folder/'accepted').glob('*.npz')):
        record=dict(np.load(p));r=attempts[int(record['attempt_index'][0])]
        trace_path=folder/'traces'/f"task{r['task']}_{r['episode']}_branch{r['branch']}_scale{r['scale']}.npz"
        trace=dict(np.load(trace_path));proof=out/(p.stem+'.json')
        if 'observed_proprio' in trace:continue
        if proof.exists():
            old=json.loads(proof.read_text())
            assert old['accepted_sha256']==sha(p) and old['trace_sha256']==sha(trace_path)
            continue
        if r['task']!=last_task:
            if env is not None:env.close();h.close()
            env=make_env(r['task'],size=128);h=h5py.File(data_file(r['task']),'r');last_task=r['task']
        obs=reset_demo(env,h['data'][r['episode']]);states=[];observed=[];rewards=[];image_error=0
        for t,a in enumerate(trace['actions']):
            states.append(env.sim.get_state().flatten());observed.append(proprio(obs))
            if t in record['time']:
                j=int(np.flatnonzero(record['time']==t)[0])
                ims=np.stack([cv2.resize(obs[k],(96,96),interpolation=cv2.INTER_AREA).transpose(2,0,1) for k in ['agentview_image','robot0_eye_in_hand_image']])
                image_error=max(image_error,int(np.abs(ims.astype('int16')-record['images'][j].astype('int16')).max()))
                assert np.array_equal(ims,record['images'][j]),f'Image mismatch {p.name} t{t}'
                assert np.array_equal(proprio(obs),record['proprio'][j]),f'Sensor mismatch {p.name} t{t}'
            obs,reward,_,_=env.step(a);rewards.append(reward)
        state_error=float(np.max(np.abs(np.asarray(states)-trace['states'])))
        assert state_error<1e-9,(p.name,state_error)
        assert np.array_equal(rewards,trace['rewards'])
        assert bool(env.check_success())==r['terminal_success']
        observed=np.asarray(observed)
        np.savez_compressed(out/(p.stem+'.npz'),observed_proprio=observed)
        result={'accepted_sha256':sha(p),'trace_sha256':sha(trace_path),'replayed_steps':len(states),'checked_rows':len(record['time']),
                'state_max_abs_error':state_error,'sensor_max_abs_error':0.,'image_max_abs_error':image_error,
                'sensor_vs_final_integrated_qpos_max':float(np.max(np.abs(observed[:,:7]-trace['states'][:,1:8]))),
                'interpretation':'Exact returned pre-action sensor observation and images verified by deterministic replay; final integrated simulator qpos is not a synchronous sensor measurement.'}
        proof.write_text(json.dumps(result,indent=2),encoding='utf8')
        checked+=1;print(library,p.stem,'PASS',checked,flush=True)
        if limit and checked>=limit:break
    if env is not None:env.close();h.close()
    print('Observation replay audit complete',library,checked,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('library');p.add_argument('--limit',type=int);a=p.parse_args();verify(a.library,a.limit)
