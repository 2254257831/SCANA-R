"""Collect executed reference-tail labels with fresh pre-action RGB observations."""
from runtime import *
from recovery_probe import normalized_pulse
import numpy as np,h5py,time,argparse,cv2,hashlib

def main(method,seed,amp,n):
    cfg=json.loads((ROOT/'protocol.json').read_text());splits=json.loads((ROOT/'protocol/source_splits.json').read_text())
    if method=='scana_r':bank=dict(np.load(ROOT/'calibration'/f'lib{seed}'/'bank.npz'))
    out=ROOT/'recovery'/f'lib{seed}_{method}';out.mkdir(parents=True,exist_ok=True)
    trace_dir=out/'traces';trace_dir.mkdir(exist_ok=True)
    run_config={'method':method,'seed':seed,'amplitude':amp,'sources_per_task':n,'reset':'source XML plus cleared Panda accumulator; deterministic cached reset within same source'}
    cp=out/'run_config.json'
    if cp.exists():assert json.loads(cp.read_text())==run_config,'refusing to mix collection protocols'
    else:write_json(str(cp.relative_to(ROOT)),run_config)
    rows=json.loads((out/'attempts.json').read_text()) if (out/'attempts.json').exists() else []
    branch_dir=out/'accepted';branch_dir.mkdir(exist_ok=True)
    start_total=time.perf_counter();sim_steps=sum(x['steps'] for x in rows)
    elapsed_before=json.loads((out/'progress.json').read_text()).get('active_seconds',0) if (out/'progress.json').exists() else 0
    complete={(x['task'],x['episode'],x['branch']) for x in rows if x['terminal_success'] or x['scale']==0}
    for tid in cfg['tasks']:
        env=make_env(tid,size=128)
        with h5py.File(data_file(tid),'r') as h:
            for ei,key in enumerate(splits[str(tid)]['train'][:n]):
                g=h['data'][key];reference=g['actions'][:];L=len(reference)
                for pi,phase in enumerate(cfg['pulse_time_fractions']):
                    if (tid,key,pi) in complete:continue
                    endpoint=int(np.clip(round(L*phase),8,L-cfg['horizon']))
                    rng=np.random.default_rng(seed+tid*10000+ei*100+pi)
                    bank_row=None
                    if method=='gaussian':pulse=normalized_pulse(rng,amp)
                    else:
                        candidates=np.flatnonzero((bank['task']==tid)&(np.abs(bank['phase']-(endpoint-8)/L)<=cfg['phase_matching_radius_fraction']))
                        if not len(candidates):raise RuntimeError('empty same-task phase neighborhood')
                        bank_row=int(rng.choice(candidates));pulse=bank['errors'][bank_row].copy();pulse[:,6]=0
                        for s in [slice(0,3),slice(3,6)]:pulse[:,s]*=amp/max(float(np.sqrt(np.mean(pulse[:,s]**2))),1e-8)
                        pulse=np.clip(pulse,-3*amp,3*amp)
                    for scale in cfg['backtracking']:
                        if any(x['task']==tid and x['episode']==key and x['branch']==pi and x['scale']==scale for x in rows):continue
                        actions=reference.copy();actions[endpoint-8:endpoint]=np.clip(reference[endpoint-8:endpoint]+scale*pulse,-1,1)
                        t0=time.perf_counter();obs=reset_demo(env,g);states=[];rewards=[];candidates=[];observed_proprio=[]
                        times=list(range(endpoint,min(endpoint+16,L-cfg['horizon']+1),4))
                        for t,a in enumerate(actions):
                            states.append(env.sim.get_state().flatten())
                            observed_proprio.append(proprio(obs))
                            if t in times:
                                assert np.array_equal(actions[t:t+cfg['horizon']],reference[t:t+cfg['horizon']])
                                ims=np.stack([cv2.resize(obs[k],(96,96),interpolation=cv2.INTER_AREA).transpose(2,0,1) for k in ['agentview_image','robot0_eye_in_hand_image']])
                                candidates.append({'images':ims,'proprio':proprio(obs),'actions':actions[t:t+cfg['horizon']].astype('float32'),'task':tid,'time':t})
                            obs,r,_,_=env.step(a);rewards.append(float(r))
                        ok=bool(env.check_success());sim_steps+=L
                        row={'method':method,'library_seed':seed,'task':tid,'episode':key,'branch':pi,'phase':phase,'endpoint':endpoint,'scale':scale,
                             'amplitude':amp,'bank_row':bank_row,'terminal_success':ok,'ever_success':bool(max(rewards)>0),'identity':scale==0,
                             'steps':L,'seconds':time.perf_counter()-t0,'accepted_rows':len(candidates) if ok else 0,
                             'effective_action_rms':float(np.sqrt(np.mean((actions[endpoint-8:endpoint,:6]-reference[endpoint-8:endpoint,:6])**2)))}
                        rows.append(row)
                        np.savez_compressed(trace_dir/f'task{tid}_{key}_branch{pi}_scale{scale}.npz',states=states,actions=actions,rewards=rewards,observed_proprio=observed_proprio)
                        if ok:
                            np.savez_compressed(branch_dir/f'task{tid}_{key}_branch{pi}.npz',images=np.stack([c['images'] for c in candidates]),
                                 proprio=np.stack([c['proprio'] for c in candidates]),actions=np.stack([c['actions'] for c in candidates]),
                                 task=np.asarray([tid]*len(candidates)),time=[c['time'] for c in candidates],attempt_index=np.asarray([len(rows)-1]*len(candidates)))
                        write_json(str((out/'attempts.json').relative_to(ROOT)),rows)
                        write_json(str((out/'progress.json').relative_to(ROOT)),{'active_seconds':elapsed_before+time.perf_counter()-start_total,'attempts':len(rows),'task':tid,'source':ei+1,'branch':pi})
                        print(method,seed,'task',tid,ei+1,'/',n,'branch',pi,'scale',scale,'success',ok,flush=True)
                        if ok:break
        env.close()
    packs=[dict(np.load(p)) for p in sorted(branch_dir.glob('*.npz'))]
    if not packs:raise RuntimeError('no accepted recovery records')
    combined={k:np.concatenate([c[k] for c in packs]) for k in packs[0]}
    images=combined.pop('images');np.save(out/'images.npy',images);np.savez(out/'labels.npz',**combined)
    write_json(str((out/'summary.json').relative_to(ROOT)),{'method':method,'library_seed':seed,'amplitude':amp,'source_episodes_per_task':n,'rows':len(images),
            'attempts':len(rows),'simulator_steps':sim_steps,'collection_wall_seconds':elapsed_before+time.perf_counter()-start_total,'attempt_seconds':sum(x['seconds'] for x in rows),
            'nonzero_successful_branches':sum(x['terminal_success'] and not x['identity'] for x in rows),'identity_branches':sum(x['terminal_success'] and x['identity'] for x in rows),
            'protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()})
    print('collection complete',out,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['gaussian','scana_r'],required=True);ap.add_argument('--seed',type=int,default=101);ap.add_argument('--amp',type=float,default=.05);ap.add_argument('--n',type=int,default=40);a=ap.parse_args();main(a.method,a.seed,a.amp,a.n)
