"""Render archived reference and fixed-layout policy records, with exact replay QA."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/metaworld_extension'))
from bridge import TASKS,make_env,HORIZON
SOURCE=ROOT/'artifacts/metaworld_extension_v1'
DEST=ROOT/'outputs/mujoco_extension_media'

def render(task,seed,source,kind):
    with np.load(source) as f:r={k:f[k] for k in f}
    env=make_env(task,seed);obs,_=env.reset();env.model.vis.global_.offwidth=640;env.model.vis.global_.offheight=480
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=[0,.6,.13];camera.distance=1.2;camera.azimuth=180;camera.elevation=-25
    frame_index=min(next((i for i,x in enumerate(r['success']) if x),HORIZON-9)+8,HORIZON-1)
    path=DEST/f'{task}-{kind}.mp4';rewards=[];success=[];max_error=0.
    with mujoco.Renderer(env.model,height=480,width=640) as renderer:
        with imageio.get_writer(str(path),fps=1/env.dt/2,codec='libx264',pixelformat='yuv420p',quality=8,
                ffmpeg_params=['-movflags','+faststart'],macro_block_size=1) as writer:
            for t,action in enumerate(r['action']):
                err=float(np.max(np.abs(obs-r['raw_obs'][t])));max_error=max(max_error,err)
                assert err<1e-9,(task,kind,t,err)
                if t%2==0 or t==frame_index:
                    renderer.update_scene(env.data,camera=camera);frame=renderer.render()
                    if t%2==0:writer.append_data(frame)
                    if t==frame_index:plt.imsave(DEST/f'{task}-{kind}.png',frame)
                obs,reward,_,_,info=env.step(action);rewards.append(reward);success.append(int(info['success']))
    np.testing.assert_array_equal(rewards,r['reward']);np.testing.assert_array_equal(success,r['success'])
    dt=env.dt;env.close()
    return dict(task=task,kind=kind,layout=seed,training_seed=7 if kind=='policy' else None,
        source=source.relative_to(ROOT).as_posix(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        video=path.name,video_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        state_max_error=max_error,rewards_identical=True,success=int(max(success)),terminal_success=int(success[-1]),
        fps=1/dt/2,frames=125,seconds=HORIZON*dt,poster_step=frame_index,
        camera=dict(lookat=[0,.6,.13],distance=1.2,azimuth=180,elevation=-25),
        presentation='Full trajectory, real-time speed, no cuts, no physics changes; offscreen framebuffer/camera only.')

def main():
    DEST.mkdir(parents=True,exist_ok=True);reports=[]
    for task in TASKS:
        manifest=json.loads((SOURCE/'references'/task/'manifest.json').read_text())
        seed=min(x['seed'] for x in manifest if x['split']=='train' and x['terminal_success'])
        src=SOURCE/'references'/task/'train'/f'{seed}.npz'
        reports.append(render(task,seed,src,'reference'))
        src=SOURCE/'test/rollouts'/task/'7/SCANA-R/nominal/30000.npz'
        if src.exists():reports.append(render(task,30000,src,'policy'))
        print(task,'media rendered',flush=True)
    (DEST/'provenance.json').write_text(json.dumps(dict(policy_selection='Always training seed7, layout30000 in every task, regardless of outcome.',reference_selection='Lowest accepted training reference seed; task illustration only, not learned-policy performance.',clips=reports),indent=2)+'\n',encoding='utf-8',newline='\n')

if __name__=='__main__':main()
