"""Replay fixed, outcome-independent examples from the current manuscript studies.

Requires the ACT environment in requirements-simulation.txt (MuJoCo 2.3.3).
This renders archived actions, not new trials or newly inferred policy actions.
"""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments/scana_r')]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def render(path, task, output):
    import mujoco
    if mujoco.__version__!='2.3.3': raise RuntimeError('Use pinned MuJoCo 2.3.3 for exact archived ACT replay.')
    from sim_bridge import sim_env, initial_pose, causal_observation
    import imageio.v2 as imageio
    with np.load(path,allow_pickle=False) as z: r={k:z[k] for k in z.files}
    layout=int(path.stem);sim_env.BOX_POSE[0]=initial_pose(task,layout)
    env=sim_env.make_sim_env('sim_'+task);ts=env.reset();errors=[];rewards=[]
    env.physics.model.vis.global_.offwidth=1280;env.physics.model.vis.global_.offheight=720
    try:
        with imageio.get_writer(str(output),fps=25,codec='libx264',pixelformat='yuv420p',quality=8,macro_block_size=1,ffmpeg_params=['-movflags','+faststart']) as writer:
            for step,action in enumerate(r['action']):
                actual=causal_observation(ts.observation,step) if 'clean_input' in r else ts.observation['qpos']
                recorded=r['clean_input'][step] if 'clean_input' in r else r['state'][step]
                err=float(np.max(np.abs(actual-recorded)));errors.append(err)
                if err>1e-5:raise AssertionError(f'Archived replay diverged at step {step}: {err}')
                if step%2==0:writer.append_data(env.physics.render(height=360,width=540,camera_id='angle'))
                ts=env.step(action);rewards.append(float(ts.reward))
        np.testing.assert_array_equal(rewards,r['reward'])
    finally: env.close()
    return dict(state_max_error=max(errors),rewards_identical=True,success=int(max(rewards)==4),terminal_success=int(rewards[-1]==4),source_sha256=sha(path),video_sha256=sha(output),frames=200,fps=25,seconds=8)

def compose(output,name,clips):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    fontpath=next((p for p in ['C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'] if Path(p).exists()),None)
    if not fontpath:raise RuntimeError('Install Arial or DejaVu Sans for neutral panel labels.')
    font=ImageFont.truetype(fontpath,22)
    readers=[imageio.get_reader(str(output/c['video'])) for c in clips]
    columns=2 if len(clips)==4 else 3;rows=2 if len(clips)==4 else 1
    target=output/(name+'.mp4');size=(columns*540,rows*420)
    try:
        with imageio.get_writer(str(target),fps=25,codec='libx264',pixelformat='yuv420p',quality=8,macro_block_size=1,ffmpeg_params=['-movflags','+faststart']) as w:
            for i in range(200):
                canvas=Image.new('RGB',size,'white');draw=ImageDraw.Draw(canvas)
                for j,(r,c) in enumerate(zip(readers,clips)):
                    x=(j%columns)*540;y=(j//columns)*420
                    draw.text((x+12,y+7),c['label'],font=font,fill='#253e4c')
                    draw.text((x+12,y+32),'Success' if c['success'] else 'Failure',font=font,fill='#34495e')
                    canvas.paste(Image.fromarray(r.get_data(i)),(x,y+60))
                w.append_data(np.asarray(canvas))
                if i==125:canvas.save(output/(name+'-poster.jpg'),quality=94)
    finally:
        for r in readers:r.close()
    return dict(video=target.name,sha256=sha(target),inputs=[c['video'] for c in clips],width=size[0],height=size[1],frames=200,fps=25,seconds=8,edits='Synchronized full-length panels with outcome labels; no cuts or speed changes.')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifacts',type=Path,default=ROOT/'artifacts/independent_libraries_v46')
    ap.add_argument('--output',type=Path,default=ROOT/'docs/assets/current-videos')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    previous=args.output/'provenance.json'
    cache={c['source']:c for c in json.loads(previous.read_text())['clips']} if previous.exists() else {}
    clips=[];compositions=[]
    for task in ['transfer_cube','insertion']:
        designs=[('libraries',[(f'evidence/replication/{task}/101/test/17/{m.replace(" ","_")}/97000.npz',m) for m in ['Clean repeat','Gaussian recovery','SCANA-R','Gaussian CPU budget']]),
                 ('noise',[(f'evidence/failure_interventions/{task}/7/Calibrated_recovery/{c}/96000.npz',label) for c,label in ([('nominal','SCANA-R: nominal'),('noise_low_support','SCANA-R: low-support noise'),('noise_supported','SCANA-R: other-channel noise')] if task=='transfer_cube' else [('nominal','SCANA-R: nominal'),('noise_joints','SCANA-R: joint noise'),('noise_objects','SCANA-R: object noise')])]),
                 ('replanning',[(f'evidence/failure_interventions/{task}/7/Calibrated_recovery/{c}/96000.npz',label) for c,label in [('nominal','Execute 8 steps'),('replan_1','Execute 1 step'),('replan_1_ensemble','1 step + temporal ensemble')]])]
        for group,records in designs:
            groupclips=[]
            for j,(rel,label) in enumerate(records):
                video=f'{task}-{group}-{j}.mp4'
                old=cache.get('independent_libraries_v46/'+rel)
                report=dict(old) if old and old['source_sha256']==sha(args.artifacts/rel) and (args.output/video).exists() and old['video_sha256']==sha(args.output/video) else render(args.artifacts/rel,task,args.output/video)
                report.update(video=video,source='independent_libraries_v46/'+rel,task=task,group=group,label=label)
                clips.append(report);groupclips.append(report);print(json.dumps(report),flush=True)
            compositions.append(compose(args.output,f'{task}-{group}',groupclips))
    report=dict(selection_rule='Outcome-independent: first task library 101, policy seed 17, layout 97000; intervention policy seed 7, layout 96000. Same fixed identifiers for all compared methods and conditions. These examples are not aggregate estimates.',rendering='Recorded actions replayed in MuJoCo 2.3.3. Every pre-action state/current clean input and every reward verified; 400 physics steps at 50 Hz, 200 rendered frames at 25 fps. No retraining.',clips=clips,compositions=compositions)
    (args.output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
