"""Render recorded evaluation actions in the pinned ACT simulator for the website.

This produces visualizations of existing records, not new evaluation trials.
The simulator state and rewards are compared with the archived rollout at every
step. No policy, task physics, action or success definition is modified.
"""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments/scana_r')]

def compose(output, font_path=None):
    """Synchronize two uncut clips; add a neutral title strip and actual-frame poster."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    candidates = [font_path, 'C:/Windows/Fonts/arial.ttf',
                  '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    font_file = next((p for p in candidates if p and Path(p).is_file()), None)
    if not font_file:
        raise RuntimeError('Provide a TrueType font with --font for video titles.')
    font = ImageFont.truetype(str(font_file), 26)
    designs = [
        ('teaser', 'transfer_cube-scana-r', 'insertion-scana-r',
         'Transfer Cube  /  SCANA-R', 'Insertion  /  SCANA-R'),
        ('transfer-comparison', 'transfer_cube-clean', 'transfer_cube-scana-r',
         'Transfer Cube  /  Clean repeat', 'Transfer Cube  /  SCANA-R'),
        ('insertion-comparison', 'insertion-clean', 'insertion-scana-r',
         'Insertion  /  Clean repeat', 'Insertion  /  SCANA-R')]
    products = []
    for name, left, right, title_left, title_right in designs:
        readers = [imageio.get_reader(str(output / (s + '.mp4'))) for s in [left, right]]
        path = output / (name + '.mp4')
        try:
            assert all(r.count_frames() == 200 for r in readers)
            with imageio.get_writer(str(path), format='FFMPEG', fps=25,
                    codec='libx264', pixelformat='yuv420p', quality=8,
                    macro_block_size=1, ffmpeg_params=['-movflags', '+faststart', '-preset', 'medium']) as writer:
                for i in range(200):
                    canvas = Image.new('RGB', (1456, 544), 'white')
                    draw = ImageDraw.Draw(canvas)
                    draw.text((20, 19), title_left, font=font, fill='#333333')
                    draw.text((756, 19), title_right, font=font, fill='#333333')
                    for x, reader in zip([0, 736], readers):
                        canvas.paste(Image.fromarray(reader.get_data(i)), (x, 64))
                    writer.append_data(np.asarray(canvas))
                    if i == 125:
                        canvas.save(output / (name + '-poster.jpg'), quality=94)
        finally:
            for reader in readers: reader.close()
        products.append(dict(video=path.name, inputs=[left + '.mp4', right + '.mp4'],
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             width=1456, height=544, frames=200, fps=25, seconds=8,
                             edits='Synchronized side-by-side composition and title strip only; no cuts or speed changes.'))
    return products

def select_examples():
    import pandas as pd
    d=pd.read_csv(ROOT/'results/independent_test/per_episode.csv')
    examples=[]
    for task in ['transfer_cube','insertion']:
        s=d[(d.task==task)&(d.seed==7)]
        w=s.pivot(index='layout',columns='method',values='success')
        eligible=w[(w['Calibrated recovery']==1)&(w['Clean repeat']==0)].index
        if not len(eligible):raise RuntimeError('No matching illustrative pair')
        examples.append((task,7,int(min(eligible))))
    return examples

def simulate(task,seed,layout,method,width,height,camera,output=None,preview=False):
    from sim_bridge import sim_env,initial_pose
    from common import WORK
    import imageio.v2 as imageio
    path=WORK/'independent_test/rollouts'/task/str(seed)/method/f'{layout}.npz'
    with np.load(path) as f:r={k:f[k] for k in f}
    sim_env.BOX_POSE[0]=initial_pose(task,layout)
    env=sim_env.make_sim_env('sim_'+task);ts=env.reset()
    # Increase only the offscreen framebuffer capacity, never a physics parameter.
    env.physics.model.vis.global_.offwidth=max(1280,width)
    env.physics.model.vis.global_.offheight=max(720,height)
    writer=None;maxerr=0.;rewards=[]
    if output:
        writer=imageio.get_writer(str(output),format='FFMPEG',fps=25,codec='libx264',pixelformat='yuv420p',quality=8,macro_block_size=1,ffmpeg_params=['-movflags','+faststart','-preset','medium'])
    try:
        for step,action in enumerate(r['action']):
            error=float(np.max(np.abs(ts.observation['qpos']-r['state'][step])))
            maxerr=max(maxerr,error)
            if error>1e-5:raise AssertionError(f'Replay state differs by {error} at {step}')
            if preview and step in [160,280]:
                for cam in ['angle','top','left_pillar']:
                    imageio.imwrite(str(preview/f'{task}-{cam}-{step}.png'),env.physics.render(height=height,width=width,camera_id=cam))
            if writer and step%2==0:
                writer.append_data(env.physics.render(height=height,width=width,camera_id=camera))
            ts=env.step(action);rewards.append(float(ts.reward))
        np.testing.assert_array_equal(rewards,r['reward'])
    finally:
        if writer:writer.close()
        env.close()
    return dict(task=task,training_seed=seed,layout=layout,method=method,
                source=path.relative_to(ROOT).as_posix(),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                state_max_error=maxerr,rewards_identical=True,success=int(max(rewards)==4),terminal_success=int(rewards[-1]==4),
                rendered_frames=200,fps=25,duration_seconds=8,camera=camera)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'docs/assets/videos')
    ap.add_argument('--preview',type=Path)
    ap.add_argument('--compose-only', action='store_true', help='Compose already rendered clips.')
    ap.add_argument('--font',type=Path,help='Optional TrueType font used for video title strips.')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.compose_only:
        path=a.output/'provenance.json'
        report=json.loads(path.read_text(encoding='utf-8'))
        report['compositions']=compose(a.output,a.font)
        path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
        print('Composed three synchronized videos and posters.');return
    if a.preview:a.preview.mkdir(parents=True,exist_ok=True)
    reports=[]
    for task,seed,layout in select_examples():
        for method,slug in [('Calibrated recovery','scana-r'),('Clean repeat','clean')]:
            if a.preview and method=='Clean repeat':continue
            output=None if a.preview else a.output/f'{task}-{slug}.mp4'
            rep=simulate(task,seed,layout,method,720,480,'angle',output,a.preview)
            if output:rep['video']=output.name;rep['video_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
            reports.append(rep);print(json.dumps(rep),flush=True)
    if not a.preview:
        (a.output/'provenance.json').write_text(json.dumps(dict(selection_rule='For each task, fix training seed 7 and select the lowest layout where recorded SCANA-R succeeds and Clean repeat fails. Intentionally selected illustrative contrast, not a random sample or a replacement for aggregate results.',rendering='Replay recorded actions under the pinned simulator; no policy retraining; every recorded state and reward verified. 50 Hz simulation rendered every second step at 25 fps, real-time speed.',clips=reports,compositions=compose(a.output,a.font)),indent=2)+'\n',encoding='utf-8',newline='\n')

if __name__=='__main__':main()
