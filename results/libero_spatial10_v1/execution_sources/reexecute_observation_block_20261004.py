"""Predeclared all-model reexecution of one observation-mismatched block.
Frozen learner and protocol files remain byte-identical. Outputs isolated.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
import psutil
import learner

ROOT=Path(__file__).resolve().parent
INCIDENT=ROOT/'analysis/initial_rgb_mismatch_20261004'
PLAN=json.loads((INCIDENT/'repair_plan.json').read_text(encoding='utf-8-sig'))
OUT=INCIDENT/'reexecution'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write_result(name,data):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');tmp.replace(p)


def main():
    assert not OUT.exists(), 'This is one predeclared run, never overwrite outcomes'
    for proc in psutil.process_iter(['pid','name']):
        try:
            if 'python' not in (proc.info['name'] or '').lower():continue
            scripts={Path(x).name for x in proc.cmdline() if x.endswith('.py')}
            if scripts & {'full_run.py','learner.py','calibrate.py','collect_recovery.py'}:
                assert Path(proc.cwd()).resolve()!=ROOT, 'Existing active study; no parallel run'
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    assert not json.loads((ROOT/'manuscript_followup_state.json').read_text(encoding='utf-8-sig'))['user_pause']['requested']
    for name,d in PLAN['sha256_frozen_files'].items():assert sha(ROOT/name)==d,name
    OUT.mkdir()
    learner.ROOT=OUT
    # Scope restriction only: same task, init, evaluation seed and policy settings.
    learner.CFG={**learner.CFG,'tasks':[3],'test_initial_states':[11]}
    learner.write_json=write_result
    original_infer=learner.infer
    for name in PLAN['models_to_reexecute']:
        for file in ['latest.pt','training.json','COMPLETE.json']:
            rel=f'models/{name}/{file}'
            assert sha(ROOT/rel)==PLAN['sha256_original_model_artifacts'][rel]
        for f,d in PLAN['sha256_frozen_files'].items():assert sha(ROOT/f)==d
        captured={'done':False}
        def capture_infer(model,ck,obs,tid):
            if not captured['done']:
                p=OUT/'evaluation'/name/'first_pre_action_rgb.npz'
                np.savez_compressed(p,agentview=obs['agentview_image'],eye_in_hand=obs['robot0_eye_in_hand_image'])
                captured['done']=True
            return original_infer(model,ck,obs,tid)
        learner.infer=capture_infer
        rows=learner.evaluate(ROOT/'models'/name/'latest.pt',split='test')
        assert len(rows)==1 and rows[0]['task']==3 and rows[0]['init_state']==11
        assert captured['done']
    learner.infer=original_infer
    reference=None
    checked=[]
    for name in PLAN['models_to_reexecute']:
        p=OUT/'evaluation'/name/'test_task3_init11.npz'
        with np.load(p) as a:
            fields={k:a[k].copy() for k in ['model_body_pos','model_body_quat','initial_proprio','initial_camera_sha256']}
            fields['initial_state']=a['states'][0].copy()
            with np.load(OUT/'evaluation'/name/'first_pre_action_rgb.npz') as rgb:
                h=hashlib.sha256(rgb['agentview'].tobytes()+rgb['eye_in_hand'].tobytes()).hexdigest()
            assert h==str(a['initial_camera_sha256'])
            assert h==PLAN['full_initial_field_mismatches'][0]['reference_rgb_sha256']
            if reference is None:reference=fields
            else:assert all(np.array_equal(reference[k],v) for k,v in fields.items()),name
            checked.append({'model':name,'initial_rgb_sha256':h,'trace_sha256':sha(p)})
    result={'completed_at':time.strftime('%Y-%m-%d %H:%M:%S'),'all_twelve_initial_fields_exact':True,'first_rgb_pixels_saved':True,'original_reference_rgb_reproduced':True,'checked':checked,'adopted_into_formal_records':False,'rule':'All twelve once; no outcome-based selection.'}
    (INCIDENT/'reexecution_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
