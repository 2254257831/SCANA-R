"""Reload frozen weights and recompute all five policies on one fixed layout per task."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/metaworld_extension'))
from run import TASKS,METHODS,OUT,load_npz,load_model,rollout,make_env,cKDTree,np

def main():
    protocol=json.loads((OUT/'protocol_frozen.json').read_text());rows=[]
    for task in TASKS:
        for method in METHODS:
            suffix='.npz' if method=='Demonstration kNN' else '.pt'
            path=OUT/'policies'/task/'7'/(method+suffix)
            assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol['checkpoints'][path.relative_to(OUT).as_posix()]
            if suffix=='.npz':
                ck=load_npz(path);model=None;knn=(cKDTree((ck['input']-ck['input_mean'])/ck['input_std']),ck['action'],ck)
            else:model,ck=load_model(path);knn=None
            env=make_env(task,30000)
            try:
                for scenario in ['nominal','pulse']:
                    got=rollout(task,30000,model,ck,knn,scenario,env)
                    expected=load_npz(OUT/'test/rollouts'/task/'7'/method/scenario/'30000.npz')
                    for key in ['input','raw_obs','action','reward','success']:
                        np.testing.assert_array_equal(got[key],expected[key],err_msg=f'{task}/{method}/{scenario}/{key}')
                    rows.append(dict(task=task,method=method,scenario=scenario,seed=7,layout=30000,all_arrays_exact=True))
            finally:env.close()
        print(task,'10 fresh policy executions match archived records exactly.',flush=True)
    result=dict(verified_rollouts=len(rows),method='Fresh model inference and environment stepping; not action-only playback.',rows=rows)
    target=ROOT/'outputs/extension_replay_audit.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Verified',len(rows),'fresh policy executions.',flush=True)

if __name__=='__main__':main()
