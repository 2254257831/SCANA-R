"""Check causal input, successful acceptance and exported executed-label alignment."""
from pathlib import Path
import json, numpy as np
from run import OUT, TASKS, HORIZON, load_npz, dump, sha

def main():
    count=0;attempts=0;source_episodes=0
    for task in TASKS:
        manifest=json.loads((OUT/'references'/task/'manifest.json').read_text())
        train={r['seed'] for r in manifest if r['split']=='train'}
        dev={r['seed'] for r in manifest if r['split']=='development'}
        assert not train.intersection(dev)
        for row in manifest:
            path=OUT/'references'/task/row['split']/f'{row["seed"]}.npz'
            assert sha(path)==row['sha256'];source_episodes+=1
        for path in (OUT/'recovery'/task).rglob('*.json'):
            a=json.loads(path.read_text());r=load_npz(path.with_suffix('.npz'))
            source=load_npz(OUT/'references'/task/'train'/f'{int(r["source"])}.npz')
            end=int(r['endpoint']);last=a['attempts'][-1]
            for attempt in a['attempts']:
                assert sha(path.parent/attempt['file'])==attempt['sha256'];attempts+=1
            accepted=load_npz(path.parent/last['file'])
            assert accepted['success'][-1]==1 and last['terminal_success']==1
            assert np.array_equal(accepted['action'][:,3],source['action'][:,3])
            assert np.array_equal(accepted['action'][end:],source['action'][end:])
            inputs=np.c_[accepted['raw_obs'][:,:18],accepted['raw_obs'][:,-3:],np.arange(HORIZON)/HORIZON].astype(np.float32)
            np.testing.assert_array_equal(inputs,accepted['input'])
            np.testing.assert_array_equal(r['input'],accepted['input'][r['time']])
            for j,t in enumerate(r['time']):
                np.testing.assert_array_equal(r['action'][j],accepted['action'][t:t+16].reshape(-1))
            count+=1
    assert count==6*4*40*6
    report=dict(accepted_branches=count,recorded_attempts=attempts,source_attempts=source_episodes,
        current_input_contract=True,train_development_episode_disjoint=True,
        executed_labels_equal_export=True,terminal_success_acceptance=True,gripper_preserved=True,
        attempted_trajectory_hashes_match=True)
    dump(OUT/'audit.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
