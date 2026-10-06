"""Verify all ten source files, causal cache indexing and content split uniqueness."""
from runtime import *
import hashlib, h5py, numpy as np


def main():
    cfg = json.loads((ROOT / 'protocol.json').read_text())
    splits = json.loads((ROOT / 'protocol/source_splits.json').read_text())
    meta = json.loads((ROOT / 'cache/metadata.json').read_text())
    labels = dict(np.load(ROOT / 'cache/labels.npz'))
    timing = json.loads((ROOT / 'preflight/timing_audit.json').read_text())
    assert len(timing['rows']) == 500 and timing['max_shifted_error'] == 0
    assert set(meta['token_ids']) == set(map(str, range(10)))
    seen = {}; records = []
    for tid in cfg['tasks']:
        assert len(splits[str(tid)]['train']) == 40 and len(splits[str(tid)]['dev']) == 10
        assert not set(splits[str(tid)]['train']) & set(splits[str(tid)]['dev'])
        with h5py.File(data_file(tid), 'r') as file:
            assert len(file['data']) == 50
            for e in (x for x in meta['episodes'] if x['task'] == tid):
                group = file['data'][e['key']]
                acts = group['actions'][:]; states = group['states'][:]
                sha = hashlib.sha256(acts.tobytes() + states.tobytes()).hexdigest()
                assert sha not in seen, f'Duplicate action/state source content: {tid}, {e["key"]}, {seen.get(sha)}'
                seen[sha] = [tid, e['key']]
                sl = slice(e['start'], e['stop']); n = e['stop'] - e['start']
                expected = np.stack([acts[t:t+cfg['horizon']] for t in range(1, n+1)]).astype('float32')
                assert np.array_equal(labels['actions'][sl], expected)
                expected_prop = np.concatenate([group['obs/joint_states'][:n], group['obs/ee_states'][:n], group['obs/gripper_states'][:n]], axis=1).astype('float32')
                assert np.array_equal(labels['proprio'][sl], expected_prop)
                assert np.all(labels['train'][sl] == (e['split'] == 'train'))
                assert np.all(labels['fold'][sl] == e['fold'])
                records.append({'task': tid, 'episode': e['key'], 'split': e['split'], 'content_sha256': sha, 'windows': n})
    assert len(records) == 500
    result = {'verified_source_episodes': len(records), 'unique_source_content': len(seen),
              'train_windows': int(labels['train'].sum()), 'development_windows': int((~labels['train']).sum()),
              'causal_alignment': 'obs[t-1] -> actions[t:t+8], all cached rows exact', 'records': records}
    write_json('analysis/source_integrity.json', result)
    write_json('task_map.json', {str(t): {'name': task_suite().get_task(t).name,
               'language': task_suite().get_task(t).language} for t in cfg['tasks']})
    print({k: v for k, v in result.items() if k != 'records'}, flush=True)


if __name__ == '__main__':
    main()
