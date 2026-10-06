"""Audit completed LIBERO records without changing the running experiment."""
from pathlib import Path
import json,hashlib,collections,time
import numpy as np

ROOT=Path(__file__).resolve().parent

def main():
    cfg=json.loads((ROOT/'protocol.json').read_text())
    split=json.loads((ROOT/'protocol/source_splits.json').read_text())
    labels=dict(np.load(ROOT/'cache/labels.npz'))
    for task,part in split.items():
        assert len(part['train'])==40 and len(part['dev'])==10
        assert not set(part['train']) & set(part['dev'])
    expected=np.flatnonzero(labels['train'])
    calibration=[]
    for p in sorted((ROOT/'calibration').glob('*/summary.json')):
        bank=dict(np.load(p.parent/'bank.npz'))
        assert np.array_equal(np.sort(bank['source_row']),expected)
        assert len(np.unique(bank['source_row']))==len(expected)
        assert np.array_equal(bank['source_episode'],labels['episode'][bank['source_row']])
        assert np.isfinite(bank['errors']).all() and bank['errors'].shape==(len(expected),8,7)
        calibration.append({'library':p.parent.name,'unique_out_of_fold_rows':len(expected)})
    zero_reference={};zero_count=0;zero_failures=set()
    for summary in sorted((ROOT/'recovery').glob('*/summary.json')):
        folder=summary.parent
        for r in json.loads((folder/'attempts.json').read_text()):
            if r['scale']!=0:continue
            p=folder/'traces'/f"task{r['task']}_{r['episode']}_branch{r['branch']}_scale{r['scale']}.npz"
            d=dict(np.load(p));key=(r['task'],r['episode']);zero_count+=1
            if not r['terminal_success']:zero_failures.add(key)
            if key in zero_reference:
                assert all(np.array_equal(zero_reference[key][k],d[k]) for k in ['states','actions','rewards']),f'Nonrepeatable identity fallback: {key}'
            else:zero_reference[key]=d
    zero_audit={'completed_libraries':len(list((ROOT/'recovery').glob('*/summary.json'))),'zero_scale_attempts_checked':zero_count,
                'unique_sources_reached_by_fallback':len(zero_reference),'sources_with_failed_zero_scale_replay':len(zero_failures),
                'exact_state_action_reward_consistency':True,'failed_sources':[list(k) for k in sorted(zero_failures)],
                'note':'Sources reaching zero scale were conditionally selected by earlier failed perturbation attempts. This is not an unconditional source replay failure rate. Zero scale does not guarantee terminal success in this native replay adaptation; failed branches are excluded.'}
    (ROOT/'analysis/identity_fallback_consistency.json').write_text(json.dumps(zero_audit,indent=2),encoding='utf8')
    initial={};initial_extra={};rows=[];hashes={};partial=[];margins=[]
    for p in sorted((ROOT/'evaluation').glob('*/test_results.json')):
        data=json.loads(p.read_text())
        if len(data)!=len(cfg['tasks'])*len(cfg['test_initial_states']):
            partial.append({'name':p.parent.name,'rows':len(data)});continue
        method=p.parent.name.split('_lib')[0];lib=int(p.parent.name.split('_lib')[1].split('_')[0])
        assert {(x['task'],x['init_state']) for x in data}=={(t,i) for t in cfg['tasks'] for i in cfg['test_initial_states']}
        hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        for r in data:
            trace=p.parent/f"test_task{r['task']}_init{r['init_state']}.npz"
            d=dict(np.load(trace));n=r['steps']
            assert n==len(d['states'])==len(d['actions'])==len(d['rewards'])==len(d['success_flags'])
            assert d['actions'].shape==(n,7)
            assert np.isfinite(d['states']).all() and np.isfinite(d['actions']).all()
            assert np.max(np.abs(d['actions']))<=1.000001
            assert bool(np.any(d['success_flags']))==bool(r['success'])
            assert np.array_equal(d['rewards']>0,d['success_flags'])
            if r['success']:assert d['success_flags'][-1] and not np.any(d['success_flags'][:-1])
            else:assert n==cfg['max_control_steps']
            key=(r['task'],r['init_state'])
            if key in initial:assert np.array_equal(initial[key],d['states'][0]),f'Unpaired start {p.parent.name} {key}'
            else:initial[key]=d['states'][0].copy()
            fields=['model_body_pos','model_body_quat','initial_proprio','initial_camera_sha256']
            if key in initial_extra:
                assert all(np.array_equal(initial_extra[key][f],d[f]) for f in fields), f'Unpaired fixture/observation: {p.parent.name} {key}'
            else:initial_extra[key]={f:d[f].copy() for f in fields}
            hashes[str(trace.relative_to(ROOT))]=hashlib.sha256(trace.read_bytes()).hexdigest()
            rows.append({**r,'method':method,'library_seed':lib})
    lookup={(r['method'],r['library_seed'],r['task'],r['init_state']):r for r in rows}
    for lib in cfg['collection_seeds']:
        for baseline in ['clean_repeat','gaussian_recovery','gaussian_same_total_cost']:
            for task in cfg['tasks']:
                pairs=[]
                for init in cfg['test_initial_states']:
                    a=lookup.get(('scana_r',lib,task,init));b=lookup.get((baseline,lib,task,init))
                    if a and b:pairs.append((a['success'],b['success']))
                if len(pairs)==len(cfg['test_initial_states']):
                    margins.append({'library_seed':lib,'task':task,'baseline':baseline,
                                    'both_success':sum(a and b for a,b in pairs),'both_failure':sum(not a and not b for a,b in pairs),
                                    'SCANA_only_success':sum(a and not b for a,b in pairs),'baseline_only_success':sum(b and not a for a,b in pairs)})
    result={'checked_at':time.strftime('%Y-%m-%d %H:%M:%S'),'complete_model_evaluations':len(rows)//(len(cfg['tasks'])*len(cfg['test_initial_states'])),
            'verified_test_episodes':len(rows),'unique_paired_initial_states':len(initial),
            'scope':'Exact initial simulator state, fixed-body poses, proprioception and RGB hash across methods AND repetitions; actions finite/in native bounds; episode lengths, official rewards and success flags agree. Does not independently prove generalization or statistical superiority.',
            'calibration':calibration,'identity_fallback_audit':zero_audit,'partial_evaluations_omitted':partial,'paired_outcome_counts':margins,'sha256':hashes}
    path=ROOT/'analysis'/f'evidence_audit_{len(rows)}.json'
    path.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['sha256','paired_outcome_counts']},indent=2))
    print('Saved',path)
    return result

if __name__=='__main__':main()
