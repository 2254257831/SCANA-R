from common import *
import pandas as pd
sim=W/'simulation';df=pd.read_csv(sim/'closed_loop_per_episode.csv');rows=[];paired=[]
rng=np.random.default_rng(92018)
for (task,method),g in df.groupby(['task','method']):
    matrix=g.pivot(index='seed',columns='layout',values='success').sort_index().sort_index(axis=1).to_numpy()
    boots=[]
    for _ in range(3000):
        si=rng.integers(5,size=5);li=rng.integers(20,size=20);boots.append(matrix[si][:,li].mean())
    lo,hi=np.quantile(boots,[.025,.975]);sr=matrix.mean(1)
    rows.append(dict(task=task,method=method,successes=int(matrix.sum()),rollouts=matrix.size,mean=float(matrix.mean()),seed_sd=float(sr.std(ddof=1)),bootstrap_low=float(lo),bootstrap_high=float(hi),degenerate=bool(matrix.max()==matrix.min()),interpretation='Two-way seed/layout empirical bootstrap; a degenerate zero interval is not a zero population-success bound.'))
    if method=='SCANA':
        for baseline in ['Clean repeat','Clip only','Conditional kNN','Deterministic MLP']:
            other=df[(df.task==task)&(df.method==baseline)].pivot(index='seed',columns='layout',values='success').sort_index().sort_index(axis=1).to_numpy()
            delta=matrix-other;boots=[]
            for _ in range(3000):
                si=rng.integers(5,size=5);li=rng.integers(20,size=20);boots.append(delta[si][:,li].mean())
            lo,hi=np.quantile(boots,[.025,.975]);paired.append(dict(task=task,contrast='SCANA - '+baseline,difference=float(delta.mean()),bootstrap_low=float(lo),bootstrap_high=float(hi)))
savecsv(sim/'closed_loop_summary.csv',rows);savecsv(sim/'paired_success_contrasts.csv',paired)
resources=[]
for p in (sim/'policies').rglob('*.json'):
    d=json.loads(p.read_text());resources.append(dict(task=p.parts[-3],seed=p.parts[-2],method=p.stem,seconds=d['seconds'],parameters=d['parameters'],bytes=p.with_suffix('.pt').stat().st_size,best_step=d['best_step'],selection_mse=d['selection_mse']))
savecsv(sim/'policy_resources.csv',resources)
dump(sim/'statistical_notes.json',dict(independent_policy_trainings_per_method=5,shared_test_layouts=20,rollouts_per_method_per_task=100,all_zero_upper95_fixed_five_policy_ensemble_new_layout=1-.05**(1/20),upper_bound_interpretation='When no layout succeeds under any of the five trained policies, a one-sided exact bound concerns the probability that at least one fixed policy succeeds on a new iid layout. It is not a bound for a new policy training seed.',zero_bootstrap_warning=True,protocol_sha256=sha(sim/'protocol_frozen.json')))
print(json.dumps(rows,ensure_ascii=False,indent=2))
