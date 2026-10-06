"""Recompute the new Spatial statistics from the included 2400-episode CSV.
No simulator, CUDA, pandas or private path is required. This does not rerun training.
Failure summaries reaggregate derived motion proxies; raw-state validation needs
local trajectories and the pinned simulator, as explained in README.md.
"""
from pathlib import Path
import csv,json,statistics,collections,argparse
HERE=Path(__file__).resolve().parent
METHODS=['clean_repeat','gaussian_recovery','scana_r','gaussian_same_total_cost']
def read(p):return list(csv.DictReader(open(p,encoding='utf-8-sig',newline='')))
def recompute(data):
 rows=read(data/'episodes.csv');assert len(rows)==2400
 keys=[(r['method'],r['library_seed'],r['task'],r['init_state']) for r in rows];assert len(set(keys))==2400
 g=collections.defaultdict(list)
 for r in rows:
  assert r['split']=='test' and int(r['task']) in range(10) and int(r['init_state']) in range(20)
  assert (r['library_seed'],r['policy_seed']) in [('101','7'),('202','11'),('303','23')]
  g[(r['method'],r['library_seed'],r['task'])].append(r['success']=='True')
 assert len(g)==120 and all(len(v)==20 for v in g.values())
 task={k:100*sum(v)/len(v) for k,v in g.items()};libs=['101','202','303'];result={'formal_episodes':2400,'macro':{},'paired':{},'tasks':{}}
 for m in METHODS:
  rates=[statistics.mean(task[(m,l,str(t))] for t in range(10)) for l in libs]
  result['macro'][m]={'mean':statistics.mean(rates),'sd':statistics.stdev(rates),'repeats':rates,'successes':sum(r['success']=='True' for r in rows if r['method']==m)}
  result['tasks'][m]={str(t):{'mean':statistics.mean(task[(m,l,str(t))] for l in libs),'sd':statistics.stdev(task[(m,l,str(t))] for l in libs)} for t in range(10)}
 for m in METHODS:
  if m=='scana_r':continue
  diffs=[a-b for a,b in zip(result['macro']['scana_r']['repeats'],result['macro'][m]['repeats'])]
  result['paired'][m]={'mean_pp':statistics.mean(diffs),'sd_pp':statistics.stdev(diffs),'repeats':diffs}
 # Cross-check archived summary files, avoiding any silent file selection.
 for r in read(data/'macro.csv'):
  x=result['macro'][r['method']];assert abs(x['mean']-float(r['mean']))<1e-10 and abs(x['sd']-float(r['sd']))<1e-10;assert x['successes']==int(r['successes'])
 for r in read(data/'paired.csv'):
  x=result['paired'][r['baseline']];assert abs(x['mean_pp']-float(r['mean_pp']))<1e-10 and abs(x['sd_pp']-float(r['sd_pp']))<1e-10
 proxies=read(data/'failure_kinematics.csv');assert len(proxies)==2400
 stages=collections.Counter((r['task'],r['method'],r['category_0.03']) for r in proxies)
 result['failure_counts']=[{'task':t,'method':m,'category':k,'count':v} for (t,m,k),v in sorted(stages.items())]
 result['cost']=[{'library_seed':r['library_seed'],'relative_deviation_pct':100*(float(r['Gaussian_total_active_seconds'])-float(r['SCANA_total_active_seconds']))/float(r['SCANA_total_active_seconds']),'gaussian_updates':int(r['Gaussian_training_updates'])} for r in read(data/'matched_cost_accounting.csv')]
 result['scope']='Ten Spatial tasks; three paired library/policy repetitions; from-scratch multitask policy; no foundation VLA or other benchmark suites.'
 return result
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=HERE/'data');ap.add_argument('--output',type=Path,default=HERE/'recomputed_statistics.json');a=ap.parse_args();res=recompute(a.data);a.output.write_text(json.dumps(res,indent=2),encoding='utf-8');print(json.dumps({'episodes':res['formal_episodes'],'macro':res['macro'],'output':str(a.output)},ensure_ascii=True))
