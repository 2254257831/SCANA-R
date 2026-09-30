from pathlib import Path
import os
import json,hashlib
import numpy as np
import pandas as pd
W=Path(os.environ.get('SCANA_V46_OUTPUT',Path(__file__).resolve().parents[2]/'artifacts/independent_libraries_v46')).resolve();D=W/'evidence';O=W/'analysis';O.mkdir(exist_ok=True)
P=json.loads((D/'protocol_frozen.json').read_text(encoding='utf-8'))
def dump(name,value):(O/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
def summation():
    if (D/'failure_interventions.csv').exists():
        a=pd.read_csv(D/'failure_interventions.csv');assert len(a)==3600
        metrics=['success','terminal_success','action_step_rms','prediction_noise_rms']+[f'reach_reward_{s}' for s in range(1,5)]
        g=a.groupby(['task','method','condition'])[metrics].mean().reset_index();g.to_csv(O/'failure_summary.csv',index=False)
        # Avoid treating 100 episodes as independent policy fits.
        rows=[];rng=np.random.default_rng(2026092902);ss=rng.integers(5,size=(5000,5));ll=rng.integers(20,size=(5000,20))
        for (task,method),group in a.groupby(['task','method']):
            mats={c:g0.pivot(index='seed',columns='layout',values='success').loc[P['diagnostics']['seeds'],P['diagnostics']['layouts']].to_numpy() for c,g0 in group.groupby('condition')}
            for treatment,control in [('noise_supported','observation_noise'),('noise_joints','nominal'),('noise_objects','nominal'),('noise_low_support','nominal'),('replan_1_ensemble','replan_1'),('replan_1','nominal')]:
                diff=mats[treatment]-mats[control];b=diff[ss[:,:,None],ll[:,None,:]].mean((1,2));lo,hi=np.quantile(b,[.025,.975])
                rows.append(dict(task=task,method=method,treatment=treatment,control=control,difference=diff.mean(),low=lo,high=hi))
        pd.DataFrame(rows).to_csv(O/'failure_contrasts.csv',index=False)
        masks=[]
        for task in P['tasks']:
            f=next((D/'failure_interventions'/task).glob('*/Calibrated_recovery/observation_noise/*.npz'));z=np.load(f)
            masks.append(dict(task=task,channels=np.flatnonzero(z['low_support_mask']).tolist(),noise_in_training_sd=z['standardized_noise_sigma'].tolist()))
        dump('noise_channels.json',masks)
        print(g[['task','method','condition','success','reach_reward_1','reach_reward_2','reach_reward_3','action_step_rms']].to_string(index=False))
    if not (D/'replication_per_episode.csv').exists():return
    a=pd.read_csv(D/'replication_per_episode.csv');assert len(a)==4200 and not a.duplicated(['task','bank','seed','method','layout']).any()
    shape=(len(P['banks']),len(P['policy_seeds']),len(P['test_layouts']));assert shape==(5,3,20)
    rng=np.random.default_rng(2026092901);bb=rng.integers(5,size=(10000,5));ss=rng.integers(3,size=(10000,5,3));ll=rng.integers(20,size=(10000,20))
    def bootstrap(x):return x[bb[:,:,None,None],ss[:,:,:,None],ll[:,None,None,:]].mean((1,2,3))
    summary=[];contrasts=[];bankrows=[];costs=[]
    for task in P['tasks']:
        arrays={}
        for method in P['methods']:
            sub=a[(a.task==task)&(a.method==method)]
            idx=pd.MultiIndex.from_product([P['banks'],P['policy_seeds'],P['test_layouts']],names=['bank','seed','layout'])
            z=sub.set_index(['bank','seed','layout']).reindex(idx).success.to_numpy().reshape(shape);assert np.isfinite(z).all();arrays[method]=z
            b=bootstrap(z);lo,hi=np.quantile(b,[.025,.975])
            summary.append(dict(task=task,method=method,successes=int(z.sum()),episodes=z.size,mean=z.mean(),low=lo,high=hi,bank_sd=z.mean((1,2)).std(ddof=1)))
            for i,bank in enumerate(P['banks']):
                bankrows.append(dict(task=task,bank=bank,method=method,success=z[i].mean()))
                path=D/'replication'/task/str(bank)
                mode='Gaussian recovery' if method=='Gaussian CPU budget' else method
                lib=json.loads((path/'libraries'/(mode.replace(' ','_')+'.json')).read_text()) if method!='Clean repeat' else dict(simulation_cpu=0,simulation_steps=0,attempts=0,identity=0,mean_scale=0,nonzero_windows=0)
                cal=np.load(path/'calibration.npz')
                aux=0 if method in ['Clean repeat','Gaussian recovery','Gaussian CPU budget'] else float(cal['insample_cpu'] if method=='No cross-fitting' else cal['crossfit_cpu'])
                for seed in P['policy_seeds']:
                    ck=json.loads((path/'policies'/str(seed)/(method.replace(' ','_')+'.json')).read_text())
                    costs.append(dict(task=task,bank=bank,seed=seed,method=method,auxiliary_cpu=aux,collection_cpu=lib['simulation_cpu'],final_training_cpu=ck['cpu_seconds'],total_cpu=aux+lib['simulation_cpu']+ck['cpu_seconds'],
                        updates=ck['actual_steps'],best_step=ck['best_step'],simulation_steps=lib['simulation_steps'],attempts=lib['attempts'],identity=lib['identity'],mean_scale=lib['mean_scale'],nonzero_windows=lib['nonzero_windows']))
        for reference in [m for m in P['methods'] if m!='SCANA-R']:
            diff=arrays['SCANA-R']-arrays[reference];b=bootstrap(diff);lo,hi=np.quantile(b,[.025,.975]);alo,ahi=np.quantile(b,[.0125,.9875])
            contrasts.append(dict(task=task,method='SCANA-R',reference=reference,difference=diff.mean(),low=lo,high=hi,family_low=alo,family_high=ahi,
                positive_banks=int(sum(diff.mean((1,2))>0)),bank_differences=diff.mean((1,2)).tolist()))
    pd.DataFrame(summary).to_csv(O/'replication_summary.csv',index=False);pd.DataFrame(contrasts).to_csv(O/'replication_contrasts.csv',index=False)
    pd.DataFrame(bankrows).to_csv(O/'bank_success.csv',index=False);c=pd.DataFrame(costs);c.to_csv(O/'cost_per_run.csv',index=False)
    c.groupby(['task','method']).mean(numeric_only=True).reset_index().to_csv(O/'cost_summary.csv',index=False)
    # CPU-budget slack reflects stopping only at a development checkpoint.
    ratios=[]
    for task in P['tasks']:
        z=c[c.task==task].pivot(index=['bank','seed'],columns='method',values='total_cpu')
        ratios.append(dict(task=task,mean_ratio=float((z['Gaussian CPU budget']/z['SCANA-R']).mean()),max_ratio=float((z['Gaussian CPU budget']/z['SCANA-R']).max())))
    dump('statistics_audit.json',dict(primary_family='two tasks, SCANA-R vs Gaussian recovery',bootstrap=10000,dimensions=shape,sampling='resample banks, resample policy seeds within each selected bank, resample common layouts',
        source_episodes_fixed=True,test_layouts_disjoint_from_prior=[92000,92039,94000,94019],diagnostic_layouts=P['diagnostics']['layouts'],replication_layouts=P['test_layouts'],
        independently_fitted_libraries=10,policy_fits=210,test_episodes=4200,diagnostic_episodes=3600,cpu_budget_ratios=ratios,protocol_sha256=hashlib.sha256((D/'protocol_frozen.json').read_bytes()).hexdigest()))
    print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(contrasts).drop(columns='bank_differences').to_string(index=False))
if __name__=='__main__':summation()
