from common import *
SRC=ROOT/'revision_work/experiment_revision_v24_20260907/rerun_outputs'
OLD=ROOT/'revision_work/full_repair_v34_20260918/evidence'
cfg=load_config(SRC/'scana_config.json')
dt0=load_chunks_npz(SRC/'dt_success_chunks_train.npz');real0=load_chunks_npz(SRC/'real_success_chunks_train.npz')
dt_test=load_chunks_npz(SRC/'dt_success_chunks_eval.npz');real_test=load_chunks_npz(SRC/'real_success_chunks_eval.npz')
dt_train,dt_val=split(dt0,91001);real_train,real_val=split(real0,91002)
train=build_calibrated_pairs(dt_train,real_train,cfg);val=build_calibrated_pairs(dt_val,real_val,cfg);test=build_calibrated_pairs(dt_test,real_test,cfg)
gamma=estimate_rbf_gamma(train.target_delta,seed=91003);con=estimate_constraints(dt_train+real_train,cfg,train.target_delta)
dump(OUT/'local_protocol.json',dict(seeds=SEEDS,pairs=[len(train.action),len(val.action),len(test.action)],gamma=gamma,scope='Retrospective offline diagnostic of historical cross-task data. Separate train/selection/evaluation episodes; no robot deployment claim.',ablation='No-MMD retrains and selects using its own declared loss. z=0, mean and condition shuffling are inference interventions.',stationary_threshold=1e-6))
rows=[];erows=[];div=[];sens=[]
for seed in SEEDS:
    models={}
    for name,slug in [('SCANA','scana'),('Deterministic MLP','deterministic'),('No MMD','no_mmd')]:
        models[name]=model_get(cfg,train,val,name,seed,OUT/'local_models'/str(seed)/slug,OLD/'models'/str(seed)/slug/'noise_generator.pt')
    ss=candidates(train,test.action,test.condition,models,seed)
    rr,ee=evaluate(ss,test,train,con,gamma,seed,dt_test,real_test);rows+=rr;erows+=ee
    sc=ss['SCANA'];within=float(sc.std(0).mean());glob=float(sc.reshape(-1,96).std(0).mean())
    div.append(dict(seed=seed,within_std=within,global_std=glob,ratio=within/glob))
    for name in ['Conditional kNN','Empirical chunks','Deterministic MLP','SCANA']:
        for scale in [.5,1,2]:
            vals=[weighted_mmd(batch_map(test.action,d,con)[1],test.target_delta,gamma*scale) for d in ss[name]]
            sens.append(dict(seed=seed,method=name,gamma_factor=scale,mmd2=float(np.mean(vals))))
    np.savez_compressed(OUT/f'local_candidates_{seed}.npz',**{n.replace(' ','_'):v for n,v in ss.items()},action=test.action,target=test.target_delta,anchor=test.matched_dt_index,real_index=test.real_index)
    savecsv(OUT/'local_per_copy.csv',rows);savecsv(OUT/'local_per_episode.csv',erows)
    savecsv(OUT/'local_summary.csv',aggregate(rows,['method','weighting']));savecsv(OUT/'local_diversity.csv',div);savecsv(OUT/'bandwidth_sensitivity.csv',sens)
    print(f'Local seed {seed} complete',flush=True)
dump(OUT/'local_complete.json',dict(complete=True,seeds=SEEDS))
