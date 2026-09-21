from common import *
from run_public import prepare
from scana_experiments.models import _chunk_descriptor

def diagnostics(task,dt,ref,train,evaldt,evalref,cfg):
    rows=[]
    for name,ds,rs in [('train',dt,ref),('evaluation',evaldt,evalref)]:
        pairs=build_calibrated_pairs(ds,rs,cfg);cnt=np.bincount(pairs.matched_dt_index,minlength=len(ds))
        rd=_chunk_descriptor(rs);dd=_chunk_descriptor(ds)
        phase_gap=np.abs(np.stack([r.condition for r in rs])[:,0]-pairs.condition[:,0])
        # Phase is a normalized-time proxy, not a contact/semantic stage label.
        constrained=[];missing=0
        for i,r in enumerate(rs):
            possible=np.array([(c.task==r.task and bool(c.task) and abs(c.condition[0]-r.condition[0])<=.1) for c in ds])
            if not possible.any():missing+=1;continue
            constrained.append(float(np.sqrt(((dd[possible]-rd[i])**2).sum(1).min())))
        rows.append(dict(task=task,split=name,queries=len(rs),pool=len(ds),anchors=int((cnt>0).sum()),coverage=float((cnt>0).mean()),max_reuse=int(cnt.max()),top10_mass=float(np.sort(cnt)[-10:].sum()/cnt.sum()),phase_gap_median=float(np.median(phase_gap)),phase_gap_gt01=float((phase_gap>.1).mean()),strict_task_phase_unmatched=missing,strict_task_phase_matches=len(constrained),strict_match_distance_median=float(np.median(constrained)) if constrained else None))
    tr=build_calibrated_pairs(dt,ref,cfg);anchors=np.unique(tr.matched_dt_index)
    anchor_desc=_chunk_descriptor([dt[i] for i in anchors]);qd=_chunk_descriptor(evaldt)
    d=np.sqrt(((qd[:,None]-anchor_desc[None])**2).sum(2).min(1))
    train_pool_dist=np.sqrt(((_chunk_descriptor(dt)[:,None]-anchor_desc[None])**2).sum(2).min(1))
    # Training-derived threshold is diagnostic only; not retroactively used in main results.
    threshold=float(np.quantile(train_pool_dist,.95))
    bins=np.quantile(train_pool_dist,[.5,.9,.95]);bins=np.unique(bins)
    support=[]
    for i in range(len(bins)+1):
        mask=np.searchsorted(bins,d,side='right')==i
        support.append(dict(task=task,bin=i,eval_centers=int(mask.sum()),median_distance=float(np.median(d[mask])) if mask.any() else None,threshold=threshold,train_edges=bins.tolist()))
    return rows,support

src=ROOT/'revision_work/experiment_revision_v24_20260907/rerun_outputs';cfg=load_config(src/'scana_config.json')
dt=load_chunks_npz(src/'dt_success_chunks_train.npz');rr=load_chunks_npz(src/'real_success_chunks_train.npz');de=load_chunks_npz(src/'dt_success_chunks_eval.npz');re=load_chunks_npz(src/'real_success_chunks_eval.npz')
rows,support=diagnostics('Historical cross-task local',dt,rr,None,de,re,cfg)
for task in ['Transfer Cube','Insertion']:
    cfg,g,_,_=prepare(task);dr,dv,de,hr,hv,he=g
    a,b=diagnostics(task,dr,hr,None,de,he,cfg);rows+=a;support+=b
savecsv(OUT/'coverage_phase.csv',rows);dump(OUT/'all_pool_support.json',support)
print(json.dumps(rows,ensure_ascii=False,indent=2))
