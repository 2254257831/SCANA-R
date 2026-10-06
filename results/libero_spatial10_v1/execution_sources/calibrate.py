"""Five episode-held-out visuomotor models and an auditable error bank."""
from learner import *
import hashlib

def main(seed):
    started=time.perf_counter();existing_training=0.;d=Data();bank=[];ids_all=[];costs=[]
    for fold in range(CFG['crossfit_folds']):
        name=f'aux_lib{seed}_fold{fold}';path=ROOT/'models'/name/'latest.pt'
        existed=path.exists()
        if existed:
            prior=torch.load(path,map_location='cpu')
            existed=prior['steps']==CFG['auxiliary_steps']
            if not existed: raise RuntimeError(f'Incomplete auxiliary checkpoint: {name}; preserve and resolve before resume')
            del prior
        if not existed:train(name,CFG['auxiliary_steps'],seed+fold,fold)
        m,ck=load(path)
        if existed:existing_training+=ck['elapsed_seconds']
        ids=np.flatnonzero(d.arr['train'] & (d.arr['fold']==fold))
        heldout=set(d.arr['episode'][ids].tolist())
        assert not heldout.intersection(ck['train_episodes']), 'episode leakage into auxiliary fitting'
        errors=[]
        with torch.no_grad():
            for ix in np.array_split(ids,max(1,int(np.ceil(len(ids)/128)))):
                x,p,t,y=d.batch(ix,ck['norm'])
                pred=np.clip(m(x,p,t).cpu().numpy()*ck['norm']['astd']+ck['norm']['amean'],-1,1)
                errors.append(pred-d.arr['actions'][ix])
        bank.append(np.concatenate(errors));ids_all.append(ids)
        costs.append({'fold':fold,'seed':seed+fold,'seconds':ck['elapsed_seconds'],'heldout_episodes':sorted(heldout),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        del m;torch.cuda.empty_cache()
    ids=np.concatenate(ids_all);errors=np.concatenate(bank)
    out=ROOT/'calibration'/f'lib{seed}';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'bank.npz',errors=errors,task=d.arr['task'][ids],phase=d.arr['phase'][ids],source_row=ids,source_episode=d.arr['episode'][ids])
    write_json(f'calibration/lib{seed}/costs.json',costs)
    write_json(f'calibration/lib{seed}/summary.json',{'rows':len(errors),'auxiliary_training_seconds':sum(x['seconds'] for x in costs),
                'calibration_active_seconds':existing_training+time.perf_counter()-started,
                'previously_fitted_models_seconds':existing_training,'note':'Same local hardware; active wall time, not energy or GPU-seconds. Existing checkpoints add their recorded fitting time.'})
    print('bank saved',len(errors),'cost_seconds',sum(x['seconds'] for x in costs),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=101);main(ap.parse_args().seed)
