from runtime import *
import h5py,numpy as np,cv2,re,time

def main():
    cfg=json.loads((ROOT/'protocol.json').read_text());splits=json.loads((ROOT/'protocol/source_splits.json').read_text())
    out=ROOT/'cache';out.mkdir(exist_ok=True)
    H=cfg['horizon'];size=cfg['image_size']; episodes=[];N=0
    for tid in cfg['tasks']:
        with h5py.File(data_file(tid),'r') as h:
            for key in sorted(h['data'],key=lambda x:int(x.split('_')[1])):
                L=len(h['data'][key]['actions']);n=L-H
                episodes.append({'task':tid,'key':key,'length':L,'start':N,'stop':N+n,'split':'train' if key in splits[str(tid)]['train'] else 'dev',
                                 'fold':splits[str(tid)]['train'].index(key)%5 if key in splits[str(tid)]['train'] else -1})
                N+=n
    images=np.lib.format.open_memmap(out/'images.npy',mode='w+',dtype='uint8',shape=(N,2,3,size,size))
    prop=np.empty((N,15),np.float32);actions=np.empty((N,H,7),np.float32);tids=np.empty(N,np.int64)
    phase=np.empty(N,np.float32);fold=np.empty(N,np.int64);episode_id=np.empty(N,np.int64);train=np.empty(N,bool)
    for eid,e in enumerate(episodes):
        with h5py.File(data_file(e['task']),'r') as h:
            g=h['data'][e['key']];a=g['actions'][:];p=np.concatenate([g['obs/joint_states'][:],g['obs/ee_states'][:],g['obs/gripper_states'][:]],axis=1)
            for k,name in enumerate(['agentview_rgb','eye_in_hand_rgb']):
                raw=g['obs'][name][:(e['length']-H)]
                for j,frame in enumerate(raw):images[e['start']+j,k]=cv2.resize(frame,(size,size),interpolation=cv2.INTER_AREA).transpose(2,0,1)
            sl=slice(e['start'],e['stop']);n=e['stop']-e['start']
            prop[sl]=p[:n];actions[sl]=np.stack([a[t:t+H] for t in range(1,n+1)])
            tids[sl]=e['task'];phase[sl]=np.arange(1,n+1)/len(a);fold[sl]=e['fold'];episode_id[sl]=eid;train[sl]=e['split']=='train'
        if eid%25==0:print('cached',eid,'/',len(episodes),flush=True)
    images.flush()
    np.savez(out/'labels.npz',proprio=prop,actions=actions,task=tids,phase=phase,fold=fold,episode=episode_id,train=train)
    task=task_suite();texts={str(i):task.get_task(i).language for i in cfg['tasks']}
    vocab={w:i+1 for i,w in enumerate(sorted(set(' '.join(texts.values()).split())))}
    token_ids={k:[vocab[w] for w in v.split()] for k,v in texts.items()}
    write_json('cache/metadata.json',{'episodes':episodes,'rows':N,'texts':texts,'vocabulary':vocab,'token_ids':token_ids,'timing':'pre-action aligned by shifting observations back one row'})
    print('dataset ready',N,'training',int(train.sum()),'development',int((~train).sum()),flush=True)

if __name__=='__main__':main()
