"""Regenerate the original successful reference episodes with ACT scripts."""
from sim_bridge import *

def main():
    for task,offset in [('transfer_cube',61000),('insertion',71000)]:
        folder=OLD/'demonstrations'/task;folder.mkdir(parents=True,exist_ok=True)
        audit=[];successes=0
        for seed in range(offset,offset+1000):
            p=folder/f'{seed}.npz'
            if p.exists():r=dict(np.load(p))
            else:
                r=collect(task,seed);np.savez_compressed(p,**r)
            successes+=int(r['success'])
            audit.append(dict(seed=seed,success=int(r['success']),max_reward=float(r['rewards'].max()),file=p.name,sha256=sha(p)))
            if successes==50:break
        if successes!=50:raise RuntimeError('Insufficient successful demonstrations')
        dump(folder/'collection_manifest.json',audit)
        print(task,successes,'successful references;',len(audit),'collection attempts',flush=True)

if __name__=='__main__':main()
