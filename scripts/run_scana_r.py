"""Portable entry point for the frozen state-based simulation experiment."""
from pathlib import Path
import argparse, importlib.metadata, os, runpy, sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments/scana_r')]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['collect','calibrate','develop','evaluate','audit','replay-check'])
    ap.add_argument('--artifacts',type=Path,default=ROOT/'artifacts')
    ap.add_argument('--main-only',action='store_true',help='Evaluate the four matched current methods only')
    args=ap.parse_args()
    if sys.version_info[:2]!=(3,10):raise SystemExit('The frozen simulator requires Python 3.10.')
    for pkg,version in [('mujoco','2.3.3'),('dm-control','1.0.9')]:
        if importlib.metadata.version(pkg)!=version:raise SystemExit(f'Install {pkg}=={version}; physics versions cannot be substituted silently.')
    os.environ['SCANA_ARTIFACTS']=str(args.artifacts.resolve())
    from experiment import crossfit
    if args.stage=='collect':
        from collect import main as collect
        collect()
    elif args.stage=='calibrate':
        for task in ['transfer_cube','insertion']:crossfit(task)
    elif args.stage=='develop':
        from recovery import development
        development([.03,.08],[7,11,23])
    elif args.stage=='evaluate':
        from final_evaluate import main as evaluate
        evaluate(main_only=args.main_only)
    else:
        file='audit_recovery.py' if args.stage=='audit' else 'replay_check.py'
        runpy.run_path(str(ROOT/'experiments/scana_r'/file),run_name='__main__')

if __name__=='__main__':main()
