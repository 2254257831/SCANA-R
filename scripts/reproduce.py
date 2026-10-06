"""Default public statistics entry: current study followed by earlier studies."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--current-only',action='store_true');args=ap.parse_args()
    scripts=['reproduce_libero.py','reproduce_current.py'] if args.current_only else ['reproduce_libero.py','reproduce_current.py','summarize.py','summarize_extensions.py']
    for name in scripts:subprocess.run([sys.executable,str(ROOT/'scripts'/name)],cwd=ROOT,check=True)
    print('Verified 2,400 LIBERO-Spatial and 11,040 ACT/Meta-World study records separately.' if args.current_only else 'Processed 28,240 records across separate studies (2,400 LIBERO-Spatial + 11,040 library/cost + 3,200 original + 11,600 extension); no pooled success rate.')
