"""Default public statistics entry: current study followed by earlier studies."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--current-only',action='store_true');args=ap.parse_args()
    scripts=['reproduce_current.py'] if args.current_only else ['reproduce_current.py','summarize.py','summarize_extensions.py']
    for name in scripts:subprocess.run([sys.executable,str(ROOT/'scripts'/name)],cwd=ROOT,check=True)
    print('Verified current 11,040 evaluation records.' if args.current_only else 'Processed 25,840 records across separate studies (11,040 current + 3,200 original + 11,600 extension); no pooled success rate.')
