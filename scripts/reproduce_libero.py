"""Recompute the complete ten-task Spatial study without a simulator or GPU."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    output=ROOT/'outputs/recomputed-libero';output.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(ROOT/'results/libero_spatial10_v1/recompute_statistics.py'),'--output',str(output/'statistics.json')],check=True,cwd=ROOT)
