from pathlib import Path
import os, sys, json, csv, hashlib, time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
ARTIFACTS=Path(os.environ.get('SCANA_ARTIFACTS',ROOT/'artifacts')).resolve()
OLD=ARTIFACTS/'legacy'
WORK=ARTIFACTS/'scana_r'
SOURCE_DIR=Path(__file__).resolve().parent
ACT=Path(os.environ.get('SCANA_ACT_ROOT',ROOT/'third_party/act')).resolve()
torch.set_num_threads(4)
SEEDS=[7,11,23,31,47]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def savecsv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
