"""Recompute extension statistics using compact records only; never refit policies."""
from pathlib import Path
import importlib.util, json, shutil
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/recomputed-extensions'
spec=importlib.util.spec_from_file_location('frozen_extension_analysis',ROOT/'experiments/metaworld_extension/analyze.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
analysis.ROOT=OUT
for name in ['act_extension_v1','metaworld_extension_v1']:
    dest=OUT/'artifacts'/name;dest.mkdir(parents=True,exist_ok=True)
    for filename in ['per_episode.csv','protocol_frozen.json']:
        shutil.copy2(ROOT/'results'/name/filename,dest/filename)
analysis.analyze('act_extension_v1',['transfer_cube','insertion'],
    ['Clean repeat','Gaussian recovery','Frozen context','Calibrated recovery'],
    ['nominal','replan_1','replan_4','replan_16','pulse_early','pulse_late','observation_noise'],94000,'Calibrated recovery')
tasks=['reach-v3','push-v3','pick-place-v3','door-open-v3','drawer-open-v3','button-press-v3']
analysis.analyze('metaworld_extension_v1',tasks,
    ['Clean repeat','Gaussian recovery','Frozen observations','SCANA-R','Demonstration kNN'],['nominal','pulse'],30000,'SCANA-R')
for name in ['act_extension_v1','metaworld_extension_v1']:
    for filename in ['summary.csv','paired_contrasts.csv','per_seed.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(ROOT/'results'/name/filename),pd.read_csv(OUT/'results'/name/filename))
print('All recomputed statistics match the checked-in evidence. Output:',OUT)
