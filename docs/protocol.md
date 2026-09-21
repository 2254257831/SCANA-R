# Frozen simulation protocol

| Item | Value |
|---|---|
| Tasks | ACT Transfer Cube and Insertion; bimanual simulation |
| Action | 14 absolute position commands; 12 joints in radians; grippers 6, 13 normalized |
| Observation | Current 14 qpos, object xyz (3 / 6), elapsed step / 400; 18 / 21 inputs |
| Episode | 400 steps, 50 Hz, 8 seconds |
| Reference split | 40 training and 10 development successful episodes per task |
| Windows | H = 16, stride 8; 1,960 train / 490 development |
| Calibration | Five source-episode folds; eight held-out sources per fold |
| Auxiliary selection | Minimum fitting-fold training MSE, checked every 100 steps; held-out fold excluded |
| Policy | d → 256 LayerNorm SiLU → 256 LayerNorm SiLU → 224 |
| Optimizer | AdamW, lr 0.001, weight decay 0.0001, batch 128, 1,200 steps |
| Pulse | 8 steps; same-task bank within ±8 steps of pulse start; no centering or random sign |
| Amplitude | Joint RMS σ, coordinate cap ±3σ; gripper perturbations zero |
| Selected σ | 0.03 Transfer Cube; 0.08 Insertion, for both calibrated and Gaussian families |
| Branches | 12 per source; endpoints stratified over 16, 24, …, 368 |
| Collection acceptance | Last reward = 4 after complete 400-step replay |
| Backtracking | Same pulse scaled by 1, 0.5, 0.25, 0; reset original initial state each attempt |
| Recovery windows | Starts u, u+8, u+16, u+24 while a full 16-step chunk fits |
| Final training mix | 2,048 original + 2,048 method windows, sampled with replacement |
| Final scalers | Fit only on 1,960 original training windows; standard deviation floor 1e−5 |
| Final selection | Minimum 490-window development MSE checked every 100 steps |
| Execution | Predict 16 actions, execute 8, reobserve |
| Test | Seeds 7, 11, 23, 31, 47 × layouts 92000–92039, two tasks |
| Main metric | Maximum episode reward = 4; terminal success reported separately |
| Statistics | 5,000 paired two-way bootstrap replicates; seed 20260919 |

The four current methods share initialization, minibatch stream and resampling
quantiles. Historical controls preserve their original checkpoints/draw streams;
they are version diagnostics rather than single-factor ablations. SCANA-R's
zero-scale branches preserve source coverage but are not new recovery experience.
The calibrated error bank and recovery bank are fixed across final policy seeds.

The frozen JSON files retain original source hashes for provenance. They do not
assert that refactored release filenames have those original hashes. See
`../provenance/source_files.json` for the release mapping.
