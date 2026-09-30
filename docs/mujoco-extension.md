# Earlier MuJoCo extension across eight task types

This earlier fixed-library study contains **11,600 closed-loop executions**: 5,600 executions of frozen
ACT policies and 6,000 on six additional Meta-World tasks. It uses simulation
only. It measures task-specific action-chunk policies and receding-horizon
control, not language planning, visual VLA transfer or real-robot performance.

The main new finding is limited transfer of the recovery benefit. On the six
Meta-World tasks, SCANA-R reaches **89.5%** nominal macro success versus **91.8%**
for Clean repeat. The paired difference is −2.33 percentage points, 95% interval
[−5.00, 0.17]. New ACT layouts retain gains over Clean repeat, but deployment
stress exposes substantial failures. Negative outcomes remain in all tables.

[Watch the six tasks and fixed-layout policy videos](mujoco-gallery.html).

## Tasks and protocol

Meta-World 3.1.1 / MuJoCo 3.3.0 / Gymnasium 1.1.1: reach-v3, push-v3,
pick-place-v3, door-open-v3, drawer-open-v3 and button-press-v3. Each task has
its own policy. This is a **custom imitation protocol**, not an official MT10,
ML10 or multi-task leaderboard result. Each episode lasts 250 steps (3.125 s),
shorter than the common 500-step benchmark horizon. Official task descriptions
and success functions are retained.

For each task, take the first 40 terminal-success training references scanning
seeds 10000–10999 and the first 10 development references scanning 20000–20999.
The official expert stops xyz commands and retains its last gripper command
after first success; this shared demonstration endpoint rule is **never used
by deployed policies**. It was fixed before training and test access. All 321
reference attempts are retained, including the 21 unsuccessful attempts.

Input: 22 dimensions, current observation indices 0:18, goal 36:39 and t/250.
The previous-frame block, future states, action-derived features and success
signal are excluded. Output: 16 × 4 xyz/gripper commands, clipped to [-1,1];
execute eight steps and reobserve. This is an official action bound, not a
collision or dynamic-feasibility certificate.

MLP: 22–256–256–64, LayerNorm/SiLU after hidden layers. AdamW 0.001, weight decay
0.0001, batch 128, 1,200 updates, minimum clean-development MSE every 100 steps.
Training-only mean/std normalization with std floor 1e-5. Each source episode
produces 30 windows (H=16, stride=8): 1,200 train and 300 development windows.
Each auxiliary policy fits 32 source episodes and predicts the eight excluded
episodes; there are five folds.

Recovery: six endpoints per source, an eight-step same-task pulse within phase
radius eight, gripper channel 3 protected, coordinate bound 3σ, backtracking
1/0.5/0.25/0, terminal-success acceptance. Gaussian uses AR(1) correlation 0.9.
Select σ from {0.10,0.25} separately for each noise family using development
policy seeds 7/11 and ten layouts; ties choose 0.10. Push selects 0.25; all
other tasks select 0.10. These are Cartesian command amplitudes, not radians.

The four MLP methods use 2,048 clean + 2,048 method draws with replacement,
sharing clean indices, auxiliary quantiles, initialization and minibatches.
Frozen observations uses SCANA-R labels/source times but original observations.
The demonstration kNN policy uses 4,096 clean draws and a five-neighbor
inverse-distance mean in train-standardized current-state space. It is not the
historical residual kNN and does not have a matched MLP optimization cost.

Freeze 120 final MLPs and 30 kNN libraries before evaluating five seeds
(7,11,23,31,47) × 20 new layouts (30000–30019) × two conditions. The nominal
primary outcome is ever-success. The pulse condition applies eight steps
starting at 64, xyz RMS 0.25, cap 0.75, unchanged gripper; post-pulse success
requires success at step 72 or later. Methods/seeds share each layout's noise.

## All six-task results

Percent success; every cell has five policy seeds × 20 layouts.

| Task | Clean | Gaussian | Frozen | SCANA-R | kNN |
|---|---:|---:|---:|---:|---:|
| Reach | 91 | 87 | 85 | 85 | 40 |
| Push | 88 | 84 | 86 | 85 | 31 |
| Pick and place | 72 | 70 | 76 | 68 | 23 |
| Open door | 100 | 100 | 100 | 99 | 100 |
| Open drawer | 100 | 100 | 100 | 100 | 100 |
| Press button | 100 | 100 | 100 | 100 | 100 |
| Macro mean | 91.8 | 90.2 | 91.2 | 89.5 | 65.7 |

After the execution pulse, macro success is 90.2/86.3/89.3/86.8/64.0% in the same
method order. SCANA-R minus Clean is −3.33 points, descriptive 95% interval
[−7.00,−0.17]. This secondary interval is not corrected across all stress tests.
No test-conditioned retuning, new checkpoint selection or endpoint switching
was performed after these results were exposed.

Paired bootstrap uses 5,000 samples of policy-seed and layout blocks. All
methods use the same resampled indices; fixed-task macro means preserve these
blocks across tasks. Six taskwise comparisons additionally report 99.1667%
marginal intervals. The intervals condition on one fixed recovery bank per
task/family/configuration and do not estimate bank recollection variability.

## Frozen ACT deployment checks

Original MuJoCo 2.3.3 / dm-control 1.0.9; 40 existing checkpoints, no retraining.
Twenty new layouts 94000–94019, five seeds, two tasks, four methods and seven
conditions give 5,600 rollouts. Conditions are execute 1/4/8/16 steps per
16-step prediction, early/late execution pulse, and current-observation noise.
The pulse starts at 64/160, lasts eight steps, joint RMS 0.08 rad, cap 0.24 rad,
and leaves grippers unchanged. Input noise is Gaussian sigma 0.01 rad for
joints and 0.005 m for object xyz, leaving grippers/clock unchanged.

At the nominal eight-step interval, SCANA-R succeeds on 93%/77% of Transfer
Cube/Insertion rollouts versus Clean 62%/48%. Paired differences are +31/+29
points, 95% intervals [16,48]/[6,50]. Comparisons with Gaussian are −1/+21
points, intervals [−12,10]/[−1,44], both including zero.

SCANA-R falls to 33%/15% when replanning every step. Late-pulse success is
0–1% for all methods. With observation noise, SCANA-R Transfer Cube is 0%
versus Gaussian 83%. These failures limit any general robustness claim;
the subsequent [current study](study-protocol.md) tests these failures on new layouts. Joint-channel interventions identify a low-support transfer joint as sufficient for the noise collapse; temporal ensembling restores much of the one-step-replanning loss. These findings concern the tested policies and do not establish a universal failure mechanism.

## Evidence and reproduction

Compact evidence: `results/metaworld_extension_v1/` and
`results/act_extension_v1/`. They contain every rollout, seed means, paired
contrasts, fixed protocols and collection costs. Current v47 manuscript Figures 6–8 and
Tables 8–10 are derived from these files. The new 11,600 executions are not
11,600 independent policy training runs, and are not pooled with the original
3,200 ACT records.

Recompute success statistics without simulators or checkpoint files:

```bash
python scripts/summarize_extensions.py
```

Download with `python scripts/download_artifacts.py --bundle extension`, then import the separate evidence bundle, checking the archive and every entry:

```bash
python scripts/import_artifacts.py artifacts/downloads/SCANA-R-MuJoCo-extension-v1.zip --manifest configs/mujoco_extension_bundle.json
```

Use a **separate Python 3.10 environment** for Meta-World, because its MuJoCo
3.3.0 dependency conflicts with the pinned ACT environment:

```bash
python -m pip install -r requirements-metaworld.txt
python experiments/metaworld_extension/audit.py
python scripts/verify_extension_replay.py
python scripts/render_extension_videos.py
python scripts/plot_extension_results.py
```

On Windows, the adapter copies unmodified official assets to an ASCII temp
path if needed for MuJoCo's file API. No package or physics source is modified.
Rendered clips verify all 39 observed coordinates and step rewards against
archived rollouts. Both reference and policy clips are 125 frames at 40 fps,
3.125 seconds at real-time speed, without cuts or retouching.
The additional replay check reloads all five methods at seed 7/layout 30000
for six tasks and both conditions: 60 fresh policy executions match every
saved input, raw observation, action, reward and success flag exactly. These
are reproducibility checks, not 60 additional independent test samples.

To regenerate from scratch, use a fresh repository copy without an imported
`artifacts/metaworld_extension_v1` directory. Entry points deliberately cache
completed work; a cached run must not be reported as new evidence:

```bash
python experiments/metaworld_extension/run.py prepare --workers 6
python experiments/metaworld_extension/audit.py
python experiments/metaworld_extension/run.py freeze
python experiments/metaworld_extension/run.py evaluate --workers 8
python experiments/metaworld_extension/analyze.py
```

Freeze refuses to overwrite an existing frozen protocol. The recorded test
layouts are now public and can be reproduced, but are no longer unseen data
for subsequent algorithm development. Regenerating ACT stress checks requires
the original ACT bundle, its environment and
`python experiments/scana_r/robustness_extension.py --workers 8`.

Original scripts, seeds, failed attempts, weights, action/reward traces and
entry hashes are kept in the separate bundle. Interpreter binaries and
third-party package installations are excluded. Code, compact results and videos are public. Full simulation trajectories and checkpoints are distributed in the [evidence release](https://github.com/2254257831/SCANA-R/releases/tag/reproducibility-v1); see [download and verification instructions](reproduction.md). The manuscript PDF remains private.

Sources: [Meta-World benchmark paper](https://proceedings.mlr.press/v100/yu20a.html),
[official repository](https://github.com/Farama-Foundation/Metaworld),
[task descriptions](https://metaworld.farama.org/benchmark/task_descriptions/).
