# Current study protocol

The current manuscript is **SCANA-R: Action noise calibration and simulation recovery learning from successful demonstrations** (v47). Its current evidence is the independent-library, cost and intervention study first archived under `v46`. The directory name records the experiment version; v47 did not run a different experiment.

## Keep the three studies separate

| Study | Evaluation records | Libraries and policies | Test layouts |
|---|---:|---|---|
| Current ACT libraries, cost and components | 4,200 | 5 libraries/task × 3 policies × 7 methods × 2 tasks | 97000–97019 |
| Current Meta-World libraries | 3,240 | 3 libraries/task × 3 policies × 3 methods × 6 tasks | 98000–98019 |
| Current ACT failure interventions | 3,600 | 5 frozen original policies × 2 methods × 9 conditions × 2 tasks | 96000–96019 |
| Earlier original ACT study | 3,200 | 5 policies × 8 methods × 2 tasks; fixed recovery libraries | 92000–92039 |
| Earlier extension | 11,600 | ACT stress 5,600; Meta-World 6,000; fixed recovery libraries | ACT 94000–94019; Meta-World 30000–30019 |

The default [results section](index.html#results) reports the first three rows (11,040 evaluations). The 92%/67% ACT result belongs to the earlier fixed-library study. Results are never pooled across these batches. An evaluation episode is not an independent policy fit.

## Algorithm and observations

Source episodes are split before window extraction: 40 training and 10 development successes per task. Five auxiliary policies fit 32 source episodes each; the eight excluded episodes supply out-of-episode errors. Auxiliary checkpoint selection uses fitting-fold MSE; final policy selection uses clean development MSE. The original source demonstrations remain fixed in the library repetition.

SCANA-R samples an eight-step error pulse from the same task and temporal neighborhood (radius eight), scales its RMS, caps coordinates at three times that scale and protects gripper channels. Every retry resets the original simulator state and reuses the pulse at scales 1, 0.5, 0.25, 0. Acceptance requires **terminal** task success. Training pairs the observations actually visited after the pulse with the executed reference-tail actions. Final training mixes 2,048 original and 2,048 method records sampled with replacement (1:1).

ACT inputs are current joint state, current object position and time; predictions contain 16 actions, with eight executed before reobserving. Twelve candidate endpoints are drawn per training source. Transfer/insertion pulse scales are 0.03/0.08 rad. ACT uses MuJoCo 2.3.3 / dm-control 1.0.9. Meta-World uses a separate MuJoCo 3.3.0 environment, a 22-dimensional current input, 16 × 4 action chunks, six endpoints, and the original development-selected task/family scales. Neither platform uses future states, images or language in this study.

## Comparisons and uncertainty

ACT compares Clean repeat, Gaussian recovery, SCANA-R, no cross-fitting, global-time errors, no intermediate backtracking, and Gaussian recovery with longer training at matched total active CPU cost. Cost includes auxiliary fitting, recovery simulation and final fitting; stopping at development checkpoints can leave budget differences, reported in the recorded cost tables.

ACT library IDs are 101, 211, 307, 401 and 503; final-policy seeds are 17, 29 and 43. Meta-World repeats three libraries per task. The exact grids and configurations are in the two frozen JSON protocols in the [compact download](assets/v46/data-and-protocols.zip).

The library analyses use 10,000 paired bootstrap replicates: libraries, policy seeds within selected libraries, and shared layouts. ACT primary comparisons report two-task adjusted 97.5% marginal intervals. Meta-World reports two-comparison macro intervals and six-task adjusted intervals. Source-demonstration population variation is not estimated. The failure analysis uses 5,000 paired policy/layout replicates with frozen original policies.

## Failure interventions

Nine deployment conditions separate current-observation noise by channel and change replanning cadence. In transfer, noise on the narrowly supported joint 0 is sufficient to cause the observed failure; removing that channel raises SCANA-R success to 65% in this diagnostic. A fixed temporal ensemble uses predictions from the latest 16 chunks, weighted by exp(−age/4). It restores much of the one-step-replanning loss and is a **deployment variant**, not an extra component of the collection algorithm. These findings are confined to the tested policies and perturbations; reduced mean action-step RMS does not support a generic command-jitter explanation.

## Manuscript map and reproduction

| Evidence | v47 figures/tables |
|---|---|
| Method | Figures 1–4 |
| Independent ACT libraries and total cost | Figure 5; Tables 5–6 |
| Independent Meta-World libraries | Table 7 |
| Earlier eight-task extension | Figures 6–8; Tables 8–10 |
| Current failure interventions | Figure 9 |

[Recompute statistics and download raw evidence](reproduction.md) · [Current videos](current-videos.md) · [Earlier original ACT protocol](protocol.md) · [Earlier extension protocol](mujoco-extension.md)
