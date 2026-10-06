# LIBERO-Spatial ten-task reviewer materials

Four methods, three paired recovery-library/policy repetitions, ten official
Spatial tasks and twenty test initial states per task: **2,400 test episodes**.
This is a custom multitask evaluation of a compact visual/instruction-conditioned
policy trained from scratch. It is not all LIBERO suites, the default lifelong
protocol, a foundation VLA, CALVIN or SimplerEnv evaluation.

## Start here (Python standard library only)

Run `python recompute_statistics.py` from this folder. The script validates all
2,400 unique rows and 120 groups, checks the supplied tables, and recomputes task
success rates, sample SD across three repetitions, paired differences, motion
proxy failure counts, and matched-cost deviations. No download or GPU is needed
for this statistics check. The derived failure proxies are not causal diagnoses.

Success means (percent): repeated original 76.33, correlated Gaussian 70.67,
SCANA-R 72.33, and Gaussian with matched total active wall time 70.33. These three
repetitions do not establish a stable calibration advantage. Original source
demonstrations remain fixed across repetitions; time matching is not energy matching.

## Raw records and models

Download the companion assets at
https://github.com/2254257831/SCANA-R/releases/tag/libero-spatial10-v1.
`SCANA-R-LIBERO-Spatial-evaluation-models-v1.zip` includes all 2,400 formal state/
action/reward traces, 12 final policies, 15 auxiliary fold policies, warm-up model,
calibration banks, cost records, and fixed-ID videos. Six `recovery-lib*-*.zip`
assets contain the accepted windows, newly captured camera arrays, attempted
rollouts, and observation-action audit evidence for each independent library.
Extract companion ZIPs into the same empty folder; their common root is
`libero_spatial10/`. File-level manifests and asset SHA-256 values are supplied.
The 720-episode pilot is a separate batch and is not pooled here.

## Scope of reproduction

`protocol/` and `audits/` record task identities, 40/10 episode splits, seeds,
600-step horizon, observation alignment, initial state/RGB pairing, fold isolation,
and evaluation-only repairs. Twelve originally discrepant initial-image records
were rerun as one paired block with unchanged actions/outcomes. The missing old
pixels prevent a diagnosis of that image-hash discrepancy; the audit retains this
limitation. Evaluation repairs did not retrain policies or change training budgets.

`execution_sources/` preserves the exact executed algorithm, collection, fitting,
evaluation and audit sources (see SOURCE_SHA256.json). Local dashboard and pause
orchestration scripts are omitted. These sources require the pinned simulation
stack, official data/assets and the original relative directory convention in
runtime.py. They are archival execution sources, not a tested cross-machine
one-command installer. Use a fresh study copy, never overwrite supplied records.
The lightweight statistics workflow is the independently checked entry point.

Raw numerical arrays are unchanged. Only local checkpoint metadata paths and
release documentation are normalized for publication; source/published hashes and
tensor-storage preservation checks are listed in RELEASE_TRANSFORMS.json.
The full paper and private real-robot data are not included.
