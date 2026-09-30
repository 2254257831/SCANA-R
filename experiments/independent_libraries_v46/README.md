# Independent libraries, costs, and failure interventions (v46)

This study adds 11,040 evaluation episodes: 4,200 ACT library/ablation episodes,
3,240 Meta-World library episodes, and 3,600 frozen-policy interventions.
Methods, library/policy seeds, and new layout IDs were fixed before these tests.
The diagnostic hypotheses were motivated by historical failures; this is not
an independently preregistered mechanism study.

Use the existing repository installation and import the original ACT and
Meta-World source demonstrations/checkpoints first. ACT requires the pinned
MuJoCo 2.3.3 environment; Meta-World requires a separate MuJoCo 3.3.0 environment.
The source demonstrations remain fixed (40 train and 10 development per task).

From the repository root:

```sh
# ACT environment, including pandas
python experiments/independent_libraries_v46/experiment_v46.py all --workers 8
python experiments/independent_libraries_v46/analyze_experiments.py

# Separate Meta-World environment
python experiments/independent_libraries_v46/metaworld_replicate.py --workers 6
python experiments/independent_libraries_v46/analyze_metaworld.py
```

Default output is `artifacts/independent_libraries_v46`. Set
`SCANA_V46_OUTPUT` to a fresh directory to run a separate replication. Do not
point portable reruns at the frozen original evidence: the original protocol
contains the SHA of the executed, machine-local script, and path-adapted script
hashes intentionally differ. ACT's original checkpoints and sources use
`SCANA_ARTIFACTS` as documented in the main README. Meta-World currently reads
the existing `artifacts/metaworld_extension_v1` source collection.

`runtime` and `mw_runtime` copy simulator assets without changing their bytes
when Windows requires an ASCII path. `SCANA_ASSET_CACHE` and
`SCANA_MW_ASSET_CACHE` can select a writable ASCII cache directory.

The portable copies change path bootstrapping only; the executed scripts,
protocol hashes, and full original file inventory are retained with the paper's
reproducibility package. `portability_manifest.json` records both versions.
Portable scripts were syntax-checked; the full reported experiment was run
using the archived original scripts, not independently repeated after export.

Recorded CSVs, protocols and analysis are in
`results/independent_libraries_v46`. Full arrays and checkpoints remain in the
local paths in `full_evidence_manifest.json`; the compact CSV directory alone
does not suffice for recomputing collection costs or verifying raw trajectories.

The results do not establish that calibration is necessary or superior at the
same total active CPU cost. Report all methods and tasks, including failures.
Library-level intervals remain conditional on the fixed original demonstrations.
