# Reproduce the reported evidence

## Statistics from a fresh clone

Only Python 3.10, NumPy and pandas are needed. No simulator, GPU, checkpoint download or original machine directory is required.

```bash
python -m pip install -r requirements-statistics.txt
python scripts/reproduce.py
```

The command first recomputes all **11,040 current-study** evaluation records, including library-level success rates, paired intervals, bank means, cost summaries and failure interventions. Nine generated tables are compared with the recorded analysis at 1e-12 tolerance. It then processes the earlier 3,200 ACT and 11,600 extension records. The 25,840 records remain separate studies; no combined success rate is calculated. CPU costs are reaggregated from recorded per-fit measurements, not measured by timing the statistics command.

For the current study alone:

```bash
python scripts/reproduce.py --current-only
```

Outputs go to `outputs/recomputed-current`, `outputs/recomputed` and `outputs/recomputed-extensions`. The checked-in evidence is never overwritten. CI runs this same default statistics command. This verifies analysis reproducibility; it does not claim fresh collection or policy training.

## Full trajectories and checkpoints

[Download the four evidence archives from the GitHub release](https://github.com/2254257831/SCANA-R/releases/tag/reproducibility-v1). They contain demonstrations, recovery attempts (including failures), calibration arrays, policy weights and recorded trajectories. They are outside the Git tree to keep clones small. Each ZIP has a per-file SHA-256 manifest; matching archive hashes and byte counts are in `configs/*bundle.json` and the [web download manifest](assets/data-downloads.json).

| Archive | Scope | Approximate download |
|---|---|---:|
| SCANA-R-artifacts-v1.zip | Original ACT sources, policies and diagnostics | 829 MB |
| SCANA-R-MuJoCo-extension-v1.zip | Earlier ACT stress and six-task Meta-World study | 998 MB |
| SCANA-R-independent-ACT-v1.zip | Current ACT libraries, costs, components and interventions | 1.96 GB |
| SCANA-R-independent-MetaWorld-v1.zip | Current Meta-World library repetitions | 281 MB |

Download, hash-verify and import (about 4.1 GB compressed; allow additional disk space for extraction):

```bash
python scripts/download_artifacts.py --bundle all --import
```

Choose `--bundle original`, `extension`, `current-act` or `current-metaworld` for one archive. Without `--import`, the command only downloads and verifies the ZIP. The importer refuses unsafe paths and refuses to overwrite locally changed files. Simulator dependencies are installed separately; no interpreter binaries, real-robot private data or manuscript PDF are distributed.

## Raw-data analysis and replay

After importing, the original frozen analysis can independently reconstruct the current cost table from per-library/per-policy metadata and read the noise masks from arrays:

```bash
python experiments/independent_libraries_v46/analyze_experiments.py
python experiments/independent_libraries_v46/analyze_metaworld.py
```

For the ACT video replay, use Python 3.10 with `requirements-simulation.txt` (MuJoCo **2.3.3**, dm-control **1.0.9**) and `requirements-video.txt`, then:

```bash
python -m pip install -e . --no-deps
python scripts/render_current_videos.py
```

The renderer checks every pre-action state or clean current input and every reward against the archived record. Meta-World requires a **separate** environment with `requirements-metaworld.txt` (MuJoCo **3.3.0**). Do not upgrade the ACT environment in place.

Fresh collection/training commands are in [the current experiment guide](https://github.com/2254257831/SCANA-R/tree/main/experiments/independent_libraries_v46). Use a fresh output directory and a new protocol for new scientific tests. Existing public test layouts are no longer unseen data for method development. Portable exported training scripts have not themselves been used for a complete independent repetition after export; replay and statistical checks are narrower guarantees.

The current manuscript text predates this raw-evidence release. The availability change adds downloads; it changes no experiment, result or algorithm. The full manuscript remains private.
