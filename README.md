# SCANA-R

**Success Demonstration Based Action Noise Calibration and Simulation Recovery Learning**

成功示范驱动的动作噪声校准与仿真恢复学习

[Project page](docs/index.html) · [中文说明](README.zh-CN.md) · [Protocol](docs/protocol.md) · [Evidence scope](docs/evidence.md) · [License](LICENSE)

SCANA-R calibrates short action perturbations from out-of-episode policy errors,
executes those perturbations in a resettable simulator, and trains on verified
recovery observations paired with actions that were actually executed. This
repository accompanies the revised manuscript and preserves its original PPT
figure organization. It is a **local release preparation**, with no GitHub remote
or public release configured yet.

## Main results

The frozen ACT/MuJoCo simulation evaluation uses five training seeds crossed with
40 common layouts per task. Each entry is successes / 200 rollouts. Policies use
current privileged state, not images, language, or future observations.

| Method | Transfer Cube | Insertion |
|---|---:|---:|
| Clean repeat | 134 / 200 (67.0%) | 86 / 200 (43.0%) |
| Gaussian recovery | 175 / 200 (87.5%) | 90 / 200 (45.0%) |
| Frozen observations | 117 / 200 (58.5%) | 87 / 200 (43.5%) |
| **SCANA-R** | **184 / 200 (92.0%)** | **134 / 200 (67.0%)** |

SCANA-R improves on Clean repeat by 25 and 24 percentage points. Paired two-way
bootstrap 95% intervals are [7.5, 42.5] and [12.5, 36.0] percentage points.
Against Gaussian recovery, the intervals are [-5.0, 16.0] and [5.0, 40.5]: the
Transfer Cube comparison does **not** establish a calibration-specific gain.
The five training seeds, not the 200 rollouts, are the independent policy fits.

All eight methods, including four historical version diagnostics, are retained
in [the 3,200-row evaluation table](results/independent_test/per_episode.csv).
Historical label-only failures are not a general claim that kNN or deterministic
regression cannot work. Read [evidence.md](docs/evidence.md) before using them.

## Repository layout

```text
src/scana_experiments/      recovery primitives and historical SCANA components
experiments/scana_r/        portable collection, cross-fitting, recovery and evaluation
scripts/                   command line entry, statistics and artifact import
tests/                     algorithm contracts and evidence checks
configs/                   frozen protocol, environment and artifact manifest
results/                   compact recorded CSV / JSON evidence
third_party/act/            pinned simulator subset, assets and original MIT notice
figures/                   editable four-figure PPT, Times New Roman
docs/                      static GitHub Pages website and reproducibility notes
legacy/                    archived diagnostic scripts; not the current method
artifacts/                 imported or regenerated arrays / weights (Git ignored)
outputs/                   local generated outputs (Git ignored)
```

## Installation

The reference simulation environment is Python **3.10.19**, MuJoCo **2.3.3**,
dm-control **1.0.9**, CPU PyTorch **2.7.1**, and NumPy **2.2.6** on Windows.
The simulator is the subset of ACT commit
`742c753c0d4a5d87076c8f69e5628c79a8cc5488`. The MLP used here is not the ACT
transformer architecture. Linux installation is supported by the dependencies
but has not been independently reproduced for this release.

From the repository root in a Python 3.10 environment:

```bash
python -m pip install -r requirements-simulation.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
```

Use the CPU PyTorch wheel for the reference environment. Do not silently replace
MuJoCo or dm-control versions; contact dynamics and successful reference replay
can change. No pretrained vision-language model is required.

## Recompute the reported statistics

This lightweight path needs only NumPy and pandas; it does not need the simulator,
raw trajectories or a GPU.

```bash
python scripts/summarize.py
```

It validates the complete seed/layout grid and writes recomputed summaries and
paired bootstrap contrasts to `outputs/recomputed/`. It never overwrites the
checked-in evidence. This is reanalysis of existing rollouts, not a new test.

## Import the separate evidence bundle

The preparation directory contains a sibling archive
`../SCANA-R-artifacts-v1.zip`. It contains recorded simulation demonstrations,
cross-fit models, calibration errors, recovery attempts, final policies and
rollouts. It excludes interpreter binaries and private real-robot data. Large
files are deliberately outside Git; the expected archive and entry hashes are
recorded in `configs/artifact_bundle.json`.

```bash
python scripts/import_artifacts.py ../SCANA-R-artifacts-v1.zip
python scripts/run_scana_r.py replay-check
python scripts/run_scana_r.py audit
```

For a minimal exact replay check, add `--profile smoke` to the import command;
the full audit requires the default full import. Only load the trusted,
hash-verified checkpoint bundle: its PyTorch checkpoints contain Python objects.
No remote download link exists yet. Future users need the matching release asset
or can regenerate the four current methods below.

## Regenerate the four current methods

Use an empty, separate artifact directory so cached historical results cannot
be confused with new computation. Each command reuses completed artifacts in
that directory. A new seed/layout protocol requires a new directory and protocol;
the recorded test layouts have already been inspected and are not a new holdout.

```bash
python scripts/run_scana_r.py collect --artifacts outputs/fresh
python scripts/run_scana_r.py calibrate --artifacts outputs/fresh
python scripts/run_scana_r.py develop --artifacts outputs/fresh
python scripts/run_scana_r.py evaluate --main-only --artifacts outputs/fresh
python scripts/run_scana_r.py audit --artifacts outputs/fresh
python scripts/summarize.py --input outputs/fresh/scana_r/independent_test/per_episode.csv --output outputs/fresh-summary
```

Collection produces 50 successful references per task (40 training / 10
development). Cross-fitting trains five auxiliary models per task. Development
compares two amplitudes for each noise family. Final evaluation trains/evaluates
the four matched methods on five seeds and 40 layouts (1,600 rollouts). The
reference full evaluation with historical controls contains 3,200 rollouts; those
four legacy policies require the frozen bundle and are not retrained by this
command. For the full frozen eight-method replay, import the complete bundle and
run `python scripts/run_scana_r.py evaluate`.

The algorithm uses four CPU workers and four Torch threads per process. The full
pipeline is substantially longer than the statistics/unit-test path. Recovery
collection includes unsuccessful attempts and extra simulator calls; it is not
equal in collection cost to simply repeating clean data.

## Project website

Open `docs/index.html` directly, or preview using a loopback-only server:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Then visit `http://127.0.0.1:8000/`. All page assets use relative URLs, with no
build step, analytics or external font dependencies. The page is prepared for
GitHub Pages `main` / `docs`; no deployment workflow is activated. See
[the local release checklist](docs/release-checklist.md) for the later transition
from a local link to an official project URL.

Optional maintenance: after changing documentation Markdown, run `npm install`
and `npm run build:docs` to rebuild the included HTML pages. Regenerate the
homepage's downloadable source package with `python scripts/package_source.py`.
Neither command uploads anything. Node is not required to view the site or run
the Python experiments.

## Reproducibility limits and provenance

This is a privileged-state simulation study of two bimanual ACT tasks. It is not
a GR00T/VLA training release, a real-robot study, or evidence of collision safety
outside the tested tasks. There is one fixed recovery bank per selected
configuration, so bootstrap intervals are conditional on that bank. There is no
same-acquisition-budget DART reproduction. The recovery-collection idea is
related to [DART](https://berkeleyautomation.github.io/DART/).

`provenance/source_files.json` maps every copied source to its original project
path and hashes. Portable adapters change paths/imports and correct an inaccurate
cross-fit checkpoint comment; the core recovery functions are byte-identical.
See [the migration note](docs/migration.md). No authors, affiliations, publication
venue, DOI or formal BibTeX record are invented; these will be filled when the
publication metadata is finalized.

## License

Original project code and website: **Apache-2.0**, selected by the project owner.
ACT subset: **MIT**, with its notice retained. See [LICENSE](LICENSE), [NOTICE](NOTICE)
and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The license does not grant
rights to excluded private data or third-party datasets and models.
