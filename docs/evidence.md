# Evidence and reproducibility scope

The latest **LIBERO-Spatial ten-task extension (2,400 test episodes)** is available in the [reviewer package, raw archives and protocol](libero-spatial.md). The older studies below are separate batches.

The independent-library and cost study contains 11,040 evaluations: 4,200 ACT library/cost/component comparisons, 3,240 Meta-World library repetitions and 3,600 frozen-policy interventions. See [the protocol and manuscript map](study-protocol.md), [compact-data reanalysis and raw downloads](reproduction.md), and [study videos](current-videos.md). The older records below remain separate studies.

The expanded eight-task simulation study adds 11,600 records in `results/act_extension_v1/` and `results/metaworld_extension_v1/`. Six-task macro performance does not improve over Clean repeat. See [the extension protocol](mujoco-extension.md) and [all-task gallery](mujoco-gallery.html); the original results below are retained as a separate study.

The checked-in `results/independent_test/per_episode.csv` has 3,200 unique records:
eight methods × two tasks × five training seeds × 40 common layouts. The four
matched current methods account for 1,600 records. Recompute rather than round
intermediate values when reporting differences.

| Paper material | Reproduction source | Availability |
|---|---|---|
| SCANA-R method and algorithm | `src/scana_experiments/recovery_augmentation.py`, `experiments/scana_r/` | Source, pinned environment, tests |
| Current independent-library, cost and intervention statistics | `results/independent_libraries_v46/`, `scripts/reproduce_current.py` | All 11,040 per-rollout rows and recorded costs included |
| Earlier success rates and contrasts | `results/independent_test/`, `scripts/summarize.py` | All 3,200 per-rollout rows included |
| Development amplitude choice | `results/development/`, `configs/protocol_frozen.json` | Recorded CSV/JSON included |
| Recovery contract, attempts, weights and trajectories | Four separate evidence bundles | [Release downloads and SHA-256 verification](reproduction.md) |
| Four current method diagrams | `docs/assets/figures/figure-1` through `figure-4` | Vector PDFs and SVGs refined for 190 mm width; the editable source PPT is preserved in `figures/` |
| Historical single-arm offline tables | `legacy/` and historical modules in `src/` | Code archived; original private data excluded |
| Historical public-data residual training | `legacy/original_v35/`, `legacy/tools/` | Requires original public caches / dataset versions; no fresh reproduction claim |
| GR00T, visual policy, real robot | None in the current study | Not evaluated |

The historical scripts are retained as research provenance and have their
original workspace-relative paths. They are not supported standalone entry
points. Use the current experiment entry point for reproducible SCANA-R results.
The complete local research archive is preserved outside this source repository.

For the earlier fixed-library ACT study, two-way bootstrap resamples both training seeds and layouts, preserving method
pairing. It does not estimate variation across newly collected recovery banks.
The two primary clean comparisons also have 97.5% intervals, [5.0,45.5] and
[10.5,38.0] percentage points. Test success is any-step maximum reward 4;
collection acceptance uses terminal reward 4. These are different definitions.

The method is related to DART-style recovery collection. No full DART baseline
with equal collection budget is available. Extra simulation collection and
auxiliary-model training costs should be reported separately from the matched
final policy budget. Statistical action bounds alone do not establish general
robot executability, and this study does not train on frozen images paired with
newly executed states.
