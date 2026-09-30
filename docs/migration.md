# Portable release changes

The original v35/v36 research code and the v38 manuscript remain untouched.
`provenance/source_files.json` records the input and output SHA-256 of copied
files. The recovery primitive module is unchanged byte-for-byte.

Changes in the supported current experiment:

- Repository-relative `src/`, `third_party/act/` and configurable artifact roots
  replace workspace-specific imports and bundled binary environments.
- The existing Standardizer and ChunkPolicy definitions are extracted into
  `policy_network.py` without numerical changes.
- Current experiment I/O helpers are separated from unused historical models.
- Abandoned continuous-noise development functions are removed from the current
  entry file; they are not part of the reported SCANA-R algorithm.
- The cross-fit comment now correctly says minimum fitting-fold training MSE,
  matching the unchanged code, rather than incorrectly saying last checkpoint.
- The clean development policy is fitted when missing, allowing a fresh run
  without relying on an earlier abandoned development directory.
- An optional four-method evaluation enables from-scratch reproduction without
  the frozen legacy control checkpoints. Eight-method replay remains available.
- Numerical summary logic is separated from the simulator and writes to a new
  output directory. Test-grid and duplicate checks are enforced.
- The ASCII simulator asset cache handles simultaneous first-use processes.

Changes to imports/paths are release maintenance, not new experimental results.
See `validation/release_validation.json` for the checks actually performed in
this release. Passing a replay check on the same host is not a fresh-machine
reproduction or an additional independent test.

Review-release metadata maintenance removes the machine-specific drive prefix
from `full_evidence_manifest.json`. Inventory paths, sizes, and SHA-256 records
are unchanged and remain relative to the separately archived research workspace.
The historical `legacy/configs/scana_single_arm.json` uses repository-relative
data and output paths; its original source hash is retained and its release hash
is updated in the provenance inventory. Historical algorithm parameters and all
current experiment scripts are unchanged.
