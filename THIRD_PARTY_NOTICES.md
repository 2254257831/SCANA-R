# Third-party components

| Component | Source | Included material | License |
|---|---|---|---|
| ACT simulator | https://github.com/tonyzhaozh/act/tree/742c753c0d4a5d87076c8f69e5628c79a8cc5488 | Six Python/license files and the assets directory | MIT; original notice retained |
| MuJoCo, dm-control and other dependencies | Declared package registries | Installed by the user; binaries are not vendored | Respective upstream licenses |
| Academic project page / Nerfies | See `docs/template-references.md` | Design references only; implementation written for this project | No template source redistributed |

The ACT Python files and assets are byte-identical to the audited local pinned
copy. The SCANA-R adapter suppresses image rendering and constructs current-state
observations; it does not change the simulator reward or physics code. The
project uses an MLP policy, not the ACT transformer policy architecture.

Private historical robot logs, public human demonstration datasets, pretrained
VLA weights, bundled Python environments and unrelated manuscripts are not
included. Their absence must not be interpreted as an open-data release.
