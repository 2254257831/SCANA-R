# Ten-task method adaptation

The algorithm and causal data interface are the same as the verified pilot. See `../libero_local_v1_20261002/METHOD_ADAPTATION.md` for the comparison with v49 ACT.

The scope now covers all ten Spatial tasks and 500 demonstrations. Task-specific error neighborhoods stay within the same task. Vocabulary, normalization, auxiliary policies, error banks, final policies and recovery libraries are rebuilt from the ten-task training split. Nothing is fit on final test outcomes.

Update budgets are multiplied by 10/3 to retain approximately the same per-task exposure. Amplitude remains 0.05; no new amplitude scan is performed. All ten tasks must pass the declared development gate before formal testing.

The same zero-scale native replay caveat and active-wall-time cost limitation apply. The result is not a foundation-VLA or full-LIBERO-all-suites experiment. Configuration is authoritative in protocol.json.

## Fixed-body reset audit

Reusing one environment on the drawer task produced different fixture body positions despite equal restored qpos/qvel and identical seeds (up to 1.55 cm in the diagnostic). Before formal results, evaluation was changed to construct and seed a fresh official environment for every episode. Task goals and physics are unchanged. Initial fixed-body poses and RGB/proprioception are now saved and paired across methods and repetitions alongside dynamic simulator state. Diagnostics are in `preflight/drawer_reset_diagnostic.json`; all-task checks in `preflight/suite_smoke.json`.

## Official evaluation horizon

Before any ten-task development or test outcomes, the limit was set to 600 control steps, matching the pinned official `libero/configs/eval/default.yaml` (20 evaluation episodes). The three-task pilot had used 300. Keep these experimental batches separate; do not pool scores. The previous setup configuration is retained in `protocol/setup_protocol_before_official_horizon.json`.
