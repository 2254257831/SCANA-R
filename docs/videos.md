# Simulation videos: provenance and reproduction

The homepage embeds real MuJoCo renderings of archived evaluation trajectories.
They are not generated illustrations, newly collected results, or real-robot footage.
The policies use current privileged simulator state; the RGB renderings are for
presentation and are not policy inputs.

## What the videos show

| Task | Training seed | Initial layout | Clean repeat | SCANA-R |
|---|---:|---:|---|---|
| Transfer Cube | 7 | 92000 | Failure | Success |
| Insertion | 7 | 92001 | Failure | Success |

Examples are selected intentionally: fix training seed 7, then choose the lowest
layout in each task where the archived SCANA-R rollout succeeds and Clean repeat
fails. This is a reproducible illustrative contrast, not a random sample. It must
not substitute for the aggregate results across five seeds and 40 layouts.

- [Two-task teaser](assets/videos/teaser.mp4): SCANA-R on both tasks.
- [Transfer Cube comparison](assets/videos/transfer-comparison.mp4): Clean repeat and SCANA-R.
- [Insertion comparison](assets/videos/insertion-comparison.mp4): Clean repeat and SCANA-R.

The two panels of each comparison share the same seed and initial layout. They
are combined into a single file so that playback, seeking and pausing remain
synchronized. Each video is uncut, eight seconds at 25 frames per second and
real-time speed. One frame is rendered for every two 50 Hz simulator steps.
Only a neutral title strip and side-by-side composition are added. Posters are
actual frames at five seconds, rather than invented or retouched robot scenes.

## Replay verification

The renderer applies all 400 recorded actions under the pinned ACT/MuJoCo
environment. At every step it compares the current 14-dimensional joint-state
observation against the archived pre-action state, then compares all rewards.
All four clips have a maximum state error of **0.0** and identical reward arrays
on the reference host. It changes only offscreen framebuffer capacity and the
camera used for rendering; no physics, policy weights or actions are changed.
This checks replay consistency, not a new independent evaluation.

[Machine-readable provenance](assets/videos/provenance.json) records the source
rollout hashes, selected layouts, replay checks and video hashes.

Individual source renderings:

- [Transfer Cube / SCANA-R](assets/videos/transfer_cube-scana-r.mp4)
- [Transfer Cube / Clean repeat](assets/videos/transfer_cube-clean.mp4)
- [Insertion / SCANA-R](assets/videos/insertion-scana-r.mp4)
- [Insertion / Clean repeat](assets/videos/insertion-clean.mp4)

## Render again

From the repository root, after importing the **full** frozen artifact bundle:

```bash
python -m pip install -r requirements-video.txt
python scripts/import_artifacts.py ../SCANA-R-artifacts-v1.zip
python scripts/render_project_videos.py
```

The default output directory is `docs/assets/videos`. To keep the checked-in
media intact, use `--output outputs/project-videos`. To compose existing clips
without replaying the simulator, add `--compose-only`. For an alternate title
font, pass `--font /path/to/font.ttf`; the default searches for system Arial on
Windows, then DejaVu Sans on Linux. Font rasterization, OpenGL drivers and video
encoder versions may change output hashes across machines.

Video encoding uses H.264, yuv420p and MP4 fast-start metadata. The page uses
native playback controls, a real-frame poster and a direct MP4 link. The teaser
starts muted when autoplay is permitted; reduced-motion preferences disable
automatic playback. Starting a comparison pauses the other videos. There is
no audio track. On a phone, full-screen playback makes the side-by-side details
easier to inspect.

ACT simulator code and assets retain the original MIT notice in `third_party/act`.
See [the evidence scope](evidence.md) and [protocol](protocol.md) for interpretation.
