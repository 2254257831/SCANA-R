# Current-study simulation videos

These six synchronized replays correspond to the current independent-library, matched-cost and failure-intervention studies. They are recorded action replays, not a new experiment. All 20 source clips verify all 400 pre-action states or current clean inputs and all step rewards exactly (maximum state error 0).

Outcome-independent: first task library 101, policy seed 17, layout 97000; intervention policy seed 7, layout 96000. Same fixed identifiers for all compared methods and conditions. These examples are not aggregate estimates.

The insertion task has no low-support joint under the frozen threshold; its noise panels therefore compare nominal, joint-only and object-only noise. Temporal ensembling is a deployment variant. An individual fixed example can disagree with an aggregate trend.


<figure class="comparison"><h3>Transfer Cube · libraries</h3><video class="film" width="1080" height="840" style="aspect-ratio:1080/840" controls muted loop playsinline preload="none" poster="assets/current-videos/transfer_cube-libraries-poster.jpg" aria-label="Transfer Cube · libraries"><source src="assets/current-videos/transfer_cube-libraries.mp4" type="video/mp4"><a href="assets/current-videos/transfer_cube-libraries.mp4">Open MP4</a></video><figcaption>Clean repeat: success · Gaussian recovery: success · SCANA-R: success · Gaussian CPU budget: success <a href="assets/current-videos/transfer_cube-libraries.mp4">Open MP4 ↗</a></figcaption></figure>

<figure class="comparison"><h3>Transfer Cube · noise</h3><video class="film" width="1620" height="420" style="aspect-ratio:1620/420" controls muted loop playsinline preload="none" poster="assets/current-videos/transfer_cube-noise-poster.jpg" aria-label="Transfer Cube · noise"><source src="assets/current-videos/transfer_cube-noise.mp4" type="video/mp4"><a href="assets/current-videos/transfer_cube-noise.mp4">Open MP4</a></video><figcaption>SCANA-R: nominal: success · SCANA-R: low-support noise: failure · SCANA-R: other-channel noise: failure <a href="assets/current-videos/transfer_cube-noise.mp4">Open MP4 ↗</a></figcaption></figure>

<figure class="comparison"><h3>Transfer Cube · replanning</h3><video class="film" width="1620" height="420" style="aspect-ratio:1620/420" controls muted loop playsinline preload="none" poster="assets/current-videos/transfer_cube-replanning-poster.jpg" aria-label="Transfer Cube · replanning"><source src="assets/current-videos/transfer_cube-replanning.mp4" type="video/mp4"><a href="assets/current-videos/transfer_cube-replanning.mp4">Open MP4</a></video><figcaption>Execute 8 steps: success · Execute 1 step: failure · 1 step + temporal ensemble: success <a href="assets/current-videos/transfer_cube-replanning.mp4">Open MP4 ↗</a></figcaption></figure>

<figure class="comparison"><h3>Insertion · libraries</h3><video class="film" width="1080" height="840" style="aspect-ratio:1080/840" controls muted loop playsinline preload="none" poster="assets/current-videos/insertion-libraries-poster.jpg" aria-label="Insertion · libraries"><source src="assets/current-videos/insertion-libraries.mp4" type="video/mp4"><a href="assets/current-videos/insertion-libraries.mp4">Open MP4</a></video><figcaption>Clean repeat: success · Gaussian recovery: success · SCANA-R: failure · Gaussian CPU budget: failure <a href="assets/current-videos/insertion-libraries.mp4">Open MP4 ↗</a></figcaption></figure>

<figure class="comparison"><h3>Insertion · noise</h3><video class="film" width="1620" height="420" style="aspect-ratio:1620/420" controls muted loop playsinline preload="none" poster="assets/current-videos/insertion-noise-poster.jpg" aria-label="Insertion · noise"><source src="assets/current-videos/insertion-noise.mp4" type="video/mp4"><a href="assets/current-videos/insertion-noise.mp4">Open MP4</a></video><figcaption>SCANA-R: nominal: failure · SCANA-R: joint noise: failure · SCANA-R: object noise: failure <a href="assets/current-videos/insertion-noise.mp4">Open MP4 ↗</a></figcaption></figure>

<figure class="comparison"><h3>Insertion · replanning</h3><video class="film" width="1620" height="420" style="aspect-ratio:1620/420" controls muted loop playsinline preload="none" poster="assets/current-videos/insertion-replanning-poster.jpg" aria-label="Insertion · replanning"><source src="assets/current-videos/insertion-replanning.mp4" type="video/mp4"><a href="assets/current-videos/insertion-replanning.mp4">Open MP4</a></video><figcaption>Execute 8 steps: failure · Execute 1 step: failure · 1 step + temporal ensemble: failure <a href="assets/current-videos/insertion-replanning.mp4">Open MP4 ↗</a></figcaption></figure>

## Replay and traceability

[All source paths, hashes and replay checks](assets/current-videos/provenance.json) · [Data archives and instructions](reproduction.md) · [Earlier selected examples](videos.md)

After importing the current ACT archive and installing the pinned ACT/video dependencies:

```bash
python scripts/render_current_videos.py
```

The video renderer does not retrain a policy or alter recorded actions. Every clip lasts 8 seconds at real-time speed (400 physics steps at 50 Hz; 200 frames at 25 fps). Labels are added above actual simulator frames; there are no cuts, interpolated actions or generated robot images.
