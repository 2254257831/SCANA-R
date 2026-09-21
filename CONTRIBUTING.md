# Contributing

Use Python 3.10 and run `python -m unittest discover -s tests -v` before changing
the algorithm or metrics. Recompute the checked-in evaluation table with
`python scripts/summarize.py`; existing evidence must never be silently replaced
by a different protocol or newly tuned method.

For algorithm changes, use a new artifact directory and record the changed
observation contract, training budget, collection budget and test-layout policy.
The existing test layouts have already been evaluated and cannot serve as a new
unseen holdout. Preserve source episode identifiers and all rejected recovery
attempts. Distinguish privileged-state simulation from visual-policy or real-robot
evidence.

Do not commit raw trajectories, model weights, private data, credentials or
Python environment directories. Keep large artifacts separate and record their
SHA-256. Keep third-party license notices intact.

For website edits, preserve the approved original-PPT visual organization.
Update documentation with `npm run build:docs`, check JavaScript syntax and local
links, then run `python scripts/package_source.py` and
`python scripts/check_release.py`. Manually preview desktop and mobile layouts.

This local preparation has no issue tracker or remote repository configured yet.
