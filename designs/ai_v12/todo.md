# ai_v12 — TODO

The **win-prob → behavior coupling** chapter. Where ai_v11 is what we can learn from an external
action distribution, **ai_v12 is how the win-probability head's knowledge becomes behavioral
force** — the head is a barometer, and these are the three routes that turn it into a coach.

Plan of record: [`design_winprob_behavior_coupling.md`](design_winprob_behavior_coupling.md).

Built and OFF (2026-08-29):

* **Route 1** — `--win-prob-pbrs-coef` (PBRS reward shaping, `γφ(s′) − φ(s)`, φ detached).
* **Routes 2+3** — `--search-teacher-mode winprob_oneply` (one-ply win-prob ranking targets
  behind a contested gate, a margin floor, and paired-rollout confirmation).

Blocking dependency: **probe L** (does the head already know about the whiffs?) — §7. Nothing
runs until an era registers arms from §6's ladder.
