# ai_v12 — TODO

The **win-prob → behavior coupling** chapter. Where ai_v11 is what we can learn from an external
action distribution, **ai_v12 is how the win-probability head's knowledge becomes behavioral
force** — the head is a barometer, and these are the three routes that turn it into a coach.

Plan of record: [`design_winprob_behavior_coupling.md`](design_winprob_behavior_coupling.md).

Built and OFF (2026-08-29):

* **Route 1** — `--win-prob-pbrs-coef` (PBRS reward shaping, `γφ(s′) − φ(s)`, φ detached).
* **Routes 2+3** — `--search-teacher-mode winprob_oneply` (one-ply win-prob ranking targets
  behind a contested gate, a margin floor, and paired-rollout confirmation).

**Probe L has landed and is folded in (§7): the head KNOWS.** It ranks an alternative above the
played action on 96.4% of immune whiffs (+0.213 over the tightest control, dice-invariant) while
the policy samples that alternative at a median p = 0.002. The distillation branch FIRES, so
**route 2 is the first arm**; route 1's E1 ladder was re-sized by two orders of magnitude in the
same pass (§7.2). Nothing runs until an era registers arms from §6's ladder.

**CAPSTONE PROBE registered (2026-08-30, owner-ordered):**
`probe_risk_modulation_capstone.md` — does the win-prob value function buy correct risk
modulation? Three offline instruments (accuracy-tradeoff curve Surf/Hydro-Pump-class ·
Explosion timing · CRN-reroll spread curve) + the guess-point entropy companion, with frozen
per-arm predictions: sparse steepest slope; shaped-world shallowest; a FLAT sparse slope
falsifies "P(win) buys risk for free" and must be reported as loudly as a pass. Baseline the
trace-only instruments on gen-15 NOW; full battery runs as the end-of-era capstone.
