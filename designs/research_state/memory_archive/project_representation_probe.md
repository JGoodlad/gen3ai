---
name: project_representation_probe
description: "prober `probe` representation-probe harness (built 2026-06-09) — linear-probes the model's internal activations to test \"is a derived fact already in the rep, or should we add it\"; first finding = speed IS represented (don't add), damage SPREAD is not"
metadata: 
  node_type: memory
  type: project
  originSessionId: b7d3c1d7-0aa2-4875-9183-6e9491acdea9
---

> **Archived 2026-09-08** — ai_v5-era probe harness, shipped; src/main/prober/CLAUDE.md is the doc of record. Preserved verbatim; nothing below is current.

Built 2026-06-09: `python -m main.prober.query probe <run> <is_faster|damage_taken|faint_soon>`
(also `ProbeSession.probe()`). Answers the "are we saturated on the info we have vs should we
hand it more" question EMPIRICALLY. Fits a cross-validated LINEAR probe on the model's internal
post-projection activations (`ProbeModel.features` → value-head `vf` / policy-head `pi`) to recover
a derived quantity, and compares it to a baseline probe on the raw obs/belief feature we ALREADY
provide. Probe recovers X ⇒ the model computed X (a new feature is redundant); can't ⇒ extraction
gap (a real obs lever — "let it learn" per [[feedback_provide_vs_learn]] hit this small net's
capacity wall for X). Stats = pure numpy (no sklearn): standardized ridge/logistic, k-fold
OUT-OF-FOLD preds, **auto-tuned l2 over a grid** (critical at d≈512 — fixed weak l2 overfits to a
negative OOF R²; caught + fixed during the build). Every result splits overall vs by-group
(easy-vs-contested = the signal) + carries a `caveat`. One checkpoint load per call. 90 prober tests.

6 targets now: is_faster, damage_taken, faint_soon, faint_healthy (HP-controlled surprise-OHKO),
big_hit_incoming (≥40% HP loss, less-RNG damage), opp_switches (opponent modeling).

**REPRESENTATION MAP (ai_v5_6_stable_70m_0608 @70M, value-head features) — answers "what does the
model need":**
- **`is_faster`**: rep AUC 0.94 on CONTESTED vs provided `active_outspeed` 0.75 → **speed already
  represented; do NOT add a speed feature.** (Answers "speed from Leftovers/Sandstorm": no.)
- **`opp_switches`**: rep AUC **0.80** (pos_rate 0.17) → **the model ALREADY does implicit opponent
  modeling (anticipates opp switches well) → an opponent-prediction / world-model head is largely
  REDUNDANT.** (Probed the old run via a temporary `git stash` of the obs files since the new arch
  made the old checkpoint encoder-incompatible.)
- **`damage_taken`** (regression): rep r2 only +0.06 vs provided active_exp +0.02 → **the rep barely
  encodes damage magnitude beyond the mean; the SPREAD is NOT there → the ONE clear extraction gap.**
  (Caveat: realized damage is RNG-capped; the rep-vs-provided delta is the signal.)
- **`faint_soon`**: belief_quiet rep AUC 0.85 vs active_pko 0.45 (caveat: HP-confounded).
Conclusion: the model extracts speed + opponent-switching from what it has; the actionable gap is
DAMAGE MAGNITUDE/SPREAD + crit separation — exactly the user's RNG/crit intuition.

**BUILT (2026-06-09): gen3_incoming_crit_split_v1** (obs 3391→3409, ARCH bumped). Acts on the map +
the user's RNG pushback. In the incoming-damage belief block (`incoming_damage.py`): PER_MON 5→8,
INCOMING_DMG_DIM 33→51. (1) SPLIT P(KO) into `*_pko_nocrit` (modal line) and `*_pko_crit` (crit-
inclusive) per phys/spec channel — the gap = crit risk, so the model prices the modal outcome without
over-weighting uncontrollable crit RNG (crit was already computed, just unblended → ~0 perf cost,
benchmark confirms `_channel_threat` not a bottleneck). (2) `threat_revealed` per-mon scalar = the
dominant threat's `p_in_set` (1.0 revealed / <1 usage-prior guess) = the "how much are we guessing"
provenance flag (provide-the-fact per [[feedback_provide_vs_learn]]). Prober `decode_incoming_belief`
made layout-robust (handles old 5-field + new 8-field traces; adds active_pko_nocrit/threat_revealed
to IncomingBeliefView). All gates pass: 2135 unit tests, obs benchmark clean, full obs+prober+model
suite green. Pairs with [[project_incoming_damage_outcome]], [[project_plateau_diagnosis_2026_06_09]].

NEXT: train a run on the new arch and A/B vs the current arch (does the crit-split + provenance move
the tail-calibration / win-rate). Probe the NEW run with `faint_healthy`/`big_hit_incoming` (killed by
resource contention on the old box) + add `crit_split` / `prior_vs_revealed` probe targets (need the
8-field belief). NOT committed (awaiting /gen3ai-ship). Runs: ai_v5_6→ai_v5_6_stable_70m_0608; user
running ai_v5_switch_bias_N_0609.
