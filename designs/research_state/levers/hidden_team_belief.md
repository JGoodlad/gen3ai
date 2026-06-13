# Lever: Hidden-team belief (in-place belief slots + supervised aux head)

**Bucket:** L3 anticipation (amortized belief over the opponent's UNREVEALED party — NOT runtime search).
**Status:** 🛠 **BUILT (2026-06-13), NOT YET RUN** — branch `claude/belief-head` (ARCH `gen3_..._v1`,
`MODEL_CONFIG_VERSION` 16, `--opp-belief-aux-coef`). Falsify-after-build metric named; gated on a fresh
retrain to know if it HELPS the policy.

## Why this lever (provenance)
When the opponent-ACTION head was FALSIFIED two ways (VoI ≈ 0.03 mon; the trunk already models the
opp's *visible-state* action at 0.90 AUC — `[[project_opp_action_head_falsified]]`), the falsification
named a **separate, un-falsified gap**: the opponent's **HIDDEN team**. Gen 3 OU has **no team
preview**, so the ~3 unrevealed slots are genuinely ABSENT from the obs (a representation probe CANNOT
recover them — the exact opposite of the opp-action head). A pre-build **learnability probe** confirmed
the signal exists: conditioning on the revealed mons beats the marginal usage prior at predicting the
hidden ones by **+7pp recall / +8–10pp top-1** (a crude pairwise model — a neural head does better).
This is the one "predict-it" lever that passes the *unknown ∩ action-changing* test the opp-action head
failed. It also SUBSUMES the bench/switch-in half of the H3 surprise-OHKO gap (a "what unseen mon could
OHKO me" query is a hidden-team belief query).

## Known
- **Architecture (BUILT).** `BeliefSlots` fills the un-revealed opp team slots with distinct learned
  unknown-mon tokens *in-place* (not zeros), refined by the same 12-token transformer so **both heads
  attend over the imagined mons as party members**. `BeliefHead` aux-supervises the refined slot-tokens
  to predict each hidden mon's **species + moves**, matched **order-invariantly (Hungarian)** so the
  anonymous slots cover the hidden SET rather than chasing a reveal-shifting target. Labels are
  privileged (the opponent's full team from `battle2.team`), training-only, and NEVER enter the forward.
- **It learns.** Smoke (CPU, coef 0.2): `belief_species_acc` ≈ 0.08–0.16 vs ~0.003 chance after a
  handful of updates — above chance immediately.
- **Robust to run.** Off byte-identical; version-gated (`opp_belief_slots`); resume reads the coef back;
  `arch_toggles` threaded into the version gate at all 4 opponent-load sites (pool / stable / eval
  sentinel / distill teacher), so a belief-ON **self-play** run doesn't FATAL on its own snapshots;
  works for self-play AND stable play (the belief is internal to the forward, the obs interface is
  unchanged). Fail-loud on out-of-vocab labels + a non-contiguous believed mask. Fuzz-validated
  (`belief_labels_fuzz_test.py`, 1792 decisions: labels == actual hidden team + no leak).
- **Observable.** 12 TB metrics incl. `species_acc_above_chance`, moves P/R, coverage, k_mean, and the
  shared-trunk **grad-balance probe** (`belief_share`/`belief_policy_cosine`).

## Not-known (the honest gates)
- **Does it HELP the policy?** UNMEASURED. The belief is a *means*; the only thing that matters is
  whether the policy makes better decisions. `species_acc` rising is necessary, NOT sufficient.
- **Generalize vs memorize?** On self-play the opponents draw the same team pool, so `species_acc`
  could reflect pool memorization rather than transferable belief. Needs a held-out / cross-pool probe.
- **Balance.** `--opp-belief-aux-coef` / `--opp-belief-moves-weight` are empirical; the grad-balance
  probe + policy-health metrics are the tuning instruments, not a known-correct value.

## Pros
- Predicts genuinely-ABSENT info (unlike the falsified opp-action head) — measured +7pp learnable.
- Amortized (a feedforward aux head, no runtime search) — clears the L1–L4 gate.
- The belief representation feeds BOTH heads, so the policy/critic reason about the hidden party.

## Cons (write the honest caveats)
- **"Learnable but inconsequential" risk** (the dominant one): a perfect belief is inert if the policy
  doesn't ACT on it — and surprise-OHKO is ~as common in WINS (56%) as losses (52%), so the belief may
  be informative-but-not-pivotal. Pairs with the under-switching/commitment lever.
- Per-decision aux loss cost (~8.6s CPU at production; lower on CUDA but extra serial kernels). Off-path
  free.
- Moves prediction is a weak secondary signal (multi-label over ~370, ~4 positives) — species dominates.

## Next-test (falsify-after-build — the metric that MUST move)
A fresh `--opp-belief-aux-coef` retrain vs a coef-0 baseline:
1. `train/belief_species_acc_above_chance` climbs through training (the head learns). [necessary]
2. The **surprise-OHKO / hidden-mon `critic_blindspot` crater share falls** (prober `triage`/`scan`) AND
   win-rate is non-regressing — the decisive test that the belief HELPS. [sufficient]
3. A held-out belief-accuracy probe (build once traces exist) shows generalization, not pool memorization.
If (1) holds but (2) doesn't, the belief is the opp-action head's quieter cousin (present-but-unactioned)
→ the lever is the policy commitment side, not more belief.

## Re-verify
`python src/agents/training/poke_env_gaps/belief_labels_fuzz_test.py` (labels correct) · the
`train/belief_*` curves + `grad/belief_share` on the retrain · `models/saved_work/team_completion_learnability*.py`
(the pre-build learnability proxy).
