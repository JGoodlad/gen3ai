# Attack type-mismatch (wrong-effectiveness move pick)

**Status:** ✅ confirmed · **Ledger id:** H2 · **Slice:** ARM A / ATTACK / move-selection

One-line claim: *the policy sometimes confidently fires a resisted/immune move while a damaging
(neutral-or-SE) move sits in the SAME mon's kit — a representation-USAGE gap (the effectiveness
signal is in obs, the policy mis-acts on it), not a reward/critic blunder.*

## Known (cleared the honesty gates)
- **The immune sub-cell is the cleanest DISCRIMINATOR in the ATTACK slice, but FAILS the
  recoverability gate** — it is a confirmed, learned, in-obs representation error whose per-decision
  win-rate headroom is LOW (14% reducible by the omniscient referee). Real, but secondary (<1% wr),
  consistent with H2's original call. It discriminates more than its "3–5 decisions" count suggested
  (98 confident playable-loss firings), but most of that is outcome-correlation, not craters.
  - DISCRIMINATION (Gate 1, ai_v5_11, model-free decode of the action-aligned move-multiplier
    `mm_off`): `immune_pick_dmg_avail` (chose a move the obs prices at mult==0 while a ≥1× move was
    legal) fires **0.86% in losses vs 0.32% in wins** all-positions; restricting to **V≥0 playable**
    positions it is **0.86% loss vs 0.32% win, Δ=+0.0053, z=7.0** — the delta SURVIVES (in fact rises)
    outcome-conditioning, the opposite of a lost-position symptom.
  - GROUND TRUTH: **93%** of these obs-mult==0 picks dealt literal 0 damage → the obs immune flag was
    correct; this is NOT ability-uncertainty noise. 65% were confident (softmax-chosen ≥ 0.3, mean 0.36).
  - CONCENTRATED → learnable: 98 playable-loss firings reduce to two canonical immunities —
    **Earthquake into Flying/Levitate** (62%: Skarmory 28, Zapdos, Aerodactyl, Gyarados, Claydol,
    Salamence) and **Thunder(bolt) into Ground/Volt-Absorb** (38%: Jolteon, Dugtrio, Swampert).
  - **RECOVERABILITY (Gate MISTAKE, omniscient re-roll falsifier) — the kill caveat.** Aggregate
    falsify of 14 immune-picks (V≥0, common-random-number re-rolls): **only 14% reducible**
    (MISTAKE 7% + MIXED 7%), **71% NEUTRAL, 14% LUCK**. Best-alt material margin **median 0.04**,
    mean 0.20 mon-equiv — the single wasted turn rarely turns the omniscient game. One clean worked
    MISTAKE (Gyarados EQ into Claydol/Levitate: Substitute +0.86, z≈10, anchor_delta −14.4) exists,
    but it is the exception. **So the z=7 LOSS>WIN discrimination is mostly outcome-CORRELATION
    (you fire EQ at Skarmory *because* you're walled — a symptom that survived the V≥0 filter
    incompletely), NOT a per-decision crater.** This reconciles with H2's original "small/<1%".
  - CROSS-RUN STABLE: ai_v5_12 (live, same arch) replicates — immune-pick V≥0 **Δ=+0.0158, z=6.8**
    (larger than ai_v5_11). Both runs z≈7, positive, playable-filtered.
- **It is a REPRESENTATION-USAGE gap, NOT a coverage gap.** The active-move type multiplier ×4 is
  ALREADY in the obs in REQUEST order (reactive.py `moves_dmg_multiplier[i]=mult/4.0`, aligned to
  action logit 6+i — see observation/CLAUDE.md). The model HAS the immunity signal (it reads 0.0) and
  confidently fires anyway. Reward + critic are correct here (this is not the self-KO over-valuation).
- BUCKET = **L2** (representation refinement). The fact is in obs; what's missing is decisiveness on
  the immune (0×) edge — a continuous `mult/4` puts 0× at the bottom of a smooth ramp, easy to wash out.

## Not-known (what would resolve / size it)
- Win-rate headroom: ~0.5–1% of decisions even at the playable level → fixing it likely moves wr <1%.
  This is a SECONDARY, cheap, stackable lever, not a primary one. The exact wr lift needs the A/B.
- Does a sharper encoding (a hard `is_immune` 0/1 flag, or a non-linear penalty on 0×) actually drop
  the immune-pick rate post-retrain without harming the calibrated ability-prior cases (firing EQ at a
  maybe-Levitate mon BEFORE Levitate is revealed is a legitimate priced guess, NOT this cell — this
  cell already filters to obs-mult==0, i.e. confirmed/by-type immunity).

## Pros
- Cheap, double-confirmed (firing-rate discrimination + omniscient falsifier + 93% zero-damage ground
  truth), cross-run stable, concentrated (2 immunity families) → genuinely learnable / amortizable (L2).
- Clean A/B: a sharper effectiveness/immune encoding ON vs OFF; distinct class from the reward levers.

## Cons (the kills around it — the gate works)
- **`hit_into_switch` KILLED**: fires MORE in wins (0.157 loss vs 0.171 win, z=−6.1) — aggressive
  attacking that forces opp switches correlates with WINNING. Not a mistake; the H3 "higher-in-wins" trap.
- **`resisted_pick_neutral_avail` KILLED**: negative delta (more in wins) — picking a neutral over a
  slightly-better neutral is normal play (STAB / secondary / PP).
- **`suboptimal_eff_pick` CONTAMINATED**: discriminates (Δ+0.018 z=7 playable) but **30% are NOT
  mistakes** — a high-BP neutral (Return/Double-Edge 120×1) out-damages a low-BP SE (HP 70×2). Only
  ~70% genuinely dominated. The immune cell has no such escape (0 damage is always dominated).
- Small absolute headroom (a fraction of a %); immune-attack cluster partly lives in won/lost games.
- Outgoing-KO mispredict (used a non-lethal when lethal existed) could NOT be cleanly measured here —
  this arch (`gen3_incoming_crit_split_v1`) has no OUTGOING-KO belief in obs (that's L3 anticipation),
  so it isn't a model-free decode; it falls to the gen3_outgoing_ko_v1 lever, not this one.

## Next test
- Sharpen the per-move-vs-active effectiveness in the action-aligned move-effect obs block (a hard
  `is_immune`/`is_resisted` flag OR a non-linear 0× emphasis), A/B the **immune-pick rate per playable
  loss** (should fall toward 0) with **no win-rate regression** and no regression on the legitimate
  ability-prior (unrevealed-Levitate) guesses. Re-verify via `attack_slice_sweep.py` + `falsify`.
