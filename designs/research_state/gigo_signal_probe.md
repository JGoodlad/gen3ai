# PRE-REGISTRATION — the GIGO accidental-signal probe (2026-08-18)

**Registered AFTER gen-14's §1 verdict (INFERIOR, Δ −38.30 CI [−55.0, −21.6], point estimate
strengthening with 4× data) and AFTER the §2 exoneration (event_seats rose +29.3% with all_off
stable ⇒ no revert), but BEFORE this probe computes anything.** With the frames exonerated, the
−38 lives among the remaining v91 bundle members, and this probe tests the one suspect with a
mechanism.

## The hypothesis

The pre-v91 magnitude column summed every residual tick on a move's target — sandstorm, burn,
poison, Leech Seed, recoil, Substitute self-cost — into that move's attributed damage. The bug
was therefore, by accident, **the event window's only residual-attrition signal**, and the
corrected window has no replacement by its own §3.2 admission ("residual damage is not an
event"; faint_cause covers the deaths, not the ongoing attrition). Gen-13 trained on the leak
and scored non-inferior; gen-14 trained on honest arithmetic and regressed with the exact
awareness profile of lost attrition history (blind-loss fraction 0.138 vs the 0.072 baseline,
lead-time down). Claim under test: **the leak was load-bearing for gen-13.**

## The probe (backward test — runs on GEN-13's side, from its pinned worktree `1fa4733`)

Recompute gen-13's event windows with the CORRECTED magnitude arithmetic via the offline
materializer path, feed gen-13's own checkpoint, and compare against its original observations
over ≥5,000 on-distribution decision states:

1. **Dependence reading:** masked-KL, flip-rate, and |dV| of (corrected obs vs original obs).
2. **Behavioral reading:** gen-13 playing eval battles WITH corrected windows vs WITH original
   windows — win-rate delta vs the standard bot set, battle-bootstrap CI.
3. **Coverage enumeration (no model):** list what the corrupted sums encoded (per-target
   residual presence, per turn, per move row) against the corrected window's columns — the
   expected verdict is "no home."

## Decision rules (fixed now)

- **CONVICTED** iff the dependence reading is at least the MEDIAN live edge family's ablation
  effect (the audit's own scale) **or** the behavioral win-rate delta's CI excludes 0.
  → The regression mechanism is "correct-but-poorer," and the remedy is to deliver honestly
  what the bug delivered by accident: a **residual-damage event row** (or per-row residual
  column) — a gen-15 rider candidate with its own pre-registered gate, in the cant_id/
  faint_cause closure pattern. NOT a revert of the fix (the arithmetic stays honest).
- **NULL** (both readings small) → the leak was not load-bearing; suspects reduce to the new
  columns (**lesion evals**: zero each column on gen-14's checkpoint and PLAY — only
  improvement-on-lesion convicts) and seed/run variance (**6M fresh-init forks**, viable
  because the gap opens by ~6M — gen-13-config vs gen-14-config vs minus-one-change arms).
- Either way the instrument hierarchy holds: post-hoc ablation = dependence; lesion-eval =
  harm-as-trained; training-time A/B = causation. None substitutes for another.

## Method lessons from the §1 arc, recorded with it

- **Naive per-node SEs on a jointly-fit ladder understate the contrast SE** (~40% here — tail
  nodes share anchors). Pair the refit (`c'Σc`) or bootstrap; the naive diagonal flipped a
  verdict.
- **Adding ladder games has a variance FLOOR**: only the frozen-pair component scales with
  games — the bot/sentinel edges and the fit's prior do not. 4× games bought 12% SE, not the
  projected 50%; the tie-break resolved on the point estimate moving, not the precision bought.
  Size future tie-breaks against the variance decomposition, not the game count.
- **`--backfill` on a complete ladder is a silent no-op** (idempotent-by-design filter) — a
  success report that measured nothing. Tripwire wanted: it should say "0 missing pairs — use
  the duplicate-append driver for variance reduction."
- The `hidden_opp` save: dV 0.0000 with flip-rate 39.6% — the pi-half/vf-half split confirmed
  as a measurement; a dV-only reading would have severed a live policy input. Read the policy
  columns before deleting anything that publishes to both heads.
- Watch-item (not a rule): `entity_pool` now carries 97.4% of the critic's route dependence
  with zero policy effect — the critic is converging toward ONE route; C1's aggregation-
  diversity concern in new form. Look at gen-15's battery; PV's offline gate is the standing
  instrument if diversity needs deliberate widening.
