# K9(b) early-probe exclusion: what the excluded rows are (2026-10-06)

**Question.** After `gen3_behaviour_tie_identity_v1` (e8008c2d) a fresh run's FIRST update is judged in full
(the zero-init scorers make the forward selection-free). Why did `rb_x5ab_fm_s1006` still stop on the
excluded-share ceiling at its SECOND probe, and is there a deterministic rule that clears the rows without
clearing a tie that can move log π?

## The live evidence (read-only, from the run dirs)

| run (pin) | probe | excluded share | max \|Δ log π\| |
|---|---|---|---|
| `rb_x5ab_fm_s1006` (708dcb0a, `fatal`) | update 1 (202,496 steps; sb3 `n_updates` 10) | **0.281** (288 of 1,024): STOP | 2.4e-7 judged; full-buffer scan 0.268, max 2.0e-5 |
| `rb_x5ab_fm_s1006b` (same seed, `warn`) | update 1 | 0.271 | 4.8e-7 |
| `rb_x5ab_fm_s1006b` | updates 2–152 | 0.030–0.099 | |
| `rb_x5ab_fm_s1001`–`s1005` (708dcb0a) | update 1 | 0.039–0.088 | |
| `rb_x5ab_oracle_full_s1001`/`b`/`s1002`/`s1003` (**77245f51**, before e8008c2d) | **update 0** | 0.220 / 0.220 / 0.239 / 0.234 | 0 |

The oracle-full readings are the update-0 cold start that e8008c2d's selection-free rule already covers: those
runs were pinned BEFORE it. Only fixed_mass seed 1006 stopped after it, at update 1, and every other
fixed_mass seed read under 0.09 there. The dumps (`behaviour_violations.jsonl`) carry only the 200 largest
JUDGED rows and the 32 worst scan rows, never the excluded ones, so the excluded rows were reproduced on CPU.

## Method (CPU only, no GPU lease)

`measure.py` is the `k9_tie_identity_2026-10-05` harness with one change: `--weights-from` loads a state
dict before the rollout. A seeded 2,048-row complete-game rollout on the Rust collector (16 envs × 128 steps,
T2 eager on CPU, p2 a seeded random policy, 24 seeded pool teams), played and judged at the weights the
s1006 violation dumped (`models/rb_x5ab_fm_s1006/behaviour_violation_u10_policy.pt`, read only). Every row
goes through the probe forward under the tie-margin recorder. Every row the rule BEFORE excluded (no payload
identity, every pair of the X5 sort head) then has its near-tied units resolved the other way (joint and per
call), and its full masked log-probs are compared bit for bit. The same driver without `--weights-from` re-runs
the 2026-10-05 fresh / perturbed arms.

Two side checks (scratch drivers, numbers below):
- **Which X5 sort caller**: at the u10 weights, 231 of the 2,048 rows have a near tie at the per-mon move
  orders of `build_op_roster` (pairs at sorted positions 2–4: 192, the cut pair (5, 6): 75), 19 at
  `other_roster`, 9 at the active's move group, 4 at the species order. The tied presences sit at 0.38–0.56
  (a near-flat move posterior one update in) and at the 1e-9 floor (109 exact ties).
- **Is the set order free?** A seeded random permutation of the first K = 6 positions of every per-mon order
  (`build_op_roster`, `other_roster`, both, 3 seeds each) leaves the full masked log-probs of all 2,048 rows
  BIT-IDENTICAL; swapping the pair across the cut moves 2,002 rows (max \|Δ\| 2.7e-5). The op reads each
  mon's first K candidates as a SET (gathers per candidate, then presence-scaled max / sums over K), the code
  path blob's `topk` feeds, and the `topk` rule already counts only its k-th vs (k+1)-th value.

## The rule (`gen3_behaviour_tie_consumed_v1`)

`hypothesis_set.stable_order` takes a `consumed` declaration that changes nothing it returns. The recorder
reads it (`selection_sites.Rule.consumed`) and counts a sort pair only where the caller reads the order:

- `build_op_roster` and `other_roster`: `SetCuts((consequence_topk, entity_topk_seats))`. The prefix is a SET,
  so only the pair straddling a cut is a boundary.
- the species order: `consumed = k`, the hidden-slot count. Positions ≥ k are OTHER's tail, read as a set, so
  only the pairs up to and across k count (the module's own `order_gap`).
- the active's move group: None, every pair of the head, as before (its seats are read in order).

A declaration the recorder cannot read (a missing local, a wrong type, a cut the head of 7 does not reach) is
a typed `TieMarginError`, never "no restriction".

## Result: excluded rows of 2,048

| state | before (no identity, no consumption) | flip-identical | flip-DISTINCT | after (production at this commit) | distinct rows cleared |
|---|---|---|---|---|---|
| **fixed_mass at s1006's update-1 weights** | **312 (15.2 %)** | 274 | 38 | **120 (5.9 %)**: sort 84, fixed dmg 25, argmax 11 | **0** |
| fixed_mass / perturbed | 128 (6.3 %) | 117 | 11 | 78 (3.8 %), was 5.1 % under the identity rule alone | 0 |
| fixed_mass / fresh | 280 (13.7 %) | 280 | 0 | 0 (selection-free) | 0 |
| blob / perturbed | 70 (3.4 %) | 60 | 10 | 35 (1.7 %), unchanged: blob has no X5 sort | 0 |
| blob / fresh | 106 (5.2 %) | 106 | 0 | 0 (selection-free) | 0 |

Every run: 0 determinism failures on untouched rows, 0 no-op flips. `fixed_mass_s1006_u10.json`,
`fixed_mass_{fresh,perturbed}.json`, `blob_{fresh,perturbed}.json` hold the per-site breakdown.

**Read.**
1. **The early false stop is the X5 per-mon move ORDER, not near-flat logits.** One update in, the move
   posterior is near-flat, so many candidates INSIDE each mon's top six tie. The sort rule counted every
   adjacent pair of the head as a boundary, but the op reads that prefix as a set: those ties cannot move log π
   (bit-identical under any permutation). The rule excluded them anyway.
2. **The new rule clears exactly those rows and no distinct row, at every state measured.** The ties left are
   the cut pair, a fixed-damage-equals-HP threshold, and the dominant-move argmax, which the flips confirm can
   move log π.
3. **Projection to the live stop, UNVERIFIED.** This harness's opponents (a random p2) read 15.2 % at the s1006
   weights where the live run read 28.1 %. If the cleared share carries over (61 %), the live probe would have
   read about 11 %, under the 0.15 ceiling. That is a proportion, not a GPU measurement.

## Not claimed

- The X5 move group (the active's seats, read in order) and the cut pairs keep their exclusion. So do the
  1e-9-floor exact ties at a cut (two floor moves swapping in and out of the top six move log π by about 1e-9
  of a presence-scaled max, but no deterministic rule here proves a bound).
- The ceiling (0.15) is unchanged, and so is the rule that a planted mismatch on a judged row is FATAL in every
  phase. `tie_identity_integration_test` plants one at every arm, fresh and perturbed.

## Files

- `measure.py`: the driver. Run it from the repo root under `scripts/ops/mem_cap.sh 24`, with
  `CUDA_VISIBLE_DEVICES=` and `PYTHONPATH=src`. Arguments: `--arm fixed_mass --weights-from <state dict> --out …`,
  or `--arm {blob,fixed_mass} --weights {fresh,perturbed} --out …`. About 1 min per arm, peak 1.2 GB.
- `*.json`: the per-arm outputs (the 2026-10-05 harness's schema; `new_rule` is production at this commit).
- torch 2.8.0+cu126, commit: the one that lands `gen3_behaviour_tie_consumed_v1`.
