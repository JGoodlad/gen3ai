# K9(b) tie IDENTITY — which ties can move log π (2026-10-05)

**Question.** K9(b) excludes every row whose forward sits within a relative 2e-4 of a selection / threshold
cutoff (rule 8), and FATALs when the excluded share passes 0.15. `rb_x5ab_oracle_full_s1001`
(`--oracle-reveal full`, pin `77245f51`) stopped at its first probe on that ceiling — excluded 0.220, max
|Δ log π| exactly 0 on every row (`models/rb_x5ab_oracle_full_s1001/crashes/restart_err_20261005_055447_9a2069.txt`).
F-XC-5 (`../x5_cost_ablation_2026-10-04/README.md` §1) saw the fixed_mass arm at 0.54–0.58. Hypothesis
under test: many excluded rows are ties whose resolution cannot change anything log π reads.

**Method** (`measure.py`, CPU only, no GPU lease; `summarize.py` folds the per-arm files into `result.json`).
Per arm, one seeded complete-game rollout on the Rust collector: 16 envs × 128 steps = 2,048 rows,
self-check build, T2 eager on CPU, run seed 1001, p2 a seeded uniform-random policy on an external route,
24 seeded pool teams. Two weight states: `fresh` is the production init at seed 1001 built under
`single_thread_build` (unperturbed, so the zero-init action scorers are exactly 0); `perturbed` is the
testkit's seeded perturbation (a non-uniform policy, the stand-in for trained weights). Every row goes
through the learner's probe forward (train mode) under the tie-margin recorder. Then every row the rule
BEFORE excluded has each near-tied unit resolved the OTHER way by a flip forward: a top-k's k-th
candidate swapped for the best unselected one, an argmax resolved to the best other candidate, a
threshold's boolean flipped, a sorted pair swapped. Variants: all of a row's units at once (JOINT), then
one per call (SINGLE). A row is **value-identical** when its full masked log-prob vector is bit-identical
under every variant, otherwise **distinct**. Checks: the baseline forward reproduces bit for bit; every
row with no flip reproduces bit for bit in every variant (0 failures in all 8 runs); a flip that changed
nothing counts as distinct (0 such rows).

The flips test SOME of the reachable resolutions — the joint flip and each single call, not every subset
of a row's units. "Value-identical" is therefore the evidence for the hypothesis, not the rule. The rule
that shipped (`gen3_behaviour_tie_identity_v1`) is deterministic and composable, and does not rest on
the flips.

## Result — the rule BEFORE, by cutoff family and flip verdict (rows of 2,048)

| arm / weights | excluded | value-identical | distinct | by cutoff (identical / distinct) |
|---|---|---|---|---|
| blob / fresh | 106 (5.2 %) | 106 | 0 | dominant-move argmax 50/0 · belief top-K cutoff 40/0 · fixed dmg ≥ HP 16/0 |
| blob / perturbed | 70 (3.4 %) | 60 | 10 | argmax 37/4 · fixed dmg 24/0 · top-K 0/6 |
| oracle_species / fresh | 102 (5.0 %) | 102 | 0 | argmax 50/0 · top-K 36/0 · fixed dmg 16/0 |
| oracle_species / perturbed | 59 (2.9 %) | 21 | 38 | argmax 11/20 · top-K 3/18 · fixed dmg 7/0 |
| **oracle_full / fresh** | **514 (25.1 %)** | **514** | 0 | top-K 279/0 · argmax 181/0 · top-K + argmax 38/0 · fixed dmg 16/0 |
| oracle_full / perturbed | 129 (6.3 %) | 75 | 54 | argmax 73/43 · top-K 1/9 · both 1/1 · fixed dmg 0/1 |
| fixed_mass / fresh | 280 (13.7 %) | 280 | 0 | X5 order (argsort) 214/0 · argmax 46/0 · fixed dmg 15/0 · mixed 5/0 |
| fixed_mass / perturbed | 128 (6.3 %) | 117 | 11 | argsort 92/5 · argmax 24/3 · fixed dmg 1/3 |

A row at two cutoffs counts under both. "belief top-K cutoff" = `pointer_head.py:176` topk + `:177` ≥ + `damage_op_pairwise.py:321` / `:924` topk
(+ `damage_op_blocks.py:920` / `:1068`); "argmax" = `damage_op.py:918` / `:924`; "fixed dmg" =
`damage_kinds.py:130`; "argsort" = `hypothesis_set.py:193`.

**Read.**
1. **The oracle-full stop is a FIRST-UPDATE artifact.** At the fresh init it reproduces (25.1 % here, 22.0 %
   live). Every excluded row is value-identical for one reason: the three action scorers are zero-init,
   so every logit is exactly 0 and NO selection reaches log π. The ties are real and level. Full reveal
   pins each opponent mon's four moves at 0.99995, and every other legal move sits at the move prior's
   floor, exactly 0.02, because the learned correction is exactly zero at init. With K = 6 the cutoff
   falls inside that 0.02 block on every fully revealed mon. The live run, in warn mode since then, read
   **0.0498, 0.0527, 0.0771, 0.0566, 0.0566, 0.0811, 0.0645, 0.0537, 0.0527, 0.0566, 0.0615** over its updates
   1–11 (`models/rb_x5ab_oracle_full_s1001b/launcher_child.log`).
2. **A real share of trained-like ties are payload-identical.** Most are exact ties between two moves
   at the dominant-move argmax that gather the same accuracy (or the same belief weight). Typically two
   revealed moves have identical capped damage at weight 1.0. These are the "~2 points of teeth" the
   K9(b) doc named as an unbuilt refinement.
3. **The tied candidates of a top-K are distinct moves.** Flipping them moves log π once the scorers are
   live, so they stay excluded.
4. **fixed_mass F-XC-5 is not reproduced at HEAD on this harness**: 13.7 % fresh, 6.3 % perturbed (regime
   A's 0.54–0.58 was measured at `889add9d`, before F-X5-44's fix, `61be9ec9`, with a seeded self-play pool).
   Regime A at HEAD on the GPU stays **UNVERIFIED**.

## The rule AFTER (`gen3_behaviour_tie_identity_v1`) — excluded rows of 2,048

| arm / weights | before | after | rows the flips call DISTINCT that the new rule clears |
|---|---|---|---|
| blob / fresh | 5.2 % | **0** (selection-free) | 0 |
| blob / perturbed | 3.4 % | 1.7 % | 0 |
| oracle_species / fresh | 5.0 % | **0** | 0 |
| oracle_species / perturbed | 2.9 % | 2.3 % | 0 |
| oracle_full / fresh | 25.1 % | **0** | 0 |
| oracle_full / perturbed | 6.3 % | 3.6 % | 0 |
| fixed_mass / fresh | 13.7 % | **0** | 0 |
| fixed_mass / perturbed | 6.3 % | 5.1 % | 0 |

The payload rule alone (without the certificate) leaves the fresh rows at 4.0 / 3.8 / 19.6 / 12.6 % (blob /
species / full / fixed_mass): the fresh oracle-full ties are mostly distinct-move top-K ties that only the
scorers' zero init makes harmless.

**Unclaimed coverage** (excluded under the new rule, yet value-identical under the flips; left
excluded, since no deterministic rule here proves them): the fixed_mass stable order's exact π ties (94
perturbed rows), and the `fixed >= current HP` exact ties (the two branches agree at equality: 24 blob
perturbed rows). Each needs a declared rule of its own (a continuity-at-the-boundary rule for a
threshold; a slot-order-invariance proof for the X5 order).

## Files

- `measure.py`: the driver (`--arm {blob,oracle_species,oracle_full,fixed_mass} --weights {fresh,perturbed}`)
- `<arm>_<weights>.json`: the per-arm output; `result.json` is the fold (`summarize.py`)
- run: `PYTHONPATH=src python designs/research_state/measurements/k9_tie_identity_2026-10-05/measure.py --arm oracle_full --weights fresh --out …` from the repo root (the team pool is read cwd-relative), under `scripts/ops/mem_cap.sh 24`; about 1 to 2 min per arm on CPU, peak 1.7 GB
- torch 2.8.0+cu126, commit: the one that lands `gen3_behaviour_tie_identity_v1`
