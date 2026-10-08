# K9(b) static-screen stop at update 1480: one species' move order, repeated in every row (2026-10-08)

**Question.** `rb_st_static_s1001` (`--token-encoding static --belief-tokens fixed_mass`, pinned at `6c6d2e09`)
stopped at update 1480 (14,653,328 steps) on the K9(b) excluded-share ceiling alone: 0.175 against 0.15, with
845 of 1,024 rows judged and a judged max |Δ log π| of 8.6e-6 against the 1e-4 bar. Its other 36 probes read a
median of 0.033 and a max of 0.062. Which selection sites produced the excluded rows? Can those ties move log π?
Is the site still there at HEAD? Is there a deterministic clearance in the `gen3_behaviour_tie_consumed_v1`
(`fe237eac`) style?

## Method (CPU only, no GPU lease, at the pin)

The dump (`behaviour_violations.jsonl`) carries only the 200 largest JUDGED rows, so the excluded rows were
reproduced on CPU. A detached worktree at `6c6d2e09` was used, never the training agent's. The harness is
`k9_early_probe_2026-10-06/measure.py`, with one added arm: `static_fm` = `fixed_mass` + `token_encoding static`.
It plays a seeded 2,048-row complete-game rollout on the Rust collector (16 envs × 128 steps, T2 eager on CPU,
p2 a seeded random policy, seeded pool teams) at the weights the violation dumped
(`models/rb_st_static_s1001/behaviour_violation_u1480_policy.pt`, read only). It then runs the learner's probe
forward under the tie-margin recorder.

- `before_after.py`: the k9_early_probe harness as is (the rule before both clearances, flip on every pair).
  Output: `before_after_u1480.json`.
- `sites.py`: for every row the PRODUCTION recorder excludes, it records the `stable_order` CALLER (from the
  issuing frame), the counted pair, its two candidates and keys, and the mon's class (addressable, believed,
  hypothesis, active, the hypothesis species). It then resolves ONLY the production-counted near pairs the
  other way (joint, plus one variant per call) and compares the full masked log-probs bit for bit. A no-op flip
  keeps a row distinct. The flip forward's call sequence is asserted equal to the recorder's. Outputs:
  `sites_u1480.json`, plus the same run at the run's 13M, 14M and 15M checkpoints (`sites_ckpt_*.json`).

Run each from the root of a `6c6d2e09` checkout under `scripts/ops/mem_cap.sh 24`, with
`CUDA_VISIBLE_DEVICES=` and `PYTHONPATH=src`. Pass `--weights-from <pt|zip> --out <json>` (`before_after.py` also
takes `--arm static_fm`). Each run takes about 3 min at a peak of 1.4 GB.

## Result

Production rule at the pin, 2,048 rows, 0 determinism failures:

| weights | excluded | largest class |
|---|---|---|
| checkpoint 13,000,092 | 41 (2.0 %) | a revealed mon's cut pair (12), the dominant-move argmax (16) |
| checkpoint 14,000,046 | 19 (0.9 %) | scattered |
| **u1480 dump (≈ 14.65M)** | **209 (10.2 %)** | **154: one hypothesis species' cut pair** |
| checkpoint 15,000,107 (after the resume) | 35 (1.7 %) | the fixed-damage-equals-HP threshold (36) |

At u1480 the exclusions break down as follows:

| site (caller) | rows | flip | max \|Δ log π\| |
|---|---|---|---|
| `hypothesis_set.py` argsort ← `build_op_roster`, cut pair (5, 6), a **Salamence (373) hypothesis slot**: Hydro Pump (56) vs Hidden Power Grass (363) at π 0.22290 vs 0.22288 (relative gap 8.3e-5 < 2e-4), the SAME two values in every row, in all 16 envs | 154 | **distinct** | 1.25e-2 (median 1.3e-3) |
| same caller, cut pair, revealed or fainted mon, two typed Hidden Power channels at bit-EQUAL presence ≈ 1e-9 (one env) | 38 | identical | 0 |
| `damage_op.py:989/995` argmax (dominant move) | 10 | distinct | 1.6e-3 |
| `damage_kinds.py:130` (fixed damage ≥ HP) | 4 (12 near calls) | identical | 0 |
| others: two revealed mons' cut pairs, one `other_roster` cut pair (one of the 154 also carries a `species_set` pair) | 3 | distinct | ≤ 5e-3 |

The rule before both clearances excluded 314 rows (15.3 %); production at the pin excludes 209 (10.2 %).

## Read

1. **Site:** the X5 per-mon move order's CUT pair (`hypothesis_tokens.build_op_roster` → `hypothesis_set.
   stable_order`, `consumed = SetCuts((6, 6))`). `gen3_behaviour_tie_consumed_v1` keeps this pair counted ON
   PURPOSE: which candidate sits inside the op's top six is read.
2. **Can it move log π: YES.** Swapping Hydro Pump and Hidden Power Grass across the cut moves the full masked
   log-probs of every one of the 154 rows, by up to 1.25e-2. The exclusion is correct, and no identity or
   consumption argument clears it. The 38 typed-HP rows are bit-identical under the flip, but as
   `k9_early_probe_2026-10-06` already noted for the 1e-9-floor ties, no deterministic rule here proves a
   bound: their payloads (the type) differ.
3. **Why static, why lumpy.** Under `--token-encoding static`, a hypothesis row reads no board fact
   (`gen3_static_tokens_v1`): its token, and so its composed move posterior, is a function of (weights,
   species) only. A near tie in one species' move order is therefore not a thin per-row slice, as under
   legacy. It sits in EVERY row where that species is a hypothesis slot. So the excluded share is lumpy: when
   the weights put one commonly-hypothesised species' cut pair inside 2e-4, the share jumps by that species'
   hypothesis frequency, and one update either side it is gone. The neighbouring weights read 0.9 % (14M) and
   1.7 % (15M) in the same harness. The resumed run finished to 15M with 22 checks passed.
4. **HEAD:** the site is unchanged. `build_op_roster`'s `stable_order(key, sel, consumed=SetCuts(cuts))` and the
   static hypothesis tokens (`static_hypothesis_tokens`, a dex-table gather) are as at the pin. The v144/v145
   diffs in `hypothesis_tokens.py` touch only `max_by_index` and the speed-spread fields.

## Not claimed

- This harness (a random p2) reads 10.2 % where the live probe read 17.5 %. The site mix is measured here, not in
  the live rows (the dump never stores excluded rows). That the live 17.5 % has the same composition is
  **UNVERIFIED**, although 154 of 209 here comes from one species/pair at the dumped weights.
- No fix is built. The dominant class CAN move log π, so a `fe237eac`-style clearance ("a tie that provably cannot
  move log π is not counted") does not apply to it. The ceiling (0.15), the margin (2e-4) and judged-row FATAL are
  untouched.

## Files

`before_after.py`, `sites.py` (drivers; they load `../k9_early_probe_2026-10-06/measure.py` from the checkout they
run in), `before_after_u1480.json`, `sites_u1480.json`, `sites_ckpt_{13000092,14000046,15000107}.json`.
torch 2.8.0+cu126; code: `6c6d2e09`.
