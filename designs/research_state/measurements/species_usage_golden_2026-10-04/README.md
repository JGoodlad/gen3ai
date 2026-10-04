# The K9 learner-golden re-bake proof for `gen3_smogon_species_usage_weighted_v1` (F-X5-47), 2026-10-04

**What changed.** The Smogon species-usage marginal `gen3_data.priors.species_usage()` weighted each species by the
chaos `Raw count`, the UNWEIGHTED set count (pkmn/stats `p.raw.count++`). It is now each species' RATING-WEIGHTED set
total W = `Σ Abilities` (Smogon's `p.raw.weight`, read through the facade's existing `_weighted_count`). That is the
population every other Smogon prior is weighted by, including the chaos `Teammates` conditional that the co-occurrence
lift divides by this share. Nothing in `data/` changed: the marginal is derived at call time, and the acquisition tool
regenerates all seven Smogon files byte-identically (`out/data_unchanged.sha256`).
`out/parent_species_usage.json` is the parent's (`7bed4347`) marginal.

**Magnitude** (`scripts/shift.py`, `out/usage_shift.txt`). Over the 216 covered species, the new / old normalised share
runs ×0.19 to ×1.75 (median ×0.69). Over the top 25 it runs ×0.92 to ×1.19. The low-rated tail was inflated most
(Shuckle ×0.19, Manectric ×0.21, Altaria ×0.25). Tyranitar goes 0.0781 → 0.0863, Jirachi ×1.16, Raikou ×1.19, Gengar
×0.94. `build_species_usage_prior` moves by at most 0.0082 (L1 0.086). Against Smogon's own latest-month weighted
`usage` (an outside reference, one month), the top-25 median relative error falls from 0.064 to 0.038.

**Claim.** Both K9 arms moved ONLY because of the marginal's values.

**Method** (`scripts/proof.py`, torch `2.8.0+cu126`, CPU, the golden's own `compute`). `--hold <file>` replaces the
facade's `gen3_data.priors.species_usage` before the learner is built. It is the one source of the marginal's values:
`build_species_usage_prior` reads it at call time, for the op's `SPECIES_USAGE_PRIOR` and for the `T0SpeciesPrior` /
`BeliefHead` marginal and lift through `build_species_cooccur_prior`. The OLD tree is `git archive 7bed4347 src data
designs/production_config.json designs/baselines.json`, extracted to scratch. `scripts/compare.py` compares every
field: init / post params, every init / post group hash, and every pinned loss (exact). `scripts/run_proof.py` drives
every step below.

| step | what | result |
|---|---|---|
| K9(b) | the committed buffers under this tree (`scripts/k9b_read.py`) | FAIL on both (blob max \|Δ log π\| 0.0054, fixed_mass 0.0077); PASS on both with the parent's marginal held (4.8e-7) — `out/k9b_*_committed*.txt` |
| A | parent's buffers: this tree + parent marginal vs the RECORDED goldens | blob 103 / 103 fields identical (post `c4f63287…`); fixed_mass 99 / 99 (post `355c9da2…`) — `out/A_*` |
| rebuild | `rebuild-buffer --arm blob / fixed_mass` | the SAME games: obs / actions / masks / rewards / episode starts byte-identical; only the behaviour columns moved (blob log_probs 29 / 64 rows, max 0.0067; fixed_mass 35 / 64, max 0.017) — `out/buffer_diff_*.json` |
| B | rebuilt buffers: OLD tree vs this tree + parent marginal | blob 103 / 103 identical; fixed_mass 99 / 99 — `out/B_*_oldtree_vs_new_parentusage.txt` |
| C | `scripts/rebuild_with_usage.py`: rebuild holding the parent marginal vs the parent's buffers | byte-identical, every field, both arms — `out/C_*` |

So the parent and this tree differ in one update ONLY through the marginal's values (A, B). The rebuilt buffers'
behaviour columns come from the marginal too (C): the seeded behaviour learner reads it, and with it held at the
parent's values the rollout reproduces the parent's buffers bit for bit.

**Recorded** (`record`, `record --arm fixed_mass`): blob post `09222426…` (= `out/B_blob_new.json`), fixed_mass post
`50659234…` (= `out/B_fixed_mass_new.json`). K9(b) PASSES on both rebuilt buffers (blob `out/k9b_blob_rebuilt.txt`;
fixed_mass in its recorded `behaviour` block). fixed_mass's excluded share is 0 % (it was 6.25 %), and its coverage is
unchanged (OTHER_species live 36 / dead 28, OTHER_move 60 / 4).

**Role set** (`python -m main.belief_roles roles` on both trees, `scripts/roles_diff.py`, `out/roles_diff.txt`): 29
roles / 3,030 pairs (`daba9995…`) → 28 / 2,798 (`823b27be…`). Ice Punch drops (0.2565 → 0.2455 expected carriers,
0.0045 under the 0.25 bar, far outside the 1e-6 rule-8 band); nothing joins.

**Revert evidence** (`scripts/revert_check.py`, `out/revert_{code,denominator}.txt`): with `priors.py` at the parent,
5 of the 5 new tests fail. With only `species_usage`'s weight line back to `Raw count` (the guard kept), 5 of 5 fail.
There is no data revert, because no data file changed.

**Two routine-gate tests had data-dependent preconditions; each was attributed before it was changed**
(`scripts/precondition_attrib.py`, `out/precondition_attribution.txt`). Both pass with the parent's marginal held and
fail at this tree, and after the edit both pass under either marginal.
- `rust_rollout/consistency_test::test_a_row_near_a_tie_is_excluded_not_judged` assumed a unique minimum tie margin.
  Under the weighted marginal three probe rows share it exactly (3.7e-5), so the test now excludes the whole tie group
  (k = 3 of 16) and expects `rows_excluded == k`.
- `model/beatup_damage_test::test_unrevealed_defender_prices_the_expected_damage_exactly` held the typeless E[mult]
  column to `abs=1e-6`, but it reads 1 + 1.07e-6 (fp32 summation rounding, 9 ulps). The bound is now the fp32
  accumulation bound `n * 2^-24`.

**Reproduce** (from the new tree's root):

    python scripts/dump_usage.py --out out/parent_species_usage.json           # with PYTHONPATH=<old tree>/src
    python scripts/run_proof.py --phase {k9b,A,rebuild,diff,B,C} --old <old tree> \
        --parent <dir with the parent's learner_golden.json + both buffers> --scratch <dir>
