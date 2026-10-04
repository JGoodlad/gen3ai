# The K9 learner-golden re-bake proof for `gen3_smogon_prior_denominator_v1` (F-X5-41), 2026-10-04

**What changed.** The Smogon move prior P(m in set | s) was the chaos `Moves[m]` (rating-WEIGHTED) over the
UNWEIGHTED `Raw count`. It is now `Moves[m] / W`, with W = `Σ Abilities`: Smogon's own `p.raw.weight`, which is
pkmn/stats `stats/src/stats.ts` `updateStats` and `reports.ts` L271. Only `data/pokemon/gen3_move_priors.json`
moved (`out/data_shift.txt`). `out/parent_gen3_move_priors.json` is the parent's (`61be9ec9`) file.

**Claim.** Both K9 arms moved ONLY because of the move prior's values.

**Method** (`scripts/proof.py`, torch `2.8.0+cu126`, CPU, the golden's own `compute`). `--prior <file>` replaces the
facade's `gen3_data.priors.move_raw` before the learner is built. That loader is the one source of the prior's
values (`build_move_prior_logits` -> `priors.moves` -> `move_raw`). The OLD tree is a detached worktree at `61be9ec9`.
`scripts/compare.py` compares every field: init / post params, every init / post group hash, every pinned loss
(exact).

| step | what | result |
|---|---|---|
| A | parent's buffers: this tree + parent prior vs the RECORDED goldens | blob 62 / 62 fields identical (post `50f12a23…`); fixed_mass 99 / 99 (post `e4f8dbab…`) — `out/A_*` |
| rebuild | K9(b) failed on both committed buffers (behaviour log-probs of the deflated prior) -> `rebuild-buffer --arm blob / fixed_mass` | games differ: blob 60 / 64 actions, fixed_mass 2 / 64 (`out/buffer_diff_*.json`) |
| B | rebuilt buffers: OLD tree vs this tree + parent prior | blob 103 / 103 identical (post `2c8aa75a…`); fixed_mass 99 / 99 (`847b42b3…`) — `out/B_*_oldtree_vs_new_parentprior.txt` |
| C | `scripts/rebuild_with_prior.py`: rebuild holding the parent prior vs the parent's buffers | fixed_mass byte-identical; blob: obs / actions / masks / rewards / starts identical, behaviour columns <= 7.15e-7 (F-X5-45's rounding) — `out/C_*` |

So the parent and this tree differ in one update ONLY through the move prior's values (A, B). The rebuilt buffers' new
games come from the prior too (C): the seeded behaviour learner reads it.

**Recorded** (`record`, `record --arm fixed_mass`): blob post `c4f63287…` (and blob's first `init_group_sha256`, 41
groups, F-X5-46), fixed_mass post `355c9da2…`. fixed_mass's K9(b) max |Δ| is 4.8e-7 with 6.25 % excluded. Its coverage
is unchanged (OTHER_species live 36 / dead 28, OTHER_move 60 / 4).

**Role set** (`python -m main.belief_roles roles`, `out/roles_{before,after}.json`, `out/roles_diff.txt`): 15 roles /
18 pairs (`4e3ab394…`) -> 29 / 3,030 (`daba9995…`); 14 join, none leave.

**Revert evidence** (`out/revert_{data,tool}.txt`): with the parent's data, 5 of the 8 new tests fail; with only the
tool's denominator reverted, 2 of 8 fail.

**Reproduce:**

    git worktree add --detach <old> 61be9ec9
    PYTHONPATH=<tree>/src python scripts/proof.py --label X --arm {blob,fixed_mass} --buffer <npz> \
        [--prior out/parent_gen3_move_priors.json] --out out/X.json
    python scripts/compare.py out/a.json out/b.json

**Two routine-gate tests had data-dependent preconditions; each was attributed before it was changed:**
- `collector_integration_test::test_the_trainees_keyed_draw_replays_from_its_key`: `len(rows) > 200` is a coverage
  floor on a seeded run. It reads 188 here and 219 with the parent's prior held (`scripts/collector_attrib.py`,
  `out/collector_attribution.txt`), so the drop is the prior's. The floor is now 150.
- `x5_opp_mon_axis_test::test_the_per_mon_selection_cut_joins_the_rule8_exclusion`: it assumed no golden-obs row is
  a near tie. On the rebuilt blob buffer 7 of 64 rows are exact ties (gap 0.0) under EITHER prior, and 0 are on the
  old buffer (`scripts/tie_attrib.py`, `out/tie_attribution.txt`). The cause is the buffer's new games, not the
  prior's values. The test now plants on the first non-tied row and requires the natural set plus that row.
