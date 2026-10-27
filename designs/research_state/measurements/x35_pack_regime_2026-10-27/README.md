# X35 pack-regime audit (2026-10-27): the training teams the pre-`d8d674b9` packer played wrong

**Question.** Every X35-frame arm (C0, D1, D3, N1-P, LR-F, and X48 when it runs) and X26c train at the production pin
`f4b25238` or a corpus-only pin on top of it. P\* (X26, 0 → 380M) trained at `9c033489`. Both pins predate the packer
fix `d8d674b99` (`gen3_team_pack_happiness_v1`, `gen3_team_pack_hptype_ivs_v1`, `gen3_team_pack_oracle_v1`). Which
training teams did they play differently from the team Showdown plays from the same paste? What share of each arm's
training DRAWS carried one? Can the difference bias any stated arm-vs-control conclusion?

**Method.** `pack_regime_audit.py`, 2 stages, all on CPU (about 2 minutes):
- `pack` runs under the OLD pin's own `src` (`git archive <pin> src`). It packs every pool team (through that pin's
  `TeamLoader`), every team of the three corpora at the tier the arm used, and the five Kakuna panels our side
  pilots. It packs exactly as the training path does (`fix_gen3_hp_ivs`, then `join_team`).
- `judge` runs under main's `showdown_pack.js` (Showdown's import, then the validator, then `Teams.pack`). It lists
  each field where our pack differs from Showdown's, then sorts each difference by what it does in a battle:
  - `frustration_bp`: Frustration at base power 1 instead of 102.
  - `return_bp`: none were found.
  - `ivs_hp_type_changed`: none were found.
  - `ivs_hp_same_type`: the same Hidden Power type and power. Up to 1 point moves between 2 stats.
  - `ivs_stat_only`: a `Hidden Power: <Type>` line on a mon without the move. Up to 1 point on up to 4 stats.
  - `happiness_inert`: `Happiness: 0` on a mon with neither Return nor Frustration. This has no gen-3 battle effect.

  Only `frustration_bp`, `return_bp` and `ivs_hp_type_changed` count as MATERIAL.
- Draw shares follow the recorded argv of each arm (`metadata.json` `original_command`) and the mix semantics in
  `designs/training/team_curriculum.md`:
  - trainee = s · corpus + (1 − s) · (0.9 · pool + 0.1 · sample);
  - opponent = s · corpus + (1 − s) · pool.
- `sft431_tally.py` counts, over every banked Kakuna `games.jsonl` under `~/gen3ai_archive` and `~/dev/gen3ai-reads`,
  the forum72 games where OUR side piloted `sft431` (`sft431_tally.out`).
- `first_pass/` holds the orchestrator's first pass (`pack_audit.py`, which counts differing teams but does not
  classify them), kept as the input this audit refines.

## Result

**Teams that differ, and the material ones.** The counts are the same at `f4b25238` and at `9c033489`, except for
the row marked †.

| set (tier) | teams | differ | material (Frustration at BP 1) | the rest |
|---|---:|---:|---:|---|
| pool | 719 | 4 († 38 at `9c033489`) | **1** (`pool_202` Snorlax) | 2 stat-only (Hidden Power line, no move), 1 inert; † plus 34 teams / 35 mons `ivs_hp_same_type` (the pre-10-19 teambuilder table) |
| sample teams (= home72, the trainee's 10 % bias) | 72 | 0 († 1, same-type) | 0 | |
| `w_forum_v1@ABC` (D1) | 1,981 | 49 | **4** | 35 teams stat-only, 11 inert (`Happiness: 0` with no Frustration or Return) |
| `d3_setmut_v1@ALL` (D3) | 2,876 | 19 († 122) | **8** | 7 stat-only, 4 inert; † plus 103 teams `ivs_hp_same_type` |
| `n1_proc_v1@ALL` (N1-P) | 12,000 | 75 | **75** | none: every Frustration set has no Happiness line, and Showdown's import reads that as 0 |
| forum72 panel | 72 | 1 | **1** (`sft431` Gyarados) | |
| home72, p72, heldout72_hpfix, rc72b | 72 each | 0 | 0 | |

The brief's "teams affected" column (4 / 49 / 19 / 75) counts every differing team. In battle, only 1 / 4 / 8 / 75
change what a move does. Each material team has exactly 1 Frustration mon.

**Draw shares of material teams.** A row lists 2 values when its "any difference" share differs between the pins.

| arm (pin) | trainee draws | opponent draws | battles with ≥ 1 | material, vs C0 | any difference (trainee / opponent) |
|---|---:|---:|---:|---:|---:|
| C0, LR-F, X26c, X48 (`f4b25238`); P\* (`9c033489`) | 0.125 % | 0.139 % | 0.264 % | — | 0.50 / 0.56 % (P\*: 4.9 / 5.3 %, almost all same-type swaps) |
| D1 (`440b61c3`) | 0.164 % | 0.171 % | 0.334 % | +0.04 pp per side | 1.49 / 1.52 % |
| D3 (`9e289ba4`) | 0.171 % | 0.181 % | 0.352 % | +0.05 pp per side | 0.55 / 0.59 % |
| N1-P (`5f8f6cfa`) | 0.375 % | 0.382 % | 0.756 % | +0.25 pp per side | 0.56 / 0.59 % |

P\*'s pool training had the same material defect at the same rate as C0. Its extra differences were all same-type
Hidden Power stat swaps.

**Our side piloting `sft431` in the Kakuna reads** (forum72 games in which our side drew `sft431`):

| read | its play pin's packer | our side on `sft431` |
|---|---|---|
| D1 final (D1 arms, C0) | pre-fix | 0 / 1,400 |
| D3 look 1 (D3: post-fix `2cc864f2f`; C0 banked from D1's read: pre-fix) | crosses the split | 0 / 420 for D3, 0 for C0 |
| N1-P look 1 (`46b9e572c` / `de15dfc7b`) | post-fix, both arms | 2 / 1,920 so far (C0 s3501 only) |
| x26c read (P\*, X26c 480M; `e05c45eeb`) | pre-fix | 6 / 400 each, at the same seeds |
| H-CTX rc72b (`905c29f5b`) | post-fix | 4 / 720 |
| D3 look 2 (the hybrid `c4a5c37b6`) | pre-fix (checked: `happiness or ""`), the same as C0's bank | to come |
| the whole archive | | 72 / 9,583 (0.75 %); our side won 22 of the 72 |

## Reading: the bias bound

**Training side.** An arm-vs-C0 contrast can only be biased by how much the defect moved each arm's outcome. Take the
lever's own size as a Lipschitz scale:
- Each lever moves a share s = 0.3–0.5 of all draws to a new team distribution. No measured effect leaves ±5.2 pp
  (the widest end among D1, D3 and N1-P on unfamiliar teams). So the outcome moves at most about 17 pp per unit of
  draw share.
- The defect moved 0.26 % (C0) to 0.76 % (N1-P) of battles. Even with no cancellation between the arms,
  |bias| ≤ 17 pp × (0.0076 + 0.0026) ≈ **0.17 pp** for N1-P, and ≈ 0.10 pp for D1 and D3. Counting only the
  differential gives 0.08 / 0.012 / 0.015 pp.
- Both figures are loose, because the defect changes one move of one mon (1 of 24 moves), which is much gentler than
  replacing a team.
- The reads resolve ±3–4 pp. **No stated conclusion moves.** D1 final (UNDECIDED), D3 look 1 (CONTINUE), N1-P look 1
  (in flight), LR-F and X48 (no mix: exposure identical to C0's, differential 0), X26c (FLAT; the same material
  exposure as P\*).

**What the defect taught.** Before `d8d674b9`, EVERY Frustration set in training played base power 1. In the pool,
none had a correct `Happiness: 0` pack. So P\* and every X35 arm learned that Frustration does nothing, on both sides.
The damage op meanwhile prices it at 102 (`damage_tables.py`, happiness assumed 0). A post-fix eval pin shows them a
102-power Frustration they never trained on. This is the same for every arm on one eval pin. It touches F only
through `sft431`, about 1 game in 72 at most.

**Eval side.**
- **Within one read.** Every read so far has its arms on one OUR-side regime, except D3 look 1. D3 look 1 crossed the
  split with zero `sft431` games on either side, so the split had no effect.
- **The goal-(b) headline** (P\* 0.487 on U) was read pre-fix: our side drew `sft431` in 6 of 400 F games. Hard bound:
  6 / 1,200 = 0.5 pp of U if every one of those games flipped. Kakuna's legacy pack crippled the same Gyarados on
  its own side in the same read, so the two partly offset. The realistic effect is ≪ 0.1 pp.

## FINDINGs

1. The `f4b25238` family launches WITHOUT the pack oracle (the guard exists only from `d8d674b9`). A new
   corpus-only pin inherits the legacy packer silently. The mitigation is the precondition in the recommendation
   below.
2. The 10-27 GIGO FIX entry's "Our own side was never affected" holds only for reads at a post-`d8d674b9` pin. Every
   pre-fix read had our side pilot `sft431` at base power 1 (72 of 9,583 banked forum72 games).
3. The 10-27 Hidden Power AUDIT's "byte for byte as Showdown does … in training" holds from `f4b25238` on. At P\*'s
   pin `9c033489`, the teambuilder's pre-10-19 table filled 35 pool mons and 106 `d3_setmut_v1` mons with other IVs
   of the same type and power.
4. `sft431` is the only material file on any Kakuna panel. home72, p72, heldout72_hpfix and rc72b pack identically
   under both pins.

## Recommendation for future arms (X63, X62, a PR-1′ control)

**Keep forks that read against a banked `f4b25238` control on that family** (production pin + one corpus-only
commit, as every X35 arm). This covers X63 against X26c, and X62 against C0.

**The trade-off.**
- **Staying.** The bias is ≤ 0.2 pp against reads that resolve ±3 pp.
- **Moving to a post-fix main pin.** That changes at least 6 training inputs, not just the packer:
  - `d8d674b9`: the α/β intent-label mask, the op's consumed item, the silent volatile ends;
  - the 10-26 engine batch: the lock-in and recharge counters, the stall-forfeit row, the Species Clause.

  The arm-vs-control contrast would then measure a regime change, not the lever.
- **A packer-only backport pin.** It would buy back ≤ 0.2 pp, at the price of re-training the control (C0′ about
  13 GPU-h for 3 seeds; X26c′ more).

**Two conditions.**
- **(a) Precondition.** Each new corpus's legacy-pack material share is computed with `pack_regime_audit.py` at its
  pin and recorded in its PREREG. Above 2 % of its teams (3× N1-P's 0.63 %), the arm needs a packer-backport pin and
  a matched control, or a bound re-derived and accepted in the PREREG.
- **(b) One eval regime.** A read never splits its arms across OUR-side pack regimes on a panel with a material file
  (today forum72). The play pin sets the regime.

**Where the post-fix main pin applies.** A fresh run with no banked control (X53-N1's arms, the next era baseline)
trains at a post-fix main pin, its arms pinned together. PR-1′ is an evaluation of existing finals, so (b) is its
only condition: it plays at one post-`b0441623e` pin, which is also post-fix on our side.

**Evidence:**
- `result_f4b2.json` and `result_9c03.json`: every differing team with its diffs and classes, input digests, and
  draw shares;
- `sft431_tally.out`;
- the guard test `src/utils/teambuilder_pack_guard_integration_test.py`: every `Gen3Teambuilder` path (pool, bias,
  `--team-mix`) raises on the legacy pack of a `Happiness: 0` Frustration team. Reverting the 3 calls fails 3 of its
  6 tests (executed).
