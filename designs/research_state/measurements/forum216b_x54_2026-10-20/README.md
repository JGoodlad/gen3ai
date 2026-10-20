# forum216b — X54's sealed confirmation panel (registered 2026-10-20)

**Status: REGISTERED and SEALED.** Registered in [`heldout_panels.json`](../../heldout_panels.json) as
`forum216b` (`held_out_from: ["training"]`) and sealed in code (`main.heldout_panels.SEALED`): no meter may play it
until X54's registration unseals it. It was carved on the orchestrator's ruling on FINDING 2 of
[`design_beat_metamon_path.md`](../../../endstate/design_beat_metamon_path.md) (2026-10-20), BEFORE X35's D1 arm
launched, so that the Metamon goal has a test that no lever was chosen on.

**Why it exists.** forum72, heldout72_hpfix and P72 have steered X37a, TSR, X37b and the D1 ranking. A goal claimed on
them carries the selection. `w_forum_v1@ABC` held every tier-A / B forum team outside the registered panels, so once
D1 trained on it no clean human-built A / B team would have been left. This panel is carved from those teams and
`w_forum_v1` is rebuilt without them and without their near neighbours.

## The draw rule (fixed before the draw; `draw_forum216b.py`, `--check` reproduces)

- **Source:** `w_forum_v1`'s tier AB as committed at `b3886800c`, before the carve: 559 teams, data sha256
  `4af495c4…7469`, AB tier sha256 `fa1219f6…1095`. The corpus had already applied every one of its rules (verbatim
  pastes, Hidden Power repair, pool-exact, the panels and their near neighbours, the validator, the engine, duplicates,
  the strength cut and the curator cap).
- **Strata:** the `ladder_weights_v1` rule, i.e. the best SPECIES overlap with one sample team (home72 + away): CORE 6,
  NEAR 4–5, FAR ≤ 3. Quotas are forum72's own counts times 3: **CORE 12 / NEAR 162 / FAR 42**. The function reproduces
  forum72's committed strata exactly (4 / 54 / 14, every id).
- **Order:** the AB ids sorted, then shuffled once with `random.Random(5400)` and taken in that order.
- **Skip** a candidate if its stratum is full, or if its curator already has **42** teams (forum72's cap of 14 per 72,
  times 3).
- **Reject** a candidate (and take the next) if any of these holds:
  - the pinned validator rejects it;
  - a Hidden Power does not play its named type at 70;
  - it is an exact team of, or shares ≥ 4 mon sets with ONE team of: the 719-team pool, home72, away, heldout72,
    heldout72_hpfix, forum72, P72, or a team drawn earlier.

**Result:** 216 teams, content digest `7831e2d2…3d02` (`gen3_team_set_digest_v1`), A 135 / B 81. 294 candidates were
read: 216 accepted, 63 skipped because their stratum was full, 15 rejected (7 near a pool team, 8 near an earlier
draw). The largest curator has 27 teams, so the cap never bound. Checked after the draw (`forum216b_manifest.json`
`overlap`): the most mon sets any team shares with the pool or a registered panel is **3**, and the same holds inside
the panel. So there is no copy or near copy (≥ 4 shared full sets) of a pool, home72, away, forum72, heldout72,
heldout72_hpfix or P72 team. The panel duplicate guard (`gen3_panel_duplicate_guard_v1`) passes: no team appears
twice.

## Why 216: X54's power (`power.py` → `power.json`)

X54's bar is a 95 % lower bound > 0.50 on the symmetric forum216b cell, Kakuna's best against ours. The planning model
is `Var(mean) = τ² / T + p(1 − p) / G`, where τ² = Var(a_t + b_t) is the per-team piloting interaction (team strength
cancels in a symmetric cell) that a team-clustered interval charges.

- **τ² measured on X37a's forum72 cell** (X26 @161M vs Kakuna, 400 games, 68 teams on both sides): the method-of-moments
  estimate is **−0.001**, i.e. 0, with a one-sided 95 % upper bound of **0.033**.
- P72's cell reads 0.039. P72 is procedural and materially more varied, so it is reported but not planned on.
- **The rule:** take the smallest T in {72, 108, 144, 216, 288} that gives ≥ 80 % power at the planning truth p = 0.55
  with G ≤ 2,400 games, under the PESSIMISTIC τ² of 0.033. Games can be added at X54; teams cannot, once the panel is
  registered.
- **T = 216** gives **0.82 at G = 1,600 and 0.88 at G = 2,400**. T = 144 gives 0.79 at G = 2,400, which fails the rule.
- If τ² is in fact near its point estimate of 0, the team count does not bind at all, and power at p = 0.55 is 0.98
  at G = 1,600.
- **At p = 0.53 no panel size reaches 80 %** under the pessimistic τ² (0.47 at T = 216, G = 2,400). At that effect,
  X54 is limited by games and by the interval it chooses (beat-Metamon FINDING 7), not by the panel.

## What the carve did to D1's corpus (`w_forum_v1`, rebuilt in place by its own builder)

| | before (`b3886800c`) | after |
|---|---|---|
| teams (A ⊂ AB ⊂ ABC) | 332 ⊂ 559 ⊂ 2,424 | **169 ⊂ 305 ⊂ 1,982** |
| ABC tier sha256 | `7bcdc835c1077fb8…f83d` | **`73a76e808431f5c4…0bdc4`** |
| data sha256 | `4af495c49a2df026…7469` | **`7f42d95f7f08f667…1336`** |
| `panel` drops | 249 | 556 (+216 drawn, +91 near neighbours of them: A 28, B 10, C 53) |
| curator cap | 242 | 198. The cap is 10 % of the capped total, so it shrank with the corpus and cut 135 more C-tier teams |

The C tier is re-shuffled because the seeded order runs over a different id list. As a result, **498 of the 1,982
teams were not in the old corpus**: they are C-tier teams from the four capped curators, re-selected. **The rebuilt corpus is NOT
soaked yet**: the committed `soak_selfcheck.json` is the 2,424-team corpus's, and 323 of the new corpus's teams are in
no soaked version. X35's launch pre-check 2d (`kit corpus-check`) therefore HOLDS D1 until a PASSing soak naming the
new data sha is committed (`python -m utils.team_corpora.soak w_forum_v1 --tier ABC --rounds 10 --out
src/utils/team_corpora/w_forum_v1/soak_selfcheck.json`, under `scripts/ops/mem_cap.sh`).

## Hazards (FINDINGS)

- The panel is sealed only where a panel is played BY NAME (`main.h2h --panel`). A `main.anchors` config names team
  DIRECTORIES, and the guard cannot see those. The panel is installed in no Metamon cache and no config; X54's
  registration does that, and unseals it in the same commit.
- The panel removes 45 % of the corpus's AB teams and 18 % of ABC. D1 now widens by 1,982 teams, not 2,424.
