# Nature / EV label coverage: why the IV-31 inversion missed half the pool (2026-09-29)

The evidence behind `gen3_true_spread_labels_v1` (M5 Lane C findings F-LC-5 and F-LC-6; TECH_DEBT rows
closed the same day). Base commit `74393cd6`.

## The question

Lane C measured that 54.6 % of the training pool's revealed-slot decisions carried no `belief_nature` /
`belief_ev` label (46,678 of 85,554), against 1.2 % on the ladder corpus. The suspected cause was the
pool's Hidden Power IV adjustment. The brief asked for the cause to be verified, not assumed.

## Method

`verify_inversion.py` takes every mon of every team, through the same packing training uses
(`main.rust_core_cutover.envs.packed_teams`: the pool through `Gen3Teambuilder`, the HP IV fix included;
the ladder corpus `full` tier). For each mon it computes the five derived stats from the declared set
with Showdown's formula (level, IVs, EVs, nature), runs the old `invert_nature_evs` on them, and compares
the result with the declared nature and stat-effective EVs (`4·⌊ev/4⌋`). `classify_iv31_mismatches.py`
splits the wrong answers on IV-31 mons. Both scripts import the old function from
`git show 74393cd6:src/agents/model/belief_tables.py > old_belief_tables.py`. No battle is played. The
figures are per mon, not per decision.

## Result

| | pool (719 teams) | ladder (22,813 teams) |
|---|---|---|
| mons | 4,314 | 136,878 |
| mons with an IV ≠ 31 stat | 2,683 | 2,574 |
| **no label (inversion found nothing)** | **2,406 (55.8 %), all IV ≠ 31** | **2,573 (1.9 %), all IV ≠ 31** |
| labelled, and equal to the declared set | 1,560 | 130,944 |
| labelled but WRONG: IV ≠ 31 (EVs off, 17 natures off on the pool) | 277 | 1 |
| labelled but WRONG nature: IV 31 | 71 | 3,352 |
| labelled but WRONG EVs only: IV 31 | 0 | 6 |
| level ≠ 100 | 0 | 0 |

- **The cause is verified.** Every unlabelled mon has an IV-30 stat, and no IV-31 mon fails. An IV-30
  stat is one point lower before the nature. With EVs in it the inversion absorbs the point by taking
  4 EVs off, which is a wrong label. With 0 EVs in it there is no decomposition, so the slot was
  unlabelled. 711 of the pool's 719 teams run Hidden Power.
- **The inversion is not identifiable, even at IV 31.** Its `Σ ≤ 510` check covers the five non-HP
  EVs only. A set with 252 HP EVs therefore has other (nature, EV) decompositions that fit the budget,
  and the inversion picks one by Smogon nature prior. Example: Bold Zapdos `252 HP / 136 Def / 84 SpD /
  36 Spe` inverted to Calm `232 Def / 36 Spe`. That is 2.5 % of the ladder's IV-31 mons.
- So of the 1,908 pool mons the inversion did label, 348 (18.2 %) were labelled wrong.

## The fix and the numbers after it

The label is now the declared set itself (`belief_tables.true_nature_ev_label`, Rust
`src/rust_env/src/labels/spread.rs`). Both readings already hold that set: poke-env backfills its own
team's spread from the team it declared, and the Rust reading mirrors that. A throwing guard checks
that the set at L100 with its true IVs reproduces the request's stats. No cache is kept on either side,
which also closes F-LC-6. The head's own formula still assumes IV 31, so for an IV-30 mon its derived
stat can read one point high. The `belief_spread` regression term sees the true stat, so it can absorb
that point.

- `src/agents/training/nature_ev_label_test.py`: 4,314 of 4,314 pool mons are labelled with their
  declared set, on all 719 teams.
- The parity gate `rust_env_labels_parity_test.py`, milestone tier, counted in slot-decisions with a
  label (nature/EV over spread):

| tier | before (Lane C) | after |
|---|---|---|
| pool | 45.4 % | **83,905 / 83,905 (100 %)** |
| ladder | 98.8 % | **91,316 / 91,316 (100 %)** |
| procedural | — | **42,539 / 42,539 (100 %)** |

  That is 714,978 key compares with 0 divergences between Rust and Python. The guard never fired in
  the 500 real battles.
- **F-LC-6:** a constructed pair (same Tyranitar / Skarmory species set, different natures and EVs)
  run back to back on one env gets its own labels. With the old cache the second team was labelled
  as the first.
