# Gen 3 full parity — the coverage table (always-current)

**The target (owner, 2026-10-27):** the Rust sim reaches FULL parity with Showdown's `gen3` mod, not just
gen3 OU: every move, ability, item and battle mechanic, for any legal Gen 3 singles team, including Ubers,
UU and below, Little Cup, Random Battle and Custom Game sets that OU does not allow. This file is the
current view. Its predecessor, [`gen3_coverage_census_2026-09-08.md`](gen3_coverage_census_2026-09-08.md),
is a dated snapshot kept as history. Update the tables below in the same commit as any batch that moves
them; the per-id counts are always recounted by the tools, never copied.

**The fail-loud policy.** Anything not yet implemented fails loudly (a panic at construction or at the
first use, a refused format), never a silent approximation. "Implemented" means "the engine does not fail
loud", which is not the same as "a gate has executed it" (the 09-08 census's tier table: 36 implemented
moves are played by no committed battle golden).

## The universes and how to recount them

The id universe is `Dex.forFormat('gen3customgame')`, every id with `exists && !isNonstandard && gen <= 3`
(`struggle` is excluded from moves: it is modeled, gated, and not selectable). Excluded as non-standard:
the 3 CAP moves, the 3 CAP abilities and `noability`, `missingno`, and 23 items (`berryjuice`,
`enigmaberry` and the Apricorn balls are `Unobtainable`; the gen-2 berries and bows are `Past`).

```bash
cd src/rust_sim && cargo build --release --bin scan_move_probe
node -e "const {Dex}=require('../../deps/pokemon-showdown/dist/sim');const d=Dex.forFormat('gen3customgame');
  for(const m of d.moves.all()){if(m.exists&&!m.isNonstandard&&m.gen<=3&&m.id!=='struggle')console.log(m.id)}" > /tmp/moves.txt
# likewise d.species / d.items / d.abilities (abilities: also drop 'noability')
PROBE_KIND=move ./target/release/scan_move_probe < /tmp/moves.txt   # one JSON verdict per id; also species|item|ability
cd harness && SCAN_UNIVERSE=1 node scan_move_coverage.js            # the 0-MISMODELED invariant gate
```

`scan_move_probe` is the engine running, so its count cannot drift; the JS scan's universe verdict is a
hand-mirrored copy of `turn.rs` and keeps only the invariant gate (`move_census.md`).

## Summary (recounted 2026-10-26 after B9 and B4)

| universe | gen3 ids | engine runs it | fails loud | how |
|---|---:|---:|---:|---|
| moves | 369 | **332** | **37** | `PROBE_KIND=move`; JS gate: 332 / 37 / **0 MISMODELED** |
| items | 106 | **106** | 0 | `PROBE_KIND=item` (B3 closed the last four) |
| abilities | 76 | **76** | 0 | `PROBE_KIND=ability`; the handler audit dispositions every hook |
| species | 392 | **392** | 0 | `PROBE_KIND=species` |
| gen3-mod singles formats | 13 | **12** exact | **1** refused (`gen31v1`) | the format table below (B1) |
| engine mechanics | — | — | — | the mechanics table below |

The item universe was cross-checked against the resolved handlers: every gen3 item outside the modeled
set has no battle handler (the fossils, evolution items, balls, the Pokéblock and EV berries, whose
`onEat` is never triggered in gen 3); `berryjuice`, the one handler-bearing item still refused, is gen-3
`Unobtainable`, outside the universe.

## Formats: the gen3-mod singles formats Showdown defines

Showdown's `config/formats.ts` defines 13 gen3-mod singles formats (Draft included). Their rule tables
differ in ways a battle can see, and `src/format.rs` (`gen3_format_rules_v1`, B1) holds each one's:
only `gen3ou`, `gen3uubl` and `gen3draft` carry Freeze Clause Mod, so only they gather BOTH
`onSetStatus` clause handlers and draw the size-2 speed-sort shuffle on every status application
(measured: a Thunder Wave turn is 4 PRNG calls there, 3 everywhere else; the gen-3 `Standard` ruleset in
`data/mods/gen3/rulesets.ts` carries Sleep Clause Mod and no Freeze Clause). The table is checked
against Showdown by `tests/format_rules_test.rs` (the vector `harness/dump_gen3_formats.js` writes,
`--check` in the same test) and `tests/format_rules_golden_test.rs` (12 formats x 6 boards: seed,
bytes, framing, `maybeTrapped`).

| format | battles (sample) | framing `\|rule\|` lines | SetStatus clause handlers | other differences | port |
|---|---:|---:|---:|---|---|
| `gen3ou` / `gen3uubl` | 639,039 / 26,426 | 12 | 2 (shuffle) | | exact |
| `gen3draft` | — | 12 | 2 (shuffle) | no Species Clause | exact |
| `gen3ubers` | 34,317 | 11 | 1 | | exact (B1) |
| `gen3uu` / `gen3ru` | 14,444 / 4,455 | 9 | 1 | the banlist removes Arena Trap from the `maybeTrapped` leak | exact (B1) |
| `gen3nu` / `gen3pu` / `gen3zu` | 4,636 / 2,001 / 53,593 | 10 / 10 / 11 | 1 | | exact (B1) |
| `gen3lc` | 18,514 | 12 | 1 | level 5 by the validator | exact (B1) |
| `gen3randombattle` | — | 9 | 1 | | exact (B1) |
| `gen3customgame` | — | 1 | 0 | exact HP, no clauses, Beat Up shows its allies | exact |
| `gen31v1` | 20,656 | — | — | Team Preview, picked team size 1 | **refused** (no team-preview phase) |

"Battles (sample)" is the Smogon chaos `number of battles` summed over the months in § Usage. **Before
B1** the engine ran every id but `gen3customgame` as OU (`!id.ends_with("customgame")`): one extra draw
per status application, Freeze Clause and OU's framing in 9 formats, all silent. Training is untouched
(every training, eval and live path runs `gen3ou`, whose behaviour B1 does not change).

**Refused, out of scope for the `gen3` singles engine:** the doubles formats (`gen3doublesou`,
`gen3doublescustomgame`, `gen3adv200doubles`, `gen3orrecolosseum`) need a doubles engine;
`gen3adv200` (`gen3rs` mod), `gen3frlgou` (`gen3frlg` mod) and `gen3hoennstadium` (`gen3colosseum`
mod) are other mods with their own data. Every one of them is an `Err` at construction.

## Engine mechanics

| # | mechanic | status | effect today | batch |
|---|---|---|---|---|
| M1 | per-format rule table (above) | **done** (B1) | — | B1 |
| M2 | an unknown or out-of-scope format id | **refused** at construction (B1) | — | B1 |
| M3 | levels | 1–100 exact; 0 reads as 100 as in Showdown; above 100 (`gen3customgame` allows 9999) **refused** (B1) | before B1 a level ≥ 256 silently became 100 | B1 |
| M4 | team size > 6, more than 4 moves (`gen3customgame`: up to 24 each) | more than 6 mons **refused** (B1, ungated); up to 24 moves run (a 6-move set is pinned, `regression_test.rs` WB1), a 25th refused | — | B1 |
| M5 | Team Preview + picked team size (`gen31v1`) | not modeled, **refused** (B1) | — | a construction-phase extension, not scheduled |
| M6 | the construction-time gender draw (`sample(['M','F'])`) | drawn on the raw-seed bridge path (`run_turn0_construction`); the post-construction-seed path (`start_with_switchins`, the goldens and fuzzers) cannot know a sampled gender, so Attract and Cute Charm PANIC on an unspecified one (fail-loud) and every harness pins genders | — | B9: resolved by design |
| M7 | Endless Battle Clause staleness (Skill Swap's external `volatileStaleness`, restorative-berry cycling through Trick / Recycle) | **not modeled, silent** | the clause can never end a battle in the port; reaching it needs both actives stale past turn 100. Affects `gen3ou` too (rare) | B10 |

## Implemented, and known to diverge

A move that runs is not thereby right. Known divergences in implemented mechanics, off the OU pool
surface:

| what | evidence | batch |
|---|---|---|
| Rest under a live Uproar succeeds in the port and fails in Showdown | TECH_DEBT (c) P3 row; `gen_sim_bridge_diff.js --mode lockin`, seed 303 battle 15 | B5 |
| Custom Game's EV-balance `\|bigerror\|` (one side over 510 EVs) was not emitted | the B9 golden's 677-EV boards; `tests/ev_balance_test.rs` | fixed in B9 |
| Rage prints `-boost` before `-damage`; Fury Cutter after a missed use; Uproar's wake prints its cure after the damage; Smelling Salts cures a fainted target; False Swipe at 1 HP drops its `-damage` line; an Insomnia / Vital Spirit Rest may draw the gen3ou clause shuffle | the random-mode fuzz of B4 and B5 (seeds 5501 / 91337), the B5 agent's reading | BF (fix batch) |
| the random-universe byte fuzz: 95 divergences in 300 battles (seeds 5501 / 91337, before ROUNDS 57-59) | `designs/research_state/measurements/rust_core_m1_2026-09-23/README.md` | recount, then triage into the batch of each mechanic class |

## Usage: how the ranking is measured

Measured 2026-10-26, numbers in the gap table below.

- **Ladder usage, every gen3 tier with published stats.** Smogon chaos stats at cutoff 0 (all battles):
  every month from 2022-01 to 2026-09 in which a non-OU gen3-mod singles tier has a file (only spotlight
  months exist, 2 to 9 per tier), plus `gen3ou` for 2026-07 to 2026-09. 54 files, 818,000 battles.
  The **carry rate** of an id in a tier is `Σ_species Moves[id] / Σ_species Σ Abilities` (the share of
  fielded mons that carry it). The **tier mean** averages that over the 10 tiers (`ou ubers uubl uu ru nu
  pu zu lc 1v1`), each weighted equally, so OU's volume does not drown the tiers this table exists to see.
- **Human replays (OU).** The share of a deterministic 1-in-10 sample of the gen3ou replay corpus
  (39,343 of 393,427 logs) in which the move is used at least once (a `|move|` line), and for Shell Bell
  a `[from] item: Shell Bell` heal.
- **Random Battle.** 5,000 generated `gen3randombattle` teams (30,000 sets) carry **none** of the 48 moves or
  4 items; 96 % of their sets are below level 100.
- **Learners**: species whose `gen3_learnset.json` row has the move.

The downloads and scripts were a one-off measurement in a scratch directory, not committed: re-measure
by fetching `https://www.smogon.com/stats/<YYYY-MM>/chaos/<tier>-0.json` for the months listed per tier
in the index pages and applying the formula above.

## The ranked gap: 37 moves

Usage columns are percentages of fielded mons (ladder) or of battles (replays). Risk: **S** reuses
existing machinery; **M** adds a self-contained volatile or mechanic; **L** crosses many mechanics (a
semi-invulnerable state every move must respect, a construction draw that moves goldens, a move that
calls arbitrary moves).

| id | family | tier mean | OU | non-OU pooled | top tier | OU replays | learners | risk | batch |
|---|---|---:|---:|---:|---|---:|---:|:-:|:-:|
| metronome | move-caller | 0.062 | 0.054 | 0.055 | 1v1 0.22 | 0.224 | 55 | L | B11 |
| spite | PP | 0.038 | 0.004 | 0.035 | 1v1 0.25 | 0.010 | 21 | M | B5 |
| swagger | confusion + boost | 0.030 | 0.000 | 0.041 | lc 0.15 | 0.005 | 373 | M | B6 |
| snore | sleep | 0.020 | 0.019 | 0.025 | 1v1 0.05 | 0.084 | 373 | M | B5 |
| grudge | PP | 0.018 | 0.007 | 0.008 | pu 0.06 | 0.031 | 12 | M | B5 |
| nightmare | sleep | 0.017 | 0.007 | 0.015 | uu 0.04 | 0.013 | 70 | M | B5 |
| assist | move-caller | 0.015 | 0.000 | 0.022 | ru 0.07 | 0.010 | 10 | L | B11 |
| ingrain | residual | 0.012 | 0.006 | 0.012 | uu 0.05 | 0.058 | 22 | M | B5 |
| foresight | identify | 0.012 | 0.001 | 0.011 | uu 0.05 | 0.010 | 45 | M | B6 |
| odorsleuth | identify | 0.010 | 0.002 | 0.017 | uubl 0.05 | 0.010 | 21 | M | B6 |
| magiccoat | reflect | 0.008 | 0.003 | 0.011 | uu 0.03 | 0.003 | 6 | M | B6 |
| stockpile | stockpile | 0.008 | 0.001 | 0.004 | pu 0.03 | 0.018 | 16 | M | B7 |
| mirrormove | move-caller | 0.006 | 0.001 | 0.006 | lc 0.01 | 0.013 | 19 | L | B11 |
| mudsport | field sport | 0.005 | 0.002 | 0.007 | lc 0.03 | 0.008 | 45 | M | B7 |
| bide | variable damage | 0.005 | 0.001 | 0.004 | 1v1 0.03 | 0.003 | 20 | M | B8 |
| magnitude | variable BP | 0.005 | 0.001 | 0.003 | ru 0.02 | 0.005 | 15 | M | B8 |
| camouflage | type change | 0.003 | 0.000 | 0.003 | 1v1 0.03 | 0.000 | 3 | S | B10 |
| triplekick | variable BP | 0.002 | 0.000 | 0.002 | 1v1 0.01 | 0.010 | 2 | M | B8 |
| spitup | stockpile | 0.002 | 0.000 | 0.002 | pu 0.01 | 0.015 | 16 | M | B7 |
| swallow | stockpile | 0.002 | 0.000 | 0.001 | pu 0.01 | 0.008 | 16 | M | B7 |
| psywave | variable damage | 0.002 | 0.001 | 0.002 | pu 0.01 | 0.003 | 14 | M | B8 |
| lockon | lock-on | 0.001 | 0.002 | 0.002 | lc 0.00 | 0.005 | 11 | M | B7 |
| followme | redirect | 0.001 | 0.000 | 0.001 | uu 0.01 | 0.000 | 13 | S | B10 |
| flatter | confusion + boost | 0.001 | 0.000 | 0.002 | uu 0.01 | 0.008 | 8 | M | B6 |
| mindreader | lock-on | 0.001 | 0.000 | 0.001 | nu 0.01 | 0.015 | 18 | M | B7 |
| teleport | escape | 0.001 | 0.000 | 0.001 | lc 0.00 | 0.000 | 11 | S | B10 |
| roleplay | ability copy | 0.001 | 0.000 | 0.001 | uu 0.01 | 0.000 | 8 | M | B10 |
| present | variable BP | 0.001 | 0.000 | 0.001 | uu 0.00 | 0.013 | 20 | M | B8 |
| watersport | field sport | 0.000 | 0.000 | 0.000 | nu 0.00 | 0.000 | 30 | M | B7 |
| mist | side condition | 0.000 | 0.000 | 0.001 | uubl 0.00 | 0.003 | 31 | M | B7 |
| sketch | move-caller | 0.000 | 0.001 | 0.000 | 1v1 0.00 | 0.003 | 1 | L | B11 |
| helpinghand | ally boost | 0.000 | 0.000 | 0.000 | uu 0.00 | 0.000 | 56 | S | B10 |
| naturepower | move-caller | 0.000 | 0.000 | 0.000 | uubl 0.00 | 0.003 | 14 | L | B11 |
| sheercold / fissure / guillotine / horndrill | OHKO | 0.000 | 0.000 | 0.000 | ubers 0.00 | ≤ 0.013 | 7–15 | M | B8 |

The OHKO moves are banned by the OHKO Clause in every Standard tier, so their ladder usage is zero by
construction; they are legal in Custom Game.

## The batch plan

Highest usage first, held back where a family depends on another (the move-callers last: each calls
arbitrary moves, so it closes only once every callable move does). Each batch: the semantics verified in
Showdown's gen3 chain with file and line, the Rust change, named regression pins plus a live differential
golden against Showdown, the A/B fuzzer in `--mode random` to 0 divergences on the touched mechanics, the
OU goldens unchanged unless a real OU bug is found (then a FINDING with its training impact), and one
commit.

| batch | content | status |
|---|---|---|
| B0 | this recount and table | shipped with this file |
| B1 | per-format rule table (M1), refuse unknown formats, levels, oversize teams and Team Preview (M2–M5) | shipped |
| B2 | Blast Burn, Frenzy Plant, Hydro Cannon | shipped |
| B3 | Shell Bell, Macho Brace, Mail, Mental Herb | shipped |
| B4 | the two-turn and semi-invulnerable family: Sky Attack, Dig, Fly, Dive, Bounce, Skull Bash, Razor Wind | shipped |
| B5 | Spite, Grudge, Snore, Nightmare, Ingrain | open |
| B6 | Swagger, Flatter, Foresight, Odor Sleuth, Magic Coat | open |
| B7 | Stockpile, Spit Up, Swallow, Mud Sport, Water Sport, Mist, Lock-On, Mind Reader | open |
| B8 | Bide, Magnitude, Psywave, Present, Triple Kick, the four OHKO moves | open |
| B9 | Attract (M6 resolved by design: harnesses pin genders); the Custom Game EV-balance warning | shipped |
| B10 | Teleport, Follow Me, Helping Hand, Role Play, Camouflage, Endless Battle Clause staleness (M7) | open |
| B11 | Metronome, Mirror Move, Assist, Nature Power, Sketch | open |
