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

## Summary (recounted 2026-10-26 after B11, on its branch)

| universe | gen3 ids | engine runs it | fails loud | how |
|---|---:|---:|---:|---|
| moves | 369 | **369** | **0** | `PROBE_KIND=move`; JS gate: 369 / 0 / **0 MISMODELED** |
| items | 106 | **106** | 0 | `PROBE_KIND=item` (B3 closed the last four) |
| abilities | 76 | **76** | 0 | `PROBE_KIND=ability`; the handler audit dispositions every hook |
| species | 392 | **392** | 0 | `PROBE_KIND=species` |
| gen3-mod singles formats | 13 | **12** exact | **1** refused (`gen31v1`) | the format table below (B1) |
| engine mechanics | — | — | — | the mechanics table below (M7, the Endless Battle Clause, B10) |

The item universe was cross-checked against the resolved handlers: every gen3 item outside the modeled
set has no battle handler (the fossils, evolution items, balls, the Pokéblock and EV berries, whose
`onEat` is never triggered in gen 3). Outside the universe, eleven handler-bearing items are REFUSED at team
construction: `berryjuice` (`Unobtainable`) and the ten gen-2 `Past` berries an unvalidated Custom Game team can
still hold (`berry`, `goldberry`, and the eight `onUpdate` cure berries: Bitter, Burnt, Ice, Mint, Miracle, Mystery,
PRZ Cure, PSN Cure; `gen3_bf_past_berries_failloud_v1`). The Past bows and the Future incenses are modeled.

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
| `gen3customgame` | — | 1 | 0 | exact HP, no clauses (no Endless Battle Clause either), Beat Up shows its allies | exact |
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
| M7 | Endless Battle Clause staleness (`FormatRules::endless_battle_clause`, every format but Custom Game) | **done** (B10, `gen3_b10_endless_battle_v1`, `turn/endless.rs`): Skill Swap's external `volatileStaleness`, the internal staleness of a restorative berry eaten after a `setItem` (Trick / Thief / Covet / Recycle; gen 3 has no external item staleness and no PP staleness), the past-turn-100 trigger, the trapped / can-switch test and the set-based loser test with its `-message` / win / tie | before B10 the clause could never end a battle in the port. Pinned by targeted boards (`tests/b10_moves_test.rs` B10-11..15, the `ebc_*` golden rows), not by random battles | B10 |
| M8 | Psywave from a user below level 100 into a Substitute | **refused** at the hit (B8): Showdown takes the RAW `random(50, 151) * level / 100` float off the sub's HP (`data/mods/gen4/moves.ts` substitute), leaving a fractional sub HP the port does not hold; every fuzz mode that picks Psywave plays level 100 | — | B8: fail-loud by design |

## Implemented, and known to diverge

A move that runs is not thereby right. Known divergences in implemented mechanics, off the OU pool
surface:

| what | evidence | batch |
|---|---|---|
| Rest under a live Uproar succeeded in the port and failed in Showdown | `gen_sim_bridge_diff.js --mode lockin`, seed 303 battle 15 | fixed in B5 |
| Custom Game's EV-balance `\|bigerror\|` (one side over 510 EVs) was not emitted | the B9 golden's 677-EV boards; `tests/ev_balance_test.rs` | fixed in B9 |
| the random-universe fuzz (the 09-23 count was 95 divergences in 300 battles) | recounted after every batch; after BF only Rest under Uproar remained on BF's own branch, which B5 fixes | — |
| RAGE (boost after `-damage`, per multi-hit strike and on fixed-damage hits; removed at any move attempt), FURY CUTTER (the multiplier doubles only on a hit), UPROAR's wake (at TryHit, before `-damage`), SMELLING SALTS (no cure on a KO'd target), FALSE SWIPE (the 0-delta `-damage` at 1 HP; Endure first), REST by an Insomnia / Vital Spirit user (fails in `onTry`, no gen3ou clause shuffle), the PURSUIT interrupt's last move, the DEFROST thaw (after BeforeMove), a FAINTED caster's ident (slot-less), LANSAT on a focused holder (no `-start`), a MIMIC'd Curse's request target | `tests/regression_test.rs` `bf1`-`bf11`, `harness/probe_bf_classes.js` | fixed in BF |
| found by the merged-tree fuzz and the B8 agent: OWN TEMPO / OBLIVIOUS gained through Skill Swap cure the confusion / infatuation at the next Update; a LOAFING Truant user's Pursuit lays no volatile (no interrupt, no residual handler); SYNCHRONIZE into a Safeguard announces `-activate`; a MULTI-HIT move stops when its user falls asleep (Effect Spore) or faints (Rough Skin); LOCK-ON before an OHKO move skips the roll (the level check too against a locked semi-invulnerable target), the B7 × B8 merge | `tests/regression_test.rs` `bf12`-`bf17`, `harness/probe_bf_classes.js` | fixed in BF (`bf17` with B8) |
| a forced STRUGGLE (every PP spent) read `multihit` from the stale chosen slot, so after a spent Bonemerang / Fury Swipes it struck two or more times (OU-reachable; found by the B7 + B8 bridge fuzz) | `tests/regression_test.rs` `bf18`, `harness/probe_bf_classes.js` BF18 | fixed with B10 |
| KING'S ROCK did not roll on BEAT UP strikes (the sim rolls the appended flinch on every strike; found by the B10 bridge fuzz) | `tests/regression_test.rs` `bf19`, `harness/probe_bf_classes.js` BF19 | fixed with B10 |
| with two or more screens up, BEAT UP strikes and the JUMP KICK / HIGH JUMP KICK crash skipped the ModifyDamagePhase1 tie shuffle (the B7 agent's finding, confirmed in the sim) | `tests/regression_test.rs` `bf20`, `harness/probe_bf_classes.js` BF20 | fixed in BF3 |
| the ten gen-2 `Past` berries with a live gen-3 handler were silently inert in the port (the sim applies them in an unvalidated Custom Game; the B6 agent's Bitter Berry finding) | `unmodeled_items_fail_loud_at_construction` | refused in BF3 (fail-loud) |
| a SNATCH-stolen move did not become the user's `lastMoveUsed`, so a later Conversion 2 read the outer move (gen 3's `useMoveInner` sets it on every `useMove`) | `tests/regression_test.rs` `bf21`, `harness/probe_bf_classes.js` BF21 | fixed in BF3 |
| the independent review (`parity_review_b6_b10_2026-10-26.md`): a PRESENT heal skipped the target's Hit-event reactions (Rage, Focus Punch; F1); BEAT UP was blocked by a non-Shedinja Wonder Guard (typeless in gen 3; F4); a base power floored to 0 by the BasePower chain (Eruption at ~1 HP into Thick Fat / Water Sport) still dealt damage (F5); the search snapshot omitted Safeguard (P1). F2 / F3 are the B11 / BF3 rows | `tests/regression_test.rs` `bf22`-`bf24`, `p1_the_search_snapshot_lists_safeguard` | fixed in BF4 |
| the Hidden Power belief tracker (`trackers/hp_belief.rs`; its Python twin was deleted in T27 P6 6d-2) ignored Foresight / Odor Sleuth: a Hidden Power Fighting into a foresighted Ghost was read against the Ghost immunity (a 2x on Sableye dropped as infeasible, a 0.5x on Gengar zeroed the true Fighting; a neutral hit is no evidence, since the reading promotes only an effectiveness line) | the B6 agent; `tests/tracker_semantics_test.rs` (the two `gen3_hp_foresight_v1` battles) | fixed 2026-10-27 (`gen3_hp_foresight_v1`, obs semantics v163) |
| a hit that confuses a Persim Berry holder: the sim eats the berry BEFORE the attacker's Shell Bell heal (gen 3 `tryMoveHit`'s Update precedes `AfterMoveSecondarySelf`); an infatuation ends at the Update right after a Future Sight strike faints its SOURCE (the generic `hitStepMoveHitLoop`'s Update, `faintMessages`, Update) | the B11 agent's fuzz (`rmvq6vp32_ab_1_4`, `rmvq9o5ho_ab_2_14`), the B11 + BF3 bridge fuzz (`rmvq9yoah_bab_1_12`) | fixed in BF4 (`bf25`, `bf26`) |
| a Sleep-Talk-called move did not set `lastMoveUsed` (Conversion 2 read `sleeptalk`), and Charge's AfterMove keyed on the caller (a Sleep-Talk-called Charge was removed at once); gen-3 `useMoveInner` sets `lastMoveUsed` and `activeMove` to the callee | `harness/probe_b11_moves.js`; `tests/b11_moves_test.rs` B11-17, the B11 golden's `st_charge` board | fixed in B11 (both Sleep Talk behaviour changes) |
| the timed side conditions (Reflect, Light Screen, Safeguard) were gathered for the residual in a FIXED order with Safeguard / Light Screen after the lock-in volatiles; the sim walks `side.sideConditions` in insertion order, which decides a tie's `-sideend` order | the B7 Mist + screens golden board (`mi_screens_b`) | fixed in B7 |

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

## The ranked gap: none

Every gen-3 move runs once B11 lands (the B11 move-callers were the last five; their measured usage was
Metronome 0.062 / Assist 0.015 / Mirror Move 0.006 / Sketch 0.000 / Nature Power 0.000 tier mean).
A batch's moves leave this table once modeled (B8's nine included the four OHKO moves, which the OHKO Clause bans
from every Standard tier).

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
| B5 | Spite, Grudge, Snore, Nightmare, Ingrain; Rest under Uproar fails | shipped |
| B6 | Swagger, Flatter, Foresight, Odor Sleuth, Magic Coat | shipped |
| B7 | Stockpile, Spit Up, Swallow, Mud Sport, Water Sport, Mist, Lock-On, Mind Reader | shipped |
| B8 | Bide, Magnitude, Psywave, Present, Triple Kick, the four OHKO moves | shipped |
| B9 | Attract (M6 resolved by design: harnesses pin genders); the Custom Game EV-balance warning | shipped |
| B10 | Teleport, Follow Me, Helping Hand, Role Play, Camouflage, Endless Battle Clause staleness (M7) | shipped |
| B11 | Metronome, Mirror Move, Assist, Nature Power, Sketch, on ONE `useMove` path shared with Sleep Talk (`turn/callers.rs`) | shipped |
| BF | the fix batch: Rage, Fury Cutter, Uproar's wake, Smelling Salts, False Swipe, Rest by an Insomnia user, the Pursuit interrupt's last move, the defrost thaw, a fainted future-move caster's ident, Lansat on a focused holder, a Mimic'd Curse's request target; then Own Tempo / Oblivious gained by Skill Swap, a loafing Truant user's Pursuit, Synchronize into Safeguard, the multi-hit user guard, and (with B8) Lock-On before an OHKO move | shipped (`bf1`-`bf17`) |
| BF2 | a forced Struggle reading the stale slot's multi-hit; King's Rock on every Beat Up strike (both OU-reachable, found by the bridge fuzz) | shipped with B10 (`bf18`, `bf19`) |
| BF3 | the screens shuffle on Beat Up and the crash; a Snatch-stolen move sets `lastMoveUsed`; the ten gen-2 berries refused | shipped with B11 (`bf20`, `bf21`) |
| BF4 | the independent review's F1, F4, F5 and P1 (F2 / F3 were already fixed by B11 / BF3); the in-tryMoveHit Update before Shell Bell; Attract's source-left Update after a Future Sight KO | shipped with B11 (`bf22`-`bf26`) |
