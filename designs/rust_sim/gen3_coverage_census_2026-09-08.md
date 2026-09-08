# The gen3 coverage census — 2026-09-08

**What this measures.** The port's target has been *gen3 **OU** singles*. The owner's 2026-09-08
directive widens it: *"we believe we have all gen3 OU, but let's drive all gen3 if we can."* So this
census asks a different question from every earlier coverage record — not *"can the port play the
762-team OU pool"* (it can: 762/762) but **"can the port execute every move, ability, item and species
that Showdown's gen3 data makes legal in ANY gen3 format"** — OU, UU, NU, Ubers, LC, randoms.

🚨 **This file is a SNAPSHOT, deliberately dated, and it is NOT the always-current number.** The
current answer is always the tool (see *How to re-measure* below). It exists so the gap is on the
record BEFORE the rounds that close it move it.

---

## The measuring instrument, and why the old one was wrong by 3

**`src/rust_sim/src/bin/scan_move_probe.rs` is the oracle**, and this census extends it from moves to
all four universes (`PROBE_KIND=move|species|item|ability`). It constructs a minimal
`gen3customgame` battle with the candidate id in the field under test, drives it through the
**unchanged public engine** (`Battle::start_with_switchins` → `run_full_battle`) under
`catch_unwind`, and reports whether the ENGINE fail-louds. It is empirical: nothing about it can
drift away from `turn.rs`, because it *is* `turn.rs` running.

⚠️ **The JS census `harness/scan_move_coverage.js` had drifted, and this is the second time that
class has bitten.** Its `SCAN_UNIVERSE=1` verdict is computed from modeled-move sets **mirrored by
hand** from `turn.rs`, and those mirrors had not been updated for ROUND 51's `defensecurl` or ROUND
52's `minimize` / `imprison`. It therefore reported **309 MODELED / 60 FAIL-LOUD** where the engine
itself runs **312 / 57** — and 309 is the number the leaf `CLAUDE.md`, the root `CLAUDE.md` and
`src/utils/bridge/README.md` all carry. **A hand-mirrored predicate that gates a census silently
shrinks that census**, which is the same shape as the ROUND-42 picker lesson and the ROUND-31
allowlist lesson. The three moves were never broken; only the map was.

The static scan keeps its job — it is the **team-pool** report and the `0 MISMODELED` invariant gate,
neither of which the engine probe answers — but **the universe COUNT is now the probe's to state**,
and `SCAN_UNIVERSE_LIST=1` was added so the scan can print *which* moves it calls unmodeled instead
of only how many, which is what made the drift visible at all.

---

## Summary — every gen3 universe, as of 2026-09-08

| Universe | gen3-legal ids | Engine RUNS it | Engine FAIL-LOUDS | Gap |
|---|---:|---:|---:|---:|
| **Moves** (at the census) | 369 | **312** | **57** | 15.4% |
| **Moves** (after ROUNDS 57-58) | 369 | **320** | **49** | 13.3% |
| **Abilities** | 76 | **76** | 0 | **CLOSED** |
| **Species** | 392 | **392** | 0 | **CLOSED** |
| **Items** | 106 | **102** | **4** | 3.8% |

**The move row moved while this file was being written, which is the point of dating it.** ROUND 57
closed the pure-confusion family (`supersonic` / `sweetkiss` / `teeterdance`) and ROUND 58 the spread
stat-drops (`leer` / `growl` / `tailwhip` / `stringshot` / `sweetscent`). The ranked list below is
the state **at the census**; strike those eight when reading it. The current number is always
`scan_move_probe`.

*Universe = every id in `Dex.forFormat('gen3customgame')` with `exists && !isNonstandard && gen <= 3`
(`struggle` excluded from moves — it is modeled and has its own differential gate, but it is not a
selectable move).*

**Abilities and species are DONE.** All 76 gen3 abilities are either modelled or a justified no-op
(ROUND 35 closed the last fail-loud one, Forecast); all 392 gen3 species construct and battle (ROUND
38 closed the last data gap, the missing forme rows). Neither needs another round.

### Gate exposure of the 312 implemented moves

"Implemented" and "gated" are separate questions, and the honest split is three-tier rather than two.
A move counts for a tier if its id appears as a standalone token in that tier's committed artifacts:

| Tier | Meaning | Moves |
|---|---|---:|
| **A** | named in a **hand-written Rust gate** (`tests/*.rs`) — a pin, a feature gate or a scenario assertion | **205** |
| **B** | named only in a **hand-written JS golden generator** (`harness/gen_*.js`) — a constructed differential scenario | **57** |
| **C** | named only in a **played sweep corpus** (the e2e capstone golden, the byte-fuzz fixtures, the protocol captures) | **14** |
| **D** | **nothing has ever executed it in a committed artifact** | **36** |

🚨 **THIS TABLE WAS WRONG WHEN FIRST PUBLISHED, AND THE CORRECTION IS THE FINDING.** The first
version reported **C = 50, D = 0** and concluded "every move the engine executes has at least
fuzz-corpus exposure". It did not: it swept `tests/vectors/` as one bucket, and that directory holds
**data dumps** — `dex_golden.txt`, the handler-audit manifest, the mechanics inventory, the protocol
inventory, the e2e taxonomy — alongside the battle corpora. A move id appearing in a **dex row** was
being counted as **battle exposure**. Excluding the dumps moves 36 moves from C to D. **A coverage
measure that cannot tell a data dump from a played turn will report a move as gated when nothing has
ever executed it.**

⚠️ **AND TIER A IS NOT DRAW COVERAGE.** `confuseray` sits in tier A — it has a named, revert-verified
ROUND-44 pin — and ROUND 57 nevertheless found that it **never rolled its accuracy**, desyncing the
PRNG on every use. The pin asserted the `-start|confusion` and `-fail` EMISSIONS and no seed, so the
missing draw was invisible to it. Tier membership answers "is there a gate", never "does that gate
assert the draw count".

⚠️ **The very first attempt at this measure reported "312/312 gated" and was worthless** — a bare
substring search over all 576 gate artifacts matches a move id inside any taxonomy list or comment.
Three drafts, and only the third is a measurement.

### The 4 uncovered items

`shellbell` · `machobrace` · `mentalherb` · `mail`. These are exactly the guarded set
`state.rs::UNMODELED_FAILLOUD_ITEMS` names (minus `berryjuice`, which is not in the gen3-legal dex
universe), each with a genuine gen-3 battle handler the port does not price:

| item | handler | why it matters |
|---|---|---|
| `machobrace` | `onModifySpe` — **halves Speed** | turn-ORDER and therefore **DRAW** relevant; a silent miss desyncs the speed-tie shuffle |
| `shellbell` | `onAfterMoveSecondarySelf` — drains 1/8 of damage dealt | a per-hit HP change |
| `mentalherb` | `onUpdate` — cures attraction | inert until Attract is modelled |
| `mail` | `onTakeItem` — blocks item theft | changes Knock Off / Thief / Trick outcomes |

They are LATENT: 0 hits across 4722 training-pool team-blocks and 0 across 3000 generated
`gen3randombattle` teams (measured at ROUND 39). They fail loud, so they cannot lie — they simply
make a team carrying one unplayable.

---

## The ranked gap — 57 moves

**Ranking metric: the number of gen3 species that legally learn the move** (`gen3_learnset.json`,
386 species). That is the brief's stated fallback and it is the right one here, because **the only
usage data committed to this repo is gen3 *OU*** (`gen3_smogon_stats.json`, `gen3ou-1500`, 2.53M
battles) — and OU usage is precisely the axis this census is trying to see past. The OU column is
kept as a second column exactly so its near-zero values make that point: **the largest remaining gap
is invisible from OU**, which is why it survived 55 rounds.

### By family — the unit a round should close

| learners | n | family | moves (by learners) |
|---:|---:|---|---|
| 441 | 2 | **sleep-interaction** | snore · nightmare |
| 432 | 5 | **confusion** — the volatile is ALREADY modelled (`confuseray`) | swagger · supersonic · sweetkiss · teeterdance · flatter |
| 338 | 1 | **attraction** — the volatile is ALREADY modelled (Cute Charm) | attract |
| 318 | 7 | **two-turn charge / semi-invulnerable** | dig · dive · fly · skyattack · skullbash · razorwind · bounce |
| 234 | 5 | **target stat-drop** — the boost machinery already exists | leer · growl · tailwhip · sweetscent · stringshot |
| 166 | 11 | misc utility | focusenergy · helpinghand · mist · spite · ingrain · teleport · followme · grudge · magiccoat · roleplay · camouflage |
| 84 | 5 | **move-caller** — each calls another move, so each MULTIPLIES the surface | metronome · mirrormove · naturepower · assist · sketch |
| 71 | 4 | identify / lock-on (accuracy bypass) | foresight · odorsleuth · mindreader · lockon |
| 48 | 2 | field sport | mudsport · watersport |
| 47 | 5 | variable-BP damaging | bide · magnitude · psywave · present · triplekick |
| 36 | 3 | Stockpile counter | stockpile · swallow · spitup |
| 32 | 4 | OHKO | fissure · horndrill · sheercold · guillotine |
| 6 | 3 | recharge (Hyper Beam siblings) | blastburn · hydrocannon · frenzyplant |

**Three families are cheap because their hard half is already built.** Confusion, attraction and
target stat-drops each need an ENTRY PATH into machinery the engine already runs bit-for-bit — the
confusion volatile (Confuse Ray, ROUND 44), the attract volatile (Cute Charm), and the
`TryBoost`→`boost` path (Intimidate, Screech, the ROUND-55 cap ordering). Together that is **11 moves
and 1,004 learner-slots for three entry paths.**

**One family should be done LAST, or not at all.** The move-callers (Metronome, Assist, Mirror Move,
Nature Power, Sketch) each dispatch *another* move, so each one's correctness is a function of the
whole rest of the census. Closing them before the tail is closed would ship a mechanic whose failure
mode is "calls something unmodeled and panics mid-battle".

### Full per-move ranking

| move | learners | gen3OU usage % | family |
|---|---:|---:|---|
| snore | 372 | 0.014 | sleep-interaction |
| swagger | 372 | 0.000 | confusion |
| attract | 338 | 0.007 | attraction |
| dig | 158 | 0.005 | two-turn charge |
| leer | 83 | 0.009 | target stat-drop |
| growl | 78 | 0.003 | target stat-drop |
| dive | 76 | 0.002 | two-turn charge |
| nightmare | 69 | 0.007 | sleep-interaction |
| metronome | 54 | 0.048 | move-caller |
| tailwhip | 48 | 0.002 | target stat-drop |
| focusenergy | 40 | 0.005 | misc utility |
| helpinghand | 40 | 0.000 | misc utility |
| fly | 39 | 0.016 | two-turn charge |
| supersonic | 39 | 0.002 | confusion |
| foresight | 31 | 0.001 | identify |
| mudsport | 28 | 0.000 | field sport |
| skyattack | 27 | 0.020 | two-turn charge |
| mist | 20 | 0.000 | misc utility |
| sweetscent | 20 | 0.000 | target stat-drop |
| watersport | 20 | 0.000 | field sport |
| odorsleuth | 18 | 0.003 | identify |
| spite | 16 | 0.002 | misc utility |
| mirrormove | 14 | 0.001 | move-caller |
| bide | 14 | 0.000 | variable-BP |
| ingrain | 13 | 0.003 | misc utility |
| magnitude | 13 | 0.000 | variable-BP |
| mindreader | 13 | 0.000 | lock-on |
| stockpile | 12 | 0.000 | Stockpile |
| swallow | 12 | 0.000 | Stockpile |
| spitup | 12 | 0.000 | Stockpile |
| sweetkiss | 11 | 0.001 | confusion |
| psywave | 10 | 0.003 | variable-BP |
| teleport | 10 | 0.000 | misc utility |
| naturepower | 10 | 0.000 | move-caller |
| fissure | 10 | 0.000 | OHKO |
| present | 9 | 0.001 | variable-BP |
| lockon | 9 | 0.001 | lock-on |
| followme | 9 | 0.000 | misc utility |
| horndrill | 8 | 0.000 | OHKO |
| sheercold | 8 | 0.000 | OHKO |
| grudge | 7 | 0.008 | misc utility |
| skullbash | 7 | 0.000 | two-turn charge |
| razorwind | 7 | 0.000 | two-turn charge |
| guillotine | 6 | 0.000 | OHKO |
| magiccoat | 5 | 0.001 | misc utility |
| teeterdance | 5 | 0.000 | confusion |
| flatter | 5 | 0.000 | confusion |
| roleplay | 5 | 0.000 | misc utility |
| stringshot | 5 | 0.000 | target stat-drop |
| assist | 5 | 0.000 | move-caller |
| bounce | 4 | 0.000 | two-turn charge |
| blastburn | 2 | 0.019 | recharge |
| hydrocannon | 2 | 0.003 | recharge |
| frenzyplant | 2 | 0.001 | recharge |
| triplekick | 1 | 0.000 | variable-BP |
| sketch | 1 | 0.000 | move-caller |
| camouflage | 1 | 0.000 | misc utility |

---

## How to re-measure

```bash
cd src/rust_sim
cargo build --release --bin scan_move_probe
# the id universes (one per line), straight from the resolved gen3 dist
node -e "const {Dex}=require('../../deps/pokemon-showdown/dist/sim');const d=Dex.forFormat('gen3customgame');
  for(const m of d.moves.all()){if(m.exists&&!m.isNonstandard&&m.gen<=3&&m.id!=='struggle')console.log(m.id)}" > /tmp/moves.txt
PROBE_KIND=move    ./target/release/scan_move_probe < /tmp/moves.txt     # verdict per id, JSON lines
PROBE_KIND=species ./target/release/scan_move_probe < /tmp/species.txt
PROBE_KIND=item    ./target/release/scan_move_probe < /tmp/items.txt
PROBE_KIND=ability ./target/release/scan_move_probe < /tmp/abilities.txt

# the team-pool report + the 0-MISMODELED invariant gate (a different question — keep both)
cd harness && SCAN_UNIVERSE=1 SCAN_UNIVERSE_LIST=1 node scan_move_coverage.js
```

A `"verdict":"ran"` means **the engine did not fail-loud**, not that the mechanic is right. The
claim that it is right rests on the separate ROUND-40 invariant — every move is MODELED or
FAIL-LOUD, **0 MISMODELED** — so a `ran` verdict cannot be hiding a silent desync.
