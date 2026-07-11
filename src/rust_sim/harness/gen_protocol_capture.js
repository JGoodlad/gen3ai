// gen_protocol_capture.js — Gen-3 PROTOCOL-EMISSION (level-2) capture harness.
//
// THE LEVEL-2 TARGET. Levels build up: the PRNG/dex/team/stat/state/turn goldens
// prove the ENGINE (bit-for-bit RNG + state). This harness captures the OTHER
// bit-for-bit contract — the RAW `|...|` PROTOCOL STREAM Showdown emits, which our
// poke-env fork parses. The Rust engine's `protocol.rs` (currently types-only) will
// later emit these exact bytes; a `protocol_test.rs` will replay this golden through
// the emitting engine and assert byte-equality per line (see
// src/rust_sim/PROTOCOL_EMISSION_DESIGN.md).
//
// WHAT WE CAPTURE — the OMNISCIENT (referee/spectator) stream, verbatim, line by
// line, per battle. This is the CLEAN full stream: no `|request|` blocks, no
// `|split|` privacy folding, full `x/y` HP for BOTH sides. It is exactly what the
// Rust engine will produce (the per-side privacy folding is the bridge's
// `getPlayerStreams` job, downstream of the engine — see local_sim_bridge.js). The
// `>start`/`>player`/choice commands + the running PRNG seed are recorded alongside
// the lines so a future Rust test can replay the identical battle and diff the bytes.
//
// DETERMINISTIC + REPRODUCIBLE: a fixed MASTER_SEED drives a well-spread seed pool
// (the same generator the other goldens use), so a plain re-run reproduces the golden
// byte-for-byte. Constructed teams/moves mirror gen_fullbattle_golden.js (the closest
// template) PLUS a handful of scenarios that exercise the long-tail line types
// (status moves, weather, secondaries, substitute, protect, spikes, phazing) so the
// captured corpus covers every distinct `|...|` line type the port must emit. This is
// the SOURCE for tests/vectors/protocol_inventory.md.
//
// Output: tests/vectors/protocol_capture_golden.txt
//
// Run:  node src/rust_sim/harness/gen_protocol_capture.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/protocol_capture_golden.txt');
const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };

// The master seed the whole corpus derives from — flip it to reshuffle the seed pool
// (a plain re-run with the same value reproduces the committed golden byte-for-byte).
const MASTER_SEED = 0x50524f54; // 'PROT'

function mon(species, moves, opts = {}) {
  return {
    species,
    item: opts.item || '',
    ability: opts.ability || 'No Ability',
    moves,
    evs: { ...EV0, ...(opts.evs || {}) },
    ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious',
    level: opts.level || 100,
    gender: opts.gender || 'N',
  };
}

function tick() { return new Promise((r) => setTimeout(r, 0)); }

// Encode a submitted choice string ('move K' / 'switch N' / null) into the compact
// golden token: 'm<K-1>' (0-based move slot) | 's<N-1>' (0-based team slot) | '-'
// (mirrors gen_fullbattle_golden.js so a future Rust replay shares the parser).
function encodeChoice(c) {
  if (!c) return '-';
  const m = c.match(/^move\s+(\d+)$/);
  if (m) return `m${Number(m[1]) - 1}`;
  const s = c.match(/^switch\s+(\d+)$/);
  if (s) return `s${Number(s[1]) - 1}`;
  throw new Error(`unencodable choice ${JSON.stringify(c)}`);
}

// A well-spread deterministic seed pool from MASTER_SEED (same LCG generator the
// other goldens use, so the corpus is reproducible).
function buildSeeds(n, master) {
  const out = [];
  let x = master >>> 0;
  const step = () => { x = (Math.imul(x, 1103515245) + 12345) >>> 0; return x & 0xffff; };
  for (let i = 0; i < n; i++) out.push([step() || 1, step() || 1, step() || 1, step() || 1]);
  return out;
}

// The forceSwitch table for the current request, per side ('move' → [false,false]).
function forceSwitchTable(battle) {
  const out = [false, false];
  if (battle.requestState !== 'switch') return out;
  for (let i = 0; i < 2; i++) {
    const req = battle.sides[i].activeRequest;
    if (req && req.forceSwitch && req.forceSwitch[0]) out[i] = true;
  }
  return out;
}

// Run ONE scenario at one seed to game-end (or the script's end), capturing the FULL
// ordered OMNISCIENT protocol stream. Records, per battle:
//   - initSeed        : the pre-first-decision PRNG seed (the Rust seeds here)
//   - decisions[]     : per boundary { request, force, choiceP1, choiceP2, seedAfter }
//   - lines[]         : EVERY omniscient `|...|` line, in emission order, verbatim
//   - winner / ended
async function runBattle(sc, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  // The OMNISCIENT stream is the clean full referee view — the bytes the Rust engine
  // emits. We keep EVERY split line verbatim (incl. |t:| and |debug|, which poke-env
  // ignores but the engine still emits — the inventory notes which are parsed) EXCEPT
  // we NORMALIZE the `|t:|` wall-clock timestamp to a fixed placeholder. The timestamp
  // is a Unix time that changes every run (non-reproducible) AND is inherently
  // un-reproducible by the Rust engine; poke-env ignores `|t:|` entirely, and the
  // byte-comparison test excludes it. Normalizing keeps the GOLDEN byte-stable across
  // regenerations while preserving that a `|t:|` line exists at exactly that position.
  (async () => {
    for await (const ch of streams.omniscient) {
      for (const l of ch.split('\n')) {
        lines.push(l.startsWith('|t:|') ? '|t:|<NORMALIZED>' : l);
      }
    }
  })();

  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(sc.p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(sc.p2) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const script = sc.makeScript();
  const rec = { initSeed: null, decisions: [], lines, winner: null, ended: false, gen: stream.battle.gen };

  let decisionNo = 0;
  let safety = 0;
  while (!stream.battle.ended && safety < 200) {
    safety++;
    const battle = stream.battle;
    const reqState = battle.requestState; // 'move' | 'switch' | 'teampreview' | ''
    if (reqState !== 'move' && reqState !== 'switch') { await tick(); continue; }
    const force = forceSwitchTable(battle);
    const seedBefore = battle.prng.getSeed();
    if (decisionNo === 0) rec.initSeed = seedBefore;

    const choices = script(decisionNo, battle, reqState, force);
    if (!choices) break;

    if (choices.p1) { try { streams.omniscient.write(`>p1 ${choices.p1}`); } catch (e) {} }
    if (choices.p2) { try { streams.omniscient.write(`>p2 ${choices.p2}`); } catch (e) {} }
    for (let i = 0; i < 16; i++) await tick();

    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter: battle.prng.getSeed(),
    });
    decisionNo++;
  }

  rec.ended = !!stream.battle.ended;
  rec.winner = stream.battle.winner;
  try { streams.omniscient.destroy(); } catch (e) {}
  return rec;
}

// ── Scenarios ──────────────────────────────────────────────────────────────
// Each is { id, p1[], p2[], makeScript() }. makeScript() returns a FRESH stateful
// closure so the move-plan index resets per battle. The mix deliberately spans the
// full set of `|...|` line types the port must emit (see protocol_inventory.md).
function scenarios() {
  const S = [];

  // Helper: a FACTORY for a fixed per-turn move/switch plan. `plan` is an array of
  // per-MOVE-request {p1, p2}; a forced-switch request uses `onForce(side)` for the
  // replacement. Returns makeScript() → a FRESH stateful closure.
  const fromPlan = (plan, onForce) => () => {
    let i = 0;
    return (decisionNo, battle, reqState, force) => {
      if (reqState === 'switch') {
        const c = { p1: null, p2: null };
        if (force[0]) c.p1 = onForce(0, battle);
        if (force[1]) c.p2 = onForce(1, battle);
        return c;
      }
      const entry = plan[i] || { p1: 'move 1', p2: 'move 1' };
      i++;
      return entry;
    };
  };

  const firstLiveBench = (side, battle) => {
    const s = battle.sides[side];
    for (let k = 0; k < s.pokemon.length; k++) {
      const p = s.pokemon[k];
      if (p !== s.active[0] && !p.fainted) return `switch ${k + 1}`;
    }
    return 'pass';
  };

  // ── (A) Core move/damage/switch/faint/win — mirrors the fullbattle golden. ──

  // Both switch (distinct speed) → trade attacks to a win. |switch| |move| |-damage|
  // |-immune| |-heal| |faint| |upkeep| |turn| |win|.
  S.push({
    id: 'both_switch_distinct',
    p1: [mon('Snorlax', ['earthquake', 'tackle'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
         mon('Suicune', ['surf', 'icebeam'], { item: 'Leftovers', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    p2: [mon('Skarmory', ['drillpeck', 'tackle'], { item: 'Leftovers', nature: 'Impish', evs: { hp: 252, def: 252 } }),
         mon('Tyranitar', ['earthquake', 'rockslide'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'switch 2', p2: 'switch 2' }, { p1: 'move 1', p2: 'move 1' }],
      (side, battle) => firstLiveBench(side, battle)),
  });

  // Post-faint sweep to a win — |faint| then a forced |switch| replacement, |win|.
  S.push({
    id: 'post_faint_sweep_win',
    p1: [mon('Aerodactyl', ['earthquake', 'rockslide'], { item: 'Choice Band', ability: 'Rock Head', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Jolteon', ['swift', 'surf'], { nature: 'Timid', evs: { spa: 252, spe: 252 } }),
         mon('Gengar', ['swift', 'surf'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 1', p2: 'move 1' }],
      (side, battle) => firstLiveBench(side, battle)),
  });

  // Double faint (mutual Explosion) → BOTH replace the same turn. |-crit| may appear;
  // exercises the double-replacement path.
  S.push({
    id: 'double_faint_replace',
    p1: [mon('Electrode', ['explosion', 'thunderbolt'], { nature: 'Hasty', evs: { atk: 252, spe: 252 } }),
         mon('Snorlax', ['earthquake', 'bodyslam'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Electrode', ['explosion', 'thunderbolt'], { nature: 'Hasty', evs: { atk: 252, spe: 252 } }),
         mon('Snorlax', ['earthquake', 'bodyslam'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 1', p2: 'move 1' }],
      (side, battle) => firstLiveBench(side, battle)),
  });

  // Last-mon double-KO → gen3 TIE — |tie| (win with no name).
  S.push({
    id: 'last_mon_double_ko_tie',
    p1: [mon('Electrode', ['explosion', 'thunderbolt'], { nature: 'Hasty', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Electrode', ['explosion', 'thunderbolt'], { nature: 'Hasty', evs: { atk: 252, spe: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 1', p2: 'move 1' }],
      (side, battle) => firstLiveBench(side, battle)),
  });

  // ── (B) Weather + boosts + type-effectiveness — |-ability| |-weather| |-unboost|
  //         |-supereffective| |-resisted|. Tyranitar Sand Stream + Salamence
  //         Intimidate; Rock Slide is SE on Salamence (Flying). ──
  S.push({
    id: 'sand_intimidate_effectiveness',
    p1: [mon('Tyranitar', ['rockslide', 'earthquake', 'crunch'], { item: 'Choice Band', ability: 'Sand Stream', nature: 'Adamant', evs: { atk: 252, spe: 252 } }),
         mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Salamence', ['icebeam', 'flamethrower'], { ability: 'Intimidate', nature: 'Modest', evs: { spa: 252, spe: 252 } }),
         mon('Jolteon', ['thunderbolt', 'surf'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 1', p2: 'move 1' },   // Rock Slide SE → KO Salamence
       { p1: 'move 1', p2: 'move 1' },   // Jolteon in; Rock Slide + sand chip
      ],
      (side, battle) => firstLiveBench(side, battle)),
  });

  // ── (C) Standalone STATUS move + status DoT + status secondary — |-status|
  //         |cant| |-boost| (from a stat-drop secondary). gen3ou would draw the
  //         SetStatus shuffle; we stay in gen3customgame like the other goldens.
  //         Thunder Wave (par) then a full-para |cant|; Crunch −1 SpD |-unboost|. ──
  S.push({
    id: 'status_para_and_boost_drop',
    p1: [mon('Tyranitar', ['thunderwave', 'crunch', 'rockslide'], { item: 'Leftovers', ability: 'Sand Stream', nature: 'Careful', evs: { hp: 252, spd: 252 } }),
         mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Starmie', ['surf', 'icebeam'], { item: 'Leftovers', nature: 'Timid', evs: { spa: 252, spe: 252 } }),
         mon('Blissey', ['seismictoss', 'icebeam'], { item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 1', p2: 'move 1' },   // TWave paralyses Starmie (Crunch is p1 slot2)
       { p1: 'move 2', p2: 'move 1' },   // Crunch −1 SpD; Starmie may be full-para |cant|
       { p1: 'move 2', p2: 'move 1' },
       { p1: 'move 3', p2: 'move 1' },
      ],
      (side, battle) => firstLiveBench(side, battle)),
  });

  // ── (D) A damaging status secondary + flinch + freeze/thaw — |-status| |cant|
  //         (flinch). Ice Beam (10% frz) + Rock Slide (30% flinch) + Body Slam
  //         (30% par). A wide net for the secondary-status line types. ──
  S.push({
    id: 'secondary_status_flinch',
    p1: [mon('Cloyster', ['icebeam', 'rockslide'], { item: 'Leftovers', nature: 'Modest', evs: { spa: 252, spe: 252 } }),
         mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Blissey', ['seismictoss', 'icebeam'], { item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } }),
         mon('Skarmory', ['drillpeck', 'tackle'], { item: 'Leftovers', ability: 'Keen Eye', nature: 'Impish', evs: { hp: 252, def: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 2', p2: 'move 1' },   // Rock Slide — 30% flinch on Blissey
       { p1: 'move 1', p2: 'move 1' },   // Ice Beam — 10% frz
       { p1: 'move 2', p2: 'move 1' },
       { p1: 'move 1', p2: 'move 1' },
      ],
      (side, battle) => firstLiveBench(side, battle)),
  });

  // ── (E) SUBSTITUTE — |-start| Substitute, |-activate| (sub absorb), |-end| on
  //         break, |-heal| Leftovers behind the sub. Suicune subs, Tyranitar hits it. ──
  S.push({
    id: 'substitute_absorb',
    p1: [mon('Suicune', ['substitute', 'surf'], { item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } }),
         mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Tyranitar', ['crunch', 'rockslide'], { item: 'Choice Band', ability: 'Sand Stream', nature: 'Adamant', evs: { atk: 252, spe: 252 } }),
         mon('Metagross', ['meteormash', 'earthquake'], { item: 'Leftovers', ability: 'Clear Body', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 1', p2: 'move 1' },   // Suicune Substitute; Tyranitar Crunch → sub
       { p1: 'move 1', p2: 'move 1' },   // (sub may already be up / break)
       { p1: 'move 2', p2: 'move 1' },
       { p1: 'move 2', p2: 'move 1' },
      ],
      (side, battle) => firstLiveBench(side, battle)),
  });

  // ── (F) PROTECT — |move| Protect|[still], |-singleturn| Protect, |-activate|
  //         Protect (block). Skarmory Protects while Blissey seismic-tosses. ──
  S.push({
    id: 'protect_block',
    p1: [mon('Skarmory', ['protect', 'drillpeck'], { item: 'Leftovers', ability: 'Keen Eye', nature: 'Impish', evs: { hp: 252, def: 252 } }),
         mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Tyranitar', ['rockslide', 'crunch'], { item: 'Choice Band', ability: 'Sand Stream', nature: 'Adamant', evs: { atk: 252, spe: 252 } }),
         mon('Metagross', ['meteormash', 'earthquake'], { item: 'Leftovers', ability: 'Clear Body', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 1', p2: 'move 1' },   // Skarmory Protect vs Rock Slide → block
       { p1: 'move 2', p2: 'move 1' },
       { p1: 'move 1', p2: 'move 1' },   // Protect again (consecutive)
       { p1: 'move 2', p2: 'move 1' },
      ],
      (side, battle) => firstLiveBench(side, battle)),
  });

  // ── (G) SPIKES (entry hazard) + PHAZE — |-sidestart| Spikes, |-damage| [from]
  //         Spikes on switch-in, and |drag| from Roar. Skarmory lays Spikes; a
  //         Roar drags a random bench mon onto the hazard. ──
  S.push({
    id: 'spikes_and_phaze',
    p1: [mon('Skarmory', ['spikes', 'roar', 'drillpeck'], { item: 'Leftovers', ability: 'Keen Eye', nature: 'Impish', evs: { hp: 252, def: 252 } }),
         mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Tyranitar', ['crunch', 'rockslide'], { item: 'Leftovers', ability: 'Sand Stream', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
         mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', ability: 'Thick Fat', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
         mon('Swampert', ['earthquake', 'surf'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 1', p2: 'move 1' },   // Skarmory Spikes → -sidestart on p2 side
       { p1: 'move 2', p2: 'move 1' },   // Roar → drag a random p2 bench mon onto Spikes
       { p1: 'move 3', p2: 'move 1' },
       { p1: 'move 2', p2: 'move 1' },   // Roar again
       { p1: 'move 3', p2: 'move 1' },
      ],
      (side, battle) => firstLiveBench(side, battle)),
  });

  // ── (H) RECOVERY + Rest (sleep) — |-heal| (Recover), |-status| slp (Rest),
  //         |cant| slp, |-curestatus| (wake). Snorlax Rests; Suicune Recovers. ──
  S.push({
    id: 'recover_and_rest',
    p1: [mon('Snorlax', ['rest', 'bodyslam'], { item: 'Leftovers', ability: 'Immunity', nature: 'Careful', evs: { hp: 252, spd: 252 } }),
         mon('Suicune', ['surf', 'icebeam'], { item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    p2: [mon('Tyranitar', ['crunch', 'rockslide'], { item: 'Choice Band', ability: 'Sand Stream', nature: 'Adamant', evs: { atk: 252, spe: 252 } }),
         mon('Metagross', ['meteormash', 'earthquake'], { item: 'Leftovers', ability: 'Clear Body', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 2', p2: 'move 1' },   // trade some HP
       { p1: 'move 1', p2: 'move 1' },   // Snorlax Rest → -status slp
       { p1: 'move 1', p2: 'move 1' },   // asleep → |cant|
       { p1: 'move 1', p2: 'move 1' },
       { p1: 'move 2', p2: 'move 1' },
      ],
      (side, battle) => firstLiveBench(side, battle)),
  });

  // ── PHASE 3 scenarios — the deferred / previously-un-emitted line types.
  // Each targets a documented protocol gap (CLAUDE.md "Protocol honesty note" +
  // the handler-audit manifest's display-only rows): the taunt/disable residual
  // `-end` lines + the Disable retro-edit forms, the Trace `-ability` reveal, the
  // Flash Fire arm/immune/`-end [silent]` cycle, the STATUS_IMMUNE `-immune [from]
  // ability:` block lines, the Synchronize→Lum `-status`/`-enditem`/`-curestatus`
  // interleave (+ LumRest), the MID-BATTLE switch-in ability lines
  // (weather/Intimidate/Pressure), and Leech Seed / Splash / Pay Day.

  // A plan filler: repeat `entry` N times (so a long battle never falls through to
  // the fromPlan default {move 1, move 1}, which could pick an ILLEGAL — taunted/
  // disabled — slot and desync the choice-boundary mapping).
  const fill = (n, entry) => Array.from({ length: n }, () => entry);

  // ── (I) TAUNT lifecycle — |-start| move: Taunt, |cant| move: Taunt, the
  //        residual expiry |-end|…|move: Taunt|[silent], the re-Taunt |-fail|. ──
  S.push({
    id: 'taunt_lifecycle',
    p1: [mon('Gengar', ['taunt', 'thunderbolt', 'psychic'], { item: 'Leftovers', nature: 'Timid', evs: { spa: 252, spe: 252 } }),
         mon('Raikou', ['thunderbolt', 'crunch'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Regice', ['thunderwave', 'icebeam', 'explosion'], { item: 'Leftovers', nature: 'Modest', evs: { hp: 252, spa: 252 } }),
         mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', ability: 'Thick Fat', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 1', p2: 'move 1' },   // Gengar Taunt → -start; Regice's queued TWave → cant
       { p1: 'move 1', p2: 'move 2' },   // re-Taunt → -fail (still taunted); Regice Ice Beam
       { p1: 'move 2', p2: 'move 1' },   // taunt expired (end of t2) → TWave lands → -status
       { p1: 'move 1', p2: 'move 2' },   // Taunt again → -start (fresh)
       ...fill(40, { p1: 'move 2', p2: 'move 2' })],
      (side, battle) => firstLiveBench(side, battle)),
  });

  // ── (J) DISABLE lifecycle — the no-lastMove |-fail| ([still] retro-edit), the
  //        55-accuracy [miss] retro-edit + |-miss|, |-start| Disable|<Move>,
  //        |cant| Disable, the already-disabled |-fail|, the residual expiry
  //        |-end|…|Disable. p1 Gengar is faster, so a landed Disable hits the
  //        move Snorlax queued THIS turn (its lastMove) → |cant|. ──
  S.push({
    id: 'disable_lifecycle',
    p1: [mon('Gengar', ['disable', 'thunderbolt', 'psychic'], { item: 'Leftovers', nature: 'Timid', evs: { spa: 252, spe: 252 } }),
         mon('Raikou', ['thunderbolt', 'crunch'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['seismictoss', 'icebeam', 'bodyslam'], { item: 'Leftovers', ability: 'Thick Fat', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
         mon('Regice', ['icebeam', 'thunderwave'], { item: 'Leftovers', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 1', p2: 'move 1' },   // Disable before Snorlax ever moved → -fail ([still])
       { p1: 'move 1', p2: 'move 1' },   // Disable Seismic Toss (lastMove) → [miss] OR -start + cant
       { p1: 'move 1', p2: 'move 2' },   // re-Disable → -fail (already disabled) / retry after a miss
       ...fill(40, { p1: 'move 2', p2: 'move 2' })],
      (side, battle) => firstLiveBench(side, battle)),
  });

  // ── (K) TRACE — the |-ability|<holder>|<Copied>|[from] ability: Trace|[of] <foe>
  //        reveal, at the LEAD switch-in (framing) AND a MID-BATTLE re-entry
  //        (switch-out reverts to Trace; re-entry re-traces + redraws). ──
  S.push({
    id: 'trace_switchin',
    p1: [mon('Gardevoir', ['psychic', 'thunderbolt'], { ability: 'Trace', nature: 'Modest', evs: { spa: 252, spe: 252 } }),
         mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Skarmory', ['drillpeck', 'tackle'], { item: 'Leftovers', ability: 'Keen Eye', nature: 'Impish', evs: { hp: 252, def: 252 } }),
         mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', ability: 'Thick Fat', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'switch 2', p2: 'move 1' }, // Gardevoir out (revert to Trace)
       { p1: 'move 1', p2: 'switch 2' }, // foe swaps to Thick Fat Snorlax
       { p1: 'switch 2', p2: 'move 1' }, // Gardevoir re-enters → re-trace (Thick Fat) mid-battle
       ...fill(40, { p1: 'move 1', p2: 'move 1' })],
      (side, battle) => firstLiveBench(side, battle)),
  });

  // ── (L) FLASH FIRE cycle — the arm |-start|…|ability: Flash Fire|, the
  //        already-armed |-immune|…|[from] ability: Flash Fire|, the switch-out
  //        |-end|…|ability: Flash Fire|[silent], the re-arm, and a faint while
  //        armed (the singleEvent(End) at faint — gen3_cloudnine_end_v1's probe). ──
  S.push({
    id: 'flashfire_cycle',
    p1: [mon('Houndoom', ['crunch', 'flamethrower'], { ability: 'Flash Fire', nature: 'Timid', evs: { spa: 252, spe: 252 } }),
         mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', ability: 'Thick Fat', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Moltres', ['flamethrower', 'drillpeck'], { ability: 'Pressure', nature: 'Modest', evs: { spa: 252, spe: 252 } }),
         mon('Metagross', ['meteormash', 'earthquake'], { item: 'Leftovers', ability: 'Clear Body', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 1', p2: 'move 1' },   // Flamethrower into FF → -start (arm)
       { p1: 'move 1', p2: 'move 1' },   // Flamethrower again → -immune [from] ability: Flash Fire
       { p1: 'switch 2', p2: 'move 2' }, // Houndoom out → -end …|ability: Flash Fire|[silent]
       { p1: 'switch 2', p2: 'move 1' }, // Houndoom back in; Flamethrower re-arms → -start
       ...fill(40, { p1: 'move 1', p2: 'move 2' })], // Drill Peck grinds the armed Houndoom down
      (side, battle) => firstLiveBench(side, battle)),
  });

  // ── (M) STATUS_IMMUNE block lines — a standalone status move into each
  //        setStatus-phase member: Limber (par), Water Veil (brn), Immunity
  //        (psn/tox), Insomnia (slp) → the |-immune|…|[from] ability: <A>| forms
  //        (the batch-3 honest residual). Hypnosis (60) / Will-O-Wisp (75) also
  //        realize the STATUS-move [miss] retro-edit across seeds. ──
  S.push({
    id: 'status_immune_lines',
    p1: [mon('Gengar', ['thunderwave', 'willowisp', 'toxic', 'hypnosis'], { item: 'Leftovers', nature: 'Timid', evs: { spa: 252, spe: 252 } }),
         mon('Raikou', ['thunderbolt', 'crunch'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Persian', ['slash', 'shadowball'], { ability: 'Limber', nature: 'Jolly', evs: { atk: 252, spe: 252 } }),
         mon('Wailord', ['surf', 'icebeam'], { ability: 'Water Veil', nature: 'Modest', evs: { hp: 252, spa: 252 } }),
         mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
         mon('Hypno', ['psychic', 'seismictoss'], { ability: 'Insomnia', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 1', p2: 'move 1' },   // TWave → Persian: -immune [from] ability: Limber
       { p1: 'move 2', p2: 'switch 2' }, // Wailord in; WoW → -immune Water Veil (or [miss])
       { p1: 'move 3', p2: 'switch 3' }, // Snorlax in; Toxic → -immune Immunity (or [miss])
       { p1: 'move 4', p2: 'switch 4' }, // Hypno in; Hypnosis → -immune Insomnia (or [miss])
       { p1: 'move 4', p2: 'move 1' },   // Hypnosis again (more [miss]/immune realizations)
       { p1: 'move 1', p2: 'move 1' },   // TWave → Hypno: LANDS → -status par (contrast)
       ...fill(40, { p1: 'move 4', p2: 'move 1' })],
      (side, battle) => firstLiveBench(side, battle)),
  });

  // ── (N) SYNCHRONIZE → LUM interleave + LumRest — |-status|holder| →
  //        |-status|source|…[from] ability: Synchronize|[of] holder| →
  //        |-enditem|source|Lum Berry|[eat]| → |-curestatus|source|; then
  //        Snorlax LumRest: |-status slp|[from] move: Rest| → |-enditem| →
  //        |-curestatus|[msg]| → |-heal|[silent]. ──
  S.push({
    id: 'synchronize_lum_rest',
    p1: [mon('Espeon', ['psychic', 'calmmind'], { ability: 'Synchronize', nature: 'Timid', evs: { spa: 252, spe: 252 } }),
         mon('Snorlax', ['rest', 'bodyslam'], { item: 'Lum Berry', ability: 'Thick Fat', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Umbreon', ['thunderwave', 'crunch'], { item: 'Lum Berry', ability: 'Synchronize', nature: 'Careful', evs: { hp: 252, spd: 252 } }),
         mon('Metagross', ['meteormash', 'earthquake'], { item: 'Leftovers', ability: 'Clear Body', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 1', p2: 'move 1' },   // TWave → Espeon par → Synchronize reflect → Umbreon's Lum eats
       { p1: 'switch 2', p2: 'move 2' }, // Snorlax in; Crunch chips it (Rest needs missing HP)
       { p1: 'move 1', p2: 'move 2' },   // Snorlax Rest → LumRest chain
       ...fill(40, { p1: 'move 2', p2: 'move 2' })],
      (side, battle) => firstLiveBench(side, battle)),
  });

  // ── (O) MID-BATTLE switch-in ability lines — a mid-battle Sand Stream
  //        (-weather SET), Pressure (-ability …|Pressure), and Intimidate three
  //        ways: vs a plain foe (-unboost), vs a SUBSTITUTE (the gen3 -immune),
  //        vs Clear Body (the blocked form). ──
  S.push({
    id: 'midswitch_ability_lines',
    p1: [mon('Salamence', ['rockslide', 'earthquake'], { ability: 'Intimidate', nature: 'Adamant', evs: { atk: 252, spe: 252 } }),
         mon('Tyranitar', ['rockslide', 'crunch'], { item: 'Leftovers', ability: 'Sand Stream', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
         mon('Suicune', ['surf', 'icebeam'], { item: 'Leftovers', ability: 'Pressure', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    p2: [mon('Metagross', ['meteormash', 'earthquake'], { item: 'Leftovers', ability: 'Clear Body', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
         mon('Suicune', ['substitute', 'surf'], { item: 'Leftovers', ability: 'Pressure', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'switch 2', p2: 'move 1' }, // Tyranitar in mid-battle → -weather SET
       { p1: 'switch 3', p2: 'move 1' }, // Suicune in → Pressure reveal
       { p1: 'move 1', p2: 'switch 2' }, // foe Suicune in
       { p1: 'move 2', p2: 'move 1' },   // foe Substitutes
       { p1: 'switch 2', p2: 'move 2' }, // Salamence in vs SUB → -ability + -immune
       { p1: 'move 1', p2: 'switch 2' }, // Metagross back in
       { p1: 'switch 2', p2: 'move 1' }, // Suicune in (Pressure again)
       { p1: 'switch 2', p2: 'move 1' }, // Salamence in vs Clear Body → blocked form
       ...fill(40, { p1: 'move 1', p2: 'move 1' })],
      (side, battle) => firstLiveBench(side, battle)),
  });

  // ── (P) LEECH SEED + SPLASH + PAY DAY — |-start|…|move: Leech Seed, the
  //        residual |-damage|…|[from] Leech Seed|[of]| + the seeder |-heal|[silent],
  //        the Grass -immune, the re-seed -fail, Splash's |-nothing|, Pay Day's
  //        |-fieldactivate|. ──
  S.push({
    id: 'leechseed_splash_payday',
    p1: [mon('Venusaur', ['leechseed', 'sludgebomb'], { item: 'Leftovers', ability: 'Overgrow', nature: 'Bold', evs: { hp: 252, def: 252 } }),
         mon('Persian', ['payday', 'splash', 'slash'], { ability: 'Limber', nature: 'Jolly', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Snorlax', ['seismictoss', 'bodyslam'], { item: 'Leftovers', ability: 'Thick Fat', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
         mon('Sceptile', ['leafblade', 'earthquake'], { ability: 'Overgrow', nature: 'Jolly', evs: { atk: 252, spe: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 1', p2: 'move 1' },   // Leech Seed → -start (or a 90-acc miss)
       { p1: 'move 1', p2: 'move 1' },   // re-seed → -fail (or land after a miss)
       { p1: 'switch 2', p2: 'move 1' }, // Persian in (Seismic Toss keeps it alive a while;
                                         //   a seeded Snorlax keeps draining → heals Persian)
       { p1: 'move 2', p2: 'move 1' },   // Splash → -nothing
       { p1: 'move 1', p2: 'move 1' },   // Pay Day → -fieldactivate
       { p1: 'switch 2', p2: 'move 1' }, // Venusaur back in
       { p1: 'move 1', p2: 'move 1' },   // Leech Seed → -fail (still seeded) or land
       { p1: 'move 1', p2: 'switch 2' }, // Sceptile in (Grass) → -immune
       ...fill(40, { p1: 'move 2', p2: 'move 1' })],
      (side, battle) => firstLiveBench(side, battle)),
  });

  // ── (Q) F1 — Leech Seed into a SUBSTITUTE'd foe: the sub blocks the volatile, so
  //        gen3 retro-edits the announce to `|move|<user>|Leech Seed||[still]` +
  //        emits `|-fail|<user>` (IDENTICAL to the already-seeded fail form, NOT a
  //        bare `|move|`). Suicune subs, Venusaur seeds the sub across turns. The
  //        Sludge Bomb line grinds the sub down so the block re-realizes on re-sub.
  //        (Review finding F1; probe `harness/probe_f1_f2_f3_lines.js`.) ──
  S.push({
    id: 'leechseed_into_substitute',
    p1: [mon('Venusaur', ['leechseed', 'sludgebomb'], { item: 'Leftovers', ability: 'Overgrow', nature: 'Bold', evs: { hp: 252, def: 252 } }),
         mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', ability: 'Thick Fat', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Suicune', ['substitute', 'surf'], { item: 'Leftovers', ability: 'Pressure', nature: 'Bold', evs: { hp: 252, def: 252 } }),
         mon('Sceptile', ['leafblade', 'earthquake'], { ability: 'Overgrow', nature: 'Jolly', evs: { atk: 252, spe: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 2', p2: 'move 1' },   // Suicune subs (Sludge Bomb chips the sub)
       { p1: 'move 1', p2: 'move 2' },   // Leech Seed into the SUB → [still] + -fail (F1)
       { p1: 'move 1', p2: 'move 2' },   // seed again into the sub → [still] + -fail
       { p1: 'move 2', p2: 'move 1' },   // Sludge Bomb; Suicune re-subs when the sub broke
       { p1: 'move 1', p2: 'move 2' },   // seed into a fresh sub → [still] + -fail
       ...fill(40, { p1: 'move 2', p2: 'move 2' })],
      (side, battle) => firstLiveBench(side, battle)),
  });

  // ── (R) F2/F3 (Flash Fire) — a gen3 `onTryHit`-class ABILITY immunity is POST-accuracy,
  //        so a MISSED Fire move into an ARMED Flash Fire holder shows `[miss]`+`-miss`
  //        (F2 — NOT `-immune`), while a LANDED one shows `|-immune|<t>|[from] ability:
  //        Flash Fire`. Houndoom arms FF turn 1 (Flamethrower), then Moltres Fire-Blasts
  //        (85 acc) the ARMED holder — realizing BOTH the miss + the armed-immune line
  //        across the seed sweep. Houndoom Earthquakes back but Moltres is FLYING → Ground
  //        0× immune (a harmless repeatable damaging-move `-immune` the engine already
  //        emits; no damage, so Moltres never dies mid-phase). Houndoom stays IN and
  //        faints in place when Suicune finally Ice-Beams it (a faint emits no FF `-end`,
  //        avoiding the alive-switch-out `-end` timing). Review finding F2 + F3; probes
  //        `probe_f1_f2_f3_lines.js` / `probe_f2_ff_armed_miss.js`. ──
  S.push({
    id: 'flashfire_tryhit_miss',
    p1: [mon('Moltres', ['flamethrower', 'fireblast'], { item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } }),
         mon('Suicune', ['icebeam', 'surf'], { item: 'Leftovers', nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Houndoom', ['earthquake', 'crunch'], { item: 'Leftovers', ability: 'Flash Fire', nature: 'Bold', evs: { hp: 252, def: 252 } }),
         mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', ability: 'Thick Fat', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 1', p2: 'move 1' },   // Flamethrower ARMS Flash Fire (-start); Houndoom EQ (Moltres immune)
       ...fill(10, { p1: 'move 2', p2: 'move 1' }), // Fire Blast into ARMED FF → -immune [from] Flash Fire / [miss]
       { p1: 'switch 2', p2: 'move 1' },  // Suicune in (Moltres stays alive — it was Ground-immune)
       ...fill(40, { p1: 'move 1', p2: 'move 1' })], // Suicune Ice Beam KOs Houndoom (faints IN → no FF -end)
      (side, battle) => firstLiveBench(side, battle)),
  });

  // ── (S) F2/F3 (Water Absorb) — the SAME POST-accuracy rule for the absorb abilities:
  //        a MISSED Water move into Water Absorb shows `[miss]`+`-miss` (F2), a LANDED one
  //        `|-immune|<t>|[from] ability: Water Absorb` (F3 — NOT a plain `-immune`).
  //        Suicune Hydro-Pumps (80 acc) a Water Absorb Politoed across the seed sweep,
  //        realizing both. Politoed Ice-Beams back (Suicune is bulky + Leftovers → survives
  //        the phase). Review finding F2 + F3; probe `probe_f1_f2_f3_lines.js`. ──
  S.push({
    id: 'waterabsorb_tryhit_miss',
    p1: [mon('Suicune', ['hydropump', 'icebeam'], { item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } }),
         mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', ability: 'Thick Fat', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Politoed', ['icebeam', 'surf'], { item: 'Leftovers', ability: 'Water Absorb', nature: 'Bold', evs: { hp: 252, def: 252 } }),
         mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', ability: 'Thick Fat', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: fromPlan(
      fill(30, { p1: 'move 1', p2: 'move 1' }), // Hydro Pump into Water Absorb → -immune [from] Water Absorb / [miss]
      (side, battle) => firstLiveBench(side, battle)),
  });

  // ── (T) MID-BATTLE Intimidate into a −6-FLOORED foe — the CLAMPED-applied delta.
  //        The port used to hardcode `|-unboost|<foe>|atk|1`, but Showdown emits the
  //        CLAMPED-APPLIED delta: a foe already at the −6 Atk floor drops by 0 →
  //        `|-unboost|<foe>|atk|0` (the line is STILL emitted — probe-verified, NOT
  //        omitted, NOT a `-fail`). p1's Intimidate Salamences pivot in one after
  //        another, each dropping p2's Snorlax Atk by 1 (atk|1 ×6) until it FLOORS at
  //        −6, then a 7th fresh Intimidate switch-in emits the `atk|0` floor line.
  //        (Bridge A/B fuzzer find; probe `harness/probe_intimidate_floor.js`.) ──
  S.push({
    id: 'intimidate_atk_floor',
    p1: [mon('Salamence', ['dragonclaw', 'splash'], { ability: 'Intimidate', nature: 'Adamant', evs: { atk: 252, spe: 252 } }),
         mon('Salamence', ['dragonclaw', 'splash'], { ability: 'Intimidate', nature: 'Adamant', evs: { atk: 252, spe: 252 } }),
         mon('Salamence', ['dragonclaw', 'splash'], { ability: 'Intimidate', nature: 'Adamant', evs: { atk: 252, spe: 252 } }),
         mon('Salamence', ['dragonclaw', 'splash'], { ability: 'Intimidate', nature: 'Adamant', evs: { atk: 252, spe: 252 } }),
         mon('Salamence', ['dragonclaw', 'splash'], { ability: 'Intimidate', nature: 'Adamant', evs: { atk: 252, spe: 252 } }),
         mon('Salamence', ['dragonclaw', 'splash'], { ability: 'Intimidate', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'splash'], { item: 'Leftovers', ability: 'Thick Fat', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: fromPlan(
      // The lead Salamence Intimidated at the >start switch-in (atk 0 → −1). Now pivot
      // in a fresh Salamence each turn: −1→−2→−3→−4→−5→−6 (atk|1 ×5) then the FLOOR
      // switch-in (−6→−6, atk|0). p2 Splashes so it just sits and eats the drops.
      [{ p1: 'switch 2', p2: 'move 2' }, // −1 → −2  (atk|1)
       { p1: 'switch 2', p2: 'move 2' }, // −2 → −3  (atk|1)
       { p1: 'switch 2', p2: 'move 2' }, // −3 → −4  (atk|1)
       { p1: 'switch 2', p2: 'move 2' }, // −4 → −5  (atk|1)
       { p1: 'switch 2', p2: 'move 2' }, // −5 → −6  (atk|1)
       { p1: 'switch 2', p2: 'move 2' }, // −6 → −6 AT FLOOR → atk|0
       ...fill(40, { p1: 'move 2', p2: 'move 2' })],
      (side, battle) => firstLiveBench(side, battle)),
  });

  return S;
}

async function main() {
  // A modest per-scenario seed sweep — enough to realize the stochastic branches
  // (crit / flinch / freeze / para / phaze target) across the corpus while keeping
  // the golden a reasonable size (the RAW-line capture is verbose).
  const SEEDS_PER_SCEN = 6;
  const seeds = buildSeeds(SEEDS_PER_SCEN, MASTER_SEED);

  const lines = [];
  lines.push('# protocol_capture_golden.txt — Gen-3 PROTOCOL-EMISSION (level-2) golden.');
  lines.push('# The RAW OMNISCIENT `|...|` protocol stream Showdown emits, verbatim, per battle.');
  lines.push('# This is the byte-for-byte TARGET the Rust engine (protocol.rs) will reproduce.');
  lines.push('# See src/rust_sim/PROTOCOL_EMISSION_DESIGN.md + tests/vectors/protocol_inventory.md.');
  lines.push('#');
  lines.push('# Record grammar (TAB-delimited headers; the L lines are RAW protocol bytes):');
  lines.push('#   SCEN  <id>');
  lines.push('#   TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('#   BATTLE  <id>  <battleNo>  <seed m,n,o,p>   (the >start seed)');
  lines.push('#   INIT  <id>  <battleNo>  <initSeed m,n,o,p>   (the pre-first-decision seed)');
  lines.push('#   DEC   <id>  <battleNo>  <decNo>  <request:move|switch>  <forceP1> <forceP2>  <choiceP1> <choiceP2>  <seedAfter m,n,o,p>');
  lines.push('#   L     <id>  <battleNo>  <lineNo>  <RAW protocol line, verbatim — MAY contain tabs/pipes>');
  lines.push('#   END   <id>  <battleNo>  <ended:0|1>  <winner:p1|p2|tie|none>  <nLines>');
  lines.push('#');
  lines.push('# The L payload is the LAST field: everything after the 4th tab is the raw line');
  lines.push('# (so a `|...` line with embedded tabs/pipes round-trips exactly). |t:| + |debug|');
  lines.push('# lines ARE captured (the engine emits them) though poke-env ignores them.');

  const S = scenarios();
  const failures = [];
  let totalLines = 0;
  let totalDecs = 0;
  let winRows = 0;
  let tieRows = 0;
  const lineTypeCounts = new Map();

  for (const sc of S) {
    lines.push(`SCEN\t${sc.id}`);
    lines.push(`TEAM\t${sc.id}\tp1\t${Teams.pack(sc.p1)}`);
    lines.push(`TEAM\t${sc.id}\tp2\t${Teams.pack(sc.p2)}`);

    let battleNo = 0;
    for (const seed of seeds) {
      let rec;
      try { rec = await runBattle(sc, seed); }
      catch (e) { failures.push(`${sc.id} seed ${seed}: ${e.message}`); battleNo++; continue; }
      if (rec.gen !== 3) { failures.push(`${sc.id}: expected gen 3, got ${rec.gen}`); break; }
      if (!rec.lines.length) { failures.push(`${sc.id} seed ${seed}: captured NO protocol lines`); battleNo++; continue; }

      const seedStr = seed.join(',');
      lines.push(['BATTLE', sc.id, battleNo, seedStr].join('\t'));
      lines.push(['INIT', sc.id, battleNo, rec.initSeed ? rec.initSeed : '-'].join('\t'));

      rec.decisions.forEach((d, di) => {
        lines.push([
          'DEC', sc.id, battleNo, di, d.request,
          d.force[0] ? 1 : 0, d.force[1] ? 1 : 0,
          d.choiceP1, d.choiceP2, d.seedAfter,
        ].join('\t'));
        totalDecs++;
      });

      // The RAW lines, verbatim, in emission order. Blank split-lines ("|") are kept
      // so the stream round-trips exactly (poke-env parses a bare "|" as a no-op).
      rec.lines.forEach((raw, li) => {
        lines.push(['L', sc.id, battleNo, li, raw].join('\t'));
        totalLines++;
        // Tally the line TYPE (the token after the first pipe) for the run report.
        const t = raw.startsWith('|') ? (raw.split('|')[1] || '<blank>') : '<no-pipe>';
        lineTypeCounts.set(t, (lineTypeCounts.get(t) || 0) + 1);
      });

      let winTok = 'none';
      if (rec.ended) {
        if (rec.winner === 'P1') winTok = 'p1';
        else if (rec.winner === 'P2') winTok = 'p2';
        else if (rec.winner === '' || rec.winner === null) winTok = 'tie';
      }
      if (winTok === 'p1' || winTok === 'p2') winRows++;
      if (winTok === 'tie') tieRows++;
      lines.push(['END', sc.id, battleNo, rec.ended ? 1 : 0, winTok, rec.lines.length].join('\t'));
      battleNo++;
    }
  }

  if (failures.length) {
    console.error('PROTOCOL CAPTURE FAILURES:\n  ' + failures.slice(0, 30).join('\n  '));
    process.exit(1);
  }
  if (totalLines < 500) {
    console.error(`PROTOCOL CAPTURE: too few lines (${totalLines}); expected a rich corpus.`);
    process.exit(1);
  }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');

  // A compact per-run report (the inventory doc is generated from these types).
  const sortedTypes = [...lineTypeCounts.entries()].sort((a, b) => b[1] - a[1]);
  console.error(
    `protocol capture: ${S.length} scenarios × ${SEEDS_PER_SCEN} seeds, ` +
    `${totalLines} raw lines, ${totalDecs} decisions, ${winRows} wins + ${tieRows} ties, ` +
    `${sortedTypes.length} distinct line types -> ${OUT}`
  );
  console.error('distinct line types (count):');
  for (const [t, c] of sortedTypes) console.error(`  |${t}|  ${c}`);
  process.exit(0);
}

// Reusable exports (gen_writeline_capture.js drives the SAME scenario corpus
// through per-write chunk capture — one scenario source, two goldens).
module.exports = { scenarios, buildSeeds, mon, encodeChoice, forceSwitchTable, MASTER_SEED, FORMAT };

if (require.main === module) {
  main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
}
