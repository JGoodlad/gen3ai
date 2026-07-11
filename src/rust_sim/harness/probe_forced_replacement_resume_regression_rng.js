// probe_forced_replacement_resume_regression_rng.js — GROUND TRUTH for the
// forced-replacement RESUME regression pin
// (`forced_replacement_resume_runs_the_post_replacement_move_decision`).
//
// The bug: after a mid-turn forced replacement changes the active mon to one with
// FEWER moves, a scripted `move K` that the NEW mon doesn't have is REJECTED by the
// sim's `side.choose` (drawing NOTHING, leaving the `move` request OPEN) — the real
// turn runs on the NEXT valid submission. The port used to RUN a full turn for that
// invalid decision (its own move no-op'd but the FOE's move + residual + Quick Claw
// drew), diverging every later seed. The FIX: `run_full_battle` validates the move
// decision (`move_decision_is_legal`) and SKIPS an out-of-range slot.
//
// The scenario is DRAW-FREE on switch-in (no weather ability / Intimidate) so the port's
// `start_with_switchins` init seed == the sim's pre-first-decision seed == the `>start`
// seed — so the Rust pin can use the SAME seed and its boundaries align 1:1.
//   p1: Aerodactyl (3 moves: earthquake, rockslide, ancientpower) + Snorlax (2 moves).
//   p2: Zapdos (thunderbolt) that KOs the Aerodactyl (Electric SE on Flying) → p1 replaces
//       with the 2-move Snorlax; we then submit the INVALID `move 3` for Snorlax (rejected,
//       draws 0), then the VALID `move 1`.
//
// Run:  node src/rust_sim/harness/probe_forced_replacement_resume_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return { species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31,
    nature: opts.nature || 'Serious', level: 100, gender: 'N' };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

const SEED = [11, 22, 33, 44];
const P1 = [
  mon('Aerodactyl', ['earthquake', 'rockslide', 'ancientpower'], { ability: 'Rock Head', nature: 'Adamant', evs: { atk: 252, spe: 252 } }),
  mon('Snorlax', ['bodyslam', 'earthquake'], { ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
];
const P2 = [
  mon('Zapdos', ['thunderbolt', 'roost'], { ability: 'Pressure', nature: 'Modest', evs: { spa: 252, spe: 252 } }),
  mon('Blissey', ['seismictoss', 'icebeam'], { ability: 'Natural Cure', nature: 'Bold', evs: { hp: 252, def: 252 } }),
];

// Submissions (per SUBMISSION, incl. the rejected phantom):
//   sub0: m0/m0  — Aerodactyl Earthquake / Zapdos Thunderbolt (Tbolt SE on Aero/Flying → KO?)
//   sub1: -/s?   — (only if p1 is KO'd sub0) p1 replaces with Snorlax
//   ... we DISCOVER the boundary shape at runtime and inject the phantom after the replacement.
const SUBS = [
  { p1: 'move 1', p2: 'move 1' },   // Aero EQ (immune vs Zapdos) / Zapdos Tbolt → SE KO Aerodactyl
];

async function main() {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(SEED)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(P1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(P2) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  let draws = 0;
  const rng = battle.prng.rng; const realNext = rng.next.bind(rng);
  rng.next = function (...a) { draws++; return realNext(...a); };

  console.log(`initSeed (pre-first-decision) = ${battle.prng.getSeed()}`);

  // Drive: turn 0 (KO Aero), then p1 forced replacement (Snorlax), then the PHANTOM
  // (`move 3` for the 2-move Snorlax → rejected), then the real `move 1`.
  const script = [
    { p1: 'move 1', p2: 'move 1' },                 // sub0: KO Aerodactyl (Tbolt SE)
    { forceP1: 'switch 2' },                         // sub1: p1 replaces with Snorlax
    { p1: 'move 3', p2: 'move 1', phantom: true },   // sub2: INVALID (Snorlax has 2 moves)
    { p1: 'move 1', p2: 'move 1' },                 // sub3: the real turn runs
  ];
  let subI = 0, safety = 0, boundaryNo = 0, lastSeed = battle.prng.getSeed();
  while (!battle.ended && safety < 40) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    if (subI >= script.length) break;
    const force = [false, false];
    for (let s = 0; s < 2; s++) { const req = battle.sides[s].activeRequest; if (req && req.forceSwitch && req.forceSwitch[0]) force[s] = true; }
    const d0 = draws;
    const entry = script[subI]; subI++;
    if (entry.forceP1 && force[0]) { try { streams.omniscient.write(`>p1 ${entry.forceP1}`); } catch (e) {} }
    if (entry.forceP2 && force[1]) { try { streams.omniscient.write(`>p2 ${entry.forceP2}`); } catch (e) {} }
    if (entry.p1) { try { streams.omniscient.write(`>p1 ${entry.p1}`); } catch (e) {} }
    if (entry.p2) { try { streams.omniscient.write(`>p2 ${entry.p2}`); } catch (e) {} }
    for (let k = 0; k < 24; k++) await tick();
    const after = battle.prng.getSeed();
    const advanced = after !== lastSeed || draws !== d0 || rs === 'switch';
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const fmt = (m) => `${m.species.name} ${m.hp}/${m.maxhp}${m.fainted ? ' FNT' : ''}${m.status ? ' ' + m.status : ''}`;
    if (advanced) {
      console.log(`BOUNDARY ${boundaryNo}: req=${rs} force=[${force}] sub=${JSON.stringify(entry)} draws=${draws - d0} seedAfter=${after}`);
      console.log(`            p1=${fmt(a0)} | p2=${fmt(a1)} left=[${battle.sides[0].pokemonLeft},${battle.sides[1].pokemonLeft}]`);
      boundaryNo++;
    } else {
      console.log(`PHANTOM (rejected, draws=0): req=${rs} sub=${JSON.stringify(entry)} seed UNCHANGED=${after}`);
    }
    lastSeed = after;
  }
  console.log(`\nended=${battle.ended} winner=${JSON.stringify(battle.winner)} totalDraws=${draws} realBoundaries=${boundaryNo}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
