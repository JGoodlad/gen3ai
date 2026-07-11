// probe_wisp_alias_regression_rng.js — GROUND-TRUTH seed for the FZ2 move-alias pin.
//
// FZ2 pins `gen3_move_alias_resolution_v1`: a packed team can carry a move ALIAS (`wisp`
// for Will-O-Wisp), which Showdown resolves at `dex.moves.get()` and RUNS as the canonical
// move (drawing its accuracy roll). The port used to return None for the alias and NO-OP
// the move (drawing nothing) -> a draw-count DESYNC (the e2e_86 cascade). This probe
// constructs the minimal scenario — a Gengar whose slot-0 move is spelled `wisp` uses it
// into a healthy foe, which gets BURNED — and captures the post-turn seed so the port's
// SEED-AFTER (with alias resolution) matches bit-for-bit.
//
// Run:  node src/rust_sim/harness/probe_wisp_alias_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31,
    nature: opts.nature || 'Hardy', level: opts.level || 100, gender: opts.gender || 'N',
  };
}

// Gengar uses `wisp` (the Will-O-Wisp alias) into a slow bulky Snorlax; Snorlax uses a
// never-miss self-move (no extra draw contention) so the only per-move draws are Gengar's
// WoW accuracy + Snorlax's Body Slam + Quick Claw. Gengar (Timid, fast) moves first.
function run(seed) {
  const p1 = [mon('Gengar', ['wisp', 'icepunch', 'firepunch', 'explosion'], { evs: { spa: 252, spe: 252 }, nature: 'Timid' })];
  // Snorlax uses Amnesia (a modeled +2 SpD self-boost, never-miss, draw-free apply) — it can't
  // miss/confound and is a move the port MODELS (Curse is type-conditional → not modeled).
  const p2 = [mon('Snorlax', ['amnesia', 'bodyslam', 'rest', 'earthquake'], { evs: { hp: 252 } })];
  const battle = new Battle({ formatid: FORMAT, seed });
  battle.setPlayer('p1', { name: 'P1', team: Teams.pack(p1) });
  battle.setPlayer('p2', { name: 'P2', team: Teams.pack(p2) });
  const seedBefore = battle.prng.getSeed();
  let draws = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = (...a) => { draws++; return realNext(...a); };

  // Submit by SLOT (the sim canonicalizes `wisp`->`willowisp` at team-import, so its moveslot
  // id is `willowisp`; `move 1` is Gengar's slot-0 Will-O-Wisp either way). The port's packed
  // team keeps the `wisp` token and resolves it via the alias map at `dex.moves()` time — the
  // exact behaviour this pin exercises.
  battle.choose('p1', 'move 1');
  battle.choose('p2', 'move 1');

  const p2a = battle.sides[1].active[0];
  return { seedBefore, seedAfter: battle.prng.getSeed(), draws, foeStatus: p2a.status || 'none', foeHp: p2a.hp, foeMax: p2a.maxhp };
}

function main() {
  // Find a seed where WoW LANDS (burns) — a clean STATE assertion (foe burned) plus the seed.
  for (let s = 0; s < 2000; s++) {
    const seed = [s & 0xffff, (s * 5 + 1) & 0xffff, (s * 9 + 3) & 0xffff, (s * 61 + 7) & 0xffff];
    const r = run(seed);
    if (r.foeStatus !== 'brn') continue;
    console.log('=== FZ2 ground truth (move-alias `wisp` resolves + runs) ===');
    console.log(`  init seed (Rust seeds this): ${seed.join(',')}`);
    console.log(`  SEED-BEFORE  = ${r.seedBefore}`);
    console.log(`  SEED-AFTER   = ${r.seedAfter}   draws=${r.draws}`);
    console.log(`  foe (Snorlax) status=${r.foeStatus} hp=${r.foeHp}/${r.foeMax}  (WoW LANDED -> burned)`);
    console.log('  => the port must resolve `wisp`->willowisp, RUN it (draw the accuracy), burn the foe, and match SEED-AFTER');
    return;
  }
  console.log('NO landed-WoW seed found in 2000 tries (unexpected).');
}

main();
