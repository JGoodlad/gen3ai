// probe_transform_pin.js — GROUND-TRUTH generator for the ROUND-33 TRANSFORM regression pins.
//
// Same methodology as `probe_ptrap_pin.js` (round 29/32): for each pinned board, drive the
// OMNISCIENT gen3 BattleStream with the SAME packed team strings the Rust pin uses, and print
//   (a) the POST-CONSTRUCTION seed the Rust pin must be seeded at (`start_with_switchins` is
//       draw-free while the sim spends its first draws on the turn-0 construction window),
//   (b) the per-decision protocol lines, and
//   (c) the per-decision post-turn seed — so a MISPLACED or MISSING draw fails the pin even
//       when the emitted lines happen to coincide. Transform is draw-FREE, so the seeds are
//       the whole proof that the copy (and in particular the HYBRID speed cache) is right.
//
// Run: node harness/probe_transform_pin.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));

const tick = () => new Promise((r) => setTimeout(r, 0));
const KEEP = (l) => l && !l.startsWith('|t:|') && l !== '|' && !l.startsWith('|debug|');

async function run(p1, p2, seed, choices) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  (async () => { for await (const ch of streams.omniscient) for (const l of String(ch).split('\n')) lines.push(l); })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2 })}`);
  for (let i = 0; i < 16; i++) await tick();
  const battle = stream.battle;
  const initSeed = String(battle.prng.getSeed());
  const per = [];
  let lo = lines.length;
  for (const [c1, c2] of choices) {
    if (c1) streams.omniscient.write(`>p1 ${c1}`);
    if (c2) streams.omniscient.write(`>p2 ${c2}`);
    for (let k = 0; k < 16; k++) await tick();
    per.push({ lines: lines.slice(lo).filter(KEEP), seed: String(battle.prng.getSeed()) });
    lo = lines.length;
    if (battle.ended) break;
  }
  return { initSeed, per, battle, lines };
}

const SEEDS = [];
for (let s = 0; s < 90; s++) SEEDS.push([s * 7 + 1, s * 3 + 2, s * 5 + 4, s * 11 + 3]);

async function scenario(name, p1, p2, choices, pred, report) {
  for (const seed of SEEDS) {
    const r = await run(p1, p2, seed, choices);
    if (pred && !pred(r)) continue;
    console.log(`\n######## ${name} ########`);
    console.log(`  raw seed          = ${JSON.stringify(seed)}`);
    console.log(`  POST-CONSTRUCTION = "${r.initSeed}"   <-- seed the Rust pin with THIS`);
    r.per.forEach((d, i) => {
      console.log(`  dec${i}: seed_after="${d.seed}"`);
      d.lines.forEach((l) => console.log(`        ${l}`));
    });
    if (report) report(r);
    return r;
  }
  console.log(`\n######## ${name} ######## -- NO SEED SATISFIED THE GUARD`);
  return null;
}

// ── Packed teams (byte-identical to the Rust pins' `opts_cg` strings) ────────────────────
// Ditto with ONE move (the gen3-randbats shape — the `ppUps[i] || 0` trap).
const DITTO_1 = 'Ditto||MetalPowder|Limber|transform|Hardy|85,85,85,85,85,85|N||||';
const DITTO_1_GENGAR =
  'Ditto||MetalPowder|Limber|transform|Hardy|85,85,85,85,85,85|N||||]Gengar||Leftovers|Levitate|splash|Hardy|85,85,85,85,85,85|M||||';
// A 4-move target with a self-target move, a foe move, a low-pp move and Splash, so the
// per-slot maxpp differences are visible.
const LAX_4 = 'Snorlax||Leftovers|ThickFat|swordsdance,bodyslam,rest,splash|Adamant|252,252,,,,4|M|31,31,31,31,31,3|||';
const LAX_SD = 'Snorlax||Leftovers|ThickFat|swordsdance,bodyslam,splash|Hardy|85,85,85,85,85,85|M||||';
const DITTO_MIRROR = 'Ditto||MetalPowder|Limber|transform,splash|Hardy|85,85,85,85,85,85|N||||';
const LAX_SUB = 'Snorlax||Leftovers|ThickFat|substitute,splash|Hardy|85,85,85,85,85,85|M||||';
const LAX_BOOM =
  'Snorlax||Leftovers|ThickFat|explosion,splash|Hardy|85,85,85,85,85,85|M||||]Blissey||Leftovers|NaturalCure|splash|Hardy|85,85,85,85,85,85|F||||';
// TF9: an INTIMIDATE carrier — proves the ability IS copied while its `onStart` does NOT
// re-fire (the `gen > 3` gate in setAbility), and that the revert goes to the SET ability.
const GYARA = 'Gyarados||Leftovers|Intimidate|splash,bodyslam|Hardy|85,85,85,85,85,85|M||||';
const BLISS_MIMIC = 'Blissey||Leftovers|NaturalCure|mimic,splash,softboiled,toxic|Hardy|85,85,85,85,85,85|F||||';
// p2 = a Mimic carrier PLUS a second mon whose moves the Mimic carrier does NOT have, so a
// transformed Ditto (which copied the carrier's WHOLE moveset) does NOT already know the foe's
// lastMove — isolating the `source.transformed` fail from the `already knows it` fail.
const BLISS_MIMIC_LAX = 'Blissey||Leftovers|NaturalCure|mimic,splash,softboiled,toxic|Hardy|85,85,85,85,85,85|F||||]Snorlax||Leftovers|ThickFat|curse,splash|Hardy|85,85,85,85,85,85|M||||';
// NO NICKNAME (the `|Species|...` packed form → `set.name` empty), so the ident must be
// derived from the mon's BASE SPECIES rather than the copied one.
const DITTO_NONICK = '|Ditto|MetalPowder|Limber|transform|Hardy|85,85,85,85,85,85|N||||';
const LAX_NONICK = '|Snorlax|Leftovers|ThickFat|bodyslam,splash|Hardy|85,85,85,85,85,85|M||||';

const dumpUser = (r) => {
  const u = r.battle.sides[0].active[0];
  console.log(`     USER: species=${u.species.id} transformed=${u.transformed} speedCache=${u.speed} stored=${JSON.stringify(u.storedStats)} hp=${u.hp}/${u.maxhp}`);
  console.log(`           ability=${u.ability} types=${JSON.stringify(u.types)} boosts=${JSON.stringify(u.boosts)}`);
  console.log(`           slots=${JSON.stringify(u.moveSlots.map((m) => `${m.id}:${m.pp}/${m.maxpp}`))}`);
};

async function main() {
  const transformed = (r) => r.per[0].lines.some((l) => l.startsWith('|-transform|'));

  // TF1 — the FULL COPY: species/stats/types/ability/boosts + the `ppUps[i] || 0` maxpp
  //       ladder, then a copied move USED (PP 5→4) — and the copied-Speed action order.
  await scenario('TF1 copy + maxpp ladder + copied-move use', DITTO_1, LAX_4,
    [['move 1', 'move 1'], ['move 2', 'move 4']], transformed, dumpUser);

  // TF2 — REVERT ON SWITCH-OUT: switch out, switch back, Transform's OWN pp persists (15/16).
  await scenario('TF2 revert on switch-out', DITTO_1_GENGAR, LAX_SD,
    [['move 1', 'move 3'], ['switch 2', 'move 3'], ['switch 2', 'move 3']], transformed, dumpUser);

  // TF3 — the DITTO MIRROR: the 2nd Transform FAILS (`target.transformed`), `[still]` + -fail.
  await scenario('TF3 ditto mirror second transform fails', DITTO_MIRROR, DITTO_MIRROR,
    [['move 1', 'move 1'], ['move 1', 'move 2']],
    (r) => r.per[0].lines.some((l) => l.includes('|-fail|')), dumpUser);

  // TF4 — a SUBSTITUTE does NOT block (bypasssub) and a PROTECT does NOT block (no protect flag).
  await scenario('TF4 transform through a Substitute', DITTO_MIRROR, LAX_SUB,
    [['move 2', 'move 1'], ['move 1', 'move 2']],
    (r) => r.per[1].lines.some((l) => l.startsWith('|-transform|')), dumpUser);

  // TF5 — REVERT ON FAINT: copy an Explosion user, blow up, the corpse reads back as Ditto.
  await scenario('TF5 revert on faint', DITTO_1_GENGAR, LAX_BOOM,
    [['move 1', 'move 2'], ['move 1', 'move 2'], ['switch 2', null]], transformed,
    (r) => console.log(`     TEAM: ${JSON.stringify(r.battle.sides[0].pokemon.map((p) => `${p.species.id}:${p.transformed}:${p.moves.join('/')}`))}`));

  // TF6 — MIMIC BY a transformed mon FAILS (`source.transformed`), PP still deducted.
  //       The foe SWITCHES to a mon whose Body Slam the copied moveset does NOT contain, so
  //       the `already knows the move` fail cannot mask the transformed-user fail.
  await scenario('TF6 mimic by a transformed user fails', DITTO_1, BLISS_MIMIC_LAX,
    [['move 1', 'move 2'], ['move 2', 'switch 2'], ['move 2', 'move 1'], ['move 1', 'move 2']],
    (r) => r.per[3] && r.per[3].lines.some((l) => l.includes('Mimic')), dumpUser);

  // TF8 — the IDENT of a NICKNAME-LESS transformed mon: `pokemon.name` is
  //       `set.name || set.species`, fixed at construction, so every line still says Ditto.
  await scenario('TF8 nickname-less ident stays the base species', DITTO_NONICK, LAX_NONICK,
    [['move 1', 'move 2'], ['move 1', 'move 2']], transformed, dumpUser);

  // TF9 — the ABILITY: copied on transform, NO onStart re-fire in gen3, reverted to the SET
  //       ability on switch-out.
  await scenario('TF9 ability copied, no onStart, reverts', DITTO_1_GENGAR, GYARA,
    [['move 1', 'move 1'], ['switch 2', 'move 1'], ['switch 2', 'move 1']], transformed,
    (r) => {
      const u = r.battle.sides[0].pokemon.find((p) => p.species.id === 'ditto');
      console.log(`     DITTO after revert: ability=${u.ability} baseAbility=${u.baseAbility} transformed=${u.transformed}`);
      console.log(`     p2 boosts (Intimidate must NOT have re-fired on the copy): ${JSON.stringify(r.battle.sides[1].active[0].boosts)}`);
      console.log(`     p1 boosts: ${JSON.stringify(r.battle.sides[0].active[0].boosts)}`);
    });

  // TF7 — the PER-SIDE `|request|` bytes for the TF1 board (the surface poke-env consumes,
  //       and the one a filtering-hidden bug would live in).
  console.log('\n######## TF7 per-side |request| frames (TF1 board) ########');
  {
    const { getPlayerStreams: gps } = require(path.join(PS, 'dist/sim/battle-stream'));
    const stream = new BattleStream();
    const st = gps(stream);
    const p1 = [];
    (async () => { for await (const c of st.p1) for (const l of String(c).split('\n')) p1.push(l); })();
    (async () => { for await (const c of st.p2) void c; })();
    (async () => { for await (const c of st.omniscient) void c; })();
    st.omniscient.write(`>start {"formatid":"gen3customgame","seed":[1,2,4,3]}`);
    st.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: DITTO_1_GENGAR })}`);
    st.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: LAX_4 })}`);
    for (let i = 0; i < 16; i++) await tick();
    console.log(`  POST-CONSTRUCTION = "${String(stream.battle.prng.getSeed())}"`);
    st.omniscient.write('>p1 move 1');
    st.omniscient.write('>p2 move 4');
    for (let i = 0; i < 16; i++) await tick();
    p1.filter((l) => l.startsWith('|request|')).forEach((l, i) => console.log(`  p1 req${i}: ${l}`));
  }
}
main().then(() => process.exit(0)).catch((e) => { console.error(e); process.exit(1); });
