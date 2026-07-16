// probe_snatch_regression_rng.js — GROUND TRUTH for the deterministic SNATCH regression
// pins (MC100-MC104, `gen3_snatch_v1`). Constructs each pin's EXACT scenario, drives the
// OMNISCIENT in-process gen3 BattleStream (no server) at a fixed seed, and prints the packed
// teams + the POST-CONSTRUCTION initSeed + the per-decision post-seed + the asserted state,
// so `tests/regression_test.rs` copies the constants verbatim.
//
// Run:  node src/rust_sim/harness/probe_snatch_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const F = 'gen3customgame';
const IV = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const E0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
const mon = (s, m, o = {}) => ({
  species: s, item: o.item || '', ability: o.ability || 'No Ability', moves: m,
  evs: { ...E0, ...(o.evs || {}) }, ivs: IV, nature: 'Serious', level: 100, gender: 'N',
});
const tick = () => new Promise((r) => setTimeout(r, 0));

async function run(label, p1, p2, plan, inject) {
  const st = new BattleStream(), ss = getPlayerStreams(st);
  const log = [];
  (async () => { for await (const c of ss.omniscient) { for (const l of c.split('\n')) if (l) log.push(l); } })();
  ss.omniscient.write(`>start {"formatid":"${F}","seed":[7,11,13,17]}`);
  ss.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1) })}`);
  ss.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const b = st.battle;
  if (inject) for (const j of inject) {
    const m = j.slot === undefined ? b.sides[j.side].active[0] : b.sides[j.side].pokemon[j.slot];
    if (j.hp !== undefined) m.hp = j.hp;
    if (j.status) m.setStatus(j.status, m, null, true);
  }
  console.log(`\n=== ${label} ===`);
  console.log(`  P1_PACKED = ${JSON.stringify(Teams.pack(p1))}`);
  console.log(`  P2_PACKED = ${JSON.stringify(Teams.pack(p2))}`);
  console.log(`  initSeed  = ${b.prng.getSeed()}`);
  let draws = [];
  const rng = b.prng.rng, rn = rng.next.bind(rng);
  rng.next = (...a) => { const v = rn(...a); draws.push(v); return v; };
  let i = 0, safety = 0;
  while (!b.ended && safety < 20) {
    safety++;
    const rs = b.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const e = plan[Math.min(i, plan.length - 1)]; i++;
    draws = [];
    if (e.p1) { try { ss.omniscient.write(`>p1 ${e.p1}`); } catch (x) {} }
    if (e.p2) { try { ss.omniscient.write(`>p2 ${e.p2}`); } catch (x) {} }
    for (let k = 0; k < 20; k++) await tick();
    const a0 = b.sides[0].active[0], a1 = b.sides[1].active[0];
    const bs = (m) => Object.entries(m.boosts).filter(([, v]) => v).map(([k, v]) => `${k}${v > 0 ? '+' : ''}${v}`).join(',') || '·';
    const pp = (m) => m.moveSlots.map((s) => `${s.id}:${s.pp}`).join(',');
    console.log(`  dec${i - 1} [${rs}] p1=${e.p1} p2=${e.p2} draws=${draws.length} -> seedAfter=${b.prng.getSeed()}`);
    if (a0) console.log(`        p1: ${a0.species.name} ${a0.hp}/${a0.maxhp} ${a0.status || '-'} b={${bs(a0)}} pp={${pp(a0)}}`);
    if (a1) console.log(`        p2: ${a1.species.name} ${a1.hp}/${a1.maxhp} ${a1.status || '-'} b={${bs(a1)}} pp={${pp(a1)}}`);
    if (e.stop) break;
  }
  ss.omniscient.destroy();
}

async function main() {
  // MC100 — FAST snatcher steals Swords Dance: Jolteon(130) > Skarmory(70). The snatcher
  //   gets +2 atk; the foe's SD PP drops (48->47), the snatcher's Snatch PP drops (16->15).
  await run('MC100 fast steal SwordsDance (Jolteon vs Skarmory)',
    [mon('Jolteon', ['snatch', 'splash'])],
    [mon('Skarmory', ['swordsdance', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // MC101 — SLOW snatcher steals SD (Snorlax(30) < Skarmory(70)): priority +4 guarantees the
  //   volatile is up before the foe's move → SAME post-seed as MC100 (the +4 proof). A naive
  //   "reactive after the foe" model would diverge on the draw order.
  await run('MC101 slow steal SwordsDance (Snorlax vs Skarmory)',
    [mon('Snorlax', ['snatch', 'splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['swordsdance', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // MC102 — steal Rest (the DRAW-COUNT teeth): the stolen Rest's sleep random(2,6) fires in
  //   the SNATCHER's context (draws=2 vs 1). The snatcher goes slp + full-heals.
  await run('MC102 steal Rest (Umbreon vs Snorlax)',
    [mon('Umbreon', ['snatch', 'splash'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['rest', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    [{ side: 0, hp: 100 }]);

  // MC103 — Thunder Wave is NOT snatchable (no flags.snatch) → passes through; the snatcher
  //   is paralyzed normally, NO -activate, NO steal.
  await run('MC103 Thunder Wave not stolen (Umbreon vs Jolteon)',
    [mon('Umbreon', ['snatch', 'splash'], { evs: { hp: 252 } })],
    [mon('Jolteon', ['thunderwave', 'splash'])],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // MC104 — the SNATCH MIRROR residual-duration tie (the CRUX): two EQUAL-speed Umbreon both
  //   cast Snatch → both `snatch` volatiles tie at the residual → ONE extra tie-shuffle draw
  //   (8 total vs the both-Splash control's 7). Neither steals (Snatch is not snatchable).
  await run('MC104 snatch MIRROR residual tie (Umbreon vs Umbreon)',
    [mon('Umbreon', ['snatch', 'splash'], { evs: { hp: 252 } })],
    [mon('Umbreon', ['snatch', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // MC104b — the CONTROL for MC104: both Splash (no snatch volatile) → NO residual tie-shuffle
  //   (7 draws). Proves the mirror's extra draw is snatch-attributable.
  await run('MC104b control both Splash (Umbreon vs Umbreon)',
    [mon('Umbreon', ['splash', 'snatch'], { evs: { hp: 252 } })],
    [mon('Umbreon', ['splash', 'snatch'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
