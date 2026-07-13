// probe_batch3_regression_rng.js — GROUND-TRUTH seeds for the MOVE-COVERAGE BATCH 3
// regression pins (MC18…MC29) in tests/regression_test.rs, captured from the OMNISCIENT
// in-process BattleStream (no server). Each scenario is a CONSTRUCTED gen3customgame board
// at a fixed raw seed; we print the per-decision post-turn PRNG seed + the key STATE the pin
// asserts. Copy the printed `seedAfter` verbatim into the pins.
//
//   MC18 — CURSE non-ghost self-boost {atk:+1, def:+1, spe:-1} (DRAW-FREE).
//   MC19 — CURSE ghost pays floor(maxhp/2) HP + lays the curse volatile on the FOE.
//   MC20 — CURSE residual chips the cursed foe floor(maxhp/4)/turn (DRAW-FREE).
//   MC21 — CURSE ghost re-curse into an already-cursed foe FAILS ([still]+-fail, DRAW-FREE).
//   MC22 — WISH heals floor(maxhp/2) at the end of the turn AFTER cast (the N+1 heal amount).
//   MC23 — the WISH RESIDUAL-ORDER SEED pin (CRITICAL): the Wish heal fires at ORDER 7,
//          BEFORE the Leftovers order-10 heal + the burn DoT. The resolve turn's seed +
//          the heal ORDER (Wish first) prove it.
//   MC24 — WISH double-Wish FAILS ([still], DRAW-FREE, the existing Wish untouched).
//   MC25 — WISH slot-keyed across a SWITCH: the incoming mon gets healed floor(ITS maxhp/2).
//   MC26 — BATON PASS boost transfer: +2 Atk passes to the entrant (DRAW-FREE copy).
//   MC27 — BATON PASS substitute transfer: the SUB HP passes to the entrant.
//   MC28 — BATON PASS leech-seed transfer: the seed passes to the entrant (the seeder keeps
//          draining the new mon).
//   MC29 — BATON PASS no-bench FAIL: a last-mon Baton Pass fails ([still]+-fail, DRAW-FREE).
//
// Run:  node src/rust_sim/harness/probe_batch3_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, p1team, p2team, plans, opts = {}) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  const seed = opts.seed || [11, 22, 33, 44];
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 10; i++) await tick();
  const b = stream.battle;
  for (const inj of (opts.inject || [])) {
    if (inj.side !== undefined) {
      const s = b.sides[inj.side];
      if (inj.status) s.active[0].setStatus(inj.status, s.active[0], null, true);
      if (inj.hp !== undefined) s.active[0].hp = inj.hp;
    }
  }
  console.log(`\n=== ${label} ===  initSeed=${b.prng.getSeed()}`);
  let i = 0, safety = 0;
  while (!b.ended && safety < 30) {
    safety++;
    const rs = b.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const force = rs === 'switch';
    let plan = plans[Math.min(i, plans.length - 1)]; i++;
    const logLen0 = log.length;
    if (plan.p1) { try { streams.omniscient.write(`>p1 ${plan.p1}`); } catch (e) {} }
    if (plan.p2) { try { streams.omniscient.write(`>p2 ${plan.p2}`); } catch (e) {} }
    for (let k = 0; k < 18; k++) await tick();
    const a0 = b.sides[0].active[0], a1 = b.sides[1].active[0];
    const cv = (a) => (a && a.volatiles && a.volatiles.curse ? 'curse' : '-');
    const sub = (a) => (a && a.volatiles && a.volatiles.substitute ? a.volatiles.substitute.hp : 0);
    const leech = (a) => (a && a.volatiles && a.volatiles.leechseed ? 'leech' : '-');
    const wish = (s) => { const sc = s.slotConditions[0]; return sc && sc.wish ? sc.wish.duration : 0; };
    const heals = log.slice(logLen0).filter((x) => x.includes('-heal') || (x.includes('-damage') && x.includes('Curse')));
    console.log(`  [${rs}] ${JSON.stringify(plan)} seedAfter=${b.prng.getSeed()}`);
    console.log(`     p1=${a0 ? a0.species.name : '-'} ${a0 ? a0.hp : 0}/${a0 ? a0.maxhp : 0} atk${a0 ? a0.boosts.atk : 0} def${a0 ? a0.boosts.def : 0} spe${a0 ? a0.boosts.spe : 0} ${cv(a0)} sub${sub(a0)} ${leech(a0)} wish${wish(b.sides[0])}`);
    console.log(`     p2=${a1 ? a1.species.name : '-'} ${a1 ? a1.hp : 0}/${a1 ? a1.maxhp : 0} atk${a1 ? a1.boosts.atk : 0} def${a1 ? a1.boosts.def : 0} spe${a1 ? a1.boosts.spe : 0} ${cv(a1)} sub${sub(a1)} ${leech(a1)} wish${wish(b.sides[1])}`);
    if (heals.length) console.log(`     residuals: ${heals.join(' || ')}`);
    if (plan.stop) break;
  }
  console.log(`  ended=${b.ended} winner=${b.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, o = {}) {
  return { species, item: o.item || '', ability: o.ability || 'No Ability', moves, evs: { ...EV0, ...(o.evs || {}) }, ivs: IV31, nature: o.nature || 'Serious', level: o.level || 100, gender: 'N' };
}

async function main() {
  // MC18: CURSE non-ghost self-boost {atk:+1, def:+1, spe:-1} (draw-free). Snorlax curses.
  await run('MC18 CURSE non-ghost self-boost (draw-free)',
    [mon('Snorlax', ['curse', 'bodyslam'], { ability: 'Immunity', evs: { hp: 252, atk: 252 } })],
    [mon('Blissey', ['splash'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);

  // MC19: CURSE ghost pays floor(maxhp/2) + lays the curse volatile on the foe. Gengar curses.
  await run('MC19 CURSE ghost HP-cost + lay volatile',
    [mon('Gengar', ['curse', 'shadowball'], { ability: 'Levitate', evs: { spa: 252, spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);

  // MC20: CURSE residual chips the cursed foe floor(maxhp/4)/turn (draw-free). Turn 2 = the chip.
  await run('MC20 CURSE residual chips the foe maxhp/4 (draw-free)',
    [mon('Gengar', ['curse', 'shadowball'], { ability: 'Levitate', evs: { spa: 252, spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }]);

  // MC21: CURSE ghost re-curse into an already-cursed foe FAILS (draw-free). Turn 2 = the fail.
  await run('MC21 CURSE re-curse FAIL (draw-free)',
    [mon('Gengar', ['curse', 'shadowball'], { ability: 'Levitate', evs: { spa: 252, spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }]);

  // MC22: WISH heals floor(maxhp/2) at end of the turn AFTER cast. Blissey injected to hp 100.
  await run('MC22 WISH heals maxhp/2 at N+1',
    [mon('Blissey', ['wish', 'splash'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }],
    { inject: [{ side: 0, hp: 100 }] });

  // MC23 (CRITICAL): the WISH RESIDUAL-ORDER pin. Blissey holds Leftovers + is burned;
  // Wishes then Splashes. On the resolve turn the Wish heal (order 7) precedes the Leftovers
  // heal (order 10) + the burn DoT — the -heal ORDER + the seed prove it.
  await run('MC23 WISH residual ORDER (order 7, before Leftovers + burn)',
    [mon('Blissey', ['wish', 'splash'], { ability: 'Natural Cure', item: 'leftovers', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }],
    { inject: [{ side: 0, status: 'brn', hp: 100 }] });

  // MC24: WISH double-Wish FAILS (draw-free). Turn 2 = the 2nd Wish (fails, existing untouched).
  await run('MC24 WISH double-Wish FAIL (draw-free)',
    [mon('Blissey', ['wish', 'splash'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }],
    { inject: [{ side: 0, hp: 100 }] });

  // MC25: WISH slot-keyed across a switch: the incoming Chansey gets healed floor(ITS maxhp/2).
  await run('MC25 WISH slot-keyed across a switch',
    [mon('Blissey', ['wish', 'splash'], { ability: 'Natural Cure', evs: { hp: 252 } }),
     mon('Chansey', ['splash'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'switch 2', p2: 'move 1' }],
    { inject: [{ side: 0, hp: 100 }] });

  // MC26: BATON PASS boost transfer. Jolteon Swords Dances (+2 Atk), Baton Passes to Snorlax.
  await run('MC26 BATON PASS boost transfer (+2 Atk)',
    [mon('Jolteon', ['swordsdance', 'batonpass'], { ability: 'Volt Absorb', evs: { spe: 252, atk: 252 } }),
     mon('Snorlax', ['bodyslam'], { ability: 'Immunity', evs: { atk: 252, hp: 252 } })],
    [mon('Blissey', ['splash'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }, { p1: 'switch 2', p2: 'move 1' }]);

  // MC27: BATON PASS substitute transfer. Jolteon Subs, Baton Passes; the sub HP passes.
  await run('MC27 BATON PASS substitute transfer',
    [mon('Jolteon', ['substitute', 'batonpass'], { ability: 'Volt Absorb', evs: { spe: 252, hp: 252 } }),
     mon('Snorlax', ['bodyslam'], { ability: 'Immunity', evs: { atk: 252, hp: 252 } })],
    [mon('Blissey', ['splash'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }, { p1: 'switch 2', p2: 'move 1' }]);

  // MC28: BATON PASS leech-seed transfer. Meganium seeds Jolteon; Jolteon Baton Passes; seed passes.
  await run('MC28 BATON PASS leech-seed transfer',
    [mon('Jolteon', ['agility', 'batonpass'], { ability: 'Volt Absorb', evs: { spe: 252, hp: 252 } }),
     mon('Snorlax', ['bodyslam'], { ability: 'Immunity', evs: { atk: 252, hp: 252 } })],
    [mon('Meganium', ['leechseed', 'splash'], { ability: 'Overgrow', evs: { hp: 252, def: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 2' }, { p1: 'switch 2', p2: 'move 2' }]);

  // MC29: BATON PASS no-bench FAIL. Jolteon is the last mon → Baton Pass fails (draw-free).
  await run('MC29 BATON PASS no-bench FAIL (draw-free)',
    [mon('Jolteon', ['batonpass', 'thunderbolt'], { ability: 'Volt Absorb', evs: { spe: 252, spa: 252 } })],
    [mon('Blissey', ['splash'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
