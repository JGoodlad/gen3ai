// probe_batch3_batonpass.js — ground-truth BATON PASS (C_BATONPASS) bit-for-bit vs the
// OMNISCIENT in-process BattleStream (no server). Baton Pass is a SWITCH that TRANSFERS the
// user's boost stages + pass-able volatiles to the incoming mon.
//
// The mod chain is the ONLY oracle. Probe:
//   - WHICH volatiles pass in gen3 (and which don't) — the pass-set
//   - the boost-stage transfer
//   - the switch draw model (the switch draws normally; the pass is draw-free)
//   - the forced-switch REQUEST it triggers (a `switch` request for the SAME side)
//   - interactions: passing a Substitute / a Leech Seed / confusion / the perish counter
//
// Run:  node src/rust_sim/harness/probe_batch3_batonpass.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: opts.gender || 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

function dumpResolved() {
  const d = Dex.forFormat(FORMAT);
  console.log('=== resolved gen3 batonpass ===');
  const m = d.moves.get('batonpass');
  console.log(`  batonpass: cat=${m.category} bp=${m.basePower} acc=${m.accuracy} target=${m.target} ` +
    `flags=${JSON.stringify(m.flags)} selfSwitch=${m.selfSwitch} priority=${m.priority} onHit=${typeof m.onHit}`);
  if (m.onHit) console.log(`  batonpass.onHit src: ${m.onHit.toString()}`);
  // The volatile pass-set is decided by each volatile's copyable flags in copyVolatileFrom.
  // Dump copyable-ness of interesting volatiles.
  console.log('=== volatile copy flags (noCopy / onCopy) ===');
  for (const id of ['substitute', 'leechseed', 'confusion', 'perishsong', 'curse', 'lockon',
    'focusenergy', 'ingrain', 'aquaring', 'magnetrise', 'stockpile', 'taunt', 'disable',
    'encore', 'attract', 'torment', 'yawn', 'nightmare', 'protect', 'flashfire', 'foresight',
    'lockedmove', 'partiallytrapped', 'gastroacid', 'embargo', 'healblock', 'telekinesis']) {
    const c = d.conditions.get(id);
    if (!c) continue;
    console.log(`  ${id}: noCopy=${c.noCopy} onCopy=${typeof c.onCopy} ` +
      `onStart=${typeof c.onStart} duration=${c.duration}`);
  }
}

async function run(label, p1team, p2team, plan, inject) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  const seed = (inject && inject.seed) || [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  for (const inj of ((inject && inject.acts) || [])) {
    const m = inj.side === undefined ? null : battle.sides[inj.side].active[0];
    if (!m) continue;
    if (inj.status) m.setStatus(inj.status, m, null, true);
    if (inj.hp !== undefined) m.hp = inj.hp;
    if (inj.boosts) m.boostBy(inj.boosts);
    if (inj.volatile) m.addVolatile(inj.volatile, m);
  }

  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 30) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const dc0 = drawCount;
    const logLen0 = log.length;
    // Which side is forced to switch?
    const force = [0, 1].map((s) => { const r = battle.sides[s].activeRequest; return r && r.forceSwitch && r.forceSwitch[0]; });
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    if (entry.p1) { try { streams.omniscient.write(`>p1 ${entry.p1}`); } catch (e) {} }
    if (entry.p2) { try { streams.omniscient.write(`>p2 ${entry.p2}`); } catch (e) {} }
    for (let k = 0; k < 18; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const vol = (m) => m ? Object.keys(m.volatiles).filter((v) => v !== 'undefined').join(',') : '-';
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'} atk${m.boosts.atk} def${m.boosts.def} spa${m.boosts.spa} spe${m.boosts.spe} vols=[${vol(m)}] sub=${m.volatiles.substitute ? m.volatiles.substitute.hp : '-'}` : '-';
    console.log(`  [${rs}] force=${JSON.stringify(force)} ${JSON.stringify(entry)} draws=${drawCount - dc0}  seedAfter=${after}`);
    console.log(`        p1=${fmt(a0)}`);
    console.log(`        p2=${fmt(a1)}`);
    const newLines = log.slice(logLen0).filter((l) => /-start|-end|-damage|-heal|-boost|-unboost|switch|drag|-activate|-fail|faint|move\|/.test(l));
    for (const l of newLines) console.log(`        LINE ${l}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  dumpResolved();

  // ---- Boost transfer: a Ninjask +2 Atk Baton Passes to a bench mon; the +2 transfers. ----
  await run('BP boost transfer: +2 Atk passes to the incoming mon',
    [mon('Jolteon', ['batonpass', 'swordsdance', 'thunderbolt'], { ability: 'Volt Absorb', evs: { spe: 252 } }),
     mon('Snorlax', ['bodyslam'], { ability: 'Immunity', evs: { hp: 252 } })],
    [mon('Blissey', ['softboiled'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    // SD to +2, then Baton Pass -> switch to Snorlax (which should keep +2 Atk).
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }, { p1: 'switch 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }]);

  // ---- Substitute pass: a mon with a Substitute Baton Passes; the incoming mon KEEPS the sub. ----
  await run('BP substitute pass: the sub transfers to the incoming mon',
    [mon('Jolteon', ['batonpass', 'substitute', 'thunderbolt'], { ability: 'Volt Absorb', evs: { hp: 252, spe: 252 } }),
     mon('Snorlax', ['bodyslam'], { ability: 'Immunity', evs: { hp: 252 } })],
    [mon('Blissey', ['pound'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }, { p1: 'switch 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }]);

  // ---- Leech Seed pass: does a passed mon KEEP leech seed? (Leech Seed is on the PASSER — it's
  //   a volatile the OPPONENT put on it. Does it transfer?) ----
  await run('BP leech seed: does the leech-seeded passer transfer the seed?',
    [mon('Jolteon', ['batonpass', 'thunderbolt'], { ability: 'Volt Absorb', evs: { hp: 252, spe: 252 } }),
     mon('Snorlax', ['bodyslam'], { ability: 'Immunity', evs: { hp: 252 } })],
    [mon('Meganium', ['leechseed', 'splash'], { ability: 'Overgrow', evs: { hp: 252 } })],
    // Meganium seeds Jolteon, then Jolteon Baton Passes to Snorlax; is Snorlax seeded?
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2' }, { p1: 'switch 2', p2: 'move 2' }, { p1: 'move 1', p2: 'move 2' }]);

  // ---- Confusion pass: does a confused mon transfer confusion? ----
  await run('BP confusion: does confusion transfer?',
    [mon('Jolteon', ['batonpass', 'thunderbolt'], { ability: 'Volt Absorb', evs: { hp: 252, spe: 252 } }),
     mon('Snorlax', ['bodyslam'], { ability: 'Immunity', evs: { hp: 252 } })],
    [mon('Blissey', ['pound'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }],
    { acts: [{ side: 0, volatile: 'confusion' }] });

  // ---- Perish counter pass: does the perish counter transfer? (No Perish Song modeled — inject.) ----
  await run('BP perish: does the perishsong volatile transfer?',
    [mon('Jolteon', ['batonpass', 'thunderbolt'], { ability: 'Volt Absorb', evs: { hp: 252, spe: 252 } }),
     mon('Snorlax', ['bodyslam'], { ability: 'Immunity', evs: { hp: 252 } })],
    [mon('Blissey', ['pound'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }],
    { acts: [{ side: 0, volatile: 'perishsong' }] });

  // ---- BP with NO eligible bench (last mon): fails? ----
  await run('BP no bench: last mon Baton Passes (fails?)',
    [mon('Jolteon', ['batonpass', 'thunderbolt'], { ability: 'Volt Absorb', evs: { hp: 252, spe: 252 } })],
    [mon('Blissey', ['pound'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);

  // ---- BP forced-switch REQUEST: what does the request state look like right after BP resolves? ----
  await run('BP negative boosts + status pass: -Spe + status stays with the mon (status NOT passed)',
    [mon('Jolteon', ['batonpass', 'thunderbolt'], { ability: 'Volt Absorb', evs: { hp: 252, spe: 252 } }),
     mon('Snorlax', ['bodyslam'], { ability: 'Immunity', evs: { hp: 252 } })],
    [mon('Blissey', ['pound'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }],
    { acts: [{ side: 0, status: 'par', boosts: { spe: -2 } }] });
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
