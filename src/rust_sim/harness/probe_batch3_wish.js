// probe_batch3_wish.js — ground-truth WISH (C_DELAYED_HEAL) bit-for-bit vs the OMNISCIENT
// in-process BattleStream (no server). Wish is a slot-keyed DELAYED heal:
//   heal the RECIPIENT's maxhp/2 at the END of the turn AFTER cast (duration 2), slot-keyed
//   (survives faint/switch/phaze), double-Wish FAILS.
//
// The mod chain is the ONLY oracle. Probe:
//   - the residual ORDER (where the Wish heal fires vs Leftovers / status-DoT / Leech Seed / Curse)
//     -- THE RISK AREA (a wrong slot = a state/seed desync)
//   - slot-keyed survive-across-switch / faint / phaze
//   - double-Wish fail
//   - heal-at-full (silent? no line?)
//   - the exact protocol tokens (-heal ... [from] move: Wish)
//   - draw-freeness
//
// Run:  node src/rust_sim/harness/probe_batch3_wish.js
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
  console.log('=== resolved gen3 wish ===');
  const m = d.moves.get('wish');
  console.log(`  wish: cat=${m.category} bp=${m.basePower} acc=${m.accuracy} target=${m.target} ` +
    `flags=${JSON.stringify(m.flags)} slotCondition=${m.slotCondition} onTryHit=${typeof m.onTryHit} ` +
    `pseudoWeather=${m.pseudoWeather} sideCondition=${m.sideCondition} volatileStatus=${m.volatileStatus}`);
  // The wish condition (slotCondition).
  const c = d.conditions.get('wish');
  console.log(`  wish cond: duration=${c && c.duration} onResidualOrder=${c && c.onResidualOrder} ` +
    `onResidualSubOrder=${c && c.onResidualSubOrder} onStart=${c && typeof c.onStart} ` +
    `onResidual=${c && typeof c.onResidual} onEnd=${c && typeof c.onEnd}`);
  if (c && c.onStart) console.log(`  wish.onStart src: ${c.onStart.toString()}`);
  if (c && c.onResidual) console.log(`  wish.onResidual src: ${c.onResidual.toString()}`);
  if (c && c.onEnd) console.log(`  wish.onEnd src: ${c.onEnd.toString()}`);
  if (m.onTryHit) console.log(`  wish.onTryHit src: ${m.onTryHit.toString()}`);
  // Compare residual orders of the interacting effects.
  console.log('=== interacting residual orders ===');
  for (const id of ['leftovers']) { const it = d.items.get(id); console.log(`  item ${id}: onResidualOrder=${it.onResidualOrder} onResidualSubOrder=${it.onResidualSubOrder}`); }
  for (const id of ['leechseed', 'brn', 'psn', 'tox', 'curse']) { const cc = d.conditions.get(id); console.log(`  cond ${id}: onResidualOrder=${cc && cc.onResidualOrder} onResidualSubOrder=${cc && cc.onResidualSubOrder}`); }
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
    if (inj.weather) { battle.field.setWeather(inj.weather, battle.sides[0].active[0]); battle.field.weatherState.duration = 0; }
    if (!m) continue;
    if (inj.status) m.setStatus(inj.status, m, null, true);
    if (inj.hp !== undefined) m.hp = inj.hp;
    if (inj.benchHp !== undefined) { const b = battle.sides[inj.side].pokemon[1]; if (b) b.hp = inj.benchHp; }
  }

  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 40) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const dc0 = drawCount;
    const logLen0 = log.length;
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    // slotConditions on each side.
    const slot = (s) => { const sc = battle.sides[s].slotConditions; const out = []; for (const arr of sc) { for (const k of Object.keys(arr)) out.push(`${k}:d${arr[k].duration}`); } return out.join(','); };
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'}` : '-';
    console.log(`  [${rs}] ${JSON.stringify(entry)} draws=${drawCount - dc0}  seedAfter=${after}`);
    console.log(`        p1=${fmt(a0)} slot=[${slot(0)}]`);
    console.log(`        p2=${fmt(a1)} slot=[${slot(1)}]`);
    const newLines = log.slice(logLen0).filter((l) => /-heal|-damage|-start|-end|faint|move\|/.test(l));
    for (const l of newLines) console.log(`        LINE ${l}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  dumpResolved();

  // ---- Basic: Wish cast turn N heals at end of turn N+1 (maxhp/2). Cast on a low-HP mon. ----
  await run('WISH basic: cast N, heals maxhp/2 at end of N+1',
    [mon('Blissey', ['wish', 'splash'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }],
    { acts: [{ side: 0, hp: 100 }] });

  // ---- Residual ORDER: Wish + Leftovers + burn on the SAME wisher. Which fires first? ----
  await run('WISH residual ORDER: Wish + Leftovers + burn on the wisher',
    [mon('Blissey', ['wish', 'splash'], { ability: 'Natural Cure', item: 'Leftovers', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }],
    { acts: [{ side: 0, hp: 100, status: 'brn' }] });

  // ---- Residual ORDER: Wish + Leech Seed. Wisher is leech-seeded. ----
  await run('WISH residual ORDER: Wish + Leech Seed on the wisher',
    [mon('Blissey', ['wish', 'splash'], { ability: 'Natural Cure', item: 'Leftovers', evs: { hp: 252 } })],
    [mon('Meganium', ['leechseed', 'splash'], { ability: 'Overgrow', evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2' }, { p1: 'move 2', p2: 'move 2' }, { p1: 'move 2', p2: 'move 2' }],
    { acts: [{ side: 0, hp: 100 }] });

  // ---- Slot-keyed survive-across-switch: cast Wish, then SWITCH; the incoming mon gets healed. ----
  await run('WISH slot-keyed switch: cast, switch out, incoming mon healed maxhp/2',
    [mon('Blissey', ['wish', 'splash'], { ability: 'Natural Cure', evs: { hp: 252 } }),
     mon('Chansey', ['splash'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'switch 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }],
    { acts: [{ side: 0, benchHp: 100 }] });

  // ---- Double-Wish fail: cast Wish twice in a row; the 2nd fails (draw-free, existing untouched). ----
  await run('WISH double-Wish fail: 2nd Wish while pending FAILS',
    [mon('Blissey', ['wish', 'splash'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }],
    { acts: [{ side: 0, hp: 100 }] });

  // ---- Heal-at-full: cast Wish on a full-HP mon; it resolves SILENTLY (no -heal line?). ----
  await run('WISH heal-at-full: silent resolve on a full-HP mon',
    [mon('Blissey', ['wish', 'splash'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }]);

  // ---- Wisher FAINTS before resolve: the replacement (in the slot) gets healed. ----
  await run('WISH wisher faints: replacement in the slot gets healed',
    [mon('Blissey', ['wish', 'splash'], { ability: 'Natural Cure', evs: { hp: 252 } }),
     mon('Chansey', ['splash'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [mon('Snorlax', ['bodyslam'], { ability: 'Immunity', nature: 'Adamant', evs: { atk: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }],
    { acts: [{ side: 0, hp: 40, benchHp: 100 }] });

  // ---- Odd-maxhp rounding: cast Wish on a mon with an ODD maxhp; heal = floor(maxhp/2)? ----
  // Charizard maxhp 297 (odd?) — check the exact floor.
  await run('WISH odd-maxhp rounding: floor(maxhp/2)',
    [mon('Charizard', ['wish', 'splash'], { ability: 'Blaze', evs: { hp: 4 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }],
    { acts: [{ side: 0, hp: 50 }] });
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
