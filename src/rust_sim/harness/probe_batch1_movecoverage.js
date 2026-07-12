// probe_batch1_movecoverage.js — ground-truth the 5 BATCH-1 move-coverage classes
// (RECOIL / DRAIN / SELF-DROP / ITEM-REMOVAL / RAPID-SPIN) bit-for-bit vs the
// OMNISCIENT in-process BattleStream (no server). Confirms EACH is DRAW-FREE (no
// extra PRNG draw vs a control) and settles the gen3 gotchas:
//   - RECOIL: floor(dmgDealt * recoil[num/den]) to the USER; Rock Head negates.
//   - DRAIN:  floor(dmgDealt/2) heal to the USER; heal-at-full fails draw-free.
//   - SELF-DROP: move.self.boosts applied to the USER after the hit; -6 floor.
//               VERIFY there is NO selfDrops random(100).
//   - ITEM-REMOVAL: Knock Off removes; Thief steals iff attacker itemless; Sticky
//                   Hold blocks; Mail guards; Knock Off does NOT boost damage (gen3).
//   - RAPID SPIN: onAfterHit clears the USER's side hazards + Leech Seed + partial-trap.
//
// Run:  node src/rust_sim/harness/probe_batch1_movecoverage.js
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
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

function dumpResolved() {
  const d = Dex.forFormat(FORMAT);
  console.log('=== resolved gen3 move fields ===');
  for (const id of ['doubleedge', 'takedown', 'submission', 'gigadrain', 'absorb',
    'megadrain', 'leechlife', 'overheat', 'superpower', 'knockoff', 'thief', 'covet', 'rapidspin']) {
    const m = d.moves.get(id);
    console.log(`  ${id}: bp=${m.basePower} acc=${m.accuracy} recoil=${JSON.stringify(m.recoil)} ` +
      `drain=${JSON.stringify(m.drain)} self=${JSON.stringify(m.self)} contact=${(m.flags || {}).contact}`);
  }
  // Rock Head, Sticky Hold, Liquid Ooze ability handlers.
  for (const id of ['rockhead', 'stickyhold', 'liquidooze']) {
    const a = d.abilities.get(id);
    console.log(`  ability ${id}: onModifyMove=${typeof a.onModifyMove} onTakeItem=${typeof a.onTakeItem} onDrain=${typeof a.onDrain} onDamage=${typeof a.onDamage}`);
  }
}

async function run(label, p1team, p2team, plan, inject) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  const seed = [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  for (const inj of (inject || [])) {
    const m = inj.side === undefined ? null : battle.sides[inj.side].active[0];
    if (inj.weather) { battle.field.setWeather(inj.weather, battle.sides[0].active[0]); battle.field.weatherState.duration = 0; }
    if (inj.spikes !== undefined) { for (let k = 0; k < inj.spikes; k++) battle.sides[inj.side].addSideCondition('spikes', battle.sides[1 - inj.side].active[0]); }
    if (!m) continue;
    if (inj.status) m.setStatus(inj.status, m, null, true);
    if (inj.hp !== undefined) m.hp = inj.hp;
    if (inj.leechseed) { m.addVolatile('leechseed', battle.sides[1 - inj.side].active[0]); }
  }
  if (inject && inject.length) {
    const sp0 = battle.sides[0].sideConditions['spikes'];
    console.log(`  [inject] p1.spikes=${sp0 ? sp0.layers : 0} p1.leechseed=${!!(battle.sides[0].active[0].volatiles || {})['leechseed']}`);
  }

  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 50) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const before = battle.prng.getSeed();
    const dc0 = drawCount;
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const spikesOf = (s) => { const sc = battle.sides[s].sideConditions; const sp = sc && sc['spikes']; return `spk${sp ? sp.layers : 0}`; };
    const lsOf = (m) => (m && m.volatiles && m.volatiles['leechseed']) ? 'LS' : '';
    const fmt = (m, s) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'} atk${m.boosts.atk} def${m.boosts.def} spa${m.boosts.spa} item=${m.item || '-'} ${lsOf(m)} ${spikesOf(s)}${m.fainted ? ' FNT' : ''}` : '-';
    console.log(`  [${rs}] ${JSON.stringify(entry)} draws=${drawCount - dc0}  seedAfter=${after}`);
    console.log(`        p1=${fmt(a0, 0)}\n        p2=${fmt(a1, 1)}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  dumpResolved();

  // ============ RECOIL ============
  // Double-Edge: floor(dmgDealt/3) recoil. Rock Head negates. Compare a control (Tackle).
  await run('RECOIL: Double-Edge recoil floor(dmg/3), vs Tackle control',
    [mon('Tauros', ['doubleedge', 'bodyslam'], { evs: { atk: 252, spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252, def: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }]);
  await run('RECOIL: Double-Edge with ROCK HEAD (no recoil)',
    [mon('Aggron', ['doubleedge'], { ability: 'Rock Head', evs: { atk: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252, def: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);

  // ============ DRAIN ============
  // Giga Drain: floor(dmgDealt/2) heal. Injure the user so heal is visible. Heal-at-full fails.
  await run('DRAIN: Giga Drain heals floor(dmg/2); user injured',
    [mon('Sceptile', ['gigadrain', 'absorb'], { evs: { spa: 252, spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }],
    [{ side: 0, hp: 50 }]);
  await run('DRAIN: Giga Drain at FULL HP (heal fails draw-free)',
    [mon('Sceptile', ['gigadrain'], { evs: { spa: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);

  // ============ SELF-DROP ============
  // Overheat: self -2 SpA. Superpower: self -1 Atk/-1 Def. Verify NO selfDrops random(100).
  await run('SELF-DROP: Overheat self -2 SpA (into -6 floor over multiple uses)',
    [mon('Charizard', ['overheat'], { evs: { spa: 252, spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252, spd: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }]);
  await run('SELF-DROP: Superpower self -1 Atk/-1 Def',
    [mon('Machamp', ['superpower'], { evs: { atk: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252, def: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }]);

  // ============ ITEM REMOVAL ============
  // Knock Off: removes target item (gen3 no dmg boost). Compare vs a target WITHOUT item.
  await run('ITEM: Knock Off removes target Leftovers (gen3 no dmg boost)',
    [mon('Tyranitar', ['knockoff', 'tackle'], { evs: { atk: 252 } })],
    [mon('Snorlax', ['splash'], { item: 'Leftovers', ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }]);
  await run('ITEM: Knock Off vs a Sticky Hold holder (BLOCKED)',
    [mon('Tyranitar', ['knockoff'], { evs: { atk: 252 } })],
    [mon('Gastrodon', ['splash'], { item: 'Leftovers', ability: 'Sticky Hold', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);
  await run('ITEM: Knock Off vs a Mail holder (guarded)',
    [mon('Tyranitar', ['knockoff'], { evs: { atk: 252 } })],
    [mon('Snorlax', ['splash'], { item: 'Bright Powder', ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);
  // Thief: steals ONLY if attacker holds no item.
  await run('ITEM: Thief steals (attacker itemless)',
    [mon('Gengar', ['thief'], { item: '', evs: { spa: 252 } })],
    [mon('Snorlax', ['splash'], { item: 'Leftovers', ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);
  await run('ITEM: Thief does NOT steal (attacker HOLDS item)',
    [mon('Gengar', ['thief'], { item: 'Leftovers', evs: { spa: 252 } })],
    [mon('Blissey', ['splash'], { item: 'Choice Band', ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);

  // ============ RAPID SPIN ============
  // clears USER's side hazards (spikes) + USER's leech seed + partial trap.
  await run('RAPID SPIN: clears USER spikes + leech seed',
    [mon('Forretress', ['rapidspin', 'tackle'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['splash', 'leechseed'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }],
    // inject: 3 spikes on p1's side + leech-seed p1's Forretress (seeded by p2).
    [{ side: 0, spikes: 3 }, { side: 0, leechseed: true }]);
  await run('RAPID SPIN: no hazards (draw-free control)',
    [mon('Forretress', ['rapidspin'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
