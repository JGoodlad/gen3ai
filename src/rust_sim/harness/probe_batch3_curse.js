// probe_batch3_curse.js — ground-truth CURSE (C_CURSE) bit-for-bit vs the OMNISCIENT
// in-process BattleStream (no server). Curse is type-conditional at onModifyMove:
//   GHOST user    — pay 1/2 max HP, lay the `curse` volatile on the FOE (a 1/4-max-HP/turn residual)
//   NON-GHOST user — self +1 Atk / +1 Def / -1 Spe (draw-free, setup shape)
//
// The mod chain is the ONLY oracle. Probe the exact:
//   - HP-cost rounding (floor(maxhp/2)? clamp?)
//   - the curse residual order (vs Leftovers / status DoT / Leech) + amount (floor(maxhp/4)?)
//   - the Ghost-into-a-curse-immune / already-cursed case
//   - the -Spe self-drop + the cached-speed timing (does it flip the first mover next turn?)
//   - draw-freeness (both branches) vs a control
//
// Run:  node src/rust_sim/harness/probe_batch3_curse.js
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
  console.log('=== resolved gen3 curse ===');
  const m = d.moves.get('curse');
  console.log(`  curse: cat=${m.category} bp=${m.basePower} acc=${m.accuracy} target=${m.target} ` +
    `flags=${JSON.stringify(m.flags)} volatileStatus=${m.volatileStatus} ` +
    `onModifyMove=${typeof m.onModifyMove} onHit=${typeof m.onHit} priority=${m.priority}`);
  console.log(`  curse.onModifyMove src: ${m.onModifyMove ? m.onModifyMove.toString() : '-'}`);
  const c = d.conditions.get('curse');
  console.log(`  curse cond: onResidualOrder=${c && c.onResidualOrder} onResidualSubOrder=${c && c.onResidualSubOrder} ` +
    `onStart=${c && typeof c.onStart} onResidual=${c && typeof c.onResidual}`);
  if (c && c.onResidual) console.log(`  curse.onResidual src: ${c.onResidual.toString()}`);
  if (c && c.onStart) console.log(`  curse.onStart src: ${c.onStart.toString()}`);
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
    if (inj.item !== undefined) m.item = inj.item;
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
    const vol = (m) => m ? Object.keys(m.volatiles).join(',') : '-';
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'} atk${m.boosts.atk} def${m.boosts.def} spe${m.boosts.spe} spd=${m.getStat('spe')} vols=[${vol(m)}]` : '-';
    console.log(`  [${rs}] ${JSON.stringify(entry)} draws=${drawCount - dc0}  seedAfter=${after}`);
    console.log(`        p1=${fmt(a0)}`);
    console.log(`        p2=${fmt(a1)}`);
    const newLines = log.slice(logLen0).filter((l) => /-start|-damage|-heal|-boost|-unboost|-fail|-immune|cant|-activate|move\|/.test(l));
    for (const l of newLines) console.log(`        LINE ${l}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  dumpResolved();

  // ---- NON-GHOST branch: self +1 Atk / +1 Def / -1 Spe. Draw-free? ----
  await run('CURSE non-ghost: Snorlax self +1Atk/+1Def/-1Spe',
    [mon('Snorlax', ['curse', 'bodyslam'], { ability: 'Immunity', evs: { hp: 252 } })],
    [mon('Blissey', ['softboiled'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }]);

  // Control: no curse (self-boost via Bulk Up-like) to compare draws.
  await run('CONTROL: Snorlax splash-only (draw baseline)',
    [mon('Snorlax', ['bodyslam', 'rest'], { ability: 'Immunity', evs: { hp: 252 } })],
    [mon('Blissey', ['softboiled'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' }]);

  // Non-ghost -Spe cached-speed timing: a fast non-ghost Curses; does the -1 Spe flip the
  // first mover NEXT turn? Use two mons close in speed.
  await run('CURSE non-ghost -Spe first-mover: does -1 Spe drop below foe next turn?',
    [mon('Miltank', ['curse', 'bodyslam'], { ability: 'Thick Fat', nature: 'Jolly', evs: { spe: 252 } })],
    [mon('Snorlax', ['bodyslam'], { ability: 'Immunity', nature: 'Careful', evs: { spe: 100 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }]);

  // ---- GHOST branch: pay 1/2 HP, lay curse on the foe (1/4-HP/turn residual). ----
  await run('CURSE ghost: Gengar pays 1/2 HP, curses the foe',
    [mon('Gengar', ['curse', 'shadowball'], { ability: 'Levitate', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }]);

  // Ghost curse residual ORDER vs Leftovers + burn + leech on the CURSED foe. Cursed foe holds
  // Leftovers + is burned + is leech-seeded → what order do the residuals fire?
  await run('CURSE ghost residual ORDER: cursed foe w/ Leftovers + burn',
    [mon('Gengar', ['curse', 'willowisp'], { ability: 'Levitate', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', item: 'Leftovers', evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }]);

  // Ghost curse into a mon at LOW HP (does the residual clamp / KO?)
  await run('CURSE ghost residual KO: cursed foe at low HP faints from the residual',
    [mon('Gengar', ['curse', 'shadowball'], { ability: 'Levitate', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }],
    { acts: [{ side: 1, hp: 100 }] });

  // Ghost curse: user at HP <= 1/2 max — pays HP anyway? (does it faint the user?)
  await run('CURSE ghost self-KO?: user at low HP pays 1/2 max',
    [mon('Gengar', ['curse', 'shadowball'], { ability: 'Levitate', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    { acts: [{ side: 0, hp: 50 }] });

  // Ghost curse: already-cursed foe (re-curse fails?) — Gengar curses twice.
  await run('CURSE ghost re-curse: already-cursed foe',
    [mon('Gengar', ['curse', 'shadowball'], { ability: 'Levitate', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }]);

  // Ghost curse into a GHOST foe (is a Ghost immune to the curse volatile?)
  await run('CURSE ghost into a GHOST foe (curse-immune?)',
    [mon('Gengar', ['curse', 'shadowball'], { ability: 'Levitate', evs: { hp: 252 } })],
    [mon('Misdreavus', ['splash'], { ability: 'Levitate', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }]);

  // Ghost curse: does the foe's SUBSTITUTE block the curse volatile?
  await run('CURSE ghost into a foe with a SUBSTITUTE',
    [mon('Gengar', ['curse', 'shadowball'], { ability: 'Levitate', evs: { hp: 252 } })],
    [mon('Snorlax', ['substitute', 'splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2' }, { p1: 'move 2', p2: 'move 2' }]);

  // Ghost curse odd maxhp rounding: pick a mon with an odd maxhp and check floor(maxhp/2)
  // for the HP cost and floor(maxhp/4) for the residual. Gengar maxhp / cursed-foe maxhp.
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
