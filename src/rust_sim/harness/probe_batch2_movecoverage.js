// probe_batch2_movecoverage.js — ground-truth the 4 BATCH-2 move-coverage classes
// (STATUS-CURE / WEATHER-SET / STAT-DROP-MOVE / SCREENS) bit-for-bit vs the
// OMNISCIENT in-process BattleStream (no server). Each is a category-Status move.
//
//  1. STATUS-CURE: Refresh (self par/psn/brn), Heal Bell + Aromatherapy (whole-team
//     major-status cure incl. bench). Draw-free onHit. Probe: team iteration + the
//     Soundproof interaction for Heal Bell (gen3 sound?) + the -curestatus lines.
//  2. WEATHER-SET: Rain Dance (Rain), Sunny Day (Sun) — a 5-turn TIMED weather.
//     THE non-trivial class. Probe: does setWeather fire the eachEvent('WeatherChange')
//     speed-tie shuffle? the 5-turn upkeep tick; overwrite vs fail when active.
//  3. STAT-DROP MOVES: Screech (-2 Def), Charm (-2 Atk), Metal Sound (-2 SpD),
//     Feather Dance (-2 Atk), Tickle (-1 Atk/-1 Def), Fake Tears (-2 SpD). accuracy
//     draw + boost() (draw-free, +-6 clamp, Clear Body/White Smoke/Keen Eye/Hyper Cutter).
//  4. SCREENS: Light Screen, Reflect (5 turns, halve special/physical to the side).
//     Draw-free. The -sidestart/-sideend lines + upkeep countdown.
//
// Run:  node src/rust_sim/harness/probe_batch2_movecoverage.js
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
  console.log('=== resolved gen3 move fields ===');
  for (const id of ['refresh', 'healbell', 'aromatherapy', 'raindance', 'sunnyday',
    'screech', 'charm', 'metalsound', 'featherdance', 'tickle', 'faketears',
    'lightscreen', 'reflect']) {
    const m = d.moves.get(id);
    console.log(`  ${id}: cat=${m.category} bp=${m.basePower} acc=${m.accuracy} target=${m.target} ` +
      `flags=${JSON.stringify(m.flags)} status=${m.status} boosts=${JSON.stringify(m.boosts)} ` +
      `weather=${m.weather} sideCondition=${m.sideCondition} pseudoWeather=${m.pseudoWeather} ` +
      `onHit=${typeof m.onHit} onTryHit=${typeof m.onTryHit} volatileStatus=${m.volatileStatus}`);
  }
  // Weather / side condition resolved durations.
  console.log('=== resolved conditions ===');
  for (const id of ['raindance', 'sunnyday', 'lightscreen', 'reflect']) {
    const c = d.conditions.get(id);
    console.log(`  cond ${id}: duration=${c ? c.duration : '?'} durationCallback=${c && typeof c.durationCallback}`);
  }
  // Soundproof: does it block Heal Bell (a sound move)?
  console.log('=== soundproof / sound flags ===');
  for (const id of ['healbell', 'aromatherapy']) {
    const m = d.moves.get(id);
    console.log(`  ${id}: flags.sound=${(m.flags || {}).sound}`);
  }
}

async function run(label, p1team, p2team, plan, inject) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  const seed = [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  for (const inj of (inject || [])) {
    const m = inj.side === undefined ? null : battle.sides[inj.side].active[0];
    if (inj.weather) { battle.field.setWeather(inj.weather, battle.sides[0].active[0]); battle.field.weatherState.duration = 0; }
    if (!m) continue;
    if (inj.status) m.setStatus(inj.status, m, null, true);
    // bench status (cure test): status the second team member
    if (inj.benchStatus) { const b = battle.sides[inj.side].pokemon[1]; if (b) b.setStatus(inj.benchStatus, b, null, true); }
    if (inj.hp !== undefined) m.hp = inj.hp;
  }

  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 60) {
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
    const wf = battle.field.weatherState;
    const scr = (s) => { const sc = battle.sides[s].sideConditions; return Object.keys(sc).map((k) => `${k}:${sc[k].duration}`).join(','); };
    const benchStat = (s) => battle.sides[s].pokemon.map((p) => `${p.species.name.slice(0, 3)}=${p.status || '-'}`).join(' ');
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'} atk${m.boosts.atk} def${m.boosts.def} spd${m.boosts.spd}` : '-';
    console.log(`  [${rs}] ${JSON.stringify(entry)} draws=${drawCount - dc0}  seedAfter=${after}`);
    console.log(`        weather=${battle.field.weather || '-'} dur=${wf ? wf.duration : '-'}  p1scr=[${scr(0)}] p2scr=[${scr(1)}]`);
    console.log(`        p1=${fmt(a0)} | ${benchStat(0)}`);
    console.log(`        p2=${fmt(a1)} | ${benchStat(1)}`);
    // Print the new protocol lines this decision (curestatus/weather/sidestart/sideend/unboost).
    const newLines = log.slice(logLen0).filter((l) => /-curestatus|-weather|-sidestart|-sideend|-unboost|-fail|cant/.test(l));
    for (const l of newLines) console.log(`        LINE ${l}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  dumpResolved();

  // ============ STATUS-CURE ============
  // Refresh: cures the USER's par/psn/brn. Inject par on the user.
  await run('CURE: Refresh cures self paralysis',
    [mon('Vaporeon', ['refresh', 'splash'], { ability: 'Water Absorb', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    [{ side: 0, status: 'par' }]);
  // Heal Bell: cures the WHOLE team (active + bench). Inject status on active + bench.
  await run('CURE: Heal Bell cures whole team (active tox + bench par)',
    [mon('Miltank', ['healbell', 'splash'], { ability: 'Thick Fat', evs: { hp: 252 } }),
     mon('Blissey', ['splash'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    [{ side: 0, status: 'tox', benchStatus: 'par' }]);
  // Heal Bell into a Soundproof teammate? (Actually Soundproof is a FOE thing here — the
  // relevant question is whether the sound flag matters for the CURE at all — it's self-side.)
  // Aromatherapy: same as Heal Bell (whole-team cure, NOT a sound move — probe the flag).
  await run('CURE: Aromatherapy cures whole team',
    [mon('Vileplume', ['aromatherapy', 'splash'], { ability: 'Chlorophyll', evs: { hp: 252 } }),
     mon('Snorlax', ['splash'], { ability: 'Thick Fat', evs: { hp: 252 } })],
    [mon('Regice', ['splash'], { ability: 'Clear Body', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    [{ side: 0, status: 'brn', benchStatus: 'slp' }]);
  // Heal Bell with a Soundproof mon ON THE SAME SIDE — does gen3 Soundproof block the cure
  // of its own holder? (gen5+ it did; gen3 healbell has NO sound flag → irrelevant.)
  await run('CURE: Heal Bell with a Soundproof teammate (bench)',
    [mon('Miltank', ['healbell', 'splash'], { ability: 'Thick Fat', evs: { hp: 252 } }),
     mon('Electrode', ['splash'], { ability: 'Soundproof', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    [{ side: 0, status: 'par', benchStatus: 'par' }]);

  // ============ WEATHER-SET ============
  // Rain Dance: set Rain for 5 turns. Does setWeather draw the eachEvent WeatherChange
  // shuffle? Use a SPEED TIE (both Snorlax) so the tie-shuffle would draw if it fires.
  await run('WEATHER: Rain Dance sets Rain 5 turns (SPEED TIE — does WeatherChange shuffle draw?)',
    [mon('Snorlax', ['raindance', 'splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }]);
  // Control: distinct speeds — no tie-shuffle, isolate the WeatherChange draw.
  await run('WEATHER: Rain Dance sets Rain (DISTINCT speeds — control)',
    [mon('Electrode', ['raindance', 'splash'], { evs: { spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }]);
  // Sunny Day.
  await run('WEATHER: Sunny Day sets Sun 5 turns (distinct speeds)',
    [mon('Electrode', ['sunnyday', 'splash'], { evs: { spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }]);
  // Rain Dance when Rain is ALREADY active (from a permanent Drizzle) — overwrite vs fail?
  await run('WEATHER: Rain Dance while Rain already active (injected permanent) — overwrite?',
    [mon('Electrode', ['raindance', 'splash'], { evs: { spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    [{ weather: 'raindance' }]);
  // Rain Dance while SUN already active — overwrite to rain.
  await run('WEATHER: Rain Dance while SUN active (injected) — overwrite to rain',
    [mon('Electrode', ['raindance', 'splash'], { evs: { spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    [{ weather: 'sunnyday' }]);

  // ============ STAT-DROP MOVES ============
  // Screech: foe -2 Def, accuracy 85. Draw = accuracy roll.
  await run('STATDROP: Screech foe -2 Def (acc 85)',
    [mon('Persian', ['screech', 'splash'], { evs: { spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }]);
  // Charm: foe -2 Atk, accuracy 100.
  await run('STATDROP: Charm foe -2 Atk (acc 100)',
    [mon('Clefable', ['charm', 'splash'], { evs: { spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);
  // Screech vs Clear Body (blocked).
  await run('STATDROP: Screech vs CLEAR BODY (blocked, acc still drawn)',
    [mon('Persian', ['screech', 'splash'], { evs: { spe: 252 } })],
    [mon('Metagross', ['splash'], { ability: 'Clear Body', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);
  // Charm vs Hyper Cutter (Atk-only immunity — blocked).
  await run('STATDROP: Charm vs HYPER CUTTER (Atk immunity, blocked)',
    [mon('Clefable', ['charm', 'splash'], { evs: { spe: 252 } })],
    [mon('Pinsir', ['splash'], { ability: 'Hyper Cutter', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);
  // Metal Sound: foe -2 SpD, acc 85. Feather Dance -2 Atk acc 100. Tickle -1/-1 acc 100. Fake Tears -2 SpD acc 100.
  await run('STATDROP: Metal Sound (-2 SpD, acc 85) / Feather Dance / Tickle / Fake Tears',
    [mon('Registeel', ['metalsound', 'featherdance', 'tickle', 'faketears'], { evs: { spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }, { p1: 'move 3', p2: 'move 1' }, { p1: 'move 4', p2: 'move 1' }]);
  // Metal Sound: is it a SOUND move? blocked by Soundproof?
  await run('STATDROP: Metal Sound vs SOUNDPROOF',
    [mon('Registeel', ['metalsound', 'splash'], { evs: { spe: 252 } })],
    [mon('Electrode', ['splash'], { ability: 'Soundproof', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);

  // ============ SCREENS ============
  // Light Screen: side condition, 5 turns. Reflect: 5 turns.
  await run('SCREEN: Light Screen + Reflect (5 turns each, upkeep countdown)',
    [mon('Blissey', ['lightscreen', 'reflect', 'splash'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }, { p1: 'move 3', p2: 'move 1' },
     { p1: 'move 3', p2: 'move 1' }, { p1: 'move 3', p2: 'move 1' }, { p1: 'move 3', p2: 'move 1' },
     { p1: 'move 3', p2: 'move 1' }]);
  // Light Screen already up — re-use fails?
  await run('SCREEN: Light Screen twice (re-use fails while up)',
    [mon('Blissey', ['lightscreen', 'splash'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
