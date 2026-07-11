// probe_switch_sand.js — pin the MISSING draw on a MID-TURN switch-in whose entrant
// TIES the opposing active under FRESHLY-SET weather (the e2e_84 dec4 desync).
//
// THE BUG (EDGE_CASES.md / CLAUDE.md): a 213-speed Tyranitar (Sand Stream) switches in
// MID-TURN while a 213-speed Suicune acts. The two actives TIE (213 == 213). When the
// Tyranitar sets sandstorm on the switch-in, `Field.setWeather` ends with
// `this.battle.eachEvent('WeatherChange', sourceEffect)` (field.ts:87) → `speedSort(actives)`
// → a 2-active SPEED-TIE Fisher-Yates `shuffle(list,0,2)` → ONE `random(0,2)` draw. The
// port's switch path sets the weather but MISSES that `eachEvent('WeatherChange')` shuffle.
//
// This probe instruments `battle.prng.next` to COUNT raw draws per decision window, and
// runs three controlled cases that ISOLATE the weather-shuffle draw:
//   (A) a Sand-Stream switch-into-a-TIE (the buggy case — expect +1 draw vs control C)
//   (B) a NO-weather switch-into-a-TIE (the baseline — Tyranitar w/ a non-weather ability)
//   (C) a Sand-Stream switch-into-a-NON-tie (distinct speed → the eachEvent speedSort has
//       no tie → no shuffle draw; isolates the "weather draws ONLY on a tie" claim)
//
// Run:  node src/rust_sim/harness/probe_switch_sand.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

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

async function run(label, p1team, p2team, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  const seed = [52903, 53571, 56373, 31187]; // the e2e_84-class init seed (a tie-shuffle exerciser)
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  const a0 = () => battle.sides[0].active[0], a1 = () => battle.sides[1].active[0];
  const spd = (m) => (m ? m.getStat('spe') : '-');
  console.log(`  speeds: p1=${spd(a0())} p2=${spd(a1())}  weather=${battle.field.weather || 'none'}`);

  let i = 0, safety = 0;
  while (!battle.ended && safety < 50) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    if (i >= plan.length) break;
    const before = battle.prng.getSeed();
    const dc0 = drawCount;
    const entry = plan[i]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const after = battle.prng.getSeed();
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} spe=${m.getStat('spe')}` : '-';
    console.log(`  [${rs}] ${JSON.stringify(entry)} draws=${drawCount - dc0}  weather=${battle.field.weather || 'none'}`);
    console.log(`        seedBefore=${before}`);
    console.log(`        seedAfter =${after}`);
    console.log(`        p1=${fmt(a0())} | p2=${fmt(a1())}`);
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // 213-speed mons that TIE. Tyranitar @ neutral nature, 0 EV ⇒ base 61 spe → 211?
  // Use explicit speed-tying spreads: Suicune (base spe 85) and a switch-in tying it.
  // To force a clean 213==213 tie we use two mons with matched final spe via EVs.
  // Simpler: Suicune base spe 85 → at 0 EV/31 IV/serious/100 = 236. Tyranitar base 61 = 188.
  // We don't need the EXACT 213; ANY exact tie exercises the weather shuffle. Use a Suicune
  // mirror-class: p2 Suicune (spe 236), p1 switches a SECOND Suicune in (spe 236) → TIE.

  // (A) Sand-Stream switch-into-a-TIE: p1 has [Skarmory lead, Tyranitar(SandStream, spe-matched)].
  //     p1 switches Tyranitar in while p2's Suicune acts; Tyranitar spe == Suicune spe → TIE +
  //     Tyranitar sets sand → the eachEvent('WeatherChange') tie-shuffle draws.
  //     To match speeds exactly we tune Tyranitar EVs so getStat('spe') == Suicune's.
  //     We'll discover the speeds from the printed header and just demonstrate the DELTA:
  //     compare the same switch-into-tie WITH sand (A) vs WITHOUT sand (B).
  // Tyranitar 252-spe-EV serious = 221 spe; Suicune 60-spe-EV serious = 221 spe → exact TIE.
  const suicune = (moves) => mon('Suicune', moves, { evs: { hp: 252, spe: 60 }, nature: 'Serious' });
  const ttarSand = mon('Tyranitar', ['crunch', 'rock slide'],
    { ability: 'Sand Stream', evs: { hp: 252, spe: 252 }, nature: 'Serious' });
  const ttarNoWeather = mon('Tyranitar', ['crunch', 'rock slide'],
    { ability: 'Pressure', evs: { hp: 252, spe: 252 }, nature: 'Serious' });

  // p1 lead = a spe-matched Suicune so the switch-in Tyranitar tie is exercised against p2's Suicune.
  await run('(A) Sand-Stream switch INTO a 213-tie (the buggy +1 draw)',
    [suicune(['surf', 'splash']), ttarSand],
    [suicune(['surf', 'splash'])],
    [
      // turn 1: p1 SWITCHES Tyranitar in (slot 2) while p2 Suicune acts (Splash). The
      // switch (order 103) runs FIRST, sand is set, then p2's Splash + the eachEvent shuffles.
      { p1: 'switch 2', p2: 'move 2' },
    ]);

  await run('(B) NO-weather switch INTO a 213-tie (the baseline, no weather draw)',
    [suicune(['surf', 'splash']), ttarNoWeather],
    [suicune(['surf', 'splash'])],
    [
      { p1: 'switch 2', p2: 'move 2' },
    ]);

  // (C) Sand-Stream switch into a NON-tie: p2 Suicune slowed to NOT tie (Brave 0-spe-EV).
  await run('(C) Sand-Stream switch into a NON-tie (distinct speed → no weather shuffle)',
    [suicune(['surf', 'splash']), ttarSand],
    [mon('Suicune', ['surf', 'splash'], { evs: { hp: 252 }, nature: 'Brave' })],
    [
      { p1: 'switch 2', p2: 'move 2' },
    ]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
