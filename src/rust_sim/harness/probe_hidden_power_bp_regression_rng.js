// probe_hidden_power_bp_regression_rng.js — GROUND-TRUTH for the R3 IV-derived Hidden
// Power BASE-POWER regression pin (`gen3_iv_derived_hidden_power_bp_v1`).
//
// gen-3 computes Hidden Power's BP from the ATTACKER's IVs (Dex.getHiddenPower, range
// 30..=70), NOT the flat 70 the port's data ships. A real gen3ou HP mon whose IVs give
// BP != 70 (e.g. a -1 Atk IV → BP 68) must damage at its IV-true BP. This probe drives a
// BP-68 Hidden Power Ice attacker into a bulky neutral wall and captures the sim's exact
// realized HP + seedAfter, so the Rust pin asserts the port now deals the SIM's damage
// (pre-fix it dealt the higher BP-70 damage → the assertion fails on revert).
//
// It ALSO prints the sim's hpType/hpPower so the pin's BP != 70 premise is self-documenting.
// Run:  node src/rust_sim/harness/probe_hidden_power_bp_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
// HP Ice, BP 68: IVs [hp31, atk28, def30, spa31, spd31, spe31] (port [hp,atk,def,spa,spd,spe]).
//   hpTypeX  = 1·1 + 2·0 + 4·0 + 8·1[spe] + 16·1[spa] + 32·1[spd] = 57 → ⌊57·15/63⌋=13 = Ice
//   hpPowerX = 1·1 + 2·0(atk28 bit1=0) + 4·1(def30) + 8·1 + 16·1 + 32·1 = 61 → ⌊61·40/63⌋+30 = 68
const HP_ICE_BP68 = { hp: 31, atk: 28, def: 30, spa: 31, spd: 31, spe: 31 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 },
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, seed, p1team, p2team, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) for (const l of ch.split('\n')) if (l) log.push(l); })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  const atk = battle.sides[0].active[0];
  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  console.log(`  attacker hpType=${atk.hpType} hpPower=${atk.hpPower}`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 40) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const before = battle.prng.getSeed();
    const dc0 = drawCount;
    const llen = log.length;
    const entry = plan[Math.min(i, plan.length - 1)];
    i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const after = battle.prng.getSeed();
    const rel = log.slice(llen).filter((l) => /-damage|-immune|-miss|-crit|faint/.test(l));
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp}${m.fainted ? ' FNT' : ''}` : '-';
    console.log(
      `  [${rs}] ${JSON.stringify(entry)} draws=${drawCount - dc0}\n` +
      `        seedBefore=${before}\n        seedAfter =${after}\n` +
      `        p1=${fmt(a0)}  p2=${fmt(a1)}`);
    for (const l of rel) console.log(`        > ${l}`);
    if (entry.stop) break;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // A BP-68 Hidden Power Ice from a Starmie (base SpA 100) into a bulky Blissey (neutral to
  // Ice, huge HP → never faints, a clear HP delta). Blissey Splashes so the ONLY draws are
  // Starmie's HP accuracy + crit + damage roll + the end-of-turn Quick Claw. Distinct speeds
  // (Starmie 115 > Blissey 55, no action-order tie) → a clean draw count. Pre-fix the port
  // uses BP 70 (more damage); with the fix it uses BP 68 = the sim's number below.
  await run('HP-ICE-BP68: Starmie HP Ice (BP68) into Blissey', [7, 7, 7, 7],
    [mon('Starmie', ['hiddenpowerice'], { evs: { spa: 252, spe: 252 }, ivs: HP_ICE_BP68 })],
    [mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);
}
main();
