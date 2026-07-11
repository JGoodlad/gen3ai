// probe_ability_batch1_regression_rng.js — GROUND TRUTH for the `gen3_ability_batch1_v1`
// regression pins (B1 CRIT_IMMUNE / B2 WEATHER_SPEED / B3 WEATHER_NEGATE / B4 RESIDUAL) in
// tests/regression_test.rs.
//
// Each scenario drives the OMNISCIENT in-process BattleStream (no server) over a CONSTRUCTED
// gen3customgame board whose EXACT packed teams + seed the Rust regression test replays. It
// RESEEDS the sim's prng to the RAW seed right before the first decision so the decision draws
// line up with the Rust's draw-free `start_with_switchins`, then prints the per-decision
// seedAfter + the observable STATE — copied verbatim into the pins as the real-Showdown ground
// truth. Each pin FAILS if its class's engine wiring is reverted.
//
// Run:  node src/rust_sim/harness/probe_ability_batch1_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));

function tick() { return new Promise((r) => setTimeout(r, 0)); }

// Drive one constructed battle, reseeding to `rawSeed` at the first decision, and print each
// decision's seedAfter + both actives' species/hp/status/speBoost + the first mover.
async function run(label, p1, p2, rawSeed, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[1,2,3,4]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2 })}`);
  for (let i = 0; i < 12; i++) await tick();
  const b = stream.battle;
  b.prng = new PRNG(rawSeed.slice());

  console.log(`\n=== ${label} (raw seed ${rawSeed.join(',')}) ===`);
  const a0 = () => b.sides[0].active[0], a1 = () => b.sides[1].active[0];
  console.log(`  weather=${b.field.weather || 'none'}  spe: p1=${a0().getStat('spe')} p2=${a1().getStat('spe')}`);

  let i = 0, safety = 0;
  while (!b.ended && safety < 60 && i < plan.length) {
    safety++;
    const rs = b.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const before = log.length;
    const entry = plan[i]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const first = (() => {
      for (let j = before; j < log.length; j++) {
        const p = log[j].split('|');
        if ((p[1] === 'move' || p[1] === 'cant') && p.length >= 3) {
          const a = (p[2] || '').trim();
          if (a.startsWith('p1a:')) return 'p1';
          if (a.startsWith('p2a:')) return 'p2';
        }
      }
      return 'none';
    })();
    const crit = log.slice(before).some((l) => l.startsWith('|-crit|'));
    const fmt = (m) => `${m.species.name} ${m.hp}/${m.maxhp}${m.status ? ' ' + m.status : ''} speB=${m.boosts.spe || 0}`;
    console.log(`  dec ${i - 1} [${rs}] ${JSON.stringify(entry)} first=${first} crit=${crit}`);
    console.log(`    seedAfter=${b.prng.getSeed()}`);
    console.log(`    p1=${fmt(a0())} | p2=${fmt(a1())}`);
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // B1 CRIT_IMMUNE: a Battle Armor Snorlax vs a high-crit Slash (Ursaring). On the chosen raw
  // seed the crit roll comes up (a control WOULD crit) but Battle Armor prevents it → the armor
  // mon takes the NON-crit hit. We ALSO run the same board with a NO-OP ability (Insomnia) to
  // print the crit-lands control (same raw seed → the crit fires, more damage).
  const armor = 'Snorlax|||BattleArmor|bodyslam,rest|Careful|252,,,,252,|N||||';
  const noop = 'Snorlax|||Insomnia|bodyslam,rest|Careful|252,,,,252,|N||||';
  const ursaring = 'Ursaring|||Guts|slash,rest|Adamant|252,252,,,,|N||||';
  // Seed chosen (scan) so the crit roll COMES UP: the Insomnia control CRITS (big damage) while
  // Battle Armor prevents it (small damage) at the IDENTICAL post-turn seed (the draw-free override).
  await run('B1 CRIT_IMMUNE armor (Battle Armor blocks the crit)', armor, ursaring,
    [21041, 42460, 1931, 46958], [{ p1: 'move 1', p2: 'move 1' }]);
  await run('B1 CRIT_IMMUNE control (Insomnia — the crit LANDS, same seed)', noop, ursaring,
    [21041, 42460, 1931, 46958], [{ p1: 'move 1', p2: 'move 1' }]);

  // B2 WEATHER_SPEED: a slow Chlorophyll Bellossom (spe 155) vs Groudon (Drought, spe 216). In
  // sun Bellossom ×2 = 310 > 216 → Bellossom moves FIRST (the flip). Control: Insomnia Bellossom
  // stays 155 < 216 → Groudon first.
  const chloro = 'Bellossom|||Chlorophyll|razorleaf,rest|Serious||||||';
  const chloroNoop = 'Bellossom|||Insomnia|razorleaf,rest|Serious||||||';
  const groudon = 'Groudon|||Drought|earthquake,rest|Serious||||||';
  await run('B2 WEATHER_SPEED chlorophyll (Bellossom FIRST in sun via ×2)', chloro, groudon,
    [23145, 51002, 8890, 44120], [{ p1: 'move 1', p2: 'move 1' }]);
  await run('B2 WEATHER_SPEED control (Insomnia — Groudon first, no ×2)', chloroNoop, groudon,
    [23145, 51002, 8890, 44120], [{ p1: 'move 1', p2: 'move 1' }]);

  // B3 WEATHER_NEGATE: a Cloud Nine Psyduck (Water, non-Rock/Ground/Steel) vs a Sand Stream
  // Tyranitar. Sand is up but Cloud Nine SUPPRESSES the chip → Psyduck takes NO sand chip (only
  // the foe's move damage). Control: a no-op-ability Psyduck (Damp) takes the sand chip.
  const cloudnine = 'Psyduck|||CloudNine|surf,rest|Bold|252,,252,,,|N||||';
  const dampctl = 'Psyduck|||Damp|surf,rest|Bold|252,,252,,,|N||||';
  const ttar = 'Tyranitar|||SandStream|rockslide,rest|Careful|252,,,,252,|N||||';
  await run('B3 WEATHER_NEGATE cloudnine (NO sand chip on Psyduck)', cloudnine, ttar,
    [40012, 7781, 55230, 19004], [{ p1: 'move 1', p2: 'move 1' }]);
  await run('B3 WEATHER_NEGATE control (Damp — Psyduck TAKES the sand chip)', dampctl, ttar,
    [40012, 7781, 55230, 19004], [{ p1: 'move 1', p2: 'move 1' }]);

  // B4 RESIDUAL: Speed Boost Ninjask +1 spe stage per active turn (activeTurns-gated → +1 on
  // the FIRST end-of-turn after its entry turn). vs a bulky Snorlax so it survives multiple
  // turns; the spe stage climbs 0 → +1 → +2. Rain Dish is validated in the golden (heals in
  // rain); here we pin Speed Boost's residual (the class member with the boost-STATE signal).
  const ninjask = 'Ninjask|||SpeedBoost|aerialace,rest|Jolly|252,252,,,,|N||||';
  const snorlax = 'Snorlax|||Immunity|bodyslam,rest|Impish|252,,252,,,|N||||';
  await run('B4 RESIDUAL speedboost (Ninjask +1 spe/turn)', ninjask, snorlax,
    [11002, 62210, 3345, 28890], [
      { p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }]);

  // B4b RESIDUAL: Rain Dish Ludicolo heals maxhp/16 each end-of-turn in rain (set by a foe
  // Kyogre's Drizzle). We drive several turns so the -heal + the exact HP are pinned.
  const raindish = 'Ludicolo|||RainDish|surf,rest|Calm|252,,,,252,|N||||';
  const kyogreWeak = 'Kyogre|||Drizzle|icebeam,rest|Modest|,,,4,,|N||||';
  await run('B4b RESIDUAL raindish (Ludicolo +maxhp/16 heal in rain)', raindish, kyogreWeak,
    [50501, 9987, 44012, 60123], [
      { p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
