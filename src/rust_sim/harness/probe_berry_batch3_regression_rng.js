// probe_berry_batch3_regression_rng.js — the GROUND-TRUTH capture for the batch-3
// (`gen3_berry_trace_shedskin_v1`) regression pins BR1-BR5 in tests/regression_test.rs.
// Runs each pin's EXACT constructed scenario (packed team + seed + scripted choices)
// on the REAL sim and prints the per-decision post-turn seed + state; the printed
// values are copied VERBATIM into the Rust pins (Mandate 4 — a probe is the truth
// source, never a Rust-run-once).
// Run: node src/rust_sim/harness/probe_berry_batch3_regression_rng.js

'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
function tick() { return new Promise((r) => setTimeout(r, 0)); }

// The packed-team strings EXACTLY as the Rust pins pass them (unpack-compatible).
const SEED = [41001, 42002, 43003, 44004];

async function run(label, p1packed, p2packed, plan, n) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(SEED)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1packed })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2packed })}`);
  for (let i = 0; i < 10; i++) await tick();
  const battle = stream.battle;
  console.log(`\n=== ${label} ===`);
  console.log(`  init seed: ${battle.prng.getSeed()}`);
  for (let d = 0; d < n && !battle.ended; d++) {
    const [c1, c2] = plan(d);
    if (battle.requestState === 'switch') {
      // forced replacements are not part of these short pins
      break;
    }
    if (c1) streams.omniscient.write(`>p1 ${c1}`);
    if (c2) streams.omniscient.write(`>p2 ${c2}`);
    for (let i = 0; i < 16; i++) await tick();
    const a = battle.sides[0].active[0];
    const b = battle.sides[1].active[0];
    console.log(`  dec${d}: seedAfter=${battle.prng.getSeed()}  ` +
      `p1[hp=${a.hp} st=${a.status || '-'} it=${a.item || '-'} ab=${a.ability} b=${a.boosts.atk}:${a.boosts.def}:${a.boosts.spa}:${a.boosts.spd}:${a.boosts.spe}]  ` +
      `p2[hp=${b.hp} st=${b.status || '-'} it=${b.item || '-'} b=${b.boosts.atk}:${b.boosts.def}:${b.boosts.spa}:${b.boosts.spd}:${b.boosts.spe}]`);
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

(async () => {
  const blissey = 'Blissey|||NoAbility|seismictoss,softboiled||N|,,,,,252|N||||';
  // BR1 sitrus: the toss grind crosses 2*hp<=maxhp → the residual eat heals +30.
  await run('BR1 sitrus (eat at the half threshold, +30)',
    blissey,
    'Snorlax||sitrusberry|NoAbility|splash,bodyslam|Adamant|252,252,,,,|N||||',
    () => ['move 1', 'move 1'], 5);
  // BR1b same but LEFTOVERS control (the tie-slot twin — no threshold, per-turn heal).
  await run('BR1b leftovers control (same slot, different behavior)',
    blissey,
    'Snorlax||leftovers|NoAbility|splash,bodyslam|Adamant|252,252,,,,|N||||',
    () => ['move 1', 'move 1'], 5);

  // BR2 lum: TWave → the IMMEDIATE eat cures before any boundary; the eat is DRAW-FREE
  // so the no-item control's seed is IDENTICAL while its status DIFFERS (par sticks).
  await run('BR2 lum (immediate draw-free cure on TWave)',
    'Jolteon|||NoAbility|thunderwave,thunderbolt|Timid|,,,,252,252|N||||',
    'Snorlax||lumberry|NoAbility|bodyslam,earthquake|Adamant|252,252,,,,|N||||',
    (d) => [d === 0 ? 'move 1' : 'move 2', 'move 1'], 2);
  await run('BR2b no-item control (identical draws, par sticks)',
    'Jolteon|||NoAbility|thunderwave,thunderbolt|Timid|,,,,252,252|N||||',
    'Snorlax|||NoAbility|bodyslam,earthquake|Adamant|252,252,,,,|N||||',
    (d) => [d === 0 ? 'move 1' : 'move 2', 'move 1'], 2);

  // BR3 starf: the toss grind crosses 4*hp<=maxhp → ONE sample + a +2 boost.
  await run('BR3 starf (pinch sample → +2 on a random stat)',
    blissey,
    'Snorlax||starfberry|NoAbility|splash,bodyslam|Adamant|252,252,,,,|N||||',
    () => ['move 1', 'move 1'], 6);

  // BR4 shed skin: a Thunder Wave paras the holder → ONE randomChance(33,100) per
  // STATUSED residual until the cure. Control: a no-op ability (Run Away) — no roll,
  // so the seed streams diverge from the first statused residual on.
  await run('BR4 shed skin (per-residual roll until the cure)',
    'Blissey|||NoAbility|thunderwave,softboiled||,,,,,252|N||||',
    'Arbok|||ShedSkin|sludgebomb,earthquake|Adamant|252,252,,,,|N||||',
    (d) => [d === 0 ? 'move 1' : 'move 2', 'move 1'], 5);
  await run('BR4b run-away control (no roll — the seed stream differs once statused)',
    'Blissey|||NoAbility|thunderwave,softboiled||,,,,,252|N||||',
    'Arbok|||RunAway|sludgebomb,earthquake|Adamant|252,252,,,,|N||||',
    (d) => [d === 0 ? 'move 1' : 'move 2', 'move 1'], 5);

  // BR5 trace: a MID-BATTLE Gardevoir switch-in draws the n=1 sample + copies
  // Immunity (live: a later Toxic into it fails). Control: a Limber Gardevoir
  // (no draw — the streams differ from the switch decision on).
  await run('BR5 trace (n=1 sample + live copy)',
    'Machamp|||Guts|crosschop,crosschop|Adamant|252,252,,,,|N||||]Gardevoir|||Trace|psychic,thunderbolt|Modest|,,,,252,252|N||||',
    'Snorlax|||Immunity|bodyslam,earthquake|Adamant|252,252,,,,|N||||',
    (d) => [d === 0 ? 'switch 2' : 'move 1', 'move 1'], 3);
  await run('BR5b limber control (no trace draw)',
    'Machamp|||Guts|crosschop,crosschop|Adamant|252,252,,,,|N||||]Gardevoir|||Limber|psychic,thunderbolt|Modest|,,,,252,252|N||||',
    'Snorlax|||Immunity|bodyslam,earthquake|Adamant|252,252,,,,|N||||',
    (d) => [d === 0 ? 'switch 2' : 'move 1', 'move 1'], 3);
})().catch((e) => { console.error(e); process.exit(1); });
