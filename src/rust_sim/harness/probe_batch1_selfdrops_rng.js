// probe_batch1_selfdrops_rng.js — settle the gen3 `move.self.boosts` SELF-DROP draw model
// with a PER-CALL-SITE PRNG trace (the mod chain is the only oracle). CORRECTS the naive
// "self-drop is draw-free" hypothesis: gen3 `selfDrops` (battle-actions.ts:1338) draws ONE
// `random(100)` (the `secondaryRoll`) when `moveData.self.boosts` exists, THEN applies the
// drop if `secondaryRoll < self.chance` OR — Overheat/Superpower have `self.chance ===
// undefined` — UNCONDITIONALLY. So the drop ALWAYS lands but the roll is ALWAYS DRAWN.
//
// The trace below wraps `battle.random` / `battle.randomChance` and prints each call's
// [args] + call-site — proving the Overheat turn draws:
//   0 randomChance[90,100]  — Overheat accuracy
//   1 randomChance[1,16]    — Overheat crit
//   2 random[16]            — Overheat damage
//   3 random[100]           — the selfDrops secondaryRoll (battle-actions.ts:1338)  <<< THE DRAW
//   4 randomChance[100,100] — the foe's move accuracy
//   5 randomChance[1,16]    — the foe's crit
//   6 random[16]            — the foe's damage
//   7 randomChance[1,5]     — Quick Claw
//
// Run:  node src/rust_sim/harness/probe_batch1_selfdrops_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function trace(label, p1team, p2team) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const c of streams.omniscient) { void c; } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[64622,12047,52124,27045]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1team })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2team })}`);
  for (let i = 0; i < 12; i++) await tick();
  const b = stream.battle;
  const stacks = [];
  const site = (n) => new Error().stack.split('\n').slice(2, 2 + n).map((x) => x.replace(/.*pokemon-showdown\//, '').trim()).join(' <- ');
  const realRandom = b.random.bind(b);
  b.random = function (...a) { stacks.push(`random ${JSON.stringify(a)} @ ${site(3)}`); return realRandom(...a); };
  const realRC = b.randomChance.bind(b);
  b.randomChance = function (...a) { stacks.push(`randomChance ${JSON.stringify(a)} @ ${site(2)}`); return realRC(...a); };
  streams.omniscient.write('>p1 move 1');
  streams.omniscient.write('>p2 move 1');
  for (let k = 0; k < 18; k++) await tick();
  console.log(`\n=== ${label} (${stacks.length} draws) ===`);
  stacks.forEach((s, i) => console.log(`  ${i}: ${s}`));
  console.log(`  p1 boosts: ${JSON.stringify(b.sides[0].active[0].boosts)}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  await trace('OVERHEAT (self -2 SpA)',
    'Charizard|||Blaze|overheat,ember|Modest|,,,252,,252|N||||',
    'Snorlax|||Immunity|pound|Careful|252,,,,252,|N||||');
  await trace('SUPERPOWER (self -1 Atk/-1 Def)',
    'Machamp|||Guts|superpower,karatechop|Adamant|,252,,,,252|N||||',
    'Snorlax|||Immunity|pound|Careful|252,,252,,,|N||||');
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
