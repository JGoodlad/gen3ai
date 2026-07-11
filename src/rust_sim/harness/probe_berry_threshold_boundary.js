// probe_berry_threshold_boundary.js — settle the EXACT berry-threshold BOUNDARY
// (`<=` vs `<`) against the resolved `Dex.mod('gen3')` sim (the only oracle).
//
// WHY: the prior probes (probe_berry_rng.js) used Snorlax boards with ODD maxhp
// (461), so exact equality (hp == maxhp/2, hp == maxhp/4) was UNREACHABLE — a
// `<=` → `<` mutation in the engine passed all 1280 golden battles + every pin
// (the reviewer's open finding). This probe constructs EVEN-maxhp boards where
// the holder lands EXACTLY on hp == maxhp/2 (sitrus/oran) and EXACTLY on
// hp == maxhp/4 (salac/liechi), plus one-HP-above controls. The PROBE decides
// whether the sim eats at equality (<=) or not (<); the engine follows the probe.
//
// TWO parts:
//   (A) the ORACLE — direct-hp-mutation boards (the omniscient probe's
//       exact-boundary tool, as in probe_berry_rng.js) on an EVEN-maxhp holder:
//       hp == maxhp/2 exactly / +1 above (heal class), hp == maxhp/4 exactly /
//       +1 above (pinch class).
//   (B) the BR6 GROUND TRUTH — the exact constructed Rust-pin scenario (packed
//       teams + seed + scripted choices, NO hp mutation): a maxhp-400 Vaporeon
//       (base HP 130, IV hp 30 → 2*130+30+0+110 = 400, EVEN) ground by Blissey's
//       Seismic Toss (fixed 100): 400 → 300 → 200 == maxhp/2 EXACTLY (sitrus) and
//       → 100 == maxhp/4 EXACTLY (salac). Per-decision seedAfter + state printed
//       verbatim for tests/regression_test.rs::BR6.
//
// Run: node src/rust_sim/harness/probe_berry_threshold_boundary.js

'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function boot(p1packed, p2packed, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of String(ch).split('\n')) if (l) lines.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1packed })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2packed })}`);
  for (let i = 0; i < 12; i++) await tick();
  return { battle: stream.battle, streams, lines };
}
async function turn(ctx, c1, c2) {
  const before = ctx.lines.length;
  if (c1) ctx.streams.omniscient.write(`>p1 ${c1}`);
  if (c2) ctx.streams.omniscient.write(`>p2 ${c2}`);
  for (let k = 0; k < 14; k++) await tick();
  return ctx.lines.slice(before);
}
function st(p) {
  return `hp=${p.hp}/${p.maxhp} item=${p.item || 'NONE'} boosts=${p.boosts.atk}:${p.boosts.def}:${p.boosts.spa}:${p.boosts.spd}:${p.boosts.spe}`;
}

const BLISSEY = 'Blissey|||NoAbility|seismictoss,softboiled||,,,,,252|N||||';
// EVEN maxhp: 2*130 (Vaporeon base HP) + IV 30 + EV 0 + 110 = 400.
const HOLDER = (item) => `Vaporeon||${item}|NoAbility|splash,watergun||,,,,,|N|30,,,,,|||`;

(async () => {
  // ───────────────────────────────────────────────────────────────────────────
  // (A) THE ORACLE — direct hp mutation, exact equality vs one-above.
  // ───────────────────────────────────────────────────────────────────────────
  console.log('=== (A) THE BOUNDARY ORACLE (even maxhp, direct hp mutation) ===');
  for (const [item, num, den, label] of [
    ['sitrusberry', 1, 2, 'HEAL sitrus thr 1/2'],
    ['oranberry', 1, 2, 'HEAL oran thr 1/2'],
    ['salacberry', 1, 4, 'PINCH salac thr 1/4'],
    ['liechiberry', 1, 4, 'PINCH liechi thr 1/4'],
  ]) {
    for (const delta of [1, 0]) { // +1 above, then EXACT equality
      const ctx = await boot(BLISSEY, HOLDER(item), [11, 22, 33, 44]);
      const holder = ctx.battle.p2.active[0];
      if (holder.maxhp % den !== 0) throw new Error(`maxhp ${holder.maxhp} not divisible by ${den}`);
      const target = holder.maxhp * num / den + delta;
      holder.hp = target;
      const lines = await turn(ctx, 'move 2', 'move 1'); // Soft-Boiled (self) + Splash: no damage
      const ate = lines.some((l) => l.includes('-enditem') && l.includes('[eat]'));
      console.log(`  ${label} maxhp=${holder.maxhp} hp=${target}` +
        ` (${delta === 0 ? `EXACT ${num}/${den}` : 'one above'}): ` +
        (ate ? 'ATE' : 'did NOT eat') + `  → ${st(holder)}`);
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // (B) BR6 GROUND TRUTH — the Rust pin's exact scenario, no mutation.
  // ───────────────────────────────────────────────────────────────────────────
  const SEED = [41001, 42002, 43003, 44004]; // same master as probe_berry_batch3_regression_rng.js
  for (const [label, item, n] of [
    ['BR6a sitrus: toss ×2 → hp 200 == maxhp/2 EXACT', 'sitrusberry', 3],
    ['BR6b salac: toss ×3 → hp 100 == maxhp/4 EXACT', 'salacberry', 4],
  ]) {
    const ctx = await boot(BLISSEY, HOLDER(item), SEED);
    const battle = ctx.battle;
    console.log(`\n=== (B) ${label} ===`);
    console.log(`  init seed: ${battle.prng.getSeed()}`);
    for (let d = 0; d < n && !battle.ended; d++) {
      await turn(ctx, 'move 1', 'move 1');
      const b = battle.sides[1].active[0];
      console.log(`  dec${d}: seedAfter=${battle.prng.getSeed()}  p2[${st(b)}]`);
    }
  }
})().catch((e) => { console.error(e); process.exit(1); });
