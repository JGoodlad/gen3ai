// probe_spiderweb_link_lifetime.js — settle the gen3 SPIDER WEB / MEAN LOOK trap-LINK
// LIFETIME against the OMNISCIENT in-process BattleStream (no server). The sim is the ONLY
// oracle — source reads are hypotheses (and two of mine were already wrong here).
//
// WHY: the 24h randbats byte-fuzz's cluster B (4 repros: fuzz24h_v2 ab_1271_13 / ab_387_12 /
// ab_943_5 + fuzz_r25 ab_533_11) shows the sim FAILING a repeat Spider Web with the
// did-nothing form (`|move|…|Spider Web||[still]` + `|-fail|<user>` + the sim's own
// `|debug|move failed because it did nothing`) where the PORT re-applies it and emits
// `|-activate|<target>|trapped`. So the port has LOST the trap link the sim still holds.
//
// In ab_387_12 the trapper (Ariados) BATON-PASSED OUT and BACK between the two casts, and
// the sim STILL failed the second one. That is the puzzle: `sim/pokemon.ts` reads as though
// the link should break —
//   * `addVolatile(status, source, sourceEffect, linkedStatus)` (pokemon.ts:2015-2025) links
//     BOTH ends symmetrically (`linkedPokemon` + `linkedStatus` on the target's `trapped` AND
//     the source's `trapper`);
//   * `clearVolatile()` (pokemon.ts:1527-1531) walks the mon's volatiles and, for any with a
//     `linkedStatus`, calls `removeLinkedVolatiles(...)` — which removes the OTHER end;
//   * BOTH `trapped` and `trapper` are `noCopy: true` (conditions.ts:208-221), so Baton Pass
//     does NOT copy/re-point them.
// ⇒ a source read predicts the trap BREAKS when the trapper leaves by any means. The golden
// says otherwise for the Baton-Pass path. Only the sim can settle it.
//
// SETTLES (each printed as a VERDICT line):
//   A. baseline           — cast, then re-cast with nobody moving. (Expect: 2nd FAILS.)
//   B. trapper NORMAL-switches out and back, then re-casts.
//   C. trapper BATON-PASSES out and back, then re-casts.   (the ab_387_12 shape)
//   D. the TRAPPED mon switches out and back (it can't — it's trapped), so instead: the
//      trapped mon is PHAZE-dragged out (Roar bypasses trapping) and comes back.
//   E. the trapper FAINTS and its replacement re-casts.
//
// The observable is the 2nd cast's protocol form:
//   FAILED (still trapped) → `|move|…|Spider Web||[still]` + `|-fail|<user>`
//   FRESH  (link broken)   → `|move|…|Spider Web|<target>` + `|-activate|<target>|trapped`
//
// Run:  node src/rust_sim/harness/probe_spiderweb_link_lifetime.js
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

// p1 lead = the TRAPPER (Ariados: Spider Web + Baton Pass + Splash filler).
// p1 slot2 = a Baton-Pass partner that can pass straight back.
// p2 lead = the TARGET (Azumarill: Splash so it never interferes), slot2 = filler.
function teams() {
  const p1 = [
    mon('Ariados', ['spiderweb', 'batonpass', 'splash', 'roar'], { level: 100 }),
    mon('Smeargle', ['batonpass', 'splash'], { level: 100 }),
    mon('Sudowoodo', ['splash'], { level: 100 }),
  ];
  const p2 = [
    mon('Azumarill', ['splash'], { level: 100 }),
    mon('Snorlax', ['splash'], { level: 100 }),
  ];
  return [p1, p2];
}

async function run(label, plan) {
  const [p1team, p2team] = teams();
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify([7, 11, 13, 17])}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;

  let i = 0, safety = 0;
  while (!battle.ended && safety < 60 && i < plan.length) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const e = plan[i]; i++;
    const a0b = battle.sides[0].active[0], a1b = battle.sides[1].active[0];
    const trapB = a1b && a1b.volatiles && !!a1b.volatiles['trapped'];
    if (e.p1) streams.omniscient.write(`>p1 ${e.p1}`);
    if (e.p2) streams.omniscient.write(`>p2 ${e.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const trapA = a1 && a1.volatiles && !!a1.volatiles['trapped'];
    const errs = log.filter((l) => l.startsWith('|error|'));
    console.log(`    step[${i - 1}] rs=${rs} ${JSON.stringify(e)} :: `
      + `p1 ${a0b && a0b.species.name}->${a0 && a0.species.name} | `
      + `p2 ${a1b && a1b.species.name}->${a1 && a1.species.name} | `
      + `trapped ${trapB}->${trapA}${errs.length ? ` ERR=${errs.length}` : ''}`);
  }

  // The observable: every Spider Web announce, in order, + whether `trapped` activated.
  const webs = log.filter((l) => l.includes('|Spider Web'));
  const acts = log.filter((l) => l.includes('|trapped'));
  const target = battle.sides[1].active[0];
  const live = target && target.volatiles && !!target.volatiles['trapped'];
  const src = battle.sides[0].active[0];
  const srcTrapper = src && src.volatiles && !!src.volatiles['trapper'];

  const second = webs[1] || '(no 2nd cast)';
  const verdict = second.endsWith('[still]') ? 'FAILED (still trapped)'
    : second.includes('|Spider Web|') ? 'FRESH APPLY (link broken)' : 'INDETERMINATE';
  console.log(`\n=== ${label} ===`);
  for (const w of webs) console.log('   ', w);
  for (const a of acts) console.log('   ', a);
  console.log(`    live: target.volatiles.trapped=${live}  source.volatiles.trapper=${srcTrapper}`);
  console.log(`    VERDICT: ${verdict}`);
  return verdict;
}

(async () => {
  // A — baseline: cast, idle a turn, re-cast. The link is untouched.
  await run('A baseline (nobody leaves)', [
    { p1: 'move spiderweb', p2: 'move splash' },
    { p1: 'move splash', p2: 'move splash' },
    { p1: 'move spiderweb', p2: 'move splash' },
  ]);

  // B — the trapper NORMAL-switches out, then back, then re-casts.
  await run('B trapper NORMAL switch out+back', [
    { p1: 'move spiderweb', p2: 'move splash' },
    { p1: 'switch 2', p2: 'move splash' },
    { p1: 'switch 1', p2: 'move splash' },
    { p1: 'move spiderweb', p2: 'move splash' },
  ]);

  // C — the trapper BATON-PASSES out, then back, then re-casts (the ab_387_12 shape).
  await run('C trapper BATON PASS out+back', [
    { p1: 'move spiderweb', p2: 'move splash' },
    { p1: 'move batonpass', p2: 'move splash' },
    { p1: 'switch 2' },                              // the BP replacement choice
    { p1: 'move batonpass', p2: 'move splash' },
    { p1: 'switch 1' },
    { p1: 'move spiderweb', p2: 'move splash' },
  ]);

  // D — the TRAPPED mon is Roar-dragged out (phaze bypasses trapping), then re-cast.
  await run('D trapped mon PHAZE-dragged out', [
    { p1: 'move spiderweb', p2: 'move splash' },
    { p1: 'move roar', p2: 'move splash' },
    { p1: 'move spiderweb', p2: 'move splash' },
  ]);
})();
