// probe_dc_batch_regression_rng.js — GROUND TRUTH for the draw-count/first-mover tail fixes
// (rmry3vbgm/rmry3ytkn round). Drives the OMNISCIENT in-process BattleStream, RESEEDED to a RAW
// seed right before the first decision (matching the Rust's draw-free `start_with_switchins`),
// and prints each decision's seedAfter + both actives' hp/status. THE SIM IS THE ONLY ORACLE.
//
//   RS1 `gen3_rest_sleep_immune_v1`   — a damaged INSOMNIA mon's Rest fails DRAW-FREE (== a
//        draw-free Amnesia control); a SLEEP-ABLE control's Rest draws the sleep random(2,6).
//   MD1 `gen3_mimic_disable_self_overwrite_v1` — a Disable of a mon whose lastMove is a Mimic
//        that overwrote its own slot FAILS; the Mimic-copied move stays usable (deals damage).
//   YW1 `gen3_yawn_recast_v1`         — a Yawn re-cast into a still-pending yawn FAILS without
//        resetting the duration → the sleep resolves on the ORIGINAL schedule.
//
// Run:  node src/rust_sim/harness/probe_dc_batch_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim'));
function tick() { return new Promise((r) => setTimeout(r, 0)); }
const st = (a) => a ? `hp=${a.hp}/${a.maxhp} ${a.status || '-'}` : '-';

async function run(label, p1, p2, rawSeed, plan, injectHp) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const _ of streams.omniscient) {} })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[1,2,3,4]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2 })}`);
  for (let i = 0; i < 12; i++) await tick();
  const b = stream.battle;
  b.prng = new PRNG(rawSeed.slice()); // align to the raw pre-first-decision seed
  if (injectHp) b.sides[injectHp.side].active[0].hp = injectHp.hp;
  console.log(`\n=== ${label} (raw seed ${rawSeed.join(',')}) ===`);
  let i = 0;
  for (const e of plan) {
    if (e.p1) streams.omniscient.write(`>p1 ${e.p1}`);
    if (e.p2) streams.omniscient.write(`>p2 ${e.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    console.log(`  dec ${i} seedAfter=${b.prng.getSeed()} | p1 ${st(b.sides[0].active[0])} | p2 ${st(b.sides[1].active[0])}`);
    i++;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  const seed = [13, 27, 41, 55];

  // RS1 — Insomnia Rest fails draw-free; a same-seed Amnesia control is IDENTICAL; a sleep-able
  //   Rest draws the random(2,6). Hypno hp injected to 100 (< max) so Rest passes the full-HP guard.
  const hypno = "Hypno|||Insomnia|rest,amnesia,,|Careful|248,,,,,|N||||";
  const snorAmn = "Snorlax|||Immunity|amnesia,rest,,|Careful|,,,,,|N||||";
  await run('RS1a Insomnia REST (damaged)', hypno, snorAmn, seed,
    [{ p1: 'move 1', p2: 'move 1' }], { side: 0, hp: 100 });
  await run('RS1b Insomnia AMNESIA control (damaged)', hypno, snorAmn, seed,
    [{ p1: 'move 2', p2: 'move 1' }], { side: 0, hp: 100 });
  const snorInsom = "Snorlax|||Insomnia|rest,amnesia,,|Careful|,,,,,|N||||"; // sleep-able? Insomnia blocks slp too
  // A genuinely SLEEP-ABLE Rest control uses a NON-blocking ability (Immunity blocks psn, not slp):
  await run('RS1c sleep-ABLE REST (draws random(2,6))', snorAmn, snorAmn, seed,
    [{ p1: 'move 2', p2: 'move 1' }], { side: 0, hp: 100 }); // Snorlax move2 = Rest

  // MD1 — Noctowl (fast, Insomnia) Mimics Grimer's Tackle then Grimer Disables it. Noctowl:
  //   [Mimic, Amnesia, Recover, Rest]; Grimer: [Tackle, Disable, Amnesia, Recover].
  const noctowl = "Noctowl|||Insomnia|mimic,amnesia,recover,rest|Careful|248,,,,,|N||||";
  const grimer = "Grimer|||Stench|tackle,disable,amnesia,recover|Careful|248,,,,,|N||||";
  await run('MD1 Mimic-then-Disable (disable must FAIL, copied move stays usable)', noctowl, grimer, seed, [
    { p1: 'move 2', p2: 'move 1' }, // dec0: Noctowl Amnesia; Grimer Tackle (sets Grimer lastMove)
    { p1: 'move 1', p2: 'move 2' }, // dec1: Noctowl Mimic(Tackle); Grimer Disable (targets lastMove=mimic → FAILS)
    { p1: 'move 1', p2: 'move 3' }, // dec2: Noctowl uses the Mimic slot (now Tackle) → Grimer takes damage
  ]);

  // YW1 — Swalot Yawns Snorlax, re-Yawns while pending; the resolve is on the ORIGINAL schedule.
  const swalot = "Swalot|||LiquidOoze|yawn,amnesia,,|Careful|248,,,,,|N||||";
  const snorTarget = "Snorlax|||Immunity|amnesia,rest,,|Careful|,,,,,|N||||";
  await run('YW1 Yawn re-cast (resolve on original schedule)', swalot, snorTarget, seed, [
    { p1: 'move 1', p2: 'move 1' }, // dec0: Yawn cast (draw-free)
    { p1: 'move 1', p2: 'move 1' }, // dec1: Yawn RE-CAST (fails, no reset) — the yawn RESOLVES here
  ]);
}
main().catch((e) => { console.error(e); process.exit(1); });
