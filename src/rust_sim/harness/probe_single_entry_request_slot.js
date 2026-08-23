// probe_single_entry_request_slot.js — the SIM is the oracle for an OUT-OF-RANGE numeric slot
// on the two request shapes that offer only ONE entry.
//
// `harness/replay_impl_parity.py` on a fresh golden caught the port SILENTLY ACCEPTING `move 2`
// from a mon whose request offered only Struggle: node emitted
// `|error|[Invalid choice] Can't move: Your Blissey doesn't have a move 2` and then took the
// follow-up policy's pick, while the port substituted Struggle without a word — a 1-vs-2
// `choices_used` and a completely different arm.
//
// The port has TWO substitution shapes that collapse a request to a single entry — a forced
// STRUGGLE and a move LOCK (charging two-turn / recharge) — and it early-outs of the reject
// classifier on BOTH. The STRUGGLE half is already node-measured, by that harness run itself:
// a real `node replay_driver.js` on a real 0-PP board, which is a better oracle than anything
// synthetic. This probe measures the OTHER half, the move LOCK, together with the CONTROL
// (`move 1`, the entry the request DOES offer) — so the fix cannot be tuned into refusing the
// one action the sim offered, the shape that killed two production launches
// (`gen3_locked_choice_never_rejected_v1`).
//
// (A Struggle arm was attempted here and dropped. Driving a mon to 0 PP inside a probe is
// three traps deep: Sand Attack saturates at -6 accuracy and the sim then repeats a
// byte-identical request forever; a fixed tick budget reads a STALE request and reports
// "accepted silently" for a token that was refused; and the drain reaches a `wait:true`
// boundary that swallows p1 tokens. A probe that measures the wrong boundary is worse than no
// probe — `bridge_choice_reject_test::a_forced_struggle_*` covers that half against the port.)
//
//   node src/rust_sim/harness/probe_single_entry_request_slot.js

'use strict';

const path = require('path');
const psPath = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(psPath, 'dist/sim/battle-stream'));

const tick = () => new Promise((r) => setImmediate(r));

function session() {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const chunks = { p1: [], p2: [] };
  for (const side of ['p1', 'p2']) {
    (async () => {
      try { for await (const c of streams[side]) chunks[side].push(c); } catch (e) { /* destroyed */ }
    })();
  }
  return { stream, streams, chunks };
}

const lines = (chunks) => chunks.join('\n').split('\n').filter(Boolean);

function lastRequest(ls) {
  for (let i = ls.length - 1; i >= 0; i--) if (ls[i].startsWith('|request|')) return ls[i];
  return null;
}

// p1 leads with a mon that can be driven into each single-entry shape; p2 just chips.
// LOCK: a Solar Beam charging turn (no sun, so it takes two turns).
const LOCK_P1 = 'Venusaur||leftovers|overgrow|solarbeam,sludgebomb,sleeppowder,synthesis|||||100|]Snorlax||leftovers|immunity|bodyslam,earthquake,rest,curse|||||100|';
const P2 = 'Blissey||leftovers|naturalcure|softboiled,calmmind,lightscreen,reflect|||||100|';

// Drive until the p1 request matches `ready`, then feed `probeToken`.
//
// ⚠️ Two ways to get this wrong, and BOTH look like a passing sim rather than a broken probe:
//   * too FEW microtask turns between decisions and the probe reads the PREVIOUS request, so a
//     token the sim refused reports as "accepted silently" (8 and 16 ticks both did this);
//   * a "wait until the request CHANGES" loop hangs once Sand Attack saturates the accuracy
//     drop at -6 — the sim then emits a byte-identical request every turn.
// So: a generous FIXED settle, and a LOUD throw if the boundary was never reached.
//
// And DESTROY each session before the next: the per-side `for await` readers stay alive
// otherwise, and six live BattleStreams starve each other's microtasks, so the LATER arms
// under-settle while the first looks fine.
const SETTLE_TICKS = 64;
const settle = async () => { for (let t = 0; t < SETTLE_TICKS; t++) await tick(); };

async function driveUntil(p1team, ready, probeToken, maxTurns) {
  const s = session();
  s.streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[7,11,13,17]}`);
  s.streams.omniscient.write(`>player p1 {"name":"P1","team":"${p1team}"}`);
  s.streams.omniscient.write(`>player p2 {"name":"P2","team":"${P2}"}`);
  await settle();
  let before = lastRequest(lines(s.chunks.p1));
  for (let i = 0; i < maxTurns && !ready(offered(before)); i++) {
    // p2 eventually faints to the drain move and must replace; while it does, p1's request is
    // `wait:true` and any p1 token is ignored. Answering only p2 keeps the drive moving —
    // without this the loop spins on a wait request and reports "never reached the boundary".
    if (before && before.includes('"wait":true')) {
      s.streams.omniscient.write(`>p2 default`);
      await settle();
      before = lastRequest(lines(s.chunks.p1));
      if (process.env.PROBE_DEBUG) console.log(`  turn ${i}: (p2 replacing)`);
      continue;
    }
    s.streams.omniscient.write(`>p1 move 1`);
    s.streams.omniscient.write(`>p2 move 1`);
    await settle();
    before = lastRequest(lines(s.chunks.p1));
    if (process.env.PROBE_DEBUG) console.log(`  turn ${i}: ${before && before.slice(9, 80)}`);
  }
  if (!ready(offered(before))) {
    try { s.stream.destroy(); } catch (e) { /* already gone */ }
    throw new Error(`never reached the boundary; last request offered ${JSON.stringify(offered(before))}`);
  }
  const mark = s.chunks.p1.length;
  s.streams.omniscient.write(`>p1 ${probeToken}`);
  await settle();
  const out = { before, after: lines(s.chunks.p1.slice(mark)) };
  try { s.stream.destroy(); } catch (e) { /* already gone */ }
  await tick();
  return out;
}

function offered(req) {
  try {
    const j = JSON.parse(req.slice('|request|'.length));
    return (j.active && j.active[0] && j.active[0].moves || []).map((m) => m.move);
  } catch (e) { return ['<unparsed>']; }
}

(async () => {
  // Solar Beam charges on the first decision, so the single-entry boundary is one turn in.
  const isSolarBeam = (mv) => mv.length === 1 && mv[0] === 'Solar Beam';

  for (const [label, team, ready, maxTurns] of [
    ['LOCK(solarbeam charging)', LOCK_P1, isSolarBeam, 4],
  ]) {
    for (const token of ['move 1', 'move 2', 'move 4']) {
      const { before, after } = await driveUntil(team, ready, token, maxTurns);
      console.log(`\n=== ${label}  <<${token}>>`);
      console.log(`  request offered : ${JSON.stringify(offered(before))}`);
      console.log(`  reply lines     :`);
      for (const l of after.slice(0, 3)) console.log(`    ${l.slice(0, 140)}`);
      if (!after.length) console.log('    <nothing — accepted silently>');
    }
  }
})();
