// probe_choice_reject_framing.js — the SIM is the oracle for the CHOICE-REJECT byte forms.
//
// The port models exactly ONE reject class (the trapped SWITCH at a move boundary). The
// search/replay drivers feed arbitrary candidate choices, so they reach the OTHER classes —
// a DISABLED move, a switch into a FAINTED slot, a switch into the ACTIVE, an out-of-range
// slot — and the parity harnesses allowlist the difference. This captures what Node actually
// emits for each, per-side, so the port can be fixed against bytes rather than a source read.
//
// For each case we print, from the OFFENDING side's stream:
//   * the exact `|error|` line (or its absence),
//   * whether a re-request FOLLOWS it, and
//   * the diff between the re-request and the request it answered.
// ...and from the OTHER side's stream, whether it was re-asked at all (the port re-opens the
// boundary to BOTH sides; the claim under test is that Node re-asks only the offender).
//
//   node src/rust_sim/harness/probe_choice_reject_framing.js

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

function lines(chunks) { return chunks.join('\n').split('\n').filter(Boolean); }

function lastRequest(ls) {
  for (let i = ls.length - 1; i >= 0; i--) if (ls[i].startsWith('|request|')) return ls[i];
  return null;
}

// Two Choice-Band Aerodactyl vs two Snorlax. CB locks Aerodactyl to its first move,
// which makes every OTHER slot `disabled` in the next request — the search harness's case.
const P1 = 'Aerodactyl||choiceband|rockhead|doubleedge,earthquake,rockslide,substitute|||||100|]' +
           'Snorlax||leftovers|immunity|bodyslam,earthquake,rest,curse|||||100|';
const P2 = 'Snorlax||leftovers|immunity|bodyslam,earthquake,rest,curse|||||100|]' +
           'Blissey||leftovers|naturalcure|seismictoss,softboiled,toxic,icebeam|||||100|';

async function run(label, script) {
  const s = session();
  s.streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[7,11,13,17]}`);
  s.streams.omniscient.write(`>player p1 {"name":"P1","team":"${P1}"}`);
  s.streams.omniscient.write(`>player p2 {"name":"P2","team":"${P2}"}`);
  await tick(); await tick();

  const marks = {};
  await script(s, marks, async () => { await tick(); await tick(); });

  const l1 = lines(s.chunks.p1);
  const l2 = lines(s.chunks.p2);
  console.log(`\n${'='.repeat(78)}\nCASE: ${label}\n${'='.repeat(78)}`);
  for (const [side, ls] of [['p1', l1], ['p2', l2]]) {
    const from = marks[side] || 0;
    const after = ls.slice(from);
    console.log(`  --- ${side} lines emitted AFTER the offending choice (${after.length}) ---`);
    for (const x of after) {
      console.log(`    ${x.length > 200 ? x.slice(0, 200) + ' …' : x}`);
    }
  }
  try { s.stream.destroy(); } catch (e) { /* already ended */ }
  return { l1, l2 };
}

(async () => {
  // ---- CASE 1: a DISABLED move (Choice lock) -------------------------------
  await run('disabled move (Choice lock)', async (s, marks, settle) => {
    s.streams.p1.write('move 1');           // Double-Edge -> locks the CB
    s.streams.p2.write('move 1');
    await settle();
    const before = lastRequest(lines(s.chunks.p1));
    console.log('\n[case1] the request BEFORE the bad choice (p1):');
    console.log('   ', before && before.slice(0, 300));
    marks.p1 = lines(s.chunks.p1).length;
    marks.p2 = lines(s.chunks.p2).length;
    s.streams.p1.write('move 2');           // Earthquake — DISABLED by the Choice lock
    await settle();
    const after = lastRequest(lines(s.chunks.p1));
    console.log('\n[case1] RE-REQUEST delta (before -> after), field by field:');
    const b = JSON.parse(before.slice('|request|'.length));
    const a = JSON.parse(after.slice('|request|'.length));
    console.log('    top-level keys before:', Object.keys(b).join(','));
    console.log('    top-level keys after :', Object.keys(a).join(','));
    console.log('    update flag after    :', JSON.stringify(a.update));
    for (let i = 0; i < a.active[0].moves.length; i++) {
      const mb = JSON.stringify(b.active[0].moves[i]);
      const ma = JSON.stringify(a.active[0].moves[i]);
      if (mb !== ma) console.log(`    move[${i}] CHANGED\n      before ${mb}\n      after  ${ma}`);
    }
  });

  // ---- CASE 2: a switch into a FAINTED slot --------------------------------
  await run('switch into a FAINTED slot', async (s, marks, settle) => {
    // Kill p1's Aerodactyl with BODY SLAM. (Earthquake was the obvious pick and is WRONG:
    // Aerodactyl is Rock/FLYING, so EQ is a 0x immune no-op and the mon never faints — the
    // fixture would then silently test nothing, which is exactly how a probe lies.)
    // Drive BOTH sides off their OWN requests. Driving only p1 stalls the moment p2 needs a
    // forced replacement (Double-Edge KOs its Snorlax), which is what left the earlier fixture
    // sitting at full HP pretending to test a fainted slot.
    for (let i = 0; i < 60; i++) {
      let wrote = false;
      for (const side of ['p1', 'p2']) {
        const req = lastRequest(lines(s.chunks[side]));
        if (!req) continue;
        const r = JSON.parse(req.slice('|request|'.length));
        if (r.wait) continue;
        if (r.forceSwitch) {
          const bench = (r.side.pokemon || [])
            .map((m, idx) => ({ m, idx }))
            .filter(({ m }) => !m.active && !/ fnt$|^0 /.test(m.condition));
          if (!bench.length) continue;
          s.streams[side].write(`switch ${bench[0].idx + 1}`);
          wrote = true;
          continue;
        }
        s.streams[side].write('move 1');
        wrote = true;
      }
      if (!wrote) break;
      await settle();
      const p1req = lastRequest(lines(s.chunks.p1));
      if (p1req) {
        const r = JSON.parse(p1req.slice('|request|'.length));
        if ((r.side.pokemon || []).some((m) => / fnt$|^0 /.test(m.condition))) break;
      }
    }
    const l1 = lines(s.chunks.p1);
    console.log('\n[case2] p1 roster fainted flags:');
    const req = lastRequest(l1);
    if (req) {
      const r = JSON.parse(req.slice('|request|'.length));
      console.log('   ', (r.side.pokemon || []).map((p) => `${p.ident}:${p.condition}`).join('  '));
    }
    marks.p1 = l1.length;
    marks.p2 = lines(s.chunks.p2).length;
    s.streams.p1.write('switch 1');         // slot 1 = the FAINTED Aerodactyl (post-swap order)
    await settle();
  });

  // ---- CASE 3: a switch into the ACTIVE ------------------------------------
  await run('switch into the ACTIVE mon', async (s, marks, settle) => {
    marks.p1 = lines(s.chunks.p1).length;
    marks.p2 = lines(s.chunks.p2).length;
    s.streams.p1.write('switch 1');         // slot 1 IS the active
    await settle();
  });

  // ---- CASE 4: an OUT-OF-RANGE slot ----------------------------------------
  await run('OUT-OF-RANGE move slot', async (s, marks, settle) => {
    marks.p1 = lines(s.chunks.p1).length;
    marks.p2 = lines(s.chunks.p2).length;
    s.streams.p1.write('move 4');           // Aerodactyl has 4 moves; try 5 below too
    await settle();
    s.streams.p1.write('move 9');
    await settle();
  });

  process.exit(0);
})();
