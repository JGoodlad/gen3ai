// probe_review_batch4_followup_rng.js — review follow-ups:
//  1. KR triple-roll ORDER: Meteor Mash (listed, contact, own 20% self-boost secondary)
//     + King's Rock + Static defender → establish [own secondary] vs [KR] vs [contact
//     proc] order by diffing a no-KR control on the same seed.
//  2. Attract onBeforeMove: |-activate|...Attract then randomChance(1,2) → ~50% cant;
//     sticks while both stay in.
// Run: node src/rust_sim/harness/probe_review_batch4_followup_rng.js
'use strict';
const { mon, run, fmtCalls } = require('./probe_batch4_lib');

function cants(d) { return d.lines.filter((l) => l.includes('|cant|')); }
function grab(d, pat) { return d.lines.filter((l) => l.includes(pat)); }

async function kr() {
  console.log('===== 1. KR ORDER: Meteor Mash + KR into Static =====');
  for (const seed of [[5, 6, 7, 8], [13, 24, 35, 46]]) {
    for (const item of ['kingsrock', '']) {
      const t = [
        [mon('Metagross', ['meteormash'], { ability: 'Clear Body', item, evs: { spe: 252 } })],
        [mon('Electabuzz', ['splash'], { ability: 'Static', evs: { hp: 252 } })],
      ];
      const r = await run(t, seed, Array(3).fill(['move 1', 'move 1']));
      r.perDecision.forEach((d, i) => {
        const seq = d.calls.map((c) => `${c.kind}(${c.args.join(',')})=${c.ret}@${c.site}`).join(' > ');
        console.log(`${item || 'noItem'} seed=${JSON.stringify(seed)} t${i + 1}: ${seq}`);
        console.log(`   boost=${grab(d, '-boost').length} status=${grab(d, '|-status|').length} cant=${cants(d).length}`);
      });
    }
  }
}

async function attract() {
  console.log('\n===== 2. ATTRACT onBeforeMove 1/2 =====');
  // seed [101,55,202,13]: Machamp(M) strength attracts to Wigglytuff CC at t3 (from the
  // main review probe). Then Machamp splashes t4..t9 while attracted — expect
  // |-activate|p1a: Machamp|move: Attract each attempt + randomChance(1,2) → ~50% cant.
  const t = [
    [mon('Machamp', ['strength', 'splash'], { ability: 'Guts', gender: 'M', level: 50 })],
    [mon('Wigglytuff', ['splash'], { ability: 'Cute Charm', gender: 'F', evs: { hp: 252 } })],
  ];
  const choices = [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1'],
    ...Array(6).fill(['move 2', 'move 1'])];
  for (const seed of [[101, 55, 202, 13], [3, 14, 15, 92], [8, 88, 888, 8]]) {
    const r = await run(t, seed, choices, {
      onBoundary: (b) => ({ att: !!b.p1.active[0].volatiles['attract'] }),
    });
    console.log(`--- seed=${JSON.stringify(seed)}`);
    r.perDecision.forEach((d, i) => {
      const act = grab(d, 'move: Attract');
      const rc2 = d.calls.filter((c) => c.kind === 'randomChance' && c.args[0] === 1 && c.args[1] === 2)
        .map((c) => c.ret);
      if (act.length || rc2.length || (r.states[i] && r.states[i].att)) {
        console.log(`t${i + 1}: attract-activate=${act.length} rc(1,2)=${JSON.stringify(rc2)} cant=${JSON.stringify(cants(d))} att=${r.states[i] && r.states[i].att}`);
      }
    });
  }
}

(async () => { await kr(); await attract(); process.exit(0); })()
  .catch((e) => { console.error(e); process.exit(1); });
