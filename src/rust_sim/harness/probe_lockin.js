// PROBE: gen-3 LOCK-IN family — OUTRAGE / PETAL DANCE / THRASH.
//
// SETTLED 2026-08-18 (run it to re-confirm; do not re-derive from source):
//   All three moves are IDENTICAL: `self: {volatileStatus: 'lockedmove'}`, acc 100 (NOT
//   never-miss), contact, protect:1. gen3 resolves the gen4 `lockedmove` condition, which
//   INHERITS the base one but DELETES `onAfterMove` — so the lock is torn down at the
//   RESIDUAL, never right after the move.
//
//   THE STATE is two counters:  duration (starts 2)  +  trueDuration (the random roll)
//     onStart   : trueDuration = random(2,4)  ∈ {2,3};  duration = 2
//     onRestart : if (trueDuration >= 2) duration = 2          [re-application refresh]
//     residual  : duration--;  if 0 -> onEnd + REMOVE (skip onResidual)
//                 else onResidual: if (status === 'slp') DELETE (no onEnd); trueDuration--
//     onEnd     : if (trueDuration > 1) return;  target.addVolatile('confusion')
//     onLockMove: return the stored move id
//
//   DRAWS the family adds, in per-turn order:
//     FIRST successful hit : random(2,4)  — INSIDE addVolatile's onStart, i.e. in
//                            runMoveEffects/selfDrops, AFTER the damage roll, BEFORE the
//                            endTurn Quick Claw. NOT at selection/cast.
//     locked turns 2..N    : NOTHING extra (the acc/crit/damage chain only)
//     the turn duration→0  : random(2,6) — the CONFUSION duration, at the RESIDUAL
//   Nothing else in the family draws. PP is deducted ONLY on the first cast.
//
//   1 LOCK LENGTH   random(2,4) -> 2  =>  2 attacking turns;  -> 3  =>  3 attacking turns.
//   2 REQUEST       {"moves":[{"move":"Outrage","id":"outrage"}],"trapped":true}
//                   — a SINGLE entry with ONLY {move,id} (no pp/maxpp/target/disabled) +
//                   FIRM trapped:true. Byte-shape identical to Hyper Beam's Recharge
//                   request except the name/id are the locked move's.
//   3 CONFUSION     ALWAYS at the duration→0 residual, PROVIDED trueDuration<=1 then:
//                   `|-start|<user>|confusion|[fatigue]` (note the [fatigue] tag) + one
//                   random(2,6). The lockedmove volatile itself emits NOTHING on removal.
//   4 INTERRUPTS    miss / any onBeforeMove cant  -> ENDS at that residual (no refresh);
//                   confusion iff trueDuration<=1 at that moment
//                   target FAINTS                 -> CONTINUES (across the replacement)
//                   target IMMUNE (Thrash->Ghost) -> NO LOCK EVER (accuracy drawn, -immune)
//                   user ASLEEP                   -> DELETED at the residual, NO confusion
//                                                    — UNLESS that turn is duration→0, where
//                                                    the end-branch wins: confusion FIRES
//                   PROTECT/DETECT                -> CONTINUES: gen4's protect onTryHit
//                                                    explicitly refreshes duration to 2
//                                                    (guarded by trueDuration >= 2)
//   5 SWITCHING     trapped:true; `switch N` -> `|error|[Invalid choice] Can't switch: The
//                   active Pokémon is trapped`, NO re-request. A PHAZE drag-out clears the
//                   lock (clearVolatile) with NO confusion and NO emission.
//
// Run: node /tmp/probe_lockin_final.js [q1|q2|q3|q4|q5|edge|tokens|all]
const path = require('path');
const PS = '/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream.js'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));
const { Battle } = require(path.join(PS, 'dist/sim/battle.js'));

// --- instrumentation: PRNG.prototype.random is the SOLE path to rng.next(); wrapping
//     randomChance/sample too would DOUBLE-COUNT (a sibling probe's lesson).
let draws = [], adds = [], TRACE = false;
const oRandom = PRNG.prototype.random;
PRNG.prototype.random = function (...a) {
  const r = oRandom.apply(this, a);
  draws.push({ s: `random(${a.join(',')})->${r}`, site: TRACE ? site(2) : '' });
  return r;
};
const oAdd = Battle.prototype.add;                       // line attribution
Battle.prototype.add = function (...p) {
  if (TRACE) adds.push({ line: p.map(x => (x && x.name) ? x.name : String(x)).join('|'), site: site(2) });
  return oAdd.apply(this, p);
};
function site(skip) {
  const out = [];
  for (const l of new Error().stack.split('\n').slice(skip)) {
    const m = l.match(/at ([^ ]+) \(.*\/(dist\/(?:sim|data)[^)]*)\)/);
    if (m) out.push(`${m[1]}@${m[2].split('/').pop()}`);
    if (out.length >= 4) break;
  }
  return out.join(' <- ');
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function run(label, p1, p2, script, opt = {}) {
  draws = []; adds = []; TRACE = !!opt.trace;
  const s = new BattleStream(); const ch = [];
  (async () => { for await (const c of s) ch.push(c); })();
  s.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(opt.seed || [9, 9, 9, 9])}}` +
    `\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await sleep(110);
  if (label) console.log(`\n######## ${label}`);
  const steps = [];
  for (const cmd of script) {
    const dm = draws.length, cm = ch.length;
    s.write(cmd); await sleep(110);
    const lines = [], reqs = [], errs = [];
    for (const c of ch.slice(cm)) {
      const cl = c.split('\n');
      if (cl[0] === 'sideupdate') {                       // DROP sideupdate from the log
        for (const l of cl.slice(2)) {
          if (l.startsWith('|request|')) reqs.push(`${cl[1]} ${l.slice(9)}`);
          if (l.startsWith('|error|')) errs.push(`${cl[1]} ${l}`);
        }
        continue;
      }
      for (const l of cl) if (l.startsWith('|') && !l.startsWith('|split|') && !l.startsWith('|t:|')) lines.push(l);
    }
    const mon = s.battle.p1.active[0], v = mon.volatiles['lockedmove'];
    const st = { cmd, lines, reqs, errs, draws: draws.slice(dm), adds: adds.slice(),
      lm: v ? `{duration:${v.duration}, trueDuration:${v.trueDuration}, move:${v.move}}` : 'ABSENT',
      conf: mon.volatiles['confusion'] ? mon.volatiles['confusion'].time : '-',
      status: mon.status || '-', pp0: mon.moveSlots[0].pp };
    steps.push(st);
    if (label) {
      console.log(`-- ${cmd.replace(/\n/g, ' ; ')}`);
      for (const l of lines) if (!opt.quiet || /move\|p1a|-start|-end|-miss|-immune|-activate|cant|faint|switch|drag|-status/.test(l)) console.log('     ' + l);
      for (const e of errs) console.log('     ' + e);
      if (opt.req) for (const r of reqs) console.log('     REQ ' + r.replace(/,"side":\{.*/, ''));
      console.log('     DRAWS: ' + (st.draws.map(d => d.s + (d.site ? ` @${d.site}` : '')).join(opt.trace ? '\n            ' : '  ') || '(none)'));
      console.log(`     lockedmove=${st.lm}  confusion=${st.conf}  status=${st.status}  pp[0]=${st.pp0}`);
    }
  }
  return steps;
}

// --- fixtures ---------------------------------------------------------------
const U = m => `Kingdra||Leftovers|SwiftSwim|${m},splash,recover,protect|Hardy|85,85,85,85,85,85|M||||`;
const U2 = m => U(m) + `],Blissey||Leftovers|NaturalCure|splash|Hardy|85,85,85,85,85,85|F||||`;
const WALL = `Blissey||Leftovers|NaturalCure|splash,protect,spore,rockslide|Hardy|85,85,85,85,85,85|F||||`;
const WALL_BP = `Blissey||BrightPowder|NaturalCure|splash,protect|Hardy|85,85,85,85,85,85|F||||`;
const WALL_FO = WALL + `],Snorlax||Leftovers|Immunity|fakeout,splash|Hardy|85,85,85,85,85,85|M||||`;
const GHOST = `Gengar||Leftovers|Levitate|splash|Hardy|85,85,85,85,85,85|M||||`;   // Normal-IMMUNE (Thrash)
const FRAIL = `Magikarp||Leftovers|SwiftSwim|splash|Hardy|85,85,85,85,85,85|M|||1|],Blissey||Leftovers|NaturalCure|splash|Hardy|85,85,85,85,85,85|F||||`;
const PHAZER = `Skarmory||Leftovers|Keeneye|whirlwind,splash|Hardy|85,85,85,85,85,85|F||||`;
const T = n => Array(n).fill('>p1 move 1\n>p2 move 1');

(async () => {
  const only = process.argv[2] || 'all', want = k => only === 'all' || only === k;

  if (want('q1')) {                                        // ---- Q1 duration + draw site
    for (const mv of ['outrage', 'petaldance', 'thrash'])
      await run(`Q1 ${mv} — 5 turns @seed 9,9,9,9`, U(mv), WALL, T(5), { trace: true, quiet: true });
    console.log('\n######## Q1 SEED SWEEP (outrage): random(2,4) -> attacking turns');
    const tally = {};
    for (let i = 0; i < 24; i++) {
      const st = await run('', U('outrage'), WALL, T(4), { seed: [i * 7 + 1, 3, 5, 7] });
      const roll = st[0].draws.map(d => d.s).find(s => s.startsWith('random(2,4)'));
      const conf = st.findIndex(s => s.lines.some(l => /-start\|p1a.*confusion/.test(l)));
      tally[`${roll} -> confusion at turn ${conf + 1}`] = (tally[`${roll} -> confusion at turn ${conf + 1}`] || 0) + 1;
    }
    console.log(JSON.stringify(tally, null, 1));
  }

  if (want('q2'))                                          // ---- Q2 the locked request
    await run('Q2 the |request| during the lock', U2('outrage'), WALL, T(4), { req: true, quiet: true });

  if (want('q3'))                                          // ---- Q3 end-of-lock confusion
    for (const seed of [[9, 9, 9, 9], [1, 3, 5, 7]])
      await run(`Q3 end-of-lock confusion @seed ${seed}`, U('outrage'), WALL, T(4), { trace: true, quiet: true, seed });

  if (want('q4')) {                                        // ---- Q4 interruptions
    for (let i = 0; i < 40; i++) {                         // (a) MISS mid-lock
      const st = await run('', U('outrage'), WALL_BP, T(4), { seed: [i * 13 + 3, 5, 7, 11] });
      if (st.findIndex(s => s.lines.some(l => /-miss/.test(l))) === 1) {
        await run(`Q4a MISS mid-lock @seed ${i * 13 + 3},5,7,11`, U('outrage'), WALL_BP, T(4), { seed: [i * 13 + 3, 5, 7, 11], quiet: true }); break;
      }
    }
    await run('Q4b TARGET FAINTS (L1 Magikarp) — the lock CONTINUES', U('outrage'), FRAIL,
      ['>p1 move 1\n>p2 move 1', '>p2 switch 2', '>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 1'], { req: true, quiet: true });
    await run('Q4c IMMUNE (Thrash -> Ghost) — NO lock is ever created', U('thrash'), GHOST, T(3), { req: true, quiet: true });
    await run('Q4d USER ASLEEP mid-lock (Spore on turn 2)', U('outrage'), WALL,
      ['>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 3', '>p1 move 1\n>p2 move 1'], { req: true, quiet: true });
    await run('Q4d2 USER ASLEEP on the FINAL locked turn (tD=2) — confusion STILL fires', U('outrage'), WALL,
      ['>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 3', '>p1 move 1\n>p2 move 1'], { seed: [16, 5, 7, 11], quiet: true });
    await run('Q4e USER FLINCHES mid-lock (Fake Out from a fresh switch-in)', U('outrage'), WALL_FO,
      ['>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 switch 2', '>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 2'], { quiet: true });
    await run('Q4f TARGET PROTECTS mid-lock — the lock CONTINUES', U('outrage'), WALL,
      ['>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 2', '>p1 move 1\n>p2 move 1'], { quiet: true });
  }

  if (want('q5')) {                                        // ---- Q5 switching / trapping
    await run('Q5a switch attempt WHILE LOCKED (draw-free reject, no re-request)', U2('outrage'), WALL,
      ['>p1 move 1\n>p2 move 1', '>p1 switch 2', '>p1 move 1\n>p2 move 1'], { req: false, quiet: true });
    await run('Q5b PHAZE (Whirlwind) drags the locked mon out — the lock is GONE', U2('outrage'), PHAZER,
      ['>p1 move 1\n>p2 move 2', '>p1 move 1\n>p2 move 1', '>p1 switch 2\n>p2 move 2'], { req: true, quiet: true });
  }

  if (want('edge')) {                                      // ---- the duration-END branch
    for (let i = 0; i < 60; i++) {
      const st = await run('', U('outrage'), WALL_BP, T(2), { seed: [i * 11 + 1, 5, 7, 11] });
      if (st[0].draws.some(d => d.s === 'random(2,4)->2') && st[1].lines.some(l => l.includes('-miss'))) {
        await run(`EDGE MISS on the FINAL locked turn (tD=2) — confusion STILL fires @seed ${i * 11 + 1},5,7,11`,
          U('outrage'), WALL_BP, T(2), { seed: [i * 11 + 1, 5, 7, 11], quiet: true }); break;
      }
    }
    for (let i = 0; i < 60; i++) {
      const sc = ['>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 2'];
      const st = await run('', U('outrage'), WALL, sc, { seed: [i * 11 + 1, 5, 7, 11] });
      if (st[0].draws.some(d => d.s === 'random(2,4)->2') && st[1].lines.some(l => /-activate\|p2a.*Protect/.test(l))) {
        await run(`EDGE PROTECT on the FINAL locked turn (tD=2) — no refresh (guard tD>=2), confusion FIRES @seed ${i * 11 + 1},5,7,11`,
          U('outrage'), WALL, sc, { seed: [i * 11 + 1, 5, 7, 11], quiet: true }); break;
      }
    }
  }

  if (want('tokens')) {                                    // ---- accepted / rejected wire tokens
    console.log('\n######## TOKENS accepted by the sim for a LOCKED mon');
    for (const tok of ['move 1', 'move outrage', 'move 2', 'move splash', 'switch 2', 'move 5', 'default', 'pass']) {
      const st = await run('', U2('outrage'), WALL, ['>p1 move 1\n>p2 move 1', `>p1 ${tok}`]);
      const s2 = st[1];
      console.log(`  >p1 ${tok.padEnd(13)} -> ${s2.errs.length ? s2.errs.join(' ') : (s2.lines.length ? 'ACCEPTED (turn ran)' : 'ACCEPTED (boundary held open for p2)')}`);
    }
  }
  process.exit(0);
})();
