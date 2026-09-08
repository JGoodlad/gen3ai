// GOLDEN GENERATOR — the FOCUS ENERGY differential vector (`gen3_focus_energy_move_v1`, ROUND 59).
//
// Focus Energy is NEVER-MISS and DRAW-FREE, so a single seed would in principle settle its own
// emission. The sweep is here for the half that is NOT draw-free: the **crit consequence**. The
// move exists to shift the crit ratio from stage 0 (1/16) to stage +2 (1/4), and that only shows
// up across seeds. So each scenario casts Focus Energy (or a control) and then attacks, and the
// vector records the sim's post-decision seed and the exact lines INCLUDING `|-crit|`.
//
//   node src/rust_sim/harness/gen_focus_energy_golden.js
//     -> tests/vectors/focus_energy_golden.txt
//   then `cargo test --test focus_energy_golden_test` re-pins the port against it.
//
// 🚨 The `splash` CONTROL scenario is the SAME board with the cast replaced by a no-op, so the
// crit-rate difference between the two scenario families is attributable to the volatile alone.
// A flag that is set but never READ would leave the two identical — and would pass any test that
// only asserted the `-start` line.
'use strict';
const fs = require('fs');
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream.js'));

const N_SEEDS = parseInt(process.env.CF_SEEDS || '120', 10);
const OUT = process.env.CF_OUT || path.resolve(__dirname, '../tests/vectors/focus_energy_golden.txt');

// Machamp so the attack (Tackle, slot 2) actually lands damage worth critting.
const GENGAR = (m) =>
  `|Machamp|Leftovers|Guts|${m},tackle,splash,splash|Hardy|85,85,85,85,85,85|M||||`;
const FOES = {
  inert: '|Snorlax|Leftovers|Immunity|splash,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||',
  snatcher: '|Snorlax|Leftovers|Immunity|snatch,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||',
};

// One decision: p1 casts from slot 1.
const T1 = ['>p1 move 1\n>p2 move 1'];
// Two decisions: p2 raises its blocker on turn 1 (p1 splashes from slot 2), cast on turn 2.
const T2 = ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 2'];
// Two decisions, both casts — reaches the ALREADY-CONFUSED branch on the second.
// Cast on turn 1, then ATTACK on turn 2 — the attack is what exposes the crit consequence.
const TCAST_ATTACK = ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1'];
// Two casts, then attack — the second cast is the already-up `[still]`+`-fail` branch.
const TWICE_ATTACK = ['>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1'];

const SCENARIOS = [];
// The CONTROL: the same board with the cast replaced by a Splash, so the crit-rate difference is
// attributable to the VOLATILE and nothing else.
SCENARIOS.push({ name: 'control-nocast-then-attack', move: 'splash', foe: 'inert', script: TCAST_ATTACK });
SCENARIOS.push({ name: 'focusenergy-then-attack', move: 'focusenergy', foe: 'inert', script: TCAST_ATTACK });
SCENARIOS.push({ name: 'focusenergy-twice-then-attack', move: 'focusenergy', foe: 'inert', script: TWICE_ATTACK });
SCENARIOS.push({ name: 'focusenergy-snatched', move: 'focusenergy', foe: 'snatcher', script: T1 });

// The gated line types — the same shape `protocol_test.rs` filters to. `|t:|` is wall-clock and
// poke-env ignores it; `debug`/`error` are dropped from BOTH sides everywhere in this project.
const KEEP = /^\|(move|-start|-end|-fail|-immune|-activate|-miss|-crit|-damage|-heal|turn)\|/;

// 🚨 READ THE **OMNISCIENT** STREAM, never the raw `BattleStream` chunks.
// The raw stream carries `|split|pN` markers: each HP-revealing event is emitted TWICE — the
// owner's exact `x/y` line and the spectator's percentage line. A naive chunk join therefore
// records every `-damage` / `-heal` twice, and the resulting golden accuses a CORRECT port of
// dropping lines. `getPlayerStreams(...).omniscient` is the referee view that resolves the split,
// and it is what `harness/gen_protocol_capture.js` has always used.
async function runOne(p1, p2, script, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const all = [];
  (async () => {
    for await (const ch of streams.omniscient) {
      for (const l of ch.split('\n')) all.push(l);
    }
  })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[${seed.join(',')}]}`);
  streams.omniscient.write(`>player p1 {"name":"P1","team":"${p1}"}`);
  streams.omniscient.write(`>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise((r) => setTimeout(r, 120));
  const before = stream.battle.prng.getSeed();
  for (const c of script) {
    for (const part of c.split('\n')) streams.omniscient.write(part);
    await new Promise((r) => setTimeout(r, 120));
  }
  const after = stream.battle.prng.getSeed();
  const i = all.indexOf('|turn|1');
  if (i < 0) throw new Error('no |turn|1 framing marker in the omniscient stream');
  const lines = all.slice(i + 1).filter((l) => KEEP.test(l));
  return { before, after, lines };
}

(async () => {
  const rows = [];
  let n = 0;
  for (const sc of SCENARIOS) {
    for (let k = 0; k < N_SEEDS; k++) {
      // A spread of start seeds; the ACCURACY branch varies with them, which is the point.
      const seed = [1000 + k * 7, 2000 + k * 13, 3000 + k * 29, 4000 + k * 51];
      const r = await runOne(GENGAR(sc.move), FOES[sc.foe], sc.script, seed);
      rows.push({
        name: sc.name,
        move: sc.move,
        p1: GENGAR(sc.move),
        p2: FOES[sc.foe],
        script: sc.script,
        seedBefore: r.before,
        seedAfter: r.after,
        lines: r.lines,
      });
      n++;
    }
  }
  // A DISCLOSED coverage census — the vector must actually contain both accuracy branches, or
  // the miss path it exists to cover is untested and nobody would know.
  const critRows = rows.filter((r) => r.lines.some((l) => l.startsWith('|-crit|')));
  const feCrit = rows.filter((r) => r.name === 'focusenergy-then-attack' && r.lines.some((l) => l.startsWith('|-crit|'))).length;
  const feTotal = rows.filter((r) => r.name === 'focusenergy-then-attack').length;
  const ctlCrit = rows.filter((r) => r.name === 'control-nocast-then-attack' && r.lines.some((l) => l.startsWith('|-crit|'))).length;
  const ctlTotal = rows.filter((r) => r.name === 'control-nocast-then-attack').length;
  const startRows = rows.filter((r) => r.lines.some((l) => l.includes('|move: Focus Energy')));
  const header = [
    '# FOCUS ENERGY GOLDEN (gen3_focus_energy_move_v1)',
    `# generated by harness/gen_focus_energy_golden.js  seeds/scenario=${N_SEEDS}`,
    `# rows=${n} scenarios=${SCENARIOS.length}`,
    `# COVERAGE: |-crit| rows = ${critRows.length}; `+
      `crit rate WITH Focus Energy = ${feCrit}/${feTotal}, CONTROL = ${ctlCrit}/${ctlTotal}; `+
      `|-start|…|move: Focus Energy rows = ${startRows.length}`,
    '# Each row is one JSON object: the board, the script, the sim SEED_BEFORE/SEED_AFTER and the',
    '# filtered protocol lines. The port is seeded at SEED_BEFORE (the pre-first-decision state).',
    '',
  ].join('\n');
  fs.writeFileSync(OUT, header + rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
  console.log(`wrote ${OUT}`);
  console.log(`rows=${n}  crit-rows=${critRows.length}  FE ${feCrit}/${feTotal}  control ${ctlCrit}/${ctlTotal}`);
  // A DISCLOSED, REFUSED precondition: if the two arms crit at the same rate the vector cannot
  // demonstrate the volatile does anything, and every row would still compare byte-equal.
  if (feCrit <= ctlCrit) {
    console.error(`REFUSING: Focus Energy (${feCrit}/${feTotal}) did not out-crit the control `
      + `(${ctlCrit}/${ctlTotal}) — the vector cannot show the volatile is READ.`);
    process.exit(1);
  }
})();
