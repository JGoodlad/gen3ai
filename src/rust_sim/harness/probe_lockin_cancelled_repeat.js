// PROBE: gen-3 OUTRAGE / THRASH / PETAL DANCE — a CANCELLED repeat (`gen3_lockin_cancelled_repeat_v1`, GIGO audit B-C1).
// The `lockedmove` condition has TWO counters (data/conditions.ts, inherited unchanged by gen 3 through gen 4):
//   duration: 2   — ticked by the residual's `duration--`; 0 ends the volatile (onEnd: confusion iff trueDuration <= 1);
//                   put back to 2 by `onRestart` (a LANDED repeat) and by gen 4's Protect `onTryHit`, only while trueDuration >= 2;
//   trueDuration  — `random(2,4)` rolled ONCE on the cast, ticked by `onResidual` on every residual the volatile survives.
// So a repeat that never lands (an immune target, a full paralysis, ...) leaves `duration` unrefreshed: the lock ENDS at that
// residual — quietly (no confusion) on a 3-turn roll's second turn, with the fatigue confusion on the roll's last turn.
// This prints, per seed, the roll and what the locked user does on turns 1-3:
//   S1  Thrash, then the foe switches in a GHOST on turn 2 (Thrash is Normal: immune, the repeat is cancelled);
//   S2  Thrash into a PROTECT on turn 2 (the repeat is blocked, which REFRESHES a 3-turn lock instead of ending it);
//   S3  Thrash with all repeats landing (the control: 2 or 3 turns, then confusion);
//   S4  the foe's slower Parasect Spores the user AFTER its turn-2 repeat: the residual ticks `duration` to 0 BEFORE
//       `onResidual`'s sleep check, so a 2-turn roll's end still confuses the (now asleep) user.
//   node harness/probe_lockin_cancelled_repeat.js [firstSeed lastSeed]
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream.js'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));
let draws = [];
const oR = PRNG.prototype.random;
PRNG.prototype.random = function (...a) { const r = oR.apply(this, a); draws.push(`random(${a})->${r}`); return r; };

const USER = 'Tauros||none|intimidate|thrash,splash|Hardy|85,85,85,85,85,85|M||||';
const FOE = 'Snorlax||none|sturdy|splash,protect|Hardy|85,85,85,85,85,85|M||||]Gengar||none|levitate|splash|Hardy|85,85,85,85,85,85|M||||';
const FOE_SPORE = 'Parasect||none|effectspore|splash,spore|Hardy|85,85,85,85,85,85|M||||';
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function run(label, seed, foeChoices, nTurns, foeTeam = FOE) {
  draws = [];
  const s = new BattleStream(); const ch = [];
  (async () => { for await (const c of s) ch.push(c); })();
  s.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}\n>player p1 {"name":"P1","team":"${USER}"}\n>player p2 {"name":"P2","team":"${foeTeam}"}`);
  await sleep(120);
  const turns = [];
  for (let i = 0; i < nTurns; i++) {
    const mark = ch.length;
    s.write(`>p1 move 1\n>p2 ${foeChoices[i]}`);
    await sleep(120);
    const seg = ch.slice(mark).filter(x => !x.startsWith('sideupdate')).join('\n');
    turns.push({
      mv: (seg.match(/\|move\|p1a: [^|]+\|([^|]+)/) || [])[1] || '-',
      locked: /\[from\] ?lockedmove/.test(seg),
      conf: /\|-start\|p1a: [^|]*\|confusion/.test(seg),
      immune: /\|-immune\|/.test(seg),
      protect: /\|-activate\|[^|]*\|Protect/.test(seg),
    });
  }
  const rolls = draws.filter(d => d.startsWith('random(2,4)')).join(',') || '-';
  const seedOut = s.battle.prng.getSeed();
  console.log(`${label} seed=${JSON.stringify(seed)} roll=${rolls} -> ` + turns.map((t, i) =>
    `t${i + 1}:${t.mv}${t.locked ? '[locked]' : ''}${t.immune ? '[immune]' : ''}${t.protect ? '[protect]' : ''}${t.conf ? '[CONFUSED]' : ''}`).join(' ') + ` seedOut=${seedOut}`);
}
(async () => {
  const lo = Number(process.argv[2] || 1), hi = Number(process.argv[3] || 12);
  for (let k = lo; k <= hi; k++) {
    const seed = [k, k, k, k];
    await run('S1 ghost-switch ', seed, ['move 1', 'switch 2', 'move 1'], 3);
    await run('S2 protect      ', seed, ['move 1', 'move 2', 'move 1'], 3);
    await run('S3 control      ', seed, ['move 1', 'move 1', 'move 1'], 3);
    await run('S4 spore after  ', seed, ['move 1', 'move 2', 'move 1'], 3, FOE_SPORE);
  }
  process.exit(0);
})();
