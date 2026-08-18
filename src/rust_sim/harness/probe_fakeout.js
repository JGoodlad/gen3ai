// PROBE: gen-3 FAKE OUT — the first-turn gate, the failure form, the draw model, priority, interactions.
//
// SETTLED 2026-08-18 (run it to re-confirm; do not re-derive from source):
//
//   DEX (resolved Dex.mod('gen3')): bp 40 PHYSICAL Normal, accuracy 100 (NOT never-miss),
//     **priority +1** (NOT the modern +3), pp 10 (maxpp 16), target normal,
//     flags {protect, mirror, metronome}  ← NO `contact` in gen3 (gen4 adds it),
//     secondaries [{chance: 100, volatileStatus: 'flinch'}].
//
//   THE GATE is `pokemon.activeMoveActions > 1` — a per-mon MOVE-ACTION counter, NOT
//     activeTurns and NOT "the turn you switched in". `runMove` does `activeMoveActions++`
//     at its very TOP (battle-actions.js:203), before BeforeMove/PP/announce; `switchIn`
//     resets it to 0 (:138). So:
//       * used on the mon's FIRST move action  -> 1 > 1 is false -> WORKS
//       * a turn spent CANT-ed (flinch/slp/par) still calls runMove -> BURNS the gate
//       * a turn whose action was CANCELLED (gen-3 faint-cancels-all) does NOT burn it
//         (activeTurns advanced, activeMoveActions did not -> Fake Out still WORKS)
//       * switching OUT and back IN **RESETS** it; so does a Roar/Whirlwind DRAG and a
//         Baton Pass entry (all three route through `switchIn`)
//       * a Sleep Talk-CALLED Fake Out reads the SLEEP TALK action's count (useMove does
//         not bump it), so it WORKS iff the Sleep Talk was the mon's first move action.
//
//   THE FAILURE FORM: `|move|<user>|Fake Out|<target>` then `|-hint|Fake Out only works on
//     your first turn out.` — NO `[still]` attr, NO `-fail`. The hint is NOT deduped (it
//     is `this.hint(text)` with `once` falsy), so three consecutive fails emit three hints.
//     The failed attempt draws **ZERO** and still costs **1 PP** (2 under a Pressure foe —
//     the DeductPP event precedes the Try gate) and still sets `lastMove = fakeout`.
//     The gate sits at gen3 `tryMoveHit`'s `singleEvent("Try", move)` (scripts.js:244),
//     which is AFTER the announce/PP/lastMove and BEFORE Invulnerability, the type-immunity
//     report, the accuracy roll and TryHit/Protect — so a FAILED Fake Out into a Ghost shows
//     only the hint (no `-immune`), and a failed Fake Out fires no in-tryMoveHit Update.
//
//   THE DRAW MODEL on a hit, in order — FOUR draws (one MORE than a plain 40-BP move):
//       randomChance(100,100)  accuracy   (acc 100 but NOT never-miss -> ALWAYS drawn)
//       randomChance(1,16)     crit
//       random(16)             damage roll
//       random(100)            **the flinch SECONDARY roll — chance 100 STILL ROLLS**
//     (`secondaries()` draws `this.battle.random(100)` unconditionally per secondary,
//      battle-actions.js; `typeof chance === 'undefined'` is the only skip and 100 is not
//      undefined). The secondary roll fires even when the hit FAINTS the target and even
//      when the hit is absorbed by a SUBSTITUTE; it does NOT fire on a miss / type-immune /
//      Protect-blocked / gate-failed attempt.
//     A LANDED Fake Out additionally fires the in-tryMoveHit `eachEvent('Update')`
//     (gen3 scripts.js:414) — visible as one extra tie-shuffle at a speed tie.
//     ⚠ Its +1 priority normally REMOVES the action-order tie-shuffle (different priority
//       brackets), so a tied mirror reads: Splash 6 shuffles / FAILED Fake Out 5 /
//       LANDED Fake Out 6 / landed Tackle 7.
//
//   PRIORITY +1 confirmed behaviourally: a 266-spe user's Fake Out resolves BEFORE a
//     359-spe foe's Tackle; against a foe's Quick Attack (also +1) the FASTER mon wins.
//
//   INTERACTIONS
//     SUBSTITUTE  -> `|-activate|<t>|Substitute|[damage]`; the secondary random(100) is
//                    STILL DRAWN, the flinch is NOT applied (the standard sub-suppression).
//     GHOST       -> accuracy drawn, then `|-immune|<t>`; NO crit/damage/secondary.
//     PROTECT     -> accuracy drawn, then `|-activate|<t>|Protect` (flags.protect).
//     INNER FOCUS -> the secondary random(100) is STILL DRAWN, the flinch is blocked at the
//                    apply (the port's existing model; identical draw count).
//     MISS (only reachable via evasion/BrightPowder) -> `|move|…|[miss]` + `|-miss|<u>|<t>`,
//                    accuracy roll only.
//     TARGET ALREADY MOVED -> `addVolatile('flinch')` IS still called and RETURNS TRUE
//                    (no moved-this-turn gate); it simply has no effect and expires at that
//                    turn's residual. It DOES join the residual duration-handler tie group.
//
// Run: node probe_fakeout.js
const path = require('path');
const PS = '/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream.js'));
const { Dex } = require(path.join(PS, 'dist/sim/dex.js'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));
const { Pokemon } = require(path.join(PS, 'dist/sim/pokemon.js'));

const d = Dex.mod('gen3').moves.get('fakeout');
console.log('dex(gen3): bp', d.basePower, 'acc', d.accuracy, 'cat', d.category, 'type', d.type,
  'priority', d.priority, 'pp', d.pp, 'target', d.target,
  '\n           flags', JSON.stringify(d.flags), 'secondaries', JSON.stringify(d.secondaries));
console.log('dex(gen3) onTry:', d.onTry.toString().replace(/\s+/g, ' '));

// --- instrumentation -------------------------------------------------------
// NOTE `randomChance(n,d)` internally calls `random(d)`, so the log shows BOTH lines for
// one real PRNG call. `random(16,)` (a trailing comma) is `battle.random(16)` = 2 args.
let draws = [], vols = [];
for (const m of ['random', 'randomChance', 'sample']) {
  const o = PRNG.prototype[m];
  PRNG.prototype[m] = function (...a) { const r = o.apply(this, a); draws.push(`${m}(${a})->${r}`); return r; };
}
const av = Pokemon.prototype.addVolatile;
Pokemon.prototype.addVolatile = function (status, source, ...rest) {
  const r = av.call(this, status, source, ...rest);
  const id = typeof status === 'string' ? status : status.id;
  if (id === 'flinch') vols.push(`addVolatile(flinch) on ${this.name} -> ${r} (movedThisTurn=${!!this.moveThisTurn})`);
  return r;
};

const LINE = /^\|(-start|-end|-fail|-immune|-activate|-damage|-crit|-supereffective|-resisted|-miss|-hint|-singleturn|-status|move|cant|switch|drag|turn|faint)\|/;

async function run(label, p1, p2, script, { seed = '[9,9,9,9]', quiet = false } = {}) {
  draws = []; vols = [];
  const s = new BattleStream();
  const ch = [];
  (async () => { for await (const c of s) ch.push(c); })();
  s.write(`>start {"formatid":"gen3customgame","seed":${seed}}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r => setTimeout(r, 200));
  const marks = [];
  for (const c of script) {
    const b = draws.length, vb = vols.length;
    s.write(c);
    await new Promise(r => setTimeout(r, 200));
    marks.push({ cmd: c.replace(/\n/g, ' | '), n: draws.length - b, d: draws.slice(b), v: vols.slice(vb) });
  }
  await new Promise(r => setTimeout(r, 200));
  // DROP the `sideupdate` chunks — that leaves the chronological OMNISCIENT log.
  const omni = ch.filter(c => !c.startsWith('sideupdate')).join('\n').split('\n').filter(l => LINE.test(l));
  const out = omni.slice(omni.indexOf('|turn|1'));
  if (!quiet) {
    console.log(`\n===== ${label}\n  LOG:\n    ` + out.join('\n    '));
    for (const m of marks) {
      console.log(`  [${m.cmd}] draws(${m.n}): ${m.d.join('  ') || '(none)'}`);
      for (const v of m.v) console.log(`      ${v}`);
    }
    for (const side of ['p1', 'p2']) {
      const a = s.battle[side].active[0];
      if (a) console.log(`  ${side} ${a.name}: activeMoveActions=${a.activeMoveActions} activeTurns=${a.activeTurns} ` +
        `lastMove=${a.lastMove && a.lastMove.id} ` + a.moveSlots.map(ms => `${ms.id}=${ms.pp}/${ms.maxpp}`).join(' '));
    }
  }
  s.destroy();
  return { log: out, marks };
}

// packed: NICK|SPECIES|ITEM|ABILITY|MOVES|NATURE|EVS|GENDER|IVS|SHINY|LEVEL|HAPPINESS
const mon = (species, moves, { ability = 'Limber', evs = '85,85,85,85,85,85', item = '', level = '' } = {}) =>
  `${species}||${item}|${ability}|${moves.join(',')}|Hardy|${evs}|M|||${level}|`;
const team = (...m) => m.join(']');
const SLOW = '85,85,85,85,85,0', FAST = '85,85,85,85,85,252', TIE = '85,85,85,85,85,85';
const FO = (evs) => mon('Persian', ['fakeout', 'splash', 'tackle', 'scratch'], { evs });

(async () => {
  // ============================================================ 1. THE GATE ==
  await run('A: the LEAD Fake Outs on turn 1 (its FIRST move action) -> WORKS',
    team(FO(FAST)), team(mon('Snorlax', ['tackle', 'splash'], { ability: 'Immunity', evs: SLOW })),
    ['>p1 move 1\n>p2 move 1']);

  await run('B: turn 2, no switch -> FAILS (the emission form + zero draws + PP burn)',
    team(FO(FAST)), team(mon('Snorlax', ['tackle', 'splash'], { ability: 'Immunity', evs: SLOW })),
    ['>p1 move 2\n>p2 move 2', '>p1 move 1\n>p2 move 2']);

  await run('B2: three consecutive fails — the |-hint| is NOT deduped, PP burns each time',
    team(FO(FAST)), team(mon('Blissey', ['splash', 'tackle'], { ability: 'Immunity', evs: SLOW })),
    ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 1']);

  await run('CANT: a turn spent FLINCHED still burns the gate (runMove ran) -> turn 2 FAILS',
    team(FO(SLOW)), team(mon('Jolteon', ['fakeout', 'splash', 'tackle'], { evs: FAST })),
    ['>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 2']);

  // THE DISCRIMINATOR: activeTurns advances, activeMoveActions does NOT.
  await run('CANCEL: p1 Splash CANCELLED by a faster Self-Destruct -> turn 2 Fake Out WORKS',
    team(FO(SLOW)),
    team(mon('Electrode', ['selfdestruct', 'splash'], { evs: FAST }),
      mon('Blissey', ['splash', 'tackle'], { ability: 'Immunity', evs: SLOW })),
    ['>p1 move 2\n>p2 move 1', '>p2 switch 2', '>p1 move 1\n>p2 move 1']);

  await run('C: switch OUT then back IN -> the gate RESETS',
    team(FO(FAST), mon('Snorlax', ['splash', 'tackle'], { ability: 'Immunity', evs: SLOW })),
    team(mon('Blissey', ['splash', 'tackle'], { ability: 'Immunity', evs: SLOW })),
    ['>p1 move 2\n>p2 move 1', '>p1 switch 2\n>p2 move 1', '>p1 switch 2\n>p2 move 1', '>p1 move 1\n>p2 move 1']);

  await run('D: DRAGGED in by Whirlwind -> the gate RESETS',
    team(mon('Snorlax', ['splash', 'tackle'], { ability: 'Immunity', evs: SLOW }), FO(FAST)),
    team(mon('Blissey', ['whirlwind', 'splash'], { ability: 'Immunity', evs: SLOW })),
    ['>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 2']);

  await run('BP: entered by a BATON PASS -> the gate RESETS',
    team(mon('Smeargle', ['batonpass', 'splash'], { evs: FAST }), FO(FAST)),
    team(mon('Blissey', ['splash', 'tackle'], { ability: 'Immunity', evs: SLOW })),
    ['>p1 move 1\n>p2 move 1', '>p1 switch 2', '>p1 move 1\n>p2 move 1']);

  // ====================================================== 2. THE FAIL FORM ===
  await run('FAILGHOST: a FAILED Fake Out into a Ghost — the gate PRECEDES the immunity report',
    team(FO(FAST)), team(mon('Gengar', ['splash', 'tackle'], { ability: 'Levitate', evs: SLOW })),
    ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 1']);

  await run('PRESSURE: a FAILED Fake Out into a Pressure foe still pays the -2 PP (16 -> 14)',
    team(FO(FAST)), team(mon('Zapdos', ['splash', 'tackle'], { ability: 'Pressure', evs: SLOW })),
    ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 1']);

  // ===================================================== 3. THE DRAW MODEL ===
  await run('CONTROL: Scratch (40 BP, no secondary) on the A board -> acc/crit/dmg only',
    team(FO(FAST)), team(mon('Snorlax', ['tackle', 'splash'], { ability: 'Immunity', evs: SLOW })),
    ['>p1 move 4\n>p2 move 1']);

  await run('KO: the hit FAINTS the target — the flinch secondary STILL rolls',
    team(FO(FAST)),
    team(mon('Caterpie', ['splash', 'tackle'], { evs: SLOW, level: '1' }),
      mon('Blissey', ['splash'], { ability: 'Immunity', evs: SLOW })),
    ['>p1 move 1\n>p2 move 1', '>p2 switch 2']);

  // The speed-TIE mirror: Splash 6 shuffles / FAILED 5 (its +1 priority removes the
  // action-order tie) / LANDED Fake Out 6 (-1 action-order, +1 in-tryMoveHit Update) /
  // landed Tackle 7.
  const TP = team(FO(TIE));
  await run('TIE-CTRL: tied mirror, Splash vs Splash (the shuffle baseline)', TP, TP,
    ['>p1 move 2\n>p2 move 2', '>p1 move 2\n>p2 move 2']);
  await run('TIE-FAIL: tied mirror, a FAILED Fake Out fires NO in-tryMoveHit Update', TP, TP,
    ['>p1 move 2\n>p2 move 2', '>p1 move 1\n>p2 move 2']);
  await run('TIE-LAND: tied mirror, a landed Tackle DOES fire it', TP, TP,
    ['>p1 move 2\n>p2 move 2', '>p1 move 3\n>p2 move 2']);

  // ======================================================== 4. PRIORITY ======
  await run('E: PRIORITY +1 — a 266-spe user outspeeds a 359-spe foe\'s Tackle',
    team(FO(SLOW)), team(mon('Jolteon', ['tackle', 'splash'], { ability: 'Immunity', evs: FAST })),
    ['>p1 move 1\n>p2 move 1']);

  await run('E2: same bracket — a FASTER foe\'s Quick Attack (+1) resolves first',
    team(FO(SLOW)), team(mon('Jolteon', ['quickattack', 'splash'], { ability: 'Immunity', evs: FAST })),
    ['>p1 move 1\n>p2 move 1']);

  // ==================================================== 5. INTERACTIONS ======
  await run('SUB: into a SUBSTITUTE — the secondary rolls, the flinch does not apply',
    team(mon('Snorlax', ['splash', 'tackle'], { ability: 'Immunity', evs: SLOW }), FO(FAST)),
    team(mon('Blissey', ['substitute', 'splash', 'tackle'], { ability: 'Immunity', evs: SLOW })),
    ['>p1 move 1\n>p2 move 1', '>p1 switch 2\n>p2 move 2', '>p1 move 1\n>p2 move 3']);

  await run('GHOST: into a GHOST — accuracy only, then -immune',
    team(FO(FAST)), team(mon('Gengar', ['splash', 'tackle'], { ability: 'Levitate', evs: SLOW })),
    ['>p1 move 1\n>p2 move 1']);

  await run('PROTECT: blocked by Protect (flags.protect) after the accuracy roll',
    team(FO(SLOW)), team(mon('Blissey', ['protect', 'splash'], { ability: 'Immunity', evs: FAST })),
    ['>p1 move 1\n>p2 move 1']);

  await run('INNERFOCUS: the secondary still rolls; the flinch is blocked at the apply',
    team(FO(FAST)), team(mon('Dragonite', ['tackle', 'splash'], { ability: 'Inner Focus', evs: SLOW })),
    ['>p1 move 1\n>p2 move 1']);

  await run('MOVED: the target already moved — addVolatile(flinch) is STILL called (no gate)',
    team(FO(SLOW)), team(mon('Jolteon', ['quickattack', 'splash'], { ability: 'Immunity', evs: FAST })),
    ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1']);

  // MISS is only reachable through evasion / Bright Powder (acc 100 x 0.9). Sweep seeds.
  const bp1 = team(FO(FAST));
  const bp2 = team(mon('Blissey', ['splash', 'tackle'], { ability: 'Immunity', evs: SLOW, item: 'Bright Powder' }));
  let found = null;
  for (let k = 0; k < 40 && !found; k++) {
    const r = await run(`miss-sweep ${k}`, bp1, bp2, ['>p1 move 1\n>p2 move 1'], { seed: `[${k},${k},${k},${k}]`, quiet: true });
    if (r.log.some(l => l.startsWith('|-miss|'))) found = { k, r };
  }
  if (found) {
    console.log(`\n===== MISS (Bright Powder, seed [${found.k}x4])\n  LOG:\n    ` + found.r.log.join('\n    '));
    for (const m of found.r.marks) console.log(`  [${m.cmd}] draws(${m.n}): ${m.d.join('  ')}`);
  } else {
    console.log('\n===== MISS: no miss realized in 40 seeds');
  }

  // ======================================================== 6. SLEEP TALK ====
  await run('ST-FAIL: Sleep Talk calls Fake Out on the mon\'s 2nd move action -> FAILS',
    team(mon('Persian', ['sleeptalk', 'fakeout'], { evs: FAST })),
    team(mon('Breloom', ['spore', 'splash'], { ability: 'Immunity', evs: SLOW })),
    ['>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 2', '>p1 move 1\n>p2 move 2']);

  await run('ST-WORK: Sleep Talk IS the mon\'s FIRST move action -> the called Fake Out WORKS',
    team(mon('Snorlax', ['splash'], { ability: 'Immunity', evs: SLOW }),
      mon('Persian', ['sleeptalk', 'fakeout'], { evs: SLOW })),
    team(mon('Breloom', ['spore', 'splash'], { ability: 'Immunity', evs: FAST })),
    ['>p1 switch 2\n>p2 move 1', '>p1 move 1\n>p2 move 2', '>p1 move 1\n>p2 move 2']);

  process.exit(0);
})();
