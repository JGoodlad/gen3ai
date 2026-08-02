// probe_choicelock_mimic_release.js — settle WHETHER the gen3 `choicelock` volatile is ADDED at
// all on a turn where a Choice-Band mon MIMICS over its own locked slot, and whether it is COUNTED
// by that same endTurn's `runEvent('DisableMove')` tie-shuffle. The sim is the ONLY oracle.
//
// WHY: `gen3_choicelock_after_move_v1` moves the port's lock set-site to AfterMove (the sim's
// `choiceband.onAfterMove`). That relocation lands AFTER Mimic's onHit, so the round-23 fix
// (`gen3_mimic_choice_lock_self_overwrite_v1`, which EAGERLY clears the lock when Mimic overwrites
// its own locked slot) no longer holds — corpus fixture 49 regressed to `kind=seed`.
//
// The sim's own sequence is not in dispute:
//     choiceband.onAfterMove -> addVolatile('choicelock')
//     choicelock.onStart     -> effectState.move = pokemon.lastMove.id      // 'mimic'
//     choicelock.onDisableMove (at endTurn's runEvent('DisableMove')):
//         if (!getItem().isChoice || !hasMove(effectState.move)) { removeVolatile; return; }
// Mimic OVERWROTE its own slot, so `hasMove('mimic')` is false and the volatile self-removes. The
// question is purely one of TIMING, and it decides a DRAW:
//   (i)  ADDED then RELEASED at that endTurn  -> it IS gathered by that DisableMove event, so it
//        COUNTS toward the handler-sort tie-shuffle on the Mimic turn (round 24 proved exactly
//        this for the ITEM half: the Knock-Off turn STILL draws the shuffle, the next turn does
//        not). The port must add-then-release, NOT clear eagerly.
//   (ii) never effectively present -> the port may keep clearing eagerly and nothing is counted.
//
// A second disabling volatile is required to make the difference OBSERVABLE: one handler alone
// never ties, so it draws nothing either way. Taunt supplies the second handler.
//
// Run:  node src/rust_sim/harness/probe_choicelock_mimic_release.js
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
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31, nature: 'Serious', level: 100, gender: 'M',
  };
}
const tick = () => new Promise((r) => setTimeout(r, 0));

// p1 = the subject: Choice Band + Mimic (+ 3 other slots so a lock is unambiguous).
// p2 = a FASTER Taunter, so the Taunt volatile is already on the subject when it moves and the
//      subject carries TWO DisableMove handlers (taunt + choicelock) at endTurn.
async function run(label, p1moves, p1item, plan) {
  const p1 = [mon('Sudowoodo', p1moves, { item: p1item, ability: 'Rock Head' })];
  const p2 = [mon('Electrode', ['taunt', 'splash'], { ability: 'Soundproof', evs: { spe: 252 } })];

  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const _ of streams.omniscient) {} })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify([7, 11, 13, 17])}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;

  // Count PRNG draws per turn by wrapping the live prng's underlying draw function. `rng` is the
  // single primitive `random`/`randomChance`/`sample`/`shuffle` all funnel through, so one call ==
  // one draw.
  let draws = 0;
  const prng = battle.prng;
  const origNext = prng.rng.next.bind(prng.rng);
  prng.rng.next = (...a) => { draws++; return origNext(...a); };

  const perTurn = [];
  for (const step of plan) {
    if (battle.ended) break;
    draws = 0;
    if (step.p1) streams.omniscient.write(`>p1 ${step.p1}`);
    if (step.p2) streams.omniscient.write(`>p2 ${step.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    const a = battle.sides[0].active[0];
    perTurn.push({
      step: JSON.stringify(step),
      draws,
      cl: !!(a.volatiles && a.volatiles['choicelock']),
      clMove: a.volatiles && a.volatiles['choicelock'] ? a.volatiles['choicelock'].move : '-',
      taunt: !!(a.volatiles && a.volatiles['taunt']),
      slots: a.moveSlots.map((m) => m.id).join(','),
    });
  }

  console.log(`\n=== ${label} ===`);
  for (const t of perTurn) {
    console.log(`    ${t.step}  draws=${t.draws}  choicelock=${t.cl}${t.cl ? `(${t.clMove})` : ''}`
      + `  taunt=${t.taunt}  slots=[${t.slots}]`);
  }
  return perTurn;
}

(async () => {
  // A — a TAUNT-CANT'D move. p2 Taunts FIRST (faster), so p1's Mimic — a Status move — is cant'd
  // and never executes (watch `slots`: unchanged). This is NOT the self-overwrite case; it is here
  // because it independently pins the ABORT rule the fix relies on:
  //     an onBeforeMove-aborted move reaches no AfterMove, so NO choicelock volatile is added.
  const a = await run('A a TAUNT-CANT\'D move adds NO choicelock (the abort rule)',
    ['mimic', 'rockslide', 'earthquake', 'toxic'], 'Choice Band', [
      { p1: 'move mimic', p2: 'move taunt' },
      { p1: 'move rockslide', p2: 'move splash' },
    ]);

  // B — CONTROL: p1 uses a NORMAL move, so the volatile is added and STAYS (hasMove is true).
  const b = await run('B CONTROL: CB + a normal move (volatile added and kept)',
    ['mimic', 'rockslide', 'earthquake', 'toxic'], 'Choice Band', [
      { p1: 'move rockslide', p2: 'move taunt' },
      { p1: 'move rockslide', p2: 'move splash' },
    ]);

  // C — CONTROL: no Choice item at all.
  const c = await run('C CONTROL: no Choice item',
    ['mimic', 'rockslide', 'earthquake', 'toxic'], 'Leftovers', [
      { p1: 'move mimic', p2: 'move taunt' },
      { p1: 'move rockslide', p2: 'move splash' },
    ]);

  // D — THE SELF-OVERWRITE CASE, uncant'd: p2 (faster) merely Splashes, so p1's Mimic RESOLVES and
  // copies Splash OVER p1's own mimic slot (watch `slots`). p1 holds a Choice Band, so AfterMove
  // adds `choicelock` keyed to 'mimic' — and the very next DisableMove finds `hasMove('mimic')`
  // false and removes it. Both readings therefore end at choicelock=false; only a DRAW could tell
  // them apart, and see the UNREACHABILITY note below for why one is never available here.
  const d = await run('D THE CASE: CB + Mimic RESOLVES and overwrites its own slot',
    ['mimic', 'rockslide', 'earthquake', 'toxic'], 'Choice Band', [
      { p1: 'move mimic', p2: 'move splash' },
      { p1: 'move rockslide', p2: 'move splash' },
    ]);

  console.log('\n--- READ ---');
  console.log(`    turn-1 draws:  A(cant'd)=${a[0].draws}  B(normal+CB)=${b[0].draws}  C(no CB)=${c[0].draws}  D(mimic resolves)=${d[0].draws}`);
  console.log(`    A slots after turn 1: [${a[0].slots}]  (unchanged => Mimic was CANT'D, not run)`);
  console.log(`    D slots after turn 1: [${d[0].slots}]  (slot 0 rewritten => Mimic RESOLVED)`);
  console.log(`    D choicelock after turn 1: ${d[0].cl}  (released — hasMove('mimic') is now false)`);
  console.log('');
  console.log('    UNREACHABILITY OF THE DRAW: telling "added then released" from "never added"');
  console.log('    needs a SECOND DisableMove handler on the mon at that same endTurn, so the two');
  console.log('    tie and the handler-sort shuffle draws. On a Mimic turn that is impossible:');
  console.log('      * a Choice-item mon can only use Mimic as its FIRST move (any earlier move');
  console.log('        locks it), so it has no lastMove yet — and DISABLE and ENCORE both require');
  console.log('        the target to already have one, so neither can be on it;');
  console.log('      * TAUNT is the only other handler, and it CANTS Mimic outright (case A), so');
  console.log('        the move never resolves and never overwrites its slot.');
  console.log('    => on the Mimic turn the choicelock handler is necessarily ALONE, never ties,');
  console.log('       and draws nothing either way. Modeling the release EAGERLY at the Mimic site');
  console.log('       is therefore observationally equivalent here.');
})();
