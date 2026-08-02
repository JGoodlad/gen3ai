// probe_choicelock_acquired_item.js — settle WHEN the gen3 CHOICE LOCK applies to a mon that
// ACQUIRED a Choice item mid-battle, against the OMNISCIENT in-process BattleStream. The sim is
// the ONLY oracle (this exact code has been changed twice, in OPPOSITE directions, off source
// reads — see the CAUTION below).
//
// WHY: the live external-consistency gate found `soak3/divergences/sbd_msb1zfxs_b97`. p1's
// Piloswine is PACKED with Leftovers but its roster shows `choiceband` — p2's Kecleon Tricked it
// one. Its `|request|` diverges:
//     node: Toxic "disabled":false            (only Protect had been used, at pp 14/16)
//     port: Toxic "disabled":true  — plus Earthquake and Ice Beam — i.e. LOCKED to Protect
// Externally visible (poke-env would hide three legal moves from the policy) and DRAW-FREE, so no
// seed / omniscient-byte gate can see it. Only the request JSON diverges.
//
// THE ISOLATION THE EARLIER PROBE COULD NOT BUILD. To separate the two candidate rules you need a
// mon that HOLDS a Choice item but has NOT MOVED SINCE ACQUIRING IT. My first attempt made the
// Trickster faster, so the Band always landed BEFORE the subject moved and the subject then moved
// while holding it — which locks under BOTH rules. The b97 board hands us the answer: at the flat
// 85-EV randbats spread, Piloswine (base spe 50) OUTSPEEDS Kecleon (base spe 40), so the subject
// moves FIRST holding Leftovers and the Band arrives AFTER. Reproduced verbatim below.
//
// THE TWO CANDIDATE RULES:
//   (A) "holds a Choice item AND has a lastMove"  → lock. (What the port's lazy request-build fold
//       `bridge.rs::move_disabled` does.)
//   (B) "the `choicelock` VOLATILE exists" — i.e. the mon MOVED **while holding** the Choice item
//       — and it still holds one and still has that move. (Showdown's `choicelock.onDisableMove`.)
// They differ EXACTLY on case 2.
//
// ⚠️ CAUTION — this code has a two-sided bug history, which is why case 4 is here:
//   * round 6 `gen3_choice_lock_request_disabled_v1` ADDED the lazy fold because the port
//     UNDER-locked a mon that GAINED a Band mid-turn (a Skarmory that Thief'd one).
//   * round 24 `gen3_choicelock_lazy_release_v1` then split the VOLATILE from the ENFORCED lock.
//   So a narrowing that fixes b97 must NOT re-break the Thief case. Case 4 is the regression guard,
//   and it is deliberately probed here rather than assumed.
//
// CASES (each prints item / lastMove / the `choicelock` volatile / the request `disabled` flags):
//   1 baseline    : HOLDS a Choice Band from the start, uses a move        → expect LOCKED
//   2 THE b97 CASE: moves FIRST with Leftovers, THEN is Tricked a Band     → locked?
//   3 then-moves  : as (2), then moves once WHILE holding the Band         → expect LOCKED to it
//   4 THIEF GUARD : itemless, STEALS a Band with Thief (round 6's case)    → locked?
//
// Run:  node src/rust_sim/harness/probe_choicelock_acquired_item.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
// The b97 randbats spread verbatim: flat 85 EVs, Hardy. This is what makes Piloswine (base spe 50)
// outspeed Kecleon (base spe 40) — the whole point of the isolation.
const EV85 = { hp: 85, atk: 85, def: 85, spa: 85, spd: 85, spe: 85 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: opts.evs || EV85, ivs: IV31, nature: 'Hardy', level: 100, gender: 'M',
  };
}
const tick = () => new Promise((r) => setTimeout(r, 0));

async function run(label, p1item, p2item, plan, expectation) {
  // p1 = the SUBJECT (4 distinct moves so a lock is unambiguous) — the b97 Piloswine.
  // p2 = the b97 Kecleon: Choice Band + Trick, and SLOWER than the subject.
  const p1 = [mon('Piloswine', ['toxic', 'protect', 'earthquake', 'icebeam'],
    { item: p1item, ability: 'Oblivious' })];
  const p2 = [mon('Kecleon', ['trick', 'return', 'brickbreak', 'shadowball'],
    { item: p2item, ability: 'Color Change' })];

  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const _ of streams.omniscient) { /* drain */ } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify([7, 11, 13, 17])}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;

  const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
  const speedNote = `spe p1=${a0.getStat('spe')} p2=${a1.getStat('spe')}`
    + ` (${a0.getStat('spe') > a1.getStat('spe') ? 'SUBJECT faster — the isolation holds' : 'SUBJECT SLOWER — isolation BROKEN'})`;

  for (const step of plan) {
    if (battle.ended) break;
    if (step.p1) streams.omniscient.write(`>p1 ${step.p1}`);
    if (step.p2) streams.omniscient.write(`>p2 ${step.p2}`);
    for (let k = 0; k < 18; k++) await tick();
  }

  const a = battle.sides[0].active[0];
  const req = battle.sides[0].activeRequest;
  const slots = (req && req.active && req.active[0] && req.active[0].moves) || [];
  const flags = slots.map((m) => `${m.id}:${m.disabled ? 'DISABLED' : 'ok'}`).join(' ');
  const nDisabled = slots.filter((m) => m.disabled).length;
  const locked = slots.length > 1 && nDisabled === slots.length - 1;
  const verdict = slots.length === 0 ? 'NO MOVE REQUEST'
    : locked ? `LOCKED (to ${slots.find((m) => !m.disabled).id})`
    : nDisabled === 0 ? 'NOT LOCKED (all selectable)' : `PARTIAL (${nDisabled} disabled)`;

  console.log(`\n=== ${label} ===`);
  console.log(`    ${speedNote}`);
  console.log(`    item now: ${a.item || '(none)'} | lastMove: ${(a.lastMove && a.lastMove.id) || '(none)'}`
    + ` | choicelock volatile: ${!!(a.volatiles && a.volatiles['choicelock'])}`
    + `${a.volatiles && a.volatiles['choicelock'] ? ` (move=${a.volatiles['choicelock'].move})` : ''}`);
  console.log(`    request: ${flags || '(none)'}`);
  console.log(`    VERDICT: ${verdict}${expectation ? `   [expected: ${expectation}]` : ''}`);
  return verdict;
}

(async () => {
  // Show the sim its own rule, resolved through the gen3 mod chain, before any behavioural claim.
  const cl = Dex.mod('gen3').conditions.get('choicelock');
  console.log('--- resolved gen3 choicelock.onDisableMove ---');
  console.log(String(cl.onDisableMove).split('\n').map((l) => '    ' + l.trim()).join('\n'));

  // 1 — baseline: born with the Band, uses Protect while holding it. The volatile IS added.
  await run('1 baseline: HELD the Band, then moved', 'Choice Band', 'Leftovers', [
    { p1: 'move protect', p2: 'move return' },
  ], 'LOCKED');

  // 2 — THE b97 CASE. Piloswine (faster) moves holding Leftovers; Kecleon then Tricks the Band on.
  // It ends holding a Choice Band with a lastMove made BEFORE the Band arrived.
  // NOTE the subject must move with TOXIC, not Protect: Trick carries `protect: 1`, so a Protect
  // BLOCKS the Trick and the item never swaps (which silently destroys the isolation — this probe
  // reported `item now: leftovers` until the move was changed).
  await run('2 moved FIRST with Leftovers, THEN Tricked a Band  <-- the b97 case', 'Leftovers', 'Choice Band', [
    { p1: 'move toxic', p2: 'move trick' },
  ], 'the question');

  // 3 — as (2), then it MOVES once while holding the Band → the volatile is added now.
  await run('3 ...then it moves once while holding the Band', 'Leftovers', 'Choice Band', [
    { p1: 'move toxic', p2: 'move trick' },
    { p1: 'move earthquake', p2: 'move return' },
  ], 'LOCKED to earthquake');

  // 4 — THE ROUND-6 REGRESSION GUARD: itemless subject STEALS the Band with its own move.
  // (Thief is given to the subject directly — gen3customgame does not validate learnsets.)
  const p1thief = [mon('Piloswine', ['thief', 'protect', 'earthquake', 'icebeam'],
    { item: '', ability: 'Oblivious' })];
  const p2band = [mon('Kecleon', ['splash', 'return', 'brickbreak', 'shadowball'],
    { item: 'Choice Band', ability: 'Color Change' })];
  {
    const stream = new BattleStream();
    const streams = getPlayerStreams(stream);
    (async () => { for await (const _ of streams.omniscient) {} })();
    streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify([7, 11, 13, 17])}}`);
    streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1thief) })}`);
    streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2band) })}`);
    for (let i = 0; i < 12; i++) await tick();
    const battle = stream.battle;
    streams.omniscient.write('>p1 move thief');
    streams.omniscient.write('>p2 move splash');
    for (let k = 0; k < 18; k++) await tick();
    const a = battle.sides[0].active[0];
    const req = battle.sides[0].activeRequest;
    const slots = (req && req.active && req.active[0] && req.active[0].moves) || [];
    const nDisabled = slots.filter((m) => m.disabled).length;
    console.log(`\n=== 4 THIEF GUARD: itemless, STOLE the Band with its own move (round 6's case) ===`);
    console.log(`    item now: ${a.item || '(none)'} | lastMove: ${(a.lastMove && a.lastMove.id) || '(none)'}`
      + ` | choicelock volatile: ${!!(a.volatiles && a.volatiles['choicelock'])}`);
    console.log(`    request: ${slots.map((m) => `${m.id}:${m.disabled ? 'DISABLED' : 'ok'}`).join(' ') || '(none)'}`);
    console.log(`    VERDICT: ${slots.length > 1 && nDisabled === slots.length - 1 ? 'LOCKED' : nDisabled === 0 ? 'NOT LOCKED' : 'PARTIAL'}`
      + `   [round 6 claims the sim LOCKS here]`);
  }
})();
