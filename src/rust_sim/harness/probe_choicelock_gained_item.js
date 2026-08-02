// probe_choicelock_gained_item.js — settle when the gen3 CHOICE LOCK applies to a mon that
// ACQUIRED a Choice item mid-battle, against the OMNISCIENT in-process BattleStream. The sim
// is the ONLY oracle (several source-read hypotheses have already died in this project).
//
// WHY: the live external-consistency gate found `soak3/divergences/sbd_msb1zfxs_b97` — a
// Piloswine whose PACKED item is Leftovers but whose roster shows `choiceband` (it was Tricked
// one). Its request differs:
//     node: Toxic "disabled":false   (only Protect had been used, at pp 14/16)
//     port: Toxic "disabled":true    — plus Earthquake and Ice Beam disabled, i.e. LOCKED to Protect
// Externally visible (poke-env would hide three legal moves from the policy) and DRAW-FREE, so
// no seed/omniscient gate can see it.
//
// THE TWO CANDIDATE RULES:
//   (A) "holds a Choice item AND has a lastMove"  → lock. (What the port's lazy fold appears to
//       do — `bridge.rs::move_disabled`, added as `gen3_choice_lock_request_disabled_v1` to fix
//       the OPPOSITE miss, a mon that GAINED a Band mid-turn and should have locked.)
//   (B) "the `choicelock` VOLATILE exists" — i.e. the mon MOVED **while holding** the Choice
//       item — and it still holds one and still has that move. (Showdown's resolved
//       `choicelock.onDisableMove`.)
// (A) and (B) differ EXACTLY when a mon moved BEFORE acquiring the item — the b97 board.
//
// CASES (each prints the post-state `disabled` flags for all four slots):
//   1 baseline: holds a Choice Band from the start, uses a move → later slots locked?
//   2 THE b97 SHAPE: uses a move with a NON-choice item, THEN is Tricked a Choice Band → locked?
//   3 as (2) but it then MOVES once while holding the Band → locked to the NEW move?
//
// Run:  node src/rust_sim/harness/probe_choicelock_gained_item.js
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
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: 'Serious', level: 100, gender: 'N',
  };
}
const tick = () => new Promise((r) => setTimeout(r, 0));

async function run(label, p1item, plan) {
  // p1 = the subject (4 distinct moves so a lock is unambiguous).
  // p2 = a Trickster holding a Choice Band it can hand over.
  const p1 = [mon('Piloswine', ['toxic', 'protect', 'earthquake', 'icebeam'], { item: p1item, ability: 'Oblivious' }),
              mon('Sudowoodo', ['splash'], { ability: 'Rock Head' })];
  const p2 = [mon('Alakazam', ['trick', 'splash'], { item: 'Choice Band', ability: 'Synchronize' }),
              mon('Snorlax', ['splash'], { ability: 'Immunity' })];

  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const _ of streams.omniscient) { /* drain */ } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify([7, 11, 13, 17])}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;

  for (const step of plan) {
    if (step.p1) streams.omniscient.write(`>p1 ${step.p1}`);
    if (step.p2) streams.omniscient.write(`>p2 ${step.p2}`);
    for (let k = 0; k < 18; k++) await tick();
  }

  const a = battle.sides[0].active[0];
  const req = battle.sides[0].activeRequest;
  const flags = (req && req.active && req.active[0] && req.active[0].moves || [])
    .map((m) => `${m.id}:${m.disabled ? 'DISABLED' : 'ok'}`).join(' ');
  console.log(`\n=== ${label} ===`);
  console.log(`    item now: ${a && a.item} | lastMove: ${a && a.lastMove && a.lastMove.id}`
    + ` | choicelock volatile: ${!!(a && a.volatiles && a.volatiles['choicelock'])}`);
  console.log(`    request flags: ${flags || '(no move request)'}`);
}

(async () => {
  // 1 — baseline: born with the Band, uses Protect. Expect a lock to Protect.
  await run('1 baseline: HELD a Choice Band, then moved', 'Choice Band', [
    { p1: 'move protect', p2: 'move splash' },
  ]);

  // 2 — THE b97 SHAPE: moved with Leftovers, THEN Tricked a Choice Band. Locked or not?
  await run('2 moved with Leftovers, THEN Tricked a Band  <-- the b97 case', 'Leftovers', [
    { p1: 'move protect', p2: 'move splash' },   // lastMove = Protect, non-choice item
    { p1: 'move protect', p2: 'move trick' },    // Alakazam Tricks the Band onto Piloswine
  ]);

  // 3 — as (2), then it MOVES once while holding the Band. Expect a lock to THAT move.
  await run('3 ...then it moves once while holding the Band', 'Leftovers', [
    { p1: 'move protect', p2: 'move splash' },
    { p1: 'move protect', p2: 'move trick' },
    { p1: 'move earthquake', p2: 'move splash' },
  ]);
})();
