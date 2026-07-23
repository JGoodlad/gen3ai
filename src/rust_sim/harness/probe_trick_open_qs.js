// probe_trick_open_qs.js — the 3 open Trick questions: Substitute block (no CB confound),
// Protect block, and the full Choice-lock interaction (user Tricks away its Band; receiver of
// a Band gets locked on its next move). Run: node src/rust_sim/harness/probe_trick_open_qs.js
'use strict';
const path = require('path');
const { mon, run, fmtCalls } = require('./probe_batch4_lib');
const SEED = [5, 4, 3, 2];

function lockOf(b, s) {
  const v = b.sides[s].active[0].volatiles.choicelock;
  return v ? (v.move || true) : null;
}
function reqMoves(b, s) {
  const a = b.sides[s].active[0];
  if (!a || !a.moveSlots) return null;
  return a.moveSlots.map((m) => `${m.id}${m.disabled ? '(dis)' : ''}`).join(',');
}
function showDec(tag, r) {
  r.perDecision.forEach((d, i) => {
    const lines = d.lines.filter((l) => l && !l.startsWith('|t:|') && !l.startsWith('|upkeep') && !l.startsWith('|debug'));
    console.log(`  ${tag} t${i + 1}: draws=${d.nexts} calls=[${fmtCalls(d.calls)}]`);
    console.log(`        lines=${JSON.stringify(lines)}`);
    if (r.states[i]) console.log(`        state=${JSON.stringify(r.states[i])}`);
  });
}

async function main() {
  // (A) Trick vs Substitute — user holds a NON-Choice item so no lock confound.
  console.log('## (A) Trick vs Substitute (user Leftovers, no CB lock)');
  {
    const teams = [
      [mon('Alakazam', ['trick', 'splash'], { item: 'Leftovers', ability: 'Synchronize' })],
      [mon('Snorlax', ['substitute', 'splash'], { item: 'Choice Band', ability: 'Own Tempo' })],
    ];
    const r = await run(teams, SEED, [
      ['move 2', 'move 1'],  // p1 Splash, p2 Substitute
      ['move 1', 'move 2'],  // p1 Trick into the subbed Snorlax, p2 Splash
    ], { onBoundary: (b) => ({ p1item: b.sides[0].active[0].item, p2item: b.sides[1].active[0].item, p2sub: !!b.sides[1].active[0].volatiles.substitute }) });
    showDec('sub', r);
  }

  // (B) Trick vs Protect.
  console.log('\n## (B) Trick vs Protect');
  {
    const teams = [
      [mon('Alakazam', ['trick'], { item: 'Leftovers', ability: 'Synchronize' })],
      [mon('Snorlax', ['protect', 'splash'], { item: 'Choice Band', ability: 'Own Tempo' })],
    ];
    const r = await run(teams, SEED, [['move 1', 'move 1']],
      { onBoundary: (b) => ({ p1item: b.sides[0].active[0].item, p2item: b.sides[1].active[0].item }) });
    showDec('protect', r);
  }

  // (C) USER is CB-locked, Tricks away its Band -> lock releases? Then next turn free move.
  console.log('\n## (C) CB user Tricks away its Band -> lock release');
  {
    const teams = [
      [mon('Alakazam', ['trick', 'splash', 'psychic'], { item: 'Choice Band', ability: 'Synchronize' })],
      [mon('Snorlax', ['splash'], { item: 'Leftovers', ability: 'Own Tempo' })],
    ];
    const r = await run(teams, SEED, [
      ['move 1', 'move 1'],  // t1 p1 Trick (locks to trick, swaps CB->Snorlax) ; p2 Splash
      ['move 2', 'move 1'],  // t2 p1 tries Splash (was locked into Trick; now unlocked?)
    ], { onBoundary: (b) => ({ p1item: b.sides[0].active[0].item, p2item: b.sides[1].active[0].item, p1lock: lockOf(b, 0), p1moves: reqMoves(b, 0), p2lock: lockOf(b, 1), p2moves: reqMoves(b, 1) }) });
    showDec('cbuser', r);
  }

  // (D) RECEIVER of a Choice Band: does it get locked, and WHEN?
  console.log('\n## (D) receiver of a CB gets locked on its NEXT move');
  {
    const teams = [
      // p1 Alakazam gives its CB to p2 Snorlax via Trick; Snorlax then moves and (should) lock.
      [mon('Alakazam', ['trick', 'splash'], { item: 'Choice Band', ability: 'Synchronize' })],
      [mon('Snorlax', ['bodyslam', 'earthquake', 'splash'], { item: 'Leftovers', ability: 'Own Tempo' })],
    ];
    const r = await run(teams, SEED, [
      ['move 1', 'move 3'],  // t1 p1 Trick (CB->Snorlax, Snorlax's Leftovers->Alakazam); p2 Splash
      ['move 2', 'move 1'],  // t2 p1 Splash; p2 Snorlax BodySlam (holds CB now -> lock to bodyslam)
      ['move 2', 'move 2'],  // t3 p1 Splash; p2 tries Earthquake (locked into BodySlam? -> reject)
    ], { onBoundary: (b) => ({ p1item: b.sides[0].active[0].item, p2item: b.sides[1].active[0].item, p2lock: lockOf(b, 1), p2moves: reqMoves(b, 1) }) });
    showDec('receiver', r);
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
