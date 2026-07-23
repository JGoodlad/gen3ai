// probe_batch89_haze_trick_yawn.js — LIVE draw-model probe for Batch-8 HAZE / TRICK / YAWN
// against the OMNISCIENT in-process BattleStream (Dex.mod('gen3'), no server). The sim is
// the ONLY oracle; the dexfacts dump is only a hypothesis.
//
// Run: node src/rust_sim/harness/probe_batch89_haze_trick_yawn.js
'use strict';
const { mon, run, fmtCalls } = require('./probe_batch4_lib');
const SEED = [5, 4, 3, 2];

function showDec(tag, r) {
  r.perDecision.forEach((d, i) => {
    const lines = d.lines.filter((l) => l && !l.startsWith('|t:|') && !l.startsWith('|upkeep') && !l.startsWith('|request'));
    console.log(`  ${tag} t${i + 1}: draws=${d.nexts} calls=[${fmtCalls(d.calls)}]`);
    console.log(`        lines=${JSON.stringify(lines)}`);
    if (r.states[i]) console.log(`        state=${JSON.stringify(r.states[i])}`);
  });
}

async function main() {
  // ---------------- HAZE ----------------
  console.log('\n############ HAZE ############');
  // Both sides set up boosts, then p1 Hazes -> both boost tables cleared. Confirm:
  //  - draw count on the Haze cast (acc:true -> no accuracy draw? draw-free?)
  //  - emission: |-clearallboost + both sides cleared (order)
  //  - it is a MOVE at priority 0 (not a residual) -> resolves in the action queue.
  {
    const teams = [
      [mon('Snorlax', ['swordsdance', 'haze', 'splash'], { ability: 'Immunity' })],
      [mon('Alakazam', ['calmmind', 'splash'], { ability: 'Synchronize' })],
    ];
    const r = await run(teams, SEED, [
      ['move 1', 'move 1'],  // t1: p1 SD (+2 atk), p2 CM (+1 spa/spd)
      ['move 1', 'move 1'],  // t2: p1 SD (+4 atk), p2 CM (+2)
      ['move 2', 'move 2'],  // t3: p1 HAZE -> clear all; p2 CM (does its cast still boost? order matters)
      ['move 3', 'move 2'],  // t4: verify tables cleared/rebuilt
    ], { onBoundary: (b) => ({ p1boosts: b.sides[0].active[0].boosts, p2boosts: b.sides[1].active[0].boosts }) });
    showDec('HAZE', r);
  }

  // ---------------- TRICK ----------------
  console.log('\n############ TRICK ############');
  // p1 Trick (holds Choice Band) into p2 (holds Leftovers): items swap.
  // Also test fail conditions: both itemless, sticky hold, substitute up.
  {
    const teams = [
      [mon('Alakazam', ['trick', 'splash'], { item: 'Choice Band', ability: 'Synchronize' })],
      [mon('Snorlax', ['splash', 'substitute'], { item: 'Leftovers', ability: 'Immunity' })],
    ];
    const r = await run(teams, SEED, [
      ['move 1', 'move 1'],  // t1: Trick swaps CB <-> Leftovers
    ], { onBoundary: (b) => ({ p1item: b.sides[0].active[0].item, p2item: b.sides[1].active[0].item }) });
    showDec('TRICK-swap', r);
  }
  {
    // Trick into a SUBSTITUTE (bypasssub NOT in trick flags -> should be blocked?)
    const teams = [
      [mon('Alakazam', ['trick', 'splash'], { item: 'Choice Band', ability: 'Synchronize' })],
      [mon('Snorlax', ['substitute', 'splash'], { item: 'Leftovers', ability: 'Immunity' })],
    ];
    const r = await run(teams, SEED, [
      ['move 2', 'move 1'],  // t1: p2 makes a Substitute, p1 Splash
      ['move 1', 'move 2'],  // t2: p1 Trick into the sub
    ], { onBoundary: (b) => ({ p1item: b.sides[0].active[0].item, p2item: b.sides[1].active[0].item, p2sub: !!b.sides[1].active[0].volatiles.substitute }) });
    showDec('TRICK-vs-sub', r);
  }
  {
    // Trick into STICKY HOLD (onTryImmunity) -> fails; and both itemless -> fails.
    const teams = [
      [mon('Alakazam', ['trick'], { item: 'Choice Band', ability: 'Synchronize' })],
      [mon('Muk', ['splash'], { item: 'Leftovers', ability: 'Sticky Hold' })],
    ];
    const r = await run(teams, SEED, [['move 1', 'move 1']],
      { onBoundary: (b) => ({ p1item: b.sides[0].active[0].item, p2item: b.sides[1].active[0].item }) });
    showDec('TRICK-vs-stickyhold', r);
  }
  {
    // Both ITEMLESS -> fail draw model
    const teams = [
      [mon('Alakazam', ['trick'], { ability: 'Synchronize' })],
      [mon('Snorlax', ['splash'], { ability: 'Immunity' })],
    ];
    const r = await run(teams, SEED, [['move 1', 'move 1']],
      { onBoundary: (b) => ({ p1item: b.sides[0].active[0].item, p2item: b.sides[1].active[0].item }) });
    showDec('TRICK-both-itemless', r);
  }
  {
    // p1 itemless, p2 has item -> one-way (trick gives p1's nothing, takes p2's item)?
    const teams = [
      [mon('Alakazam', ['trick'], { ability: 'Synchronize' })],
      [mon('Snorlax', ['splash'], { item: 'Leftovers', ability: 'Immunity' })],
    ];
    const r = await run(teams, SEED, [['move 1', 'move 1']],
      { onBoundary: (b) => ({ p1item: b.sides[0].active[0].item, p2item: b.sides[1].active[0].item }) });
    showDec('TRICK-p1-itemless', r);
  }

  // ---------------- YAWN ----------------
  console.log('\n############ YAWN ############');
  // p1 Yawn on p2: -start yawn (turn cast), then at end of NEXT turn -> slp (draws random(2,6) at RESOLVE).
  {
    const teams = [
      [mon('Snorlax', ['yawn', 'splash'], { ability: 'Immunity' })],
      [mon('Blissey', ['splash'], { ability: 'Natural Cure' })],
    ];
    const r = await run(teams, SEED, [
      ['move 1', 'move 1'],  // t1: Yawn cast (draw-free? emission -start)
      ['move 2', 'move 1'],  // t2: yawn volatile resolves at END -> slp set (random(2,6) here?)
      ['move 2', 'move 1'],  // t3: p2 asleep -> cant/wake draws
    ], { onBoundary: (b) => ({ p2status: b.sides[1].active[0].status, p2vol: Object.keys(b.sides[1].active[0].volatiles), slpTime: b.sides[1].active[0].statusState && b.sides[1].active[0].statusState.time }) });
    showDec('YAWN', r);
  }
  {
    // Yawn onto an already-statused target -> onTryHit fails (target.status). Draw model of the fail?
    const teams = [
      [mon('Snorlax', ['yawn'], { ability: 'Immunity' })],
      [mon('Blissey', ['splash'], { ability: 'Natural Cure', status: 'par' })],
    ];
    // Can't set status via set; instead paralyze first. Use Glare-less: just do a Body Slam? keep simple - use a par lead.
    const r = await run([
      [mon('Snorlax', ['yawn', 'thunderwave'], { ability: 'Immunity' })],
      [mon('Blissey', ['splash'], { ability: 'Natural Cure' })],
    ], SEED, [
      ['move 2', 'move 1'],  // t1: TWave -> par
      ['move 1', 'move 1'],  // t2: Yawn into a par'd target -> fail?
    ], { onBoundary: (b) => ({ p2status: b.sides[1].active[0].status, p2vol: Object.keys(b.sides[1].active[0].volatiles) }) });
    showDec('YAWN-vs-statused', r);
  }
  {
    // Yawn vs a sleep-IMMUNE target (Insomnia / Vital Spirit) -> runStatusImmunity('slp') false -> onTryHit fail.
    const r = await run([
      [mon('Snorlax', ['yawn'], { ability: 'Immunity' })],
      [mon('Primeape', ['splash'], { ability: 'Vital Spirit' })],
    ], SEED, [['move 1', 'move 1']],
      { onBoundary: (b) => ({ p2vol: Object.keys(b.sides[1].active[0].volatiles) }) });
    showDec('YAWN-vs-insomnia', r);
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
