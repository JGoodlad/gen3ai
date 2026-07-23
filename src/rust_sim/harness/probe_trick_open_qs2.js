// probe_trick_open_qs2.js — clean Substitute block + the Knock-Off-then-Trick (itemKnockedOff)
// gate. Run: node src/rust_sim/harness/probe_trick_open_qs2.js
'use strict';
const { mon, run, fmtCalls } = require('./probe_batch4_lib');
const SEED = [5, 4, 3, 2];
function showDec(tag, r) {
  r.perDecision.forEach((d, i) => {
    const lines = d.lines.filter((l) => l && !l.startsWith('|t:|') && !l.startsWith('|upkeep') && !l.startsWith('|debug'));
    console.log(`  ${tag} t${i + 1}: draws=${d.nexts} calls=[${fmtCalls(d.calls)}]`);
    console.log(`        lines=${JSON.stringify(lines)}`);
    if (r.states[i]) console.log(`        state=${JSON.stringify(r.states[i])}`);
  });
}
async function main() {
  // (A') Trick vs Substitute, user Leftovers, target Silk Scarf (no CB lock either side).
  console.log("## (A') Trick vs Substitute (no CB either side)");
  {
    const teams = [
      [mon('Alakazam', ['trick', 'splash'], { item: 'Leftovers', ability: 'Synchronize' })],
      [mon('Snorlax', ['substitute', 'splash'], { item: 'Silk Scarf', ability: 'Own Tempo' })],
    ];
    const r = await run(teams, SEED, [
      ['move 2', 'move 1'],  // p1 Splash, p2 Substitute
      ['move 1', 'move 2'],  // p1 Trick into subbed Snorlax
    ], { onBoundary: (b) => ({ p1item: b.sides[0].active[0].item, p2item: b.sides[1].active[0].item, p2sub: !!b.sides[1].active[0].volatiles.substitute }) });
    showDec('sub', r);
  }
  // (E) Knock Off (marks knockedOff) then Trick -> FAIL.
  console.log('\n## (E) Knock Off then Trick into the knocked-off mon -> fail');
  {
    const teams = [
      [mon('Alakazam', ['knockoff', 'trick', 'splash'], { item: 'Leftovers', ability: 'Synchronize' })],
      [mon('Snorlax', ['splash'], { item: 'Sitrus Berry', ability: 'Own Tempo' })],
    ];
    const r = await run(teams, SEED, [
      ['move 1', 'move 1'],  // t1 p1 Knock Off (removes+marks p2 item); p2 Splash
      ['move 2', 'move 1'],  // t2 p1 Trick (p1 Leftovers, p2 knocked-off itemless) -> FAIL
    ], { onBoundary: (b) => ({ p1item: b.sides[0].active[0].item, p2item: b.sides[1].active[0].item, p2knocked: !!b.sides[1].active[0].itemKnockedOff }) });
    showDec('ko-trick', r);
  }
  // (E2) The USER itself is knocked-off, then Tricks a target with an item -> FAIL.
  console.log('\n## (E2) knocked-off USER Tricks a target with an item -> fail');
  {
    const teams = [
      [mon('Alakazam', ['trick', 'splash'], { item: 'Leftovers', ability: 'Synchronize' })],
      [mon('Snorlax', ['knockoff', 'splash'], { item: 'Leftovers', ability: 'Own Tempo' })],
    ];
    const r = await run(teams, SEED, [
      ['move 2', 'move 1'],  // t1 p1 Splash; p2 Knock Off (removes+marks p1 item)
      ['move 1', 'move 2'],  // t2 p1 Trick (p1 knocked-off itemless, p2 Leftovers) -> FAIL
    ], { onBoundary: (b) => ({ p1item: b.sides[0].active[0].item, p2item: b.sides[1].active[0].item, p1knocked: !!b.sides[0].active[0].itemKnockedOff }) });
    showDec('ko-user-trick', r);
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
