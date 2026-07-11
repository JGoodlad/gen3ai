// probe_focusband_confusion_rng.js — the FB gap probe_focusband_rng.js left open (its
// confusion scenario never used Confuse Ray): does a CONFUSION SELF-HIT into an FB
// holder DRAW the randomChance(1,10)? Can FB SURVIVE a lethal self-hit (the gen4
// conditions confusion self-damage is dealt with effectType:'Move')?
// Run: node src/rust_sim/harness/probe_focusband_confusion_rng.js
'use strict';
const { mon, run, fmtCalls } = require('./probe_batch4_lib');

async function main() {
  console.log('=== confusion self-hit draw: FB Snorlax confused, splashes (self-hit on 1/2)');
  const mk = (item) => [
    [mon('Snorlax', ['splash'], { ability: 'Thick Fat', item })],
    [mon('Jolteon', ['confuseray', 'splash'], { ability: 'Sturdy', evs: { spe: 252 } })],
  ];
  for (const seed of [[1,1,1,1],[2,2,2,2],[3,3,3,3],[5,5,5,5]]) {
    for (const item of ['Focus Band', '']) {
      const r = await run(mk(item), seed, [['move 1','move 1'],['move 1','move 2'],['move 1','move 2']]);
      r.perDecision.forEach((d,i)=>{
        const hurt = d.lines.some(l=>l.includes('[from] confusion'));
        console.log(`item=${item||'none'} seed=${seed[0]} t${i+1} selfhit=${hurt}: [${fmtCalls(d.calls)}]`);
      });
    }
  }
  console.log('=== lethal self-hit survive: lv-9 FB Rattata confused (self-hit can be lethal)');
  const lethal = [
    [mon('Rattata', ['splash'], { ability: 'Guts', item: 'Focus Band', level: 9 })],
    [mon('Jolteon', ['confuseray', 'splash'], { ability: 'Sturdy', evs: { spe: 252 } })],
  ];
  for (let s = 1; s <= 20; s++) {
    const seed = [s, s*3+1, s*5+2, s*7+3];
    const r = await run(lethal, seed, [['move 1','move 1'],['move 1','move 2'],['move 1','move 2'],['move 1','move 2']], {
      onBoundary: (b) => ({ hp: b.p1.pokemon[0].hp, fnt: b.p1.pokemon[0].fainted }),
    });
    const act = r.lines.filter(l=>l.includes('Focus Band'));
    const hurt = r.lines.some(l=>l.includes('[from] confusion'));
    if (act.length || (hurt && r.states.some(st=>st && st.fnt)))
      console.log(`seed=${s}: FBact=${JSON.stringify(act)} states=${JSON.stringify(r.states)}`);
  }
}
main().catch((e)=>{console.error(e);process.exit(1);});
