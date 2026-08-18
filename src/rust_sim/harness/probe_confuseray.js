// PROBE: gen-3 CONFUSE RAY — draw model + emission + the fail/immune gates.
//
// SETTLED 2026-08-18 (run it to re-confirm; do not re-derive from source):
//   A plain hit      : random(100) accuracy  THEN random(2,6) duration
//                      -> `|-start|<target>|confusion`
//   already confused : accuracy roll ONLY, NO duration draw -> `|move|…|[still]` (the move fails)
//   OWN TEMPO target : accuracy roll ONLY, NO duration draw
//                      -> `|-immune|<target>|confusion|[from] ability: Own Tempo`
//
// The port ALREADY has the hard half: `secondaries.rs::add_confusion` implements the KO /
// already-confused / Own-Tempo gates and the random(2,6) duration draw, and emits the
// `-start|confusion`. What is missing to MODEL the move is only the two MOVE-LEVEL emissions
// (the `[still]`+fail form and the `-immune … Own Tempo` form) plus the dispatch entry, and
// then removing `confuseray` from the three fail-loud mirrors (state.rs UNMODELED_FAILLOUD_MOVES,
// gen_e2e_fuzz.js REJECT_MOVES, scan_move_coverage.js).
//
// WHY IT IS WORTH DOING: confuseray carries 0.37 of the gen3ou move-slot prior mass — joint
// third on the remaining unmodeled list behind sandattack (0.72) and level with recycle/safeguard.
const path=require('path'); const PS='/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const {BattleStream}=require(path.join(PS,'dist/sim/battle-stream.js'));
const {Dex}=require(path.join(PS,'dist/sim/dex.js'));
const {PRNG}=require(path.join(PS,'dist/sim/prng'));
const d=Dex.mod('gen3').moves.get('confuseray');
console.log('dex: acc',d.accuracy,'cat',d.category,'target',d.target,'volatileStatus',d.volatileStatus,'flags',JSON.stringify(d.flags));
let draws=[];
for (const m of ['random','randomChance','sample']) { const o=PRNG.prototype[m];
  PRNG.prototype[m]=function(...a){ const r=o.apply(this,a); draws.push(`${m}(${a})->${r}`); return r; }; }
async function run(label,p1,p2,script){
  draws=[]; const s=new BattleStream(); const ch=[];
  (async()=>{ for await(const c of s) ch.push(c); })();
  s.write(`>start {"formatid":"gen3customgame","seed":[9,9,9,9]}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r=>setTimeout(r,250)); const base=draws.length;
  for(const c of script){ s.write(c); await new Promise(r=>setTimeout(r,250)); }
  await new Promise(r=>setTimeout(r,250));
  const omni=ch.filter(c=>!c.startsWith('sideupdate')).join('\n').split('\n')
    .filter(l=>/^\|(-start|-fail|-immune|-activate|move|turn)\|/.test(l));
  const i=omni.indexOf('|turn|1');
  console.log(`\n== ${label}\n  ${omni.slice(i+1,i+7).join('\n  ')}`);
  console.log('  DRAWS this turn:', draws.slice(base).join('  ') || '(none)');
}
(async()=>{
  const user='Gengar||Leftovers|Levitate|confuseray,splash,nightshade,willowisp|Hardy|85,85,85,85,85,85|M||||';
  await run('A: plain hit', user, 'Snorlax||Leftovers|Immunity|splash|Hardy|85,85,85,85,85,85|M||||', ['>p1 move 1\n>p2 move 1']);
  await run('B: already confused (2nd cast)', user, 'Snorlax||Leftovers|Immunity|splash|Hardy|85,85,85,85,85,85|M||||', ['>p1 move 1\n>p2 move 1','>p1 move 1\n>p2 move 1']);
  await run('C: OWN TEMPO target', user, 'Slowbro||Leftovers|OwnTempo|splash|Hardy|85,85,85,85,85,85|M||||', ['>p1 move 1\n>p2 move 1']);
})();
