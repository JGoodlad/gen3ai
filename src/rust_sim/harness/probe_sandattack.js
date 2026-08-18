// PROBE: gen-3 SAND ATTACK — draw model + emissions, and the ACCURACY/EVASION STAGE TABLE
// measured off the to-hit draw ARGUMENTS.
//
// SETTLED 2026-08-18 (run it to re-confirm; do not re-derive from source):
//
//  1. THE TO-HIT DRAW CARRIES THE FOLDED ACCURACY IN ITS ARGS.
//     gen3 `tryMoveHit` ends the accuracy block with `randomChance(accuracy, 100)` where
//     `accuracy` has ALREADY had the acc/eva stage table + ModifyAccuracy folded in. So the
//     stage multiplier is READ DIRECTLY off the draw arguments — the draw ARGS change, the
//     comparison does not. (`randomChance(n,d)` = `random(d) < n`, so it is ONE PRNG call.)
//
//  2. SAND ATTACK's draw model: exactly ONE `randomChance(100,100)` (acc 100, NOT never-miss),
//     then a DRAW-FREE apply. It is structurally a pure declarative foe stat-drop STATUS move
//     — identical in shape to Screech/Charm, only the stat is `accuracy` (boost index 5).
//        plain hit      -> `|-unboost|<target>|accuracy|1`
//        target at -6   -> STILL draws the roll, STILL emits `|-unboost|<target>|accuracy|0`
//                          (the delta-0 primary-drop line), NO `-fail`
//        Clear Body     -> `|-fail|<t>|unboost|[from] ability: Clear Body|[of] <t>`   (no stat token)
//        Keen Eye       -> `|-fail|<t>|unboost|accuracy|[from] ability: Keen Eye|[of] <t>` (stat token)
//        Protect        -> accuracy drawn, then `|-activate|<t>|Protect`
//        Substitute     -> accuracy drawn, then `|move|<u>|Sand Attack||[still]` + `|-fail|<u>`
//        GROUND target  -> NOT immune (a Status move defaults `ignoreImmunity = true`)
//
//  3. THE STAGE TABLE, boostTable = [1, 4/3, 5/3, 2, 7/3, 8/3, 3], applied to a 100-acc move:
//        attacker accuracy stage s<0 : acc /= T[-s]   -1..-6 -> 75, 60, 50, 42.857…, 37.5, 33.333…
//        attacker accuracy stage s>0 : acc *= T[s]    (UNREACHABLE — no gen3 move raises accuracy)
//        defender evasion  stage s>0 : acc /= T[s]    +1..+6 -> 75, 60, 50, 42.857…, 37.5, 33.333…
//        defender evasion  stage s<0 : acc *= T[-s]   -1     -> 133.33333333333331
//     ⚠ BIT-EXACTNESS: the MULTIPLY direction must use the PRECOMPUTED table entry.
//        100*T[1] = 133.33333333333331  but  100*4/3 = 133.33333333333334  — DIFFERENT f64.
//     The DIVIDE direction happens to agree with the naive form at every stage.
//
//  4. YES, a 100%-accuracy move MISSES at -6 accuracy (effAcc 33.33) — measured below, three
//     misses at -4/-5/-6 off r=99/64/65. Sand Attack itself is subject to BOTH its own user's
//     accuracy stage AND the target's evasion stage (it is `accuracy: 100`, not `true`).
//
//  5. The accuracy check is the SAME code path for Status and damaging moves, in the SAME
//     position: accuracy -> TryHit(Protect) -> {status apply | crit -> damage}.
//     gen3 has NO ModifyBoost handlers (the event is identity) and NO ignoreAccuracy /
//     ignoreEvasion move, so reading the raw boost stages is faithful.
//
// EXPOSURE (measured): 0 of 3000 gen3randombattle teams; 1 of 773 data/teams files, and that
// one FAILS gen3ou validation (Speed Pass Clause) so it is outside the 719-team training pool.
// The value is the `ourandom` Smogon-native gen3ou surface, where sandattack is 0.72 of
// move-slot prior mass — the single largest remaining coverage gap.
const path=require('path'); const PS='/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const {BattleStream}=require(path.join(PS,'dist/sim/battle-stream.js'));
const {PRNG}=require(path.join(PS,'dist/sim/prng'));
let draws=[];
for (const m of ['random','randomChance','sample']) { const o=PRNG.prototype[m];
  PRNG.prototype[m]=function(...a){ const r=o.apply(this,a); draws.push(`${m}(${a})->${r}`); return r; }; }
async function run(label,p1,p2,script,opts={}){
  draws=[]; const s=new BattleStream(); const ch=[];
  (async()=>{ for await(const c of s) ch.push(c); })();
  s.write(`>start {"formatid":"gen3customgame","seed":[9,9,9,9]}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r=>setTimeout(r,200)); let base=draws.length; const per=[];
  for(const c of script){ s.write(c); await new Promise(r=>setTimeout(r,200));
    per.push(draws.slice(base)); base=draws.length; }
  await new Promise(r=>setTimeout(r,200));
  const all=ch.filter(c=>!c.startsWith('sideupdate')).join('\n').split('\n')
    .filter(l=>/^\|(-start|-end|-fail|-immune|-activate|-boost|-unboost|-miss|move|turn)\|/.test(l));
  const last=per[per.length-1]||[];
  if(opts.argsOnly){
    const acc=last.find(d=>/^randomChance\(.*,100\)->/.test(d));
    console.log(`  ${label.padEnd(30)} TO-HIT DRAW: ${acc||'(NONE)'}   ${last.some(d=>/->false/.test(d)&&/,100\)/.test(d))?'MISS':''}`);
    return;
  }
  const i=all.lastIndexOf('|turn|'+per.length);
  console.log(`\n== ${label}\n  ${all.slice(i).join('\n  ')}`);
  console.log('  DRAWS[final turn]: '+(last.join('  ')||'(none)'));
}
const SA  ='Sandslash||Leftovers|SandRush|sandattack,splash,scratch,sweetscent|Hardy|85,85,85,85,85,85|M||||';
const TGT ='Snorlax||Leftovers|Immunity|sandattack,splash,doubleteam,protect|Hardy|85,85,85,85,85,85|M||||';
const SUBM='Snorlax||Leftovers|Immunity|substitute,splash,doubleteam,protect|Hardy|85,85,85,85,85,85|M||||';
const CB  ='Metagross||Leftovers|ClearBody|splash|Hardy|85,85,85,85,85,85|N||||';
const KE  ='Skarmory||Leftovers|KeenEye|splash|Hardy|85,85,85,85,85,85|M||||';
const GRD ='Marowak||Leftovers|RockHead|splash|Hardy|85,85,85,85,85,85|M||||';
const rep=(n,c)=>new Array(n).fill(c);
(async()=>{
  console.log('######## A: SAND ATTACK draw model + emissions ########');
  await run('A1 plain hit',        SA, TGT, ['>p1 move 1\n>p2 move 2']);
  await run('A2 SEVENTH cast (at the -6 cap)', SA, TGT, rep(7,'>p1 move 1\n>p2 move 2'));
  await run('A3 vs CLEAR BODY',    SA, CB,  ['>p1 move 1\n>p2 move 1']);
  await run('A4 vs KEEN EYE',      SA, KE,  ['>p1 move 1\n>p2 move 1']);
  await run('A5 vs GROUND type',   SA, GRD, ['>p1 move 1\n>p2 move 1']);
  await run('A6 into PROTECT',     SA, TGT, ['>p1 move 1\n>p2 move 4']);
  await run('A7 into SUBSTITUTE',  SA, SUBM,['>p1 move 2\n>p2 move 1','>p1 move 1\n>p2 move 2']);

  console.log('\n######## B: the STAGE TABLE, read off the to-hit draw ARGS (Scratch, acc 100) ########');
  console.log('-- attacker ACCURACY stage (p2 Sand Attacks p1 N times) --');
  for(let n=0;n<=6;n++) await run(`acc -${n}`, SA, TGT,
    [...rep(n,'>p1 move 2\n>p2 move 1'), '>p1 move 3\n>p2 move 2'], {argsOnly:1});
  console.log('-- defender EVASION stage (p2 Double Teams N times) --');
  for(const n of [1,2,3,6]) await run(`eva +${n}`, SA, TGT,
    [...rep(n,'>p1 move 2\n>p2 move 3'), '>p1 move 3\n>p2 move 2'], {argsOnly:1});
  console.log('-- defender EVASION -1 (p1 Sweet Scent) = the MULTIPLY direction --');
  await run('eva -1', SA, TGT, ['>p1 move 4\n>p2 move 2','>p1 move 3\n>p2 move 2'], {argsOnly:1});
  console.log('-- SAND ATTACK is itself stage-modified --');
  await run('SA into +1 evasion', SA, TGT, ['>p1 move 2\n>p2 move 3','>p1 move 1\n>p2 move 2'], {argsOnly:1});
  await run('SA at -2 own acc',   SA, TGT, [...rep(2,'>p1 move 2\n>p2 move 1'),'>p1 move 1\n>p2 move 2'], {argsOnly:1});

  console.log('\n######## C: f64 bit patterns the port must reproduce ########');
  const T=[1,4/3,5/3,2,7/3,8/3,3];
  for(let s=1;s<=6;s++) console.log(`  100/T[${s}] = ${100/T[s]}`);
  console.log(`  100*T[1] = ${100*T[1]}   vs naive 100*4/3 = ${100*4/3}   IDENTICAL=${100*T[1]===100*4/3}`);
  process.exit(0);
})();
