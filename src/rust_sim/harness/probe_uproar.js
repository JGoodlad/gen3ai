// PROBE: gen-3 UPROAR — the multi-turn LOCK on the user + the field-wide SLEEP BLOCK.
//
// SETTLED 2026-08-18 (run it to re-confirm; do not re-derive from source):
//
// THE LOCK (drawn ONCE, on the CAST turn only, AFTER the damage):
//   cast turn : random(100) acc -> random(16) crit -> random(16,) dmg -> random(2,6) DURATION
//               -> `|-start|<user>|Uproar`            [+ endTurn random(5) Quick Claw]
//   locked    : random(100) acc -> random(16) crit -> random(16,) dmg   (NO duration re-draw, NO PP)
//   residual  : order 10 / subOrder 11 -> `|-start|<user>|Uproar|[upkeep]` (ticks the duration)
//               ... on the tick that reaches 0 -> `|-end|<user>|Uproar` INSTEAD  (the duration-end
//               `continue` branch; NO end-of-lock confusion, unlike outrage/thrash)
//   duration D = random(2,6) in {2,3,4,5}; the user uses uproar EXACTLY D times (the CAST turn's
//   own residual already ticks it). PP is deducted ONCE, on the cast (16 -> 15 over a 5-turn lock).
//   request while locked (the FIRM single-move shape, no pp/maxpp/target/disabled):
//     {"active":[{"moves":[{"move":"Uproar","id":"uproar"}],"trapped":true}], ...}
//     a different slot -> `|error|[Invalid choice] Can't move: Your X doesn't have a move N`
//     a switch         -> `|error|[Invalid choice] Can't switch: The active Pokemon is trapped`
//
// THE SLEEP BLOCK (the uproar condition's `onAnySetStatus`, so it fires INSIDE runEvent('SetStatus')):
//   vs the UPROARER   -> `|-fail|<mon>|slp|[from] Uproar|[msg]`
//   vs anyone ELSE    -> `|-fail|<mon>|slp|[from] Uproar`          (no [msg])
//   the sleep move STILL DRAWS its accuracy roll; the sleep `random(2,6)` is NOT drawn.
//   gen3ou: the SetStatus handler-sort shuffle STILL draws exactly ONE call — the uproar handler has
//   a DEFINED speed so it takes index 0 and the 2 Standard clauses stay a size-2 tie:
//   `random(1,3)` vs the control's `random(0,2)`. SAME draw COUNT (PRNG.random calls next() once
//   regardless of range), so the seed is unaffected. customgame: uproar is the ONLY handler -> size 1
//   -> NO shuffle at all.
//   REST is fully blocked: `|-fail|<mon>|slp|[from] Uproar`, NO heal, NO random(2,6).
//   YAWN resolving at 10/19 is blocked the same way -- BUT on the residual where uproar EXPIRES
//   (10/11) the `-end` runs FIRST and the yawn then LANDS (`|-status|<mon>|slp` + its random(2,6)).
//
// THE WAKE (the MOVE's own `onTryHit`, fired by singleEvent BEFORE the damage):
//   a landed uproar cures slp on BOTH ACTIVES -> `|-curestatus|<mon>|slp|[msg]` (bare, no [from]),
//   emitted after the `|move|` line and before `|-resisted|`/`|-damage|`. DRAW-FREE.
//   NOT on a MISS, NOT on a type-IMMUNE (Ghost) target, and NEVER for a BENCHED sleeper.
//
// INTERRUPTIONS -- NONE of these end the lock (the move desc says otherwise; the desc is WRONG
// for Showdown-gen3). Each still ticks the duration at the residual:
//   MISS      -> `|move|<u>|Uproar|<t>|[from] lockedmove|[miss]` + `|-miss|`, acc draw only
//   IMMUNE    -> `|-immune|<t>`, acc draw only  (and a CAST into a Ghost applies NO volatile at all:
//                no `-start`, no duration draw, a NORMAL request next turn, PP still spent)
//   PROTECT   -> `|-activate|<t>|Protect`, acc draw only
//   full PARA / FLINCH -> `|cant|<u>|par` / `|cant|<u>|flinch`, no acc/crit/dmg, no PP
//   target FAINTS -> the lock survives the forced replacement and re-targets the entrant
//   the uproarer PHAZED out -> clearVolatile, SILENT (no `-end` line)
//
// Run: node /tmp/probe_uproar.js [section]   (sections: lock sleep wake end interrupt all)
const path=require('path'); const PS='/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const {BattleStream}=require(path.join(PS,'dist/sim/battle-stream.js'));
const {Dex}=require(path.join(PS,'dist/sim/dex.js'));
const {PRNG}=require(path.join(PS,'dist/sim/prng'));      // SOLE path to rng.next(); wrapping
const {Battle}=require(path.join(PS,'dist/sim/battle.js'));// randomChance TOO would double-count
let draws=[];
{ const o=PRNG.prototype.random;
  PRNG.prototype.random=function(...a){ const r=o.apply(this,a); draws.push(`random(${a})->${r}`); return r; }; }
let traceOn=false, traced=[];
{ const o=Battle.prototype.add;
  Battle.prototype.add=function(...a){
    if(traceOn) traced.push({args:a.map(x=>x&&x.name?x.name:String(x)).join('|').slice(0,70),
      st:(new Error().stack||'').split('\n').slice(2,6).map(s=>s.trim().replace(/^at /,'').replace(/ \(.*\)$/,''))
         .filter(s=>!s.startsWith('Battle.add')).slice(0,3).join(' <- ')});
    return o.apply(this,a); }; }
const KEEP=/^\|(-start|-end|-fail|-status|-curestatus|-activate|-immune|-miss|-damage|-heal|move|turn|cant|faint|switch|drag)\|/;
async function newBattle(p1,p2,seed='[9,9,9,9]',fmt='gen3customgame'){
  const s=new BattleStream(); const ch=[];
  (async()=>{ for await(const c of s) ch.push(c); })();
  s.write(`>start {"formatid":"${fmt}","seed":${seed}}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r=>setTimeout(r,300)); return {s,ch};
}
async function step(s,ch,cursor,cmd){
  const b=draws.length, tb=traced.length; s.write(cmd); await new Promise(r=>setTimeout(r,250));
  const omni=[],reqs={},errs=[];
  for(const c of ch.slice(cursor)){
    if(c.startsWith('sideupdate')){ const ls=c.split('\n');
      for(const l of ls.slice(2)){ if(l.startsWith('|request|')) reqs[ls[1]]=l.slice(9); if(l.startsWith('|error|')) errs.push(ls[1]+' '+l); } }
    else for(const l of c.split('\n')) if(l.startsWith('|')&&!l.startsWith('|t:|')) omni.push(l);
  }
  return {omni,reqs,errs,next:ch.length,draws:draws.slice(b),traced:traced.slice(tb)};
}
async function scen(label,p1,p2,script,{seed='[9,9,9,9]',fmt='gen3customgame',req=false,raw=false}={}){
  const {s,ch}=await newBattle(p1,p2,seed,fmt); let cur=ch.length; draws.length=0;
  console.log(`\n=========== ${label}   [${fmt} ${seed}]`);
  const out=[];
  for(const [i,cmd] of script.entries()){
    const r=await step(s,ch,cur,cmd); cur=r.next; out.push(r);
    console.log(` T${i+1} [${cmd.replace(/\n/g,' / ')}]\n    ${(raw?r.omni:r.omni.filter(l=>KEEP.test(l))).join('\n    ')}`);
    console.log('    DRAWS:', r.draws.join('  ')||'(none)');
    if(req&&r.reqs.p1) console.log('    REQ p1 active:',JSON.stringify(JSON.parse(r.reqs.p1).active));
    for(const e of r.errs) console.log('    ERR:',e);
  }
  return {s,ch,out,cur};
}
// --- teams -------------------------------------------------------------------
const UPW  ='Blissey||Leftovers|NaturalCure|uproar,splash,rest,spore|Hardy|85,0,85,85,85,85|F|0,0,31,31,31,31|||'; // weak uproarer
const UPW2 = UPW+']Vaporeon||Leftovers|WaterAbsorb|splash,rest,tackle,tackle|Hardy|85,85,85,85,85,85|M||||';
const UPS  ='Snorlax||Leftovers|Immunity|uproar,splash,rest,tackle|Hardy|85,85,85,85,85,85|M||||';
const STEEL='Steelix||Leftovers|RockHead|spore,rest,splash,yawn|Hardy|85,85,85,85,85,85|M||||';
const PROT ='Steelix||Leftovers|RockHead|protect,rest,splash,tackle|Hardy|85,85,85,85,85,85|M||||';
const EVA  ='Steelix||Leftovers|RockHead|doubleteam,rest,splash,tackle|Hardy|85,85,85,85,85,85|M||||';
const TWAVE='Steelix||Leftovers|RockHead|thunderwave,rest,splash,tackle|Hardy|85,85,85,85,85,85|M||||';
const ROAR ='Steelix||Leftovers|RockHead|roar,rest,splash,tackle|Hardy|85,85,85,85,85,85|M||||';
const GHOST='Gengar||Leftovers|Levitate|splash,protect,rest,spore|Hardy|85,85,85,85,85,85|M|31,31,31,31,31,0|||';
const FRAIL='Magikarp||Leftovers|SwiftSwim|splash,rest,tackle,tackle|Hardy|0,0,0,0,0,0|M|0,0,0,0,0,0|||]Vaporeon||Leftovers|WaterAbsorb|splash,rest,tackle,tackle|Hardy|85,85,85,85,85,85|M||||';
const RESTER='Blissey||Leftovers|NaturalCure|rest,splash,spore,tackle|Hardy|85,85,85,85,85,85|F||||';
const U5=['>p1 move 1\n>p2 move 3','>p1 move 1\n>p2 move 3','>p1 move 1\n>p2 move 3','>p1 move 1\n>p2 move 3','>p1 move 1\n>p2 move 3','>p1 move 1\n>p2 move 3'];
// -----------------------------------------------------------------------------
(async()=>{
 const S=(process.argv[2]||'all');
 if(S=='all'||S=='lock'){
  console.log('\n########## 1. THE LOCK — duration draw, [upkeep] ticks, the locked |request|');
  for(const sd of ['[11,22,33,44]','[9,9,9,9]','[42,42,42,42]'])
    await scen('lock lifecycle (duration = the number of uproar uses)',UPW,STEEL,U5,{seed:sd,req:true});
  console.log('\n-- PP: one deduction for the WHOLE lock (watch pp in the post-lock request) --');
  await scen('PP across a 5-turn lock',UPW,STEEL,U5,{seed:'[42,42,42,42]',req:true});
  console.log('\n-- the reject forms while locked --');
  { const {s,ch,cur}=await scen('reject probes',UPW2,STEEL,['>p1 move 1\n>p2 move 3'],{req:true});
    let c=cur;
    for(const bad of ['>p1 move 2','>p1 switch 2']){ const r=await step(s,ch,c,bad); c=r.next;
      console.log(`   sent "${bad}" ->`, r.errs.join(' ')||'(no output)'); } }
 }
 if(S=='all'||S=='sleep'){
  console.log('\n########## 2. THE SLEEP BLOCK');
  await scen('Spore -> the UPROARER ([msg] form)',UPW,STEEL,['>p1 move 1\n>p2 move 3','>p1 move 1\n>p2 move 1','>p1 move 1\n>p2 move 1']);
  await scen('Spore -> the UPROARER, gen3ou (shuffle range 1,3)',UPW,STEEL,['>p1 move 1\n>p2 move 3','>p1 move 1\n>p2 move 1'],{fmt:'gen3ou'});
  await scen('CONTROL Spore lands, no uproar, gen3ou (shuffle range 0,2)',UPW,STEEL,['>p1 move 2\n>p2 move 1'],{fmt:'gen3ou'});
  await scen('a DAMAGED foe RESTs during uproar (no [msg]; no heal; no random(2,6))',UPS,RESTER,
    ['>p1 move 1\n>p2 move 2','>p1 move 1\n>p2 move 1'],{seed:'[42,42,42,42]',fmt:'gen3ou'});
  await scen('CONTROL the same Rest with no uproar, gen3ou',UPS,RESTER,
    ['>p1 move 4\n>p2 move 2','>p1 move 4\n>p2 move 1'],{seed:'[42,42,42,42]',fmt:'gen3ou'});
 }
 if(S=='all'||S=='wake'){
  console.log('\n########## 3. THE WAKE (move.onTryHit) — hit only, actives only');
  traceOn=true;
  await scen('asleep foe -> uproar CAST wakes it (traced below)',UPW,STEEL,['>p1 move 4\n>p2 move 3','>p1 move 1\n>p2 move 3']);
  traceOn=false;
  for(const t of traced) if(/curestatus|-start\|.*Uproar/.test(t.args)) console.log('   ATTRIB',t.args.padEnd(46),'<-',t.st);
  await scen('asleep GHOST (Normal-IMMUNE) is NOT woken',UPW,GHOST,
    ['>p1 move 4\n>p2 move 1','>p1 move 1\n>p2 move 1','>p1 move 1\n>p2 move 1','>p1 move 1\n>p2 move 1'],{seed:'[42,42,42,42]'});
  await scen('a BENCHED sleeper is NOT woken (it wakes only when it is ACTIVE and hit)',UPW,
    STEEL+']Vaporeon||Leftovers|WaterAbsorb|splash,rest,tackle,tackle|Hardy|85,85,85,85,85,85|M||||',
    ['>p1 move 4\n>p2 move 3','>p1 move 1\n>p2 switch 2','>p1 move 1\n>p2 switch 2']);
 }
 if(S=='all'||S=='end'){
  console.log('\n########## 4/5. THE END — `-end` at 10/11, NO confusion; and the YAWN edge');
  await scen('RAW final locked turn (grep for confusion: there is none)',UPW,STEEL,U5.slice(0,3),{seed:'[11,22,33,44]',raw:true});
  console.log('\n-- Yawn resolving DURING the lock is blocked; on the EXPIRY residual it LANDS --');
  for(const [sd,D] of [['[9,9,9,9]',4],['[11,22,33,44]',3]]){
    const sc=[]; for(let t=1;t<=D+1;t++) sc.push(`>p1 move 1\n>p2 move ${t===D-1?4:3}`);
    await scen(`Yawn cast on turn ${D-1} so it resolves on the uproar-EXPIRY residual`,UPW,STEEL,sc,{seed:sd});
  }
 }
 if(S=='all'||S=='interrupt'){
  console.log('\n########## 6. INTERRUPTIONS — none of these end the lock');
  await scen('MISS mid-lock (Double Team)',UPW,EVA,U5.map(c=>c.replace('move 3','move 1')),{seed:'[42,42,42,42]'});
  await scen('IMMUNE mid-lock (Ghost switches in); and an immune CAST never locks',UPW,
    STEEL+']'+GHOST,['>p1 move 1\n>p2 move 3','>p1 move 1\n>p2 switch 2','>p1 move 1\n>p2 move 1','>p1 move 1\n>p2 move 1'],
    {seed:'[42,42,42,42]',req:true});
  await scen('PROTECT mid-lock',UPW,PROT,['>p1 move 1\n>p2 move 3','>p1 move 1\n>p2 move 1','>p1 move 1\n>p2 move 3'],{seed:'[42,42,42,42]'});
  await scen('full PARA mid-lock',UPW,TWAVE,['>p1 move 1\n>p2 move 1','>p1 move 1\n>p2 move 3','>p1 move 1\n>p2 move 3','>p1 move 1\n>p2 move 3','>p1 move 1\n>p2 move 3']);
  await scen('target FAINTS mid-lock -> replacement, lock re-targets',UPS,FRAIL,
    ['>p1 move 1\n>p2 move 1','>p2 switch 2','>p1 move 1\n>p2 move 1'],{seed:'[42,42,42,42]',req:true});
  await scen('the uproarer is ROARed out -> volatile cleared SILENTLY (no -end)',UPW2,ROAR,
    ['>p1 move 1\n>p2 move 3','>p1 move 1\n>p2 move 1','>p1 move 1\n>p2 move 3'],{seed:'[42,42,42,42]',req:true});
  await scen('MIRROR uproar at a speed TIE -> ONE extra residual handler-sort shuffle random(2,4)',UPS,UPS,
    ['>p1 move 1\n>p2 move 1','>p1 move 1\n>p2 move 1'],{seed:'[42,42,42,42]'});
  await scen('CONTROL the same mirror, both Splash (no random(2,4))',UPS,UPS,
    ['>p1 move 2\n>p2 move 2','>p1 move 2\n>p2 move 2'],{seed:'[42,42,42,42]'});
 }
})();
