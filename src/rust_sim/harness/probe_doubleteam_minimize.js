// PROBE: gen-3 SELF-EVASION — DOUBLE TEAM + MINIMIZE. Draw model, emissions, the +6 cap,
// the DEFENDER-evasion stage table, and what Minimize's `volatileStatus` ACTUALLY does in gen 3.
//
// SETTLED 2026-08-18 (run it to re-confirm; do not re-derive from source):
//
//  1. DOUBLE TEAM IS A PURE DECLARATIVE SELF-BOOST. Resolved `Dex.mod('gen3')`:
//     `acc=true cat=Status bp=0 target=self boosts={evasion:1} volatileStatus=undefined
//      self=undefined secondary=undefined onHit/onTryHit/heal=undefined selfdestruct=undefined
//      flags={snatch:1, metronome:1}` — structurally IDENTICAL to Swords Dance, only the stat
//     differs (boost index 6). NEVER-MISS ⇒ NO accuracy draw; the `boost()` apply is DRAW-FREE.
//        plain cast   -> `|move|<u>|Double Team|<u>` + `|-boost|<u>|evasion|1`, 0 draws
//        7th cast     -> STILL succeeds, emits the DELTA-0 line `|-boost|<u>|evasion|0`, no `-fail`
//        Clear Body / White Smoke / Keen Eye cast it on THEMSELVES normally (`onTryBoost`
//          early-returns on `target === source`; nothing in gen3 blocks an evasion RAISE)
//
//  2. MINIMIZE IS **NOT** A PURE SELF-BOOST — its volatile is LIVE IN GEN 3.
//     Same declarative shape PLUS `volatileStatus: 'minimize'`, whose resolved gen-3 condition is
//     `{noCopy: true, onRestart: () => null, onSourceModifyDamage, onAccuracy: UNDEFINED}`:
//       * **onAccuracy is UNDEFINED in gen 3** — the later-gen "listed moves can't miss a
//         minimized target" bypass DOES NOT EXIST here (measured: Stomp into +3 evasion still
//         draws `randomChance(50,100)` and MISSES).
//       * **onSourceModifyDamage doubles the damage of any move with `flags.minimize`.**
//         `if (move.flags['minimize']) return this.chainModify(2);` — a FLAG, not an id list.
//         GEN-3-LEGAL CARRIERS (exactly four): **stomp, astonish, extrasensory, needlearm**.
//         ⚠️ NOT bodyslam/dragonrush/heatcrash/heavyslam — those gained the flag in gen 9.
//     MEASURED, the controlled pair (both give +1 evasion; only Minimize adds the volatile),
//     same seed, byte-identical draw streams:
//        Stomp        into +1 eva via Double Team -> 86 dmg | via Minimize -> 172   (EXACTLY x2)
//        Extrasensory into +1 eva via Double Team -> 31 dmg | via Minimize ->  62   (EXACTLY x2)
//        Tackle (no `flags.minimize`) -> 48 dmg in BOTH                              (no change)
//     So the x2 is DRAW-FREE and DAMAGE-ONLY; it never alters the draw count or order.
//        cast         -> `|move|<u>|Minimize|<u>` + `|-boost|<u>|evasion|1`. **NO `-start` line**
//                        (the condition has NO `onStart`) — the volatile is added SILENTLY.
//        re-cast      -> boosts again (`evasion|1`); `onRestart: () => null` leaves it unchanged
//        7th cast     -> the delta-0 `|-boost|<u>|evasion|0`, exactly like Double Team
//        switch out   -> volatile CLEARED, SILENTLY (no `-end`); re-entry does NOT restore the x2
//        **HAZE**     -> clears the evasion BOOST but **NOT** the volatile: the x2 SURVIVES
//                        (measured 178 vs the Double-Team control's 89 on the post-Haze turn)
//        BATON PASS   -> `noCopy: true` ⇒ the volatile is NOT passed (the eva STAGE is)
//        SUBSTITUTE   -> the DOUBLED number hits the sub (measured: it BREAKS a 120-HP sub
//                        that the un-doubled hit only chips)
//
//  3. THE DEFENDER-EVASION STAGE TABLE, read off the `randomChance(acc,100)` ARGS
//     (Scratch, acc 100) — IDENTICAL via Double Team and via Minimize, and identical to the
//     attacker-accuracy side `harness/probe_sandattack.js` measured:
//        +0 -> 100 | +1 -> 75 | +2 -> 60 | +3 -> 50
//        +4 -> 42.857142857142854 | +5 -> 37.5 | +6 -> 33.333333333333336
//     i.e. `acc /= boostTable[s]`, boostTable = [1, 4/3, 5/3, 2, 7/3, 8/3, 3] — exactly the
//     DIVIDE direction `src/turn/speed.rs::effective_accuracy` already implements for boosts[6].
//
//  4. NOTHING IN GEN 3 BLOCKS AN EVASION RAISE OR IGNORES EVASION.
//     Zero gen3 moves carry `ignoreEvasion`/`ignoreAccuracy`. The only gen3 `onTryBoost`
//     abilities are Clear Body / White Smoke / Hyper Cutter / Keen Eye, and ALL FOUR open with
//     `if (source && target === source) return;` — they gate FOE-inflicted DROPS only (Keen Eye
//     is `boost.accuracy < 0`). No gen3 ability has `onModifyBoost`. Bright Powder / Lax Incense
//     touch ModifyAccuracy, not boosts — already modeled by the accMod pipeline.
//
// EXPOSURE (MEASURED — the honest part): **ZERO on every surface we play.** gen3ou's rule table
// carries `-move:minimize` and `-move:doubleteam` (Standard -> Evasion Moves Clause), so they are
// BANNED in the training pool, in `ourandom` (which validates through TeamValidator('gen3ou')),
// and in the e2e/pool corpus: 0 of 773 `data/teams/` files mention either, and their gen3ou
// Smogon move-slot prior mass is 0.000 (vs sandattack's 0.723). 0 of 393 curated gen3-randbats
// sets carry either. They are reachable only in a clause-free format (`gen3customgame`) — i.e.
// the `--mode random` fuzz surface and hand-built teams. This is a LATENT-HAZARD unlock, NOT an
// active gap: unlike Sand Attack, nothing measured is waiting on it.
const path=require('path'); const PS='/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const {BattleStream}=require(path.join(PS,'dist/sim/battle-stream.js'));
const {Dex}=require(path.join(PS,'dist/sim/dex.js'));
const {PRNG}=require(path.join(PS,'dist/sim/prng'));
const g3=Dex.mod('gen3');

// ---------------------------------------------------------------- 0: DEX FACTS
console.log('######## 0: DEX FACTS (resolved Dex.mod("gen3")) ########');
for(const id of ['doubleteam','minimize','swordsdance']){
  const d=g3.moves.get(id);
  console.log(`  ${id.padEnd(12)} acc=${d.accuracy} cat=${d.category} bp=${d.basePower} target=${d.target} `
    +`boosts=${JSON.stringify(d.boosts)} volatileStatus=${d.volatileStatus} self=${JSON.stringify(d.self)} `
    +`secondary=${JSON.stringify(d.secondary)} onHit=${typeof d.onHit} onTryHit=${typeof d.onTryHit} `
    +`heal=${JSON.stringify(d.heal)} selfdestruct=${d.selfdestruct} flags=${JSON.stringify(d.flags)}`);
  if(d.condition) for(const k of Object.keys(d.condition))
    console.log(`      condition.${k} = ${typeof d.condition[k]==='function'?String(d.condition[k]).replace(/\s+/g,' '):JSON.stringify(d.condition[k])}`);
}
console.log('\n  -- gen3-LEGAL carriers of `flags.minimize` (what the x2 applies to) --');
console.log('   ', g3.moves.all().filter(m=>m.exists&&!m.isNonstandard&&m.gen<=3&&m.flags&&m.flags.minimize)
  .map(m=>`${m.id}(bp${m.basePower} ${m.type})`).join(', '));
console.log('  -- the GEN 9 list, for contrast (do NOT use it) --');
console.log('   ', Dex.mod('gen9').moves.all().filter(m=>m.exists&&m.flags&&m.flags.minimize).map(m=>m.id).join(', '));
console.log('\n  -- anything that ignores evasion or blocks an evasion RAISE? --');
console.log('    moves w/ ignoreEvasion|ignoreAccuracy:',
  g3.moves.all().filter(m=>m.exists&&!m.isNonstandard&&m.gen<=3&&(m.ignoreEvasion||m.ignoreAccuracy)).map(m=>m.id).join(', ')||'(NONE)');
for(const a of g3.abilities.all()){
  if(!a.exists||a.isNonstandard||a.gen>3) continue;
  if(a.onTryBoost) console.log(`    ${a.id}.onTryBoost = ${String(a.onTryBoost).replace(/\s+/g,' ').slice(0,110)}...`);
  if(a.onModifyBoost) console.log(`    ${a.id}.onModifyBoost EXISTS (would break the raw-stage read)`);
}

// ---------------------------------------------------------------- driver
// PASS A wraps ONLY `random` (the sole path to rng.next()) -> honest DRAW COUNTS.
// PASS B additionally wraps `randomChance` -> the folded-accuracy ARGS, at the cost of
// double-counting (randomChance calls random internally). Never read counts from pass B.
let draws=[]; let argsMode=false;
{const o=PRNG.prototype.random;
 PRNG.prototype.random=function(...a){const r=o.apply(this,a);draws.push(`random(${a})->${r}`);return r;};}
{const o=PRNG.prototype.randomChance;
 PRNG.prototype.randomChance=function(...a){const r=o.apply(this,a);
   if(argsMode) draws.push(`randomChance(${a})->${r}`); return r;};}
async function run(label,p1,p2,script,opts={}){
  draws=[]; argsMode=!!opts.argsOnly; const s=new BattleStream(); const ch=[];
  (async()=>{ for await(const c of s) ch.push(c); })();
  s.write(`>start {"formatid":"gen3customgame","seed":[9,9,9,9]}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r=>setTimeout(r,200)); let base=draws.length; const per=[];
  for(const c of script){ s.write(c); await new Promise(r=>setTimeout(r,200));
    per.push(draws.slice(base)); base=draws.length; }
  await new Promise(r=>setTimeout(r,220)); argsMode=false;
  const all=[...new Set(ch.filter(c=>!c.startsWith('sideupdate')).join('\n').split('\n')
    .filter(l=>/^\|(-start|-end|-fail|-immune|-activate|-boost|-unboost|-miss|-crit|-damage|switch|move|turn)\|/.test(l)))];
  const last=per[per.length-1]||[];
  if(opts.argsOnly){
    const acc=last.find(d=>/^randomChance\(.*,100\)->/.test(d));
    console.log(`  ${label.padEnd(30)} TO-HIT: ${acc||'(NO ACCURACY DRAW)'}   ${all.some(l=>l.startsWith('|-miss|'))?'MISS':''}`);
    return;
  }
  const i=all.lastIndexOf('|turn|'+per.length);
  console.log(`\n== ${label}\n  ${all.slice(opts.allLines?0:i).join('\n  ')}`);
  console.log('  DRAWS[final turn]: '+(last.join('  ')||'(none)'));
}
// p1 1=stomp 2=tackle 3=scratch 4=splash 5(alt team)=extrasensory
const ATK ='Miltank||Leftovers|ThickFat|stomp,tackle,scratch,splash|Hardy|85,85,85,85,85,85|F||||';
const ATK2='Miltank||Leftovers|ThickFat|stomp,extrasensory,haze,splash|Hardy|85,85,85,85,85,85|F||||';
// p2 1=doubleteam 2=minimize 3=splash 4=batonpass   (+ a Blissey bench for the switch tests)
const DEF ='Snorlax||Leftovers|Immunity|doubleteam,minimize,splash,batonpass|Hardy|85,85,85,85,85,85|M||||]Blissey||Leftovers|NaturalCure|splash,stomp,haze,protect|Hardy|85,85,85,85,85,85|F||||';
const CB  ='Metagross||Leftovers|ClearBody|doubleteam,minimize,splash,protect|Hardy|85,85,85,85,85,85|N||||';
const KE  ='Skarmory||Leftovers|KeenEye|doubleteam,minimize,splash,protect|Hardy|85,85,85,85,85,85|M||||';
const WS  ='Torkoal||Leftovers|WhiteSmoke|doubleteam,minimize,splash,protect|Hardy|85,85,85,85,85,85|M||||';
const rep=(n,c)=>new Array(n).fill(c);
const DT=n=>rep(n,'>p1 move 4\n>p2 move 1'), MZ=n=>rep(n,'>p1 move 4\n>p2 move 2');

(async()=>{
  console.log('\n######## A: DOUBLE TEAM — draw model + emissions ########');
  await run('A1 plain cast',            ATK, DEF, DT(1));
  await run('A2 SEVENTH cast (+6 cap)',  ATK, DEF, DT(7));
  await run('A3 cast by CLEAR BODY',     ATK, CB,  DT(1));
  await run('A4 cast by WHITE SMOKE',    ATK, WS,  DT(1));
  await run('A5 cast by KEEN EYE',       ATK, KE,  DT(1));

  console.log('\n######## B: MINIMIZE — draw model + emissions (note: NO `-start` line) ########');
  await run('B1 plain cast',            ATK, DEF, MZ(1));
  await run('B2 SEVENTH cast (+6 cap)',  ATK, DEF, MZ(7));
  await run('B3 re-cast (onRestart)',    ATK, DEF, MZ(2));

  console.log('\n######## C: what the VOLATILE does — the controlled +1-evasion pair ########');
  await run('C1 STOMP vs +1 eva via DOUBLE TEAM', ATK, DEF, [...DT(1),'>p1 move 1\n>p2 move 3']);
  await run('C2 STOMP vs +1 eva via MINIMIZE   ', ATK, DEF, [...MZ(1),'>p1 move 1\n>p2 move 3']);
  await run('C3 TACKLE (no flags.minimize) vs DT', ATK, DEF, [...DT(1),'>p1 move 2\n>p2 move 3']);
  await run('C4 TACKLE (no flags.minimize) vs MZ', ATK, DEF, [...MZ(1),'>p1 move 2\n>p2 move 3']);
  await run('C5 EXTRASENSORY vs DT',              ATK2,DEF, [...DT(1),'>p1 move 2\n>p2 move 3']);
  await run('C6 EXTRASENSORY vs MZ',              ATK2,DEF, [...MZ(1),'>p1 move 2\n>p2 move 3']);

  console.log('\n######## D: the DEFENDER-EVASION stage table off the randomChance ARGS (Scratch acc 100) ########');
  for(const n of [0,1,2,3,4,5,6]) await run(`eva +${n} (DT)`, ATK, DEF, [...DT(n),'>p1 move 3\n>p2 move 3'], {argsOnly:1});
  for(const n of [0,1,2,3,4,5,6]) await run(`eva +${n} (MZ)`, ATK, DEF, [...MZ(n),'>p1 move 3\n>p2 move 3'], {argsOnly:1});
  console.log('  -- STOMP into MINIMIZE: gen3 has NO onAccuracy bypass, so it CAN still miss --');
  for(const n of [1,3]) await run(`STOMP eva +${n} (MZ)`, ATK, DEF, [...MZ(n),'>p1 move 1\n>p2 move 3'], {argsOnly:1});

  console.log('\n######## E: the volatile LIFECYCLE — switch-out / HAZE / Baton Pass ########');
  await run('E1 MZ -> switch out -> back -> STOMP', ATK2, DEF,
    [...MZ(1),'>p1 move 4\n>p2 switch 2','>p1 move 4\n>p2 switch 2','>p1 move 1\n>p2 move 1'], {allLines:1});
  await run('E2 MZ -> p1 HAZE -> STOMP  (x2 SURVIVES Haze)', ATK2, DEF, [...MZ(1),'>p1 move 3\n>p2 move 3','>p1 move 1\n>p2 move 3']);
  await run('E2c CONTROL DT -> p1 HAZE -> STOMP',            ATK2, DEF, [...DT(1),'>p1 move 3\n>p2 move 3','>p1 move 1\n>p2 move 3']);
  await run('E3 MZ -> BATON PASS -> STOMP into entrant', ATK2, DEF,
    [...MZ(1),'>p1 move 4\n>p2 move 4','>p2 switch 2','>p1 move 1\n>p2 move 1']);
  await run('E3c CONTROL no boost -> BATON PASS -> STOMP', ATK2, DEF,
    ['>p1 move 4\n>p2 move 3','>p1 move 4\n>p2 move 4','>p2 switch 2','>p1 move 1\n>p2 move 1']);

  console.log('\n######## F: f64 the port must reproduce (boostTable, DIVIDE direction) ########');
  const T=[1,4/3,5/3,2,7/3,8/3,3];
  for(let s=1;s<=6;s++) console.log(`  100/T[${s}] = ${100/T[s]}`);
  process.exit(0);
})();
