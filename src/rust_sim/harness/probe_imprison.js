// PROBE: gen-3 IMPRISON — restriction / duration / emissions / draw model / request JSON / edges.
//
// SETTLED 2026-08-18 (run it to re-confirm; do not re-derive from source):
//   CAST (a shared move exists)  : DRAW-FREE -> |move|<u>|Imprison|<u>  +  |-start|<u>|move: Imprison
//   CAST (no shared move)        : DRAW-FREE -> |move|<u>|Imprison||[still]  +  |-fail|<u>
//   RE-CAST (already imprisoned) : DRAW-FREE -> the same [still]+-fail  + |debug|move failed because it did nothing
//   BLOCKED foe move (queued)    : DRAW-FREE -> |cant|<foe>|move: Imprison|<Move>, NO PP spent
//   there is NO -end line, NO duration, NO accuracy roll and NO residual tick.
const path=require('path'); const PS='/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const {BattleStream}=require(path.join(PS,'dist/sim/battle-stream.js'));
const {PRNG}=require(path.join(PS,'dist/sim/prng'));
let draws=[];
{ const o=PRNG.prototype.random;                       // the SOLE path to rng.next()
  PRNG.prototype.random=function(...a){ const r=o.apply(this,a); draws.push(`random(${a})->${r}`); return r; }; }
const S=(n,sp,mv,it)=>`${n}|${sp}|${it||'Leftovers'}|Insomnia|${mv}|Hardy|85,85,85,85,85,85|M||||`;
async function run(label,p1,p2,script,seed='[9,9,9,9]'){
  draws=[]; const s=new BattleStream(); const ch=[];
  (async()=>{ for await(const c of s) ch.push(c); })();
  s.write(`>start {"formatid":"gen3customgame","seed":${seed}}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r=>setTimeout(r,200));
  console.log(`\n===== ${label}`);
  for(const c of script){ const b=draws.length, cb=ch.length;
    s.write(c); await new Promise(r=>setTimeout(r,200));
    const nu=ch.slice(cb);
    console.log(`  > ${c.replace(/\n/g,' ; ')}   [draws ${draws.length-b}: ${draws.slice(b).join(' ')||'NONE'}]`);
    nu.filter(x=>!x.startsWith('sideupdate')).join('\n').split('\n')      // the chronological omniscient log
      .filter(l=>l.trim()&&!/^\|(t:|upkeep|split|-heal|update|$)/.test(l)).forEach(l=>console.log('      '+l));
    for (const cc of nu.filter(x=>x.startsWith('sideupdate'))) {
      const L=cc.split('\n'), side=L[1];
      for (const l of L) {
        if (l.startsWith('|error|')) console.log(`      ${side} ${l}`);
        else if (l.startsWith('|-activate|')) console.log(`      ${side} SIDE-ONLY ${l}`);
        else if (l.startsWith('|request|')) { const r=JSON.parse(l.slice(9)); if(r.wait) continue;
          const a=(r.active&&r.active[0])||{};
          console.log(`      ${side} REQ ${a.moves?JSON.stringify(a.moves):'forceSwitch'}`
            +` maybeDisabled=${a.maybeDisabled} maybeLocked=${a.maybeLocked}`); }
      }
    }
  }
}
(async()=>{
  const user     = S('Imp','Alakazam','imprison,icebeam,splash,thunderbolt');            // fast
  const share    = S('Foe','Snorlax','icebeam,bodyslam,splash,rest');                    // slow; icebeam+splash shared
  const noShare  = S('Foe','Snorlax','bodyslam,rest,earthquake,curse');
  const allShare = S('Foe','Snorlax','icebeam,splash,thunderbolt,imprison');
  const oneShare = S('Foe','Snorlax','icebeam');

  await run('Q1a CAST, shared move exists',            user, share,   ['>p1 move 1\n>p2 move 2']);
  await run('Q1a CONTROL (Splash, identical board)',   user, share,   ['>p1 move 3\n>p2 move 2']);
  await run('Q1b CAST, NO shared move -> fail',        user, noShare, ['>p1 move 1\n>p2 move 1']);
  await run('Q3  RE-CAST while already imprisoned',    user, share,   ['>p1 move 1\n>p2 move 2','>p1 move 3\n>p2 move 3']);
  await run('Q3  BLOCKED queued move -> |cant| (PP unspent)', user, share, ['>p1 move 1\n>p2 move 1']);
  await run('Q5  REJECT + re-request for a NEXT-turn imprisoned pick', user, share,
    ['>p1 move 1\n>p2 move 2','>p1 move 3\n>p2 move 1']);
  await run('Q5  ALL foe moves imprisoned -> the request still lists them; move N becomes STRUGGLE',
    user, allShare, ['>p1 move 1\n>p2 move 2','>p1 move 3\n>p2 move 1','>p1 move 3\n>p2 move struggle']);
  await run('Q5  SINGLE-move foe (its only move shared)', user, oneShare, ['>p1 move 1\n>p2 move 1','>p1 move 3\n>p2 move 1']);
  await run('Q2  DURATION: the USER switches out -> the foe is freed',
    user+']'+S('Alt','Machamp','crosschop,rest,earthquake,curse'), share,
    ['>p1 move 1\n>p2 move 2','>p1 switch 2\n>p2 move 2','>p1 move 2\n>p2 move 1']);
  await run('Q2  the TARGET switches out -> the ENTRANT is imprisoned too',
    user, share+']'+S('Foe2','Gengar','icebeam,shadowball,splash,thunderbolt'),
    ['>p1 move 1\n>p2 move 2','>p1 move 3\n>p2 switch 2','>p1 move 3\n>p2 move 1']);
  await run('Q6  BATON PASS (noCopy) -> the entrant does NOT imprison',
    S('Imp','Alakazam','imprison,batonpass,splash,icebeam')+']'+S('Alt','Machamp','crosschop,rest,earthquake,curse'),
    share, ['>p1 move 1\n>p2 move 2','>p1 move 2\n>p2 move 2','>p1 switch 2','>p1 move 1\n>p2 move 1']);
  await run('Q6  CHOICE-LOCKED foe locked into a move that becomes imprisoned',
    user, S('Foe','Snorlax','icebeam,bodyslam,splash,rest','Choice Band'),
    ['>p1 move 3\n>p2 move 1','>p1 move 1\n>p2 move 1','>p1 move 3\n>p2 move 1']);
  await run('Q4  onFoeBeforeMove PRIORITY 4 > paralysis 1: the cant suppresses the para roll',
    S('Imp','Alakazam','imprison,thunderwave,splash,icebeam'), share,
    ['>p1 move 2\n>p2 move 3','>p1 move 1\n>p2 move 1']);
})();
