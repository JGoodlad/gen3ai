// PROBE: gen-3 TORMENT — duration, restriction, request shape, emissions, draw model, edges.
//
// SETTLED 2026-08-18 (run it to re-confirm; do not re-derive from source):
//   DURATION      : PERMANENT until the volatile is cleared. The gen-3 `torment` condition has
//                   `duration: undefined` and NO `onResidual` -> NO duration draw, NO residual
//                   handler, no residual duration-tie group membership. Verified live for 6 turns.
//   THE DRAW MODEL: the CAST draws EXACTLY ONE `random(100)` (accuracy 100, NOT never-miss) and
//                   NOTHING else. The volatile itself is DRAW-FREE. BUT its `onDisableMove`
//                   handler JOINS the endTurn `runEvent('DisableMove')` handler-sort tie group
//                   (with taunt / disable / choicelock / encore): n handlers on one mon -> n-1
//                   `random` draws per endTurn. MEASURED n=1 -> 0 draws; n=2 -> `random(0,2)`;
//                   n=3 -> `random(0,3)` + `random(1,3)`.
//   RESTRICTION   : SELECTION-TIME ONLY. `onDisableMove` disables `pokemon.lastMove.id` (skipping
//                   `struggle`), so the request marks that ONE move `"disabled":true`. There is NO
//                   `onBeforeMove` -> a move already SELECTED when torment lands mid-turn STILL
//                   EXECUTES (no `|cant|`). All moves unusable -> the request collapses to
//                   `[{"move":"Struggle","id":"struggle","target":"randomNormal","disabled":false}]`
//                   + an owner-only `|-activate|<mon>|move: Struggle`.
//   EMISSIONS     : cast `|move|<u>|Torment|<t>` then `|-start|<t>|Torment` (NO `move: ` prefix).
//                   re-cast into an already-tormented foe: `|move|<u>|Torment||[still]` +
//                   `|-fail|<u>` (the accuracy roll IS still drawn). Blocked selection ->
//                   `|error|[Unavailable choice] Can't move: <Mon>'s <Move> is disabled` + a
//                   re-request whose slot gains `"disabledSource":""`; a repeat -> `[Invalid
//                   choice]`, no re-request. `onEnd` (`|-end|<t>|Torment`) is UNREACHABLE in gen 3:
//                   nothing removes the volatile, and `clearVolatile` wipes `volatiles = {}`
//                   WITHOUT firing End -> a switch-out / faint clears it SILENTLY.
//   EDGES         : PROTECT blocks it (`-activate Protect`, after the accuracy roll, no volatile).
//                   SUBSTITUTE does NOT (`bypasssub: 1`). BATON PASS does NOT transfer it
//                   (`noCopy: true`) - VERIFIED, the entrant repeats a move freely.
//                   MOVE-LOCKED turns (Hyper Beam recharge, Solar Beam fire) are IMMUNE - the
//                   locked request carries no `disabled` key at all and the move executes; the
//                   lastMove is unchanged by the recharge `|cant|`, so the move is disabled the
//                   turn AFTER. CHOICE item stacks: CB-locked slot + torment on that same slot =
//                   every slot unusable -> the mon alternates <locked move> / Struggle forever
//                   (Struggle is exempt from `onDisableMove`, so the next turn frees the slot).
//                   SLEEP TALK: torment disables `sleeptalk` ITSELF, never the CALLED move -
//                   `runMove` sets `lastMove` for the OUTER move only, `useMove` does not.
const path=require('path'); const PS='/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const {BattleStream}=require(path.join(PS,'dist/sim/battle-stream.js'));
const {Dex}=require(path.join(PS,'dist/sim/dex.js'));
const {PRNG}=require(path.join(PS,'dist/sim/prng'));
const d=Dex.mod('gen3').moves.get('torment');
console.log('dex: acc',d.accuracy,'cat',d.category,'target',d.target,'prio',d.priority,
            'volatileStatus',d.volatileStatus,'flags',JSON.stringify(d.flags),
            'cond.duration',d.condition.duration,'cond.noCopy',d.condition.noCopy,
            'cond hooks',Object.keys(d.condition).filter(k=>typeof d.condition[k]==='function'));
let draws=[];
{ const o=PRNG.prototype.random;   // the SOLE path to rng.next(); wrapping randomChance too double-counts
  PRNG.prototype.random=function(...a){ const r=o.apply(this,a); draws.push(`random(${a})->${r}`); return r; }; }
const SP='Hardy|85,85,85,85,85,85|M||||';
const mon=(n,i,a,m)=>`${n}||${i}|${a}|${m}|${SP}`;
async function run(label,p1t,p2t,script){
  draws=[]; const s=new BattleStream(); const ch=[];
  (async()=>{ for await(const c of s) ch.push(c); })();
  s.write(`>start {"formatid":"gen3customgame","seed":[9,9,9,9]}\n>player p1 {"name":"P1","team":"${p1t}"}\n>player p2 {"name":"P2","team":"${p2t}"}`);
  await new Promise(r=>setTimeout(r,200));
  const marks=[{n:draws.length,cmd:'(start)'}];
  for(const c of script){ s.write(c); await new Promise(r=>setTimeout(r,200)); marks.push({n:draws.length,cmd:c.replace(/\n/g,' | ')}); }
  await new Promise(r=>setTimeout(r,250)); marks.push({n:draws.length,cmd:'(end)'});
  const omni=[]; const side={p1:[],p2:[]};
  for (const c of ch){ if (c.startsWith('sideupdate')){ const L=c.split('\n'); const w=L[1];
      for (const l of L.slice(2)) if(l) side[w].push(l); }        // per-side: keep, we need |request|
    else for (const l of c.split('\n')) if(l && !l.startsWith('update')) omni.push(l); }
  console.log(`\n================ ${label}`);
  for (const l of omni) if(/^\|(-start|-end|-fail|-immune|-activate|-miss|-prepare|move|cant|turn|switch|faint|error|-mustrecharge|-status|-curestatus|-singleturn|-damage)\|/.test(l)) console.log('   '+l);
  console.log('--- REQ p2');
  for (const l of side.p2) { if (l.startsWith('|request|')) { const j=JSON.parse(l.slice(9));
      if (j.wait) continue; if (j.forceSwitch) { console.log('   forceSwitch'); continue; }
      const a=j.active[0];
      console.log('   '+(a.moves||[]).map(m=>m.id+(m.disabled?'*DISABLED*':'')).join(' ')+(a.trapped?' [trapped]':'')); }
    else console.log('   '+l); }
  console.log('--- DRAWS per write');
  for (let k=1;k<marks.length;k++){ const seg=draws.slice(marks[k-1].n,marks[k].n);
    console.log(`   [${marks[k].cmd}] n=${seg.length} ${seg.join(' ')}`); }
}
(async()=>{
  const TORM=mon('Gengar','Leftovers','Levitate','torment,splash,nightshade,willowisp');
  const V4=mon('Snorlax','Leftovers','Immunity','tackle,scratch,pound,splash');
  const V1=mon('Snorlax','Leftovers','Immunity','tackle');
  const BLIS=mon('Blissey','Leftovers','NaturalCure','tackle,splash,pound,scratch');

  await run('A: cast + consecutive-use block + PERMANENCE (6 turns, no duration draw)', TORM, V4,
    ['>p1 move 1\n>p2 move 1','>p1 move 2\n>p2 move 2','>p1 move 2\n>p2 move 1',
     '>p1 move 2\n>p2 move 2','>p1 move 2\n>p2 move 1','>p1 move 2\n>p2 move 2']);
  await run('B: re-cast into an already-tormented foe ([still]+-fail, accuracy STILL drawn)', TORM, V4,
    ['>p1 move 1\n>p2 move 1','>p1 move 1\n>p2 move 2']);
  await run('C: 1-move victim -> forced Struggle -> struggle is EXEMPT -> move free again', TORM, V1,
    ['>p1 move 1\n>p2 move 1','>p1 move 2\n>p2 move 1','>p1 move 2\n>p2 move 1']);
  await run('D: illegal selection -> [Unavailable choice] + re-request w/ disabledSource', TORM,
    V4+']'+BLIS, ['>p1 move 1\n>p2 switch 2','>p1 move 2\n>p2 move 1','>p1 move 2\n>p2 move 1']);
  await run('E: switch-out clears SILENTLY (no -end), and does not come back', TORM, V4+']'+BLIS,
    ['>p1 move 1\n>p2 move 1','>p1 move 2\n>p2 switch 2','>p1 move 2\n>p2 switch 2','>p1 move 2\n>p2 move 1']);
  await run('F: selection-time ONLY — torment landing mid-turn does NOT cant a queued repeat',
    mon('Snorlax','Leftovers','Immunity','torment,splash,tackle,scratch'), V4,
    ['>p1 move 3\n>p2 move 1','>p1 move 1\n>p2 move 1']);
  await run('G-control: DisableMove n=1 (choicelock only) -> NO shuffle draw',
    mon('Gengar','Leftovers','Levitate','splash,torment,taunt,nightshade'),
    mon('Snorlax','ChoiceBand','Immunity','tackle,scratch,pound,splash'),
    ['>p1 move 1\n>p2 move 1','>p1 move 1\n>p2 move 1']);
  await run('G: DisableMove n=2 (torment+choicelock) -> ONE random(0,2); CB+torment = Struggle lock',
    TORM, mon('Snorlax','ChoiceBand','Immunity','tackle,scratch,pound,splash'),
    ['>p1 move 1\n>p2 move 1','>p1 move 2\n>p2 move 1']);
  await run('G3: DisableMove n=3 (torment+taunt+choicelock) -> random(0,3)+random(1,3)',
    mon('Gengar','Leftovers','Levitate','torment,taunt,splash,nightshade'),
    mon('Snorlax','ChoiceBand','Immunity','tackle,scratch,pound,splash'),
    ['>p1 move 1\n>p2 move 1','>p1 move 2\n>p2 move 1']);
  await run('H1: PROTECT blocks torment (accuracy drawn, -activate Protect, no volatile)', TORM,
    mon('Snorlax','Leftovers','Immunity','protect,tackle,scratch,splash'), ['>p1 move 1\n>p2 move 1']);
  await run('H2: SUBSTITUTE does NOT block (bypasssub)', TORM,
    mon('Snorlax','Leftovers','Immunity','substitute,tackle,scratch,splash'),
    ['>p1 move 2\n>p2 move 1','>p1 move 1\n>p2 move 2']);
  await run('I: BATON PASS does NOT transfer it (noCopy)', TORM,
    mon('Smeargle','Leftovers','OwnTempo','batonpass,tackle,scratch,splash')+']'+BLIS,
    ['>p1 move 1\n>p2 move 2','>p1 move 2\n>p2 move 1','>p2 switch 2',
     '>p1 move 2\n>p2 move 1','>p1 move 2\n>p2 move 1']);
  await run('J: SOLAR BEAM fire turn is IMMUNE (locked request has no `disabled` key)',
    mon('Gengar','Leftovers','Levitate','splash,torment,nightshade,willowisp'),
    mon('Snorlax','Leftovers','Immunity','solarbeam,tackle,scratch,splash'),
    ['>p1 move 1\n>p2 move 1','>p1 move 2\n>p2 move 1','>p1 move 1\n>p2 move 1']);
  await run('K: HYPER BEAM recharge turn is IMMUNE; lastMove survives the |cant|recharge',
    mon('Blissey','Leftovers','NaturalCure','splash,torment,pound,scratch'),
    mon('Snorlax','Leftovers','Immunity','hyperbeam,tackle,scratch,splash'),
    ['>p1 move 1\n>p2 move 1','>p1 move 2\n>p2 move 1','>p1 move 1\n>p2 move 1','>p1 move 1\n>p2 move 2']);
  await run('L: asleep SLEEP TALK — torment disables sleeptalk ITSELF, not the CALLED move',
    mon('Gengar','Leftovers','Levitate','psychic,torment,splash,nightshade'),
    mon('Snorlax','Leftovers','Immunity','rest,sleeptalk,tackle,scratch'),
    ['>p1 move 1\n>p2 move 1','>p1 move 2\n>p2 move 2','>p1 move 3\n>p2 move 2','>p1 move 3\n>p2 move 3']);
})();
