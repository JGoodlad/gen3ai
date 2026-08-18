// PROBE: gen-3 CONVERSION + CONVERSION 2 — draw model, selection set, emission, failure.
//
// SETTLED 2026-08-18 (run it to re-confirm; do not re-derive from source):
//   BOTH moves are category Status, accuracy `true` (NEVER-MISS => NO accuracy draw),
//   priority 0, and each consumes EXACTLY ONE PRNG draw when it SUCCEEDS: `random(n)`
//   (Showdown `this.sample(list)` == `random(list.length)`), and ZERO draws when it FAILS.
//
//   CONVERSION (num 160, target:self, flags {metronome})
//     list = the USER's LIVE move slots, IN SLOT ORDER, keeping `move.type` for every slot
//            whose id != "curse" AND whose type the user does NOT currently have.
//            DUPLICATES ARE KEPT (3 Ice moves => a 3-entry list => random(3)).
//     draw = ONE `random(n)`; n==1 STILL DRAWS.
//     ok   -> `|move|<u>|Conversion|<u>` + `|-start|<u>|typechange|<Type>`
//     n==0 -> `|move|<u>|Conversion||[still]` + `|-fail|<u>`, ZERO draws.
//
//   CONVERSION 2 (num 176, target:normal, flags {bypasssub, metronome})
//     keys off the TARGET's `lastMoveUsed` (reset by a switch-out); type-changes the SOURCE.
//     attackType = lastMoveUsed.type ("struggle" is special-cased to Normal; in gen 3
//                  struggle's dex type is ALREADY Normal, so the special case is a no-op).
//     list = every type T, in `Dex.mod('gen3').types.names()` ORDER, with
//            damageTaken[attackType] in {2 (resist), 3 (immune)}. It does NOT exclude the
//            source's own current types.
//     draw = ONE `random(n)`.
//     ok   -> `|move|<u>|Conversion 2|<foe>` + `|-start|<u>|typechange|<Type>`
//     fail -> `|move|<u>|Conversion 2||[still]` + `|-fail|<u>`, ZERO draws, when either
//             (a) the target has NO lastMoveUsed, or (b) attackType is `???` (Curse) => the
//             list is empty. Protect does NOT block it (no `protect` flag); Substitute does
//             not either (`bypasssub`); a Ghost target is not immune (Status ignoreImmunity).
//
//   BOTH: setType() REPLACES the whole type array with ONE type (Gengar Ghost/Poison ->
//   Ghost), takes effect immediately for STAB/chart/etc, and is REVERTED by a switch-out.
const path=require('path'); const PS='/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const {BattleStream}=require(path.join(PS,'dist/sim/battle-stream.js'));
const {Dex}=require(path.join(PS,'dist/sim/dex.js'));
const {PRNG}=require(path.join(PS,'dist/sim/prng.js'));
let draws=[];
for (const m of ['random','randomChance','sample']) { const o=PRNG.prototype[m];
  PRNG.prototype[m]=function(...a){ const r=o.apply(this,a);
    const arg=m==='sample'?'['+a[0].map(x=>(x&&x.name)||x).join(',')+']':a;
    draws.push(`${m}(${arg})->${m==='sample'?JSON.stringify((r&&r.name)||r):r}`); return r; }; }
async function run(label,p1,p2,script,seed){
  draws=[]; const s=new BattleStream(); const ch=[];
  (async()=>{ for await(const c of s) ch.push(c); })();
  s.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed||[9,9,9,9])}}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r=>setTimeout(r,200)); const base=draws.length;
  for(const c of script){ s.write(c); await new Promise(r=>setTimeout(r,200)); }
  await new Promise(r=>setTimeout(r,200));
  const omni=[...new Set(ch.filter(c=>!c.startsWith('sideupdate')))].join('\n').split('\n')
    .filter(l=>/^\|(-start|-fail|-immune|-activate|move|turn|switch)\|/.test(l));
  const i=omni.indexOf('|turn|1');
  console.log(`\n== ${label}  seed=${JSON.stringify(seed||[9,9,9,9])}`);
  console.log('  '+omni.slice(i+1).join('\n  '));
  console.log('  DRAWS:', draws.slice(base).join('  ')||'(none)');
  const a=s.battle.sides.map(sd=>sd.active[0]);
  console.log('  TYPES:', a.map(p=>p.name+'='+p.types.join('/')+'(last='+(p.lastMoveUsed?p.lastMoveUsed.id:'NONE')+')').join(' | '));
}
const T=(sp,mv,x)=>`${sp}||${(x&&x.item)||''}|${(x&&x.ability)||''}|${mv}|Hardy|85,85,85,85,85,85|M||||`;
(async()=>{
  const d=Dex.mod('gen3'), N=d.types.names();
  console.log('gen3 types.names() ORDER =',JSON.stringify(N));
  console.log('CONVERSION-2 CANDIDATE TABLE:');
  for (const a of N) console.log('  '+a.padEnd(9),JSON.stringify(N.filter(t=>[2,3].includes(d.types.get(t).damageTaken[a]))));

  const foe=T('Snorlax','splash,tackle,thunderwave,surf');
  // --- CONVERSION -------------------------------------------------------
  for (const seed of [[9,9,9,9],[1,2,3,4],[42,42,42,42]])
    await run('C1 conversion n=3 (Ice/Electric/Ghost) - RANDOM across seeds',
      T('Porygon2','conversion,icebeam,thunderbolt,shadowball'), foe, ['>p1 move 1\n>p2 move 1'], seed);
  await run('C2 conversion n=1 STILL DRAWS random(1)',
    T('Porygon2','conversion,icebeam,recover,splash'), foe, ['>p1 move 1\n>p2 move 1']);
  await run('C3 conversion n=0 -> [still]+-fail, ZERO draws',
    T('Porygon2','conversion,recover,splash,doubleedge'), foe, ['>p1 move 1\n>p2 move 1']);
  await run('C4 conversion keeps DUPLICATES in slot order -> random(3) over [Ice,Ice,Ice]',
    T('Porygon2','conversion,icebeam,blizzard,icywind'), foe, ['>p1 move 1\n>p2 move 1']);
  await run('C5 conversion: a typed-HP slot is stored BARE (type Normal) -> excluded',
    T('Porygon2','conversion,hiddenpowerice,recover,splash'), foe, ['>p1 move 1\n>p2 move 1']);
  await run('C6 conversion: CURSE is excluded by id',
    T('Porygon2','conversion,curse,recover,splash'), foe, ['>p1 move 1\n>p2 move 1']);
  await run('C7 conversion twice: the 2nd list is built vs the CURRENT types',
    T('Porygon2','conversion,icebeam,thunderbolt,shadowball'), foe,
    ['>p1 move 1\n>p2 move 1','>p1 move 1\n>p2 move 1']);
  await run('C8 conversion is REVERTED by a switch-out',
    T('Porygon2','conversion,icebeam,thunderbolt,shadowball')+']'+T('Blissey','splash,recover,tackle,icebeam'),
    foe, ['>p1 move 1\n>p2 move 1','>p1 switch 2\n>p2 move 1','>p1 switch 2\n>p2 move 1']);
  await run('C9 conversion reads the LIVE Mimic-overwritten slot',
    T('Shuckle','conversion,mimic,recover,splash'), T('Snorlax','icebeam,splash,tackle,surf'),
    ['>p1 move 2\n>p2 move 1','>p1 move 1\n>p2 move 2']);
  await run('C10 conversion excludes a COLOR-CHANGED live type (Ice Beam excluded)',
    T('Kecleon','conversion,icebeam,thunderbolt,shadowball',{ability:'Color Change'}),
    T('Snorlax','icebeam,splash,tackle,surf'), ['>p1 move 4\n>p2 move 1','>p1 move 1\n>p2 move 2']);
  // --- CONVERSION 2 -----------------------------------------------------
  const slow=T('Shuckle','conversion2,splash,recover,icebeam');
  for (const seed of [[9,9,9,9],[1,2,3,4],[42,42,42,42]])
    await run('D1 conv2 vs foe lastMove Tackle (Normal) -> [Ghost,Steel,Rock]', slow, foe,
      ['>p1 move 1\n>p2 move 2'], seed);
  await run('D2 conv2 with NO foe lastMoveUsed -> [still]+-fail, ZERO draws',
    T('Jolteon','conversion2,splash,recover,icebeam'), foe, ['>p1 move 1\n>p2 move 2']);
  await run('D3 conv2 vs foe lastMove Surf (Water) -> [Grass,Dragon,Water]', slow, foe,
    ['>p1 move 2\n>p2 move 4','>p1 move 1\n>p2 move 4']);
  await run('D4 conv2 vs foe lastMove CURSE (???) -> empty list -> FAIL, ZERO draws',
    slow, T('Snorlax','curse,splash,tackle,surf'), ['>p1 move 1\n>p2 move 1']);
  await run('D5 conv2 is NOT blocked by Protect', slow, T('Snorlax','protect,tackle,splash,recover'),
    ['>p1 move 2\n>p2 move 2','>p1 move 1\n>p2 move 1']);
  await run('D6 conv2 is NOT blocked by an UP Substitute (bypasssub)', slow,
    T('Snorlax','substitute,tackle,splash,surf'), ['>p1 move 2\n>p2 move 1','>p1 move 1\n>p2 move 2']);
  await run('D7 conv2 CAN pick a type the SOURCE already has, and REPLACES both types',
    T('Gengar','conversion2,splash,recover,icebeam',{ability:'Levitate'}),
    T('Snorlax','tackle,splash,thunderwave,surf'), ['>p1 move 2\n>p2 move 1','>p1 move 1\n>p2 move 1']);
  await run('D8 conv2 FAILS after the foe switched out and back (lastMoveUsed reset)', slow,
    T('Snorlax','tackle,splash,thunderwave,surf')+']'+T('Blissey','splash,recover,tackle,icebeam'),
    ['>p1 move 2\n>p2 move 1','>p1 move 2\n>p2 switch 2','>p1 move 2\n>p2 switch 2','>p1 move 1\n>p2 move 2']);
})();
