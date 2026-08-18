// PROBE: gen-3 SAFEGUARD — draw model + emission forms + the block matrix.
//
// SETTLED 2026-08-18 (run it to re-confirm; do not re-derive from source):
//
//   DEX: num 219, accuracy TRUE (never-miss), Status, pp 25, priority 0,
//        target 'allySide', sideCondition 'safeguard', flags {snatch:1, metronome:1}
//        (NO protect, NO reflectable, NO mirror). Condition: duration 5,
//        durationCallback -> 5 (Persistent is gen5+; deterministic, DRAW-FREE),
//        onSideResidualOrder 4, onSideResidualSubOrder UNDEFINED (== reflect 1 /
//        lightscreen 2 / mist 3 — the same SideCondition family).
//
//   CAST (success)      : ZERO draws. never-miss => NO accuracy roll; the side-condition
//                         add + the fixed duration 5 draw nothing. `landed` = false.
//                         -> `|move|<u>|Safeguard|<u>` then `|-sidestart|<side>|Safeguard`
//   CAST (already up)   : ZERO draws (addSideCondition false; no onSideRestart)
//                         -> `|move|<u>|Safeguard||[still]` + `|-fail|<u>`   (fail on the USER)
//   EXPIRY              : end-of-turn SIDE residual, order 4, decrement 5..0
//                         -> `|-sideend|<side>|Safeguard`
//                         Emitted BEFORE sand `[upkeep]`(8) / Leftovers `-heal`(10.4) /
//                         status DoT(10.6); AFTER a same-turn Reflect `-sideend`(1).
//   RESIDUAL TIE  (!!)  : BOTH sides carrying Safeguard TIE at order 4 => ONE EXTRA
//                         `random(0,2)` per residual (the both-sides-Light-Screen sibling).
//                         Measured on a Snorlax mirror: 1 safeguard = 8 draws/turn,
//                         2 safeguards = 9 draws/turn.
//
//   BLOCK MATRIX (the block itself is always DRAW-FREE; the ATTEMPT keeps its own draws):
//     Thunder Wave / Toxic / Will-O-Wisp / Spore / Confuse Ray / Swagger / Yawn-cast
//                         : accuracy roll DRAWN, then blocked
//                           -> `|-activate|<target>|move: Safeguard`   (on the TARGET MON)
//                           The sleep `random(2,6)` is NOT drawn (blocked before onStart).
//     Body Slam par-30% / Water Pulse confusion-20%
//                         : the secondary's `random(100)` STILL DRAWS (same count as the
//                           control) and the effect is SILENTLY suppressed — NO line at all.
//                           A confusion secondary's follow-on `random(2,6)` is NOT drawn.
//     Static / contact-proc abilities
//                         : the proc's `randomChance(1,3)` STILL DRAWS, SILENTLY suppressed.
//     Synchronize reflect : BLOCKED, and it DOES emit `-activate|<caster>|move: Safeguard`
//                           (the one non-Move effect the condition announces).
//
//   gen3ou SetStatus SHUFFLE (the draw-count crux): a safeguard-BLOCKED status still draws
//     EXACTLY ONE shuffle at `Pokemon.setStatus`, but the RANGE shifts `random(0,2)` ->
//     `random(1,3)`: safeguard's onSetStatus is a 3rd handler that sorts into its OWN group
//     (index 0), leaving the 2 Standard clauses a size-2 tie. Same COUNT, one `rng.next()`
//     => SEED-NEUTRAL. gen3customgame has no clauses => safeguard's handler is alone => NO
//     shuffle at all. (Identical to the STATUS_IMMUNE-ability pattern already modeled.)
//
//   GATE PRECEDENCE (which line wins):
//     already-statused  >  status-TYPE immunity  >  STATUS_IMMUNE ability  >  SAFEGUARD  >  sleep clause
//     (`-fail|<t>|<status>`)  (`-immune|<t>`)   (`-immune|…|[from] ability:`) (`-activate`)  (never reached)
//
//   EDGES:
//     SELF-inflicted (REST) PASSES — `onSetStatus` returns early when target === source.
//       Rest under your own Safeguard sleeps + heals and draws its `random(2,6)` normally.
//     CONFUSION is blocked (`onTryAddVolatile`), including Confuse Ray and Swagger.
//       SWAGGER IS PARTIAL: its `-boost|<t>|atk|2` STILL APPLIES and only the confusion is
//       blocked -> `|-boost|<t>|atk|2` THEN `|-activate|<t>|move: Safeguard`. (An earlier
//       draft of this probe claimed the whole move failed — its line filter simply omitted
//       `-boost`. Widen the filter before concluding a line is absent.)
//     YAWN: the CAST is blocked (`onTryAddVolatile` yawn arm, `-activate`), but a
//       PENDING yawn's RESOLVE is EXEMPT (`if (effect.id === 'yawn') return`) — it sleeps
//       the mon through an active Safeguard, drawing its `random(2,6)`.
//     BATON PASS / any switch: Safeguard is a SIDE condition, not a volatile — it PERSISTS
//       and protects the ENTRANT. Nothing to copy.
//     SNATCH steals it (flags.snatch:1): `-activate|<snatcher>|move: Snatch|[of] <user>` then
//       `|move|<snatcher>|Safeguard|<snatcher>|[from] Snatch` + `-sidestart|<SNATCHER side>`.
//
// WHY IT IS WORTH DOING: safeguard carries 0.37 of the gen3ou move-slot prior mass — level
// with recycle/confuseray, behind sandattack (0.72).
//
// DRAW COUNTING: `PRNG.random()` is the ONE method that reaches `rng.next()` (prng.js:86);
// randomChance (:108), sample (:127) and shuffle (:142) all funnel through it. Wrap ONLY
// `random` — one logged line == exactly one PRNG draw. (Wrapping randomChance too
// double-counts every roll, which is what made the first pass of this probe unreadable.)
const path=require('path'); const PS='/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const {BattleStream}=require(path.join(PS,'dist/sim/battle-stream.js'));
const {Dex}=require(path.join(PS,'dist/sim/dex.js'));
const {PRNG}=require(path.join(PS,'dist/sim/prng'));

// ---------------------------------------------------------------- dex facts
{
  const d=Dex.mod('gen3'), mv=d.moves.get('safeguard'), c=mv.condition||{};
  console.log('=== DEX safeguard ===', JSON.stringify({
    num:mv.num, accuracy:mv.accuracy, category:mv.category, pp:mv.pp, priority:mv.priority,
    target:mv.target, sideCondition:mv.sideCondition, flags:mv.flags,
    duration:c.duration, onSideResidualOrder:c.onSideResidualOrder,
    onSideResidualSubOrder:c.onSideResidualSubOrder, handlers:Object.keys(c),
  }));
  for (const id of ['reflect','lightscreen','mist']) {
    const cc=d.moves.get(id).condition||{};
    console.log(`  anchor ${id}: order=${cc.onSideResidualOrder} sub=${cc.onSideResidualSubOrder} dur=${cc.duration}`);
  }
}

// ---------------------------------------------------------------- harness
let draws=[];
{ const o=PRNG.prototype.random;
  PRNG.prototype.random=function(...a){ const r=o.apply(this,a);
    const site=(new Error().stack||'').split('\n').slice(2,6)
      .map(s=>(s.match(/at ([\w.<>\[\] ]+)/)||[])[1]||'?').join('<-');
    draws.push(`random(${a})->${r} @${site}`); return r; }; }

const KEEP=/^\|(-sidestart|-sideend|-activate|-start|-end|-status|-curestatus|-immune|-fail|-miss|-damage|-heal|-boost|-unboost|-weather|-singleturn|-message|move|switch|drag|cant|faint|turn|upkeep|win)\b/;

async function run(label, p1, p2, script, fmt='gen3customgame') {
  draws=[]; const s=new BattleStream(); const ch=[];
  (async()=>{ for await(const c of s) ch.push(c); })();
  s.write(`>start {"formatid":"${fmt}","seed":[9,9,9,9]}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r=>setTimeout(r,220));
  const m=[draws.length];
  for(const c of script){ s.write(c); await new Promise(r=>setTimeout(r,220)); m.push(draws.length); }
  await new Promise(r=>setTimeout(r,220)); m[m.length-1]=draws.length;
  const omni=ch.filter(c=>!c.startsWith('sideupdate')).join('\n').split('\n').filter(l=>KEEP.test(l));
  console.log(`\n===== ${label} [${fmt}] =====`);
  console.log(omni.map(l=>'  '+l).join('\n'));
  for(let i=0;i<script.length;i++){
    console.log(`  [w${i}] ${script[i].replace(/\n/g,' | ')}  -> ${m[i+1]-m[i]} DRAWS`);
    for(const d of draws.slice(m[i],m[i+1])) console.log('        '+d);
  }
}

// ---------------------------------------------------------------- teams
// p1 safeguard user:              1=safeguard 2=splash 3=rest 4=tackle
const SG    = 'Snorlax||Leftovers|Thickfat|safeguard,splash,rest,tackle|Hardy|85,85,85,85,85,85|M||||';
// same, but slot 3 is THUNDER WAVE (for the Synchronize-reflect scenario)
const SG_TW = 'Snorlax||Leftovers|Thickfat|safeguard,splash,thunderwave,tackle|Hardy|85,85,85,85,85,85|M||||';
const SG_BP = SG+']Blissey||Leftovers|Thickfat|splash,softboiled,tackle,rest|Hardy|85,85,85,85,85,85|F||||';
// MIRROR (identical speed => every eachEvent/residual group ties): 1=safeguard 2=splash
const SGM   = 'Snorlax||Leftovers|Thickfat|safeguard,splash,reflect,tackle|Hardy|85,85,85,85,85,85|F||||';
// GROUND safeguard user (Thunder Wave is type-immune):            1=safeguard 2=splash
const SGG   = 'Swampert||Leftovers|Torrent|safeguard,splash,toxic,tackle|Hardy|85,85,85,85,85,85|M||||';
// LIMBER + safeguard (ability-vs-safeguard precedence):           1=safeguard 2=splash
const LIMB  = 'Persian||Leftovers|Limber|safeguard,splash,rest,tackle|Hardy|85,85,85,85,85,85|M||||';
// p2 status attacker:  1=twave 2=splash 3=toxic 4=willowisp
const ST1   = 'Gengar||Leftovers|Levitate|thunderwave,splash,toxic,willowisp|Hardy|85,85,85,85,85,85|M||||';
// p2 volatile/secondary attacker: 1=spore 2=splash 3=confuseray 4=bodyslam
const ST2   = 'Gengar||Leftovers|Levitate|spore,splash,confuseray,bodyslam|Hardy|85,85,85,85,85,85|M||||';
// p2 yawn / waterpulse:           1=yawn 2=splash 3=waterpulse 4=swagger
const ST3   = 'Gengar||Leftovers|Levitate|yawn,splash,waterpulse,swagger|Hardy|85,85,85,85,85,85|M||||';
const STATIC= 'Electabuzz||Leftovers|Static|splash,tackle,thunderwave,reflect|Hardy|85,85,85,85,85,85|M||||';
const SYNC  = 'Espeon||Leftovers|Synchronize|splash,tackle,rest,recover|Hardy|85,85,85,85,85,85|M||||';
const SNATCH= 'Gengar||Leftovers|Levitate|snatch,splash,thunderwave,spore|Hardy|85,85,85,85,85,85|M||||';
const SPL   = 'Miltank||Leftovers|Thickfat|splash,reflect,tackle,lightscreen|Hardy|85,85,85,85,85,85|F||||';
const TTAR  = 'Tyranitar||Leftovers|Sandstream|splash,tackle,toxic,swagger|Hardy|85,85,85,85,85,85|M||||';

// Every "into safeguard" scenario opens with `>p2 move 2` (SPLASH) so nothing lands before
// Safeguard is up. Do NOT let the foe use a status move on turn 1: the first draft did, the
// target ended up already-statused, and every later block read as an already-statused FAIL.
const OPEN = '>p1 move 1\n>p2 move 2';

(async()=>{
  // --- 1. DRAW MODEL ---------------------------------------------------------
  await run('1a CONTROL splash/splash',           SG, SPL, ['>p1 move 2\n>p2 move 1']);
  await run('1b CAST (== control => DRAW-FREE)',  SG, SPL, ['>p1 move 1\n>p2 move 1']);
  await run('1c CAST then RE-CAST while active',  SG, SPL, ['>p1 move 1\n>p2 move 1','>p1 move 1\n>p2 move 1']);

  // --- 2. EXPIRY + residual position ----------------------------------------
  await run('2a EXPIRY (cast + 5 idle turns)', SG, SPL,
    ['>p1 move 1\n>p2 move 1','>p1 move 2\n>p2 move 1','>p1 move 2\n>p2 move 1','>p1 move 2\n>p2 move 1','>p1 move 2\n>p2 move 1','>p1 move 2\n>p2 move 1']);
  await run('2b -sideend order: p2 Reflect(1) BEFORE p1 Safeguard(4)', SG, SPL,
    ['>p1 move 1\n>p2 move 2','>p1 move 2\n>p2 move 1','>p1 move 2\n>p2 move 1','>p1 move 2\n>p2 move 1','>p1 move 2\n>p2 move 1','>p1 move 2\n>p2 move 1']);
  await run('2c -sideend(4) BEFORE sand(8) / Leftovers(10.4) / tox DoT(10.6)', SG, TTAR,
    ['>p1 move 1\n>p2 move 3','>p1 move 2\n>p2 move 1','>p1 move 2\n>p2 move 1','>p1 move 2\n>p2 move 1','>p1 move 2\n>p2 move 1','>p1 move 2\n>p2 move 1']);

  // --- 3. RESIDUAL TIE: both sides safeguard --------------------------------
  await run('3a ONE safeguard on a mirror (control)', SG, SGM, ['>p1 move 1\n>p2 move 2','>p1 move 2\n>p2 move 2']);
  await run('3b BOTH sides safeguard (+1 draw/turn)', SG, SGM, ['>p1 move 1\n>p2 move 1','>p1 move 2\n>p2 move 2']);

  // --- 4. BLOCKED STATUS MOVES ----------------------------------------------
  await run('4a THUNDER WAVE into safeguard', SG, ST1, [OPEN,'>p1 move 2\n>p2 move 1']);
  await run('4b TOXIC into safeguard',        SG, ST1, [OPEN,'>p1 move 2\n>p2 move 3']);
  await run('4c WILL-O-WISP into safeguard',  SG, ST1, [OPEN,'>p1 move 2\n>p2 move 4']);
  await run('4d SPORE into safeguard (NO random(2,6))', SG, ST2, [OPEN,'>p1 move 2\n>p2 move 1']);
  await run('4e CONTROL twave lands',         SG, ST1, ['>p1 move 2\n>p2 move 2','>p1 move 2\n>p2 move 1']);
  await run('4f CONTROL spore lands (+random(2,6))', SG, ST2, ['>p1 move 2\n>p2 move 2','>p1 move 2\n>p2 move 1']);

  // --- 5. SECONDARIES + contact procs: draw KEPT, effect SILENTLY suppressed --
  await run('5a BODY SLAM into safeguard (secondary random(100) STILL draws)', SG, ST2,
    [OPEN,'>p1 move 2\n>p2 move 4','>p1 move 2\n>p2 move 4','>p1 move 2\n>p2 move 4']);
  await run('5b CONTROL body slam (par lands on the same roll)', SG, ST2,
    ['>p1 move 2\n>p2 move 2','>p1 move 2\n>p2 move 4','>p1 move 2\n>p2 move 4','>p1 move 2\n>p2 move 4']);
  await run('5c WATER PULSE confusion-secondary into safeguard', SG, ST3,
    [OPEN,'>p1 move 2\n>p2 move 3','>p1 move 2\n>p2 move 3','>p1 move 2\n>p2 move 3','>p1 move 2\n>p2 move 3']);
  await run('5d STATIC contact-proc under own safeguard (random(3) STILL draws)', SG, STATIC,
    ['>p1 move 1\n>p2 move 1','>p1 move 4\n>p2 move 1','>p1 move 4\n>p2 move 1','>p1 move 4\n>p2 move 1']);
  await run('5e CONTROL static (par lands)', SG, STATIC,
    ['>p1 move 2\n>p2 move 1','>p1 move 4\n>p2 move 1','>p1 move 4\n>p2 move 1','>p1 move 4\n>p2 move 1']);

  // --- 6. CONFUSION + YAWN --------------------------------------------------
  await run('6a CONFUSE RAY into safeguard',      SG, ST2, [OPEN,'>p1 move 2\n>p2 move 3']);
  await run('6b CONTROL confuse ray lands',       SG, ST2, ['>p1 move 2\n>p2 move 2','>p1 move 2\n>p2 move 3']);
  await run('6c SWAGGER into safeguard (the +2 Atk STILL applies; only confusion is blocked)', SG, ST3, [OPEN,'>p1 move 2\n>p2 move 4']);
  await run('6d YAWN cast into safeguard',        SG, ST3, [OPEN,'>p1 move 2\n>p2 move 1','>p1 move 2\n>p2 move 2','>p1 move 2\n>p2 move 2']);
  await run('6e YAWN pending -> safeguard up -> RESOLVE IS EXEMPT', SG, ST3,
    ['>p1 move 2\n>p2 move 1','>p1 move 1\n>p2 move 2','>p1 move 2\n>p2 move 2']);

  // --- 7. SELF-INFLICTED, SNATCH, SIDE PERSISTENCE ---------------------------
  await run('7a REST under own safeguard (self-inflicted PASSES)', SG, ST2,
    [OPEN,'>p1 move 4\n>p2 move 4','>p1 move 3\n>p2 move 2']);
  await run('7b SNATCH steals safeguard', SG, SNATCH, ['>p1 move 1\n>p2 move 1','>p1 move 2\n>p2 move 3']);
  await run('7c safeguard persists across a switch, protects the ENTRANT', SG_BP, ST1,
    ['>p1 move 1\n>p2 move 2','>p1 switch 2\n>p2 move 2','>p1 move 1\n>p2 move 1']);

  // --- 8. GATE PRECEDENCE ----------------------------------------------------
  await run('8a already-statused BEATS safeguard (-fail|t|par)', SG, ST1,
    ['>p1 move 2\n>p2 move 1','>p1 move 1\n>p2 move 2','>p1 move 2\n>p2 move 1']);
  await run('8b type-immunity BEATS safeguard (-immune)', SGG, ST1, [OPEN,'>p1 move 2\n>p2 move 1']);
  await run('8c STATUS_IMMUNE ability BEATS safeguard (Limber)', LIMB, ST1, [OPEN,'>p1 move 2\n>p2 move 1']);
  // 8d needs the SAFEGUARDED side to be the one INFLICTING the status, so it uses its own
  // p1 team whose slot 3 is Thunder Wave (SG's slot 3 is Rest — using SG here silently
  // Rests instead and tests nothing).
  await run('8d SYNCHRONIZE reflect blocked, DOES emit -activate', SG_TW, SYNC,
    ['>p1 move 1\n>p2 move 1','>p1 move 3\n>p2 move 1']);
  await run('8e CONTROL synchronize reflect lands (no safeguard)', SG_TW, SYNC,
    ['>p1 move 2\n>p2 move 1','>p1 move 3\n>p2 move 1']);

  // --- 9. gen3ou SetStatus shuffle: same COUNT, shifted RANGE -----------------
  await run('9a ou twave LANDS  -> shuffle(list,0,2)',   SG, ST1, ['>p1 move 2\n>p2 move 2','>p1 move 2\n>p2 move 1'], 'gen3ou');
  await run('9b ou twave BLOCKED -> shuffle(list,1,3)',  SG, ST1, [OPEN,'>p1 move 2\n>p2 move 1'], 'gen3ou');
  await run('9c cg twave BLOCKED -> NO shuffle',         SG, ST1, [OPEN,'>p1 move 2\n>p2 move 1'], 'gen3customgame');
})();
