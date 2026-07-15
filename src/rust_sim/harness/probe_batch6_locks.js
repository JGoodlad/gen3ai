// probe_batch6_locks.js — ground-truth GROUP A of move-coverage BATCH 6 bit-for-bit vs the
// OMNISCIENT in-process gen3 BattleStream (no server): ENCORE / DESTINY BOND / ENDURE.
//
// Questions this probe settles BEHAVIORALLY (the sim is the only oracle):
//   ENCORE:
//     1. the resolved gen3 duration model — durationCallback draw (random(3,7)? (2,6)?) + the
//        willMove onStart branch (the Disable stored = rolled vs rolled+1 precedent).
//     2. the draw model of the Encore move itself (accuracy 100 → one roll? fail paths draw-free?).
//     3. fail conditions: no lastMove / lastMove == Struggle / already-encored / (0-PP lastMove?).
//     4. the move restriction + REQUEST shape (other slots disabled? trapped?), the PP
//        interaction (the encored move consumes PP normally?) and what happens when the encored
//        move hits 0 PP (early -end at the residual? Struggle?).
//   DESTINY BOND:
//     5. the volatile's LIFETIME window — persists until the user's NEXT MOVE (removed at
//        onBeforeMove?), so a faster foe KOing the user before it moves still triggers it.
//     6. WHAT triggers it: a FOE MOVE KO only? a residual (sand) KO? (confusion self-hit is the
//        same non-Move class as the residual.)
//     7. the mutual-faint ORDER (|faint| lines), pokemon_left, the double forced switch, and the
//        both-last-mons WIN/TIE semantics.
//     8. re-cast while the volatile is up (fail? refresh?) + the cast's own draw model.
//   ENDURE:
//     9. the stall-counter machinery — SHARED with Protect/Detect? (endure then protect: does the
//        protect roll read the counter endure set?), the first-use no-draw, the 2/4/8 ladder.
//    10. the survive-at-1 gate — a lethal MOVE hit (incl. FIXED damage / MULTIHIT strikes), and
//        what it does NOT guard (the end-of-turn residual after surviving at 1).
//    11. the willAct() gate (endure vs a foe switch → fails draw-free like protect?).
//
// Run:  node src/rust_sim/harness/probe_batch6_locks.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: opts.gender || 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

function dumpResolved() {
  const d = Dex.forFormat(FORMAT);
  for (const id of ['encore', 'destinybond', 'endure', 'protect']) {
    const m = d.moves.get(id);
    console.log(`=== resolved gen3 ${id} ===`);
    console.log(`  cat=${m.category} bp=${m.basePower} acc=${JSON.stringify(m.accuracy)} type=${m.type} prio=${m.priority} pp=${m.pp} target=${m.target}`);
    console.log(`  flags=${JSON.stringify(m.flags)} volatileStatus=${JSON.stringify(m.volatileStatus)} stallingMove=${JSON.stringify(m.stallingMove)}`);
    for (const k of Object.keys(m)) {
      const v = m[k];
      if (typeof v === 'function') console.log(`  fn ${k}: ${v.toString().replace(/\s+/g, ' ')}`);
    }
    if (m.condition) {
      console.log('  --- condition:');
      for (const k of Object.keys(m.condition)) {
        const v = m.condition[k];
        if (typeof v === 'function') console.log(`      fn ${k}: ${v.toString().replace(/\s+/g, ' ')}`);
        else console.log(`      ${k}=${JSON.stringify(v)}`);
      }
    }
    for (const g of ['gen4', 'gen9']) {
      const gm = Dex.mod(g).moves.get(id);
      const gc = gm.condition || {};
      console.log(`  [${g}] acc=${JSON.stringify(gm.accuracy)} prio=${gm.priority} cond.duration=${JSON.stringify(gc.duration)} cond.durationCallback=${gc.durationCallback ? gc.durationCallback.toString().replace(/\s+/g, ' ') : 'none'}`);
    }
  }
  // The stall condition (shared protect/endure machinery?) as gen3 resolves it.
  const stall = d.conditions.get('stall');
  console.log('=== resolved gen3 condition: stall ===');
  for (const k of Object.keys(stall)) {
    const v = stall[k];
    if (typeof v === 'function') console.log(`  fn ${k}: ${v.toString().replace(/\s+/g, ' ')}`);
    else if (k !== 'name' && k !== 'exists') console.log(`  ${k}=${JSON.stringify(v)}`);
  }
}

function drawLabel() {
  const st = new Error().stack.split('\n');
  const frames = [];
  for (let i = 3; i < st.length && frames.length < 6; i++) {
    const mm = st[i].match(/at ([\w.<>]+) /);
    if (mm) frames.push(mm[1]);
  }
  return frames.join('<');
}
const P32 = 2 ** 32;
const rTo = (val, from, to) => Math.floor((val * (to - from)) / P32) + from;

function volStr(m) {
  if (!m) return '-';
  return Object.entries(m.volatiles).map(([k, v]) => {
    const parts = [];
    for (const f of ['duration', 'counter', 'time', 'move', 'sourceSlot', 'trueDuration', 'hp', 'lostFocus']) {
      if (v[f] !== undefined) parts.push(`${f}=${JSON.stringify(v[f])}`);
    }
    return `${k}{${parts.join(',')}}`;
  }).join(' ') || '(none)';
}
function reqStr(side) {
  const r = side.activeRequest;
  if (!r) return 'null';
  if (r.wait) return 'wait';
  if (r.forceSwitch) return 'forceSwitch';
  if (r.active) {
    const a = r.active[0];
    const mv = (a.moves || []).map((x) => `${x.id}${x.pp !== undefined ? `(${x.pp}pp)` : ''}${x.disabled ? '[DIS]' : ''}`).join(',');
    return `moves=[${mv}]${a.trapped ? ' TRAPPED' : ''}${a.maybeTrapped ? ' maybeTrapped' : ''}`;
  }
  return '?';
}

// plan entries: { p1, p2, pre: [injections], stop }
// inject: { seed, acts: [{side, slot, status, hp, faint, item, boosts, pp:{idx:val}}] }
async function run(label, p1team, p2team, plan, inject) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  const seed = (inject && inject.seed) || [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  const applyActs = (acts) => {
    for (const inj of acts || []) {
      const side = battle.sides[inj.side];
      const m = inj.slot === undefined ? side.active[0] : side.pokemon[inj.slot];
      if (!m) continue;
      if (inj.status) m.setStatus(inj.status, m, null, true);
      if (inj.hp !== undefined) m.hp = inj.hp;
      if (inj.faint) { m.hp = 0; m.fainted = true; }
      if (inj.item !== undefined) m.item = inj.item;
      if (inj.boosts) Object.assign(m.boosts, inj.boosts);
      if (inj.pp) for (const [k, v] of Object.entries(inj.pp)) m.moveSlots[Number(k)].pp = v;
    }
  };
  applyActs(inject && inject.acts);

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);

  let draws = [];
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { const v = realNext(...a); draws.push({ v, l: drawLabel() }); return v; };

  let i = 0, safety = 0;
  while (!battle.ended && safety < 30) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    applyActs(entry.pre);
    draws = [];
    const logLen0 = log.length;
    const before = battle.prng.getSeed();
    const hpBefore = battle.sides.map((s) => s.active[0] ? `${s.active[0].species.name}:${s.active[0].hp}/${s.active[0].maxhp}` : '-');
    if (entry.p1) { try { streams.omniscient.write(`>p1 ${entry.p1}`); } catch (e) {} }
    if (entry.p2) { try { streams.omniscient.write(`>p2 ${entry.p2}`); } catch (e) {} }
    for (let k = 0; k < 20; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const ppStr = (m) => m ? m.moveSlots.map((s) => `${s.id}:${s.pp}/${s.maxpp}`).join(',') : '-';
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'} pp={${ppStr(m)}}` : '-';
    console.log(`  [${rs}] ${JSON.stringify({ p1: entry.p1, p2: entry.p2 })} draws=${draws.length}  seed ${before}->${after}`);
    console.log(`        pre : p1=${hpBefore[0]}  p2=${hpBefore[1]}  left=${battle.sides.map((s) => s.pokemonLeft)}`);
    console.log(`        post: p1=${fmt(a0)}  vols: ${volStr(a0)}`);
    console.log(`              p2=${fmt(a1)}  vols: ${volStr(a1)}`);
    console.log(`        req : p1=${reqStr(battle.sides[0])}`);
    console.log(`              p2=${reqStr(battle.sides[1])}`);
    draws.forEach((d, k) => console.log(`        DRAW[${k}] val=${d.v} r(3,7)=${rTo(d.v, 3, 7)} r(3,8)=${rTo(d.v, 3, 8)} r(2,6)=${rTo(d.v, 2, 6)}  ${d.l}`));
    const newLines = log.slice(logLen0).filter((l) =>
      /\|move\||\|turn\||-damage|-heal|-boost|-unboost|-fail|-immune|-miss|-crit|-supereffective|-resisted|cant|-activate|-hitcount|-end\b|-start|-singlemove|-singleturn|switch|drag|faint|-prepare|-nothing|-enditem|-status|-curestatus|win\|/.test(l));
    for (const l of newLines) console.log(`        LINE ${l}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner === undefined ? 'undefined' : JSON.stringify(battle.winner)}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  dumpResolved();

  // ================================================================ ENCORE
  // EN1: FAST encore user (Jolteon 130 > Skarmory 70) — the target splashed LAST turn; encore
  // lands BEFORE the target moves this turn (willMove TRUE branch). Read the duration draw
  // (raw val + candidates), the stored volatile, the REQUEST lock shape, and ride the lock to
  // its -end (the observable lock window).
  for (const seed of [[7, 11, 13, 17], [21, 22, 23, 24], [101, 5, 9, 3]]) {
    await run(`EN1 encore FAST user (willMove TRUE) seed=${JSON.stringify(seed)}`,
      [mon('Jolteon', ['encore', 'splash'])],
      [mon('Skarmory', ['splash', 'drillpeck'], { evs: { hp: 252 } })],
      [{ p1: 'move 2', p2: 'move 1' },                       // t1: both splash (target lastMove=splash)
       { p1: 'move 1', p2: 'move 1' },                       // t2: encore lands BEFORE skarm splashes
       ...Array(8).fill({ p1: 'move 2', p2: 'move 1' })],    // ride the lock to -end
      { seed });
  }

  // EN2: SLOW encore user (Snorlax 30 < Skarmory 70) — the target moves FIRST this turn, THEN
  // encore lands (willMove FALSE branch). Same reads; compare stored duration vs the raw roll.
  for (const seed of [[7, 11, 13, 17], [21, 22, 23, 24], [101, 5, 9, 3]]) {
    await run(`EN2 encore SLOW user (willMove FALSE) seed=${JSON.stringify(seed)}`,
      [mon('Snorlax', ['encore', 'splash'], { evs: { hp: 252 } })],
      [mon('Skarmory', ['splash', 'drillpeck'], { evs: { hp: 252 } })],
      [{ p1: 'move 2', p2: 'move 1' },
       { p1: 'move 1', p2: 'move 1' },
       ...Array(8).fill({ p1: 'move 2', p2: 'move 1' })],
      { seed });
  }

  // EN3: encore with NO lastMove (turn 1) → fail. Draw count?
  await run('EN3 encore vs a fresh target (no lastMove) — fail draws?',
    [mon('Jolteon', ['encore', 'splash'])],
    [mon('Skarmory', ['splash', 'drillpeck'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // EN4: the target's lastMove is STRUGGLE (splash pp injected 0) → encore fails?
  await run('EN4 encore vs a lastMove=Struggle target',
    [mon('Jolteon', ['encore', 'splash'])],
    [mon('Skarmory', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' },                          // t1: skarm STRUGGLES (0 pp)
     { p1: 'move 1', p2: 'move 1', stop: true }],             // t2: encore -> fail?
    { acts: [{ side: 1, pp: { 0: 0 } }] });

  // EN5: the PP interaction — encored splash at pp 2: lock, spend to 0, watch the encore END
  // early (the 0-PP residual check) + the request shape at 0 PP.
  await run('EN5 encore PP: encored move runs to 0 PP',
    [mon('Jolteon', ['encore', 'splash'])],
    [mon('Skarmory', ['splash', 'drillpeck'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' },                          // t1: splash (pp 2->1)
     { p1: 'move 1', p2: 'move 1' },                          // t2: encore lands; skarm splash (1->0)
     { p1: 'move 2', p2: 'move 1' },                          // t3: skarm at 0 pp — what runs? encore ends?
     { p1: 'move 2', p2: 'move 1', stop: true }],
    { acts: [{ side: 1, pp: { 0: 2 } }] });

  // EN6: RE-encore an already-encored target → fail? draws?
  await run('EN6 re-encore an already-encored target',
    [mon('Jolteon', ['encore', 'splash'])],
    [mon('Skarmory', ['splash', 'drillpeck'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' },                          // t2: encore lands
     { p1: 'move 1', p2: 'move 1', stop: true }]);            // t3: encore again -> ?

  // ============================================================ DESTINY BOND
  // DB1: the LIFETIME window — cast turn 1; on turn 2 the FASTER foe KOs the user BEFORE it
  // moves → the volatile is still up → MUTUAL faint. Print the faint order + left + requests.
  await run('DB1 destiny bond: cast t1, KOd t2 BEFORE moving (window OPEN)',
    [mon('Gengar', ['destinybond', 'splash'], { ability: 'Levitate' }), mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [mon('Starmie', ['splash', 'thunderbolt']), mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },                          // t1: DB cast; starmie splash
     { pre: [{ side: 0, hp: 1 }], p1: 'move 2', p2: 'move 2' }, // t2: starmie (faster) tbolt KOs first
     { p1: 'switch 2', p2: 'switch 2', stop: true }]);

  // DB2: the window CLOSES when the user MOVES again — cast t1; t2 the user (faster) moves
  // (splash) FIRST, then the slower foe KOs it → NO mutual faint?
  await run('DB2 destiny bond: user moves again t2 THEN is KOd (window CLOSED?)',
    [mon('Gengar', ['destinybond', 'splash'], { ability: 'Levitate' }), mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['splash', 'shadowball'], { evs: { hp: 252 } }), mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },                          // t1: DB cast
     { pre: [{ side: 0, hp: 1 }], p1: 'move 2', p2: 'move 2' }, // t2: gengar splashes FIRST, then shadowball KOs
     { p1: 'switch 2', p2: 'move 1', stop: true }]);

  // DB3: same-turn cast-then-KO — the user (faster) casts DB THIS turn, the slower foe KOs it
  // → mutual faint (the classic line).
  await run('DB3 destiny bond: cast and KOd the SAME turn',
    [mon('Gengar', ['destinybond', 'splash'], { ability: 'Levitate' }), mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['splash', 'shadowball'], { evs: { hp: 252 } }), mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ pre: [{ side: 0, hp: 1 }], p1: 'move 1', p2: 'move 2' }, // t1: DB first, shadowball KOs
     { p1: 'switch 2', p2: 'move 1', stop: true }]);

  // DB4: a RESIDUAL (sand) KO with the volatile up → does the foe faint? (non-Move source)
  await run('DB4 destiny bond vs a SAND-chip KO (residual source)',
    [mon('Gengar', ['destinybond', 'splash'], { ability: 'Levitate' }), mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [mon('Tyranitar', ['splash'], { ability: 'Sand Stream', evs: { hp: 252 } }), mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ pre: [{ side: 0, hp: 5 }], p1: 'move 1', p2: 'move 1' }, // t1: DB cast; sand chip KOs gengar at residual
     { p1: 'switch 2', p2: 'move 1', stop: true }]);

  // DB5: BOTH LAST MONS — DB user + foe both 1-mon teams; mutual faint → win/tie semantics?
  await run('DB5 destiny bond mutual faint with BOTH last mons (win/tie?)',
    [mon('Gengar', ['destinybond', 'splash'], { ability: 'Levitate' })],
    [mon('Snorlax', ['splash', 'shadowball'], { evs: { hp: 252 } })],
    [{ pre: [{ side: 0, hp: 1 }], p1: 'move 1', p2: 'move 2', stop: true }]);

  // DB6: RE-cast while the volatile is up (t1 DB, t2 DB again) — refresh? fail? draw model?
  await run('DB6 destiny bond re-cast on consecutive turns',
    [mon('Gengar', ['destinybond', 'splash'], { ability: 'Levitate' }), mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['splash', 'shadowball'], { evs: { hp: 252 } }), mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' },                          // t2: DB again (onBeforeMove sees own DB move)
     { pre: [{ side: 0, hp: 1 }], p1: 'move 2', p2: 'move 2' }, // t3: gengar splash first then KOd — window?
     { p1: 'switch 2', p2: 'move 1', stop: true }]);

  // ================================================================ ENDURE
  // ED1: first endure — non-lethal hit (no clamp line?) then a LETHAL hit → survive at 1 HP.
  // Draw model of the first use (no stall roll?) + the stall counter volatile.
  await run('ED1 endure: non-lethal then LETHAL hit (survive at 1)',
    [mon('Snorlax', ['endure', 'splash'], { evs: { hp: 252 } }), mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['drillpeck', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },                          // t1: endure vs a NON-lethal peck
     { pre: [{ side: 0, hp: 60 }], p1: 'move 1', p2: 'move 1', stop: true }]); // t2: consecutive endure vs a LETHAL peck

  // ED2: consecutive endures ×4 (foe splashes) — the stall ladder draws (2 -> 4 -> 8) + counter.
  for (const seed of [[7, 11, 13, 17], [3, 1, 4, 1]]) {
    await run(`ED2 consecutive endures (stall ladder) seed=${JSON.stringify(seed)}`,
      [mon('Snorlax', ['endure', 'splash'], { evs: { hp: 252 } })],
      [mon('Skarmory', ['splash'], { evs: { hp: 252 } })],
      [{ p1: 'move 1', p2: 'move 1' },
       { p1: 'move 1', p2: 'move 1' },
       { p1: 'move 1', p2: 'move 1' },
       { p1: 'move 1', p2: 'move 1', stop: true }],
      { seed });
  }

  // ED3: endure THEN protect — does the protect roll read the endure-set stall counter (SHARED)?
  await run('ED3 endure then PROTECT (shared stall counter?)',
    [mon('Snorlax', ['endure', 'protect', 'splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },                          // t1: endure (counter -> 2)
     { p1: 'move 2', p2: 'move 1', stop: true }]);            // t2: protect — draws randomChance(1,2)?

  // ED4: protect THEN endure — the mirror order.
  await run('ED4 protect then ENDURE (shared stall counter?)',
    [mon('Snorlax', ['endure', 'protect', 'splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1', stop: true }]);

  // ED5: endure survives the MOVE at 1 HP, then the same turn's BURN residual KOs (endure does
  // NOT guard residuals?).
  await run('ED5 endure at 1 HP then the burn residual (not guarded?)',
    [mon('Snorlax', ['endure', 'splash'], { evs: { hp: 252 } }), mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['drillpeck', 'splash'], { evs: { hp: 252 } })],
    [{ pre: [{ side: 0, status: 'brn', hp: 60 }], p1: 'move 1', p2: 'move 1' }, // endure -> 1hp -> brn KO
     { p1: 'switch 2', p2: 'move 2', stop: true }]);

  // ED6: endure vs a lethal FIXED-DAMAGE hit (Seismic Toss 100 into 60) → survive at 1?
  await run('ED6 endure vs a lethal Seismic Toss (fixed damage)',
    [mon('Snorlax', ['endure', 'splash'], { evs: { hp: 252 } }), mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } })],
    [{ pre: [{ side: 0, hp: 60 }], p1: 'move 1', p2: 'move 1', stop: true }]);

  // ED7: endure vs a foe SWITCH (the willAct gate) — fails draw-free, no volatile?
  await run('ED7 endure vs a foe SWITCH (willAct gate)',
    [mon('Snorlax', ['endure', 'splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['splash'], { evs: { hp: 252 } }), mon('Forretress', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'switch 2', stop: true }]);

  // ED8: endure vs a lethal MULTIHIT (Arm Thrust) — every strike clamps at 1?
  await run('ED8 endure vs a lethal multihit (Arm Thrust)',
    [mon('Snorlax', ['endure', 'splash'], { evs: { hp: 252 } }), mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [mon('Hariyama', ['armthrust', 'splash'], { evs: { hp: 252, atk: 252 } })],
    [{ pre: [{ side: 0, hp: 40 }], p1: 'move 1', p2: 'move 1', stop: true }]);

  // ============================================== FOLLOW-UPS (ties + override)
  // EN7: the SAME-TURN onOverrideAction — the FAST encore lands while the target has QUEUED a
  // DIFFERENT move (drillpeck): the queued action is overridden to the encored move (splash)?
  // WHICH slot's PP deducts? What draws?
  await run('EN7 encore onOverrideAction: overrides the queued DIFFERENT move',
    [mon('Jolteon', ['encore', 'splash'])],
    [mon('Skarmory', ['splash', 'drillpeck'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' },                          // t1: both splash
     { p1: 'move 1', p2: 'move 2', stop: true }]);            // t2: encore vs a QUEUED drillpeck

  // EN8: encore at a SPEED TIE (Skarmory mirror) — the eachEvent tie shuffles + does the landed
  // encore fire the in-tryMoveHit Update? (t1 both-splash = the control draw count.) Then a
  // REJECTED disabled-slot choice (move 2 while encored) → draw-free open boundary?
  await run('EN8 encore at a speed TIE + a rejected disabled-slot choice',
    [mon('Skarmory', ['encore', 'splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['splash', 'drillpeck'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' },                          // control: both splash at a tie
     { p1: 'move 1', p2: 'move 1' },                          // encore lands at a tie
     { p1: 'move 2', p2: 'move 2' },                          // p2 picks the DISABLED drillpeck -> rejected?
     { p1: 'move 2', p2: 'move 1', stop: true }]);

  // EN10: encore into a 0-PP lastMove (the Disable-TD5 analog) — the target spends its LAST
  // splash PP on t1 (drillpeck still has PP so no Struggle), then t2 encore → the onStart
  // `moveSlot.pp <= 0` guard fails AFTER the accuracy + duration draws?
  await run('EN10 encore vs a 0-PP lastMove (onStart pp guard)',
    [mon('Jolteon', ['encore', 'splash'])],
    [mon('Skarmory', ['splash', 'drillpeck'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' },                          // t1: skarm's LAST splash pp (1 -> 0)
     { p1: 'move 1', p2: 'move 2', stop: true }],             // t2: encore vs lastMove=splash @ 0 pp
    { acts: [{ side: 1, pp: { 0: 1 } }] });

  // ED9: endure at a SPEED TIE (Snorlax mirror) — control then endure (the in-tryMoveHit
  // Update / landed question for the protect family at a tie).
  await run('ED9 endure at a speed TIE (control then endure)',
    [mon('Snorlax', ['endure', 'splash'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['splash', 'bodyslam'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' },                          // control: both splash at a tie
     { p1: 'move 1', p2: 'move 1', stop: true }]);            // endure at a tie

  // DB7: destiny bond CAST at a SPEED TIE (Gengar mirror) — control then cast (the
  // in-tryMoveHit Update / landed question for a -singlemove volatile).
  await run('DB7 destiny bond cast at a speed TIE (control then cast)',
    [mon('Gengar', ['destinybond', 'splash'], { ability: 'Levitate' })],
    [mon('Gengar', ['splash', 'shadowball'], { ability: 'Levitate' })],
    [{ p1: 'move 2', p2: 'move 1' },                          // control: both splash at a tie
     { p1: 'move 1', p2: 'move 1', stop: true }]);            // DB cast at a tie
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
