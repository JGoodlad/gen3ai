// probe_varbp_cluster_final.js — ground-truth the gen-3 VARIABLE-BASE-POWER cluster
//   ERUPTION / REVENGE / SMELLING SALTS / FURY CUTTER
// bit-for-bit vs the OMNISCIENT in-process gen3 BattleStream (no server). All four sit in the
// port's 16 SILENT-DESYNC fail-loud list (`state.rs::UNMODELED_FAILLOUD_MOVES`): each would run
// at its FLAT data BP (150 / 60 / 60 / 10) while its true gen-3 BP is DERIVED.
//
// SETTLED 2026-08-18 (run it to re-confirm; do not re-derive from source):
//
//  ERUPTION (Fire, SPECIAL, dataBP 150, acc 100, prio 0, target allAdjacentFoes)
//    bp = clampIntRange(150 * user.hp / user.maxhp, 1) = max(floor(150*hp/maxhp), 1).
//    Measured at maxhp 297: hp 297->150, 198->100 (exact), 149->75, 148->74, 3->1, 2->1,
//    1-> raw 0.505 floors to 0 -> CLAMPED to 1 (a min-damage HIT, never a fail).
//    IDENTICAL to WATER SPOUT (same resolved callback; probed side by side at hp 150 -> both 75).
//    The hp is read INSIDE getDamage, i.e. AT DAMAGE TIME: a faster foe's same-turn chip counts
//    (Swift 297->226 first => bp floor(150*226/297)=114). DRAW-NEUTRAL: full-hp and 1-hp arms at
//    the same seed end on the SAME seed with the same 4 draws.
//
//  REVENGE (Fighting, PHYSICAL, dataBP 60, acc 100, contact, PRIORITY -4)
//    bp = 120 iff the user was damaged THIS TURN, BEFORE revenge resolves, by a MOVE of the
//    current target that dealt damage to the MON; else 60.
//    Priority pinned BEHAVIOURALLY: a max-speed Aerodactyl still moves after a Snorlax; Focus
//    Punch (-3) resolves BEFORE revenge; Counter (-5) resolves AFTER it. So -4, strictly.
//    NOT doubling: a sub-absorbed hit (the user's Substitute ate it) -> 60; a MISSED foe move
//    -> 60; damage taken on a PREVIOUS turn -> 60 (incl. a sandstorm residual); a foe move that
//    resolves AFTER revenge (Counter at -5) -> 60. The "this turn" flag is read AT RESOLUTION.
//
//  SMELLING SALTS (Normal, PHYSICAL, dataBP 60, acc 100, contact, prio 0)
//    bp = 120 iff target.status == 'par' at getDamage time, else 60 (burn/sleep/frz do NOT
//    double). On a LANDED hit that reaches the MON it then CURES the paralysis, in this order:
//      |move|…|Smelling Salts|<target>
//      |-damage|<target>|<hp>/<max> par      <- the damage line still carries `par`
//      |-curestatus|<target>|par|[msg]
//      (…then any contact-ability proc, e.g. |-status|<user>|par|[from] ability: Static|[of]…)
//    NO cure when: the move MISSES; the target is type-IMMUNE (Ghost — no BP callback runs at
//    all); the hit is absorbed by the target's SUBSTITUTE (bp is still 120 — the status read is
//    the MON's — but no `-curestatus`); the hit KOs the target (no `-curestatus` line at all).
//
//  FURY CUTTER (Bug, PHYSICAL, dataBP 10, acc 95, contact, prio 0)
//    A per-USER multiplier m (1,2,4,8,16, capped at 16): bp = clampIntRange(10*m, 1, 160)
//    => 10, 20, 40, 80, 160, 160, 160 … (probed 8 forced-hit turns).
//    The multiplier lives in a duration-2 volatile on the USER:
//      * the BP callback CREATES it at m=1 when absent (this is what a first use reads);
//      * the move's onHit DOUBLES it (m<16) and RE-ARMS duration to 2 — and onHit runs only on
//        a hit that reaches the MON.
//    RESETS (all probed): a MISS; using a DIFFERENT move; the user SWITCHING OUT; the target
//    PROTECTING; a turn the user cannot act (asleep). Rule: the escalation continues iff fury
//    cutter LANDED ON THE MON on the immediately preceding turn and the user never left.
//    DOES NOT reset: the TARGET switching — the counter is on the user and target-agnostic
//    (x4 carried straight onto a freshly switched-in Snorlax for bp 40, then 80).
//    ⚠ SUBSTITUTE is the trap: a sub-absorbed hit still COMPUTES bp from the current m (probed
//    bp 40 into a sub) but SKIPS onHit — so m never doubles and the volatile expires that turn.
//    A chain fired into a standing sub reads 40 once and then sits at 10 forever.
//
//  DRAW MODEL — all four are the ORDINARY single-hit damaging draw model; none of the four BP
//  derivations, the salts cure, or the fury-cutter volatile consumes or skips a draw. Per
//  damaging turn: randomChance(100,100) accuracy -> randomChance(1,16) crit -> random(16) damage
//  (+ the per-turn `Battle.endTurn randomChance(1,5)` that every gen3 turn takes regardless).
//  A miss stops after the accuracy roll. THE ONE DOWNSTREAM DELTA: curing paralysis REMOVES the
//  target's per-turn full-para randomChance for the rest of the battle — measured 4 draws
//  (salts, cures) vs 5 draws (a flat control move into the same par target).
//
// METHOD (sibling-probe lessons):
//   * draws are counted at `battle.prng.rng.next` — the SOLE path (PRNG.random / randomChance /
//     shuffle all funnel through it). Wrapping random AND randomChance double-counts.
//   * the derived BP is read by wrapping `move.basePowerCallback` for the duration of ONE
//     `BattleActions.getDamage` call — NEVER by calling the callback a second time: fury
//     cutter's callback has a SIDE EFFECT (it `addVolatile`s itself), so a double call would
//     corrupt the very escalation being measured.
//   * a forced miss/hit is a one-shot wrapper on `randomChance(_,100)` that STILL CONSUMES the
//     real draw and only flips the boolean, so the seed evolves exactly as it would have.
//   * `pre(battle)` hooks set hp / status / refill PP mid-battle (eruption has 5 PP).
//   * chunks starting with `sideupdate` are dropped by using the `omniscient` player stream.
//
// Run:  node /tmp/probe_varbp_cluster_final.js
'use strict';
const path = require('path');
const PS = '/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));
const { BattleActions } = require(path.join(PS, 'dist/sim/battle-actions'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
const MOVES = ['eruption', 'waterspout', 'revenge', 'smellingsalts', 'furycutter'];

function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: { ...IV31, ...(opts.ivs || {}) },
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: opts.gender || 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

let bpLog = [];
const origGetDamage = BattleActions.prototype.getDamage;
BattleActions.prototype.getDamage = function (source, target, move, suppressMessages = false) {
  if (move && typeof move === 'object' && move.basePowerCallback) {
    const cb = move.basePowerCallback; const src = source, tgt = target;
    move.basePowerCallback = function (...a) {
      const bp = cb.apply(this, a);
      const mult = src.volatiles && src.volatiles.furycutter ? ` fcMult=${src.volatiles.furycutter.multiplier}` : '';
      bpLog.push(`BP ${move.id}: dataBP=${move.basePower} type=${move.type} cat=${move.category} -> derived=${bp} ` +
        `(floored->${Math.max(Math.floor(bp), 1)}) [user ${src.name} ${src.hp}/${src.maxhp} st=${src.status || '-'}${mult}] ` +
        `[target ${tgt.name} ${tgt.hp}/${tgt.maxhp} st=${tgt.status || '-'} sub=${!!tgt.volatiles.substitute}]`);
      return bp;
    };
    try { return origGetDamage.call(this, source, target, move, suppressMessages); }
    finally { move.basePowerCallback = cb; }
  }
  if (move && typeof move === 'object' && move.id) {
    bpLog.push(`BP ${move.id}: FLAT dataBP=${move.basePower} type=${move.type} cat=${move.category}`);
  }
  return origGetDamage.call(this, source, target, move, suppressMessages);
};

function dumpResolved() {
  const d = Dex.forFormat(FORMAT);
  console.log('=== RESOLVED gen3 handlers (shown for the reader; BEHAVIOUR below is the oracle) ===');
  for (const id of MOVES) {
    const m = d.moves.get(id);
    console.log(`--- ${id}: cat=${m.category} bp=${m.basePower} acc=${m.accuracy} type=${m.type} ` +
      `prio=${m.priority} pp=${m.pp} target=${m.target} flags=${JSON.stringify(m.flags)}`);
    for (const k of ['basePowerCallback', 'onHit', 'onTry', 'onTryHit', 'onModifyMove', 'onMoveFail', 'condition']) {
      if (m[k] === undefined) continue;
      const v = typeof m[k] === 'function' ? m[k].toString().replace(/\s+/g, ' ')
        : JSON.stringify(m[k], (kk, vv) => (typeof vv === 'function' ? vv.toString().replace(/\s+/g, ' ') : vv));
      console.log(`    ${k}: ${v}`);
    }
  }
}

function drawLabel() {
  const st = new Error().stack.split('\n'); const frames = [];
  for (let i = 3; i < st.length && frames.length < 3; i++) {
    const mm = st[i].match(/at ([\w.<>]+) /); if (mm) frames.push(mm[1]);
  }
  return frames.join('<');
}

async function run(label, p1team, p2team, plan, inject = {}) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);   // omniscient => no `sideupdate` chunks
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  const seed = inject.seed || [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  for (const inj of (inject.acts || [])) {
    const m = battle.sides[inj.side].active[0]; if (!m) continue;
    if (inj.status) m.setStatus(inj.status, m, null, true);
    if (inj.hp !== undefined) m.hp = inj.hp;
  }
  let draws = [];
  const rng = battle.prng.rng; const realNext = rng.next.bind(rng);
  rng.next = function (...a) { const v = realNext(...a); draws.push(drawLabel()); return v; };
  let force = null;
  const realRC = battle.prng.randomChance.bind(battle.prng);
  battle.prng.randomChance = function (n, den) {
    const v = realRC(n, den);
    if (force && den === 100) { const f = force; force = null; return f === 'hit'; }
    return v;
  };
  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  let i = 0, safety = 0; const trace = [];
  while (!battle.ended && safety < 40) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    if (entry.pre) entry.pre(battle);
    force = entry.forceMiss ? 'miss' : (entry.forceHit ? 'hit' : null);
    draws = []; bpLog = [];
    const logLen0 = log.length; const before = battle.prng.getSeed();
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 25; k++) await tick();
    const after = battle.prng.getSeed(); trace.push(`${before}->${after}`);
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const fmt = (m) => (m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'} vols=[${Object.keys(m.volatiles).map((k) => (k === 'furycutter' ? `furycutter(x${m.volatiles[k].multiplier},d${m.volatiles[k].duration})` : k)).join(',')}]` : '-');
    console.log(`  T${i} ${entry.label || JSON.stringify({ p1: entry.p1, p2: entry.p2 })}  draws=${draws.length}  seed ${before} -> ${after}`);
    for (const b of bpLog) console.log(`        ${b}`);
    for (const l of log.slice(logLen0).filter((l) => /\|(move|-damage|-heal|-fail|-immune|-crit|-miss|-activate|-start|-end|-status|-curestatus|switch|faint|cant)\|/.test(l))) console.log(`        LINE ${l}`);
    draws.forEach((dl, k) => console.log(`        DRAW[${k}] ${dl}`));
    console.log(`        p1=${fmt(a0)}`);
    console.log(`        p2=${fmt(a1)}`);
    if (entry.stop) break;
  }
  try { streams.omniscient.destroy(); } catch (e) { /* ignore */ }
  return trace;
}

async function main() {
  dumpResolved();

  // ================= A. ERUPTION =================
  console.log('\n\n############ A. ERUPTION ############');
  const erupUser = () => [mon('Typhlosion', ['eruption', 'waterspout', 'splash'], { evs: { spa: 252 } })];
  const erupFoe = () => [mon('Blissey', ['softboiled', 'splash'], { evs: { hp: 252, def: 252, spd: 252 } })];
  const setHp = (hp) => (b) => {
    const m = b.sides[0].active[0]; m.hp = hp;
    for (const sl of m.moveSlots) sl.pp = sl.maxpp;
    const f = b.sides[1].active[0]; f.hp = f.maxhp;
  };
  await run('A1 ERUPTION hp sweep (maxhp 297)', erupUser(), erupFoe(), [
    { label: 'hp=297 -> 150', pre: setHp(297), p1: 'move 1', p2: 'move 1' },
    { label: 'hp=198 -> 100 exact', pre: setHp(198), p1: 'move 1', p2: 'move 1' },
    { label: 'hp=149 -> floor 75.25 = 75', pre: setHp(149), p1: 'move 1', p2: 'move 1' },
    { label: 'hp=148 -> floor 74.74 = 74', pre: setHp(148), p1: 'move 1', p2: 'move 1' },
    { label: 'hp=3   -> floor 1.51 = 1', pre: setHp(3), p1: 'move 1', p2: 'move 1' },
    { label: 'hp=2   -> floor 1.01 = 1', pre: setHp(2), p1: 'move 1', p2: 'move 1' },
    { label: 'hp=1   -> raw 0.505 floors to 0 -> CLAMPED to 1, still HITS', pre: setHp(1), p1: 'move 1', p2: 'move 1', stop: true },
  ]);
  await run('A2 ERUPTION vs WATER SPOUT at the same hp (formula identity)', erupUser(), erupFoe(), [
    { label: 'eruption at hp=150', pre: setHp(150), p1: 'move 1', p2: 'move 1' },
    { label: 'waterspout at hp=150', pre: setHp(150), p1: 'move 2', p2: 'move 1', stop: true },
  ]);
  const a3full = await run('A3a ERUPTION full hp (draw-neutrality arm 1)', erupUser(), erupFoe(),
    [{ p1: 'move 1', p2: 'move 2', stop: true }]);
  const a3low = await run('A3b ERUPTION hp=1 (draw-neutrality arm 2, same seed)', erupUser(), erupFoe(),
    [{ p1: 'move 1', p2: 'move 2', stop: true }], { acts: [{ side: 0, hp: 1 }] });
  console.log(`\n  A3 SEED COMPARE: full=${JSON.stringify(a3full)} low=${JSON.stringify(a3low)} IDENTICAL=${JSON.stringify(a3full) === JSON.stringify(a3low)}`);
  await run('A4 ERUPTION reads hp AT DAMAGE TIME (never-miss Swift from a faster foe lands first)',
    [mon('Typhlosion', ['eruption'], { evs: { spa: 252 } })],
    [mon('Aerodactyl', ['swift'], { evs: { atk: 252, spe: 252 } })],
    [{ label: '297 -> 226 chip, so bp must be floor(150*226/297)=114', p1: 'move 1', p2: 'move 1', stop: true }]);

  // ================= B. REVENGE =================
  console.log('\n\n############ B. REVENGE ############');
  await run('B1 REVENGE priority: a MAX-SPEED user still moves after a slow foe',
    [mon('Aerodactyl', ['revenge', 'splash'], { evs: { atk: 252, spe: 252 } })],
    [mon('Snorlax', ['bodyslam', 'splash'], { evs: { hp: 252, atk: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);
  await run('B2 REVENGE BP: undamaged / damaged-this-turn / damaged-LAST-turn-only',
    [mon('Machamp', ['revenge', 'splash'], { evs: { atk: 252 } })],
    [mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252, def: 252 } })], [
      { label: 'T1 foe splashes (undamaged) -> 60', p1: 'move 1', p2: 'move 2' },
      { label: 'T2 foe seismic tosses first (damaged) -> 120', p1: 'move 1', p2: 'move 1' },
      { label: 'T3 foe splashes (damaged LAST turn only) -> 60', pre: (b) => { for (const sl of b.sides[0].active[0].moveSlots) sl.pp = sl.maxpp; }, p1: 'move 1', p2: 'move 2', stop: true },
    ]);
  await run('B3 REVENGE after the foe hit the user\'s SUBSTITUTE -> 60 (sub-absorbed does NOT count)',
    [mon('Machamp', ['revenge', 'substitute'], { evs: { atk: 252, hp: 252 } })],
    [mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252, def: 252 } })], [
      { label: 'T1 user subs', p1: 'move 2', p2: 'move 2' },
      { label: 'T2 seismic toss breaks the sub, then revenge -> BP?', p1: 'move 1', p2: 'move 1', stop: true },
    ]);
  await run('B4 REVENGE after a FORCED-MISS foe move -> 60',
    [mon('Machamp', ['revenge'], { evs: { atk: 252 } })],
    [mon('Blissey', ['bodyslam'], { evs: { hp: 252, atk: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', forceMiss: true, stop: true }]);
  await run('B5 REVENGE with only a PREVIOUS-turn sandstorm residual -> 60',
    [mon('Machamp', ['revenge', 'splash'], { evs: { atk: 252 } })],
    [mon('Tyranitar', ['sandstorm', 'confuseray', 'splash'], { ability: 'Sand Stream', evs: { hp: 252 } })], [
      { label: 'T1 sandstorm chips the user at the RESIDUAL', p1: 'move 2', p2: 'move 1' },
      { label: 'T2 revenge -> BP?', p1: 'move 1', p2: 'move 3', stop: true },
    ]);
  await run('B6a REVENGE (-4) vs FOCUS PUNCH (-3): focus punch first, revenge DOUBLED',
    [mon('Machamp', ['revenge'], { evs: { atk: 252 } })],
    [mon('Blissey', ['focuspunch'], { evs: { hp: 252, atk: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);
  await run('B6b REVENGE (-4) vs COUNTER (-5): revenge first, and UNdoubled',
    [mon('Machamp', ['revenge'], { evs: { atk: 252 } })],
    [mon('Blissey', ['counter'], { evs: { hp: 252, atk: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);
  // B7: a ZERO-NET-damage hit (Focus Band save at 1 HP) must NOT double. Focus Band is a
  // random save, so seed [5,5,5,5] is the one pinned to proc here.
  await run('B7 REVENGE after a ZERO-net-damage hit (Focus Band save at 1 HP) -> 60',
    [mon('Machamp', ['revenge'], { item: 'Focus Band', evs: { atk: 252 } })],
    [mon('Blissey', ['seismictoss'], { evs: { hp: 252 } })],
    [{ label: 'user at 1 HP, band saves, 0 net damage', p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 0, hp: 1 }], seed: [5, 5, 5, 5] });

  // ================= C. SMELLING SALTS =================
  console.log('\n\n############ C. SMELLING SALTS ############');
  await run('C1 SMELLING SALTS vs a PARALYZED target: 120 + cure, damage-then-cure order',
    [mon('Machamp', ['smellingsalts', 'splash'], { evs: { atk: 252 } })],
    [mon('Blissey', ['splash'], { evs: { hp: 252, def: 252 } })], [
      { label: 'T1 par -> 120 + |-curestatus|…|par|[msg]', p1: 'move 1', p2: 'move 1' },
      { label: 'T2 now cured -> 60', p1: 'move 1', p2: 'move 1', stop: true },
    ], { acts: [{ side: 1, status: 'par' }] });
  await run('C2 SMELLING SALTS vs a BURNED target -> 60, no cure (par only)',
    [mon('Machamp', ['smellingsalts'], { evs: { atk: 252 } })],
    [mon('Blissey', ['splash'], { evs: { hp: 252, def: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }], { acts: [{ side: 1, status: 'brn' }] });
  await run('C3 SMELLING SALTS FORCED MISS vs par -> no cure',
    [mon('Machamp', ['smellingsalts'], { evs: { atk: 252 } })],
    [mon('Blissey', ['splash'], { evs: { hp: 252, def: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', forceMiss: true, stop: true }], { acts: [{ side: 1, status: 'par' }] });
  await run('C4 SMELLING SALTS into a par target behind a SUBSTITUTE -> bp 120, NO cure',
    [mon('Machamp', ['smellingsalts', 'splash'], { evs: { atk: 252 } })],
    [mon('Blissey', ['substitute', 'splash'], { evs: { hp: 252, def: 252 } })], [
      { label: 'T1 foe subs', p1: 'move 2', p2: 'move 1' },
      { label: 'T2 salts into the sub', pre: (b) => { const f = b.sides[1].active[0]; if (!f.status) f.setStatus('par', f, null, true); }, p1: 'move 1', p2: 'move 2', stop: true },
    ], { acts: [{ side: 1, status: 'par' }] });
  await run('C5 SMELLING SALTS vs a paralyzed GHOST: -immune, no BP callback, no cure',
    [mon('Machamp', ['smellingsalts'], { evs: { atk: 252 } })],
    [mon('Gengar', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }], { acts: [{ side: 1, status: 'par' }] });
  await run('C6 SMELLING SALTS that KOs a par target: bp 120, faint, NO -curestatus line',
    [mon('Machamp', ['smellingsalts'], { evs: { atk: 252 } })],
    [mon('Blissey', ['splash'], { evs: { hp: 252 } }), mon('Snorlax', ['splash'], {})],
    [{ p1: 'move 1', p2: 'move 1', stop: true }], { acts: [{ side: 1, status: 'par', hp: 1 }] });
  const fast = (moves) => [mon('Aerodactyl', moves, { evs: { atk: 252, spe: 252 } })];
  const slow = () => [mon('Blissey', ['splash'], { evs: { hp: 252, def: 252 } })];
  const c7a = await run("C7a SMELLING SALTS vs par (cures) — 4 draws/turn", fast(['smellingsalts', 'strength']), slow(),
    [{ label: 'salts into par', p1: 'move 1', p2: 'move 1' }, { label: 'target cured -> no para roll', p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 1, status: 'par' }] });
  const c7b = await run("C7b STRENGTH control vs the SAME par target — 5 draws/turn (the para roll)", fast(['smellingsalts', 'strength']), slow(),
    [{ label: 'strength into par', p1: 'move 2', p2: 'move 1' }, { label: 'target still par -> para roll', p1: 'move 2', p2: 'move 1', stop: true }],
    { acts: [{ side: 1, status: 'par' }] });
  console.log(`\n  C7 SEED TRACES: salts=${JSON.stringify(c7a)}\n                  control=${JSON.stringify(c7b)}`);
  // cure ORDER vs a contact proc: sweep seeds until Static's 1/3 fires.
  for (const seed of [[3, 3, 3, 3], [5, 5, 5, 5], [9, 9, 9, 9]]) {
    await run(`C8 SMELLING SALTS cure BEFORE the contact proc (Static), seed ${seed}`,
      [mon('Machamp', ['smellingsalts'], {})],
      [mon('Blissey', ['splash'], { ability: 'Static', evs: { hp: 252, def: 252 } })],
      [{ p1: 'move 1', p2: 'move 1', stop: true }], { acts: [{ side: 1, status: 'par' }], seed });
  }

  // ================= D. FURY CUTTER =================
  console.log('\n\n############ D. FURY CUTTER ############');
  const fcUser = () => [mon('Scizor', ['furycutter', 'splash'], { evs: { atk: 4 } }), mon('Scyther', ['furycutter', 'splash'], {})];
  const fcFoe = () => [mon('Blissey', ['splash', 'protect', 'substitute'], { evs: { hp: 252, def: 252 } }), mon('Snorlax', ['splash'], { evs: { hp: 252, def: 252 } })];
  const heal = (b) => { const f = b.sides[1].active[0]; f.hp = f.maxhp; for (const sl of b.sides[0].active[0].moveSlots) sl.pp = sl.maxpp; for (const sl of f.moveSlots) sl.pp = sl.maxpp; };
  await run('D1 FURY CUTTER escalation + CAP (8 FORCED hits): 10,20,40,80,160,160,160,160',
    [mon('Scizor', ['furycutter'], { evs: { atk: 4 } })], [mon('Blissey', ['splash'], { evs: { hp: 252, def: 252 } })],
    [1, 2, 3, 4, 5, 6, 7, 8].map((n) => ({ label: `forced hit ${n}`, pre: heal, p1: 'move 1', p2: 'move 1', forceHit: true, stop: n === 8 })));
  await run('D2 a MISS resets it', fcUser(), fcFoe(), [
    { label: 'hit 1', pre: heal, p1: 'move 1', p2: 'move 1', forceHit: true },
    { label: 'hit 2', pre: heal, p1: 'move 1', p2: 'move 1', forceHit: true },
    { label: 'FORCED MISS', pre: heal, p1: 'move 1', p2: 'move 1', forceMiss: true },
    { label: 'after the miss -> 10?', pre: heal, p1: 'move 1', p2: 'move 1', forceHit: true, stop: true },
  ]);
  await run('D3 a DIFFERENT move resets it', fcUser(), fcFoe(), [
    { label: 'hit 1', pre: heal, p1: 'move 1', p2: 'move 1', forceHit: true },
    { label: 'hit 2', pre: heal, p1: 'move 1', p2: 'move 1', forceHit: true },
    { label: 'SPLASH', pre: heal, p1: 'move 2', p2: 'move 1' },
    { label: 'fury cutter again -> 10?', pre: heal, p1: 'move 1', p2: 'move 1', forceHit: true, stop: true },
  ]);
  await run('D4 SWITCHING OUT resets it', fcUser(), fcFoe(), [
    { label: 'hit 1', pre: heal, p1: 'move 1', p2: 'move 1', forceHit: true },
    { label: 'hit 2', pre: heal, p1: 'move 1', p2: 'move 1', forceHit: true },
    { label: 'switch out', pre: heal, p1: 'switch 2', p2: 'move 1' },
    { label: 'switch back', pre: heal, p1: 'switch 2', p2: 'move 1' },
    { label: 'fury cutter -> 10?', pre: heal, p1: 'move 1', p2: 'move 1', forceHit: true, stop: true },
  ]);
  await run('D5 the foe PROTECTING resets it', fcUser(), fcFoe(), [
    { label: 'hit 1', pre: heal, p1: 'move 1', p2: 'move 1', forceHit: true },
    { label: 'hit 2', pre: heal, p1: 'move 1', p2: 'move 1', forceHit: true },
    { label: 'foe PROTECTS', pre: heal, p1: 'move 1', p2: 'move 2' },
    { label: 'fury cutter -> 10?', pre: heal, p1: 'move 1', p2: 'move 1', forceHit: true, stop: true },
  ]);
  await run('D6 the FOE switching does NOT reset it (counter is on the USER)', fcUser(), fcFoe(), [
    { label: 'hit 1', pre: heal, p1: 'move 1', p2: 'move 1', forceHit: true },
    { label: 'hit 2', pre: heal, p1: 'move 1', p2: 'move 1', forceHit: true },
    { label: 'foe switches; hit the NEW target -> 40?', pre: heal, p1: 'move 1', p2: 'switch 2', forceHit: true },
    { label: 'again -> 80?', pre: heal, p1: 'move 1', p2: 'move 1', forceHit: true, stop: true },
  ]);
  await run('D7 a turn the user CANNOT act (asleep) resets it', fcUser(), fcFoe(), [
    { label: 'hit 1', pre: heal, p1: 'move 1', p2: 'move 1', forceHit: true },
    { label: 'hit 2', pre: heal, p1: 'move 1', p2: 'move 1', forceHit: true },
    { label: 'user asleep, cant move', pre: (b) => { heal(b); const u = b.sides[0].active[0]; u.setStatus('slp', u, null, true); u.statusState.time = 3; }, p1: 'move 1', p2: 'move 1' },
    { label: 'woken; fury cutter -> 10?', pre: (b) => { heal(b); b.sides[0].active[0].cureStatus(true); }, p1: 'move 1', p2: 'move 1', forceHit: true, stop: true },
  ]);
  await run('D8 ⚠ THE SUBSTITUTE TRAP: bp reads the multiplier but onHit is SKIPPED, so it never escalates',
    fcUser(), fcFoe(), [
      { label: 'T1 bare target (hit -> mult 2)', pre: heal, p1: 'move 1', p2: 'move 1', forceHit: true },
      { label: 'T2 bare target (hit -> mult 4)', pre: heal, p1: 'move 1', p2: 'move 3', forceHit: true },
      { label: 'T3 into the SUB -> bp 40 but NO escalation', pre: heal, p1: 'move 1', p2: 'move 1', forceHit: true },
      { label: 'T4 into the SUB again -> bp 10 (reset)', pre: heal, p1: 'move 1', p2: 'move 1', forceHit: true, stop: true },
    ]);
  // D9: does the duration-2 volatile register a RESIDUAL duration handler that adds a
  // speed-tie shuffle draw (the hazard the port's duration-1 `reactive` volatile documents)?
  // Mirror-matched speeds so every tie group is size 2. Compare draws/turn against a control.
  const mirror = (mv) => run(`D9 ${mv === 1 ? 'FURY CUTTER (volatile up)' : 'STRENGTH control (no volatile)'} — equal speeds, residual tie groups`,
    [mon('Scizor', ['furycutter', 'strength'], { evs: { atk: 4 } })],
    [mon('Scizor', ['splash'], { evs: { atk: 4 } })],
    [1, 2, 3].map((n) => ({
      label: `turn ${n}`,
      pre: (b) => { const u = b.sides[0].active[0], f = b.sides[1].active[0]; u.hp = u.maxhp; f.hp = f.maxhp; for (const sl of u.moveSlots) sl.pp = sl.maxpp; },
      p1: `move ${mv}`, p2: 'move 1', forceHit: true, stop: n === 3,
    })));
  const d9a = await mirror(1); const d9b = await mirror(2);
  console.log(`\n  D9 SEED TRACES: furycutter=${JSON.stringify(d9a)}\n                  control   =${JSON.stringify(d9b)}`);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
