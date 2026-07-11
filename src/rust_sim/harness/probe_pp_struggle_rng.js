// probe_pp_struggle_rng.js — instrument the gen3 PP-tracking + STRUGGLE draw model
// bit-for-bit, against the OMNISCIENT in-process BattleStream (no server).
//
// SETTLES (do NOT trust the task hints — the sim is the source of truth):
//   1. PP INIT: is a moveslot's PP the moveset's base PP or base*8/5 (3 PP-ups default)?
//      Dump side.active[0].moveSlots[k].pp / .maxpp at battle start.
//   2. PP DECREMENT: −1 per USE. Does a MISS / an IMMUNE hit still decrement (yes)?
//      Does a FULL-PARA / SLEEP / FLINCH turn (mon can't move) decrement (no — deductPP
//      runs AFTER runEvent('BeforeMove') passes)? Is it DRAW-FREE (yes)?
//   3. PRESSURE −2: a move TARGETING a Pressure holder deducts 2 PP (DeductPP event, no RNG).
//   4. FORCED STRUGGLE: when ALL moves are 0 PP the request offers ONLY Struggle and the
//      sim auto-selects it. What choice does side.choose accept / what does the request show?
//   5. STRUGGLE MECHANICS + DRAW MODEL: type ('???' typeless), BP 50, category (phys), does
//      it draw accuracy (gen3 struggle accuracy === true → never-miss → NO draw)? crit roll?
//      damage roll? recoil = ¼ maxhp (gen<=4 struggleRecoil = trunc(maxhp/4), min 1) vs
//      ¼ damage-dealt — PROVE via the recoil HP delta vs damage dealt vs maxhp/4. Draw COUNT
//      of a Struggle turn (acc? crit? dmg? recoil draw-free?).
//   6. PP does NOT reset on switch-out in gen3 (confirm moveSlots[k].pp persists across a
//      switch-out/in of the SAME mon).
//
// We wrap battle.prng.next to count raw draws per decision window + dump moveSlots PP each
// boundary. Struggle is FORCED by exhausting a Choice-Band-style single usable move (we give
// a mon ONE damaging move + fillers we never pick, then spam that move past its PP).
//
// Run:  node src/rust_sim/harness/probe_pp_struggle_rng.js
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
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

function dumpResolved() {
  const d = Dex.forFormat(FORMAT);
  console.log('=== resolved gen3 move facts (calculatePP over data.pp) ===');
  for (const id of ['surf', 'earthquake', 'struggle', 'thunderbolt', 'seismictoss', 'thunder']) {
    const m = d.moves.get(id);
    // Emulate the Pokemon ctor: ppUps = noPPBoosts||trumpcard ? 0 : 3.
    const ppUps = (m.noPPBoosts || id === 'trumpcard') ? 0 : 3;
    const basePP = m.pp;
    const maxpp = m.noPPBoosts ? m.pp : Math.floor(m.pp * (5 + ppUps) / 5);
    console.log(`  ${id}: type=${m.type} bp=${m.basePower} cat=${m.category} acc=${m.accuracy} ` +
      `pp=${basePP} noPPBoosts=${!!m.noPPBoosts} => maxpp(ppUps=${ppUps})=${maxpp} ` +
      `struggleRecoil=${!!m.struggleRecoil} recoil=${JSON.stringify(m.recoil)} priority=${m.priority}`);
  }
}

function moveSlotsOf(m) {
  if (!m || !m.moveSlots) return '-';
  return m.moveSlots.map((s) => `${s.id}:${s.pp}/${s.maxpp}`).join(' ');
}

async function run(label, p1team, p2team, plan, inject) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  const seed = [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  for (const inj of (inject || [])) {
    const m = inj.side === undefined ? null : battle.sides[inj.side].active[0];
    if (!m) continue;
    if (inj.status) m.setStatus(inj.status, m, null, true);
    if (inj.hp !== undefined) m.hp = inj.hp;
    // Force a single move's PP down to `setpp` (to exhaust it fast).
    if (inj.setpp !== undefined && m.moveSlots[inj.setpp.slot]) {
      m.moveSlots[inj.setpp.slot].pp = inj.setpp.pp;
      if (m.baseMoveSlots[inj.setpp.slot]) m.baseMoveSlots[inj.setpp.slot].pp = inj.setpp.pp;
    }
  }

  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  console.log(`  START moveSlots p1=[${moveSlotsOf(battle.sides[0].active[0])}] ` +
    `p2=[${moveSlotsOf(battle.sides[1].active[0])}]`);

  let i = 0, safety = 0;
  while (!battle.ended && safety < 80) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const before = battle.prng.getSeed();
    const dc0 = drawCount;
    // Snapshot HP for the recoil measurement.
    const hpBeforeP1 = battle.sides[0].active[0] ? battle.sides[0].active[0].hp : 0;
    // What does the ACTIVE request offer this decision (Struggle-only detection)?
    let reqMoves = '-';
    const ar = battle.sides[0].activeRequest;
    if (ar && ar.active && ar.active[0] && ar.active[0].moves) {
      reqMoves = ar.active[0].moves.map((mm) => `${mm.id}${mm.pp !== undefined ? `(${mm.pp})` : ''}${mm.disabled ? 'X' : ''}`).join(',');
    }
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    try { if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`); } catch (e) { console.log('  p1 choose err', e.message); }
    try { if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`); } catch (e) { console.log('  p2 choose err', e.message); }
    for (let k = 0; k < 18; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'}${m.fainted ? ' FNT' : ''}` : '-';
    console.log(`  [${rs}] req=[${reqMoves}] ${JSON.stringify(entry)} draws=${drawCount - dc0}  before=${before} after=${after}`);
    console.log(`        p1=${fmt(a0)} slots=[${moveSlotsOf(a0)}] | p2=${fmt(a1)} slots=[${moveSlotsOf(a1)}]`);
    if (a0 && entry.p1 && /move/.test(entry.p1 || '')) {
      const dealt = (a1 ? (a1.maxhp) : 0); // not used
      const recoil = hpBeforeP1 - a0.hp; // p1's own HP loss this turn (recoil + any foe dmg)
      if (recoil !== 0) console.log(`        p1 HP delta this turn = ${recoil} (maxhp/4=${Math.floor(a0.maxhp / 4)})`);
    }
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  dumpResolved();

  // (A) PP INIT + basic decrement. A mon with Surf(15→24) + Earthquake(10→16). Confirm
  //     maxpp = base*8/5 and −1 per use, draw-free-relative (the PP decrement adds NO draw
  //     beyond the move's own acc/crit/dmg).
  await run('PP init + decrement (Surf/Earthquake, 3 PP-ups default)',
    [mon('Suicune', ['surf', 'earthquake'], { evs: { hp: 252, spa: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // Surf → pp 24->23
      { p1: 'move 2', p2: 'move 1' }, // EQ (immune? Snorlax is Normal → not immune) pp 16->15
      { p1: 'move 1', p2: 'move 1' },
    ]);

  // (B) MISS still decrements. A low-accuracy move (Thunder 70%) may miss; PP still drops.
  await run('MISS still decrements PP (Thunder 70%)',
    [mon('Zapdos', ['thunder', 'thunderbolt'], { evs: { spa: 252, spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // Thunder (may miss) pp drops regardless
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
    ]);

  // (C) IMMUNE hit still decrements. EQ into a Levitate/Flying mon: -immune but PP -1.
  await run('IMMUNE hit still decrements PP (EQ into Levitate)',
    [mon('Tyranitar', ['earthquake', 'crunch'], { evs: { atk: 252 } })],
    [mon('Gengar', ['splash'], { ability: 'Levitate', evs: { hp: 252 } })], // Levitate → EQ immune
    [
      { p1: 'move 1', p2: 'move 1' }, // EQ → -immune, pp still -1
      { p1: 'move 1', p2: 'move 1' },
    ]);

  // (D) FULL-PARA / can't-move does NOT decrement. Paralyze p1; a full-para turn draws the
  //     para roll but skips deductPP (BeforeMove aborts).
  await run("can't-move (full-para) does NOT decrement PP",
    [mon('Snorlax', ['bodyslam', 'splash'], { evs: { hp: 252, atk: 252 } })],
    [mon('Regirock', ['splash'], { evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // Body Slam (pp -1)
      { p1: 'move 1', p2: 'move 1' }, // possibly full-para → no pp decrement on the para'd turn
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
    ],
    [{ side: 0, status: 'par' }]);

  // (E) PRESSURE −2. A move targeting a Pressure holder deducts 2 PP.
  await run('PRESSURE holder: foe move deducts 2 PP',
    [mon('Snorlax', ['bodyslam', 'splash'], { evs: { hp: 252, atk: 252 } })],
    [mon('Suicune', ['splash'], { ability: 'Pressure', evs: { hp: 252, def: 252 } })], // Pressure
    [
      { p1: 'move 1', p2: 'move 1' }, // Body Slam vs Pressure Suicune → pp -2
      { p1: 'move 1', p2: 'move 1' }, // -2 again
    ]);

  // (F) FORCED STRUGGLE + its draw model + recoil. A Choice-Band Tyranitar with ONE usable
  //     move (Rock Slide) whose PP we crush to 1; spam it → out of PP → forced Struggle.
  //     (Choice Band locks it to that move; the fillers are the other slots.) We measure the
  //     Struggle turn's draw count + the recoil (p1 HP loss vs damage dealt vs maxhp/4).
  await run('FORCED STRUGGLE (CB TTar out of Rock Slide PP) + draw model + recoil',
    [mon('Tyranitar', ['rockslide', 'earthquake', 'crunch', 'roar'],
      { item: 'Choice Band', ability: 'Sand Stream', evs: { hp: 252, atk: 252 } })],
    [mon('Skarmory', ['splash'], { evs: { hp: 252, def: 252 } })], // bulky, survives; Steel
    [
      { p1: 'move 1', p2: 'move 1' }, // Rock Slide (Choice-locks to it) — pp crushed to 1 → 0
      { p1: 'move 1', p2: 'move 1' }, // now 0 PP → request forces Struggle; we submit `move 1` (the sim auto-substitutes) OR `move 1`
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
    ],
    [{ side: 0, setpp: { slot: 0, pp: 1 } }]);

  // (G) STRUGGLE into a GHOST (typeless '???' — does it hit? gen3 struggle onTryHit).
  await run('STRUGGLE into a GHOST (typeless — hits? recoil?)',
    [mon('Tyranitar', ['rockslide', 'earthquake', 'crunch', 'roar'],
      { item: 'Choice Band', ability: 'Sand Stream', evs: { hp: 252, atk: 252 } })],
    [mon('Gengar', ['splash'], { ability: 'Levitate', evs: { hp: 252 } })], // Ghost
    [
      { p1: 'move 1', p2: 'move 1' }, // Rock Slide (pp 1 -> 0)
      { p1: 'move 1', p2: 'move 1' }, // Struggle into Gengar (Ghost) — hit? recoil?
      { p1: 'move 1', p2: 'move 1' },
    ],
    [{ side: 0, setpp: { slot: 0, pp: 1 } }]);

  // (H) PP does NOT reset on switch-out (gen3). Use a move, switch out, switch back → pp stays.
  await run('PP persists across switch-out (gen3, no reset)',
    [mon('Suicune', ['surf', 'icebeam'], { evs: { hp: 252, spa: 252 } }),
     mon('Snorlax', ['bodyslam'], { evs: { hp: 252, atk: 252 } })],
    [mon('Regirock', ['splash'], { evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // Surf (pp 24->23)
      { p1: 'switch 2', p2: 'move 1' }, // switch to Snorlax
      { p1: 'switch 2', p2: 'move 1' }, // switch back to Suicune → Surf pp STILL 23?
      { p1: 'move 1', p2: 'move 1' }, // Surf again → 22
    ]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
