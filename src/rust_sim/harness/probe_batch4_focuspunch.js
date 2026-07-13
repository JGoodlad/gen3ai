// probe_batch4_focuspunch.js — ground-truth FOCUS PUNCH (focuspunch) bit-for-bit vs the
// OMNISCIENT in-process BattleStream (no server). The mod chain is the ONLY oracle.
//
// Resolved gen3 (via the gen4 mod, inherited): Focus Punch is
//   - a 150-BP PHYSICAL Fighting move, priority -3, flags contact+protect+punch,
//   - beforeTurnCallback(pokemon){ pokemon.addVolatile('focuspunch'); } — queued as a
//     `beforeTurnMove` action (order 5) at turn start; the volatile's onStart emits
//     `|-singleturn|…|move: Focus Punch`,
//   - the volatile's onHit sets lostFocus=true when the user is hit by a NON-Status move,
//   - the volatile's onTryAddVolatile blocks FLINCH (a Focus Punch user can't be flinched),
//   - the move's onTry cancels the punch (attrLastMove('[still]') + `|cant|…|Focus Punch`,
//     returns null) if lostFocus.
//   NOTE: gen3 has NO beforeMoveCallback / priorityChargeCallback (gen4 nulls them) and
//   NO onTryMove — the cancel is the move's onTry.
//
// This probe nails down, against the sim's PRNG:
//   1. WHERE the beforeTurn/beforeTurnMove actions insert + whether they DRAW (relative to
//      the eachEvent('BeforeTurn') shuffle the port already models).
//   2. The "took damage this turn" cancel condition + the EXACT draw model of a cancelled
//      punch (which pre-move events still fire before the cancel: acc? para? flinch?
//      confusion self-hit? sleep? freeze?).
//   3. A LANDED Focus Punch's normal draw model (acc/crit/dmg/QuickClaw).
//   4. Edge interactions: user flinched / fully-para / asleep / hit-behind-Substitute (does
//      chip through a Sub count as "took damage"?).
//
// Run:  node src/rust_sim/harness/probe_batch4_focuspunch.js
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
  console.log('=== resolved gen3 focuspunch ===');
  const m = d.moves.get('focuspunch');
  console.log(`  focuspunch: cat=${m.category} bp=${m.basePower} acc=${m.accuracy} pri=${m.priority} ` +
    `target=${m.target} type=${m.type} flags=${JSON.stringify(m.flags)}`);
  console.log(`  beforeTurnCallback: ${m.beforeTurnCallback ? m.beforeTurnCallback.toString() : '-'}`);
  console.log(`  onTry:              ${m.onTry ? m.onTry.toString() : '-'}`);
  console.log(`  beforeMoveCallback: ${m.beforeMoveCallback === undefined ? 'undefined' : typeof m.beforeMoveCallback}`);
  console.log(`  priorityChargeCallback: ${m.priorityChargeCallback === undefined ? 'undefined' : typeof m.priorityChargeCallback}`);
  const c = d.conditions.get('focuspunch');
  for (const k of ['onStart', 'onHit', 'onTryAddVolatile']) {
    if (c && c[k]) console.log(`  cond.${k}: ${c[k].toString()}`);
  }
  console.log(`  cond.duration=${c && c.duration} noCopy=${c && c.noCopy}`);
}

// Instrument the PRNG with call-site tags so we can see WHICH draws fire.
function instrument(battle) {
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  const draws = [];
  rng.next = function (...a) {
    // capture a compact call-site (the sim frame just above prng)
    const st = new Error().stack.split('\n').slice(2, 7)
      .map((l) => l.trim().replace(/^at\s+/, ''))
      .filter((l) => /battle|actions|pokemon|field|dex/.test(l))
      .map((l) => l.replace(/\s*\(.*$/, '').replace(/^Battle(Actions)?\./, ''))
      .slice(0, 2).join(' < ');
    const v = realNext(...a);
    draws.push(st);
    return v;
  };
  return draws;
}

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
  for (const inj of ((inject && inject.acts) || [])) {
    const m = inj.side === undefined ? null : battle.sides[inj.side].active[0];
    if (!m) continue;
    if (inj.status) m.setStatus(inj.status, m, null, true);
    if (inj.hp !== undefined) m.hp = inj.hp;
    if (inj.item !== undefined) m.item = inj.item;
    if (inj.boost) m.boostBy(inj.boost);
  }

  const draws = instrument(battle);

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 40) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const dc0 = draws.length;
    const logLen0 = log.length;
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const vol = (m) => m ? Object.keys(m.volatiles).join(',') : '-';
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'} vols=[${vol(m)}]` : '-';
    console.log(`  [${rs}] ${JSON.stringify(entry)} draws=${draws.length - dc0}  seedAfter=${after}`);
    console.log(`        p1=${fmt(a0)}`);
    console.log(`        p2=${fmt(a1)}`);
    for (let k = dc0; k < draws.length; k++) console.log(`        DRAW#${k - dc0} ${draws[k]}`);
    const newLines = log.slice(logLen0).filter((l) =>
      /-start|-singleturn|-damage|-heal|-boost|-unboost|-fail|-immune|-activate|-end|cant|move\|/.test(l));
    for (const l of newLines) console.log(`        LINE ${l}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  dumpResolved();

  // ---- 1. LANDED Focus Punch: foe uses a non-damaging move so the user keeps focus.
  //         Enumerate the full draw model (beforeTurn shuffle? beforeTurnMove? acc/crit/dmg/QuickClaw).
  await run('FP LANDED: foe Splash, user Focus Punch lands (draw model)',
    [mon('Machamp', ['focuspunch', 'splash'], { evs: { atk: 252 } })],
    [mon('Blissey', ['splash', 'softboiled'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }]);

  // Control: user Splash instead of Focus Punch — isolates the beforeTurnMove/onTry draws.
  await run('CONTROL: user Splash vs foe Splash (no focuspunch volatile)',
    [mon('Machamp', ['focuspunch', 'splash'], { evs: { atk: 252 } })],
    [mon('Blissey', ['splash', 'softboiled'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' }]);

  // ---- 2. CANCELLED Focus Punch: foe damages the user first (pri -3 → user moves last).
  await run('FP CANCELLED: foe Tackle hits first → lostFocus → cant (draw model)',
    [mon('Machamp', ['focuspunch', 'splash'], { evs: { atk: 252, hp: 252 } })],
    [mon('Snorlax', ['tackle', 'splash'], { evs: { atk: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);

  // Control for the cancel: same board but foe Splash (lands) — diff the draw sequences.
  await run('CONTROL for cancel: foe Splash (FP lands) — same board',
    [mon('Machamp', ['focuspunch', 'splash'], { evs: { atk: 252, hp: 252 } })],
    [mon('Snorlax', ['tackle', 'splash'], { evs: { atk: 252 } })],
    [{ p1: 'move 1', p2: 'move 2' }]);

  // ---- 3. CANCELLED but a STATUS move hit the user: does a status move set lostFocus?
  //         (onHit: move.category !== 'Status' → so a Thunder Wave should NOT cancel.)
  await run('FP vs foe status move: Thunder Wave hits, then FP — does it still land?',
    [mon('Machamp', ['focuspunch', 'splash'], { evs: { atk: 252, hp: 252 } })],
    [mon('Jolteon', ['thunderwave', 'splash'], { evs: { spe: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);

  // ---- 4a. FULLY PARALYZED user with a pending Focus Punch: does the para roll draw
  //          BEFORE the onTry cancel/land? (para is onBeforeMove, before useMove/onTry.)
  await run('FP + PARALYSIS: user para, foe Splash — para roll vs onTry order',
    [mon('Machamp', ['focuspunch', 'splash'], { evs: { atk: 252, hp: 252 } })],
    [mon('Blissey', ['splash'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    { acts: [{ side: 0, status: 'par' }] });

  // ---- 4b. ASLEEP user with a pending Focus Punch: sleep cant before onTry (no acc/land).
  await run('FP + SLEEP: user asleep, foe Splash — sleep cant, focuspunch volatile fate',
    [mon('Machamp', ['focuspunch', 'splash'], { evs: { atk: 252, hp: 252 } })],
    [mon('Blissey', ['splash'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    { acts: [{ side: 0, status: 'slp' }] });

  // ---- 5. FLINCH: the focuspunch volatile's onTryAddVolatile blocks flinch. A faster foe
  //         Fake Out / Rock Slide should NOT flinch the Focus Punch user — but it DOES damage
  //         it (non-status) → lostFocus → cant. Probe whether flinch is blocked AND cancel fires.
  await run('FP + FLINCH: foe Rock Slide (flinch move) hits — flinch blocked? cancel?',
    [mon('Machamp', ['focuspunch', 'splash'], { evs: { atk: 252, hp: 252 } })],
    [mon('Aerodactyl', ['rockslide', 'splash'], { evs: { spe: 252, atk: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);

  // ---- 6. BEHIND A SUBSTITUTE: user subs turn 1; turn 2 user Focus Punch while foe Tackles
  //         the SUB. Does chip absorbed by the sub count as "took damage" (lostFocus)?
  await run('FP behind SUB: foe Tackle breaks/chips the sub — does FP keep focus?',
    [mon('Machamp', ['focuspunch', 'substitute'], { evs: { atk: 252, hp: 252 } })],
    [mon('Snorlax', ['tackle', 'splash'], { evs: { atk: 4 } })],
    [{ p1: 'move 2', p2: 'move 2' }, { p1: 'move 1', p2: 'move 1' }]);

  // ---- 7. BOTH sides Focus Punch at a SPEED TIE: two beforeTurnMove(order 5) actions tie →
  //         does the action-sort draw an extra shuffle? (the beforeTurn-phase draw question.)
  await run('FP MIRROR speed-tie: both Focus Punch — beforeTurnMove tie-shuffle?',
    [mon('Machamp', ['focuspunch', 'splash'], { evs: { atk: 252 } })],
    [mon('Machamp', ['focuspunch', 'splash'], { evs: { atk: 252 } })],
    [{ p1: 'move 2', p2: 'move 2' }, { p1: 'move 1', p2: 'move 1' }]);

  // Control for the mirror: both Splash (no beforeTurnMove) at the same tie — baseline draws.
  await run('CONTROL mirror: both Splash speed-tie (no beforeTurnMove)',
    [mon('Machamp', ['focuspunch', 'splash'], { evs: { atk: 252 } })],
    [mon('Machamp', ['focuspunch', 'splash'], { evs: { atk: 252 } })],
    [{ p1: 'move 2', p2: 'move 2' }]);

  // Half-mirror: only ONE side Focus Punch at a tie — one beforeTurnMove (order 5), no tie there.
  await run('HALF-mirror: p1 Focus Punch, p2 Splash, speed-tie — one beforeTurnMove',
    [mon('Machamp', ['focuspunch', 'splash'], { evs: { atk: 252 } })],
    [mon('Machamp', ['focuspunch', 'splash'], { evs: { atk: 252 } })],
    [{ p1: 'move 1', p2: 'move 2' }]);

  // ---- 8. BULKY Snorlax mirror (NOBODY dies → full turns WITH Quick Claw), so the
  //         beforeTurnMove per-action Update delta is CLEAN (not faint-truncated).
  //         Result (each a single full turn, speed TIE):
  //           both Splash      → 7 draws (6 shuffle + QuickClaw)  [no beforeTurnMove]
  //           one FP + Splash  → 11 (7 shuffle + acc/crit/dmg + QC) [+1 shuffle: the FP
  //             user's beforeTurnMove runAction Update]  (the move-priority tie is GONE so the
  //             move-sort tie-shuffle is replaced by the beforeTurnMove Update — net a wash vs
  //             a distinct-speed board, but tie-visible here)
  //           both FP          → 15 (11 shuffle + acc/crit/dmg + QC) [2 beforeTurnMove Updates
  //             + the beforeTurnMove(order 5) action-sort tie]; the FIRST FP lands, the SECOND
  //             is CANCELLED by lostFocus (draw-free) — one FP's acc/crit/dmg only.
  const bulk = () => [mon('Snorlax', ['focuspunch', 'splash'], { evs: { hp: 252, def: 252 } })];
  await run('BULKY mirror both SPLASH (tie, no beforeTurnMove)', bulk(), bulk(),
    [{ p1: 'move 2', p2: 'move 2', stop: true }]);
  await run('BULKY mirror ONE FP + Splash (tie, one beforeTurnMove, FP lands)', bulk(), bulk(),
    [{ p1: 'move 1', p2: 'move 2', stop: true }]);
  await run('BULKY mirror both FP (tie, two beforeTurnMove; 1st lands, 2nd cancelled)', bulk(), bulk(),
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // ---- 9. FLINCH move that HITS (seed chosen so Rock Slide connects): the FP user is
  //         damaged → lostFocus → FP CANCELLED, the flinch is BLOCKED by focuspunch's
  //         onTryAddVolatile, and the flinch SECONDARY random(100) STILL DRAWS (draw-then-block).
  //         Draw model of the cancelled turn: RS acc + crit + dmg + flinch-random(100) + QuickClaw
  //         (5 draws, distinct speed), FP itself draws NOTHING.
  await run('FP + FLINCH HIT: Rock Slide connects → cancel; flinch blocked; secondary draws',
    [mon('Machamp', ['focuspunch', 'splash'], { evs: { hp: 252, def: 252 } })],
    [mon('Aerodactyl', ['rockslide', 'splash'], { evs: { spe: 252, atk: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { seed: [1, 2, 3, 4] });
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
