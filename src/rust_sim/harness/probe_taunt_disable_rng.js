// probe_taunt_disable_rng.js — instrument the gen3 TAUNT + DISABLE draw model
// bit-for-bit, against the OMNISCIENT in-process BattleStream (no server).
//
// SETTLES (do NOT trust the task hints — the sim is the source of truth):
//   1. TAUNT: type (Dark), accuracy (100), draw model of the MOVE (accuracy roll? — it's
//      NOT never-miss so it DRAWS randomChance(100,100)). The `taunt` VOLATILE's duration:
//      gen3 is a FIXED 2 turns (`duration: 2`, NO durationCallback) → NO duration draw.
//      Dump target.volatiles.taunt.duration + the per-decision raw draw count to PROVE
//      the taunt-move turn draws ONLY accuracy (+ any action-order/eachEvent shuffle).
//      When does the volatile TICK DOWN / expire (onResidualOrder 10, subOrder 15)?
//      Does onDisableMove make every Status move un-selectable (request `disabled`)?
//   2. DISABLE: type (Normal), accuracy (55 — NOT 100! the task hint was WRONG), and the
//      DURATION DRAW: `durationCallback: this.random(2, 6)` → ONE draw at onStart. PROVE
//      the raw draw count of a Disable turn = accuracy + ONE random(2,6) (+ shuffles).
//      Which move gets disabled (target's lastMove). Disable into a mon that has NOT moved
//      yet (`!pokemon.lastMove` → onStart returns false → FAILS; is the random(2,6) still
//      drawn? durationCallback runs BEFORE onStart in addVolatile — PROVE the draw count).
//      The `willMove`/duration++ subtlety. The tick-down + expiry (does the disabled move
//      free up after the duration?).
//   3. SELECTION RESTRICTION: while taunted, the request offers Status moves as
//      `disabled:true` (un-selectable); while disabled, the ONE disabled move is
//      `disabled:true`. Forced Struggle when Taunt+Disable leave nothing usable.
//   4. DRAW-COUNT NEUTRALITY: Taunt = accuracy-only (no duration draw); Disable =
//      accuracy + ONE random(2,6). Pin the exact per-move draw sequence.
//
// We wrap battle.prng.next to count raw draws per decision window + dump the volatile
// state + the request's disabled flags each boundary.
//
// ===================================================================================
// CONFIRMED FINDINGS (vs the omniscient sim — the source of truth; the task hints were
// WRONG on Disable's accuracy):
//
//   TAUNT  (`taunt`): Dark, Status, accuracy 100 (NOT never-miss → DRAWS randomChance(100,100)),
//     flags protect:1 (Protect BLOCKS it), bypasssub:1 (Substitute does NOT block it). Volatile
//     `duration: 2` FIXED — NO duration draw (no durationCallback). `onDisableMove` disables
//     EVERY Status-category move (all become un-selectable → the request shows them disabled;
//     if all usable moves are Status → forced Struggle). `onBeforeMove` ALSO blocks a Status
//     move at execution (`cant`), but the selection restriction is the mechanism the fuzz hits.
//     Residual duration handler at onResidualOrder 10, onResidualSubOrder 15 (a NEW order-10
//     slot AFTER Leftovers/leech/DoT sub 4/5/6); it decrements the duration each residual and
//     ENDs the volatile at 0. So Taunt = accuracy-only, draw-free duration + a residual tick.
//
//   DISABLE (`disable`): Normal, Status, accuracy 55 (NOT 100! — the task hint was wrong; NOT
//     never-miss → DRAWS randomChance(55,100)), flags protect:1 (Protect BLOCKS), bypasssub:1
//     (Substitute does NOT block), noCopy. The DURATION is `durationCallback: this.random(2,6)`
//     → ONE random(2,6) draw (result ∈ {2,3,4,5}) at addVolatile, +1 in onStart iff the target
//     has ALREADY moved this turn (`!willMove(target)` → disabler is SLOWER / moves 2nd). BUT
//     the move's `onTryHit` FAILS (draw-free, BEFORE addVolatile) if the target has NO lastMove
//     OR its lastMove is Struggle — so a Disable into a not-yet-moved target draws ONLY accuracy
//     (NO random(2,6)). On a landed hit with a real lastMove, that ONE move (the target's
//     lastMove) becomes un-selectable; the others stay usable (or Struggle if disable+PP leave
//     nothing). Residual duration handler at order FALSE (NO_ORDER), subOrder 2 (ties with
//     protect/stall/flinch — the same VolatileDuration machinery), ticking −1 per residual +
//     ENDing at 0. Clears on switch-out.
//
//   DRAW-COUNT: Taunt adds ZERO extra PRNG (accuracy-only). Disable adds exactly ONE random(2,6)
//     on a LANDED hit into a mon with a lastMove; a no-lastMove Disable adds nothing past
//     accuracy. Both residual handlers participate in the residual speed-sort tie-shuffle.
// ===================================================================================
//
// Run:  node src/rust_sim/harness/probe_taunt_disable_rng.js
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
  console.log('=== resolved gen3 move facts ===');
  for (const id of ['taunt', 'disable']) {
    const m = d.moves.get(id);
    const c = m.condition || {};
    console.log(`  ${id}: type=${m.type} cat=${m.category} acc=${m.accuracy} bp=${m.basePower} ` +
      `priority=${m.priority} target=${m.target} volatileStatus=${m.volatileStatus} ` +
      `flags=${JSON.stringify(m.flags)}`);
    console.log(`       condition.duration=${c.duration} onResidualOrder=${c.onResidualOrder} ` +
      `onResidualSubOrder=${c.onResidualSubOrder} durationCallback=${!!c.durationCallback} ` +
      `onBeforeMovePriority=${c.onBeforeMovePriority}`);
  }
}

function volInfo(m) {
  if (!m) return '-';
  const v = [];
  if (m.volatiles.taunt) v.push(`taunt(dur=${m.volatiles.taunt.duration})`);
  if (m.volatiles.disable) v.push(`disable(dur=${m.volatiles.disable.duration},move=${m.volatiles.disable.move})`);
  return v.length ? v.join(',') : 'none';
}
function reqMovesOf(battle, side) {
  const ar = battle.sides[side].activeRequest;
  if (!ar || !ar.active || !ar.active[0] || !ar.active[0].moves) return '-';
  return ar.active[0].moves.map((mm) =>
    `${mm.id}${mm.pp !== undefined ? `(${mm.pp})` : ''}${mm.disabled ? '[X]' : ''}`).join(',');
}
function lastMoveOf(m) { return m && m.lastMove ? m.lastMove.id : '-'; }

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
    if (inj.setpp !== undefined && m.moveSlots[inj.setpp.slot]) {
      m.moveSlots[inj.setpp.slot].pp = inj.setpp.pp;
      if (m.baseMoveSlots[inj.setpp.slot]) m.baseMoveSlots[inj.setpp.slot].pp = inj.setpp.pp;
    }
  }

  let drawCount = 0;
  const draws = [];
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);

  let i = 0, safety = 0;
  while (!battle.ended && safety < 80) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const before = battle.prng.getSeed();
    const dc0 = drawCount;
    const reqP1 = reqMovesOf(battle, 0);
    const reqP2 = reqMovesOf(battle, 1);
    const lmP1 = lastMoveOf(battle.sides[0].active[0]);
    const lmP2 = lastMoveOf(battle.sides[1].active[0]);
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    try { if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`); } catch (e) { console.log('  p1 choose err', e.message); }
    try { if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`); } catch (e) { console.log('  p2 choose err', e.message); }
    for (let k = 0; k < 18; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'}${m.fainted ? ' FNT' : ''}` : '-';
    console.log(`  [${rs}] ${JSON.stringify(entry)} draws=${drawCount - dc0}  before=${before} after=${after}`);
    console.log(`        p1=${fmt(a0)} vol=[${volInfo(a0)}] lastMove(pre)=${lmP1} req(pre)=[${reqP1}]`);
    console.log(`        p2=${fmt(a1)} vol=[${volInfo(a1)}] lastMove(pre)=${lmP2} req(pre)=[${reqP2}]`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  dumpResolved();

  // (A) TAUNT lands: p1 Taunts p2. Confirm: taunt-move turn draws ONLY accuracy (100 → always
  //     passes, NO duration draw), the taunt volatile duration=2, and p2's Status moves become
  //     disabled in its request (Toxic [X], but Body Slam usable). Then watch the duration tick
  //     down (2 → 1 → expire) and the Status move free up.
  await run('TAUNT lands (Dark, acc100, NO duration draw) + Status disabled + expiry',
    [mon('Gengar', ['taunt', 'shadowball'], { evs: { spa: 252, spe: 252 } })],
    [mon('Snorlax', ['bodyslam', 'toxic'], { evs: { hp: 252, atk: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // Taunt (draws accuracy only) → p2 taunted, Toxic disabled
      { p1: 'move 2', p2: 'move 1' }, // p2 can only Body Slam now (Toxic [X])
      { p1: 'move 2', p2: 'move 1' }, // taunt still up? tick
      { p1: 'move 2', p2: 'move 1' }, // taunt expired → Toxic usable again
      { p1: 'move 2', p2: 'move 2' }, // p2 Toxic now allowed
    ]);

  // (B) TAUNT into a mon whose ONLY moves are Status → forced Struggle. p2 has Toxic + Recover
  //     (both Status). After Taunt, both are disabled → p2 must Struggle.
  await run('TAUNT forces Struggle (all Status moves disabled)',
    [mon('Gengar', ['taunt', 'shadowball'], { evs: { spa: 252, spe: 252 } })],
    [mon('Blissey', ['toxic', 'softboiled'], { evs: { hp: 252, def: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // Taunt → both p2 Status moves disabled
      { p1: 'move 2', p2: 'move 1' }, // p2 request: Struggle-only?
      { p1: 'move 2', p2: 'move 1' },
    ]);

  // (C) DISABLE lands: p1 Disables p2's last move. p2 must move FIRST (be faster) so it HAS a
  //     lastMove. Then p1 Disables it. Confirm: Disable turn draws accuracy(55) + ONE
  //     random(2,6) duration draw, the disable volatile records the move + duration, and that
  //     move is [X] in p2's request while the others remain usable. Watch tick-down + free-up.
  //     p1 = a WALL (Blissey) that survives a Body Slam so the battle lasts.
  await run('DISABLE lands (Normal, acc55, ONE random(2,6) duration) + move disabled + expiry',
    [mon('Blissey', ['disable', 'softboiled'], { evs: { hp: 252, def: 252 }, nature: 'Bold', ivs: { ...IV31, spe: 0 } })],
    [mon('Snorlax', ['bodyslam', 'rest'], { evs: { hp: 252, atk: 252 } })], // faster → moves first, sets lastMove
    [
      { p1: 'move 2', p2: 'move 1' }, // p2 Body Slam (sets lastMove=bodyslam); p1 Soft-Boiled
      { p1: 'move 1', p2: 'move 1' }, // p1 Disable bodyslam (acc 55 may miss!) → draw model
      { p1: 'move 2', p2: 'move 2' }, // p2 tries Rest (bodyslam [X])
      { p1: 'move 2', p2: 'move 2' },
      { p1: 'move 2', p2: 'move 2' },
      { p1: 'move 2', p2: 'move 2' }, // disable should expire → bodyslam usable again
      { p1: 'move 2', p2: 'move 1' },
    ]);

  // (D) DISABLE into a mon that has NOT moved yet (no lastMove) → onStart returns false → FAILS.
  //     Is the random(2,6) still drawn? durationCallback runs in addVolatile BEFORE onStart, so
  //     the draw likely fires even on a fail — PROVE the draw count. p1 must be FASTER so it
  //     Disables BEFORE p2 has moved this battle. p1 = fast + BULKY so it survives.
  await run('DISABLE into a mon that has NOT moved (fail) — is random(2,6) still drawn?',
    [mon('Aerodactyl', ['disable', 'rockslide'], { evs: { hp: 252, spe: 252 } })], // fast, Flying (Body Slam not immune)
    [mon('Snorlax', ['bodyslam', 'rest'], { evs: { hp: 252, atk: 252 }, nature: 'Brave', ivs: { ...IV31, spe: 0 } })], // slow
    [
      { p1: 'move 1', p2: 'move 1' }, // p1 Disable (p2 has no lastMove yet — p1 faster) → fail? draw?
      { p1: 'move 2', p2: 'move 1' },
    ]);

  // (E) TAUNT + DISABLE stacked → forced Struggle. p2 has 1 attacking + 1 status; Taunt kills
  //     the status, Disable kills the attack → Struggle. (Two-turn setup.)
  await run('TAUNT + DISABLE force Struggle (attack disabled, status taunted)',
    [mon('Gengar', ['taunt', 'disable', 'shadowball'], { evs: { spa: 252, spe: 252 } }),
     mon('Alakazam', ['disable', 'psychic'], { evs: { spa: 252, spe: 252 } })],
    [mon('Snorlax', ['bodyslam', 'toxic'], { evs: { hp: 252, atk: 252 }, nature: 'Brave', ivs: { ...IV31, spe: 0 } })], // slow, moves 2nd
    [
      { p1: 'move 1', p2: 'move 1' }, // p1 Taunt (p2 Body Slam sets lastMove); Toxic disabled
      { p1: 'move 2', p2: 'move 1' }, // p1 Disable bodyslam (p2's lastMove) → now bodyslam [X] AND toxic taunted
      { p1: 'move 3', p2: 'move 1' }, // p2 forced Struggle?
      { p1: 'move 3', p2: 'move 1' },
    ]);

  // (F) DISABLE clears on switch-out. p2 disabled, switches out, switches back → move usable.
  await run('DISABLE clears on switch-out',
    [mon('Slowbro', ['disable', 'surf'], { evs: { hp: 252, def: 252 }, nature: 'Relaxed', ivs: { ...IV31, spe: 0 } })],
    [mon('Zapdos', ['thunderbolt', 'roost'], { evs: { spa: 252, spe: 252 } }),
     mon('Skarmory', ['spikes', 'roost'], { evs: { hp: 252, def: 252 } })],
    [
      { p1: 'move 2', p2: 'move 1' }, // p2 Thunderbolt (lastMove); p1 Surf
      { p1: 'move 1', p2: 'move 1' }, // p1 Disable thunderbolt
      { p1: 'move 2', p2: 'switch 2' }, // p2 switch to Skarmory (disable clears)
      { p1: 'move 2', p2: 'switch 2' }, // p2 switch back to Zapdos → thunderbolt usable?
      { p1: 'move 2', p2: 'move 1' }, // p2 Thunderbolt allowed
    ]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
