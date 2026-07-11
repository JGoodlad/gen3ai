// probe_taunt_disable_onbeforemove_rng.js — the EXECUTION-TIME block (onBeforeMove) for
// TAUNT + DISABLE, against the OMNISCIENT in-process BattleStream (no server).
//
// The selection restriction (`onDisableMove` → the request's `disabled` flags) can't stop a
// move that was ALREADY QUEUED the same turn the volatile landed (the target's choice was
// committed before the FASTER taunter/disabler moved). The sim then blocks it at EXECUTION:
// `runMove` → `runEvent('BeforeMove')` → the volatile's `onBeforeMove` `cant`s the move.
//
// SETTLES (the sim is the source of truth — the port's `on_before_move(is_status, ...)` must
// match):
//   1. TAUNT-OBM: taunter FASTER + the target QUEUED a Status move the same turn → is the
//      queued move blocked at execution? What protocol line (`|cant|<mon>|move: Taunt|<Move>`)?
//      How many draws does the blocked action consume (expected ZERO — no accuracy/crit/dmg)?
//      Is PP deducted (expected NO — deductPP runs only after BeforeMove passes)?
//   2. TAUNT-OBM-PAR: the taunted mon is ALSO paralyzed. gen3 taunt deletes
//      `onBeforeMovePriority` (gen4's 5 → undefined → 0), so taunt sorts AFTER paralysis
//      (priority 1): the para roll (`randomChance(1,4)`) DRAWS FIRST; only a not-fully-para'd
//      mon reaches the taunt cant. Expected: +1 draw vs scenario 1, same `cant Taunt` line
//      (or `cant par` on the 25%).
//   3. DISABLE-OBM: disabler FASTER + the target QUEUED the very move being disabled → the
//      queued move is blocked (`|cant|<mon>|Disable|<Move>`), draw-free, no PP.
//   4. DISABLE-OBM-PAR: the disabled mon is ALSO paralyzed. Disable's `onBeforeMovePriority: 7`
//      (base, inherited through gen4→gen3) sorts BEFORE confusion (3) + paralysis (1): the
//      disabled move is cant'd BEFORE the para roll → ZERO draws for the blocked action
//      (the OPPOSITE of taunt).
//   5. TAUNT-FIXED-DAMAGE: a taunted mon's QUEUED Seismic Toss (Showdown category Physical,
//      bp 0) is NOT blocked (taunt blocks `move.category === 'Status'` only) and stays
//      SELECTABLE in the request.
//   6. TAUNT-RESIDUAL-ORDER: gen3 taunt inherits gen4's `onResidualOrder: 10,
//      onResidualSubOrder: 15` (NOT the base's order 15 — gen4 shadows it). A FAST taunted mon
//      + a SLOW burned foe expiring/ticking the same residual: at order 10 the tie breaks on
//      SPEED before subOrder, so the fast mon's taunt `-end` precedes the slow foe's brn
//      `-damage` (base order-15 would put taunt strictly AFTER every order-10 handler).
//
// ==================================================================================
// CONFIRMED FINDINGS (run 2026-07-01 vs the omniscient sim; verbatim from the output):
//   1. TAUNT-OBM (seed [1,2,3,4]): the queued Thunder Wave is blocked at execution —
//      `|cant|p2a: Blissey|move: Taunt|Thunder Wave` — dec-0 draws=2 (taunt accuracy + Quick
//      Claw ONLY → the blocked action drew ZERO) and NO PP deducted (twave stays 32). The next
//      request shows `thunderwave(DISABLED)`; the request AFTER the expiry shows it usable
//      (the exact 2-turn window, `-end ... [silent]` at the 2nd residual).
//   2. TAUNT-OBM-PAR (same seed + injected par): dec-0 draws=3 — EXACTLY +1 vs scenario 1
//      (the para `randomChance(1,4)` fires BEFORE the taunt cant), then the SAME
//      `cant ... move: Taunt` line. Taunt sorts at priority 0, AFTER paralysis (1).
//   3. DISABLE-OBM (seed [3,4,5,6]): the queued Earthquake is blocked —
//      `|-start|p2a: Snorlax|Disable|Earthquake` then `|cant|p2a: Snorlax|Disable|Earthquake`
//      — dec-1 draws=3 (disable accuracy + its random(2,6) + Quick Claw ONLY → the blocked
//      action drew ZERO) and NO PP deducted (earthquake stays 15 from turn 1).
//   4. DISABLE-OBM-PAR (same seed + injected par): dec-1 draws=3 — IDENTICAL to scenario 3
//      (NO para roll: disable's priority 7 cants the move BEFORE paralysis 1 — the OPPOSITE
//      of taunt). Same `-start` + `cant ... Disable` lines.
//   5. TAUNT-FIXED-DAMAGE: Seismic Toss EXECUTES under taunt (deals level=100 damage,
//      `|move|p2a: Blissey|Seismic Toss|...` — no cant) and the request does NOT disable it
//      (`seismictoss thunderwave(DISABLED)`) — taunt keys on the Showdown category (Physical).
//   6. TAUNT-RESIDUAL-ORDER: at the SAME residual the FAST mon's
//      `|-end|p2a: Aerodactyl|move: Taunt|[silent]` precedes the SLOW foe's
//      `|-damage|p1a: Snorlax|285/524 brn|[from] brn` — order 10/subOrder 15 with SPEED
//      breaking before subOrder CONFIRMED (the gen4-inherited values; the base's
//      `onResidualOrder: 15` is SHADOWED by gen4 and would have reversed the two lines).
// ==================================================================================
//
// Run:  node src/rust_sim/harness/probe_taunt_disable_onbeforemove_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

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

async function run(label, seed, p1team, p2team, plan, inject) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  for (const inj of (inject || [])) {
    const m = battle.sides[inj.side].active[0];
    if (!m) continue;
    if (inj.status) m.setStatus(inj.status, m, null, true);
    if (inj.volatile) m.addVolatile(inj.volatile);
  }
  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  console.log(`\n=== ${label} ===  seed=${JSON.stringify(seed)} initSeed=${battle.prng.getSeed()}`);
  const ppOf = (m) => m ? m.moveSlots.map((s) => `${s.id}:${s.pp}`).join(',') : '-';
  let i = 0, safety = 0;
  while (!battle.ended && safety < 40) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const dc0 = drawCount, logLen0 = log.length;
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    try { if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`); } catch (e) {}
    try { if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`); } catch (e) {}
    for (let k = 0; k < 18; k++) await tick();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    console.log(`  [dec ${i - 1}] ${JSON.stringify(entry)} draws=${drawCount - dc0} after=${battle.prng.getSeed()}`);
    console.log(`        p1 pp=[${ppOf(a0)}] | p2 pp=[${ppOf(a1)}]`);
    for (const l of log.slice(logLen0)) {
      if (/\|(move|cant|-start|-end|-damage|-status|-fail|-miss|switch|turn|-heal)\|/.test(l) || /\|turn\|/.test(l)) {
        console.log(`        ${l}`);
      }
    }
    // Request `disabled` flags for p2 (the restriction view).
    const req = battle.sides[1].activeRequest;
    if (req && req.active && req.active[0] && req.active[0].moves) {
      console.log(`        p2 request moves: ${req.active[0].moves.map((mv) => `${mv.id}${mv.disabled ? '(DISABLED)' : ''}`).join(' ')}`);
    }
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  const SEED = [1, 2, 3, 4];

  // 1. TAUNT-OBM: fast Aerodactyl taunts; slow Blissey QUEUED Thunder Wave (status) the same
  //    turn. Expect the queued TWave blocked at execution: `|cant|p2a: Blissey|move: Taunt|
  //    Thunder Wave`, ZERO draws for the blocked action, twave PP unchanged.
  await run('1 TAUNT-OBM: queued status move blocked at execution (taunter faster)', SEED,
    [mon('Aerodactyl', ['taunt', 'earthquake'], { evs: { atk: 252, spe: 252 } })],
    [mon('Blissey', ['thunderwave', 'icebeam'], { evs: { hp: 252, def: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 2', stop: true }]);

  // 2. TAUNT-OBM-PAR: same, Blissey pre-paralyzed. Expect +1 draw (the para randomChance(1,4)
  //    fires BEFORE the taunt cant — taunt priority 0 < par 1).
  await run('2 TAUNT-OBM-PAR: paralyzed + taunted → para roll BEFORE the taunt cant', SEED,
    [mon('Aerodactyl', ['taunt', 'earthquake'], { evs: { atk: 252, spe: 252 } })],
    [mon('Blissey', ['thunderwave', 'icebeam'], { evs: { hp: 252, def: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 2', stop: true }],
    [{ side: 1, status: 'par' }]);

  // 3. DISABLE-OBM: T1 both attack (Snorlax Earthquake sets lastMove into the bulky Suicune,
  //    which survives). T2 the faster Suicune disables EQ (seed picked so acc-55 LANDS) while
  //    Snorlax QUEUED EQ. Expect `|cant|p2a: Snorlax|Disable|Earthquake`, the blocked action
  //    draw-free, earthquake PP unchanged from T1.
  const DSEED = [3, 4, 5, 6];
  await run('3 DISABLE-OBM: queued disabled move blocked at execution (disabler faster)', DSEED,
    [mon('Suicune', ['disable', 'surf'], { evs: { hp: 252, def: 252 } })],
    [mon('Snorlax', ['earthquake', 'bodyslam'], { ability: 'Immunity', evs: { hp: 252, atk: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1', stop: true }]);

  // 4. DISABLE-OBM-PAR: same, Snorlax pre-paralyzed. Expect the blocked action STILL draw-free
  //    (disable priority 7 cants BEFORE the para roll at 1) — the OPPOSITE of taunt: the cant
  //    turn's draw count is IDENTICAL to scenario 3's (same seed, +0 for the para).
  await run('4 DISABLE-OBM-PAR: paralyzed + disabled → cant BEFORE the para roll (no para draw)', DSEED,
    [mon('Suicune', ['disable', 'surf'], { evs: { hp: 252, def: 252 } })],
    [mon('Snorlax', ['earthquake', 'bodyslam'], { ability: 'Immunity', evs: { hp: 252, atk: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1', stop: true }],
    [{ side: 1, status: 'par' }]);

  // 5. TAUNT-FIXED-DAMAGE: taunted Blissey's QUEUED Seismic Toss (Physical, bp 0) is NOT
  //    blocked — it executes (level=100 damage) and stays selectable in the next request.
  await run('5 TAUNT-FIXED-DAMAGE: Seismic Toss executes under taunt (not a Status move)', SEED,
    [mon('Aerodactyl', ['taunt', 'earthquake'], { evs: { atk: 252, spe: 252 } })],
    [mon('Blissey', ['seismictoss', 'thunderwave'], { evs: { hp: 252, def: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1', stop: true }]);

  // 6. TAUNT-RESIDUAL-ORDER: FAST taunted Aerodactyl (volatile injected, duration 2) + SLOW
  //    burned Snorlax. Residual 2 (the taunt expiry) must show `-end ... Taunt|[silent]`
  //    BEFORE `-damage ... [from] brn` (order 10/subOrder 15, speed breaks before subOrder;
  //    base order-15 would reverse them).
  await run('6 TAUNT-RESIDUAL-ORDER: fast taunt -end before slow brn -damage at the same residual', SEED,
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [mon('Aerodactyl', ['splash', 'earthquake'], { evs: { spe: 252 } })],
    // Turn 1: p2's queued Splash is taunt-cant'd (splash is DISABLED in the next request, so
    // turn 2 must pick earthquake). Turn 2's residual is the taunt expiry (2→1→0).
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2', stop: true }],
    [{ side: 0, status: 'brn' }, { side: 1, volatile: 'taunt' }]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
