// probe_taunt_disable_regression_rng.js — GROUND-TRUTH seeds + exact per-decision state for
// the deterministic TAUNT + DISABLE regression pins in tests/regression_test.rs. Constructs
// the SAME gen3customgame scenarios the pins use (fixed seed + scripted choices) and prints
// the post-decision SEED + the taunt presence / disabled slot / HP / status the pins assert.
// Ground truth is COPIED VERBATIM from here into the pins.
//
// Pins covered:
//   - taunt_blocks_status_move_selection_for_the_sim_window_draw_free :
//       a landed Taunt (acc-100 draw only — the volatile duration is a draw-free FIXED 2)
//       makes the target's Status moves un-selectable for EXACTLY the sim's window (the
//       next selection boundary), the queued status move is cant'd at execution, and the
//       free-up boundary lets Thunder Wave run again (the paralysis it inflicts is the
//       free-up proof).
//   - disable_duration_stored_per_branch_matches_sim :
//       BOTH duration branches — the FASTER disabler (target still to move,
//       `willMove(target)` TRUE → stored = random(2,6)) and the SLOWER disabler (target
//       already moved → stored = random(2,6)+1) — with the exact per-boundary disabled-slot
//       timeline to the FREE-UP turn + the post-decision SEEDs.
//   - taunt_plus_disable_forces_struggle :
//       Disable takes the only damaging move, Taunt takes the Status moves → the target is
//       FORCED to Struggle (the sim's request offers only Struggle).
//   - taunt_and_disable_onbeforemove_priority_vs_paralysis :
//       the onBeforeMove PRIORITY ordering vs the para roll — taunt (priority 0) cants
//       AFTER paralysis (1): a taunted+paralyzed queued status move DRAWS the
//       randomChance(1,4) first; disable (priority 7) cants BEFORE it: NO para roll.
//       (In-engine Thunder Wave paralysis — no injection — so a Rust pin replays it.)
//
// Run:  node src/rust_sim/harness/probe_taunt_disable_regression_rng.js
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

function disabledSlotOf(a) {
  if (!a || !a.volatiles || !a.volatiles['disable']) return -1;
  const disId = a.volatiles['disable'].move;
  if (!disId) return -1;
  for (let k = 0; k < a.moveSlots.length; k++) if (a.moveSlots[k].id === disId) return k;
  return -1;
}

async function run(label, seed, p1team, p2team, plan, opts = {}) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;

  if (!opts.quiet) console.log(`\n=== ${label} ===  seed=${JSON.stringify(seed)} initSeed=${battle.prng.getSeed()}`);
  const rows = [];
  let i = 0, safety = 0;
  while (!battle.ended && i < plan.length && safety < 60) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const logLen0 = log.length;
    const entry = plan[i]; i++;
    try { if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`); } catch (e) {}
    try { if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`); } catch (e) {}
    for (let k = 0; k < 18; k++) await tick();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const row = {
      dec: i - 1, choice: entry, after: battle.prng.getSeed(),
      p1: a0 ? `${a0.species.name} ${a0.hp}/${a0.maxhp}${a0.status ? ' ' + a0.status : ''} taunt=${a1 && a0.volatiles.taunt ? 1 : (a0.volatiles.taunt ? 1 : 0)} dis=${disabledSlotOf(a0)}` : '-',
      p2: a1 ? `${a1.species.name} ${a1.hp}/${a1.maxhp}${a1.status ? ' ' + a1.status : ''} taunt=${a1.volatiles.taunt ? 1 : 0} dis=${disabledSlotOf(a1)}` : '-',
    };
    rows.push(row);
    if (!opts.quiet) {
      console.log(`  [dec ${row.dec}] ${JSON.stringify(entry)} after=${row.after}`);
      console.log(`        p1: ${row.p1} | p2: ${row.p2}`);
      for (const l of log.slice(logLen0)) {
        if (/\|(move|cant|-start|-end|-damage|-status|-fail|-miss)\|/.test(l)) console.log(`        ${l}`);
      }
      const req = battle.sides[1].activeRequest;
      if (req && req.active && req.active[0] && req.active[0].moves) {
        console.log(`        p2 request: ${req.active[0].moves.map((mv) => `${mv.id}${mv.disabled ? '(DISABLED)' : ''}`).join(' ')}`);
      }
    }
  }
  const result = { battle, rows, log };
  if (!opts.quiet) console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
  return result;
}

async function main() {
  // ---------------------------------------------------------------------------
  // PIN taunt_blocks_status_move_selection_for_the_sim_window_draw_free
  // Fast Aerodactyl (taunt, earthquake) vs slow Blissey (thunderwave, icebeam).
  //   dec0: taunt lands (acc-100 draw only) + Blissey's QUEUED Thunder Wave is cant'd.
  //   dec1: the RESTRICTED window — twave un-selectable (Blissey must Ice Beam); the taunt
  //         expires at this decision's residual (2 → 1 at dec0's, 1 → 0 at dec1's).
  //   dec2: the FREE-UP — Thunder Wave is selectable again and RUNS: Aerodactyl is par'd
  //         (the free-up proof).
  // ---------------------------------------------------------------------------
  await run('PIN taunt window (Aerodactyl taunt vs Blissey)', [1, 2, 3, 4],
    [mon('Aerodactyl', ['taunt', 'earthquake'], { evs: { atk: 252, spe: 252 } })],
    [mon('Blissey', ['thunderwave', 'icebeam'], { evs: { hp: 252, def: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 2' }, { p1: 'move 2', p2: 'move 1' }]);

  // ---------------------------------------------------------------------------
  // PIN disable_duration_stored_per_branch_matches_sim — branch A: FASTER disabler.
  // Suicune (disable, surf) FAST vs Snorlax (earthquake, bodyslam) SLOW.
  //   dec0: both attack → Snorlax lastMove = earthquake (slot 0).
  //   dec1: Suicune disables FIRST (Snorlax still to move → willMove TRUE → stored =
  //         random(2,6) = the ROLLED value) + Snorlax's queued EQ is cant'd.
  //   dec2+: Snorlax on bodyslam; the disabled slot ticks down at each residual and FREES
  //         at the boundary where stored - (elapsed residuals) == 0.
  // ---------------------------------------------------------------------------
  const rolledOf = (r) => {
    const l = r.log.find((x) => /\|-start\|p2a: Snorlax\|Disable\|/.test(x));
    return l ? 'landed' : 'no-land';
  };
  const fast = await run('PIN disable duration — FASTER disabler (Suicune disable, willMove TRUE)', [3, 4, 5, 6],
    [mon('Suicune', ['disable', 'surf'], { evs: { hp: 252, def: 252 } })],
    [mon('Snorlax', ['earthquake', 'bodyslam'], { ability: 'Immunity', evs: { hp: 252, atk: 252 } })],
    [
      { p1: 'move 2', p2: 'move 1' },  // dec0: surf / EQ (lastMove=EQ)
      { p1: 'move 1', p2: 'move 1' },  // dec1: DISABLE (lands) / queued EQ -> cant
      { p1: 'move 2', p2: 'move 2' },  // dec2: surf / bodyslam
      { p1: 'move 2', p2: 'move 2' },  // dec3
      { p1: 'move 2', p2: 'move 2' },  // dec4
      { p1: 'move 2', p2: 'move 2' },  // dec5
      { p1: 'move 2', p2: 'move 2' },  // dec6
    ]);
  console.log(`  (disable ${rolledOf(fast)})`);

  // Branch B: SLOWER disabler. Snorlax (disable, bodyslam) SLOW vs Suicune (surf, icebeam)
  // FAST. dec1: Suicune surfs FIRST, then Snorlax disables (target ALREADY moved →
  // willMove FALSE → stored = random(2,6) + 1).
  // Seed picked so the acc-55 Disable LANDS at dec1 (a small deterministic sweep).
  let slowSeed = null;
  for (let s = 1; s <= 40 && !slowSeed; s++) {
    const seed = [s, s + 7, s + 13, s + 21];
    const r = await run('sweep', seed,
      [mon('Snorlax', ['disable', 'bodyslam'], { ability: 'Immunity', evs: { hp: 252, atk: 252 } })],
      [mon('Suicune', ['surf', 'icebeam'], { evs: { hp: 252, def: 252 } })],
      [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }], { quiet: true });
    const landed = r.log.some((x) => /\|-start\|p2a: Suicune\|Disable\|Surf/.test(x));
    if (landed) slowSeed = seed;
  }
  console.log(`\n  slower-branch seed sweep picked: ${JSON.stringify(slowSeed)}`);
  await run('PIN disable duration — SLOWER disabler (Snorlax disable, willMove FALSE → +1)', slowSeed,
    [mon('Snorlax', ['disable', 'bodyslam'], { ability: 'Immunity', evs: { hp: 252, atk: 252 } })],
    [mon('Suicune', ['surf', 'icebeam'], { evs: { hp: 252, def: 252 } })],
    [
      { p1: 'move 2', p2: 'move 1' },  // dec0: bodyslam / surf (lastMove=surf, slot 0)
      { p1: 'move 1', p2: 'move 1' },  // dec1: surf runs FIRST, then DISABLE lands (+1)
      { p1: 'move 2', p2: 'move 2' },  // dec2: bodyslam / icebeam
      { p1: 'move 2', p2: 'move 2' },  // dec3
      { p1: 'move 2', p2: 'move 2' },  // dec4
      { p1: 'move 2', p2: 'move 2' },  // dec5
      { p1: 'move 2', p2: 'move 2' },  // dec6
      { p1: 'move 2', p2: 'move 2' },  // dec7
    ]);

  // ---------------------------------------------------------------------------
  // PIN taunt_plus_disable_forces_struggle
  // Gengar (shadowball, disable, taunt) FAST vs Blissey (softboiled, toxic, icebeam) SLOW.
  //   dec0: shadowball / icebeam → Blissey lastMove = icebeam (slot 2).
  //   dec1: DISABLE lands → icebeam disabled; Blissey's queued icebeam cant'd.
  //   dec2: TAUNT lands → the Status moves un-selectable too; Blissey's queued softboiled
  //         cant'd. Now softboiled+toxic (taunted) + icebeam (disabled) = NOTHING usable.
  //   dec3: Blissey is FORCED to Struggle (the request offers ONLY struggle; `move 1`).
  // Seed picked so dec1's acc-55 Disable LANDS and dec3's request is struggle-only.
  // ---------------------------------------------------------------------------
  let strSeed = null;
  for (let s = 1; s <= 60 && !strSeed; s++) {
    const seed = [s * 3, s + 11, s * 5 + 1, s + 29];
    const r = await run('sweep', seed,
      [mon('Gengar', ['shadowball', 'disable', 'taunt'], { evs: { spa: 252, spe: 252 } })],
      [mon('Blissey', ['softboiled', 'toxic', 'icebeam'], { evs: { hp: 252, def: 252 } })],
      [
        { p1: 'move 1', p2: 'move 3' },
        { p1: 'move 2', p2: 'move 3' },
        { p1: 'move 3', p2: 'move 1' },
        { p1: 'move 1', p2: 'move 1' },
      ], { quiet: true });
    const landed = r.log.some((x) => /\|-start\|p2a: Blissey\|Disable\|Ice Beam/.test(x));
    const struggled = r.log.some((x) => /\|move\|p2a: Blissey\|Struggle\|/.test(x));
    if (landed && struggled) strSeed = seed;
  }
  // ---------------------------------------------------------------------------
  // PIN taunt_and_disable_onbeforemove_priority_vs_paralysis
  // (a) TAUNT side: Aerodactyl (thunderwave, taunt, earthquake) FAST vs Blissey
  //     (thunderwave, icebeam) SLOW. dec0: Aero paralyzes Blissey IN-ENGINE. dec1: Aero
  //     taunts while the PARALYZED Blissey QUEUED Thunder Wave — the sim draws the para
  //     randomChance(1,4) BEFORE the taunt cant (taunt onBeforeMove priority 0 < par 1).
  //     Need a seed where dec1's para roll does NOT full-para (so the cant line shows).
  // (b) DISABLE side: Suicune (thunderwave, disable, surf) FAST vs Snorlax (earthquake,
  //     bodyslam) SLOW. dec0: Suicune paralyzes Snorlax. dec1: both attack (Snorlax must
  //     NOT be full-para'd so lastMove = EQ). dec2: Suicune disables (must LAND) while the
  //     PARALYZED Snorlax QUEUED EQ — the cant fires BEFORE the para roll (disable priority
  //     7 > par 1): the turn draws disable acc + random(2,6) + Quick Claw, NO para roll.
  // ---------------------------------------------------------------------------
  let tpSeed = null;
  for (let s = 1; s <= 60 && !tpSeed; s++) {
    const seed = [s * 7 + 1, s + 17, s * 3 + 5, s + 41];
    const r = await run('sweep', seed,
      [mon('Aerodactyl', ['thunderwave', 'taunt', 'earthquake'], { evs: { atk: 252, spe: 252 } })],
      [mon('Blissey', ['thunderwave', 'icebeam'], { evs: { hp: 252, def: 252 } })],
      [{ p1: 'move 1', p2: 'move 2' }, { p1: 'move 2', p2: 'move 1' }], { quiet: true });
    const par = r.log.some((x) => /\|-status\|p2a: Blissey\|par/.test(x));
    const cant = r.log.some((x) => /\|cant\|p2a: Blissey\|move: Taunt\|Thunder Wave/.test(x));
    if (par && cant) tpSeed = seed;
  }
  console.log(`\n  taunt-para seed sweep picked: ${JSON.stringify(tpSeed)}`);
  await run('PIN OBM priority (a): taunted+paralyzed draws the para roll BEFORE the taunt cant', tpSeed,
    [mon('Aerodactyl', ['thunderwave', 'taunt', 'earthquake'], { evs: { atk: 252, spe: 252 } })],
    [mon('Blissey', ['thunderwave', 'icebeam'], { evs: { hp: 252, def: 252 } })],
    [
      { p1: 'move 1', p2: 'move 2' },  // dec0: thunderwave (par) / icebeam
      { p1: 'move 2', p2: 'move 1' },  // dec1: TAUNT / queued twave: para roll THEN cant
    ]);

  let dpSeed = null;
  for (let s = 1; s <= 80 && !dpSeed; s++) {
    const seed = [s + 3, s * 5 + 2, s + 23, s * 2 + 9];
    const r = await run('sweep', seed,
      [mon('Suicune', ['thunderwave', 'disable', 'surf'], { evs: { hp: 252, def: 252 } })],
      [mon('Snorlax', ['earthquake', 'bodyslam'], { ability: 'Immunity', evs: { hp: 252, atk: 252 } })],
      [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 3', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }], { quiet: true });
    const par = r.log.some((x) => /\|-status\|p2a: Snorlax\|par/.test(x));
    const eq = r.log.some((x) => /\|move\|p2a: Snorlax\|Earthquake\|/.test(x));
    const cant = r.log.some((x) => /\|cant\|p2a: Snorlax\|Disable\|Earthquake/.test(x));
    if (par && eq && cant) dpSeed = seed;
  }
  console.log(`\n  disable-para seed sweep picked: ${JSON.stringify(dpSeed)}`);
  await run('PIN OBM priority (b): paralyzed+disabled is cant\'d with NO para roll', dpSeed,
    [mon('Suicune', ['thunderwave', 'disable', 'surf'], { evs: { hp: 252, def: 252 } })],
    [mon('Snorlax', ['earthquake', 'bodyslam'], { ability: 'Immunity', evs: { hp: 252, atk: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' },  // dec0: thunderwave (par) / EQ (para roll passes)
      { p1: 'move 3', p2: 'move 1' },  // dec1: surf / EQ (lastMove=EQ; para roll passes)
      { p1: 'move 2', p2: 'move 1' },  // dec2: DISABLE lands / queued EQ: cant, NO para roll
    ]);

  console.log(`\n  struggle seed sweep picked: ${JSON.stringify(strSeed)}`);
  await run('PIN taunt+disable forces Struggle (Gengar vs Blissey)', strSeed,
    [mon('Gengar', ['shadowball', 'disable', 'taunt'], { evs: { spa: 252, spe: 252 } })],
    [mon('Blissey', ['softboiled', 'toxic', 'icebeam'], { evs: { hp: 252, def: 252 } })],
    [
      { p1: 'move 1', p2: 'move 3' },  // dec0: shadowball / icebeam (lastMove=icebeam slot 2)
      { p1: 'move 2', p2: 'move 3' },  // dec1: DISABLE icebeam; queued icebeam cant'd
      { p1: 'move 3', p2: 'move 1' },  // dec2: TAUNT; queued softboiled cant'd
      { p1: 'move 1', p2: 'move 1' },  // dec3: Blissey FORCED Struggle
    ]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
