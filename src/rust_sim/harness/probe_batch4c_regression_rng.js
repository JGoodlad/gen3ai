// probe_batch4c_regression_rng.js — REAL-Showdown ground-truth seeds/state for the
// `gen3_move_coverage_batch4c_v1` regression pins (MC49+): Hyper Beam's mustrecharge,
// Solar Beam's two-turn charge, Doom Desire / Future Sight's future strike.
//
// Each scenario mirrors its `tests/regression_test.rs` pin EXACTLY (teams, seed, choices,
// injections); the printed per-boundary `seedAfter` + state are copied verbatim into the
// pin's constants. Re-run after any PRNG/draw-order change.
//
// Run:  node src/rust_sim/harness/probe_batch4c_regression_rng.js
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
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: opts.gender || 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

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
    }
  };
  applyActs(inject && inject.acts);

  console.log(`\n=== ${label} ===  seed=${JSON.stringify(seed)} initSeed=${battle.prng.getSeed()}`);
  const ppStr = (m) => m ? m.moveSlots.map((s) => `${s.id}:${s.pp}/${s.maxpp}`).join(',') : '-';
  let i = 0, safety = 0;
  while (!battle.ended && safety < 20) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    if (i >= plan.length) break;
    const entry = plan[i]; i++;
    applyActs(entry.pre);
    const before = battle.prng.getSeed();
    const l0 = log.length;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'}` : '-';
    console.log(`  dec${i - 1} [${rs}] ${JSON.stringify({ p1: entry.p1, p2: entry.p2 })} seed ${before} -> ${battle.prng.getSeed()}`);
    console.log(`      p1=${fmt(a0)} pp={${ppStr(a0)}}  p2=${fmt(a1)}`);
    const key = log.slice(l0).filter((l) => /-mustrecharge|-prepare|-anim|cant|-start\||-end\||-miss|faint|-heal/.test(l));
    for (const l of key) console.log(`      LINE ${l}`);
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  const hbP1 = () => [mon('Snorlax', ['hyperbeam', 'splash']), mon('Blissey', ['softboiled'])];
  const hbP2 = () => [mon('Skarmory', ['spikes', 'protect', 'splash', 'roar'], { evs: { hp: 252 } }),
                      mon('Forretress', ['spikes', 'splash'], { evs: { hp: 252 } })];

  // MC49 — HB hit -> recharge (draws only endTurn) -> free HB again.
  await run('MC49 HB hit->recharge->free', hbP1(), hbP2(),
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 3' }],
    { seed: [9, 9, 9, 9] });

  // MC50 — HB natural MISS (no recharge; PP consumed; user acts freely next turn).
  await run('MC50 HB miss no-recharge', hbP1(), hbP2(),
    [{ p1: 'move 1', p2: 'move 3' },
     { p1: 'move 1', p2: 'move 3' }]);

  // MC51 — HB KO: -mustrecharge before faint; lock persists across p2's force-switch.
  await run('MC51 HB KO lock persists', hbP1(), hbP2(),
    [{ p1: 'move 1', p2: 'move 1' },
     { p2: 'switch 2' },
     { p1: 'move 1', p2: 'move 2' }],
    { seed: [9, 9, 9, 9], acts: [{ side: 1, slot: 0, hp: 20 }] });

  // MC52 — par'd user on the locked turn: NO para roll (seed advance identical to MC49 dec1).
  await run('MC52 HB recharge with par (no para draw)', hbP1(), hbP2(),
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1', pre: [{ side: 0, slot: 0, status: 'par' }] },
     { p1: 'move 2', p2: 'move 3' }],
    { seed: [9, 9, 9, 9] });

  // MC53 — SB charge (0 move draws) -> fire (3 draws, KOs Swampert) -> replacement (1 draw)
  //        -> fresh charge vs Blissey (1 draw incl. the failed full-HP SoftBoiled).
  await run('MC53 SB charge->fire->fresh charge',
    [mon('Venusaur', ['solarbeam', 'razorleaf']), mon('Snorlax', ['bodyslam'])],
    [mon('Swampert', ['surf'], { evs: { hp: 252 } }), mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' },
     { p2: 'switch 2' },
     { p1: 'move 1', p2: 'move 1' }]);

  // MC54 — SB SUN SKIP (7 draws whole turn: EQ 3 + SB 3 + QC).
  await run('MC54 SB sun skip',
    [mon('Venusaur', ['solarbeam'])],
    [mon('Groudon', ['earthquake'], { ability: 'Drought', evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);

  // MC55 — SB in RAIN (halved BP, state-only) + the no-weather control (same seeds).
  await run('MC55 SB rain halved',
    [mon('Venusaur', ['solarbeam'])],
    [mon('Kyogre', ['calmmind'], { ability: 'Drizzle', evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' }]);
  await run('MC55 control (Shell Armor Kyogre, no weather)',
    [mon('Venusaur', ['solarbeam'])],
    [mon('Kyogre', ['calmmind'], { ability: 'Shell Armor', evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' }]);

  // MC56 — full-para on the CHARGE turn (no charge, PP untouched) at seed [5,6,7,8].
  await run('MC56 SB full-para on the charge turn',
    [mon('Venusaur', ['solarbeam'])],
    [mon('Swampert', ['curse'], { evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    { seed: [5, 6, 7, 8], acts: [{ side: 0, slot: 0, status: 'par' }] });

  // MC57 — DD cast (2 draws) -> idle (1) -> resolve (2, damage 366) + FS variant (45).
  await run('MC57 DD cast/idle/resolve',
    [mon('Jirachi', ['doomdesire', 'splash'])],
    [mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' }]);
  await run('MC57b FS cast/idle/resolve',
    [mon('Jirachi', ['futuresight', 'splash'])],
    [mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' }]);

  // MC58 — DD double-cast on the idle turn: 1 draw (QC only), PP 7->6, the pending strike
  //        resolves on schedule.
  await run('MC58 DD double-cast fail',
    [mon('Jirachi', ['doomdesire', 'splash'])],
    [mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' }]);

  // MC59 — DD resolve KO (Gengar hp 60): the resolve turn draws acc ONLY (no QC — the
  //        deferred faint), the QC lands at the forced-replacement boundary.
  await run('MC59 DD resolve KO defers the Quick Claw',
    [mon('Jirachi', ['doomdesire', 'splash'])],
    [mon('Gengar', ['splash'], { ability: 'Levitate', evs: { hp: 252 } }),
     mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p2: 'switch 2' },
     { p1: 'move 2', p2: 'move 1' }],
    { acts: [{ side: 1, slot: 0, hp: 60 }] });

  // MC60 — the residual ORDER composition: Wish(7) -> sand(8) -> Leftovers(10.4) -> DD(11).
  //        Celebi (Leftovers, hp 150) vs Sand Stream Tyranitar (Leftovers).
  await run('MC60 DD residual order (Wish/sand/Leftovers/DD)',
    [mon('Celebi', ['doomdesire', 'wish', 'splash'], { item: 'Leftovers' })],
    [mon('Tyranitar', ['splash'], { ability: 'Sand Stream', item: 'Leftovers', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 3', p2: 'move 1' }],
    { acts: [{ side: 0, slot: 0, hp: 150 }] });
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
