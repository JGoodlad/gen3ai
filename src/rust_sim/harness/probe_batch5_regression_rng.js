// probe_batch5_regression_rng.js — REAL-Showdown ground-truth seeds/state for the
// `gen3_move_coverage_batch5_v1` regression pins (MC61+): Counter / Mirror Coat /
// Endeavor (the reactive fixed-damage family), Return / Frustration / Flail / Reversal /
// Low Kick (the variable-BP family), and Sleep Talk.
//
// Each scenario mirrors its `tests/regression_test.rs` pin EXACTLY (teams, seed, choices,
// injections); the printed per-boundary `seedAfter` + state are copied verbatim into the
// pin's constants. Re-run after any PRNG/draw-order change.
//
// Run:  node src/rust_sim/harness/probe_batch5_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  const m = {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: opts.gender || 'N',
  };
  if (opts.happiness !== undefined) m.happiness = opts.happiness;
  return m;
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
      if (inj.pp) m.moveSlots[inj.pp.moveSlot].pp = inj.pp.val;
    }
  };
  applyActs(inject && inject.acts);

  console.log(`\nTEAM p1 ${Teams.pack(p1team)}`);
  console.log(`TEAM p2 ${Teams.pack(p2team)}`);
  console.log(`=== ${label} ===  seed=${JSON.stringify(seed)} initSeed=${battle.prng.getSeed()}`);
  const ppStr = (m) => m ? m.moveSlots.map((s) => `${s.id}:${s.pp}/${s.maxpp}`).join(',') : '-';
  let i = 0, safety = 0;
  while (!battle.ended && safety < 30) {
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
    const slp = (m) => (m && m.status === 'slp' && m.statusState) ? ` slp(t=${m.statusState.time},sk=${m.statusState.skippedTime | 0})` : '';
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'}${slp(m)}` : '-';
    console.log(`  dec${i - 1} [${rs}] ${JSON.stringify({ p1: entry.p1, p2: entry.p2 })} seed ${before} -> ${battle.prng.getSeed()}`);
    console.log(`      p1=${fmt(a0)} pp={${ppStr(a0)}}  p2=${fmt(a1)}`);
    const key = log.slice(l0).filter((l) => /move\||-damage|-fail|-immune|-miss|cant|faint|-end\b|-activate|-heal|-status/.test(l));
    for (const l of key) console.log(`      LINE ${l}`);
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // MC61 — Counter returns 2x a physical hit; the NEXT turn's counter (foe splashes)
  // fails ZERO-DRAW (the onStart reset — prev-turn damage never counts).
  await run('MC61 counter 2x + prev-turn reset zero-draw fail',
    [mon('Snorlax', ['counter', 'splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['drillpeck', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 2' }]);

  // MC62 — the WRONG-CATEGORY gates: counter vs a SPECIAL hit fails; mirror coat vs a
  // PHYSICAL hit fails (both zero-draw past the foe's own move).
  await run('MC62 wrong-category fails (counter vs Surf, MC vs Return)',
    [mon('Snorlax', ['counter', 'mirrorcoat'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['surf', 'return'], { evs: { hp: 252 }, happiness: 255 })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 2' }]);

  // MC63 — the RETURN-FIRE type immunity: an armed Counter into a GHOST + an armed
  // Mirror Coat into a DARK both draw accuracy then `-immune`.
  await run('MC63a counter into a Ghost (Shadow Ball arms; Fighting immune)',
    [mon('Machamp', ['counter', 'splash'], { evs: { hp: 252 } })],
    [mon('Gengar', ['shadowball', 'splash'], { ability: 'Levitate', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);
  await run('MC63b mirror coat into a Dark (Crunch arms; Psychic immune)',
    [mon('Blissey', ['mirrorcoat', 'splash'], { evs: { hp: 252 } })],
    [mon('Tyranitar', ['crunch', 'splash'], { ability: 'No Ability', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);

  // MC64 — a SUB-ABSORBED foe hit does NOT arm Counter (the mon's Damage event never
  // fires behind its own sub) → the counter fails zero-draw.
  await run('MC64 sub-absorbed hit not recorded',
    [mon('Snorlax', ['counter', 'substitute', 'splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['drillpeck', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 2' },   // sub up (Skarm splashes)
     { p1: 'move 1', p2: 'move 1' }]); // drill peck into the sub -> counter fails

  // MC65 — Seismic Toss (fixed damage, Fighting → Physical) IS countered (2x100);
  // Beat Up's strikes (Special) arm MIRROR COAT with 2x the LAST strike.
  await run('MC65a seismic toss countered',
    [mon('Snorlax', ['counter', 'splash'], { evs: { hp: 252 } })],
    [mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);
  await run('MC65b beat up arms mirror coat (2x the LAST strike)',
    [mon('Blissey', ['mirrorcoat', 'splash'], { evs: { hp: 252 } })],
    [mon('Smeargle', ['beatup', 'splash'], { evs: { hp: 252 } }),
     mon('Snorlax', ['splash'], { evs: { hp: 252 } }),
     mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);

  // MC66 — the COUNTER-MIRROR speed tie: a both-splash CONTROL turn then a both-counter
  // turn (the +4 draw delta: the order-5 pair tie + 2 trailing Updates + the residual
  // duration tie).
  await run('MC66 counter mirror tie (control then both-counter)',
    [mon('Snorlax', ['counter', 'splash'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['counter', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 2' },
     { p1: 'move 1', p2: 'move 1' }]);

  // MC67 — Endeavor: the delta (target -> the user's hp), then the EQUALITY fail
  // (zero-draw `-fail`).
  await run('MC67 endeavor delta then equality fail',
    [mon('Swellow', ['endeavor', 'splash'])],
    [mon('Snorlax', ['splash', 'drillpeck'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' }],
    { acts: [{ side: 0, hp: 50 }] });

  // MC68 — Endeavor into a GHOST (`-immune` after the accuracy draw) + into a
  // SUBSTITUTE (the delta reads the MON's hp; the sub breaks, no carry).
  await run('MC68a endeavor into a Ghost',
    [mon('Swellow', ['endeavor', 'splash'])],
    [mon('Gengar', ['splash'], { ability: 'Levitate', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    { acts: [{ side: 0, hp: 50 }] });
  await run('MC68b endeavor into a substitute (no carry)',
    [mon('Swellow', ['endeavor', 'splash'])],
    [mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 2' }],
    { acts: [{ side: 0, hp: 50 }] });

  // MC69 — RETURN happiness extremes: h255 (bp 102) vs h3 (bp 1) — the two runs must
  // end at the SAME seed (draw-neutrality) with different damage.
  await run('MC69a return h255',
    [mon('Tauros', ['return', 'splash'], { evs: { atk: 252 }, happiness: 255 })],
    [mon('Skarmory', ['splash', 'drillpeck'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);
  await run('MC69b return h3 (bp 1 via the ||1 clamp)',
    [mon('Tauros', ['return', 'splash'], { evs: { atk: 252 }, happiness: 3 })],
    [mon('Skarmory', ['splash', 'drillpeck'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);
  await run('MC69c frustration h0 (bp 102)',
    [mon('Tauros', ['frustration', 'splash'], { evs: { atk: 252 }, happiness: 0 })],
    [mon('Skarmory', ['splash', 'drillpeck'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);

  // MC70 — the FLAIL band boundary at Snorlax maxhp 524: hp 21 (ratio 1 → bp 200) vs
  // hp 22 (ratio 2 → bp 150) — seed-identical, damage-different.
  await run('MC70a flail hp=21 (bp 200)',
    [mon('Snorlax', ['flail', 'splash'], { evs: { hp: 252, atk: 252 } })],
    [mon('Skarmory', ['splash', 'drillpeck'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    { acts: [{ side: 0, hp: 21 }] });
  await run('MC70b flail hp=22 (bp 150)',
    [mon('Snorlax', ['flail', 'splash'], { evs: { hp: 252, atk: 252 } })],
    [mon('Skarmory', ['splash', 'drillpeck'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    { acts: [{ side: 0, hp: 22 }] });

  // MC71 — the LOW KICK weight ladder: Pichu (20 hg → bp 20) vs Wobbuffet (285 → 60)
  // vs Snorlax (4600 → 120) — same board otherwise, seed-identical, damage per rung.
  for (const [tag, sp] of [['a Pichu bp20', 'Pichu'], ['b Wobbuffet bp60', 'Wobbuffet'], ['c Snorlax bp120', 'Snorlax']]) {
    await run(`MC71${tag}`,
      [mon('Blissey', ['lowkick', 'splash'], { evs: { atk: 252 } })],
      [mon(sp, ['splash'], { evs: { hp: 252 } })],
      [{ p1: 'move 1', p2: 'move 1' }]);
  }

  // MC72 — SLEEP TALK: the n=1 sample (pool [rest] — Rest picked while asleep silently
  // no-ops) + the EMPTY pool ([sleeptalk, solarbeam] → [still]+fail). Rest-based sleep
  // (fixed 3) so the counter is deterministic.
  await run('MC72a sleep talk n=1 pool [rest] (sample drawn; called Rest no-ops)',
    [mon('Snorlax', ['sleeptalk', 'rest'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['drillpeck', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' },   // Rest (hurt by the pre-inject? no — drill peck hits first turn? Snorlax slower; rest at full FAILS)
     { p1: 'move 1', p2: 'move 2' }],  // asleep -> sleep talk: sample random(1) -> rest -> silent no-op
    { acts: [{ side: 0, hp: 300 }] });
  await run('MC72b sleep talk empty pool ([still]+fail)',
    [mon('Snorlax', ['sleeptalk', 'solarbeam'], { evs: { hp: 252 } })],
    [mon('Breloom', ['spore', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },   // Breloom (faster) Spores; the awake Sleep Talk pick never runs (cant slp? no — spore lands FIRST, then slp cant + proceed? time rolled)
     { p1: 'move 1', p2: 'move 2' }],  // asleep -> sleep talk: pool [] -> [still] + -fail, zero draws
    );

  // MC72c — the called Rest-while-asleep fires on a DAMAGED sleeper (Drill Peck lands
  // the same turn BEFORE the talk): the silent no-op must NOT heal / redraw / reset the
  // counter (the teeth for the run_rest asleep guard — a full-HP board masks it behind
  // the full-HP guard).
  await run('MC72c called Rest on a DAMAGED sleeper (silent no-op, no heal)',
    [mon('Snorlax', ['sleeptalk', 'rest'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['drillpeck', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' },   // Rest (hurt via inject) -> Sleep(3), full heal
     { p1: 'move 1', p2: 'move 1' }],  // Drill Peck damages the sleeper; the talk picks Rest -> NO heal
    { acts: [{ side: 0, hp: 300 }] });

  // MC73 — the CHOICE-LOCKED Sleep Talk: a CB sleeper's FIRST Sleep Talk (the lock
  // records Sleep Talk itself) samples + executes; the SECOND fails `[still]`+`-fail`
  // BEFORE the sample. Breloom's Spore supplies the sleep so the CB lock is fresh.
  await run('MC73 CB sleep talk works once then choicelock-fails',
    [mon('Snorlax', ['sleeptalk', 'bodyslam'], { item: 'Choice Band', evs: { hp: 252, atk: 252 } })],
    [mon('Breloom', ['spore', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },   // Breloom Spores first; Snorlax's Sleep Talk turn is cant'd? (slp lands then the queued sleeptalk proceeds via sleepUsable)
     { p1: 'move 1', p2: 'move 2' },   // sleep talk #1: locks to sleeptalk, samples [bodyslam] -> executes
     { p1: 'move 1', p2: 'move 2' }],  // sleep talk #2: choicelock -> [still]+fail, no sample
    );

  // MC74 — the 0-PP PICK: the pool is exactly [bodyslam] (pp 0) → the n=1 sample DRAWS
  // then `|cant|…|nopp|bodyslam` (the turn wasted). Spore-based sleep.
  await run('MC74 sleep talk 0-PP pick (nopp cant after the sample)',
    [mon('Snorlax', ['sleeptalk', 'bodyslam'], { evs: { hp: 252 } })],
    [mon('Breloom', ['spore', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },   // Breloom Spores
     { p1: 'move 1', p2: 'move 2' }],  // asleep -> pool [bodyslam(0pp)] -> sample then nopp cant
    { acts: [{ side: 0, pp: { moveSlot: 1, val: 0 } }] });

  // MC76 — a FIXED-DAMAGE hit sets the Focus-Punch user's lostFocus (the e2e_202
  // admission bug: the port's run_fixed_damage_move never set it, so the punch landed
  // where the sim cants it). Seismic Toss (priority 0) lands before the -3 punch.
  await run('MC76 seismic toss cancels a queued Focus Punch',
    [mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } })],
    [mon('Dragonite', ['focuspunch', 'splash'], { ability: 'No Ability', evs: { hp: 252, atk: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);

  // MC75 — skippedTime RESTORE: rest → talk ×2 (time 3 → 1, skipped 2) → switch out +
  // back in → the slp time reads 3 again.
  await run('MC75 skippedTime restore on switch',
    [mon('Snorlax', ['sleeptalk', 'rest'], { evs: { hp: 252 } }),
     mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['drillpeck', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' },   // Rest (time 3)
     { p1: 'move 1', p2: 'move 2' },   // talk (time 2, skipped 1)
     { p1: 'move 1', p2: 'move 2' },   // talk (time 1, skipped 2)
     { p1: 'switch 2', p2: 'move 2' }, // pivot out
     { p1: 'switch 2', p2: 'move 2' }, // pivot back (slp time restored to 3)
     { p1: 'move 1', p2: 'move 2' }],  // talk (time 2 again)
    { acts: [{ side: 0, hp: 300 }] });
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
