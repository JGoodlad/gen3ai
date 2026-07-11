// probe_residual_tail_regression_rng.js — GROUND TRUTH for the 2026-07-10 A/B
// residual-tail regression pins (tests/regression_test.rs):
//   CN1 cloud_nine_switch_out_fires_the_weatherchange_shuffle   (`gen3_cloudnine_end_v1`)
//   CN2 cloud_nine_faint_fires_the_weatherchange_shuffle        (`gen3_cloudnine_end_v1`)
//   FZ3 frozen_flash_fire_holder_is_not_fire_immune             (`gen3_ff_frozen_no_absorb_v1`)
//   FN1 fainted_replacement_sort_clears_status_and_boosts       (`gen3_fnt_clears_status_v1`)
//   TC1 traced_status_immune_ability_cures_the_status_on_update (`gen3_statusimmune_onupdate_cure_v1`)
//
// THE BUGS (the auto_0709_0805 re-triage residue, all probe-settled 2026-07-10):
//  CN: the resolved gen3 Cloud Nine / Air Lock `onEnd` fires `eachEvent("WeatherChange")`
//      — at switchIn's alive-outgoing ability-End AND at faintMessages' pre-`fainted=true`
//      ability-End — drawing ONE tie-shuffle iff the actives tie on cached speed.
//  FZ3: `flashfire.onTryHit` returns early for a `frz` holder — a FROZEN FF mon is NOT
//      fire-immune (full move draws; the fire-move thaw cures it post-hit).
//  FN1: `checkFainted` sets `status="fnt"` and `faintMessages→clearVolatile` zeroes
//      boosts — so a fainted formerly-para'd/+spe corpse sorts at its PLAIN speed in the
//      replacement instaswitch tie.
//  TC1: the STATUS_IMMUNE members carry an `onUpdate` CURE — a slept mon that TRACES
//      Insomnia is cured at the first Update after the copy.
//
// Run:  node src/rust_sim/harness/probe_residual_tail_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, p1, p2, rawSeed, plan, pre) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[1,2,3,4]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2 })}`);
  for (let i = 0; i < 12; i++) await tick();
  const b = stream.battle;
  b.prng = new PRNG(rawSeed.slice());
  if (pre) pre(b);
  console.log(`\n=== ${label} (raw seed ${rawSeed.join(',')}) ===`);
  let i = 0, safety = 0;
  while (!b.ended && safety < 80 && i < plan.length) {
    safety++;
    const rs = b.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[i]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const u = b.sides[0].active[0];
    const t = b.sides[1].active[0];
    console.log(`  dec ${i - 1} ${JSON.stringify(entry)} req=${b.requestState} seedAfter=${b.prng.getSeed()}`);
    console.log(`    p1=${u.species.name} ${u.hp}/${u.maxhp} st=${u.status || '-'}  p2=${t.species.name} ${t.hp}/${t.maxhp} st=${t.status || '-'}`);
  }
  return { b, log };
}

// Flat sets, explicit gender/level, Hardy 85-EVs — packed strings the Rust pins reuse.
const S = (name, item, abil, moves, lvl = 100) =>
  `${name}||${item}|${abil}|${moves}|Hardy|85,85,85,85,85,85|M|||${lvl}|`;

async function main() {
  // CN1: a Golduck-L81 Cloud Nine MIRROR; p1 voluntarily SWITCHES OUT its Golduck —
  // the onEnd WeatherChange fires with both Golducks tied → ONE extra draw vs a
  // same-board control where p1's Golduck has Damp.
  const golduckCN = S('Golduck', 'Leftovers', 'CloudNine', 'surf,splash', 81);
  const golduckDamp = S('Golduck', 'Leftovers', 'Damp', 'surf,splash', 81);
  const snorlax = S('Snorlax', 'Leftovers', 'ThickFat', 'splash,bodyslam');
  await run('CN1_switchout (Cloud Nine leaves, mirror tie)',
    golduckCN + ']' + snorlax, golduckCN, [0, 0, 0, 11],
    [{ p1: 'switch 2', p2: 'move 2' }, { p1: 'move 1', p2: 'move 1' }]);
  await run('CN1_control (Damp leaves — no onEnd WeatherChange)',
    golduckDamp + ']' + snorlax, golduckCN, [0, 0, 0, 11],
    [{ p1: 'switch 2', p2: 'move 2' }, { p1: 'move 1', p2: 'move 1' }]);

  // CN2: the FAINT site — a Golduck-L81 MIRROR where p2's Cloud Nine Golduck sits at
  // 1 HP: p1's Surf KOs it while the actives TIE on cached speed; the faintMessages
  // ability-End fires WeatherChange (the dying mon still gathered) → the tie draw.
  const golduckCNweak = `Golduck||Leftovers|CloudNine|splash,surf|Hardy|85,85,85,85,85,85|M|||81|`;
  await run('CN2_faint (Cloud Nine KO under a mirror tie)',
    golduckCN, golduckCNweak + ']' + snorlax, [0, 0, 0, 5],
    [{ p1: 'move 1', p2: 'move 1' }, { p2: 'switch 2' }, { p1: 'move 2', p2: 'move 1' }],
    (b) => { b.sides[1].active[0].hp = 1; });

  // FZ3: a FROZEN Flash Fire Houndoom takes a Flamethrower — full draws, damage lands,
  // the fire-move thaw cures it, and its own move then runs with NO thaw roll.
  const mewtwo = S('Mewtwo', 'Leftovers', 'Pressure', 'flamethrower,splash', 66);
  const houndoomFF = S('Houndoom', 'Leftovers', 'FlashFire', 'crunch,splash', 79);
  await run('FZ3_frozen_ff (Flamethrower into a FROZEN FF Houndoom)',
    mewtwo, houndoomFF, [0, 0, 0, 17],
    [{ p1: 'move 1', p2: 'move 1' }],
    (b) => { b.sides[1].active[0].setStatus('frz'); });

  // FN1: a +6-spe PARALYZED p1 corpse vs a plain p2 corpse of EQUAL plain speed —
  // the mutual Explosion double faint; the replacement instaswitch sort must TIE
  // (fnt erases par, clearVolatile erases the +6) → the shuffle draw.
  const mukA = S('Muk', 'Leftovers', 'StickyHold', 'explosion,agility,splash', 84);
  const mukB = S('Muk', 'Leftovers', 'StickyHold', 'splash,explosion', 84);
  await run('FN1_double_faint (boosted+para corpse ties plain corpse)',
    mukA + ']' + snorlax, mukB + ']' + S('Blissey', 'Leftovers', 'NaturalCure', 'splash,icebeam'),
    [0, 0, 0, 23],
    [{ p1: 'move 2', p2: 'move 1' },        // Agility +2
     { p1: 'move 2', p2: 'move 1' },        // +4
     { p1: 'move 2', p2: 'move 1' },        // +6
     { p1: 'move 1', p2: 'move 2' },        // p1 explodes; p2 explodes — order by speed; mutual double faint
     { p1: 'switch 2', p2: 'switch 2' },    // double replacement — the tie shuffle + resumed QC
     { p1: 'move 1', p2: 'move 1' }],
    (b) => { b.sides[0].active[0].setStatus('par'); });

  // TC1: a SLEPT Trace Porygon2 re-enters vs an Insomnia Hypno — the traced Insomnia's
  // onUpdate cures the sleep at the first Update (draw-free).
  const porygon2 = `Porygon2||Leftovers|Trace|recover,splash|Hardy|85,85,85,85,85,85|N|||80|`;
  const hypno = S('Hypno', 'Leftovers', 'Insomnia', 'hypnosis,splash', 85);
  await run('TC1_trace_cure (slept Porygon2 traces Insomnia)',
    porygon2 + ']' + snorlax, hypno, [0, 0, 0, 29],
    [{ p1: 'switch 2', p2: 'move 2' },      // P2 out (asleep pre-set), Snorlax in
     { p1: 'switch 2', p2: 'move 2' },      // P2 back IN → traces Insomnia → cured at Update
     { p1: 'move 2', p2: 'move 2' }],
    (b) => { b.sides[0].active[0].setStatus('slp'); });
}
main();
