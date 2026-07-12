// probe_batch1_regression_rng.js — capture the REAL-Showdown ground-truth seedAfter +
// per-mon STATE for the MOVE-COVERAGE BATCH 1 regression pins
// (`gen3_move_coverage_batch1_v1`). Each scenario is a CONSTRUCTED gen3customgame board
// with an explicit seed + scripted choices; the printed `seedAfter` (+ HP / boosts / item /
// spikes / leech) are copied verbatim into tests/regression_test.rs as constants.
//
// Run:  node src/rust_sim/harness/probe_batch1_regression_rng.js
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

async function run(label, p1, p2, plan, seed, inject) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  for (const inj of (inject || [])) {
    const m = battle.sides[inj.side].active[0];
    if (inj.spikes) for (let k = 0; k < inj.spikes; k++) battle.sides[inj.side].addSideCondition('spikes', battle.sides[1 - inj.side].active[0]);
    if (inj.leechseed) m.addVolatile('leechseed', battle.sides[1 - inj.side].active[0]);
    if (inj.hp !== undefined) m.hp = inj.hp;
  }
  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 30) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const spk = (s) => { const sc = battle.sides[s].sideConditions; const sp = sc && sc['spikes']; return sp ? sp.layers : 0; };
    const ls = (m) => (m && m.volatiles && m.volatiles['leechseed']) ? 'LS' : '-';
    const fmt = (m, s) => m ? `${m.species.name} ${m.hp}/${m.maxhp}${m.fainted ? ' FNT' : ''} atk${m.boosts.atk} def${m.boosts.def} spa${m.boosts.spa} item=${m.item || '-'} spk${spk(s)} ${ls(m)}` : '-';
    console.log(`  [${rs}] ${JSON.stringify(entry)} seedAfter=${battle.prng.getSeed()}`);
    console.log(`        p1=${fmt(a0, 0)}\n        p2=${fmt(a1, 1)}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  const SEED = [11, 22, 33, 44];

  // MC1 — RECOIL: Double-Edge recoils floor(dmg/3) to the USER (Rock-Head-free Tauros).
  await run('MC1 Double-Edge recoil floor(dmg/3)',
    [mon('Tauros', ['doubleedge'], { ability: 'Sturdy', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    [mon('Snorlax', ['pound'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 252, def: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }], SEED);

  // MC1b — RECOIL negated by ROCK HEAD (Aggron Double-Edge takes NO recoil; identical seed).
  await run('MC1b Rock Head negates recoil',
    [mon('Aggron', ['doubleedge'], { ability: 'Rock Head', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    [mon('Snorlax', ['pound'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 252, def: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }], SEED);

  // MC2 — DRAIN: Giga Drain heals floor(dmg/2) to the injured USER (seed-identical to a
  //       non-drain move — draw-free).
  await run('MC2 Giga Drain heals floor(dmg/2)',
    [mon('Sceptile', ['gigadrain'], { ability: 'Overgrow', nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    [mon('Snorlax', ['pound'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }], SEED, [{ side: 0, hp: 80 }]);

  // MC3 — SELF-DROP: Overheat self -2 SpA. gen3 `selfDrops` DRAWS ONE random(100) (the
  //       secondaryRoll) then applies unconditionally — NOT draw-free.
  await run('MC3 Overheat self -2 SpA',
    [mon('Charizard', ['overheat'], { ability: 'Blaze', nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    [mon('Snorlax', ['pound'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 252, spd: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }], SEED);

  // MC4 — ITEM: Knock Off removes the target's Leftovers (gen3 no dmg boost); a 2nd Knock
  //       Off is a no-op (target itemless).
  await run('MC4 Knock Off removes Leftovers',
    [mon('Tyranitar', ['knockoff'], { ability: 'Sand Stream', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    [mon('Snorlax', ['pound'], { item: 'Leftovers', ability: 'Immunity', nature: 'Careful', evs: { hp: 252, spd: 252, def: 4 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }], SEED);

  // MC4b — ITEM: Knock Off BLOCKED by Sticky Hold (Muk keeps Leftovers).
  await run('MC4b Knock Off blocked by Sticky Hold',
    [mon('Tyranitar', ['knockoff'], { ability: 'Sand Stream', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    [mon('Muk', ['pound'], { item: 'Leftovers', ability: 'Sticky Hold', nature: 'Careful', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }], SEED);

  // MC5 — ITEM: Thief STEALS (attacker itemless → Gengar gains Leftovers). A 2nd Thief does
  //       nothing (now holds an item).
  await run('MC5 Thief steals (attacker itemless)',
    [mon('Gengar', ['thief'], { item: '', ability: 'Levitate', nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    [mon('Snorlax', ['pound'], { item: 'Leftovers', ability: 'Immunity', nature: 'Careful', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }], SEED);

  // MC5b — ITEM: Thief does NOT steal when the attacker HOLDS an item (both keep items).
  await run('MC5b Thief no steal (attacker holds an item)',
    [mon('Gengar', ['thief'], { item: 'Leftovers', ability: 'Levitate', nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    [mon('Snorlax', ['pound'], { item: 'Choice Band', ability: 'Immunity', nature: 'Careful', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }], SEED);

  // MC6 — RAPID SPIN: clears the USER's own Spikes (injected 3) + Leech Seed (injected).
  await run('MC6 Rapid Spin clears user spikes + leech',
    [mon('Forretress', ['rapidspin'], { ability: 'Sturdy', nature: 'Relaxed', evs: { hp: 252, def: 252 } })],
    [mon('Snorlax', ['pound'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }], SEED, [{ side: 0, spikes: 3, leechseed: true }]);

  // MC7 — gen3 `itemKnockedOff`: a mon whose item was KNOCKED OFF can neither have its item
  //       taken NOR gain one (takeItem returns false if the target OR source is knocked-off).
  //       So a Thief by a Knocked-Off attacker does NOTHING. dec0: Snorlax Knock Offs Skarmory
  //       (removes item + flags it); dec1: Skarmory Thiefs → nothing (the e2e_83 real-team bug).
  await run('MC7 Knocked-Off attacker Thiefs -> nothing',
    [mon('Skarmory', ['spikes', 'thief'], { item: 'Leftovers', ability: 'Keen Eye', nature: 'Impish', evs: { hp: 252, def: 252 } })],
    [mon('Snorlax', ['knockoff', 'pound'], { item: 'Leftovers', ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    // dec0: Skarmory Spikes (m1), Snorlax Knock Off (m1). dec1: Skarmory Thief (m2), Snorlax Pound (m2).
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 2' }], SEED);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
