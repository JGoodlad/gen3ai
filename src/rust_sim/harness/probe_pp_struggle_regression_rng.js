// probe_pp_struggle_regression_rng.js — GROUND-TRUTH seeds + exact numbers for the
// deterministic regression pins in tests/regression_test.rs. Constructs the SAME
// gen3customgame scenarios the pins use (fixed seed + scripted choices) and prints the
// post-decision SEED + the exact HP/PP/recoil the pins assert. Ground truth is COPIED
// VERBATIM from here into the pins.
//
// Pins covered:
//   - pp_decrements_on_use_draw_free       : a normal move decrements the used slot's PP by
//                                            1 and the SEED matches a no-PP-tracking baseline
//                                            (PP decrement is draw-free).
//   - pressure_decrements_two_pp           : a move into a Pressure holder decrements 2 PP.
//   - no_usable_move_forces_struggle       : a mon with its only-PP move exhausted is FORCED
//                                            to Struggle (the request offers Struggle only;
//                                            the sim auto-substitutes it), damaging the foe.
//   - struggle_recoil_is_gen3_quarter_damage_dealt : gen3 Struggle recoil = max(floor(dmg/4),1)
//     (gen3 calcRecoilDamage uses Math.floor — NOT round, NOT the gen4+ trunc(maxhp/4))
//                                            (the `recoil:[1,4]` path, NOT struggleRecoil=maxhp/4),
//                                            applied draw-free after the hit.
//
// Run:  node src/rust_sim/harness/probe_pp_struggle_regression_rng.js
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
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  console.log(`\n=== ${label} ===  seed=${JSON.stringify(seed)} initSeed=${battle.prng.getSeed()}`);
  const ppOf = (m) => m ? m.moveSlots.map((s) => `${s.id}:${s.pp}`).join(',') : '-';
  let i = 0, safety = 0;
  while (!battle.ended && safety < 60) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const dc0 = drawCount, logLen0 = log.length;
    const a0b = battle.sides[0].active[0], a1b = battle.sides[1].active[0];
    const hp1b = a1b ? a1b.hp : 0, hp0b = a0b ? a0b.hp : 0;
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    try { if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`); } catch (e) {}
    try { if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`); } catch (e) {}
    for (let k = 0; k < 18; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    // Recoil = the mover's HP loss this turn attributable to the Recoil line.
    const recoilLine = log.slice(logLen0).find((l) => /\|-damage\|.*\[from\] Recoil/.test(l));
    console.log(`  [${rs}] ${JSON.stringify(entry)} draws=${drawCount - dc0} after=${after}`);
    console.log(`        p1=${a0 ? `${a0.species.name} ${a0.hp}/${a0.maxhp}` : '-'} pp=[${ppOf(a0)}] | ` +
      `p2=${a1 ? `${a1.species.name} ${a1.hp}/${a1.maxhp}` : '-'} pp=[${ppOf(a1)}]`);
    if (recoilLine) {
      const dmgToFoe = hp1b - (a1 ? a1.hp : 0);
      console.log(`        RECOIL line: ${recoilLine}`);
      console.log(`        damage-dealt-to-foe=${dmgToFoe} => round(dmg/4)=${Math.round(dmgToFoe / 4)} floor=${Math.floor(dmgToFoe / 4)}`);
    }
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  const SEED = [1, 2, 3, 4];

  // PIN pp_decrements_on_use_draw_free: Suicune Surf into a bulky Snorlax. First decision's
  // seedAfter + Surf pp 24->23. (PP decrement is draw-free — the seed is a pure function of
  // the move's own acc/crit/dmg + residual + Quick Claw.)
  await run('pp_decrements_on_use_draw_free (Suicune Surf)', SEED,
    [mon('Suicune', ['surf', 'icebeam'], { evs: { hp: 252, spa: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // PIN pressure_decrements_two_pp: Snorlax Body Slam into a Pressure Suicune → bodyslam
  // pp 24->22 (−2). First decision's seedAfter + pp.
  await run('pressure_decrements_two_pp (Body Slam into Pressure Suicune)', SEED,
    [mon('Snorlax', ['bodyslam', 'splash'], { evs: { hp: 252, atk: 252 } })],
    [mon('Suicune', ['splash'], { ability: 'Pressure', evs: { hp: 252, def: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // PIN no_usable_move_forces_struggle + struggle_recoil_is_gen3_quarter_damage_dealt
  // (NO PP injection — pure natural exhaustion so a Rust pin can reproduce it with the SAME
  // scripted choices): a CB Snorlax with Extreme Speed (m0, 8 PP, 5-base-PP) LOCKS to it and
  // spams it 8× into a LEVITATE Gengar (Extreme Speed [Normal] → `-immune`, 0 damage, PP still
  // −1). After 8 uses ES is at 0 PP → decision 8 (0-based) FORCES Struggle (the other slots are
  // Choice-disabled). Struggle (typeless '???') HITS the Ghost; Gengar uses Splash (no damage
  // to Snorlax) so Snorlax's HP loss on the Struggle turn is EXACTLY the recoil = floor(dmg/4).
  // We print all decisions so the pin can copy dec 7 (ES pp 0, the last immune ES) + dec 8 (the
  // Struggle turn: seedAfter + Snorlax recoil + Gengar HP loss).
  await run('forced Struggle + gen3 recoil (natural exhaustion: CB Snorlax ES into Levitate Gengar)', SEED,
    [mon('Snorlax', ['extremespeed', 'bodyslam', 'crunch', 'shadowball'],
      { item: 'Choice Band', ability: 'Immunity', evs: { hp: 252, atk: 252 } })],
    [mon('Gengar', ['splash'], { ability: 'Levitate', evs: { hp: 252, spe: 252 } })],
    // 9 decisions: 8× Extreme Speed (immune, pp 8→0), then the forced Struggle.
    Array.from({ length: 9 }, (_v, i) => ({ p1: 'move 1', p2: 'move 1', stop: i === 8 })));
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
