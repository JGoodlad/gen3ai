// probe_regression_rng.js — capture GROUND-TRUTH seeds for the seed-dependent
// engine-bug regression tests (the gaps the e2e fuzz found that need a real-sim
// oracle, not a pure state assertion):
//
//   #3-para  — the cached `pokemon.speed` model: a mon paralyzed WHILE ACTIVE keeps
//              its full turn-start speed through the move-phase `eachEvent('Update')`
//              tie-shuffles (drops to para-speed only at the residual), while one that
//              SWITCHES IN paralyzed ties on its para-speed AT ONCE. Reverting that to
//              a live-speed read flips a speed TIE → a different shuffle draw count →
//              a divergent post-turn seed.
//   #7       — the forced-replacement `updateSpeed` on commit: `commitChoices()` runs
//              `updateSpeed()` at its top on EVERY submit INCLUDING a mid-turn forced
//              replacement, so a foe paralyzed mid-turn drops to its para-speed before
//              the resumed-turn tail's tie-shuffles. Not re-caching there leaves it on
//              its stale full speed → a phantom tie → a divergent seed.
//   #6       — the residual handler GATHER order: a 2-mon DoT+Leftovers speed-tie's
//              Fisher-Yates shuffle permutes handlers in their PRE-SORT (gather) order,
//              so the status-DoT must be gathered before Leftovers per mon. (A STATE
//              desync the seed match masked — captured here for the HP state too.)
//
// For each constructed decision it logs seedBefore, seedAfter, the raw draw count for
// the decision, the first mover, and per-side active HP/status — the exact ground-truth
// the Rust regression test hardcodes + asserts against. Run:
//   node src/rust_sim/harness/probe_regression_rng.js
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

async function run(label, seed, p1team, p2team, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) for (const l of ch.split('\n')) if (l) log.push(l); })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  console.log(`\n=== ${label} ===  start_seed=${JSON.stringify(seed)}  initSeed=${battle.prng.getSeed()}`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 80) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const before = battle.prng.getSeed();
    const dc0 = drawCount;
    const entry = plan[Math.min(i, plan.length - 1)];
    i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const after = battle.prng.getSeed();
    const drew = drawCount - dc0;
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'}${m.fainted ? ' FNT' : ''} spe=${m.getStat ? m.getStat('spe') : '?'}` : '-';
    console.log(
      `  [${rs}] ${JSON.stringify(entry)} draws=${drew}\n` +
      `        seedBefore=${before}\n` +
      `        seedAfter =${after}\n` +
      `        p1=${fmt(a0)}\n        p2=${fmt(a1)}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // ----------------------------------------------------------------------------
  // #3-para: a SPEED-TIE Jirachi mirror. Both Jirachi have IDENTICAL stats (same
  // EV/IV/nature) so at full speed they TIE on every `eachEvent` shuffle (a draw
  // each). p2's Jirachi is paralyzed by p1's Thunder Wave on turn 1. On turn 2 the
  // para'd p2 mon keeps its FULL turn-start speed through the move-phase shuffles
  // (still tying p1 → shuffles still draw), and only drops to para-speed at the
  // residual. If the cached-speed model were reverted to a live read, the para'd mon
  // would read its quartered speed mid-turn → NO tie mid-turn → fewer shuffle draws →
  // a divergent turn-2 post-turn seed. Both use a NEVER-MISS, no-secondary, low-power
  // move (Swift) so turns 2+ are pure tie-shuffle + acc/dmg draws (no para full-para
  // draw asymmetry beyond the modeled para roll). We capture turn 2's seed.
  await run('PARA cached-speed: Jirachi mirror (para while active)', [3, 5, 7, 11],
    [mon('Jirachi', ['thunderwave', 'swift'], { evs: { hp: 252, spe: 252 }, nature: 'Timid' })],
    [mon('Jirachi', ['swift', 'thunderwave'], { evs: { hp: 252, spe: 252 }, nature: 'Timid' })],
    [
      { p1: 'move 1', p2: 'move 1' }, // p1 Thunder Wave paralyzes p2 ; p2 Swift
      { p1: 'move 2', p2: 'move 1' }, // both Swift — p2 para'd but still full-speed-tie this turn
      { p1: 'move 2', p2: 'move 1' }, // again — confirm cross-turn carry
    ]);

  // #3-para contrast: a mon that SWITCHES IN paralyzed ties on its PARA speed at once.
  // p2's bench Jirachi is pre-paralyzed; p2 switches it in. The post-switch shuffles
  // read its para speed immediately (53), so vs a faster p1 it does NOT tie → fewer
  // draws than the para-while-active case. (We pre-para via a Glare from a 3rd mon is
  // overkill — instead we just observe the para-on-entry path's draw count to confirm
  // the asymmetry the model encodes.)
  await run('PARA cached-speed: switch IN a (Thunder-Wave-on-prev-turn) para mon', [3, 5, 7, 11],
    [mon('Jirachi', ['thunderwave', 'swift'], { evs: { hp: 252, spe: 252 }, nature: 'Timid' })],
    [mon('Jirachi', ['swift'], { evs: { hp: 252, spe: 252 }, nature: 'Timid' }),
     mon('Jirachi', ['swift'], { evs: { hp: 252, spe: 252 }, nature: 'Timid' })],
    [
      { p1: 'move 1', p2: 'move 1' }, // p1 TWave para p2 active
      { p1: 'move 2', p2: 'switch 2' }, // p2 switches to the (unparalyzed) bench Jirachi
    ]);

  // ----------------------------------------------------------------------------
  // #7: forced-replacement updateSpeed. A Jirachi mirror where p1 KOs p2's lead AND
  // p2's lead got paralyzed the turn before. After the KO p2 must replace; the
  // replacement-commit `updateSpeed` refreshes BOTH actives. We want a case where, on
  // the resumed tail / next turn, the para'd-then-refreshed speeds matter. Simplest:
  // p1 (faster, unpara) vs p2 (para'd) — after p2's lead faints to an attack and a
  // fresh p2 mon enters, the next turn's ordering + shuffle count must reflect the
  // refreshed speeds. We capture the forced-switch boundary + the following turn.
  await run('#7 forced-replacement updateSpeed: para mirror + KO', [9, 13, 17, 19],
    [mon('Jirachi', ['thunderwave', 'swift', 'psychic'], { evs: { hp: 252, spa: 252, spe: 4 }, nature: 'Timid' })],
    [mon('Jirachi', ['swift'], { level: 5, evs: { hp: 252, spe: 252 }, nature: 'Timid' }),
     mon('Jirachi', ['swift'], { evs: { hp: 252, spe: 252 }, nature: 'Timid' })],
    [
      { p1: 'move 1', p2: 'move 1' }, // p1 TWave para p2 lvl5 lead
      { p1: 'move 3', p2: 'move 1' }, // p1 Psychic KOs the lvl5 lead → p2 forced replace
      { p1: 'move 2', p2: 'switch 2' }, // p2 brings in the full-speed Jirachi
      { p1: 'move 2', p2: 'move 1' }, // both Swift — tie-shuffle count on the refreshed speeds
    ]);

  // #7 (the TRUE surface): a foe paralyzed MID-TURN must drop to its para speed when a
  // forced replacement commits, so the resumed tail's `eachEvent('Update')` shuffles do
  // NOT spuriously tie it with the fresh entrant. Construct: p1 lead Jirachi-A (frail,
  // lvl 1) Thunder-Waves p2 Jirachi-C THIS turn (para'd mid-turn) and is itself KO'd by
  // C's Swift → p1 forced to replace with Jirachi-B (full 328). The resumed tail / next
  // turn compares C (para'd: stale full 328 vs live para 82) with B (328). Under the
  // bug (no re-cache on the forced commit) C reads its STALE 328 → phantom tie with B →
  // an extra shuffle draw → a divergent seed. We capture the forced-switch boundary +
  // the next turn's seeds.
  await run('#7 para-mid-turn foe vs fresh entrant after a forced replace', [4, 8, 15, 16],
    [mon('Jirachi', ['thunderwave', 'swift'], { level: 1, evs: { spe: 252 }, nature: 'Timid' }),
     mon('Jirachi', ['swift'], { evs: { hp: 252, spe: 252 }, nature: 'Timid' })],
    [mon('Jirachi', ['swift'], { evs: { hp: 252, spe: 252 }, nature: 'Timid' })],
    [
      { p1: 'move 1', p2: 'move 1' }, // p1-A(lvl1) TWaves C (para) ; C Swift KOs lvl1 A → p1 replace
      { p1: 'switch 2', p2: 'move 1' }, // (forced) p1 brings in Jirachi-B (328) vs para'd C
      { p1: 'move 1', p2: 'move 1' }, // both Swift — tie count on B(328) vs C(para 82)
      { p1: 'move 1', p2: 'move 1' },
    ]);

  // ----------------------------------------------------------------------------
  // #6: residual handler GATHER order. Two Gengar (Ghost/Poison) both burned + holding
  // Leftovers, at IDENTICAL speed (mirror) so the residual handler-sort ties — the
  // Fisher-Yates shuffle permutes the gathered handlers, and the gather order
  // (status-DoT-before-Leftovers per mon) decides the resulting side-order. One Gengar
  // is at low HP so its burn DoT self-KOs; if that KO ends the game, the survivor's
  // Leftovers must NOT have ticked first. Both use Tackle (Normal → Ghost-immune, 0
  // dmg) so the only HP change is the residual. We capture the post-turn seed + HP.
  await run('#6 residual gather order: 2-mon burn+Leftovers speed tie', [2, 3, 5, 7],
    [mon('Gengar', ['tackle'], { item: 'Leftovers', evs: { hp: 252, spe: 252 }, nature: 'Timid' }),
     mon('Gengar', ['tackle'], { item: 'Leftovers', evs: { hp: 252, spe: 252 }, nature: 'Timid' })],
    [mon('Gengar', ['tackle'], { item: 'Leftovers', evs: { hp: 252, spe: 252 }, nature: 'Timid' }),
     mon('Gengar', ['tackle'], { item: 'Leftovers', evs: { hp: 252, spe: 252 }, nature: 'Timid' })],
    [
      // We can't burn via a move easily in a mirror; instead this scenario is primarily
      // a STATE pin built in Rust by directly setting the burn + low HP. Here we just
      // surface the clean speed-tie residual draw count for reference.
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
    ]);

  // #3a: burn-DoT self-KO ends the game BEFORE the foe's Leftovers heal — a FASTER
  // burned mon at low HP self-KOs (status DoT sub 6, but SPEED outranks subOrder), and
  // being its side's last mon ENDS the game, so the slower foe's Leftovers (sub 4)
  // never ticks. Primarily a Rust STATE pin (constructed directly); here we just note
  // it is a distinct-speed (no tie) case so it draws cleanly.
  await run('#3a burn-DoT self-KO ends before foe Leftovers (distinct speed)', [2, 3, 5, 7],
    [mon('Gengar', ['tackle'], { item: 'Leftovers', evs: { hp: 252, spe: 252 }, nature: 'Timid' })], // FAST, will be burned + low HP in Rust
    [mon('Snorlax', ['tackle'], { item: 'Leftovers', evs: { hp: 252 }, nature: 'Brave' })],          // SLOW Leftovers holder
    [
      { p1: 'move 1', p2: 'move 1' },
    ]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
