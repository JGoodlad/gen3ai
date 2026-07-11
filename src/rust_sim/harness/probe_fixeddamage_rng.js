// probe_fixeddamage_rng.js — instrument the gen3 FIXED-DAMAGE draw model bit-for-bit.
//
// Fixed-damage moves are implemented in Showdown with a `damageCallback` that returns a
// fixed/derived number, BYPASSING getDamage — so NO crit roll, NO 16-way damage roll.
// This probe pins the exact per-decision draw COUNT + order against the OMNISCIENT
// in-process BattleStream (no server), so the Rust engine can mirror it.
//
// What we verify (each hint from the orchestrator is CONFIRMED here, not trusted):
//   1. The DRAW MODEL: does a fixed-damage move draw its accuracy `randomChance(acc,100)`?
//      (Seismic Toss / Night Shade / Dragon Rage are acc-100 but never_miss=false, so —
//      like the phaze acc-100 case — they STILL draw one accuracy roll. Sonic Boom /
//      Super Fang are acc-90 and can actually miss.) Does it draw crit? (Expected NO.)
//      Does it draw the damage roll? (Expected NO — fixed.) → diff a fixed-damage turn
//      vs a splash/splash turn and read the raw `battle.prng.next` delta.
//   2. TYPE IMMUNITY + its draw interaction: a Ghost is immune to Seismic Toss (Fighting
//      0x vs Ghost) — is accuracy drawn THEN -immune (like the damaging-move immune
//      short-circuit)? Same for Night Shade→Normal, Sonic Boom→Ghost, Super Fang→Ghost.
//   3. THE DAMAGE VALUE: Seismic Toss / Night Shade = user's LEVEL exactly. Sonic Boom =
//      20, Dragon Rage = 40. Super Fang = floor(target.hp/2), min 1.
//   4. SUBSTITUTE: a fixed-damage move into a Substitute — does it hit the SUB (sub HP
//      drops / breaks) or the mon? And for Super Fang, is the halving off the SUB's HP?
//   5. A fixed-damage KO → faint/win via the deferred-faint protocol.
//
// We wrap `battle.prng.next` to count raw draws per decision window, then run a few
// constructed turns and print the per-window draw count + realized state.
//
// Run:  node src/rust_sim/harness/probe_fixeddamage_rng.js
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

async function run(label, p1team, p2team, plan, seed = [7, 11, 13, 17]) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) for (const l of ch.split('\n')) if (l) log.push(l); })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 10; i++) await tick();

  const battle = stream.battle;
  let drawCount = 0;
  const drawLog = [];
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  console.log(`\n=== ${label} ===  seed=${JSON.stringify(seed)}`);
  let i = 0, safety = 0, logIdx = 0;
  while (!battle.ended && safety < 60) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const before = battle.prng.getSeed();
    const dc0 = drawCount;
    logIdx = log.length;
    const entry = plan[Math.min(i, plan.length - 1)];
    i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 16; k++) await tick();
    const after = battle.prng.getSeed();
    const drew = drawCount - dc0;
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const sub0 = a0 && a0.volatiles.substitute ? ` sub=${a0.volatiles.substitute.hp}` : '';
    const sub1 = a1 && a1.volatiles.substitute ? ` sub=${a1.volatiles.substitute.hp}` : '';
    const relevant = log.slice(logIdx).filter((l) =>
      /-damage|-immune|-miss|-activate|-end|-start|faint|-fail|-status/.test(l));
    console.log(
      `  [${rs}] ${JSON.stringify(entry)} draws=${drew} seedBefore=${before} seedAfter=${after}\n` +
      `        p1=${a0 ? a0.species.name + ' ' + a0.hp + '/' + a0.maxhp + sub0 + (a0.fainted ? ' FNT' : '') : '-'} ` +
      `p2=${a1 ? a1.species.name + ' ' + a1.hp + '/' + a1.maxhp + sub1 + (a1.fainted ? ' FNT' : '') : '-'}`);
    for (const l of relevant) console.log(`        > ${l}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // BASELINE: a splash/splash turn's draws are the same-speed-tie PRE-MOVE eachEvent
  // shuffles (a Snorlax mirror TIES → BeforeTurn/Update/Weather tie-shuffles) + the endTurn
  // Quick Claw = 7 for this Snorlax-vs-Snorlax mirror (NOT 1). The fixed-damage validation
  // is via the ABSOLUTE seedBefore/seedAfter per window, not a "=1" delta — a fixed-damage
  // move adds exactly its accuracy roll (2 relative to its own before/after seed).
  await run('BASELINE splash/splash (control window)',
    [mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1', stop: true }]);

  // (A) SEISMIC TOSS lands: user's LEVEL (100) damage. Draw model = accuracy (acc 100,
  //     NOT never-miss → still draws) + NO crit + NO damage roll, then Quick Claw. So a
  //     fixed-damage move adds exactly its accuracy roll (draws=2 measured against this
  //     window's own seedBefore/seedAfter — the absolute seed is the source of truth).
  await run('A: Seismic Toss lands (= level 100)',
    [mon('Machamp', ['seismictoss'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // (B) SEISMIC TOSS into a GHOST (Fighting 0x vs Ghost) → IMMUNE. Accuracy drawn THEN
  //     -immune (like the damaging-move immune short-circuit). Draw = acc + QuickClaw = 2.
  await run('B: Seismic Toss into a Ghost (immune, accuracy-only)',
    [mon('Machamp', ['seismictoss'], { evs: { hp: 252 } })],
    [mon('Gengar', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // (C) NIGHT SHADE into a NORMAL (Ghost 0x vs Normal) → IMMUNE. Same acc-then-immune.
  await run('C: Night Shade into a Normal (immune)',
    [mon('Gengar', ['nightshade'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // (C2) NIGHT SHADE lands (= level 100) into a non-Normal.
  await run('C2: Night Shade lands (= level 100)',
    [mon('Gengar', ['nightshade'], { evs: { hp: 252 } })],
    [mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // (D) A fixed-damage KO to WIN: Seismic Toss (100) KOs a level-1 last mon. The deciding
  //     faint draws NO Quick Claw. Draw = acc(1) only (the KO cancels the tail).
  await run('D: Seismic Toss KO-to-win (last mon, no Quick Claw)',
    [mon('Machamp', ['seismictoss'], { evs: { hp: 252 } })],
    [mon('Diglett', ['splash'], { level: 1 })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // (E) SEISMIC TOSS into a SUBSTITUTE: does it hit the sub (sub HP drops/breaks) or the
  //     mon? p2 subs turn 1 (Snorlax sub hp = floor(524/4)=131), then p1 Seismic Tosses
  //     (100) into the sub → sub 131→31 (survives). Draw for the ST turn = acc + QuickClaw.
  await run('E: Seismic Toss into a Substitute',
    [mon('Machamp', ['seismictoss'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // p1 ST (100) into no sub yet ; p2 Substitute (pays 131)
      { p1: 'move 1', p2: 'move 2', stop: true }, // p1 ST (100) into the sub → sub 131->31 ; p2 Splash
    ]);

  // (F) SONIC BOOM (fixed 20, acc 90, Normal) — CAN miss (acc 90). A Ghost is immune.
  await run('F: Sonic Boom lands (= 20)',
    [mon('Snorlax', ['sonicboom'], { evs: { hp: 252 } })],
    [mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);
  await run('F2: Sonic Boom into a Ghost (immune, Normal 0x)',
    [mon('Snorlax', ['sonicboom'], { evs: { hp: 252 } })],
    [mon('Gengar', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // (G) DRAGON RAGE (fixed 40, acc 100, Dragon — no gen3 type immunity).
  await run('G: Dragon Rage lands (= 40)',
    [mon('Salamence', ['dragonrage'], { ability: 'Intimidate', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // (H) SUPER FANG (halves TARGET current hp, floor, min 1, acc 90, Normal; Ghost immune).
  //     Snorlax maxhp 524 → 262 dealt (524->262). A 2nd Super Fang: 262 -> floor(262/2)=131.
  await run('H: Super Fang (halves current HP)',
    [mon('Raticate', ['superfang'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // 524 -> 262 (dealt 262)
      { p1: 'move 1', p2: 'move 1', stop: true }, // 262 -> 131 (dealt 131)
    ]);
  // (H2) SUPER FANG into a Ghost (immune, Normal 0x).
  await run('H2: Super Fang into a Ghost (immune)',
    [mon('Raticate', ['superfang'], { evs: { hp: 252 } })],
    [mon('Gengar', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);
  // (H3) SUPER FANG into a Substitute — does the halving use the SUB's HP or the mon's?
  //      p2 Snorlax subs (sub hp 131), then p1 Super Fangs the sub. If it uses the sub's
  //      hp: floor(131/2)=65 dealt to the sub (131->66). If it uses the mon's current hp
  //      (393 after paying sub): floor(393/2)=196 to the sub (breaks 131->0).
  await run('H3: Super Fang into a Substitute (whose HP is halved?)',
    [mon('Raticate', ['superfang'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // p1 SF (no sub yet: 524->262) ; p2 Substitute (pays 131, sub hp 131)
      { p1: 'move 1', p2: 'move 2', stop: true }, // p1 SF into the sub ; p2 Splash
    ]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
