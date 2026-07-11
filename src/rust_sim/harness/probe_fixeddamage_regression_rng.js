// probe_fixeddamage_regression_rng.js — capture GROUND-TRUTH seeds + draw counts for the
// DETERMINISTIC fixed-damage regression tests (the gen-3 fixed-damage draw-model edge cases
// this layer surfaced, each pinned in tests/regression_test.rs against a real-sim oracle):
//
//   FD-LEVEL  — Seismic Toss deals the USER's LEVEL exactly (level 100 → 100), drawing ONLY
//               its accuracy roll (acc 100, NOT never-miss → still draws) — NO crit, NO damage
//               roll. A spurious crit/damage roll would desync the seed; a wrong amount the HP.
//   FD-GHOST  — Seismic Toss (Fighting) into a GHOST is IMMUNE (0×): accuracy drawn THEN
//               `-immune` (SAME draw count as a landed hit). A wrong "never-miss on immune" or
//               a crit/damage roll would desync the seed.
//   FD-NORMAL — Night Shade (Ghost) into a NORMAL is IMMUNE (0×): zero damage, `-immune`.
//   FD-SUB    — Seismic Toss (100) into a Substitute hits the SUB (the NUMBER hits the sub HP,
//               breaks with no carry). The draw model is identical to a bare hit (acc only).
//
// For each constructed decision it logs seedBefore/seedAfter, the raw draw count, per-side
// active HP + sub HP — the exact ground-truth the Rust regression tests hardcode + assert.
// Run:  node src/rust_sim/harness/probe_fixeddamage_regression_rng.js
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
  (async () => { for await (const ch of streams.omniscient) for (const l of ch.split('\n')) if (l) log.push(l); })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  if (inject) inject(battle);
  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 40) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const before = battle.prng.getSeed();
    const dc0 = drawCount;
    const llen = log.length;
    const entry = plan[Math.min(i, plan.length - 1)];
    i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const after = battle.prng.getSeed();
    const rel = log.slice(llen).filter((l) => /-damage|-immune|-miss|-activate|-end|-start|faint/.test(l));
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const sub1 = a1 && a1.volatiles.substitute ? ` sub=${a1.volatiles.substitute.hp}` : '';
    const fmt = (m, s) => m ? `${m.species.name} ${m.hp}/${m.maxhp}${s || ''}${m.fainted ? ' FNT' : ''}` : '-';
    console.log(
      `  [${rs}] ${JSON.stringify(entry)} draws=${drawCount - dc0}\n` +
      `        seedBefore=${before}\n        seedAfter =${after}\n` +
      `        p1=${fmt(a0, '')}  p2=${fmt(a1, sub1)}`);
    for (const l of rel) console.log(`        > ${l}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // FD-LEVEL: Seismic Toss deals the user's level (100). Machamp (fast enough vs Snorlax?
  // no — Snorlax is slow). Machamp ST (100) into Snorlax (524 → 424). Snorlax Splashes so the
  // ONLY draws are Machamp's ST accuracy + the end-of-turn Quick Claw. Distinct speeds (no
  // action-order tie), so the count is clean.
  await run('FD-LEVEL: Seismic Toss = user level 100', [3, 5, 7, 9],
    [mon('Machamp', ['seismictoss'], { ability: 'Guts', evs: { hp: 252, atk: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // FD-GHOST: Seismic Toss into a Ghost (Gengar) → IMMUNE, accuracy-drawn-then-immune (SAME
  // seedAfter as a landed hit would be). Gengar Splashes.
  await run('FD-GHOST: Seismic Toss into a Ghost is immune (accuracy-only)', [3, 5, 7, 9],
    [mon('Machamp', ['seismictoss'], { ability: 'Guts', evs: { hp: 252, atk: 252 } })],
    [mon('Gengar', ['splash'], { ability: 'Levitate', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // FD-NORMAL: Night Shade (Ghost) into a Normal (Snorlax) → IMMUNE. Gengar Night Shades;
  // Snorlax Splashes. Zero damage, `-immune`, accuracy drawn.
  await run('FD-NORMAL: Night Shade into a Normal is immune', [3, 5, 7, 9],
    [mon('Gengar', ['nightshade'], { ability: 'Levitate', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // FD-SUB: Seismic Toss (100) into a held Substitute. p2 Snorlax already has a sub (injected
  // to hp 131 = floor(524/4)); Machamp ST → sub 131→31 (survives → -activate Substitute
  // [damage]). Draw model is a bare acc + Quick Claw (unchanged). We inject the sub so the
  // decision is a single clean ST-into-sub turn.
  await run('FD-SUB: Seismic Toss into a Substitute (sub 131 -> 31)', [3, 5, 7, 9],
    [mon('Machamp', ['seismictoss'], { ability: 'Guts', evs: { hp: 252, atk: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    (battle) => {
      // INJECT a substitute on the p2 Snorlax at floor(524/4)=131 HP (mirrors the engine's
      // create), and dock the mon's HP by that much (as if it had paid for the sub).
      const s = battle.sides[1].active[0];
      const cost = Math.floor(s.maxhp / 4);
      s.hp -= cost;
      s.volatiles.substitute = { hp: cost };
    });
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
