// probe_phaze_regression_rng.js — capture GROUND-TRUTH seeds + draw counts for the
// DETERMINISTIC phaze regression tests (the gen-3 Roar/Whirlwind draw-model edge cases this
// layer surfaced, each pinned in tests/regression_test.rs against a real-sim oracle):
//
//   PHAZE-ACC   — gen-3 Roar/Whirlwind resolve to `accuracy: 100` (NOT `true`!), so a
//                 successful phaze turn draws `randomChance(100,100)` (the accuracy roll,
//                 always passes but CONSUMES a draw) THEN one `sample`/`random(n)` (the
//                 random target). A wrong "never-miss" model would skip the accuracy draw.
//   PHAZE-N1    — a phaze with EXACTLY ONE eligible foe bench mon STILL draws the `sample`
//                 (`random(1)` returns 0 but calls rng.next()). A wrong "n=1 draw-free"
//                 model desyncs the seed.
//   PHAZE-FAIL  — a phaze with NO eligible foe (the foe's last mon) draws ONLY the accuracy
//                 roll (no `sample`). A wrong model that drew the sample anyway desyncs.
//   PHAZE-KO    — a phaze that drags a (pre-chipped) grounded mon into a 3-layer-Spikes KO
//                 faints it on entry → forces a NORMAL replacement (the composition).
//
// For each constructed decision it logs seedBefore/seedAfter, the raw draw count, the first
// mover, the dragged species, and per-side active HP — the exact ground-truth the Rust
// regression tests hardcode + assert against. Run:
//   node src/rust_sim/harness/probe_phaze_regression_rng.js
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

// `inject` runs once after the leads are in, before the plan, to set HP / status directly.
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
    const drags = log.slice(llen).filter((l) => l.startsWith('|drag|')).map((l) => l.split('|')[3].split(',')[0].trim());
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'}${m.fainted ? ' FNT' : ''}` : '-';
    console.log(
      `  [${rs}] ${JSON.stringify(entry)} draws=${drawCount - dc0} drags=${JSON.stringify(drags)}\n` +
      `        seedBefore=${before}\n        seedAfter =${after}\n` +
      `        p1=${fmt(a0)}  p2=${fmt(a1)}  p2Left=${battle.sides[1].pokemonLeft}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // PHAZE-ACC + PHAZE-N1: a SLOW Suicune Roars a foe whose ONLY bench is one Snorlax → the
  // phaze draws the accuracy roll (randomChance(100,100)) THEN the n=1 `sample`. (Snorlax is
  // grounded but there are no Spikes, so it takes no chip.) p2 Blissey uses a draw-free,
  // never-miss self-move (Soft-Boiled at full HP fails → draw-free) so the ONLY draws are the
  // phaze accuracy + sample + the end-of-turn Quick Claw. We capture the post-turn seed.
  await run('PHAZE-ACC+N1: Roar with one eligible bench (acc roll + n=1 sample)', [3, 5, 7, 9],
    [mon('Suicune', ['roar', 'surf'], { ability: 'Pressure', evs: { hp: 252, def: 252 }, nature: 'Relaxed' })],
    [mon('Blissey', ['softboiled'], { ability: 'Pressure', evs: { hp: 252 } }),
     mon('Snorlax', ['bodyslam'], { ability: 'Pressure', evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1', stop: true }, // Roar → drag Snorlax (n=1) ; Blissey Soft-Boiled (full HP → fail)
    ]);

  // PHAZE-FAIL: a Roar when the foe has NO eligible bench (its last mon) draws ONLY the
  // accuracy roll (no sample). Same Suicune; p2 has ONLY Blissey. The post-turn seed reflects
  // exactly ONE fewer draw than the n=1 case (no sample) + the Quick Claw.
  await run('PHAZE-FAIL: Roar with no eligible bench (accuracy only, no sample)', [3, 5, 7, 9],
    [mon('Suicune', ['roar', 'surf'], { ability: 'Pressure', evs: { hp: 252, def: 252 }, nature: 'Relaxed' })],
    [mon('Blissey', ['softboiled'], { ability: 'Pressure', evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1', stop: true }, // Roar → -fail (no bench) ; Blissey Soft-Boiled
    ]);

  // PHAZE-KO: a Roar that drags a PRE-CHIPPED grounded lvl-1 mon into 3-layer Spikes → the
  // spikes KO it on entry → forces a NORMAL replacement. We set 3 spikes on the p2 side +
  // pre-chip the lvl-1 bench mons so the floor(maxhp/4) spikes chip is LETHAL on entry, and
  // capture the forced-switch boundary. (Skarmory Roars; p2's Diglett/Sandshrew bench are
  // lvl-1 at low HP.) The drag's runSwitch (EntryHazard → KO) draws nothing beyond the
  // phaze accuracy + sample; the resulting forced replacement is a NORMAL switch request.
  await run('PHAZE-KO: Roar drags a pre-chipped lvl-1 mon into a 3-layer Spikes KO', [1, 2, 3, 4],
    [mon('Skarmory', ['roar', 'drillpeck'], { ability: 'Keen Eye', evs: { hp: 252, atk: 252 } })],
    [mon('Blissey', ['softboiled'], { ability: 'Pressure', evs: { hp: 252 } }),
     mon('Diglett', ['scratch'], { level: 1, ability: 'Sand Veil' }),
     mon('Sandshrew', ['scratch'], { level: 1, ability: 'Sand Veil' })],
    [
      { p1: 'move 1', p2: 'move 1' }, // Roar → drag a lvl-1 bench → 3-layer spikes KO on entry → p2 forced replace
      { p1: 'move 1', p2: 'move 1', stop: true },
    ],
    (battle) => {
      // INJECT: 3 layers of Spikes on the p2 side + pre-chip the lvl-1 bench mons to 1 HP so
      // the spikes chip KOs them on entry. (lvl-1 mons have ~11-12 maxhp; floor(maxhp/4) ~ 3.)
      battle.sides[1].addSideCondition('spikes', 'debug');
      battle.sides[1].addSideCondition('spikes', 'debug');
      battle.sides[1].addSideCondition('spikes', 'debug');
      for (const p of battle.sides[1].pokemon) {
        if (p.level === 1) p.hp = 1;
      }
    });

  // PHAZE-PROTECT: gen-3 Roar / Whirlwind carry the `protect: 1` flag, so a Protect / Detect
  // on the TARGET BLOCKS the phaze at `TryHit` (AFTER the accuracy roll) → NO forceSwitchFlag →
  // NO drag → NO `sample` draw (`-activate Protect`, the target STAYS active). This was the
  // multi-phaze `sample` draw-POSITION desync (the port dragged an EXTRA `sample` into a
  // protected foe the sim left in place, shifting every later phaze's sample). p1 has a FAST
  // Skarmory that Protects (priority 3, never-miss → no accuracy draw) + one bench (Blissey);
  // p2 has a SLOW Suicune (Relaxed) that Roars (priority −6 → resolves last, into the up
  // Protect). The Roar draws its accuracy roll then is BLOCKED (no sample), so p1's Skarmory
  // STAYS active. Draws this turn: Skarmory Protect (first-protect, counter 0 → NO stall roll)
  // + Suicune Roar accuracy (randomChance(100,100)) + the end-of-turn Quick Claw = 2, NO sample.
  await run('PHAZE-PROTECT: a Protect BLOCKS a Roar (accuracy drawn, NO sample, target stays)', [11, 22, 33, 44],
    [mon('Skarmory', ['protect', 'steelwing'], { ability: 'Keen Eye', evs: { hp: 252, spe: 252 } }),
     mon('Blissey', ['softboiled'], { ability: 'Pressure', evs: { hp: 252 } })],
    [mon('Suicune', ['roar', 'surf'], { ability: 'Pressure', evs: { hp: 252, def: 252 }, nature: 'Relaxed' }),
     mon('Snorlax', ['bodyslam'], { ability: 'Pressure', evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1', stop: true }, // Skarmory Protect ; Suicune Roar → blocked, no drag
    ]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
