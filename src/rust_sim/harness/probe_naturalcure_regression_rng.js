// probe_naturalcure_regression_rng.js — GROUND-TRUTH seeds + post-switch STATUS for the
// `tests/regression_test.rs` Natural Cure pins (`gen3_natural_cure_v1`). Constructed
// `gen3customgame` scenarios with EXPLICIT seeds; prints the per-decision `seedAfter` + the
// switched-out mon's status so the Rust pins copy them verbatim. THE PROBE IS THE ONLY ORACLE.
//
// Mirrors the packed teams + seeds the Rust pins use EXACTLY (so the captured seeds line up):
//   NC1  natural_cure_cures_status_on_voluntary_switch_out — Starmie(NC, badly-poisoned) pivots
//        to a bench mon; a bench-back pivot returns it UNSTATUSED. + a non-NC control (returns
//        still toxic). Foe = Flygon (Ground, EQ) so the ONLY thing acting on Starmie is the
//        cure / the residual — the switch-out cure is draw-free.
//   NC2  natural_cure_is_a_no_op_on_a_faint — a burned 1-HP NC Blissey is KO'd by the foe; it
//        does NOT route through the SwitchOut cure (a fainted mon has nothing to cure). The
//        forced replacement is a normal switch. STATE-only pin.
//   NC3  natural_cure_phaze_drag_cures_the_dragged_out_mon — the foe ROARS the toxic'd NC
//        Starmie OUT; the phaze DRAG cures the dragged-out (now-bench) mon. + SEED (draw-free).
//
// Run:  node src/rust_sim/harness/probe_naturalcure_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const FORMAT = 'gen3customgame';

function tick() { return new Promise((r) => setTimeout(r, 0)); }

// Run a packed-team scenario with a fixed seed + scripted (p1,p2) choices; return per-decision
// seedAfter + a full team dump. `inject` sets status/hp on an active before the plan runs.
async function run(p1packed, p2packed, seed, plan, inject = []) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1packed })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2packed })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  // The ALIGNED init seed: the sim's PRNG state right before the FIRST decision (after the
  // `>start` switch-in setup). The Rust `start_with_switchins` is DRAW-FREE at start, so the
  // pins must seed with THIS value (NOT the raw `>start` seed) for the per-decision seeds to
  // line up bit-for-bit — the same alignment the golden's `initSeed` uses.
  const initSeed = battle.prng.getSeed();
  for (const inj of inject) {
    const m = battle.sides[inj.side].active[0];
    if (!m) continue;
    if (inj.status) m.setStatus(inj.status, m, null, true);
    if (inj.tox !== undefined && m.statusState) m.statusState.stage = inj.tox;
    if (inj.hp !== undefined) m.hp = inj.hp;
  }
  const seeds = [];
  seeds.initSeed = initSeed;
  for (const step of plan) {
    if (step.p1) { try { streams.omniscient.write(`>p1 ${step.p1}`); } catch (e) {} }
    if (step.p2) { try { streams.omniscient.write(`>p2 ${step.p2}`); } catch (e) {} }
    for (let k = 0; k < 14; k++) await tick();
    seeds.push({ note: step.note || '', seed: battle.prng.getSeed() });
  }
  return { battle, seeds, log };
}

function dumpTeam(battle, side) {
  return battle.sides[side].pokemon.map((p) => {
    const ss = p.statusState || {};
    const tag = ss.stage !== undefined ? ` tox=${ss.stage}` : (ss.time !== undefined ? ` slp=${ss.time}` : '');
    return `${p.species.name}(${p.status || 'ok'}${tag} hp=${p.hp}/${p.maxhp} fnt=${p.fainted ? 1 : 0} act=${p.isActive ? 1 : 0})`;
  }).join(' ');
}

(async () => {
  // ── NC1 — voluntary switch-out cure (+ non-NC control). p1 Starmie(NC) badly-poisoned +
  //   bench Snorlax; p2 Flygon (Ground/Levitate, EQ). Pivot to Snorlax (Starmie cured), pivot
  //   back to Starmie (returns clean). Foe EQ can't touch Starmie/Snorlax specially; the cure is
  //   the only status change → draw-free (seedAfter == the non-NC control's).
  const starmieNC = 'Starmie|||naturalcure|surf,recover|Timid|4,,,252,,252|||||]Snorlax|||owntempo|bodyslam,rest|Careful|252,,,,252,|||||';
  const starmieNonNC = 'Starmie|||illuminate|surf,recover|Timid|4,,,252,,252|||||]Snorlax|||owntempo|bodyslam,rest|Careful|252,,,,252,|||||';
  const flygon = 'Flygon|||levitate|earthquake|Adamant|,252,,,,252|||||';
  const nc1Plan = [
    { p1: 'switch 2', p2: 'move 1', note: 'NC Starmie pivots OUT (cured); Snorlax in' },
    { p1: 'switch 2', p2: 'move 1', note: 'Starmie pivots BACK (returns unstatused)' },
  ];
  const nc1a = await run(starmieNC, flygon, [11, 22, 33, 44], nc1Plan, [{ side: 0, status: 'tox', tox: 5 }]);
  const nc1b = await run(starmieNonNC, flygon, [11, 22, 33, 44], nc1Plan, [{ side: 0, status: 'tox', tox: 5 }]);
  console.log('=== NC1 natural_cure_cures_status_on_voluntary_switch_out ===');
  console.log('  NC   initSeed:', nc1a.seeds.initSeed, '| nonNC initSeed:', nc1b.seeds.initSeed);
  console.log('  NC seeds:     ', nc1a.seeds.map((s) => s.seed).join('  |  '));
  console.log('  nonNC seeds:  ', nc1b.seeds.map((s) => s.seed).join('  |  '));
  console.log('  seed-neutral (NC==nonNC):', JSON.stringify(nc1a.seeds.map((s) => s.seed)) === JSON.stringify(nc1b.seeds.map((s) => s.seed)));
  console.log('  NC    p1 team after:', dumpTeam(nc1a.battle, 0));
  console.log('  nonNC p1 team after:', dumpTeam(nc1b.battle, 0));

  // ── NC2 — faint is a no-op. p1 Blissey(NC), burned + 1 HP + bench Skarmory; p2 Tyranitar EQ
  //   (Ground; EQ KOs the 1-HP Blissey). Blissey faints; no SwitchOut cure; a forced replacement
  //   sends Skarmory. STATE pin (the fainted Blissey has no status to cure).
  const blisseyNC = 'Blissey|||naturalcure|softboiled|Calm|252,,252,,,|||||]Skarmory|||keeneye|spikes|Impish|252,,252,,,|||||';
  const ttarEQ = 'Tyranitar|||sandstream|earthquake|Adamant|,252,,,,252|||||';
  const nc2 = await run(blisseyNC, ttarEQ, [9, 10, 11, 12],
    [{ p1: 'move 1', p2: 'move 1', note: 'Tyranitar EQ KOs the 1-HP burned NC Blissey' },
      { p1: 'switch 2', note: 'p1 sends Skarmory (forced replacement)' }],
    [{ side: 0, status: 'brn', hp: 1 }]);
  console.log('\n=== NC2 natural_cure_is_a_no_op_on_a_faint ===');
  console.log('  initSeed:', nc2.seeds.initSeed);
  console.log('  seeds:', nc2.seeds.map((s) => `${s.note}: ${s.seed}`).join('  |  '));
  console.log('  p1 team after:', dumpTeam(nc2.battle, 0));

  // ── NC3 — phaze-drag cure. p1 Starmie(NC, toxic'd) + bench Snorlax; p2 Suicune (0-Spe Bold,
  //   slower → Roars AFTER Starmie moves; Roar priority -6). Starmie Surfs, Suicune ROARS →
  //   Starmie DRAGGED OUT (cured, n=1 sample of the only bench mon Snorlax). + SEED (draw-free
  //   cure; the drag's accuracy + n=1 sample draw as normal — the cure adds nothing).
  const suicuneRoar = 'Suicune|||pressure|toxic,roar|Bold|252,,252,,,|||||';
  const nc3 = await run(starmieNC, suicuneRoar, [5, 6, 7, 8],
    [{ p1: 'move 1', p2: 'move 1', note: 'Starmie Surf; Suicune Toxic (Starmie tox)' },
      { p1: 'move 1', p2: 'move 2', note: 'Starmie Surf; Suicune ROAR -> Starmie dragged out CURED' }],
    []);
  console.log('\n=== NC3 natural_cure_phaze_drag_cures_the_dragged_out_mon ===');
  console.log('  initSeed:', nc3.seeds.initSeed);
  console.log('  seeds:', nc3.seeds.map((s) => `${s.note}: ${s.seed}`).join('  |  '));
  console.log('  p1 team after:', dumpTeam(nc3.battle, 0));
  console.log('  drag/curestatus lines:');
  for (const l of nc3.log) if (l.includes('|drag|') || l.includes('curestatus') || l.includes('|-status|')) console.log('    ', l);
})();
