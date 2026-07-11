// probe_shadowtag_rng.js — settle SHADOW TAG (universal trap) vs the resolved gen3 sim.
// Hypotheses (resolved dist): onFoeTrapPokemon → trapped=true UNCONDITIONALLY (no
// grounded/type gate — Flying + Levitate + Ghost all trapped); onFoeMaybeTrapPokemon
// skips ST holders (maybeTrapped display only — the TRAP itself has NO mirror
// exemption in gen3: a Wobbuffet mirror is MUTUALLY trapped). Draw model: onFoe*
// handlers → 1 handler per trap event → NO tie-shuffle (0 extra draws, like the
// Dugtrio mirror; UNLIKE Magnet Pull's onAny 4-draw mirror). Cross ST-vs-MP at equal
// speed: the MP holder's trap events carry 2 handlers → shuffle draws.
// Run: node src/rust_sim/harness/probe_shadowtag_rng.js

'use strict';
const { mon, run, fmtCalls } = require('./probe_batch4_lib');

const SEED = [7, 7, 7, 7];

async function trapFacts() {
  console.log('=== trap facts: who is trapped by Shadow Tag? (request.trapped / rejected switch)');
  const cases = [
    ['normal foe', mon('Zangoose', ['scratch'], { ability: 'Immunity' })],
    ['Flying foe', mon('Skarmory', ['scratch'], { ability: 'Keen Eye' })],
    ['Levitate foe', mon('Gengar', ['scratch'], { ability: 'Levitate' })],
    ['ST mirror', mon('Wobbuffet', ['scratch'], { ability: 'Shadow Tag' })],
  ];
  for (const [label, foe] of cases) {
    const teams = [
      [mon('Wobbuffet', ['splash'], { ability: 'Shadow Tag' }), mon('Rattata', ['scratch'], { ability: 'Guts' })],
      [foe, mon('Rattata', ['scratch'], { ability: 'Guts' })],
    ];
    const r = await run(teams, SEED, [['move 1', 'move 1'], ['move 1', 'switch 2'], ['move 1', 'move 1']], {
      onBoundary: (b) => ({
        p2trapped: b.p2.active[0].trapped, p2maybe: b.p2.active[0].maybeTrapped,
        p1trapped: b.p1.active[0].trapped, p2active: b.p2.active[0].species.id,
      }),
    });
    const errs = r.lines.filter((l) => l.includes('error') && l.includes('trapped'));
    console.log(`${label}: states=${JSON.stringify(r.states)} trapErr=${JSON.stringify(errs.slice(0, 1))}`);
  }
}

async function drawModel() {
  console.log('=== draw model: per-turn nexts — ST mirror vs no-trap control vs ST-vs-MagnetPull (all speed-tied)');
  const mk = (a1, a2, s1, s2) => [
    [mon(s1, ['splash'], { ability: a1 })],
    [mon(s2, ['splash'], { ability: a2 })],
  ];
  // Wobbuffet mirror: same species = speed tie. Control: Guts mirror on Wobbuffet.
  for (const [label, teams] of [
    ['ST mirror (Wobb)', mk('Shadow Tag', 'Shadow Tag', 'Wobbuffet', 'Wobbuffet')],
    ['control mirror (Wobb, Guts)', mk('Guts', 'Guts', 'Wobbuffet', 'Wobbuffet')],
    ['ST vs Guts (Wobb)', mk('Shadow Tag', 'Guts', 'Wobbuffet', 'Wobbuffet')],
    ['ST vs MagnetPull (Wobb)', mk('Shadow Tag', 'Magnet Pull', 'Wobbuffet', 'Wobbuffet')],
    ['MP vs Guts (Wobb)', mk('Magnet Pull', 'Guts', 'Wobbuffet', 'Wobbuffet')],
  ]) {
    const r = await run(teams, SEED, Array(3).fill(['move 1', 'move 1']));
    console.log(`${label}: perTurnNexts=${JSON.stringify(r.perDecision.map((d) => d.nexts))} t1=[${fmtCalls(r.perDecision[0].calls)}]`);
  }
}

(async () => { await trapFacts(); await drawModel(); })().catch((e) => { console.error(e); process.exit(1); });
