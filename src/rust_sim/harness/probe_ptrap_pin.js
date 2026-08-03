// probe_ptrap_pin.js — GROUND-TRUTH generator for the ROUND-32 partial-trap regression pins.
//
// For each pinned board it drives the OMNISCIENT gen3 BattleStream with the SAME packed team
// strings the Rust pin uses, sweeps raw seeds until the scenario's guard predicate holds, and
// prints (a) the POST-CONSTRUCTION seed the Rust pin must be seeded at (the round-29
// methodology note: `start_with_switchins` is draw-free while the sim spends its first draws on
// the turn-0 construction window, so seeding the port with the RAW seed silently replays those
// as move-phase draws), (b) the per-decision protocol lines, and (c) the per-decision
// post-turn seed.
//
// Run: node harness/probe_ptrap_pin.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));

const tick = () => new Promise((r) => setTimeout(r, 0));
const KEEP = (l) => l && !l.startsWith('|t:|') && l !== '|';

async function run(p1, p2, seed, choices) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  (async () => { for await (const ch of streams.omniscient) for (const l of String(ch).split('\n')) lines.push(l); })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2 })}`);
  for (let i = 0; i < 16; i++) await tick();
  const battle = stream.battle;
  const initSeed = String(battle.prng.getSeed());
  const per = [];
  let lo = lines.length;
  for (const [c1, c2] of choices) {
    if (c1) streams.omniscient.write(`>p1 ${c1}`);
    if (c2) streams.omniscient.write(`>p2 ${c2}`);
    for (let k = 0; k < 14; k++) await tick();
    per.push({ lines: lines.slice(lo).filter(KEEP), seed: String(battle.prng.getSeed()) });
    lo = lines.length;
    if (battle.ended) break;
  }
  return { initSeed, per, battle, lines };
}

const SEEDS = [];
for (let s = 0; s < 80; s++) SEEDS.push([s * 7 + 1, s * 3 + 2, s * 5 + 4, s * 11 + 3]);

async function scenario(name, p1, p2, choices, pred) {
  for (const seed of SEEDS) {
    const r = await run(p1, p2, seed, choices);
    if (!pred(r)) continue;
    console.log(`\n######## ${name} ########`);
    console.log(`  raw seed          = ${JSON.stringify(seed)}`);
    console.log(`  POST-CONSTRUCTION = "${r.initSeed}"   <-- seed the Rust pin with THIS`);
    r.per.forEach((d, i) => {
      console.log(`  dec${i}: seed_after="${d.seed}"`);
      d.lines.forEach((l) => console.log(`        ${l}`));
    });
    return r;
  }
  console.log(`\n######## ${name} ######## -- NO SEED SATISFIED THE GUARD`);
  return null;
}

// Packed-team strings — byte-identical to the ones the Rust pins pass to `opts_cg`.
const DNITE_WRAP = 'Dragonite||Leftovers|InnerFocus|wrap,splash|Hardy|85,85,85,85,85,85|M||||';
const DNITE_WRAP_SPLASH = 'Dragonite||Leftovers|InnerFocus|wrap,splash|Hardy|85,85,85,85,85,85|M||||]Gengar||Leftovers|Levitate|splash|Hardy|85,85,85,85,85,85|M||||';
const LAX_2 = 'Snorlax||Leftovers|Immunity|splash|Hardy|85,85,85,85,85,85|M||||]Blissey||Leftovers|NaturalCure|splash|Hardy|85,85,85,85,85,85|F||||';
const LAX_SUB = 'Snorlax||Leftovers|Immunity|substitute,splash|Hardy|85,85,85,85,85,85|M||||';
const STARMIE_SPIN = 'Starmie||Leftovers|NaturalCure|rapidspin,splash|Hardy|85,85,85,85,85,85|N||||';
const LAX_BP = 'Snorlax||Leftovers|Immunity|batonpass,splash|Hardy|85,85,85,85,85,85|M||||]Blissey||Leftovers|NaturalCure|splash|Hardy|85,85,85,85,85,85|F||||';

const wrapLanded = (r) => r.per[0].lines.some((l) => l.includes('|-activate|') && l.includes('move: Wrap'));

async function main() {
  // PT1 — the FULL LIFECYCLE: cast (+the one random(3,7)) → chips → the NON-silent `-end`,
  //       plus the FIRM switch-block (a `switch 2` mid-trap must be rejected draw-free).
  await scenario('PT1 lifecycle + switch-block', DNITE_WRAP, LAX_2,
    [['move 1', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'],
     ['move 2', 'move 1'], ['move 2', 'move 1']],
    (r) => wrapLanded(r)
      // duration 5 → 4 chips → the `-end` lands inside the scripted window
      && r.per[0].lines.some((l) => l.includes('[partiallytrapped]'))
      && r.per.some((d) => d.lines.some((l) => l.startsWith('|-end|') && l.includes('[partiallytrapped]') && !l.endsWith('[silent]'))));

  // PT2 — the TRAPPER LEAVES: the residual sees `!source.isActive` → SILENT `-end`, NO chip.
  await scenario('PT2 trapper switches out (silent release)', DNITE_WRAP_SPLASH, LAX_2,
    [['move 1', 'move 1'], ['switch 2', 'move 1'], ['move 1', 'move 1']],
    (r) => wrapLanded(r) && r.per[1].lines.some((l) => l.includes('[partiallytrapped]|[silent]')));

  // PT3 — SUBSTITUTE: the sub intercepts before runMoveEffects → NO volatile, NO duration draw.
  await scenario('PT3 substitute blocks the trap', DNITE_WRAP, LAX_SUB,
    [['move 2', 'move 1'], ['move 1', 'move 2'], ['move 1', 'move 2']],
    (r) => r.per[1].lines.some((l) => l.includes('|-activate|p2a: Snorlax|Substitute|[damage]')));

  // PT4 — RAPID SPIN by the victim: the NON-silent `-end`, no `[from] move: Rapid Spin` tag.
  await scenario('PT4 rapid spin releases the trap', DNITE_WRAP, STARMIE_SPIN,
    [['move 1', 'move 2'], ['move 2', 'move 1'], ['move 2', 'move 2']],
    (r) => wrapLanded(r) && r.per[1].lines.some((l) => l.startsWith('|-end|') && l.includes('[partiallytrapped]')));

  // PT5 — BATON PASS: `partiallytrapped` has no `noCopy` → the entrant inherits it (same source,
  //       same duration) and is chipped off ITS OWN maxhp.
  await scenario('PT5 baton pass carries the trap', DNITE_WRAP, LAX_BP,
    [['move 1', 'move 2'], ['move 2', 'move 1'], [null, 'switch 2'], ['move 2', 'move 1'], ['move 2', 'move 1']],
    (r) => wrapLanded(r) && r.per[2].lines.some((l) => l.includes('[from] Baton Pass'))
      && r.per[2].lines.some((l) => l.includes('[partiallytrapped]')));
  // PT6 — PRECEDENCE: the turn the duration hits 0 AND the trapper leaves. `fieldEvent`
  //       decrements BEFORE calling onResidual, so the NON-silent onEnd form must win.
  await scenario('PT6 duration-0 beats trapper-gone', DNITE_WRAP_SPLASH, LAX_2,
    [['move 1', 'move 1'], ['move 2', 'move 1'], ['switch 2', 'move 1'], ['move 1', 'move 1']],
    (r) => wrapLanded(r)
      && r.per[2].lines.some((l) => l.startsWith('|-end|') && l.includes('[partiallytrapped]') && !l.includes('[silent]'))
      && r.per[2].lines.some((l) => l.startsWith('|switch|p1a')));
  // PT7 — the endTurn TrapPokemon HANDLER-SORT tie: `trapped` (Block) + `partiallytrapped`
  //       (Sand Tomb) on the same mon are two Conditions at the same speed AND subOrder 2 ⇒
  //       ONE extra `random(0,2)` per endTurn while both are live.
  await scenario('PT7 block + sandtomb tie the trap-event sort',
    'Onix||BlackBelt|Sturdy|block,sandtomb,splash|Hardy|85,85,85,85,85,85|M||||',
    'Wartortle||Leftovers|Torrent|splash|Hardy|85,85,85,85,85,85|M||||',
    [['move 2', 'move 1'], ['move 1', 'move 1'], ['move 3', 'move 1'], ['move 3', 'move 1']],
    (r) => r.per[0].lines.some((l) => l.includes('move: Sand Tomb'))
      && r.per[1].lines.some((l) => l.includes('|-activate|p2a: Wartortle|trapped')));
}
main().catch((e) => { console.error(e); process.exit(1); });
