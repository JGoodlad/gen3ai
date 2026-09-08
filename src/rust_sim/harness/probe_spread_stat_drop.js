// PROBE: the gen-3 `allAdjacentFoes` STAT-DROP moves — are they, in SINGLES, the same mechanic
// as the already-modelled `target: normal` stat-drop family?
//
// THE QUESTION. `_stat_drop_boosts` in `tools/pokemon_data_extractor/sync.py` gates on
// `target in ("normal","adjacentFoe","any")`, so it emits `statDropBoosts` for Screech / Charm /
// Metal Sound / Feather Dance / Tickle / Fake Tears / Cotton Spore / Scary Face / Sand Attack —
// but NOT for the five gen-3 stat-drop moves whose target is `allAdjacentFoes`:
//
//   leer {def:-1} acc100 (83 learners) · growl {atk:-1} acc100 SOUND (78) ·
//   tailwhip {def:-1} acc100 (48) · sweetscent {evasion:-1} acc100 (20) ·
//   stringshot {spe:-1} acc95 (5)
//
// In SINGLES `allAdjacentFoes` resolves to the one foe, so the mechanic SHOULD be identical — but
// "should" is exactly the word the mod-chain law exists to distrust, and the sibling exclusion in
// that same function (accuracy/evasion, excluded for a reason that was already false) kept four
// moves fail-loud for nothing. So MEASURE, on every axis where a spread target could differ:
//
//   1. the DRAW MODEL      — accuracy roll then a draw-free boost, or something else?
//   2. the `|move|` ANNOUNCE — does a spread move still render the FOE as its target?
//   3. the -unboost FORM    — including the DELTA-0 line at the −6 floor
//   4. SOUNDPROOF           — Growl is the family's `sound` move
//   5. SUBSTITUTE           — none carry `bypasssub`
//   6. the onTryBoost gates — Clear Body (all) / Hyper Cutter (atk) / Keen Eye (accuracy)
//   7. SWEET SCENT drops EVASION on the FOE — the only foe-evasion drop in gen 3
//
// RUN IT: `node src/rust_sim/harness/probe_spread_stat_drop.js`
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream.js'));
const { Dex } = require(path.join(PS, 'dist/sim/dex.js'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));

const SPREAD = ['leer', 'growl', 'tailwhip', 'stringshot', 'sweetscent'];
const NORMAL = ['screech', 'charm', 'scaryface'];
const d3 = Dex.mod('gen3');
console.log('=== gen3-RESOLVED dex rows ===');
for (const id of [...SPREAD, ...NORMAL]) {
  const m = d3.moves.get(id);
  console.log(
    `${id.padEnd(11)} target=${m.target.padEnd(16)} boosts=${JSON.stringify(m.boosts)} ` +
    `acc=${String(m.accuracy).padStart(3)} flags=${JSON.stringify(m.flags)} ` +
    `status=${m.status} vol=${m.volatileStatus} onHit=${!!m.onHit} self=${JSON.stringify(m.self)}`
  );
}

let draws = [];
for (const m of ['random', 'randomChance', 'sample']) {
  const o = PRNG.prototype[m];
  PRNG.prototype[m] = function (...a) {
    const r = o.apply(this, a);
    draws.push(`${m}(${a})->${r}`);
    return r;
  };
}

async function run(label, p1, p2, script) {
  draws = [];
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const all = [];
  (async () => { for await (const ch of streams.omniscient) for (const l of ch.split('\n')) all.push(l); })();
  streams.omniscient.write('>start {"formatid":"gen3customgame","seed":[9,9,9,9]}');
  streams.omniscient.write(`>player p1 {"name":"P1","team":"${p1}"}`);
  streams.omniscient.write(`>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise((r) => setTimeout(r, 180));
  let base = draws.length;
  for (const c of script) {
    base = draws.length;
    for (const part of c.split('\n')) streams.omniscient.write(part);
    await new Promise((r) => setTimeout(r, 180));
  }
  const i = all.indexOf('|turn|1');
  const shown = all.slice(i + 1).filter((l) =>
    /^\|(move|-start|-end|-fail|-immune|-activate|-miss|-boost|-unboost)\|/.test(l));
  console.log(`\n== ${label}\n  ${shown.join('\n  ')}`);
  console.log('  DRAWS (last decision):', draws.slice(base).join('  ') || '(none)');
}

const T = (species, ability, moves, item = 'Leftovers') =>
  `|${species}|${item}|${ability}|${moves}|Hardy|85,85,85,85,85,85|M||||`;

(async () => {
  const inert = T('Snorlax', 'Immunity', 'splash,splash,splash,splash');
  const clearbody = T('Regirock', 'ClearBody', 'splash,splash,splash,splash');
  const hypercutter = T('Pinsir', 'HyperCutter', 'splash,splash,splash,splash');
  const keeneye = T('Hitmonchan', 'KeenEye', 'splash,splash,splash,splash');
  const soundproof = T('Voltorb', 'Soundproof', 'splash,splash,splash,splash');
  const subber = T('Snorlax', 'Immunity', 'substitute,splash,splash,splash');

  // 1-3. plain casts — draw model, announce target, -unboost form. A `normal`-target CONTROL
  //      (screech) runs alongside so any difference is attributable to the TARGET field alone.
  for (const mv of [...SPREAD, 'screech']) {
    await run(`PLAIN ${mv}`, T('Gengar', 'Levitate', `${mv},splash,splash,splash`), inert,
      ['>p1 move 1\n>p2 move 1']);
  }

  // 3b. THE −6 FLOOR — does a spread drop emit the delta-0 line like a `normal` one?
  //     Six Leers take the foe's Def to −6; the seventh is the floor case.
  await run('LEER x7 — the last one is AT the −6 Def floor',
    T('Gengar', 'Levitate', 'leer,splash,splash,splash'), inert,
    Array(7).fill('>p1 move 1\n>p2 move 1'));

  // 4. SOUNDPROOF vs Growl (sound) and vs Leer (the control — not a sound move).
  await run('GROWL into SOUNDPROOF', T('Gengar', 'Levitate', 'growl,splash,splash,splash'),
    soundproof, ['>p1 move 1\n>p2 move 1']);
  await run('LEER into SOUNDPROOF (control — not a sound move)',
    T('Gengar', 'Levitate', 'leer,splash,splash,splash'), soundproof, ['>p1 move 1\n>p2 move 1']);

  // 5. SUBSTITUTE.
  await run('LEER into a SUBSTITUTE', T('Gengar', 'Levitate', 'leer,splash,splash,splash'),
    subber, ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 2']);

  // 6. the onTryBoost immunity gates.
  await run('GROWL into CLEAR BODY', T('Gengar', 'Levitate', 'growl,splash,splash,splash'),
    clearbody, ['>p1 move 1\n>p2 move 1']);
  await run('GROWL into HYPER CUTTER (atk-only blocker)',
    T('Gengar', 'Levitate', 'growl,splash,splash,splash'), hypercutter, ['>p1 move 1\n>p2 move 1']);
  await run('LEER into HYPER CUTTER (control — Leer drops DEF, not Atk)',
    T('Gengar', 'Levitate', 'leer,splash,splash,splash'), hypercutter, ['>p1 move 1\n>p2 move 1']);
  await run('SWEET SCENT into KEEN EYE (control — Keen Eye guards ACCURACY, not evasion)',
    T('Gengar', 'Levitate', 'sweetscent,splash,splash,splash'), keeneye, ['>p1 move 1\n>p2 move 1']);
})();
