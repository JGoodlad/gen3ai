// GROUND-TRUTH RNG PROBE for the CONFUSION-MOVE FAMILY (`gen3_confusion_move_family_v1`).
//
// Prints the REAL Showdown's PRNG seed BEFORE the first decision and AFTER the scripted
// decisions, for every disposition `tests/confusion_family_test.rs::CF1` pins. The printed
// `AFTER=` values are copied VERBATIM into that test as constants — regenerate with this
// script after ANY draw-order change, then update the constants.
//
//   node src/rust_sim/harness/probe_confusion_family_rng.js
//
// The BEFORE value is the port's seeding convention: `Battle::start_with_switchins` is given
// the sim's PRE-FIRST-DECISION state, so construction draws are not double-counted. It is the
// same for every board here because they share the `[9,9,9,9]` start seed and a one-mon team.
//
// 🚨 THE `splash` CONTROL ROW IS LOAD-BEARING and must stay first. Without it a seed mismatch
// cannot be distinguished from a wrong seeding convention — and comparing the port and the sim
// across DIFFERENT windows is exactly the mistake that made the first read of the ROUND-57
// Confuse Ray accuracy bug ambiguous.
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream.js'));

const GENGAR = (m) =>
  `|Gengar|Leftovers|Levitate|${m},splash,splash,splash|Hardy|85,85,85,85,85,85|M||||`;
const INERT = '|Snorlax|Leftovers|Immunity|splash,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||';
const SUBBER = '|Snorlax|Leftovers|Immunity|substitute,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||';
const OWN_TEMPO = '|Slowbro|Leftovers|OwnTempo|splash,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||';
const SOUNDPROOF = '|Voltorb|Leftovers|Soundproof|splash,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||';
const SAFEGUARDER = '|Snorlax|Leftovers|Immunity|safeguard,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||';

async function go(label, p1, p2, script) {
  const s = new BattleStream();
  const ch = [];
  (async () => { for await (const c of s) ch.push(c); })();
  s.write(
    `>start {"formatid":"gen3customgame","seed":[9,9,9,9]}\n` +
    `>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`
  );
  await new Promise((r) => setTimeout(r, 250));
  const before = s.battle.prng.getSeed();
  for (const c of script) {
    s.write(c);
    await new Promise((r) => setTimeout(r, 250));
  }
  console.log(
    `${label.padEnd(34)} BEFORE=${JSON.stringify(before)}  AFTER=${JSON.stringify(s.battle.prng.getSeed())}`
  );
}

const T1 = ['>p1 move 1\n>p2 move 1'];
// The blocker goes up on turn 1 (p1 splashes from slot 2), the cast lands on turn 2.
const T2 = ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 2'];

(async () => {
  await go('CONTROL splash', GENGAR('splash'), INERT, T1);
  for (const mv of ['confuseray', 'supersonic', 'sweetkiss', 'teeterdance']) {
    await go(`${mv} plain`, GENGAR(mv), INERT, T1);
  }
  await go('confuseray x2 (already confused)', GENGAR('confuseray'), INERT, [...T1, ...T1]);
  await go('supersonic vs Own Tempo', GENGAR('supersonic'), OWN_TEMPO, T1);
  await go('supersonic vs Soundproof', GENGAR('supersonic'), SOUNDPROOF, T1);
  await go('confuseray vs Substitute', GENGAR('confuseray'), SUBBER, T2);
  await go('supersonic vs Safeguard', GENGAR('supersonic'), SAFEGUARDER, T2);
})();
