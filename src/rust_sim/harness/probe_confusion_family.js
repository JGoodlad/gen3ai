// PROBE: the gen-3 CONFUSION-INFLICTING MOVE FAMILY — draw model, emission forms, and the
// gates. The five moves left unmodeled after ROUND 44 shipped Confuse Ray:
//
//   supersonic (acc 55, SOUND) · sweetkiss (acc 75) · teeterdance (acc 100, allAdjacent)
//   swagger (acc 90, +2 Atk to the TARGET, then confuse) · flatter (acc 100, +1 SpA, then confuse)
//
// Every one carries `volatileStatus: 'confusion'`, so the hard half — the random(2,6) duration
// draw, the KO / already-confused / Own-Tempo gates and the `-start|confusion` emission — is the
// SAME `secondaries.rs::add_confusion` path Confuse Ray already uses. What has to be settled here
// is what the OTHER halves do, and specifically the two things no reading of the move data would
// tell you:
//
//   (1) for swagger/flatter, whether the BOOST and the CONFUSION are independent — i.e. what
//       happens when one of them cannot apply (target at +6 Atk; target already confused; Own
//       Tempo). A move that fails as a UNIT and a move whose two halves fail separately consume
//       DIFFERENT numbers of draws.
//   (2) whether Safeguard / Substitute / Soundproof gate the family, and at which step — each is
//       a different draw count.
//
// RUN IT (do not re-derive from the source): `node harness/probe_confusion_family.js`
//
// SETTLED — see the SETTLED block appended at the bottom of this file after the first run.
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream.js'));
const { Dex } = require(path.join(PS, 'dist/sim/dex.js'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));

const FAMILY = ['confuseray', 'supersonic', 'sweetkiss', 'teeterdance', 'swagger', 'flatter'];
const d3 = Dex.mod('gen3');
console.log('=== dex rows (gen3-RESOLVED — the mod-chain law: never read a single data file) ===');
for (const id of FAMILY) {
  const m = d3.moves.get(id);
  console.log(
    `${id.padEnd(12)} acc=${String(m.accuracy).padStart(3)} type=${m.type.padEnd(8)} ` +
    `target=${m.target.padEnd(11)} vol=${m.volatileStatus} boosts=${JSON.stringify(m.boosts || null)} ` +
    `flags=${JSON.stringify(m.flags)}`
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
  const s = new BattleStream();
  const ch = [];
  (async () => { for await (const c of s) ch.push(c); })();
  s.write(
    `>start {"formatid":"gen3customgame","seed":[9,9,9,9]}\n` +
    `>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`
  );
  await new Promise((r) => setTimeout(r, 200));
  const marks = [];
  for (const c of script) {
    const base = draws.length;
    s.write(c);
    await new Promise((r) => setTimeout(r, 200));
    marks.push(base);
  }
  await new Promise((r) => setTimeout(r, 200));
  const omni = ch.filter((c) => !c.startsWith('sideupdate')).join('\n').split('\n')
    .filter((l) => /^\|(-start|-end|-fail|-immune|-boost|-unboost|-activate|-miss|move|turn|-status)\|/.test(l));
  const i = omni.indexOf('|turn|1');
  console.log(`\n== ${label}`);
  console.log('  ' + omni.slice(i + 1).slice(0, 12).join('\n  '));
  console.log('  DRAWS (last decision):', draws.slice(marks[marks.length - 1]).join('  ') || '(none)');
}

// Teams. `Hardy` + flat 85 EVs mirrors probe_confuseray.js so the boards are comparable.
const T = (species, ability, moves, item = 'Leftovers') =>
  `${''}|${species}|${item}|${ability}|${moves}|Hardy|85,85,85,85,85,85|M||||`;

(async () => {
  const inert = T('Snorlax', 'Immunity', 'splash');
  const ownTempo = T('Slowbro', 'OwnTempo', 'splash');
  const soundproof = T('Voltorb', 'Soundproof', 'splash');

  // ── 1. the plain hit for each of the five, one at a time ──────────────────
  for (const mv of ['supersonic', 'sweetkiss', 'teeterdance', 'swagger', 'flatter']) {
    await run(`PLAIN ${mv}`, T('Gengar', 'Levitate', `${mv},splash,splash,splash`), inert,
      ['>p1 move 1\n>p2 move 1']);
  }

  // ── 2. swagger/flatter: are the BOOST and the CONFUSION independent? ──────
  // 2a. target already at +6 Atk (Swagger's boost cannot apply). Belly Drum takes the
  //     inert mon to +6 Atk on turn 1; Swagger lands on turn 2.
  await run('SWAGGER into a target ALREADY at +6 Atk (boost half cannot apply)',
    T('Gengar', 'Levitate', 'swagger,splash,splash,splash'),
    T('Snorlax', 'Immunity', 'bellydrum,splash,splash,splash'),
    ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 2']);

  // 2b. target ALREADY CONFUSED (the confusion half cannot apply, the boost can).
  await run('SWAGGER into an ALREADY-CONFUSED target (confusion half cannot apply)',
    T('Gengar', 'Levitate', 'confuseray,swagger,splash,splash'), inert,
    ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1']);

  // 2c. OWN TEMPO — immune to the confusion, but is the Atk boost still handed over?
  await run('SWAGGER into OWN TEMPO', T('Gengar', 'Levitate', 'swagger,splash,splash,splash'),
    ownTempo, ['>p1 move 1\n>p2 move 1']);
  await run('FLATTER into OWN TEMPO', T('Gengar', 'Levitate', 'flatter,splash,splash,splash'),
    ownTempo, ['>p1 move 1\n>p2 move 1']);
  await run('TEETER DANCE into OWN TEMPO', T('Gengar', 'Levitate', 'teeterdance,splash,splash,splash'),
    ownTempo, ['>p1 move 1\n>p2 move 1']);

  // ── 3. SOUNDPROOF vs the one SOUND move in the family ─────────────────────
  await run('SUPERSONIC into SOUNDPROOF', T('Gengar', 'Levitate', 'supersonic,splash,splash,splash'),
    soundproof, ['>p1 move 1\n>p2 move 1']);
  await run('SWEET KISS into SOUNDPROOF (control — NOT a sound move)',
    T('Gengar', 'Levitate', 'sweetkiss,splash,splash,splash'), soundproof,
    ['>p1 move 1\n>p2 move 1']);

  // ── 4. SAFEGUARD — does the ward block a confusion move, and at which step? ─
  await run('SUPERSONIC into SAFEGUARD',
    T('Gengar', 'Levitate', 'supersonic,splash,splash,splash'),
    T('Snorlax', 'Immunity', 'safeguard,splash,splash,splash'),
    ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 2']);
  await run('SWAGGER into SAFEGUARD (does the BOOST still land?)',
    T('Gengar', 'Levitate', 'swagger,splash,splash,splash'),
    T('Snorlax', 'Immunity', 'safeguard,splash,splash,splash'),
    ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 2']);

  // ── 5. SUBSTITUTE — a sub blocks foe-targeting status moves in gen3 ────────
  await run('SWAGGER into a SUBSTITUTE',
    T('Gengar', 'Levitate', 'swagger,splash,splash,splash'),
    T('Snorlax', 'Immunity', 'substitute,splash,splash,splash'),
    ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 2']);

  // ── 6. a SECOND cast of each — the already-confused fail form, per move ────
  for (const mv of ['supersonic', 'sweetkiss', 'teeterdance', 'flatter']) {
    await run(`SECOND ${mv} (target already confused)`,
      T('Gengar', 'Levitate', `${mv},splash,splash,splash`), inert,
      ['>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 1']);
  }
})();
