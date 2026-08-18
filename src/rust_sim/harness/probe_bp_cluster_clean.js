// PROBE (clean): per-TURN HP deltas for the gen-3 BP-modifier cluster, with the two
// confounds that ruined the first pass removed:
//   * CRIT — a crit is a x2 that mimics a BP doubling. Every damage line is tagged with
//     whether `|-crit|` appeared, and the ladder rows are read only from NON-crit turns.
//   * MISS — Fury Cutter is acc 95, so a miss silently shifts the whole ladder. Misses are
//     printed explicitly.
// Board: Mew mirror (base 100 all round, Psychic neutral, no STAB) so damage is a pure
// function of base power.
const path = require('path');
const PS = '/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream.js'));

const MEW = (moves) => `Mew||none|synchronize|${moves}|Hardy|85,85,85,85,85,85|N||||`;

async function turns(label, p1, p2, script, seed) {
  const s = new BattleStream(); const ch = [];
  (async () => { for await (const c of s) ch.push(c); })();
  s.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r => setTimeout(r, 150));
  const rows = [];
  for (const c of script) {
    const before = s.battle.sides.map(sd => sd.active[0].hp);
    const mark = ch.length;
    s.write(c); await new Promise(r => setTimeout(r, 150));
    const after = s.battle.sides.map(sd => sd.active[0].hp);
    const seg = ch.slice(mark).filter(x => !x.startsWith('sideupdate')).join('\n');
    rows.push({
      cmd: c.replace(/\n/g, ' | '),
      p2delta: before[1] - after[1],
      crit: /\|-crit\|/.test(seg),
      miss: /\|-miss\|/.test(seg),
      immune: /\|-immune\|/.test(seg),
      fail: /\|-fail\|/.test(seg),
      cure: (seg.match(/\|-curestatus\|[^\n]*/g) || []).join(','),
      drain: /\[from\] drain/.test(seg),
    });
  }
  console.log(`\n== ${label}  seed=${JSON.stringify(seed)}`);
  for (const r of rows) {
    console.log(`   ${r.cmd.padEnd(30)} p2dmg=${String(r.p2delta).padStart(4)}` +
      `${r.crit ? ' CRIT' : ''}${r.miss ? ' MISS' : ''}${r.immune ? ' IMMUNE' : ''}` +
      `${r.fail ? ' FAIL' : ''}${r.drain ? ' DRAIN' : ''}${r.cure ? ' cure:' + r.cure : ''}`);
  }
}

(async () => {
  // REVENGE: bp 60 -> 120 when the user was damaged BY THE TARGET this turn. Priority -4, so
  // the foe's Tackle always lands first on the same turn.
  await turns('REVENGE: hit-first (2x) vs not-hit (base)',
    MEW('revenge,splash'), MEW('tackle,splash'),
    ['>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 2'], [3,3,3,3]);

  // SMELLING SALTS: bp 60 -> 120 vs a PARALYZED target, and the onHit CURES it.
  // p2 has no Synchronize here (Sturdy) so the user never gets reflected par.
  await turns('SMELLING SALTS: vs par (2x + cure) then vs the now-cured foe (base)',
    MEW('thunderwave,smellingsalts,splash'),
    'Snorlax||none|sturdy|splash|Hardy|85,85,85,85,85,85|M||||',
    ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1', '>p1 move 2\n>p2 move 1'], [3,3,3,3]);

  // FURY CUTTER: bp 10, multiplier doubles per CONSECUTIVE use, capped at 160, and the
  // volatile is duration 2 so ONE non-Fury-Cutter turn lapses it.
  for (const seed of [[3,3,3,3], [11,11,11,11]]) {
    await turns('FURY CUTTER: 5 consecutive (ladder 10/20/40/80/160)',
      MEW('furycutter,splash'), 'Snorlax||none|sturdy|splash|Hardy|85,85,85,85,85,85|M||||',
      Array(5).fill('>p1 move 1\n>p2 move 1'), seed);
  }
  await turns('FURY CUTTER: broken by another move -> resets to 10',
    MEW('furycutter,splash'), 'Snorlax||none|sturdy|splash|Hardy|85,85,85,85,85,85|M||||',
    ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 1'], [3,3,3,3]);

  // DREAM EATER: onTryImmunity requires a SLEEPING target AND no substitute; drains 1/2.
  await turns('DREAM EATER: awake (immune) -> asleep (hits + drains)',
    MEW('spore,dreameater,seismictoss,splash'),
    'Snorlax||none|sturdy|splash|Hardy|85,85,85,85,85,85|M||||',
    ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1'], [3,3,3,3]);

  // FALSE SWIPE: onDamage clamps to target.hp - 1 when it WOULD KO.
  await turns('FALSE SWIPE: would-KO clamps to 1 HP',
    MEW('falseswipe,splash'),
    'Magikarp||none|swiftswim|splash|Hardy|0,0,0,0,0,0|M|0,0,0,0,0,0||5|',
    ['>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 1'], [3,3,3,3]);
})();
