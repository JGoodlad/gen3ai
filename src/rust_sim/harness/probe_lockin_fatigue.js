// PROBE: the END-OF-LOCK confusion (`lockedmove.onEnd` -> `addVolatile('confusion')`) and the
// SELF-SOURCED confusion gates — `gen3_lockin_fatigue_v1`.
//
// SETTLED 2026-09-24 (run it to re-confirm; do not re-derive from source):
//
//   EMISSION: the lock-end confusion is `|-start|<user>|confusion|[fatigue]` — Outrage, Thrash and
//     Petal Dance alike. `confusion.onStart` tags `[fatigue]` when `sourceEffect.id ===
//     'lockedmove'`, and inside `lockedmove.onEnd` the ambient `battle.effect` IS the lockedmove
//     condition, so every lock-end add carries it. Every other gen-3 confusion source (Confuse
//     Ray / Supersonic / Teeter Dance / a Water Pulse secondary / a Figy-family berry) stays BARE.
//     Draw model unchanged: ONE `random(2,6)` at the add.
//
//   SAFEGUARD DOES NOT BLOCK A SELF-SOURCED CONFUSION. `safeguard.onTryAddVolatile` blocks
//     confusion only when `target !== source`, and `addVolatile` defaults `source` to the ambient
//     `battle.event.source`, else to the holder itself:
//       - the lock end runs in `removeVolatile`'s `singleEvent('End', …, this)` (no source) ->
//         source = the user -> NOT blocked: the user is confused and the `random(2,6)` DRAWS.
//       - a Figy-family berry is eaten at the residual (`eatItem` sets source = event.target =
//         the holder, and `singleEvent('Eat', …, this, source)`) -> NOT blocked either.
//     A foe's Confuse Ray / a Water Pulse secondary (target !== source) IS blocked (probe_safeguard).
//     A port that gates every confusion on the side's Safeguard drops one `random(2,6)` here.
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream.js'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));
let draws = [];
const oR = PRNG.prototype.random;
PRNG.prototype.random = function (...a) { const r = oR.apply(this, a); draws.push(`random(${a})->${r}`); return r; };

async function run(label, p1, p2, script, seed) {
  draws = [];
  const s = new BattleStream(); const ch = [];
  (async () => { for await (const c of s) ch.push(c); })();
  s.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r => setTimeout(r, 100));
  console.log(`\n== ${label} seed=${JSON.stringify(seed)}`);
  for (let i = 0; i < script.length; i++) {
    const mark = ch.length, dm = draws.length;
    s.write(`>p1 ${script[i][0]}\n>p2 ${script[i][1]}`);
    await new Promise(r => setTimeout(r, 100));
    const seg = ch.slice(mark).filter(x => !x.startsWith('sideupdate')).join('\n').split('\n').filter(l => !/^\|split\|/.test(l));
    const conf = seg.filter(l => /confusion|Safeguard|-enditem/.test(l));
    const d = draws.slice(dm).filter(x => /random\(2,6\)/.test(x));
    console.log(`   t${i + 1} ${conf.join('  ') || '-'}   ${d.join(' ')}   seedAfter=${s.battle.prng.getSeed()}`);
  }
}

(async () => {
  const snorlax = 'Snorlax||none|thickfat|splash|Hardy|85,85,85,85,85,85|M||||';
  for (const [mv, sp] of [['outrage', 'Dragonite'], ['thrash', 'Tauros'], ['petaldance', 'Vileplume']]) {
    const p1 = `${sp}||none|innerfocus|${mv},splash|Hardy|85,85,85,85,85,85|M||||`;
    await run(`${mv.toUpperCase()} lock end`, p1, snorlax, Array(5).fill(['move 1', 'move 1']), [3, 3, 3, 3]);
  }
  // The user's OWN side carries Safeguard when the lock ends.
  const dnite = 'Dragonite||none|innerfocus|outrage,safeguard|Hardy|85,85,85,85,85,85|M||||';
  const sg = [['move 2', 'move 1'], ...Array(5).fill(['move 1', 'move 1'])];
  await run('OUTRAGE lock end UNDER THE USER\'S SAFEGUARD', dnite, snorlax, sg, [3, 3, 3, 3]);
  await run('OUTRAGE lock end, no Safeguard (control)', dnite, snorlax, sg.map((c, i) => i === 0 ? ['move 1', 'move 1'] : c), [3, 3, 3, 3]);
  // A Figy Berry (confuses a -Atk nature) eaten under the holder's own Safeguard.
  const figy = 'Snorlax||figyberry|thickfat|safeguard,splash|Modest|85,85,85,85,85,85|M||||';
  const tosser = 'Blissey||none|naturalcure|seismictoss|Hardy|85,85,85,85,85,85|F||||';
  await run('FIGY BERRY under the holder\'s Safeguard', figy, tosser, [['move 1', 'move 1'], ...Array(4).fill(['move 2', 'move 1'])], [5, 5, 5, 5]);
  await run('FIGY BERRY, no Safeguard (control)', figy, tosser, Array(5).fill(['move 2', 'move 1']), [5, 5, 5, 5]);
})();
