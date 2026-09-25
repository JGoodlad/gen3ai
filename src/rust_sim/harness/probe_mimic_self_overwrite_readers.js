// PROBE: the readers of `lastMove` after a MIMIC that OVERWROTE ITS OWN SLOT —
// `gen3_mimic_self_overwrite_readers_v1`.
//
// SETTLED 2026-09-24 (run it to re-confirm; do not re-derive from source):
//
//   A successful Mimic replaces its OWN slot with the copied move, but the move the mon USED — its
//   `lastMove` — is still `mimic`, which is now in NO slot. Every reader resolves `lastMove` by ID:
//     E1  ENCORE into it FAILS: `encore.onStart` -> `getMoveData('mimic')` is undefined (and Mimic
//         carries `failencore`) -> `|move|<u>|Encore||[still]` + `|-fail|<u>`, after drawing the
//         accuracy roll AND the `durationCallback` `random(3,7)` (the EN3 fail shape).
//     M1  a NEW foe's MIMIC of it FAILS: `target.lastMove` is `mimic` (`failmimic`) ->
//         `|move|<u>|Mimic||[still]` + `|-fail|<u>`, draw-free after the accuracy roll.
//   (DISABLE fails the same way — MD1, `gen3_mimic_disable_self_overwrite_v1`.) A port that
//   stores `lastMove` as a SLOT reads the COPIED move there instead and encores / mimics it.
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream.js'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));
let draws = [];
const oR = PRNG.prototype.random;
PRNG.prototype.random = function (...a) { const r = oR.apply(this, a); draws.push(`random(${a})->${r}`); return r; };

async function run(label, p1, p2, script, seed, quiet) {
  draws = [];
  const s = new BattleStream(); const ch = [];
  (async () => { for await (const c of s) ch.push(c); })();
  s.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r => setTimeout(r, 60));
  const out = [];
  for (let i = 0; i < script.length; i++) {
    const mark = ch.length, dm = draws.length;
    s.write(script[i]);
    await new Promise(r => setTimeout(r, 60));
    const seg = ch.slice(mark).filter(x => !x.startsWith('sideupdate')).join('\n').split('\n')
      .filter(l => /^\|(move|-fail|-start|-activate|cant|switch|-status|-curestatus)\|/.test(l));
    out.push(`   t${i + 1} ${seg.join('  ')}   [${draws.slice(dm).length} draws] seedAfter=${s.battle.prng.getSeed()}`);
  }
  if (!quiet) { console.log(`\n== ${label} seed=${JSON.stringify(seed)}`); out.forEach(l => console.log(l)); }
  return out.join('\n');
}

(async () => {
  // E1 — Alakazam (faster) uses Psychic, Snorlax Mimics it (slot 1 := Psychic); then Encore.
  const snorlax = 'Snorlax||none|thickfat|mimic,bodyslam|Hardy|85,85,85,85,85,85|M||||';
  const kazam = 'Alakazam||none|synchronize|psychic,encore|Hardy|85,85,85,85,85,85|M||||';
  await run('E1 ENCORE into a mon whose lastMove is a self-overwriting MIMIC', snorlax, kazam,
    ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 2'], [3, 3, 3, 3]);
  // M1 — Breloom Mach Punches, Snorlax Mimics it, Breloom Spores it (it sleeps through the
  // switch), a Mr. Mime comes in and Mimics the sleeping Snorlax (whose lastMove is still Mimic).
  const breloom = 'Breloom||none|effectspore|machpunch,spore|Hardy|85,85,85,85,85,85|M||||]Mr. Mime||none|soundproof|mimic,splash|Hardy|85,85,85,85,85,85|M||||';
  const script = ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 2', '>p1 move 2\n>p2 switch 2', '>p1 move 2\n>p2 move 1'];
  for (let k = 1; k <= 40; k++) {
    const seed = [k, k, k, k];
    const o = await run('', snorlax, breloom, script, seed, true);
    const t = o.split('\n');
    // Want: T2 and T3 both `cant ... slp` for Snorlax (it never acts after its Mimic), T4 Mr. Mime Mimics.
    if (/cant\|p1a: Snorlax\|slp/.test(t[1]) && /cant\|p1a: Snorlax\|slp/.test(t[2]) && /Mr\. Mime\|Mimic/.test(t[3])) {
      console.log(`\n== M1 MIMIC of a mon whose lastMove is a self-overwriting MIMIC seed=${JSON.stringify(seed)}`);
      console.log(o);
      break;
    }
  }
})();
