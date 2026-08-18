// PROBE 2: gen-3 WEATHER BALL follow-ups the first probe could not reach.
//   A. FREEZE THAW — does a SUN (runtime Fire) Weather Ball thaw a frozen defender?
//      gen3 frz.onDamagingHit keys on the BASE-dex type, and weatherball's base type is
//      Normal. Probe 1 could not test it: SUN BLOCKS FREEZE APPLICATION, so the freeze
//      must land FIRST (no weather), then sun, then the ball. Seeds are swept until a
//      10% Ice Beam freeze lands.
//   B. The CATEGORY FLIP's consequence — Counter (Physical) vs Mirror Coat (Special).
//      In sandstorm the ball is Rock/PHYSICAL, in rain Water/SPECIAL, so the reactive
//      pair must swap over.
const path = require('path');
const PS = '/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream.js'));
const { BattleActions } = require(path.join(PS, 'dist/sim/battle-actions.js'));

let dmgCalls = [];
const origGetDamage = BattleActions.prototype.getDamage;
BattleActions.prototype.getDamage = function (source, target, move) {
  const rec = (move && typeof move === 'object' && move.id)
    ? { id: move.id, type: move.type, bp: move.basePower, cat: move.category } : null;
  const r = origGetDamage.apply(this, arguments);
  if (rec) { rec.damage = r; dmgCalls.push(rec); }
  return r;
};

const RE = /^\|(move|-damage|-crit|-supereffective|-resisted|-immune|-miss|-fail|-status|-curestatus|-activate|-start|-weather|cant|turn|faint)\|/;

async function play(seed, p1, p2, script) {
  dmgCalls = [];
  const s = new BattleStream(); const ch = [];
  (async () => { for await (const c of s) ch.push(c); })();
  s.write(`>start {"formatid":"gen3customgame","seed":[${seed},${seed},${seed},${seed}]}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r => setTimeout(r, 160));
  for (const c of script) { s.write(c); await new Promise(r => setTimeout(r, 160)); }
  await new Promise(r => setTimeout(r, 160));
  const omni = ch.filter(c => !c.startsWith('sideupdate')).join('\n').split('\n').filter(l => RE.test(l));
  return { omni, calls: dmgCalls.slice() };
}

(async () => {
  // ---------- A. FREEZE THAW ----------
  const USER = 'Mew||Leftovers|Synchronize|icebeam,sunnyday,weatherball,flamethrower,splash|Hardy|0,0,0,0,0,252|||||';
  const FOE = 'Snorlax||Leftovers|Immunity|splash|Hardy|0,0,0,0,0,0|M||||';
  let found = null;
  for (let seed = 1; seed <= 60 && !found; seed++) {
    // turn1 Ice Beam (NO weather, so a freeze can land) -> hunt for |-status|...|frz
    const r = await play(seed, USER, FOE, ['>p1 move 1\n>p2 move 1']);
    if (r.omni.some(l => /^\|-status\|p2a: Snorlax\|frz/.test(l))) found = seed;
  }
  console.log('== A. FREEZE THAW — freezing seed found:', found);
  if (found) {
    // frozen turn1 -> sun turn2 -> WEATHER BALL (runtime Fire) turn3 -> FLAMETHROWER turn4 (control)
    const r = await play(found, USER, FOE,
      ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1', '>p1 move 3\n>p2 move 1', '>p1 move 4\n>p2 move 1']);
    console.log('   LOG:\n     ' + r.omni.join('\n     '));
    for (const c of r.calls) if (c.id !== 'splash') console.log(`   getDamage(${c.id}): type=${c.type} bp=${c.bp} cat=${c.cat} -> ${c.damage}`);
  }

  // ---------- B. COUNTER / MIRROR COAT (the category flip) ----------
  // p2 is SLOWER and answers with Counter (physical) or Mirror Coat (special).
  const WBUSER = 'Mew||Leftovers|Synchronize|weatherball,raindance,sandstorm,splash|Hardy|0,0,0,0,0,252|||||';
  const REACT = 'Blissey||Leftovers|NaturalCure|counter,mirrorcoat,splash|Hardy|0,0,0,0,0,0|F||||';
  for (const [label, setter] of [['SANDSTORM (Rock/PHYSICAL)', '>p1 move 3'], ['RAIN (Water/SPECIAL)', '>p1 move 2']]) {
    for (const [rname, rmove] of [['Counter', '>p2 move 1'], ['MirrorCoat', '>p2 move 2']]) {
      const r = await play(9, WBUSER, REACT, [`${setter}\n>p2 move 3`, `>p1 move 1\n${rmove}`]);
      const t2 = r.omni.slice(r.omni.indexOf('|turn|2') + 1);
      console.log(`\n== B. ${label} answered by ${rname}\n     ` + t2.join('\n     '));
    }
  }
})();
