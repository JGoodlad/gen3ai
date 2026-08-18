// PROBE: the gen-3 BP-MODIFIER / GATE cluster — FALSE SWIPE, REVENGE, SMELLING SALTS,
// FURY CUTTER, DREAM EATER. Draw model + BP ladder + gates + emission forms.
//
// Method: omniscient BattleStream (no server), fixed seed, gen3customgame.
//   * PRNG.prototype.random wrapped (the SOLE path to rng.next()) -> exact draw counts.
//   * BattleActions.prototype.getDamage wrapped at ENTRY -> the RESOLVED activeMove's
//     type/basePower/category and the returned damage.
// Boards use a MEW MIRROR where the number must be read (base 100 across the board, Psychic
// neutral to everything, no STAB) so realized damage is a pure function of base power.
const path = require('path');
const PS = '/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream.js'));
const { Dex } = require(path.join(PS, 'dist/sim/dex.js'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));
const { BattleActions } = require(path.join(PS, 'dist/sim/battle-actions.js'));

console.log('=== RESOLVED gen3 dex rows ===');
for (const id of ['falseswipe','revenge','smellingsalts','furycutter','dreameater','defensecurl','rollout','iceball','outrage','petaldance','thrash','rage','secretpower']) {
  const m = Dex.mod('gen3').moves.get(id);
  console.log(' ', id.padEnd(14),
    'num', String(m.num).padEnd(4), 'bp', String(m.basePower).padEnd(4),
    'acc', String(m.accuracy).padEnd(5), 'cat', m.category.padEnd(9), 'type', m.type.padEnd(9),
    'pri', m.priority,
    'flags', JSON.stringify(m.flags),
    m.basePowerCallback ? 'BPCB' : '', m.onModifyMove ? 'onModifyMove' : '',
    m.onTryHit ? 'onTryHit' : '', m.onHit ? 'onHit' : '',
    m.self ? 'self='+JSON.stringify(m.self) : '',
    m.drain ? 'drain='+JSON.stringify(m.drain) : '',
    m.volatileStatus ? 'vol='+m.volatileStatus : '',
    m.condition ? 'cond='+JSON.stringify(Object.keys(m.condition)) : '');
}

let draws = [];
const oR = PRNG.prototype.random;
PRNG.prototype.random = function (...a) { const r = oR.apply(this, a); draws.push(`random(${a})->${r}`); return r; };
let dmg = [];
const oG = BattleActions.prototype.getDamage;
BattleActions.prototype.getDamage = function (src, tgt, mv, sup) {
  const r = oG.apply(this, arguments);
  if (mv && typeof mv === 'object' && mv.id) dmg.push(`${mv.id}: bp=${mv.basePower} cat=${mv.category} type=${mv.type} -> ${r}`);
  return r;
};

async function run(label, p1, p2, script, seed) {
  draws = []; dmg = [];
  const s = new BattleStream(); const ch = [];
  (async () => { for await (const c of s) ch.push(c); })();
  s.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed || [9,9,9,9])}}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r => setTimeout(r, 150));
  const base = draws.length, dbase = dmg.length;
  const marks = [];
  for (const c of script) { const b = draws.length; s.write(c); await new Promise(r => setTimeout(r, 150)); marks.push(draws.length - b); }
  await new Promise(r => setTimeout(r, 150));
  const omni = [...new Set(ch.filter(c => !c.startsWith('sideupdate')))].join('\n').split('\n')
    .filter(l => /^\|(-damage|-heal|-fail|-immune|-miss|-curestatus|-start|-end|-status|move|cant|faint|-activate|-boost|-unboost)\|/.test(l));
  console.log(`\n== ${label}`);
  console.log('   ' + omni.join('\n   '));
  console.log('   DRAWS/write:', marks.join(','), '|', draws.slice(base).join('  ') || '(none)');
  console.log('   DMG:', dmg.slice(dbase).join(' ; ') || '(none)');
  const a = s.battle.sides.map(sd => sd.active[0]);
  console.log('   STATE:', a.map(p => `${p.name} ${p.hp}/${p.maxhp}${p.status ? ' ' + p.status : ''}`).join(' | '));
}
module.exports = { run };
if (require.main === module) require('./probe_bp_cluster_cases.js');
