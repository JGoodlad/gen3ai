// PROBE: gen-3 SKILL SWAP — draw model, emissions, the onStart NON-re-fire, the
// unconditional onEnd, the failure gates, switch-out restore, and Trace/Forecast.
//
// SETTLED 2026-08-18 (run it to re-confirm; do not re-derive from source):
//   NEVER-MISS (accuracy: true) -> NO accuracy roll. The move itself draws NOTHING.
//   success  : `|move|<u>|Skill Swap|<t>` then EXACTLY ONE line
//              `|-activate|<u>|Skill Swap|||[of] <t>`   (the gen<=4 branch: TWO empty
//              fields where gen5+ names the two abilities). NO -endability, NO -ability.
//   fail     : `|move|<u>|Skill Swap||[still]` + `|-fail|<u>`   (the attrLastMove('[still]')
//              did-nothing form). PP is deducted either way; zero extra draws.
//   FAIL iff : (a) EITHER side's ability carries flags.failskillswap -> in gen3 that is
//              WONDER GUARD **only** (multitype is gen4+), or
//              (b) gen <= 5 and the two ability IDS ARE EQUAL (incl. noability==noability).
//              A mon with "No Ability" is NOT a failure case on its own — `noability` swaps
//              like any other id.
//   onStart  : the swapped-in abilities do **NOT** re-fire their switch-in onStart
//              (`if (this.gen > 3)` gates the two singleEvent('Start') calls). Verified for
//              Intimidate / Drought<->Drizzle / Sand Stream / Trace / Forecast.
//   onEnd    : singleEvent('End') DOES fire, UNCONDITIONALLY, on BOTH abilities. The three
//              gen-3 carriers are flashfire / cloudnine / airlock:
//                flashfire -> removeVolatile -> `|-end|<loser>|ability: Flash Fire|[silent]`
//                             emitted AFTER the -activate; the receiver is NOT armed.
//                cloudnine/airlock -> eachEvent('WeatherChange') -> +1 `random(0,2)` at a
//                             cached-speed TIE (A/B measured 8 vs 7; 0 at distinct speed).
//   SetAbility: runEvent('SetAbility') fires twice but NO gen-3 ability has a handler ->
//              zero draws, never blocks.
//   switch   : the swap does NOT survive a switch-out — `ability` reverts to `baseAbility`
//              on re-entry (both directions).
//   Trace    : a Trace holder swaps its CURRENT (traced) ability; `baseAbility` stays trace,
//              so it RE-TRACES on re-entry (drawing `random(1)`). Receiving Trace is inert
//              (no onStart -> no copy); the receiver just carries a dead `trace`.
//   Forecast : swapping it OFF a formed Castform does NOT revert the forme, and the stale
//              forme then NEVER changes again. Swapping it ONTO a non-Castform is inert.
//   Protect BLOCKS it (flags.protect); Substitute does NOT (flags.bypasssub).
//   PP       : maxpp 16 (10*8/5); -1 on use, **-2 into a PRESSURE foe** (target:normal is
//              foe-directed), and PP is deducted on the FAIL and PROTECT paths too.
//   Type immunity is irrelevant (Status category ignores it) — a Dark target is hit.
const path = require('path');
const PS = '/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream.js'));
const { Battle } = require(path.join(PS, 'dist/sim/battle.js'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));
const { Dex } = require(path.join(PS, 'dist/sim/dex.js'));

let draws = [];
const origRandom = PRNG.prototype.random;   // the SOLE path to rng.next()
PRNG.prototype.random = function (...a) { const r = origRandom.apply(this, a); draws.push(`random(${a})->${r}`); return r; };
let adds = [];
const origAdd = Battle.prototype.add;       // stack-trace attribution for each emitted line
Battle.prototype.add = function (...a) {
  const st = (new Error().stack || '').split('\n').slice(2, 6).map((s) => s.trim().replace(/^at\s+/, '').replace(/\s*\(.*\)$/, ''))
    .filter((s) => s && !s.startsWith('Battle.add')).join(' <- ');
  adds.push({ line: a.map((x) => (x === null || x === undefined ? '' : String(x))).join('|'), st });
  return origAdd.apply(this, a);
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function mon(sp, mv, o = {}) {
  return `${sp}||${o.item || ''}|${o.ability === undefined ? '' : o.ability}|${mv.join(',')}|` +
    `${o.nature || 'Hardy'}|${o.evs || '85,85,85,85,85,85'}|${o.gender || 'M'}||||`;
}
function st(b) {
  return b.sides.map((s) => { const p = s.active[0]; return `${p.name}[${p.species.id}/${p.ability} base=${p.baseAbility} spe=${p.speed} atk${p.boosts.atk} vol:${Object.keys(p.volatiles)}]`; })
    .join('  ') + `  weather=${b.field.weather || 'none'}`;
}
async function run(label, p1, p2, script) {
  draws = []; adds = [];
  const s = new BattleStream(); const ch = [];
  (async () => { for await (const c of s) ch.push(c); })();
  s.write(`>start {"formatid":"gen3customgame","seed":[9,9,9,9]}\n>player p1 {"name":"P1","team":"${p1.join(']')}"}\n>player p2 {"name":"P2","team":"${p2.join(']')}"}`);
  await sleep(200);
  const b = s.battle;
  // DROP chunks starting with 'sideupdate'; drop |t:| / |split| / |debug| noise.
  const flat = () => ch.filter((c) => !c.startsWith('sideupdate')).join('\n').split('\n')
    .filter((l) => l.startsWith('|') && !l.startsWith('|t:|') && !l.startsWith('|split') && !l.startsWith('|debug') && l !== '|');
  console.log(`\n================= ${label}\n  START ${st(b)}`);
  let lm = flat().length, dm = draws.length, am = adds.length; const per = [];
  for (const w of script) {
    s.write(w); await sleep(200);
    const L = flat(); const nl = L.slice(lm); lm = L.length;
    const nd = draws.slice(dm); dm = draws.length;
    const na = adds.slice(am); am = adds.length; per.push(nd.length);
    console.log(`  --- ${JSON.stringify(w)}  draws=${nd.length}`);
    nl.forEach((l) => console.log(`      ${l}`));
    if (nd.length) console.log(`      DRAWS: ${nd.join('  ')}`);
    na.filter((x) => /Skill Swap|-formechange|-end\b|-ability/.test(x.line))
      .forEach((x) => console.log(`      [add] ${x.line}\n            ^ ${x.st}`));
    console.log(`      ${st(b)}`);
    console.log(`      p1a pp: ${b.sides[0].active[0].moveSlots.map((m) => `${m.id}:${m.pp}/${m.maxpp}`).join(' ')}`);
    if (b.ended) break;
  }
  console.log(`  draws/write = ${JSON.stringify(per)}`);
  return per;
}
function census() {
  const g3 = Dex.mod('gen3'); const m = g3.moves.get('skillswap');
  console.log(`== skillswap: num ${m.num} acc ${m.accuracy} cat ${m.category} target ${m.target} pri ${m.priority} flags ${JSON.stringify(m.flags)}`);
  const onEnd = [], setAb = [], fail = [];
  for (const a of g3.abilities.all()) {
    if (!a.exists || a.num < 0 || a.num > 76) continue;
    if (typeof a.onEnd === 'function') onEnd.push(a.id);
    for (const k of Object.keys(a)) if (/SetAbility/i.test(k)) setAb.push(`${a.id}.${k}`);
    if (a.flags && a.flags.failskillswap) fail.push(`${a.id}(${a.num})`);
  }
  console.log(`== gen3 abilities with onEnd (skillSwap fires it unconditionally): ${onEnd.join(', ')}`);
  console.log(`== gen3 abilities with a *SetAbility* hook: ${setAb.join(', ') || 'NONE'}`);
  console.log(`== gen3 abilities with flags.failskillswap: ${fail.join(', ') || 'NONE'}`);
}
const SS = ['skillswap', 'splash'];

(async () => {
  census();

  // ---------- 1. DRAW MODEL + EMISSIONS ----------
  await run('A: plain swap (Synchronize <-> Immunity)',
    [mon('Alakazam', SS, { ability: 'Synchronize' })], [mon('Snorlax', ['splash'], { ability: 'Immunity' })],
    ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1']);

  // ---------- 2. DOES A SWAPPED-IN ABILITY RE-FIRE ITS switch-in onStart? ----------
  await run('B: INTO Intimidate (a re-fire would add a 2nd -unboost)',
    [mon('Alakazam', SS, { ability: 'Synchronize' })], [mon('Gyarados', ['splash'], { ability: 'Intimidate' })],
    ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1']);
  await run('C: Drought <-> Drizzle (a re-fire would FLIP the weather)',
    [mon('Groudon', SS, { ability: 'Drought' })], [mon('Kyogre', ['splash'], { ability: 'Drizzle' })],
    ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1']);
  await run('C2: INTO Sand Stream',
    [mon('Alakazam', SS, { ability: 'Synchronize' })], [mon('Tyranitar', ['splash'], { ability: 'Sand Stream' })],
    ['>p1 move 1\n>p2 move 1']);
  await run('D: INTO Trace (a live-Trace holder hands Trace to the foe)',
    [mon('Porygon2', SS, { ability: 'Trace' })], [mon('Gardevoir', ['splash'], { ability: 'Trace' }), mon('Gyarados', ['splash'], { ability: 'Intimidate' })],
    ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 switch 2', '>p1 move 1\n>p2 move 1']);
  await run('E: Forecast OFF a formed Castform, then the weather CHANGES',
    [mon('Alakazam', ['raindance', 'skillswap', 'sunnyday', 'splash'], { ability: 'Synchronize' })],
    [mon('Castform', ['splash'], { ability: 'Forecast' })],
    ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1', '>p1 move 3\n>p2 move 1']);
  await run('E2: Forecast ONTO a non-Castform, then set rain',
    [mon('Alakazam', ['skillswap', 'raindance', 'splash'], { ability: 'Synchronize' })],
    [mon('Castform', ['splash'], { ability: 'Forecast' })],
    ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1']);

  // ---------- 2b. THE onEnd THAT *DOES* FIRE (A/B: swap vs splash, same board+seed) ----------
  const cnA = await run('F: CLOUD NINE swapped away at a SPEED TIE',
    [mon('Psyduck', ['sandstorm', 'skillswap', 'splash'], { ability: 'Cloud Nine' })], [mon('Psyduck', ['splash'], { ability: 'Damp' })],
    ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1']);
  const cnB = await run('F-control: same board, turn 2 SPLASH instead',
    [mon('Psyduck', ['sandstorm', 'skillswap', 'splash'], { ability: 'Cloud Nine' })], [mon('Psyduck', ['splash'], { ability: 'Damp' })],
    ['>p1 move 1\n>p2 move 1', '>p1 move 3\n>p2 move 1']);
  console.log(`\n  >>> CLOUD NINE onEnd WeatherChange delta at a TIE = ${cnA[1] - cnB[1]} (expect 1)`);
  const ctA = await run('F2: NO onEnd ability, SKILL SWAP', [mon('Psyduck', ['sandstorm', 'skillswap', 'splash'], { ability: 'Damp' })], [mon('Psyduck', ['splash'], { ability: 'Water Absorb' })], ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1']);
  const ctB = await run('F2-control: NO onEnd ability, SPLASH', [mon('Psyduck', ['sandstorm', 'skillswap', 'splash'], { ability: 'Damp' })], [mon('Psyduck', ['splash'], { ability: 'Water Absorb' })], ['>p1 move 1\n>p2 move 1', '>p1 move 3\n>p2 move 1']);
  console.log(`  >>> NO-onEnd delta = ${ctA[1] - ctB[1]} (expect 0)`);
  await run('G: FLASH FIRE armed on the USER, then swapped away',
    [mon('Houndoom', SS, { ability: 'Flash Fire' })], [mon('Charizard', ['ember', 'splash'], { ability: 'Blaze' })],
    ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 2']);
  await run('G2: FLASH FIRE armed on the TARGET, then swapped away',
    [mon('Alakazam', ['ember', 'skillswap', 'splash'], { ability: 'Synchronize' })], [mon('Houndoom', ['splash'], { ability: 'Flash Fire' })],
    ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1']);
  await run('H: CLOUD NINE swapped away WHILE a Forecast Castform is out (the End formes it)',
    [mon('Castform', ['raindance', 'splash'], { ability: 'Forecast' })], [mon('Psyduck', ['skillswap', 'splash'], { ability: 'Cloud Nine' })],
    ['>p1 move 1\n>p2 move 2', '>p1 move 2\n>p2 move 1', '>p1 move 2\n>p2 move 2']);

  // ---------- 3. FAILURE CASES ----------
  await run('I: target has WONDER GUARD', [mon('Alakazam', SS, { ability: 'Synchronize' })], [mon('Shedinja', ['splash'], { ability: 'Wonder Guard' })], ['>p1 move 1\n>p2 move 1']);
  await run('I2: USER has WONDER GUARD', [mon('Shedinja', SS, { ability: 'Wonder Guard' })], [mon('Snorlax', ['splash'], { ability: 'Immunity' })], ['>p1 move 1\n>p2 move 1']);
  await run('J: SAME ability both sides', [mon('Alakazam', SS, { ability: 'Synchronize' })], [mon('Espeon', ['splash'], { ability: 'Synchronize' })], ['>p1 move 1\n>p2 move 1']);
  await run('K: target has No Ability (SUCCEEDS)', [mon('Alakazam', SS, { ability: 'Synchronize' })], [mon('Snorlax', ['splash'], { ability: 'No Ability' })], ['>p1 move 1\n>p2 move 1']);
  await run('K2: BOTH have No Ability (same id -> FAILS)', [mon('Alakazam', SS, { ability: 'No Ability' })], [mon('Snorlax', ['splash'], { ability: 'No Ability' })], ['>p1 move 1\n>p2 move 1']);
  await run('L: target PROTECTs', [mon('Alakazam', SS, { ability: 'Synchronize' })], [mon('Snorlax', ['protect'], { ability: 'Immunity' })], ['>p1 move 1\n>p2 move 1']);
  await run('M: target behind a SUBSTITUTE (bypasssub -> hits)', [mon('Alakazam', SS, { ability: 'Synchronize' })], [mon('Snorlax', ['substitute', 'splash'], { ability: 'Immunity' })], ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 2']);
  await run('N: DARK target (Status category ignores type immunity)', [mon('Alakazam', SS, { ability: 'Synchronize' })], [mon('Houndoom', ['splash'], { ability: 'Flash Fire' })], ['>p1 move 1\n>p2 move 1']);

  // ---------- 4. DOES IT SURVIVE A SWITCH-OUT? ----------
  await run('O: the USER switches out and back',
    [mon('Alakazam', SS, { ability: 'Synchronize' }), mon('Jolteon', ['splash'], { ability: 'Volt Absorb' })], [mon('Snorlax', ['splash'], { ability: 'Immunity' })],
    ['>p1 move 1\n>p2 move 1', '>p1 switch 2\n>p2 move 1', '>p1 switch 2\n>p2 move 1']);
  await run('O2: the TARGET switches out and back',
    [mon('Alakazam', SS, { ability: 'Synchronize' })], [mon('Snorlax', ['splash'], { ability: 'Immunity' }), mon('Blissey', ['splash'], { ability: 'Natural Cure' })],
    ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 switch 2', '>p1 move 2\n>p2 switch 2']);

  // ---------- 5. TRACE ----------
  await run('P: a TRACED ability is swapped away, then the tracer pivots (re-traces)',
    [mon('Porygon2', SS, { ability: 'Trace' }), mon('Jolteon', ['splash'], { ability: 'Volt Absorb' })],
    [mon('Gyarados', ['splash'], { ability: 'Intimidate' }), mon('Snorlax', ['splash'], { ability: 'Immunity' })],
    ['>p1 move 2\n>p2 switch 2', '>p1 move 1\n>p2 move 1', '>p1 switch 2\n>p2 move 1', '>p1 switch 2\n>p2 move 1']);
  await run('P2: swapping onto a Porygon2 that TRACED the swapper -> same id -> FAIL',
    [mon('Alakazam', SS, { ability: 'Synchronize' })], [mon('Porygon2', ['splash'], { ability: 'Trace' })], ['>p1 move 1\n>p2 move 1']);

  // ---------- 6. bonus: a swapped-in RESIDUAL ability is live the SAME turn ----------
  await run('Q: swap INTO Speed Boost (its residual fires that same end-of-turn)',
    [mon('Alakazam', SS, { ability: 'Synchronize' })], [mon('Ninjask', ['splash'], { ability: 'Speed Boost' })],
    ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1']);
  await run('Q2: swap INTO Truant', [mon('Alakazam', SS, { ability: 'Synchronize' })], [mon('Slaking', ['splash'], { ability: 'Truant' })],
    ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1', '>p1 move 2\n>p2 move 1']);
  await run('R: PRESSURE foe deducts TWO Skill Swap PP (target:normal is foe-directed)',
    [mon('Alakazam', SS, { ability: 'Synchronize' })], [mon('Zapdos', ['splash'], { ability: 'Pressure' })], ['>p1 move 1\n>p2 move 1']);
  await run('R-control: non-Pressure foe deducts ONE',
    [mon('Alakazam', SS, { ability: 'Synchronize' })], [mon('Zapdos', ['splash'], { ability: 'Static' })], ['>p1 move 1\n>p2 move 1']);
})().then(() => process.exit(0)).catch((e) => { console.error(e); process.exit(1); });
