// probe_batch5_reactive.js — ground-truth the REACTIVE FIXED-DAMAGE family bit-for-bit vs the
// OMNISCIENT in-process gen3 BattleStream (no server): COUNTER / MIRROR COAT / ENDEAVOR.
//
// The resolved gen3 sources (dumped below) say:
//   counter / mirrorcoat: bp 0, acc 100 (NOT never-miss), priority -5, target 'scripted',
//     ignoreImmunity FALSE (counter Fighting -> Ghost immune; mirrorcoat Psychic -> Dark immune),
//     a beforeTurnCallback that addVolatile()s a duration-1 volatile whose onStart RESETS
//     {slot:null, damage:0}, and an onDamage (priority -101) recorder:
//       counter:    effect.effectType==='Move' && foe && (category==='Physical' || effect.id==='hiddenpower')
//       mirrorcoat: effect.effectType==='Move' && foe &&  category==='Special'  && effect.id!=='hiddenpower'
//     damageCallback = volatile.damage (2x the recorded hit) || 1; onTry fails when no volatile
//     or slot===null.
//   endeavor: bp 0, acc 100 (NOT never-miss), priority 0, Normal, ignoreImmunity FALSE,
//     damageCallback = target.hp - pokemon.hp, onTryImmunity pokemon.hp < target.hp,
//     onTry (hp >= target.hp) -> |-fail|.
//
// This probe settles BEHAVIORALLY (the sim is the only oracle):
//   1. the exact draw model per move (accuracy drawn? crit? damage roll? the beforeTurn
//      volatile add draw-free? the duration-1 residual handler tie draws?)
//   2. the "damage to return" source: same-turn only (the onStart reset), physical-vs-special
//      category, the bare-hiddenpower special case BOTH ways, sub-absorbed hits, multihit
//      (last strike vs total), prev-turn damage.
//   3. the fail conditions + their draw counts (no damage / wrong category / foe switched).
//   4. type immunity: Counter->Ghost, Mirror Coat->Dark, Endeavor->Ghost (accuracy-then-immune?).
//   5. Endeavor's exact rule (hp >= target.hp fails incl. EQUALITY; the delta; behind a sub).
//
// Run:  node src/rust_sim/harness/probe_batch5_reactive.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: opts.gender || 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

function dumpResolved() {
  const d = Dex.forFormat(FORMAT);
  for (const id of ['counter', 'mirrorcoat', 'endeavor']) {
    const m = d.moves.get(id);
    console.log(`=== resolved gen3 ${id} ===`);
    console.log(`  cat=${m.category} bp=${m.basePower} acc=${JSON.stringify(m.accuracy)} type=${m.type} prio=${m.priority} pp=${m.pp} target=${m.target}`);
    console.log(`  flags=${JSON.stringify(m.flags)} ignoreImmunity=${JSON.stringify(m.ignoreImmunity)}`);
    for (const k of Object.keys(m)) {
      const v = m[k];
      if (typeof v === 'function') console.log(`  fn ${k}: ${v.toString().replace(/\s+/g, ' ')}`);
    }
    if (m.condition) {
      console.log('  --- condition:');
      for (const k of Object.keys(m.condition)) {
        const v = m.condition[k];
        if (typeof v === 'function') console.log(`      fn ${k}: ${v.toString().replace(/\s+/g, ' ')}`);
        else console.log(`      ${k}=${JSON.stringify(v)}`);
      }
    }
    // gen4 / base deltas (the mod-chain law: know what gen3 REPLACED).
    for (const g of ['gen4', 'gen9']) {
      const gm = Dex.mod(g === 'gen9' ? 'gen9' : g).moves.get(id);
      console.log(`  [${g}] cat=${gm.category} acc=${JSON.stringify(gm.accuracy)} prio=${gm.priority} ignoreImmunity=${JSON.stringify(gm.ignoreImmunity)}`);
    }
  }
  for (const id of ['hiddenpower', 'hiddenpowerfighting', 'hiddenpowerice', 'doubleslap']) {
    const m = d.moves.get(id);
    console.log(`  ${id}: cat=${m.category} type=${m.type} bp=${m.basePower} acc=${JSON.stringify(m.accuracy)} multihit=${JSON.stringify(m.multihit)}`);
  }
}

function drawLabel() {
  const st = new Error().stack.split('\n');
  const frames = [];
  for (let i = 3; i < st.length && frames.length < 5; i++) {
    const mm = st[i].match(/at ([\w.<>]+) /);
    if (mm) frames.push(mm[1]);
  }
  return frames.join('<');
}

// plan entries: { p1, p2, pre: [injections], stop }
// inject: { seed, acts: [{side, slot, status, hp, faint, item, boosts}] }
async function run(label, p1team, p2team, plan, inject) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  const seed = (inject && inject.seed) || [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  const applyActs = (acts) => {
    for (const inj of acts || []) {
      const side = battle.sides[inj.side];
      const m = inj.slot === undefined ? side.active[0] : side.pokemon[inj.slot];
      if (!m) continue;
      if (inj.status) m.setStatus(inj.status, m, null, true);
      if (inj.hp !== undefined) m.hp = inj.hp;
      if (inj.faint) { m.hp = 0; m.fainted = true; }
      if (inj.item !== undefined) m.item = inj.item;
      if (inj.boosts) Object.assign(m.boosts, inj.boosts);
    }
  };
  applyActs(inject && inject.acts);

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);

  let draws = [];
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { const v = realNext(...a); draws.push(drawLabel()); return v; };

  let i = 0, safety = 0;
  while (!battle.ended && safety < 16) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    applyActs(entry.pre);
    draws = [];
    const logLen0 = log.length;
    const before = battle.prng.getSeed();
    const hpBefore = battle.sides.map((s) => s.active[0] ? `${s.active[0].species.name}:${s.active[0].hp}/${s.active[0].maxhp}` : '-');
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const ppStr = (m) => m ? m.moveSlots.map((s) => `${s.id}:${s.pp}/${s.maxpp}`).join(',') : '-';
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'} vols=[${Object.keys(m.volatiles).join(',')}] pp={${ppStr(m)}}` : '-';
    console.log(`  [${rs}] ${JSON.stringify({ p1: entry.p1, p2: entry.p2 })} draws=${draws.length}  seed ${before}->${after}`);
    console.log(`        pre : p1=${hpBefore[0]}  p2=${hpBefore[1]}`);
    console.log(`        post: p1=${fmt(a0)}`);
    console.log(`              p2=${fmt(a1)}`);
    draws.forEach((dl, k) => console.log(`        DRAW[${k}] ${dl}`));
    const newLines = log.slice(logLen0).filter((l) =>
      /\|move\||\|turn\||-damage|-heal|-boost|-unboost|-fail|-immune|-miss|-crit|-supereffective|-resisted|cant|-activate|-hitcount|-end\b|-start|switch|drag|faint|-prepare|-singleturn|-nothing/.test(l));
    for (const l of newLines) console.log(`        LINE ${l}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  dumpResolved();

  // ---------------------------------------------------------------- COUNTER
  // C1: baseline success (2x a physical hit), prev-turn NON-persistence, special fail, multihit.
  await run('C1 counter: physical 2x / prev-turn? / special fail / multihit',
    [mon('Snorlax', ['counter', 'splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['drillpeck', 'surf', 'splash', 'armthrust'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },              // drill peck (phys) -> counter 2x
     { p1: 'move 1', p2: 'move 3' },              // splash -> does LAST TURN's damage count?
     { p1: 'move 1', p2: 'move 2' },              // surf (special) -> counter fails?
     { p1: 'move 1', p2: 'move 4', stop: true }]); // armthrust (acc 100 multihit) -> 2x last strike or total?

  // C2: no damage at all + the foe SWITCHES (volatile armed at beforeTurn, switch resolves first).
  await run('C2 counter: no-damage fail draws + foe switch',
    [mon('Snorlax', ['counter', 'splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['drillpeck', 'splash'], { evs: { hp: 252 } }), mon('Forretress', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 2' },              // splash -> counter fail: draw count?
     { p1: 'move 1', p2: 'switch 2', stop: true }]); // foe switches -> counter fail? draws?

  // C3: counter INTO a Ghost — the damager is Gengar hitting a FIGHTING-type counter user with
  // Shadow Ball (Ghost = PHYSICAL in gen3, neutral into Fighting). Armed counter -> Fighting
  // into Ghost: -immune? accuracy drawn?
  await run('C3 counter INTO a Ghost (Fighting immunity)',
    [mon('Machamp', ['counter', 'splash'], { evs: { hp: 252 } })],
    [mon('Gengar', ['shadowball', 'splash'], { ability: 'Levitate', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // C4: the foe's physical hit lands on the counter user's SUBSTITUTE — recorded?
  await run('C4 counter: foe physical hit absorbed by OUR sub — recorded?',
    [mon('Snorlax', ['counter', 'substitute', 'splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['drillpeck', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 2' },              // sub up (skarm splashes)
     { p1: 'move 1', p2: 'move 1', stop: true }]); // drill peck into the sub -> counter?

  // C5a: bare HIDDENPOWER (all-31 IVs -> HP Dark 70, SPECIAL in gen3) — counter's
  // `effect.id === 'hiddenpower'` special case: countered DESPITE being special?
  // Then mirrorcoat vs the same bare HP: EXCLUDED (`id !== 'hiddenpower'`) -> fail?
  await run('C5a bare hiddenpower (Dark/special): counter YES? mirrorcoat NO?',
    [mon('Snorlax', ['counter', 'mirrorcoat', 'splash'], { evs: { hp: 252 } })],
    [mon('Alakazam', ['hiddenpower', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },              // counter vs bare HP
     { p1: 'move 2', p2: 'move 1', stop: true }]); // mirrorcoat vs bare HP

  // C5b: the TYPED HP move ids (hiddenpowerfighting = Physical, hiddenpowerice = Special) —
  // these have their OWN ids (NOT 'hiddenpower'): category rules only?
  await run('C5b typed HP ids: counter vs hp-ice (fail?) / mirrorcoat vs hp-ice (2x?) / counter vs hp-fighting (2x?)',
    [mon('Snorlax', ['counter', 'mirrorcoat', 'splash'], { evs: { hp: 252 } })],
    [mon('Alakazam', ['hiddenpowerice', 'hiddenpowerfighting', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },              // counter vs hp-ice (special id) -> fail?
     { p1: 'move 2', p2: 'move 1' },              // mirrorcoat vs hp-ice -> 2x?
     { p1: 'move 1', p2: 'move 2', stop: true }]); // counter vs hp-fighting (physical id) -> 2x?

  // ------------------------------------------------------------- MIRROR COAT
  // M1: baseline 2x a special hit; physical fail; no-damage fail.
  await run('M1 mirrorcoat: special 2x / physical fail / no-damage fail',
    [mon('Blissey', ['mirrorcoat', 'splash'], { evs: { hp: 252 } })],
    [mon('Alakazam', ['psychic', 'return', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },              // psychic -> 2x
     { p1: 'move 1', p2: 'move 2' },              // return (physical) -> fail
     { p1: 'move 1', p2: 'move 3', stop: true }]); // splash -> fail

  // M2: mirror coat INTO a Dark type (gen3 Dark is Psychic-immune). Tyranitar Crunch is
  // Dark = SPECIAL in gen3, so it arms mirrorcoat; the return fire is immune?
  await run('M2 mirrorcoat INTO a Dark (Psychic immunity)',
    [mon('Blissey', ['mirrorcoat', 'splash'], { evs: { hp: 252 } })],
    [mon('Tyranitar', ['crunch', 'splash'], { ability: 'No Ability', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // CM: an equal-speed COUNTER MIRROR (both fail, no damage) vs a both-splash control —
  // isolates the beforeTurnMove order-5 tie + the duration-1 residual handler tie draws.
  await run('CM counter mirror at an equal speed (draw count vs splash control)',
    [mon('Snorlax', ['counter', 'splash'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['counter', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 2' },              // both splash: the control draw count
     { p1: 'move 1', p2: 'move 1', stop: true }]); // both counter: the mirror draw count

  // CT: a LANDED counter at an EQUAL-SPEED board — does the landed counter fire the
  // in-tryMoveHit eachEvent('Update') shuffle (landed=true), like other landed moves?
  // Queue: p1 BeforeTurnMove(order 5, alone) + p2 return(prio 0) + p1 counter(prio -5)
  // — no action-order ties. Count the eachEvent shuffles.
  await run('CT landed counter at an equal-speed tie (in-tryMoveHit Update?)',
    [mon('Snorlax', ['counter', 'splash'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['return', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // CS: counter vs a FIXED-DAMAGE physical hit (Seismic Toss: category Physical, bp 0,
  // damageCallback = level). Recorded (2x100)?
  await run('CS counter vs Seismic Toss (fixed-damage, category Physical)',
    [mon('Snorlax', ['counter', 'splash'], { evs: { hp: 252 } })],
    [mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // ---------------------------------------------------------------- ENDEAVOR
  // E1: baseline (sets target hp to user hp) then the EQUALITY fail on the next turn.
  await run('E1 endeavor: baseline delta then hp-equality fail',
    [mon('Swellow', ['endeavor', 'splash'])],
    [mon('Snorlax', ['splash', 'drillpeck'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },              // user hp 50 vs full Snorlax -> hp = 50
     { p1: 'move 1', p2: 'move 1', stop: true }],  // 50 vs 50 -> fail (draws?)
    { acts: [{ side: 0, hp: 50 }] });

  // E2: target BELOW the user -> fail; which line + draws?
  await run('E2 endeavor: target.hp < user.hp fail',
    [mon('Swellow', ['endeavor', 'splash'])],
    [mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 1, hp: 40 }] });

  // E3: endeavor into a GHOST (Normal immunity) — accuracy drawn then -immune?
  await run('E3 endeavor INTO a Ghost (Normal immunity)',
    [mon('Swellow', ['endeavor', 'splash'])],
    [mon('Gengar', ['splash'], { ability: 'Levitate', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 0, hp: 50 }] });

  // E4: endeavor into a SUBSTITUTE — the callback reads target.hp (the MON), where does the
  // number land (sub absorb / break, carry?)?
  await run('E4 endeavor into a substitute',
    [mon('Swellow', ['endeavor', 'splash'])],
    [mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' },              // snorlax subs (swellow splashes)
     { p1: 'move 1', p2: 'move 2', stop: true }],  // endeavor vs the subbed snorlax
    { acts: [{ side: 0, hp: 50 }] });

  // E5: endeavor KO probe — target hp above user, user hp 1: delta = target.hp - 1 (never a KO).
  await run('E5 endeavor at user hp 1 (max delta, target left at 1)',
    [mon('Swellow', ['endeavor', 'splash'])],
    [mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 0, hp: 1 }] });
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
