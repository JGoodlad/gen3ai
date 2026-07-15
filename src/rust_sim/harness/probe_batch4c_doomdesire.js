// probe_batch4c_doomdesire.js — ground-truth DOOM DESIRE (id `doomdesire`) + its sibling
// FUTURE SIGHT (id `futuresight`) bit-for-bit vs the OMNISCIENT in-process gen3 BattleStream
// (no server). The gen3 FUTURE-move class: announced now, the strike lands at the end of a
// LATER turn against the target SLOT (the Wish slot-condition precedent, but a damaging strike).
//
// The resolved gen3 source (dumped below) says:
//   - onTry: addSlotCondition('futuremove') + getDamage(source, target, {bp 120/80, cat
//     Physical/Special, willCrit:false, type:'???'}, true) AT CAST — the DAMAGE NUMBER is
//     SNAPSHOT at cast (typeless, no crit), stored in the slot condition's moveData together
//     with accuracy 85/90 and basePower 0.
//   - condition: onResidualOrder 11, onStart sets endingTurn = turn-1+2; onResidual removes the
//     slot condition at endingTurn -> onEnd -> trySpreadMoveHit([target], source, hitMove) with
//     the STORED damage; target.removeVolatile('Protect'/'Endure') first; skips if the slot
//     occupant is fainted (or === source).
// The mod chain is the ONLY oracle — probe the exact:
//   1. the CAST-turn draw model (accuracy at cast? the getDamage random(16) at cast? crit?)
//   2. the RESOLVE-turn draw model (accuracy rolled at resolve? no crit? no damage roll?)
//      + WHICH turn the resolve lands on (end of N+1 or N+2 — the endingTurn off-by-one).
//   3. stats snapshot: a boost BETWEEN cast and resolve must NOT change the damage (attacker
//      Calm Mind / defender Amnesia controls at the same seed).
//   4. the residual ORDER vs Wish(7) / sand(8) / Leftovers(10.4) — order 11 = LAST?
//   5. slot semantics: target switches (hits the new occupant at the OLD-computed damage?),
//      caster switches / faints (still resolves?), double-cast fails (draw count?),
//      resolve into a Substitute / a Protect / a "would-be-immune" type (typeless).
//   6. Future Sight — the same mechanic at bp 80 / acc 90 / Special?
//   7. miss + crit sweeps at the resolve (acc 85/90 realized; crits expected ZERO).
//
// Run:  node src/rust_sim/harness/probe_batch4c_doomdesire.js
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
  for (const id of ['doomdesire', 'futuresight']) {
    const m = d.moves.get(id);
    console.log(`=== resolved gen3 ${id} ===`);
    console.log(`  cat=${m.category} bp=${m.basePower} acc=${JSON.stringify(m.accuracy)} type=${m.type} target=${m.target} ` +
      `priority=${m.priority} flags=${JSON.stringify(m.flags)} critRatio=${m.critRatio} willCrit=${m.willCrit} ignoreImmunity=${JSON.stringify(m.ignoreImmunity)}`);
    if (m.onTry) console.log(`  onTry src: ${m.onTry.toString().replace(/\s+/g, ' ')}`);
  }
  const c = d.conditions.get('futuremove');
  console.log(`=== resolved gen3 futuremove condition ===`);
  console.log(`  onResidualOrder=${c.onResidualOrder} duration=${c.duration} noCopy=${c.noCopy} affectsFainted=${c.affectsFainted}`);
  console.log(`  onStart src: ${c.onStart.toString().replace(/\s+/g, ' ')}`);
  console.log(`  onResidual src: ${c.onResidual.toString().replace(/\s+/g, ' ')}`);
  console.log(`  onEnd src: ${c.onEnd.toString().replace(/\s+/g, ' ')}`);
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

const LINE_RE = /\|move\||-damage|-heal|-boost|-unboost|-fail|-immune|-crit|-supereffective|-resisted|\|cant\||-activate|-hitcount|-end\b|-start\b|switch|drag|faint|-miss|-sideend|-sidestart|-weather|-status|-singleturn|-enditem|turn\|/;

async function run(label, p1team, p2team, plan, opts = {}) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  const seed = opts.seed || [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  for (const inj of (opts.acts || [])) {
    const side = battle.sides[inj.side];
    const m = inj.slot === undefined ? side.active[0] : side.pokemon[inj.slot];
    if (!m) continue;
    if (inj.status) m.setStatus(inj.status, m, null, true);
    if (inj.hp !== undefined) m.hp = inj.hp;
  }
  if (!opts.quiet) console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  const LRE = opts.raw ? /./ : LINE_RE;

  let draws = [];
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { const v = realNext(...a); draws.push(drawLabel()); return v; };

  let i = 0, safety = 0;
  while (!battle.ended && safety < 16) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    if (i >= plan.length) break;
    draws = [];
    const logLen0 = log.length;
    const before = battle.prng.getSeed();
    const entry = plan[i]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    const after = battle.prng.getSeed();
    if (!opts.quiet) {
      const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
      const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'} vols=[${Object.keys(m.volatiles).join(',')}] boosts=${JSON.stringify(m.boosts)}` : '-';
      const sc = (s) => JSON.stringify(Object.keys(battle.sides[s].slotConditions[0] || {}));
      console.log(`  [${rs}] ${JSON.stringify(entry)} draws=${draws.length}  seed ${before}->${after}`);
      console.log(`        p1=${fmt(a0)} slotCond=${sc(0)}`);
      console.log(`        p2=${fmt(a1)} slotCond=${sc(1)}`);
      draws.forEach((dl, k) => console.log(`        DRAW[${k}] ${dl}`));
      for (const l of log.slice(logLen0)) if (LRE.test(l)) console.log(`        LINE ${l}`);
      if (opts.pp) {
        const ms = battle.sides[0].pokemon[0].moveSlots.map((m) => `${m.id}:${m.pp}/${m.maxpp}`).join(' ');
        console.log(`        P1[0] PP: ${ms}`);
      }
    }
    if (entry.stop) break;
  }
  if (!opts.quiet) console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  const out = { log, battle };
  try { streams.omniscient.destroy(); } catch (e) {}
  return out;
}

// Sweep the baseline scenario across seeds; tally resolve hit/miss/crit + the damage dealt.
async function sweep(label, moveId, nSeeds) {
  let hit = 0, miss = 0, crit = 0, resolved = 0;
  const damages = new Set();
  for (let s = 0; s < nSeeds; s++) {
    const { log } = await run('sweep',
      [mon('Jirachi', [moveId, 'splash'])],
      [mon('Blissey', ['splash'], { evs: { hp: 252 } })],
      [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1', stop: true }],
      { quiet: true, seed: [s + 1, s * 3 + 5, s * 7 + 11, s * 11 + 13] });
    // find the -end futuremove line, then look at what follows
    const endIdx = log.findIndex((l) => l.startsWith('|-end|') && /Doom Desire|Future Sight/.test(l));
    if (endIdx < 0) continue;
    resolved++;
    const tail = log.slice(endIdx + 1, endIdx + 4).join('\n');
    if (/\|-miss\|/.test(tail)) miss++;
    else if (/\|-damage\|/.test(tail)) {
      hit++;
      const dm = tail.match(/\|-damage\|p2a: Blissey\|(\d+)\/(\d+)/);
      if (dm) damages.add(714 - Number(dm[1]) >= 0 ? String(Number(dm[2]) - Number(dm[1])) : '?');
    }
    if (/\|-crit\|/.test(tail)) crit++;
  }
  console.log(`\n=== SWEEP ${label}: ${nSeeds} seeds ===`);
  console.log(`  resolved=${resolved} hit=${hit} miss=${miss} (rate=${(hit / Math.max(1, hit + miss)).toFixed(3)}) crit=${crit}`);
  console.log(`  distinct hp-losses on Blissey at the resolve: ${[...damages].sort((a, b) => a - b).join(',')}`);
}

async function main() {
  dumpResolved();

  // 1) BASELINE — Doom Desire cast turn / idle turn / resolve turn: the per-turn draw model,
  //    WHICH turn resolves (N+1 vs N+2), and the -start/-end line shapes.
  await run('DD baseline: cast t1, splash t2, splash t3 (which end-of-turn resolves? draws per turn?)',
    [mon('Jirachi', ['doomdesire', 'splash'])],
    [mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1', stop: true }]);

  // 1b) CONTROL — splash/splash turn 1 at the SAME seed: diff the cast-turn draw count.
  await run('CONTROL: splash both, same seed (subtract to get the DD cast draws)',
    [mon('Jirachi', ['doomdesire', 'splash'])],
    [mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1', stop: true }]);

  // 2) STAT SNAPSHOT (attacker): Future Sight, then Calm Mind between cast and resolve vs a
  //    splash control at the SAME seed — equal resolve damage proves the cast-time snapshot.
  await run('FS + CALM MIND after cast (snapshot? damage must equal the control)',
    [mon('Jirachi', ['futuresight', 'calmmind', 'splash'])],
    [mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 3', p2: 'move 1', stop: true }]);
  await run('FS control: splash instead of Calm Mind (same seed)',
    [mon('Jirachi', ['futuresight', 'calmmind', 'splash'])],
    [mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 3', p2: 'move 1' },
     { p1: 'move 3', p2: 'move 1', stop: true }]);

  // 2b) STAT SNAPSHOT (defender): the target Amnesias (+2 SpD) after the FS cast.
  await run('FS + target AMNESIA after cast (defender snapshot?)',
    [mon('Jirachi', ['futuresight', 'splash'])],
    [mon('Blissey', ['splash', 'amnesia'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 2' },
     { p1: 'move 2', p2: 'move 1', stop: true }]);

  // 3) TARGET SWITCH: cast DD at Skarmory (huge Def), p2 switches to Blissey (tiny Def) before
  //    the resolve — does Blissey take the SKARMORY-computed damage (slot semantics + snapshot)?
  await run('DD cast at SKARMORY, Blissey switches in before the resolve (stored damage vs new occupant)',
    [mon('Jirachi', ['doomdesire', 'splash'])],
    [mon('Skarmory', ['splash'], { evs: { hp: 252 } }),
     mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'switch 2' },
     { p1: 'move 2', p2: 'move 1', stop: true }]);
  await run('DD control: cast DIRECTLY at Blissey (same seed) — the big-damage reference',
    [mon('Jirachi', ['doomdesire', 'splash'])],
    [mon('Blissey', ['splash'], { evs: { hp: 252 } }),
     mon('Skarmory', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1', stop: true }]);

  // 4) CASTER SWITCHES OUT: Jirachi casts then pivots out — does the strike still resolve
  //    (and from the benched caster)?
  await run('DD then the CASTER SWITCHES OUT before the resolve',
    [mon('Jirachi', ['doomdesire', 'splash']),
     mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'switch 2', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1', stop: true }]);

  // 5) CASTER FAINTS: Jirachi casts, is KO'd on the idle turn — does the strike still resolve?
  await run('DD then the CASTER is KO\'d before the resolve (still strikes?)',
    [mon('Jirachi', ['doomdesire', 'splash']),
     mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [mon('Blissey', ['seismictoss'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },      // Seismic Toss 100 KOs the injected-1hp... see acts
     { p1: 'switch 2', p2: 'move 1' },    // forced replacement
     { p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 0, slot: 0, hp: 90 }] });

  // 6) DOUBLE-CAST: a 2nd DD while one is pending — fails? draw count?
  await run('DD DOUBLE-CAST (2nd while pending): fail shape + draw count',
    [mon('Jirachi', ['doomdesire', 'splash'])],
    [mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1', stop: true }]);

  // 6b) CROSS double: FS pending, then DD cast onto the SAME slot (both are 'futuremove').
  await run('FS pending then DD onto the same slot (one futuremove slot condition?)',
    [mon('Jirachi', ['futuresight', 'doomdesire', 'splash'])],
    [mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 3', p2: 'move 1', stop: true }]);

  // 7) RESOLVE into a SUBSTITUTE: target subs on the idle turn.
  await run('DD resolve into a SUBSTITUTE (sub absorbs the stored damage?)',
    [mon('Jirachi', ['doomdesire', 'splash'])],
    [mon('Blissey', ['splash', 'substitute'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 2' },
     { p1: 'move 2', p2: 'move 1', stop: true }]);

  // 8) RESOLVE through a PROTECT on the resolve turn (onEnd removes the volatile first).
  await run('DD resolve vs PROTECT on the resolve turn (hits through?)',
    [mon('Jirachi', ['doomdesire', 'splash'])],
    [mon('Blissey', ['splash', 'protect'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 2', stop: true }]);

  // 8b) PROTECT on the CAST turn: does Protect block the announce/slot condition?
  await run('PROTECT on the CAST turn (does the cast go through a protecting target?)',
    [mon('Jirachi', ['doomdesire', 'splash'])],
    [mon('Blissey', ['splash', 'protect'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 2' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1', stop: true }]);

  // 9) TYPELESS effectiveness: DD (Steel) into a FIRE type (would resist Steel) — typeless '???'
  //    should be NEUTRAL, no -resisted; and FS ('???') into a DARK type (would be Psychic-immune)
  //    should HIT.
  await run('DD into CHARIZARD (Fire resists Steel — typeless => neutral? no -resisted?)',
    [mon('Jirachi', ['doomdesire', 'splash'])],
    [mon('Charizard', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1', stop: true }]);
  await run('FS into UMBREON (Dark is Psychic-immune — typeless => hits?)',
    [mon('Jirachi', ['futuresight', 'splash'])],
    [mon('Umbreon', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1', stop: true }]);

  // 10) RESIDUAL ORDER: sand (Tyranitar) + Leftovers + a Wish resolving the SAME turn as the DD.
  //     Wish(7) -> sand(8) -> Leftovers(10.4) -> futuremove(11)? Watch the line order at t3 end.
  await run('RESIDUAL ORDER: Wish + sand + Leftovers + DD all at the t3 residual',
    [mon('Jirachi', ['doomdesire', 'wish', 'splash'], { item: 'Leftovers' })],
    [mon('Tyranitar', ['splash'], { ability: 'Sand Stream', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },   // Wish cast t2 -> resolves end of t3, same turn as DD
     { p1: 'move 3', p2: 'move 1', stop: true }]);

  // 11) RESOLVE KO: the strike KOs the slot occupant — faint/replacement protocol + draws.
  await run('DD resolve KOs the target (faint -> forced replacement; Quick Claw drawn?)',
    [mon('Jirachi', ['doomdesire', 'splash'])],
    [mon('Gengar', ['splash'], { ability: 'Levitate' }),
     mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p2: 'switch 2' },
     { p1: 'move 2', p2: 'move 1', stop: true }],
    { acts: [{ side: 1, slot: 0, hp: 60 }] });

  // 12) FUTURE SIGHT baseline (same mechanic? Special bp 80 acc 90).
  await run('FS baseline: cast t1 / splash / resolve',
    [mon('Jirachi', ['futuresight', 'splash'])],
    [mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1', stop: true }]);

  // 13) SWEEPS: resolve-time accuracy realization + crit count (expect ~85%/~90%, crit 0).
  await sweep('DOOM DESIRE (acc 85)', 'doomdesire', 60);
  await sweep('FUTURE SIGHT (acc 90)', 'futuresight', 60);

  // ---------- PART 2: the follow-up gaps ----------

  // 14) RESIDUAL ORDER v2 — OBSERVABLE: p1 Celebi (Grass — takes the sand chip; Leftovers;
  //     damaged) has a Wish resolving the SAME residual the DD strikes p2 Tyranitar.
  //     Expect the line order: Wish(7) -> sand(8) -> Leftovers(10.4) -> futuremove(11).
  await run('RESIDUAL ORDER v2: Wish-heal + sand-chip + Leftovers on p1, DD -end on p2, same residual',
    [mon('Celebi', ['doomdesire', 'wish', 'splash'], { item: 'Leftovers' })],
    [mon('Tyranitar', ['splash'], { ability: 'Sand Stream', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 3', p2: 'move 1', stop: true }],
    { acts: [{ side: 0, slot: 0, hp: 150 }] });

  // 15) CASTER KO'd — FULL: play through to the resolve turn (the caster dead on the bench).
  await run('DD then the CASTER is KO\'d — play to the resolve (strikes from a fainted, benched caster?)',
    [mon('Jirachi', ['doomdesire', 'splash']),
     mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [mon('Blissey', ['seismictoss'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'switch 2' },                 // forced replacement after the t1 KO
     { p1: 'move 1', p2: 'move 1' },     // turn 2 (idle)
     { p1: 'move 1', p2: 'move 1' },     // turn 3 (resolve at its end)
     { p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 0, slot: 0, hp: 90 }] });

  // 16) DOUBLE-CAST raw lines + PP: what exactly does the failed 2nd cast emit ([still]? -fail?)
  //     and is its PP still deducted?
  await run('DD DOUBLE-CAST raw lines + PP bookkeeping',
    [mon('Jirachi', ['doomdesire', 'splash'])],
    [mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1', stop: true }],
    { raw: true, pp: true });

  // 17) MIRROR FS at EQUAL speed: does the PENDING futuremove condition register a residual
  //     handler that TIES (extra shuffle draws) on idle + resolve turns? vs a DISTINCT-speed control.
  await run('FS MIRROR, EQUAL speed (Jirachi vs Jirachi): idle/resolve residual tie draws + -end order',
    [mon('Jirachi', ['futuresight', 'splash'])],
    [mon('Jirachi', ['futuresight', 'splash'])],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 2' },
     { p1: 'move 2', p2: 'move 2', stop: true }]);
  await run('FS BOTH-CAST, DISTINCT speed (Jirachi vs Snorlax): control',
    [mon('Jirachi', ['futuresight', 'splash'])],
    [mon('Snorlax', ['futuresight', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 2' },
     { p1: 'move 2', p2: 'move 2', stop: true }]);

  // 18) A resolve-turn MISS: hunt a missing seed quietly, then show the raw resolve lines.
  for (let s = 0; s < 200; s++) {
    const seed = [s + 1, s * 3 + 5, s * 7 + 11, s * 11 + 13];
    const { log } = await run('hunt', [mon('Jirachi', ['doomdesire', 'splash'])],
      [mon('Blissey', ['splash'], { evs: { hp: 252 } })],
      [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1', stop: true }],
      { quiet: true, seed });
    const endIdx = log.findIndex((l) => l.startsWith('|-end|') && /Doom Desire/.test(l));
    if (endIdx >= 0 && /\|-miss\|/.test(log.slice(endIdx + 1, endIdx + 3).join('\n'))) {
      await run(`DD resolve MISS (seed ${JSON.stringify(seed)}) — raw resolve-turn lines`,
        [mon('Jirachi', ['doomdesire', 'splash'])],
        [mon('Blissey', ['splash'], { evs: { hp: 252 } })],
        [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1', stop: true }],
        { raw: true, seed });
      break;
    }
  }
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
