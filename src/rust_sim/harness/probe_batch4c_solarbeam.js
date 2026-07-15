// probe_batch4c_solarbeam.js — ground-truth SOLAR BEAM (id `solarbeam`) bit-for-bit vs the
// OMNISCIENT in-process BattleStream (no server). Solar Beam is the 2-TURN CHARGE move:
//   turn 1: |-prepare| (no hit), the `twoturnmove` volatile locks the user;
//   turn 2: the move FIRES (accuracy/crit/damage);
//   in SUN it skips the charge and fires immediately.
//
// The mod chain is the ONLY oracle. Probe the exact:
//   1. the CHARGE-turn vs FIRE-turn draw split (accuracy drawn on which turn? does the
//      charge turn draw ANYTHING move-wise? the -prepare / [still] line shapes; PP timing).
//   2. the SUN skip (draw count when it fires immediately; decided by effectiveWeather —
//      does a Cloud Nine foe force the charge back?).
//   3. gen3 rain/sand/hail BP interaction (the resolved onBasePower — do NOT assume the
//      modern halving; compare realized damage rain/sand vs a no-weather control).
//   4. interruptions: user asleep/paralyzed on the fire turn (is the charge lost? does it
//      re-charge after? the volatile's lifecycle), target switches between charge and fire,
//      user hit during the charge.
//   5. the charging mon's REQUEST on the fire turn (locked to solarbeam? trapped? can it
//      switch? what does a rejected switch look like?).
// Also enumerate every other gen3 flags.charge two-turn move (the fail-loud siblings; the
// semi-invulnerable ones — Fly/Dig/Bounce/Dive — DIFFER via their own condition).
//
// Run:  node src/rust_sim/harness/probe_batch4c_solarbeam.js
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

function srcOf(fn) { return fn ? fn.toString().replace(/\s+/g, ' ') : 'undefined'; }

function dumpResolved() {
  const d = Dex.forFormat(FORMAT);
  const m = d.moves.get('solarbeam');
  console.log('=== resolved gen3 solarbeam ===');
  console.log(`  cat=${m.category} bp=${m.basePower} acc=${m.accuracy} type=${m.type} target=${m.target} ` +
    `priority=${m.priority} flags=${JSON.stringify(m.flags)} critRatio=${m.critRatio}`);
  console.log(`  secondary=${JSON.stringify(m.secondary)} secondaries=${JSON.stringify(m.secondaries)}`);
  for (const k of ['onTryMove', 'onTry', 'onBasePower', 'onModifyMove', 'onPrepareHit', 'onHit',
    'basePowerCallback', 'beforeTurnCallback', 'priorityChargeCallback']) {
    if (m[k]) console.log(`  ${k} src: ${srcOf(m[k])}`);
  }
  if (m.condition) console.log(`  condition: ${JSON.stringify(Object.keys(m.condition))}`);
  // The twoturnmove condition (the charge-lock volatile) — the resolved gen3 shape.
  const tt = d.conditions.get('twoturnmove');
  console.log('=== resolved gen3 twoturnmove condition ===');
  console.log(`  duration=${tt.duration}`);
  for (const k of ['onStart', 'onEnd', 'onLockMove', 'onMoveAborted', 'durationCallback']) {
    if (tt[k]) console.log(`  ${k} src: ${srcOf(tt[k])}`);
  }
  // Cross-gen inheritance check.
  for (const g of ['gen3', 'gen4', 'gen5']) {
    const dm = Dex.mod(g).moves.get('solarbeam');
    console.log(`  [${g}] bp=${dm.basePower} acc=${dm.accuracy} onTryMove=${!!dm.onTryMove} onBasePower=${!!dm.onBasePower}` +
      ` onBasePowerSrc=${dm.onBasePower ? srcOf(dm.onBasePower) : '-'}`);
  }
  // Enumerate EVERY gen3 charge move (the fail-loud sibling list).
  console.log('=== gen3 flags.charge two-turn movers (the sibling class) ===');
  const g3 = Dex.mod('gen3');
  for (const raw of g3.moves.all()) {
    if (!raw.exists || raw.isNonstandard) continue;
    if (raw.gen > 3) continue;
    if (!raw.flags || !raw.flags.charge) continue;
    const cond = raw.condition || {};
    const semiInvuln = !!(cond.onInvulnerability !== undefined || cond.onSourceModifyDamage ||
      (raw.condition && srcOf(cond.onInvulnerability).length > 12));
    console.log(`  ${raw.id}: bp=${raw.basePower} acc=${raw.accuracy} type=${raw.type} cat=${raw.category}` +
      ` ownCondition=${!!raw.condition}${raw.condition ? ' condKeys=' + JSON.stringify(Object.keys(raw.condition)) : ''}` +
      ` semiInvulnerable=${semiInvuln} onTryMoveSrc=${raw.onTryMove ? srcOf(raw.onTryMove).slice(0, 220) : '-'}`);
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

async function run(label, p1team, p2team, plan, inject) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  // Also capture the per-side p1 stream so the |request| / |error| frames are visible verbatim.
  const p1log = [];
  (async () => { for await (const ch of streams.p1) { for (const l of ch.split('\n')) if (l) p1log.push(l); } })();
  const seed = (inject && inject.seed) || [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  for (const inj of ((inject && inject.acts) || [])) {
    const side = battle.sides[inj.side];
    const m = inj.slot === undefined ? side.active[0] : side.pokemon[inj.slot];
    if (!m) continue;
    if (inj.status) m.setStatus(inj.status, m, null, true);
    if (inj.hp !== undefined) m.hp = inj.hp;
    if (inj.faint) { m.hp = 0; m.fainted = true; }
    if (inj.item !== undefined) m.item = inj.item;
  }

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);

  let draws = [];
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { const v = realNext(...a); draws.push(drawLabel()); return v; };

  const fmtReq = (side) => {
    const r = side.activeRequest;
    if (!r) return 'null';
    if (r.wait) return 'wait';
    if (r.forceSwitch) return `forceSwitch=${JSON.stringify(r.forceSwitch)}`;
    if (!r.active) return JSON.stringify(r).slice(0, 200);
    const a = r.active[0];
    return `moves=${JSON.stringify(a.moves)}${a.trapped ? ' trapped=true' : ''}${a.maybeTrapped ? ' maybeTrapped=true' : ''}` +
      `${a.canSwitch !== undefined ? ' canSwitch=' + JSON.stringify(a.canSwitch) : ''}`;
  };
  const fmtPP = (m) => m ? m.moveSlots.map((s) => `${s.id}:${s.pp}/${s.maxpp}${s.disabled ? '(dis)' : ''}`).join(' ') : '-';
  const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'} vols=[${Object.keys(m.volatiles).join(',')}]` : '-';

  let i = 0, safety = 0;
  while (!battle.ended && safety < 24 && i < plan.length) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    draws = [];
    const logLen0 = log.length;
    const p1LogLen0 = p1log.length;
    const before = battle.prng.getSeed();
    // Print the OPEN request BEFORE submitting (what the charging mon is offered).
    console.log(`  [request:${rs}] p1req: ${fmtReq(battle.sides[0])}`);
    console.log(`                 p2req: ${fmtReq(battle.sides[1])}`);
    const entry = plan[i]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    console.log(`  [${rs}] ${JSON.stringify(entry)} draws=${draws.length}  seed ${before}->${after}`);
    console.log(`        p1=${fmt(a0)}  pp[${fmtPP(a0)}]`);
    console.log(`        p2=${fmt(a1)}  pp[${fmtPP(a1)}]`);
    console.log(`        weather=${battle.field.weather || 'none'}`);
    draws.forEach((dl, k) => console.log(`        DRAW[${k}] ${dl}`));
    const newLines = log.slice(logLen0).filter((l) =>
      /\|move\||-damage|-heal|-boost|-unboost|-fail|-immune|-crit|-supereffective|-resisted|cant|-activate|-prepare|-anim|-end\b|-start|switch|drag|faint|-status|-curestatus|-weather|error/.test(l));
    for (const l of newLines) console.log(`        LINE ${l}`);
    const p1new = p1log.slice(p1LogLen0).filter((l) => /error/.test(l));
    for (const l of p1new) console.log(`        P1SIDE ${l}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  dumpResolved();

  // ---------------------------------------------------------------------------------
  // 1) BASELINE charge + fire, distinct speeds. Venusaur (spe 80) solarbeams a slower
  //    Swampert (spe 60) that Surfs back — so the user is HIT DURING THE CHARGE (does the
  //    charge survive a hit?). Capture: the charge turn's draw count + line shape
  //    (-prepare / [still]), PP timing (deducted on charge or fire?), the volatile keys,
  //    the FIRE-turn request (locked? trapped?), and the fire turn's draw count.
  await run('SOLARBEAM baseline: charge turn then fire turn (user hit mid-charge)',
    [mon('Venusaur', ['solarbeam', 'razorleaf']),
     mon('Snorlax', ['bodyslam'])],
    [mon('Swampert', ['surf'], { evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },   // charge (Venusaur faster: prepare, then Surf hits it)
     { p1: 'move 1', p2: 'move 1' },   // fire (KOs Swampert → the twoturnmove volatile LINGERS through the faint pause)
     { p2: 'switch 2' },               // the replacement — then: is p1's next request still locked?
     { p1: 'move 1', p2: 'move 1' },   // what does the lingering twoturnmove do here? (re-charge? insta-fire?)
     { p1: 'move 1', p2: 'move 1', stop: true }]);

  // 2) FIRE-turn ILLEGAL choices: on the fire turn try `switch 2` FIRST (rejected? what
  //    error?), then `move 2` (razorleaf — rejected? the lock), then the accepted `move 1`.
  await run('SOLARBEAM fire-turn legality: try switch + another move while locked',
    [mon('Venusaur', ['solarbeam', 'razorleaf']),
     mon('Snorlax', ['bodyslam'])],
    [mon('Swampert', ['surf'], { evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },        // charge
     { p1: 'switch 2', p2: 'move 1' },      // fire turn: attempt a switch (expect reject)
     { p1: 'move 2' },                      // attempt the OTHER move (expect reject or auto-solarbeam?)
     { p1: 'move 1', stop: true }]);        // the legal locked choice

  // 3) SUN SKIP: p2 Groudon (Drought) → permanent sun. Solar Beam should fire IMMEDIATELY
  //    (no -prepare, no volatile). Count the draws (acc + crit + dmg like a normal move?).
  await run('SOLARBEAM in SUN (Drought): the charge is skipped',
    [mon('Venusaur', ['solarbeam'])],
    [mon('Groudon', ['earthquake'], { ability: 'Drought', evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1', stop: true }]);

  // 4) CLOUD NINE vs the sun skip: sun is UP (p1's own Groudon partner? — simpler: the
  //    USER is Groudon-Drought itself) but the FOE has Cloud Nine. If the skip is decided
  //    by effectiveWeather() the suppression forces a normal CHARGE.
  await run('SOLARBEAM in sun but a CLOUD NINE foe (effectiveWeather suppressed): charge or skip?',
    [mon('Groudon', ['solarbeam'], { ability: 'Drought' })],
    [mon('Golduck', ['watergun'], { ability: 'Cloud Nine', evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1', stop: true }]);

  // 5) RAIN BP: p2 Kyogre (Drizzle) → permanent rain. Solar Beam must CHARGE (no sun) and
  //    the fire-turn damage shows whether gen3 halves BP in rain (compare vs baseline-class
  //    damage on a similar bulk target). Also: does the fire turn draw differently? (No —
  //    BP mod is draw-free.)
  await run('SOLARBEAM in RAIN (Drizzle): charge + fire, is BP halved?',
    [mon('Venusaur', ['solarbeam'])],
    [mon('Kyogre', ['calmmind'], { ability: 'Drizzle', evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1', stop: true }]);

  // 5b) A no-weather CONTROL on the SAME target (Kyogre, ability swapped) so the rain
  //     halving is read off the damage delta directly.
  await run('SOLARBEAM no-weather CONTROL on the same Kyogre target',
    [mon('Venusaur', ['solarbeam'])],
    [mon('Kyogre', ['calmmind'], { ability: 'Shell Armor', evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1', stop: true }]);

  // 5c) SANDSTORM: p2 Tyranitar (Sand Stream). Charge + fire; is BP halved in sand?
  //     (Venusaur is also sand-chipped — Grass is not immune — the residual shows.)
  await run('SOLARBEAM in SAND (Sand Stream): charge + fire, is BP halved?',
    [mon('Venusaur', ['solarbeam'])],
    [mon('Tyranitar', ['curse'], { ability: 'Sand Stream', evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1', stop: true }]);

  // 6) TARGET SWITCHES between charge and fire: p2 switches on the fire turn (a switch
  //    resolves before the move) — Solar Beam hits the ENTRANT.
  await run('SOLARBEAM target switches on the fire turn (hits the entrant?)',
    [mon('Venusaur', ['solarbeam'])],
    [mon('Swampert', ['surf'], { evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'switch 2' },
     { p1: 'move 1', p2: 'move 1', stop: true }]);

  // 7) ASLEEP ON THE FIRE TURN: a FASTER Breloom (spe 70) Spores the slower charging
  //    Exeggutor (spe 55) on its fire turn — the onBeforeMove sleep abort. Is the charge
  //    LOST? Does the volatile survive? What happens when it wakes (re-charge or fire)?
  await run('SOLARBEAM user put to SLEEP on the fire turn (charge lost? re-charge on wake?)',
    [mon('Exeggutor', ['solarbeam'])],
    [mon('Breloom', ['machpunch', 'spore'], { evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },            // charge (Breloom mach-punches)
     { p1: 'move 1', p2: 'move 2' },            // fire turn: Breloom (faster) Spores first → cant
     { p1: 'move 1', p2: 'move 1' },            // asleep / wake?
     { p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1', stop: true }]);

  // 8) PARALYZED sweep: user paralyzed at start (injected). The para roll fires on BOTH
  //    the charge turn and the fire turn (onBeforeMove precedes onTryMove). Sweep seeds to
  //    realize a full-para on the CHARGE turn and on the FIRE turn; observe the volatile's
  //    fate + whether the next turn re-charges.
  for (const seed of [[7, 11, 13, 17], [1, 2, 3, 4], [42, 42, 42, 42], [9, 99, 999, 9999], [5, 6, 7, 8]]) {
    await run(`SOLARBEAM user PARALYZED (injected) seed=${JSON.stringify(seed)}: full-para on charge/fire turns`,
      [mon('Venusaur', ['solarbeam'])],
      [mon('Swampert', ['curse'], { evs: { hp: 252 } }),
       mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
      [{ p1: 'move 1', p2: 'move 1' },
       { p1: 'move 1', p2: 'move 1' },
       { p1: 'move 1', p2: 'move 1' },
       { p1: 'move 1', p2: 'move 1', stop: true }],
      { seed, acts: [{ side: 0, slot: 0, status: 'par' }] });
  }

  // 9) PP + PRESSURE: does the fire turn deduct PP (or the charge turn)? And under a
  //    PRESSURE foe, is the deduction -2, and on WHICH turn?
  await run('SOLARBEAM vs a PRESSURE foe (PP timing + the -2)',
    [mon('Venusaur', ['solarbeam'])],
    [mon('Zapdos', ['agility'], { ability: 'Pressure', evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1', stop: true }]);

  // 10) FIRE TURN INTO A PROTECT: the foe Protects on the fire turn — accuracy drawn then
  //     blocked (like any move)? Is the charge consumed (no re-fire next turn)?
  await run('SOLARBEAM fire turn into a PROTECT (charge consumed?)',
    [mon('Venusaur', ['solarbeam'])],
    [mon('Swampert', ['protect', 'surf'], { evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 2' },   // charge; Swampert surfs
     { p1: 'move 1', p2: 'move 1' },   // fire into Protect
     { p1: 'move 1', p2: 'move 2', stop: true }]);  // next turn: re-charge or fire?

  // 11) FIRE TURN INTO AN IMMUNE-ish / SUB: fire into a SUBSTITUTE (normal sub absorb?)
  await run('SOLARBEAM fire turn into a SUBSTITUTE',
    [mon('Venusaur', ['solarbeam'])],
    [mon('Snorlax', ['substitute', 'bodyslam'], { evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },   // Venusaur charges; Snorlax subs
     { p1: 'move 1', p2: 'move 2', stop: true }]);  // fire into the sub

  // 12) CHARGE while the FOE SWITCHES (charge turn is unaffected by the target acting).
  //     And: the user KO'd mid-charge — foe KOs the charging mon on the fire turn BEFORE
  //     it moves (faster foe, huge hit) → no fire, replacement.
  await run('SOLARBEAM user KO-d on the fire turn before it moves',
    [mon('Exeggutor', ['solarbeam'], { ivs: { ...IV31, hp: 0 } }),
     mon('Snorlax', ['bodyslam'])],
    [mon('Gengar', ['icepunch', 'shadowball'], { ability: 'Levitate', evs: { spa: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },   // charge; Gengar ice-punches
     { p1: 'move 1', p2: 'move 1' },   // Gengar (faster) hits; if KO → no fire
     { p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1', stop: true }]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
