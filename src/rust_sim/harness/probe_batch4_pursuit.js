// probe_batch4_pursuit.js — ground-truth PURSUIT (id `pursuit`) bit-for-bit vs the
// OMNISCIENT in-process BattleStream (no server). Pursuit is the SWITCH-INTERRUPT move:
//   - base BP 40 Dark (SPECIAL in gen3), acc 100, contact/protect/mirror flags
//   - basePowerCallback DOUBLES to 80 when target.beingCalledBack || target.switchFlag
//   - onModifyMove sets accuracy=true (never-miss) on that same interrupt condition
//   - beforeTurnCallback lays a `pursuit` volatile on the TARGET (duration 1) recording
//     the pursuers in effectState.sources
//   - the volatile's onBeforeSwitchOut is the INTERRUPT: when the target voluntarily
//     switches out, it CANCELS each source's queued Pursuit and immediately runs Pursuit
//     against the switching mon BEFORE the switch resolves.
//
// The mod chain is the ONLY oracle. Probe the exact:
//   1. action-reordering when the FOE switches + we Pursuit — where the strike inserts,
//      what draws fire at the switch boundary (acc? crit? dmg?), and whether the switch's
//      own runSwitch / entry-hazard sequence follows.
//   2. the ×2 basePower trigger (beingCalledBack / switchFlag) + never-miss-on-interrupt;
//      and that NORMAL (foe stays) Pursuit is a plain bp-40 hit with the ordinary draw model.
//   3. edge cases: Pursuit KOs the switcher (does the replacement still come in?),
//      FORCED switch (Roar/Whirlwind) vs a CHOSEN switch, Ghost target (Dark hits Ghost),
//      Substitute, target faster / slower.
//
// Run:  node src/rust_sim/harness/probe_batch4_pursuit.js
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
  const m = d.moves.get('pursuit');
  console.log('=== resolved gen3 pursuit ===');
  console.log(`  cat=${m.category} bp=${m.basePower} acc=${m.accuracy} type=${m.type} target=${m.target} ` +
    `priority=${m.priority} flags=${JSON.stringify(m.flags)}`);
  console.log(`  basePowerCallback src: ${m.basePowerCallback.toString().replace(/\s+/g, ' ')}`);
  console.log(`  onModifyMove src: ${m.onModifyMove.toString().replace(/\s+/g, ' ')}`);
  console.log(`  beforeTurnCallback src: ${m.beforeTurnCallback.toString().replace(/\s+/g, ' ')}`);
  const c = m.condition;
  console.log(`  condition.duration=${c.duration}`);
  console.log(`  condition.onBeforeSwitchOut src: ${c.onBeforeSwitchOut.toString().replace(/\s+/g, ' ')}`);
}

// A short call-site label from the PRNG-draw stack.
function drawLabel() {
  const st = new Error().stack.split('\n');
  // skip: Error, drawLabel, the rng.next wrapper -> first meaningful frame at idx 3
  const frames = [];
  for (let i = 3; i < st.length && frames.length < 3; i++) {
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
  const seed = (inject && inject.seed) || [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  for (const inj of ((inject && inject.acts) || [])) {
    const m = inj.side === undefined ? null : battle.sides[inj.side].active[0];
    if (!m) continue;
    if (inj.status) m.setStatus(inj.status, m, null, true);
    if (inj.hp !== undefined) m.hp = inj.hp;
    if (inj.item !== undefined) m.item = inj.item;
  }

  let draws = [];
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { const v = realNext(...a); draws.push(drawLabel()); return v; };

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 6) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    draws = [];
    const logLen0 = log.length;
    const before = battle.prng.getSeed();
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'} vols=[${Object.keys(m.volatiles).join(',')}]` : '-';
    console.log(`  [${rs}] ${JSON.stringify(entry)} draws=${draws.length}  seed ${before}->${after}`);
    console.log(`        p1=${fmt(a0)}`);
    console.log(`        p2=${fmt(a1)}`);
    draws.forEach((dl, k) => console.log(`        DRAW[${k}] ${dl}`));
    const newLines = log.slice(logLen0).filter((l) =>
      /\|move\||-damage|-heal|-boost|-unboost|-fail|-immune|-crit|-supereffective|-resisted|cant|-activate|switch|drag|faint|-start|-end/.test(l));
    for (const l of newLines) console.log(`        LINE ${l}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  dumpResolved();

  // 1) FOE SWITCHES + we PURSUIT — the interrupt. p1 Pursuit, p2 switches its active out.
  //    Pursuit should strike the SWITCHING mon (×2 BP, never-miss) BEFORE the switch resolves,
  //    then the replacement comes in.
  await run('PURSUIT interrupt: p2 switches, p1 Pursuits (×2, never-miss, strikes before switch)',
    [mon('Tyranitar', ['pursuit', 'crunch'], { ability: 'Sand Stream', evs: { spa: 252 } })],
    [mon('Jolteon', ['thunderbolt'], { evs: { spe: 252 } }),
     mon('Snorlax', ['bodyslam'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'switch 2', stop: true }]);

  // 2) NORMAL Pursuit — foe STAYS IN and attacks. Plain bp-40 hit + ordinary draw model
  //    (accuracy + crit + damage). No ×2, accuracy roll present.
  await run('PURSUIT normal: foe stays in — plain bp40, acc+crit+dmg draws',
    [mon('Tyranitar', ['pursuit', 'crunch'], { ability: 'Sand Stream', evs: { spa: 252 } })],
    [mon('Snorlax', ['bodyslam'], { evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // 3) Pursuit KOs the SWITCHER — does the replacement still come in afterwards?
  //    Low-HP foe active switches out into Pursuit that KOs it.
  await run('PURSUIT KOs the switcher: does the replacement still come in?',
    [mon('Tyranitar', ['pursuit', 'crunch'], { ability: 'Sand Stream', evs: { spa: 252 } })],
    [mon('Gengar', ['shadowball'], { ability: 'Levitate', evs: { spe: 252 } }),
     mon('Snorlax', ['bodyslam'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'switch 2' }, { p1: 'move 2', p2: 'move 1' }],
    { acts: [{ side: 1, hp: 40 }] });

  // 4) FORCED switch (Roar) vs the pursuer — does Pursuit intercept a PHAZE-forced switch?
  //    p2 Roars p1 (forces p1 to switch); p1 also carries Pursuit but is the one being phazed.
  //    Better: p1 Pursuit, p2 Roar — but Roar forces p1 out, not p2. To test forced switch of
  //    the FOE: p1 Pursuit + p1's ally... singles. Instead: does a mon dragged by Roar trigger
  //    the pursuer's Pursuit? Set p2 to Roar p1 while p1 tries Pursuit — the FORCED-out mon is p1.
  //    We want the PURSUED side forced out. Use p1 Whirlwind on p2 + a Pursuit... can't Pursuit+phaze same turn.
  //    So: p2 has Pursuit, p1 Roars p2 out. Does p2's Pursuit fire on p1? No — p1 isn't switching.
  //    The real question: a mon FORCED out by Roar — is it Pursuit-intercepted? p1 Pursuit, p2 has
  //    NO switch move; we can't force p2 out AND Pursuit in one turn from one side in singles.
  //    Test instead: p1 Pursuit vs p2 that is FORCED out by p1's OWN... no. Use two turns is moot.
  //    Cleanest forced-switch probe: p1 Pursuit, p2 uses a SELF-SWITCH? none in gen3 (no U-turn/Baton on this mon).
  //    So probe the DRAG path directly: give p1 Roar, see it does NOT trigger p2's Pursuit at all
  //    (control), then note gen3 pursuit only intercepts VOLUNTARY (beingCalledBack) switches.
  await run('CONTROL forced switch: p1 Roars p2 (drag), p2 also holds Pursuit — no Pursuit interrupt on a DRAG',
    [mon('Suicune', ['roar', 'surf'], { evs: { hp: 252 } })],
    [mon('Tyranitar', ['pursuit', 'crunch'], { ability: 'Sand Stream', evs: { spa: 252 } }),
     mon('Snorlax', ['bodyslam'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);

  // 5) GHOST target — Pursuit is Dark, hits Ghost (Dark→Ghost super effective in gen3).
  //    Ghost foe switches out into Pursuit.
  await run('PURSUIT into a GHOST switcher: Dark hits Ghost (super effective), ×2 interrupt',
    [mon('Tyranitar', ['pursuit', 'crunch'], { ability: 'Sand Stream', evs: { spa: 252 } })],
    [mon('Gengar', ['shadowball'], { ability: 'Levitate', evs: { spe: 252 } }),
     mon('Misdreavus', ['shadowball'], { ability: 'Levitate', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'switch 2', stop: true }]);

  // 6) SUBSTITUTE — foe has a sub up then switches out. Does the ×2 interrupt hit the SUB or the mon?
  await run('PURSUIT into a switching SUB holder: sub or mon?',
    [mon('Tyranitar', ['pursuit', 'crunch'], { ability: 'Sand Stream', evs: { spa: 252 } })],
    [mon('Snorlax', ['substitute', 'bodyslam'], { evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'switch 2', stop: true }]);

  // 7) TARGET FASTER — the switching foe is FASTER than the pursuer. The interrupt fires at
  //    switch time regardless of speed. Confirm ×2 + no double hit.
  await run('PURSUIT target FASTER: interrupt fires at switch time regardless of speed',
    [mon('Snorlax', ['pursuit', 'bodyslam'], { evs: { spa: 100 } })],
    [mon('Jolteon', ['thunderbolt'], { evs: { spe: 252 } }),
     mon('Aerodactyl', ['rockslide'], { evs: { spe: 252 } })],
    [{ p1: 'move 1', p2: 'switch 2', stop: true }]);

  // 8) The FORCED-out-by-own-move edge is n/a in singles; also probe: foe switches but the
  //    pursuer is SLOWER and would normally move 2nd — the interrupt still fires (already in #1).
  //    Extra: pursuer PARALYZED / does the interrupt still consume its PP + move? Observe PP note.
  await run('PURSUIT interrupt draw-order detail (slow pursuer, fast switcher, clean board)',
    [mon('Blissey', ['pursuit', 'softboiled'], { evs: { hp: 252 } })],
    [mon('Starmie', ['surf'], { evs: { spe: 252 } }),
     mon('Skarmory', ['spikes'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'switch 2', stop: true }]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
