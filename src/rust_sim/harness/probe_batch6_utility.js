// probe_batch6_utility.js — ground-truth GROUP C of move-coverage batch 6 bit-for-bit vs the
// OMNISCIENT in-process gen3 BattleStream (no server):
//   BELLY DRUM / CHARGE / MEMENTO / MIMIC / PAIN SPLIT / PSYCH UP.
//
// Questions this probe settles BEHAVIORALLY (the sim is the only oracle):
//   bellydrum: the exact HP gate (hp <= maxhp/2 fails — float or floor compare? probe the
//     even-maxhp boundary hp == maxhp/2 vs +1), the +6 SET (from any stage, incl. negative),
//     the HP cost floor(maxhp/2), the Atk-already-+6 fail, and the draw model (never-miss —
//     zero draws expected everywhere?).
//   charge: the gen3 effect — the 'charge' volatile that doubles the NEXT Electric move's BP
//     (gen3 has NO +1 SpD — that's gen4+; boosts printed to verify), the volatile's lifetime
//     (duration 2? consumed by ANY move or only an Electric one? survives an idle splash
//     turn?), and the draw model.
//   memento: resolved acc=true (NEVER-miss, no accuracy draw!) + selfdestruct:'ifHit' +
//     protect:1 — does the user faint on a Protect block / a Substitute block / a Clear Body
//     boost-immune target? the line ORDER (-unboost before faint?), the draw model, and the
//     forced replacement.
//   mimic: the gen3 copy semantics — which SLOT (the Mimic slot), what PP the copied move
//     gets (probe moveSlots after the copy), the fail conditions (no lastMove /
//     already-known via failmimic), the lifetime (switch out+in → reverted?), bypasssub:1.
//   painsplit: the exact average floor((u+t)/2) (odd-sum probe), both directions, vs a
//     SUBSTITUTE (blocked? painsplit has protect:1 but NOT bypasssub), draw model.
//   psychup: copies the target's boost STAGES exactly (incl. NEGATIVE stages, overwriting
//     the user's own), bypasssub:1 (works through a sub), draw-free?
//
// Run:  node src/rust_sim/harness/probe_batch6_utility.js
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
  for (const id of ['bellydrum', 'charge', 'memento', 'mimic', 'painsplit', 'psychup']) {
    const m = d.moves.get(id);
    console.log(`=== resolved gen3 ${id} ===`);
    console.log(`  cat=${m.category} bp=${m.basePower} acc=${JSON.stringify(m.accuracy)} type=${m.type} prio=${m.priority} pp=${m.pp} target=${m.target}`);
    console.log(`  flags=${JSON.stringify(m.flags)} selfdestruct=${JSON.stringify(m.selfdestruct)} volatileStatus=${JSON.stringify(m.volatileStatus)} boosts=${JSON.stringify(m.boosts)} ignoreImmunity=${JSON.stringify(m.ignoreImmunity)} noPPBoosts=${JSON.stringify(m.noPPBoosts)}`);
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
    // the mod-chain law: know what gen3 REPLACED (base + gen4 deltas).
    for (const g of ['gen4', 'gen9']) {
      const gm = Dex.mod(g).moves.get(id);
      console.log(`  [${g}] acc=${JSON.stringify(gm.accuracy)} selfdestruct=${JSON.stringify(gm.selfdestruct)} boosts=${JSON.stringify(gm.boosts)} volatile=${JSON.stringify(gm.volatileStatus)}`);
    }
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
  while (!battle.ended && safety < 24) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    if (i >= plan.length) break;
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
    const ppStr = (m) => m ? m.moveSlots.map((s) => `${s.id}:${s.pp}/${s.maxpp}${s.virtual ? '(virt)' : ''}`).join(',') : '-';
    const boostStr = (m) => m ? JSON.stringify(Object.fromEntries(Object.entries(m.boosts).filter(([, v]) => v !== 0))) : '-';
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'} boosts=${boostStr(m)} vols=[${Object.keys(m.volatiles).join(',')}] pp={${ppStr(m)}}` : '-';
    console.log(`  [${rs}] ${JSON.stringify({ p1: entry.p1, p2: entry.p2 })} draws=${draws.length}  seed ${before}->${after}`);
    console.log(`        pre : p1=${hpBefore[0]}  p2=${hpBefore[1]}`);
    console.log(`        post: p1=${fmt(a0)}`);
    console.log(`              p2=${fmt(a1)}`);
    draws.forEach((dl, k) => console.log(`        DRAW[${k}] ${dl}`));
    const newLines = log.slice(logLen0).filter((l) =>
      /\|move\||\|turn\||-damage|-heal|-boost|-unboost|-setboost|-copyboost|-sethp|-fail|-immune|-miss|-crit|-supereffective|-resisted|cant|-activate|-hitcount|-end\b|-start|switch|drag|faint|-prepare|-singleturn|-singlemove|-nothing/.test(l));
    for (const l of newLines) console.log(`        LINE ${l}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  dumpResolved();

  // -------------------------------------------------------------- BELLY DRUM
  // BD1: full-HP success — +6 Atk SET (what line? -setboost?), pays floor(maxhp/2)=262 of 524,
  // draw count; then a 2nd Belly Drum at Atk +6 -> fail (draws?); then splash control.
  await run('BD1 bellydrum: full-HP success (+6, -262) then Atk-already-6 fail',
    [mon('Snorlax', ['bellydrum', 'splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1', stop: true }]);

  // BD2: the EXACT HP GATE at an EVEN maxhp (Snorlax 524; maxhp/2 = 262 exactly):
  // hp == 262 -> fail? (hp <= maxhp/2). Then hp == 263 -> success (cost 262 -> hp 1).
  await run('BD2 bellydrum: hp == maxhp/2 (262/524) fail boundary, then 263 success -> hp 1',
    [mon('Snorlax', ['bellydrum', 'splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },                                  // hp 262 -> fail?
     { p1: 'move 1', p2: 'move 1', pre: [{ side: 0, hp: 263 }], stop: true }], // hp 263 -> +6, hp 1?
    { acts: [{ side: 0, hp: 262 }] });

  // BD3: from a NEGATIVE Atk stage (-2 via injection) -> SET to +6 (not +8-clamped etc.).
  await run('BD3 bellydrum: from Atk -2 -> SET to exactly +6',
    [mon('Snorlax', ['bellydrum', 'splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 0, boosts: { atk: -2 } }] });

  // ------------------------------------------------------------------ CHARGE
  // CH1: control tbolt -> charge (boosts? volatile? draws?) -> charged tbolt (x2 damage?)
  // -> next tbolt (volatile consumed?).
  await run('CH1 charge: control tbolt / charge turn / charged tbolt x2 / post tbolt',
    [mon('Lanturn', ['thunderbolt', 'charge', 'surf', 'splash'], { evs: { spa: 252 } })],
    [mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },              // control damage
     { p1: 'move 2', p2: 'move 1' },              // charge: volatile + any SpD boost? draws?
     { p1: 'move 1', p2: 'move 1' },              // charged tbolt: ~2x the control?
     { p1: 'move 1', p2: 'move 1', stop: true }]); // volatile gone -> control damage again?

  // CH2: charge then an IDLE splash turn: does the volatile survive to turn 3's tbolt
  // (duration 2 -> expires at the end of the splash turn?); then charge -> SURF (non-Electric):
  // consumed by any move or only Electric?
  await run('CH2 charge lifetime: charge/splash/tbolt (expired?) then charge/surf/tbolt',
    [mon('Lanturn', ['thunderbolt', 'charge', 'surf', 'splash'], { evs: { spa: 252 } })],
    [mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' },              // charge
     { p1: 'move 4', p2: 'move 1' },              // splash: volatile still up at end?
     { p1: 'move 1', p2: 'move 1' },              // tbolt: doubled or expired?
     { p1: 'move 2', p2: 'move 1' },              // charge again
     { p1: 'move 3', p2: 'move 1' },              // surf: does a non-Electric move consume it?
     { p1: 'move 1', p2: 'move 1', stop: true }]); // tbolt: doubled iff surf did NOT consume

  // ----------------------------------------------------------------- MEMENTO
  // ME1: baseline — foe -2 Atk / -2 SpA, the user FAINTS; line order; draws (acc=true ->
  // zero accuracy draw?); then the forced replacement.
  await run('ME1 memento: baseline -2/-2 + self-faint + replacement',
    [mon('Dugtrio', ['memento', 'splash']), mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['splash', 'protect', 'substitute'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },              // memento
     { p1: 'switch 2' },                          // forced replacement
     { p1: 'move 1', p2: 'move 1', stop: true }]);

  // ME2: memento INTO A PROTECT (protect:1) — blocked; does the user faint anyway
  // (selfdestruct:'ifHit' says NO — probe it)?
  await run('ME2 memento into a Protect: user faints or not?',
    [mon('Dugtrio', ['memento', 'splash']), mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['protect', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // ME3: memento INTO A SUBSTITUTE — blocked? user faints?
  await run('ME3 memento into a Substitute: blocked? user faints?',
    [mon('Dugtrio', ['memento', 'splash']), mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['substitute', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' },              // skarm subs (dugtrio splashes)
     { p1: 'move 1', p2: 'move 2', stop: true }]); // memento into the sub

  // ME4: memento vs CLEAR BODY (both drops blocked) — does the move "hit" (user faints)?
  await run('ME4 memento vs Clear Body: drops blocked, user faints?',
    [mon('Dugtrio', ['memento', 'splash']), mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [mon('Metagross', ['splash'], { ability: 'Clear Body', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // ME5: memento vs a foe ALREADY at -6/-6 — gen3 boost() into the floor: fail? user faints?
  await run('ME5 memento vs a -6/-6 foe: into-the-floor behavior',
    [mon('Dugtrio', ['memento', 'splash']), mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 1, boosts: { atk: -6, spa: -6 } }] });

  // ------------------------------------------------------------------- MIMIC
  // MI1: SLOWER mimic user — the foe's lastMove exists by the time Mimic runs; the copy
  // semantics: which slot, what PP/maxPP (probe moveSlots), then USE the copied move,
  // then switch out + back in -> reverted?
  await run('MI1 mimic: copy psychic (slot/pp?), use it, switch-revert',
    [mon('Snorlax', ['mimic', 'splash'], { evs: { hp: 252 } }), mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [mon('Alakazam', ['psychic', 'splash'], { evs: {} })],
    [{ p1: 'move 1', p2: 'move 1' },              // zam psychic first; lax mimic -> copies?
     { p1: 'move 1', p2: 'move 2' },              // what does 'move 1' resolve to now (copied slot)?
     { p1: 'switch 2', p2: 'move 2' },            // lax out
     { p1: 'switch 2', p2: 'move 2' },            // lax back in: moveSlots reverted? pp?
     { p1: 'move 1', p2: 'move 2', stop: true }]); // mimic again (fresh copy? zam lastMove=splash)

  // MI2: FASTER mimic user on turn 1 — the target has NO lastMove -> fail (draws? lines?).
  await run('MI2 mimic with no target lastMove: fail',
    [mon('Alakazam', ['mimic', 'splash'])],
    [mon('Snorlax', ['bodyslam', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 2', stop: true }]);

  // MI3: the ALREADY-KNOWN fail — the user already knows the target's lastMove (splash).
  await run('MI3 mimic when the user already knows the move: fail?',
    [mon('Snorlax', ['mimic', 'splash'], { evs: { hp: 252 } })],
    [mon('Alakazam', ['splash', 'psychic'])],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]); // zam splash first -> lax mimics splash (known)

  // MI4: mimic THROUGH a substitute (bypasssub:1) — copies anyway?
  await run('MI4 mimic through a substitute (bypasssub)',
    [mon('Snorlax', ['mimic', 'splash'], { evs: { hp: 252 } })],
    [mon('Alakazam', ['substitute', 'psychic', 'splash'])],
    [{ p1: 'move 2', p2: 'move 1' },              // zam subs
     { p1: 'move 2', p2: 'move 2' },              // zam psychic (lastMove=psychic, sub up)
     { p1: 'move 1', p2: 'move 3', stop: true }]); // lax mimic through the sub

  // -------------------------------------------------------------- PAIN SPLIT
  // PS1: baseline — user low, target high, ODD sum: Gengar 41 + Blissey 101 = 142 -> 71 each?
  // (even) — use 41+100=141 -> floor 70.5 = 70. Then direction 2: user HIGH -> user damaged.
  await run('PS1 painsplit: odd-sum floor both directions',
    [mon('Gengar', ['painsplit', 'splash'], { ability: 'Levitate' })],
    [mon('Blissey', ['splash', 'painsplit'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },              // gengar 41 + blissey 100 -> 70/70?
     { p1: 'move 2', p2: 'move 2', stop: true }],  // blissey painsplits back (both at 70; equal -> ?)
    { acts: [{ side: 0, hp: 41 }, { side: 1, hp: 100 }] });

  // PS1b: painsplit where the AVERAGE EXCEEDS the user's maxhp path is impossible (avg <= max
  // of the two), but the TARGET-damaged direction + a healthy gap: user 41, target 651 -> 346.
  await run('PS1b painsplit: big gap (41 + 651 -> 346 each), draws?',
    [mon('Gengar', ['painsplit', 'splash'], { ability: 'Levitate' })],
    [mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 0, hp: 41 }] });

  // PS2: painsplit INTO A SUBSTITUTE — blocked (no bypasssub)? what line? draws?
  await run('PS2 painsplit into a substitute: blocked?',
    [mon('Gengar', ['painsplit', 'splash'], { ability: 'Levitate' })],
    [mon('Blissey', ['substitute', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' },              // blissey subs
     { p1: 'move 1', p2: 'move 2', stop: true }],  // painsplit into the sub
    { acts: [{ side: 0, hp: 41 }] });

  // PS3: painsplit vs a GHOST/Normal-immunity question — painsplit is typeless-Normal Status;
  // ignoreImmunity? probe vs a Gengar target (Normal->Ghost immune if type-checked).
  await run('PS3 painsplit into a Ghost: type-immune or not?',
    [mon('Blissey', ['painsplit', 'splash'], { evs: { hp: 252 } })],
    [mon('Gengar', ['splash'], { ability: 'Levitate' })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 0, hp: 100 }] });

  // ---------------------------------------------------------------- PSYCH UP
  // PU1: copies the foe's EXACT stages incl. NEGATIVE, overwriting the user's own (+1 spe pre-set).
  await run('PU1 psychup: copy {atk:+2, def:-1, spa:+3} over own {spe:+1}',
    [mon('Snorlax', ['psychup', 'splash'], { evs: { hp: 252 } })],
    [mon('Alakazam', ['splash'], {})],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 1, boosts: { atk: 2, def: -1, spa: 3 } }, { side: 0, boosts: { spe: 1 } }] });

  // PU2: psychup THROUGH a substitute (bypasssub:1) — copies anyway?
  await run('PU2 psychup through a substitute',
    [mon('Snorlax', ['psychup', 'splash'], { evs: { hp: 252 } })],
    [mon('Alakazam', ['substitute', 'calmmind', 'splash'])],
    [{ p1: 'move 2', p2: 'move 1' },              // zam subs
     { p1: 'move 2', p2: 'move 2' },              // zam calm minds (+1/+1 behind the sub)
     { p1: 'move 1', p2: 'move 3', stop: true }]); // psych up through the sub

  // PU3: psychup vs a PROTECTING foe — psychup has NO protect flag -> NOT blocked?
  await run('PU3 psychup vs Protect: blocked or not (no protect flag)?',
    [mon('Snorlax', ['psychup', 'splash'], { evs: { hp: 252 } })],
    [mon('Alakazam', ['protect', 'splash'], {})],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 1, boosts: { spa: 2 } }] });

  // PU4: psychup with ALL-ZERO foe boosts — clears the user's own stages (copy of zeros)?
  await run('PU4 psychup vs zero boosts: zeroes the user\'s own +2?',
    [mon('Snorlax', ['psychup', 'splash'], { evs: { hp: 252 } })],
    [mon('Alakazam', ['splash'], {})],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 0, boosts: { atk: 2, spe: 1 } }] });

  // ------------------------------------------------------- SUPPLEMENTAL EDGES
  // BD4: the ODD-maxhp COST rounding — Snorlax hp-ev 248 -> maxhp 523; directDamage(261.5)
  // -> cost 261 (floor) leaving 262, or 262 (round) leaving 261?
  await run('BD4 bellydrum odd maxhp 523: cost floor(261.5)?',
    [mon('Snorlax', ['bellydrum', 'splash'], { evs: { hp: 248 } })],
    [mon('Skarmory', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // ME6: memento as the LAST mon — the self-faint loses the game (foe wins)? Quick Claw drawn?
  await run('ME6 memento as the last mon: self-faint -> loss',
    [mon('Dugtrio', ['memento', 'splash'])],
    [mon('Skarmory', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
