// probe_contact_proc_rng.js — settle the EXACT draw model of the CONTACT_PROC ability
// class (Static / Poison Point / Flame Body / Effect Spore / Cute Charm) + the DRAW-FREE
// contact reactions (Rough Skin / Color Change), against the resolved gen3 sim's PRNG
// (Mandate 1 — the sim is the only oracle).
//
// The crux this probe nails (NOT assumable from the count alone):
//   (1) The EXACT roll: randomChance(1,3) for static/poisonpoint/flamebody/cutecharm,
//       randomChance(1,10)+sample(3) for effectspore.
//   (2) The POSITION relative to the move's OWN secondary — the sim fires runEvent(
//       'DamagingHit') (gen<5, INSIDE runMoveEffects) BEFORE secondaries() → the contact
//       proc's randomChance draws BEFORE the move's own secondary random(100).
//   (3) Effect Spore's NESTED sample(3): one randomChance(1,10) gate, THEN on-land one
//       sample(['slp','par','psn']).
//   (4) A NON-CONTACT move must NOT proc (no randomChance at all).
//   (5) The status lands on the ATTACKER (source), gen-3 type/ability immunities apply.
//   (6) Serene Grace / Shield Dust: do they touch the FOE's contact proc? (they must NOT —
//       Shield Dust is the HOLDER's own-move-secondary gate; the contact proc is a FOE
//       ability on the attacker.)
//
// It instruments prng.random / prng.randomChance / prng.sample and records EACH call with
// its arguments + a coarse call-site label (parsed from the stack), so the exact sequence
// is visible. Run: node src/rust_sim/harness/probe_contact_proc_rng.js

'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

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

// Parse a coarse label from the call stack: the first battle-* frame that isn't the prng.
function siteLabel(stack) {
  const lines = String(stack).split('\n').slice(2);
  for (const l of lines) {
    const m = l.match(/at (?:Battle|BattleActions|Pokemon|Side|Field)\.?(\w+)/);
    if (m && !['random', 'randomChance', 'sample', 'shuffle'].includes(m[1])) return m[1];
  }
  return '?';
}

async function run(teamsFn, ability, seed, choices, opts = {}) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of String(ch).split('\n')) lines.push(l); } })();
  const [p1, p2] = teamsFn(ability);
  streams.omniscient.write(`>start {"formatid":"${opts.fmt || 'gen3customgame'}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2) })}`);
  for (let i = 0; i < 14; i++) await tick();
  const battle = stream.battle;
  const prng = battle.prng;
  const calls = []; // { kind, args, site }
  const wrap = (name) => {
    const orig = prng[name].bind(prng);
    prng[name] = (...a) => {
      const r = orig(...a);
      calls.push({ kind: name, args: a, site: siteLabel(new Error().stack), ret: r });
      return r;
    };
  };
  wrap('random'); wrap('randomChance'); wrap('sample');

  const perDecision = [];
  for (const [c1, c2] of choices) {
    const before = calls.length;
    if (c1) streams.omniscient.write(`>p1 ${c1}`);
    if (c2) streams.omniscient.write(`>p2 ${c2}`);
    for (let k = 0; k < 12; k++) await tick();
    perDecision.push(calls.slice(before));
    if (battle.ended) break;
  }
  return { calls, perDecision, lines, ended: battle.ended };
}

function fmtCalls(cs) {
  return cs.map((c) => `${c.kind}(${JSON.stringify(c.args)})@${c.site}=>${JSON.stringify(c.ret)}`).join('  ');
}

// Report the FIRST decision's call sequence for a probed ability + a control, so the exact
// contact-proc draw + its position vs the move secondary is visible.
async function report(label, teamsFn, probed, control, choices, opts = {}) {
  const seed = opts.seed || [1, 2, 3, 4];
  const a = await run(teamsFn, probed, seed, choices, opts);
  const b = await run(teamsFn, control, seed, choices, opts);
  console.log(`\n=== ${label}  [probed=${probed} vs control=${control}]  seed=${JSON.stringify(seed)} ===`);
  for (let i = 0; i < a.perDecision.length; i++) {
    console.log(`  dec${i} PROBED : ${fmtCalls(a.perDecision[i])}`);
    console.log(`  dec${i} CONTROL: ${fmtCalls((b.perDecision[i] || []))}`);
  }
  // status witness on the ATTACKER (p1 active) from the lines
  const statusLines = a.lines.filter((l) => /-status|-start|attract/.test(l));
  console.log(`  probed status/volatile lines: ${JSON.stringify(statusLines)}`);
}

(async () => {
  const contactMon = (ab) => [
    // p1: a Body-Slam user (Body Slam IS contact — and has its OWN 30% par secondary,
    // so we see the contact proc AND the move secondary in one decision).
    [mon('Snorlax', ['bodyslam', 'bodyslam'])],
    // p2: the contact-proc holder (bulky so it survives + procs).
    [mon('Suicune', ['recover', 'recover'], { ability: ab })],
  ];

  await report('CONTACT_PROC Static (par) — Body Slam (contact + own par secondary)', contactMon,
    'Static', 'Shell Armor', [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']]);

  await report('CONTACT_PROC Poison Point (psn)', contactMon,
    'Poison Point', 'Shell Armor', [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']]);

  await report('CONTACT_PROC Flame Body (brn)', contactMon,
    'Flame Body', 'Shell Armor', [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']]);

  await report('CONTACT_PROC Effect Spore (randomChance(1,10)+sample(3))', contactMon,
    'Effect Spore', 'Shell Armor', [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']]);

  await report('CONTACT_PROC Cute Charm (attract volatile — DEFER)', contactMon,
    'Cute Charm', 'Shell Armor', [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']]);

  // NON-CONTACT move (Earthquake is NOT contact) into a Static mon → must NOT proc.
  await report('NON-CONTACT Earthquake into Static — must NOT proc', (ab) => [
    [mon('Snorlax', ['earthquake', 'earthquake'])],
    [mon('Suicune', ['recover', 'recover'], { ability: ab })],
  ], 'Static', 'Shell Armor', [['move 1', 'move 1'], ['move 1', 'move 1']]);

  // Rough Skin — draw-free recoil.
  await report('Rough Skin (recoil baseMaxhp/16 — draw-free)', (ab) => [
    [mon('Snorlax', ['bodyslam', 'bodyslam'])],
    [mon('Suicune', ['recover', 'recover'], { ability: ab })],
  ], 'Rough Skin', 'Shell Armor', [['move 1', 'move 1'], ['move 1', 'move 1']]);

  // Color Change — draw-free type change (Body Slam Normal → target becomes Normal).
  await report('Color Change (target type := move.type — draw-free)', (ab) => [
    [mon('Snorlax', ['bodyslam', 'bodyslam'])],
    [mon('Kecleon', ['recover', 'recover'], { ability: ab })],
  ], 'Color Change', 'Shell Armor', [['move 1', 'move 1'], ['move 1', 'move 1']]);

  // Serene Grace on the ATTACKER — does it double the FOE's contact proc? (must NOT: SG is
  // onModifyMove for the user's OWN move secondaries.) Compare Static-proc count SG vs not.
  await report('Serene Grace attacker vs Static foe — SG must NOT touch the contact proc', (ab) => [
    [mon('Jirachi', ['bodyslam', 'bodyslam'], { ability: 'Serene Grace' })],
    [mon('Suicune', ['recover', 'recover'], { ability: ab })],
  ], 'Static', 'Shell Armor', [['move 1', 'move 1'], ['move 1', 'move 1']]);

  // Shield Dust on the HOLDER (the contact-proc target): does it suppress the foe's contact
  // proc? Shield Dust only blocks the HOLDER's incoming-move secondaries — NOT its own
  // reactive ability. But the mon can't have BOTH Shield Dust AND Static. So test Shield
  // Dust as the target of a Body Slam: does the ATTACKER's... no — the contact proc is the
  // TARGET's ability. Shield Dust IS a target ability. So a Shield-Dust mon simply doesn't
  // have Static. Skip — the interaction is N/A (one ability slot). We note it in the report.

  console.log('\n\nNOTE: Shield Dust vs contact-proc is N/A — both are the TARGET mon\'s ability slot (a mon holds one). Shield Dust blocks the move\'s OWN secondary (the attacker\'s), not a foe ability.');
})();
