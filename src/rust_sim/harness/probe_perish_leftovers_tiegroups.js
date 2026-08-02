// probe_perish_leftovers_tiegroups.js — root-cause `sbd_msb1zfxs_b134`: on a SPEED-TIED mutual
// perish-out, node emits `|-start|p2a|perish0` then p1a (and faints in that order) while the port
// emits p1a first. The perish3/2/1 ticks and the two Leftovers `-heal` lines on the SAME turn all
// MATCH, so the shuffle DRAW agrees and only the perish pair's PRE-SHUFFLE order differs.
//
// WHY THE EARLIER PROBE MISSED IT: `probe_perish_double_faint_order.js` gave the mons NO ITEM, so
// the residual had exactly ONE tie group (perish). With a single group nothing perturbs it and both
// engines agree — which is why that probe pronounced the port "correct in isolation". The b134
// board has TWO tie groups: Leftovers (order 10, subOrder 4) and perish (order 12). Showdown's
// `speedSort` is a NON-STABLE selection sort that, per round, collects ALL tied-best indices and
// SWAPS them into place — and those swaps can REORDER the still-unsorted tail, i.e. the perish pair,
// before its own shuffle ever reads it. This probe instruments `speedSort` to print the array at
// each stage so the pre-shuffle order is observed rather than inferred.
//
// Run:  node src/rust_sim/harness/probe_perish_leftovers_tiegroups.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
// The b134 spread: flat 85 EVs / Hardy — which is what puts Misdreavus (base spe 85 via its 227
// stat) and Aipom on the SAME 227 speed. Reproduced so the tie is genuine, not hand-forced.
const EV85 = { hp: 85, atk: 85, def: 85, spa: 85, spd: 85, spe: 85 };
function mon(species, moves, item, ability) {
  return {
    species, item: item || '', ability: ability || 'No Ability', moves,
    evs: EV85, ivs: IV31, nature: 'Hardy', level: 100, gender: 'M',
  };
}
const tick = () => new Promise((r) => setTimeout(r, 0));

function describe(h) {
  const eff = (h.effect && h.effect.id) || '?';
  // The mon a residual handler belongs to is `effectHolder` (`target` is unset for these).
  const t = h.effectHolder || h.target;
  const who = t && t.side ? t.side.id : '?';
  return `${eff}@${who}`;
}

async function run(label, item) {
  // Two speed-TIED mons. p1 carries Perish Song (which hits BOTH actives), so a mutual perish-out
  // follows. `item` is the variable under test: with Leftovers there are TWO residual tie groups,
  // without it only ONE.
  // Real b134 abilities (both residual-handler-free, so they cannot perturb the gather).
  const p1 = [mon('Misdreavus', ['perishsong', 'return'], item, 'Levitate')];
  const p2 = [mon('Aipom', ['return', 'splash'], item, 'Run Away')];

  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify([51782, 41991, 10548, 37725])}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  const spe = (s) => battle.sides[s].active[0].getStat('spe');

  // Instrument speedSort: dump the residual handler array pre-sort, after each round's swaps, and
  // the exact slice each tie-shuffle sees.
  let trace = [];
  let capture = false;
  const origSort = battle.speedSort.bind(battle);
  battle.speedSort = (list, comparator) => {
    if (!capture || list.length < 2 || !list[0] || !list[0].effect) return origSort(list, comparator);
    const before = list.map(describe);
    // Re-implement the source's loop so the per-round slices can be observed, delegating the
    // actual shuffling to the real prng so draws stay bit-identical.
    const cmp = comparator || ((a, b) => battle.comparePriority(a, b));
    let sorted = 0;
    const rounds = [];
    while (sorted + 1 < list.length) {
      let nextIndexes = [sorted];
      for (let i = sorted + 1; i < list.length; i++) {
        const delta = cmp(list[nextIndexes[0]], list[i]);
        if (delta < 0) continue;
        if (delta > 0) nextIndexes = [i];
        else nextIndexes.push(i);
      }
      for (let i = 0; i < nextIndexes.length; i++) {
        const index = nextIndexes[i];
        if (index === sorted + i) continue;
        const t = list[sorted + i]; list[sorted + i] = list[index]; list[index] = t;
      }
      const preShuffle = list.slice(sorted, sorted + nextIndexes.length).map(describe);
      if (nextIndexes.length > 1) battle.prng.shuffle(list, sorted, sorted + nextIndexes.length);
      const postShuffle = list.slice(sorted, sorted + nextIndexes.length).map(describe);
      rounds.push({ at: sorted, size: nextIndexes.length, preShuffle, postShuffle });
      sorted += nextIndexes.length;
    }
    trace.push({ before, rounds });
  };

  // BOTH mons must take damage, so BOTH Leftovers actually heal and their group's order is
  // observable in the emitted lines (at full HP the handler still runs but prints nothing).
  const plan = [
    { p1: 'move perishsong', p2: 'move return' },
    { p1: 'move return', p2: 'move return' },
    { p1: 'move return', p2: 'move return' },
    { p1: 'move return', p2: 'move return' },
  ];
  console.log(`\n=== ${label} ===  spe p1=${spe(0)} p2=${spe(1)} ` +
    `(${spe(0) === spe(1) ? 'TIE — good' : 'NOT tied — repro broken'})`);
  for (const step of plan) {
    if (battle.ended) break;
    if (battle.requestState === 'move') {
      trace = []; capture = true;
      if (step.p1) streams.omniscient.write(`>p1 ${step.p1}`);
      if (step.p2) streams.omniscient.write(`>p2 ${step.p2}`);
      for (let k = 0; k < 20; k++) await tick();
      capture = false;
      // Only the RESIDUAL sort (the one carrying a perish handler) is interesting.
      for (const t of trace) {
        if (!t.before.some((s) => s.startsWith('perishsong@'))) continue;
        console.log(`  residual handlers pre-sort: [${t.before.join(', ')}]`);
        for (const r of t.rounds) {
          console.log(`    round@${r.at} size=${r.size}  preShuffle=[${r.preShuffle.join(', ')}]`
            + `  -> postShuffle=[${r.postShuffle.join(', ')}]`);
        }
      }
    }
  }
  const perish = log.filter((l) => /\|-start\|p[12]a: \w+\|perish\d/.test(l))
    .map((l) => `${l.match(/perish(\d)/)[1]}:p${l.match(/p([12])a/)[1]}`);
  const heals = log.filter((l) => l.includes('[from] item: Leftovers')).map((l) => `p${l.match(/p([12])a/)[1]}`);
  console.log(`  Leftovers heal order (in order): ${heals.join(' ')}`);
  const faints = log.filter((l) => /^\|faint\|/.test(l)).map((l) => `p${l.match(/p([12])a/)[1]}`);
  console.log(`  emitted perish marks (tick:side, in order): ${perish.join('  ')}`);
  console.log(`  faint order: [${faints.join(' then ')}]`);
}

(async () => {
  // ONE tie group (no item) — what the earlier probe tested, and why it saw no problem.
  await run('A no item  => ONE residual tie group (perish only)', '');
  // TWO tie groups (the b134 board) — Leftovers sorts FIRST and its swaps can reorder the
  // still-unsorted perish pair before the perish shuffle reads it.
  await run('B Leftovers => TWO residual tie groups (the b134 board)', 'Leftovers');
})();
