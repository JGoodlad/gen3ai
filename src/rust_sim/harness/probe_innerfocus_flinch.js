// probe_innerfocus_flinch.js — settle whether Inner Focus (flinch immunity) changes the
// draw COUNT relative to a control that DOES flinch, on the turn AFTER a flinch would land.
//
// Inner Focus blocks the flinch volatile (onTryAddVolatile). The question: when Rock Slide's
// 30% flinch WOULD apply, a flinched mon's NEXT action is a `|cant|flinch` (drawing NO
// acc/crit/dmg), while an Inner-Focus mon MOVES (drawing acc/crit/dmg). That is a draw-COUNT
// difference — so Inner Focus is NOT a no-op if a flinch lands. We instrument prng.next per
// turn and find a seed where the control FLINCHES, then compare that turn's draw count.
//
// Method: fast Aerodactyl Rock Slide (30% flinch) into a Snorlax that also attacks. On a
// flinch, Snorlax's Body Slam is cancelled (fewer draws). With Inner Focus it attacks
// (more draws). We look at per-turn draws + whether a `|cant|...flinch` was emitted.
//
// Run: node src/rust_sim/harness/probe_innerfocus_flinch.js

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
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31, nature: opts.nature || 'Serious', level: 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(defAbility, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of String(ch).split('\n')) lines.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
  // Fast Aerodactyl uses Rock Slide (flinch 30%); slow Snorlax uses Body Slam. Snorlax bulky
  // so both survive many turns. Aerodactyl is a low-def frail — give it max HP so it lives.
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([mon('Aerodactyl', ['rockslide', 'rockslide'], { nature: 'Jolly', evs: { spe: 252, hp: 252 } })]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([mon('Snorlax', ['bodyslam', 'bodyslam'], { ability: defAbility, evs: { hp: 252, def: 252 } })]) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  let n = 0; rng.next = (...a) => { n += 1; return realNext(...a); };

  const per = [], cants = [];
  for (let t = 0; t < 20; t++) {
    const b = n;
    const lb = lines.length;
    streams.omniscient.write('>p1 move 1');
    streams.omniscient.write('>p2 move 1');
    for (let k = 0; k < 10; k++) await tick();
    per.push(n - b);
    const turnLines = lines.slice(lb);
    cants.push(turnLines.some((l) => l.startsWith('|cant|') && l.includes('flinch')));
    if (battle.ended || battle.sides[0].active[0].hp === 0 || battle.sides[1].active[0].hp === 0) break;
  }
  return { totalDraws: n, per, cants };
}

(async () => {
  console.log('=== Inner Focus flinch draw-count probe (Rock Slide flinch into Snorlax) ===');
  const seeds = [];
  for (let i = 0; i < 12; i++) seeds.push([i * 7 + 1, i * 13 + 3, i * 17 + 5, i * 3 + 2]);
  let anyFlinchDiff = false;
  for (const seed of seeds) {
    const ifoc = await run('Inner Focus', seed);
    const ctl = await run('Shell Armor', seed); // control: CAN flinch
    const ctlFlinched = ctl.cants.some(Boolean);
    const ifocFlinched = ifoc.cants.some(Boolean);
    const drawsMatch = ifoc.totalDraws === ctl.totalDraws && JSON.stringify(ifoc.per) === JSON.stringify(ctl.per);
    // Only interesting when the control actually flinched.
    if (ctlFlinched) {
      console.log(`  seed ${JSON.stringify(seed)}: control FLINCHED=${ctlFlinched} innerFocusFlinched=${ifocFlinched} drawsMatch=${drawsMatch}`);
      console.log(`      InnerFocus per=${JSON.stringify(ifoc.per)} total=${ifoc.totalDraws}`);
      console.log(`      Control    per=${JSON.stringify(ctl.per)} total=${ctl.totalDraws}`);
      if (!drawsMatch) anyFlinchDiff = true;
    }
  }
  console.log('');
  console.log(anyFlinchDiff
    ? '=> Inner Focus CHANGES the draw count when a flinch would land (the cant vs move draw difference). It is STRUCTURAL, not a no-op.'
    : '=> On every seed where the control flinched, the draw counts MATCHED — inspect: either the flinch never gated a move, or Inner Focus is draw-neutral here.');
})();
