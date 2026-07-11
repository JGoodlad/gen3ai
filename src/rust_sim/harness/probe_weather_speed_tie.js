// probe_weather_speed_tie.js — settle the WEATHER_SPEED (Chlorophyll / Swift Swim) draw
// model AT the tie boundary (the real risk: a wrong effective speed desyncs the
// action-order + eachEvent tie-shuffles, which DRAW from the PRNG).
//
// The generic draw-model probe used never-miss moves at distinct speeds, so no tie-shuffle
// fired. Here we CONSTRUCT a speed TIE created BY the ×2 weather boost: a Chlorophyll mon
// whose sun-doubled speed EXACTLY equals the foe's speed → the action-order speed-tie
// shuffle draws ONE random(0,2). We prove: (1) with the boost the mon TIES (an extra draw
// vs a no-boost control at the same base speed), and (2) the boosted-speed action ORDER
// matches — by comparing the sim's first-mover + total draw count between the Chlorophyll
// mon and a same-base-speed no-op mon that does NOT tie.
//
// We instrument prng.next and also record who moved first (the |move| order) so a wrong
// effective speed is caught in BOTH the draw count AND the order.
//
// Run: node src/rust_sim/harness/probe_weather_speed_tie.js

'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Dex, Teams } = require(path.join(PS, 'dist/sim'));

const d3 = Dex.mod('gen3');
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

// Compute a mon's raw speed (storedStats.spe) for planning a tie.
function speedOf(species, evs, nature) {
  const sp = d3.species.get(species);
  const base = sp.baseStats.spe;
  const iv = 31; const ev = (evs && evs.spe) || 0;
  let stat = Math.floor(((2 * base + iv + Math.floor(ev / 4)) * 100) / 100) + 5;
  // nature
  const nat = d3.natures.get(nature || 'Serious');
  if (nat && nat.plus === 'spe') stat = Math.floor(stat * 1.1);
  if (nat && nat.minus === 'spe') stat = Math.floor(stat * 0.9);
  return stat;
}

async function run(p1sets, p2sets, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const moveOrder = [];
  (async () => {
    for await (const ch of streams.omniscient) {
      for (const l of String(ch).split('\n')) {
        if (l.startsWith('|move|')) moveOrder.push(l.split('|')[2]); // p1a/p2a
      }
    }
  })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1sets) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2sets) })}`);
  for (let i = 0; i < 14; i++) await tick();
  const battle = stream.battle;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  let n = 0; rng.next = (...a) => { n += 1; return realNext(...a); };

  const per = [];
  for (let t = 0; t < 3; t++) {
    const b = n;
    streams.omniscient.write('>p1 move 1');
    streams.omniscient.write('>p2 move 1');
    for (let k = 0; k < 10; k++) await tick();
    per.push(n - b);
    if (battle.ended) break;
  }
  // effective speeds the sim actually used
  const es1 = battle.sides[0].active[0].getActionSpeed ? battle.sides[0].active[0].getActionSpeed() : null;
  const es2 = battle.sides[1].active[0].getActionSpeed ? battle.sides[1].active[0].getActionSpeed() : null;
  return { totalDraws: n, per, firstMovers: moveOrder, es1, es2 };
}

(async () => {
  // Exeggutor base spe 55 → raw 155 (0 EV). ×2 in sun = 310. Find a foe with raw speed 310.
  // Persian base 115 → 285; Aerodactyl base 130 → 315; Jolteon base 130 → 315. We want the
  // foe at 310. Electrode base 140 → 325; Crobat 130→315. Hmm. Instead pick a foe with a
  // spread that lands 310. Easier: use damaging moves so a TIE both-attack turn draws the
  // action-order shuffle. We just want to demonstrate the boost CREATES a tie.
  //
  // Simpler + robust: Exeggutor 155 → ×2 = 310. Foe = a mon at raw 310. Zapdos base 100 →
  // 240; +252 EV +Timid → floor((2*100+31+63)*100/100+5)*1.1 = floor(294+5=299 *1.1)=328.
  // Let's just brute-force a foe raw speed from a few species/EV combos to hit 310.
  const targets = [
    ['Jolteon', { spe: 0 }, 'Serious'], // 130 -> 315
    ['Aerodactyl', { spe: 0 }, 'Serious'], // 130 -> 315
    ['Alakazam', { spe: 0 }, 'Serious'], // 120 -> 305
    ['Gengar', { spe: 0 }, 'Serious'], // 110 -> 295
    ['Starmie', { spe: 0 }, 'Serious'], // 115 -> 300
    ['Sceptile', { spe: 0 }, 'Serious'], // 120 -> 305
  ];
  const exegSpe = speedOf('Exeggutor', {}, 'Serious');
  const exegSun = exegSpe * 2;
  console.log(`Exeggutor raw spe = ${exegSpe}, ×2 (sun) = ${exegSun}`);
  for (const [sp, evs, nat] of targets) {
    console.log(`   ${sp} raw spe = ${speedOf(sp, evs, nat)}`);
  }
  // Find a foe whose raw speed == exegSun (310) — tune EVs. Alakazam base 120: to hit 310,
  // need floor((2*120+31+ev/4)*100/100)+5 = 310 → 2*120+31+ev/4 = 305 → ev/4 = 34 → ev=136.
  const zamNeed = exegSun; // 310
  // solve ev: (240+31+floor(ev/4))+5 = 310 → floor(ev/4)=34 → ev in [136,139]
  const foeEvs = { spe: 136 };
  const foeSpe = speedOf('Alakazam', foeEvs, 'Serious');
  console.log(`   Alakazam @${JSON.stringify(foeEvs)} EV raw spe = ${foeSpe} (want ${exegSun})`);

  console.log('\n=== TIE test: Exeggutor(Chlorophyll) in sun vs Alakazam tuned to the SAME sun-doubled speed ===');
  // Both use a damaging move so the action-order shuffle fires on a TIE.
  const p1 = [mon('Groudon', ['fireblast', 'fireblast'], { ability: 'Drought' })]; // sun setter, also attacks
  // Actually we want Exeggutor + Alakazam to be the two actives. Put Groudon as p1 lead to
  // set sun, but then Exeggutor isn't active. Simpler: use a Chlorophyll mon that ALSO could
  // set... no. Use Drought on Groudon as p1, Exeggutor as p2 — but then the tie is Groudon vs
  // Exeggutor. Let me instead make BOTH teams 1 mon and set sun via a move? Sunny Day isn't
  // modeled but the SIM has it. Use Groudon (Drought) as the Chlorophyll-side lead's TEAMMATE?
  // Cleanest: p1 = [Groudon(Drought)], p2 = [Exeggutor(Chlorophyll)]. Tie = Groudon vs Exeg.
  // Groudon base spe 90 → 245. Not 310. So tune: we want the ACTIVE p1 to tie Exeggutor@310.
  // Give Groudon +Spe EVs+nature to reach 310: base 90 → need (2*90+31+ev/4)+5=310/1.1?
  // Groudon isn't Chlorophyll; its speed is fixed 245-ish. To reach 310 raw: floor((180+31+
  // floor(ev/4))+5)=310 → floor(ev/4)=94 → ev=376 >255. With +nature: 310/1.1=281.8 →
  // (216 base-calc)+5... too low. Use a faster sun-setter: there's only Groudon in gen3.
  // OK — pivot the design: use Sunny Day as p1's move (the SIM models it) so p1 can be ANY
  // fast mon. p1 = Jolteon (raw 315) with Sunny Day; p2 = Exeggutor(Chlorophyll) 155→310.
  // Jolteon 315 vs Exeg 310 → NOT a tie. Tune Jolteon DOWN to 310 via -Spe or fewer... it's
  // base 130 → 315 at 0 EV already >310, can't go below 315 without -nature (315*0.9=283).
  // Use Alakazam (base 120 → 305) +136 EV → 310 as p1 with Sunny Day; p2 Exeggutor 155→310.
  const seeds = [[1, 2, 3, 4], [7, 11, 13, 17], [100, 200, 300, 400], [5, 5, 5, 5], [42, 42, 42, 42], [8, 8, 8, 8]];
  // Exeggutor sun-doubled = 292. Tune Alakazam to raw 292: base 120 → (240+31+floor(ev/4))+5
  // = 292 → floor(ev/4)=16 → ev=64.
  const foeTieEvs = { spe: 64 };
  console.log(`   Alakazam @${JSON.stringify(foeTieEvs)} EV raw spe = ${speedOf('Alakazam', foeTieEvs, 'Serious')} (want ${exegSun})`);
  let tieExtra = 0, distinctSame = 0;
  for (const seed of seeds) {
    // TIE case: p1 Alakazam@292 (Sunny Day then Psychic), p2 Exeggutor(Chlorophyll)@292 in sun.
    const tie = await run(
      [mon('Alakazam', ['sunnyday', 'psychic'], { evs: foeTieEvs })],
      [mon('Exeggutor', ['psychic', 'psychic'], { ability: 'Chlorophyll' })],
      seed,
    );
    // CONTROL: same, but p2 has a NO-OP ability (Insomnia) so its speed stays 146 (distinct).
    const noTie = await run(
      [mon('Alakazam', ['sunnyday', 'psychic'], { evs: foeTieEvs })],
      [mon('Exeggutor', ['psychic', 'psychic'], { ability: 'Insomnia' })],
      seed,
    );
    // Turn 2 is the both-Psychic turn (turn 1: Alakazam Sunny Day + Exeg Psychic).
    console.log(`  seed ${JSON.stringify(seed)}:`);
    console.log(`    Chlorophyll(tie@310):   draws=${tie.totalDraws} per=${JSON.stringify(tie.per)} es1=${tie.es1} es2=${tie.es2} firstMovers=${JSON.stringify(tie.firstMovers)}`);
    console.log(`    Insomnia(distinct 310 vs 155): draws=${noTie.totalDraws} per=${JSON.stringify(noTie.per)} es1=${noTie.es1} es2=${noTie.es2} firstMovers=${JSON.stringify(noTie.firstMovers)}`);
    // On the both-Psychic turn(s), the TIE should draw ONE MORE (the action-order shuffle)
    // than the distinct-speed control.
  }
  console.log('\n=> Read the per-turn draw counts: the Chlorophyll ×2 speed should TIE (an extra action-order shuffle draw) where the Insomnia control (155 vs 310) does not. The es2 field shows the sim used 310 for Chlorophyll-in-sun.');
})();
