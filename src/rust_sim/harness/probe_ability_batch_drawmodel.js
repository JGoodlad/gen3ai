// probe_ability_batch_drawmodel.js — settle the DRAW MODEL of the batch-1 class-(b)
// draw-free/structural abilities + the class-(a) no-op candidates, against the actual
// PRNG (Mandate 1 — the resolved sim is the only oracle).
//
// For each ability, run TWO battles at the same seed/teams/choices that DIFFER ONLY in the
// probed mon's ability (probed vs a chosen CONTROL), instrumenting `prng.rng.next()`, and
// report per-turn + total draw counts + whether they MATCH. A draw-free/structural ability
// keeps the draw count identical to a no-op control (only STATE differs); a draw-bearing
// one differs. Each case also declares the SCENARIO that makes its handler REACHABLE (a
// weather-speed tie under the right weather, a phaze into a Suction-Cups foe, etc.), so a
// "no-op" claim is proven where the handler WOULD fire, not just where it's dormant.
//
// Run: node src/rust_sim/harness/probe_ability_batch_drawmodel.js

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
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

// Run a scripted battle; `abilityOverride` replaces p2's mon[activeIdx] ability.
// teams = [p1sets, p2sets]; choices = [[p1choice, p2choice], ...] per decision.
// Returns { totalDraws, perDecision, lines }.
async function run(teamsFn, ability, seed, choices, opts = {}) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of String(ch).split('\n')) lines.push(l); } })();
  const [p1team, p2team] = teamsFn(ability);
  streams.omniscient.write(`>start {"formatid":"${opts.fmt || 'gen3customgame'}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 14; i++) await tick();
  const battle = stream.battle;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  let nextCount = 0;
  rng.next = (...a) => { nextCount += 1; return realNext(...a); };

  const perDecision = [];
  for (const [c1, c2] of choices) {
    const before = nextCount;
    if (c1) streams.omniscient.write(`>p1 ${c1}`);
    if (c2) streams.omniscient.write(`>p2 ${c2}`);
    for (let k = 0; k < 12; k++) await tick();
    perDecision.push(nextCount - before);
    if (battle.ended) break;
  }
  return { totalDraws: nextCount, perDecision, lines, ended: battle.ended };
}

// Compare a probed ability vs a control over several seeds; report draw parity + a STATE
// witness (a line-set diff hint) so a claim is proven where the handler is REACHABLE.
async function compare(label, teamsFn, probed, control, choices, opts = {}) {
  const seeds = opts.seeds || [[1, 2, 3, 4], [7, 11, 13, 17], [100, 200, 300, 400], [5, 5, 5, 5], [42, 42, 42, 42]];
  let allMatch = true;
  let anyStateDiff = false;
  const details = [];
  for (const seed of seeds) {
    const a = await run(teamsFn, probed, seed, choices, opts);
    const b = await run(teamsFn, control, seed, choices, opts);
    const match = a.totalDraws === b.totalDraws && JSON.stringify(a.perDecision) === JSON.stringify(b.perDecision);
    if (!match) allMatch = false;
    // A crude STATE witness: do the emitted line sets differ (excluding |t:|)?
    const filt = (ls) => ls.filter((l) => l && !l.startsWith('|t:|') && !l.startsWith('|request') && !l.startsWith('|debug')).join('\n');
    const stateDiff = filt(a.lines) !== filt(b.lines);
    if (stateDiff) anyStateDiff = true;
    details.push({ seed, aD: a.totalDraws, bD: b.totalDraws, aPer: a.perDecision, bPer: b.perDecision, match, stateDiff });
  }
  console.log(`\n### ${label}  [probed=${probed} vs control=${control}]`);
  for (const d of details) {
    console.log(`   seed ${JSON.stringify(d.seed)}: probed draws=${d.aD} ${JSON.stringify(d.aPer)} | control draws=${d.bD} ${JSON.stringify(d.bPer)} | DRAW-MATCH=${d.match} | state-differs=${d.stateDiff}`);
  }
  console.log(`   => DRAW-FREE (count identical to control): ${allMatch}   |   handler REACHABLE (state differs somewhere): ${anyStateDiff}`);
  return { allMatch, anyStateDiff };
}

(async () => {
  const results = {};

  // -------- WEATHER_SPEED: Chlorophyll (spe×2 in sun) --------
  // Make the Chlorophyll mon TIE the foe under sun (so the action-order + eachEvent
  // tie-shuffles are REACHABLE — a wrong speed changes the tie, hence the draw count).
  // Base: a Chlorophyll mon at a speed that, ×2 in sun, TIES the foe → tie-shuffle draws.
  // Control: Insomnia (a no-op speed-wise). If Chlorophyll DOUBLED speed but drew the SAME
  // count, that only means neither tied; we ALSO run a case where the ×2 CREATES a tie.
  results.chlorophyll = await compare(
    'WEATHER_SPEED Chlorophyll — ×2 spe in sun (Groudon sun; a Chlorophyll mon vs a same-speed foe)',
    (ab) => [
      // p1: Groudon (Drought → permanent sun) + a mon; p2 the Chlorophyll mon.
      [mon('Groudon', ['recover', 'recover'], { ability: 'Drought' })],
      // p2: a mon whose sun-doubled speed ties p1's Groudon. We just measure draw parity
      // between Chlorophyll and a no-op; the STATE (who moves first) will differ.
      [mon('Exeggutor', ['recover', 'recover'], { ability: ab })],
    ],
    'Chlorophyll', 'Insomnia',
    [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']],
  );

  // -------- WEATHER_SPEED: Swift Swim (spe×2 in rain) --------
  results.swiftswim = await compare(
    'WEATHER_SPEED Swift Swim — ×2 spe in rain (Kyogre rain; a Swift Swim mon vs a foe)',
    (ab) => [
      [mon('Kyogre', ['recover', 'recover'], { ability: 'Drizzle' })],
      [mon('Kingdra', ['rest', 'rest'], { ability: ab })],
    ],
    'Swift Swim', 'Shell Armor',
    [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']],
  );

  // -------- WEATHER_NEGATE: Cloud Nine --------
  // A Cloud Nine mon SUPPRESSES the sand chip (Tyranitar Sand Stream). Control: a no-op.
  // Draw-free (chip is deterministic); the STATE (HP) differs (no chip). We run under Sand.
  results.cloudnine = await compare(
    'WEATHER_NEGATE Cloud Nine — suppress sand chip (Tyranitar Sand Stream vs a Cloud Nine mon)',
    (ab) => [
      [mon('Tyranitar', ['recover', 'recover'], { ability: 'Sand Stream' })],
      // Golduck (not Rock/Ground/Steel → normally chipped) with Cloud Nine.
      [mon('Golduck', ['recover', 'recover'], { ability: ab })],
    ],
    'Cloud Nine', 'Damp', // Damp = a no-op control here (no Explosion)
    [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']],
  );

  // -------- WEATHER_NEGATE: Air Lock (Rayquaza) --------
  results.airlock = await compare(
    'WEATHER_NEGATE Air Lock — suppress sand chip (Tyranitar Sand Stream vs an Air Lock mon)',
    (ab) => [
      [mon('Tyranitar', ['recover', 'recover'], { ability: 'Sand Stream' })],
      [mon('Rayquaza', ['recover', 'recover'], { ability: ab })],
    ],
    'Air Lock', 'Damp',
    [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']],
  );

  // -------- RESIDUAL: Speed Boost (+1 spe end of turn) --------
  results.speedboost = await compare(
    'RESIDUAL Speed Boost — +1 spe at end of turn (a Speed Boost mon vs a no-op)',
    (ab) => [
      [mon('Snorlax', ['recover', 'recover'])],
      [mon('Ninjask', ['recover', 'recover'], { ability: ab })],
    ],
    'Speed Boost', 'Shell Armor',
    [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']],
  );

  // -------- RESIDUAL: Rain Dish (heal maxhp/16 in rain) --------
  results.raindish = await compare(
    'RESIDUAL Rain Dish — heal maxhp/16 in rain (Kyogre rain; a Rain Dish mon takes chip first)',
    (ab) => [
      [mon('Kyogre', ['surf', 'surf'], { ability: 'Drizzle' })], // deals damage so heal is observable
      [mon('Ludicolo', ['rest', 'rest'], { ability: ab })],
    ],
    'Rain Dish', 'Shell Armor',
    [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']],
  );

  // -------- BLOCK: Suction Cups (blocks phaze drag) --------
  // A Roar into a Suction-Cups foe: the drag is blocked → NO `sample` draw (draw-COUNT
  // change, like Protect-blocks-phaze). Control: a no-op (the drag fires → sample draws).
  // p1 has a bench so a drag is possible; p2 Roars p1... wait — we want the SUCTION CUPS mon
  // to be the TARGET of the phaze. So p2 phazes the Suction-Cups p1 mon.
  results.suctioncups = await compare(
    'BLOCK Suction Cups — phaze (Roar) into a Suction Cups mon draws NO sample (vs a no-op that gets dragged)',
    (ab) => [
      // p1: the Suction-Cups mon (the phaze TARGET) + a bench so a drag WOULD be possible.
      [mon('Cradily', ['recover', 'recover'], { ability: ab }), mon('Snorlax', ['recover', 'recover'])],
      // p2: a slow Roarer (priority -6 so it moves last, into the up p1 mon).
      [mon('Suicune', ['roar', 'roar'])],
    ],
    'Suction Cups', 'Shell Armor',
    [['move 1', 'move 1'], ['move 1', 'move 1']],
  );

  // -------- BLOCK: Soundproof (blocks sound moves) --------
  // Which MODELED move is sound? Roar/Whirlwind are NOT sound in gen3. In the modeled move
  // universe the sound moves are... none damaging. We probe with Roar (is it sound? no) —
  // instead we test whether Soundproof is even reachable by a modeled move. Uproar/Perish
  // Song/Metal Sound/GrassWhistle/Sing/Snore/Hyper Voice are sound. Of the MODELED set:
  // Sing / Grass Whistle (sleep status moves) ARE sound. So Soundproof blocks Sing.
  results.soundproof = await compare(
    'BLOCK Soundproof — Sing (a sound sleep move) into a Soundproof mon is immune (vs a no-op that sleeps)',
    (ab) => [
      [mon('Jynx', ['sing', 'sing'])],
      [mon('Electrode', ['recover', 'recover'], { ability: ab })],
    ],
    'Soundproof', 'Shell Armor',
    [['move 1', 'move 1'], ['move 1', 'move 1']],
    { fmt: 'gen3customgame' },
  );

  // -------- BLOCK: Damp (blocks Explosion/Self-Destruct — a MODELED move) --------
  results.damp = await compare(
    'BLOCK Damp — Explosion into/near a Damp mon is prevented (vs a no-op where Explosion fires)',
    (ab) => [
      [mon('Snorlax', ['explosion', 'explosion'])],
      [mon('Golduck', ['recover', 'recover'], { ability: ab })],
    ],
    'Damp', 'Shell Armor',
    [['move 1', 'move 1']],
  );

  // -------- no-op candidate: Wonder Guard (Shedinja) — type immunity gate --------
  // Wonder Guard: only super-effective moves hit. A neutral move → immune (accuracy drawn
  // THEN immune, like a type-immune move — draw model = a normal immune move). We test with
  // a NEUTRAL move (Body Slam vs Shedinja Bug/Ghost = neutral? Ghost: Normal→Ghost is 0×
  // anyway). Use Surf (Water vs Bug/Ghost = neutral) → Wonder Guard immune.
  results.wonderguard = await compare(
    'STRUCTURAL Wonder Guard — a neutral move into Shedinja is immune (draw = a normal immune move?)',
    (ab) => [
      [mon('Vaporeon', ['surf', 'surf'])],
      [mon('Shedinja', ['recover', 'recover'], { ability: ab })],
    ],
    'Wonder Guard', 'Shell Armor',
    [['move 1', 'move 1']],
  );

  // -------- no-op candidate: Inner Focus (flinch immunity) --------
  // Rock Slide (30% flinch) into an Inner Focus mon: the flinch volatile is blocked
  // (onTryAddVolatile). The secondary random(100) STILL draws (like Own Tempo's confusion
  // block — the effect is suppressed, the roll fires). Draw-free vs a no-op? PROBE.
  results.innerfocus = await compare(
    'STRUCTURAL Inner Focus — Rock Slide flinch into an Inner Focus mon (flinch blocked; does the secondary draw match?)',
    (ab) => [
      [mon('Aerodactyl', ['rockslide', 'rockslide'], { nature: 'Jolly', evs: { spe: 252 } })],
      [mon('Snorlax', ['recover', 'recover'], { ability: ab })],
    ],
    'Inner Focus', 'Shell Armor',
    [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']],
  );

  // -------- no-op candidate: Truant (every-other-turn cant) --------
  // Truant makes the mon cant every other turn. Does the cant change the draw count vs a
  // normal move? A cant means no move → no acc/crit/dmg draws that turn. So Truant is NOT a
  // no-op (it changes the draw count). PROBE to confirm.
  results.truant = await compare(
    'STRUCTURAL Truant — cant every other turn (a Truant mon vs a no-op — draw count differs?)',
    (ab) => [
      [mon('Snorlax', ['recover', 'recover'])],
      [mon('Slaking', ['bodyslam', 'bodyslam'], { ability: ab })],
    ],
    'Truant', 'Shell Armor',
    [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']],
  );

  console.log('\n\n================ SUMMARY ================');
  for (const [k, v] of Object.entries(results)) {
    console.log(`${k.padEnd(14)}: draw-free=${v.allMatch}  handler-reachable(state-differs)=${v.anyStateDiff}`);
  }
})();
