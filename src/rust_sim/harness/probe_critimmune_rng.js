// probe_critimmune_rng.js — settle the CRIT_IMMUNE (Battle Armor / Shell Armor) draw model.
//
// THE CRUX: does a crit-immune defender make the ATTACKER SKIP the crit `randomChance`
// roll (a draw-COUNT change) or is the roll DRAWN then the crit OVERRIDDEN to false
// (draw-FREE)?
//
// The RESOLVED source (base battle-actions.ts:1642-1651, gen3 inherits — the gen3 mod
// overrides only `modifyDamage`, NOT the crit determination) says:
//   moveHit.crit = move.willCrit || false;
//   if (move.willCrit === undefined) {
//     if (critRatio) { moveHit.crit = this.battle.randomChance(1, critMult[critRatio]); }  // ALWAYS drawn
//   }
//   if (moveHit.crit) { moveHit.crit = this.battle.runEvent('CriticalHit', target, null, move); }  // onCriticalHit=false → false
// And gen3 battlearmor/shellarmor resolve `onCriticalHit = false` (a boolean, not a fn).
// So the ROLL is drawn regardless; only the resulting boolean is overridden to false when
// the roll succeeded → DRAW-FREE. This probe PROVES it against the actual PRNG.
//
// Method: run a fixed battle (a HIGH-CRIT physical move each side) at fixed seeds, with the
// defender holding Battle Armor vs a no-op ability (Sturdy), counting the sim's raw
// prng.next() draws AND observing the emitted `|-crit|` lines. If the counts are IDENTICAL
// and NO `|-crit|` is ever emitted against the crit-immune mon (while the control DOES get
// crit), CRIT_IMMUNE is draw-free.
//
// Run: node src/rust_sim/harness/probe_critimmune_rng.js

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
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31, nature: 'Serious', level: 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

// Count raw prng draws over a fixed battle where p1 attacks p2 with a HIGH-CRIT physical
// move (Slash, critRatio 2 → 1/8). p2's ability is the variable (Battle Armor vs a no-op
// control). We record `|-crit|<p2>` emissions (crit ON the crit-immune defender) + total
// draws + per-turn draws.
async function run(defAbility, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const critLines = [];
  (async () => {
    for await (const ch of streams.omniscient) {
      for (const ln of String(ch).split('\n')) {
        if (ln.startsWith('|-crit|')) critLines.push(ln);
      }
    }
  })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
  // A WEAK high-crit move (Leer is Status — instead use Karate Chop, critRatio 2, BP 50,
  // Fighting) into a super-bulky RESISTED defender so the fight runs MANY turns and MANY
  // crit rolls fire against the crit-immune mon. Snorlax (Normal) resists nothing, but a
  // low-attack Persian + Karate Chop (Fighting SE vs Normal but low BP) still lets Snorlax
  // survive long enough to expose ~15+ crit rolls.
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([mon('Persian', ['karatechop', 'karatechop'], { evs: {} })]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([mon('Snorlax', ['recover', 'recover'], { ability: defAbility, evs: { hp: 252, def: 252 } })]) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;

  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  let nextCount = 0;
  rng.next = (...a) => { nextCount += 1; return realNext(...a); };

  const seedTrace = [];
  for (let t = 0; t < 25; t++) {
    const before = nextCount;
    streams.omniscient.write('>p1 move 1');
    streams.omniscient.write('>p2 move 1');
    for (let k = 0; k < 10; k++) await tick();
    seedTrace.push(nextCount - before);
    const p2 = battle.sides[1].active[0];
    if (battle.ended || p2.hp === 0 || battle.sides[0].active[0].hp === 0) break;
  }
  // critLines against p2a (Snorlax = the defender).
  const critOnDef = critLines.filter((l) => l.includes('p2a'));
  return { totalDraws: nextCount, perTurn: seedTrace, critOnDef: critOnDef.length, ended: battle.ended };
}

(async () => {
  const seeds = [[1, 2, 3, 4], [7, 11, 13, 17], [100, 200, 300, 400], [5, 5, 5, 5], [42, 42, 42, 42], [9, 8, 7, 6]];
  console.log('=== CRIT_IMMUNE draw-model probe (Slash into a Battle-Armor vs Sturdy Snorlax) ===');
  let allMatch = true;
  let sawCritOnControl = 0;
  let sawCritOnBattleArmor = 0;
  for (const seed of seeds) {
    const ba = await run('Battle Armor', seed);
    const ctl = await run('Sturdy', seed); // Sturdy = a no-op here (no OHKO moves)
    const match = ba.totalDraws === ctl.totalDraws && JSON.stringify(ba.perTurn) === JSON.stringify(ctl.perTurn);
    if (!match) allMatch = false;
    sawCritOnControl += ctl.critOnDef;
    sawCritOnBattleArmor += ba.critOnDef;
    console.log(`  seed ${JSON.stringify(seed)}:`);
    console.log(`    BattleArmor: draws=${ba.totalDraws} perTurn=${JSON.stringify(ba.perTurn)} critOnDef=${ba.critOnDef}`);
    console.log(`    Sturdy(ctl): draws=${ctl.totalDraws} perTurn=${JSON.stringify(ctl.perTurn)} critOnDef=${ctl.critOnDef}`);
    console.log(`    DRAW-COUNT MATCH: ${match}`);
  }
  console.log('');
  console.log(`SUMMARY: draw-count identical across all seeds = ${allMatch}`);
  console.log(`         control (Sturdy) crit the defender N times = ${sawCritOnControl}`);
  console.log(`         Battle Armor crit N times                  = ${sawCritOnBattleArmor}`);
  console.log('');
  console.log(allMatch && sawCritOnControl > 0 && sawCritOnBattleArmor === 0
    ? '=> CONFIRMED: CRIT_IMMUNE is DRAW-FREE (roll drawn, crit overridden to false). Control crits, Battle Armor never does, IDENTICAL draw count.'
    : '=> NOT confirmed as the clean draw-free-with-override model — inspect above.');
})();
