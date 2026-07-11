// probe_accuracy_intguard.js — the runEvent integer-guard on chainModify acc mods.
//
// runEvent (battle.ts:929) applies the accumulated chainModify via
// `this.modify(relayVar, event.modifier)` ONLY if relayVar is a non-negative
// INTEGER (`relayVar === Math.abs(Math.floor(relayVar))`). The gen3 accuracy
// chainModify members are Compound Eyes(1.3)/Sand Veil(0.8)/Hustle(3277/4096);
// the direct-multiply members are Bright Powder(*0.9)/Lax Incense(*0.95).
//
// This probes the CROSS cases where accuracy is non-integer WHEN the chainModify
// member's contribution would apply — to confirm the guard SKIPS it, and to
// settle the exact end-to-end effAcc when a chainModify member coexists with a
// stage (float) or a direct-multiply member (float).
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return { species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: 'N' };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function scenario(desc, p1mon, p2mon, p1move, setup) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) {} })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":[1,2,3,4]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([p1mon]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([p2mon]) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  if (setup) setup(battle);
  let logging = false;
  const rc = [];
  const realRC = battle.prng.randomChance.bind(battle.prng);
  battle.prng.randomChance = function (num, den) {
    const res = realRC(num, den);
    if (logging && den === 100) rc.push(num);
    return res;
  };
  logging = true;
  streams.omniscient.write(`>p1 move ${p1move}`);
  streams.omniscient.write(`>p2 move 1`);
  for (let k = 0; k < 20; k++) await tick();
  logging = false;
  console.log(`${desc}\n   -> effAcc reaching randomChance(_,100): ${JSON.stringify(rc[0])}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}
function setB(b, side, key, val) { b.sides[side].active[0].boosts[key] = val; }

async function main() {
  console.log('=== INTEGER-GUARD probe: chainModify acc mod × float accuracy ===\n');

  // A) Compound Eyes(1.3, chainModify) on Thunder70, with target eva -1 (float 93.33)
  //    Without guard: 93.33 * 1.3 = 121.33.  WITH guard (93.33 non-int): chainModify SKIPPED -> 93.33.
  await scenario('A) Compound Eyes + Thunder70 + target eva-1 (float 93.33): guard should SKIP the 1.3',
    mon('Butterfree', ['thunder'], { ability: 'Compound Eyes' }),
    mon('Snorlax', ['tackle'], { evs: { hp: 252 } }),
    1, (b) => setB(b, 1, 'evasion', -1));

  // B) Compound Eyes(1.3) + Thunder70 + Bright Powder on... same mon? No, CE is attacker, BP defender.
  //    Thunder70: CE chainModify 1.3 accumulates; BP direct *0.9 mutates relayVar to float 63.
  //    Order: CE(prio9) chainModify -> BP(prio5) returns 70*0.9=63 (float? 63.0 integer!) -> end modify(63,1.3).
  //    63 IS integer -> modify(63,1.3) = 81. Let's confirm.
  await scenario('B) Compound Eyes(attacker) + Bright Powder(defender) + Thunder70: CE chain then BP*0.9=63(int) then modify(63,1.3)=81?',
    mon('Butterfree', ['thunder'], { ability: 'Compound Eyes' }),
    mon('Snorlax', ['tackle'], { item: 'brightpowder', evs: { hp: 252 } }),
    1, null);

  // C) Compound Eyes(1.3) + Lax Incense(*0.95) + Thunder70:
  //    CE chain 1.3; Lax returns 70*0.95=66.5 (FLOAT) -> end guard SKIPS modify -> effAcc 66.5.
  await scenario('C) Compound Eyes(attacker) + Lax Incense(defender) + Thunder70: Lax makes 66.5 FLOAT -> CE 1.3 SKIPPED -> 66.5',
    mon('Butterfree', ['thunder'], { ability: 'Compound Eyes' }),
    mon('Snorlax', ['tackle'], { item: 'laxincense', evs: { hp: 252 } }),
    1, null);

  // D) Sand Veil(0.8, chainModify) + Bright Powder(*0.9) on the SAME defender, in sand, Tackle95:
  //    Both target handlers. SV(prio8) chainModify 0.8; BP(prio5) returns 95*0.9=85.5 (FLOAT)
  //    -> end guard SKIPS the SV 0.8 -> effAcc 85.5.
  await scenario('D) Sand Veil + Bright Powder same defender in sand + Tackle95: BP*0.9=85.5 FLOAT -> SV 0.8 SKIPPED -> 85.5',
    mon('Tauros', ['tackle'], {}),
    mon('Cacturne', ['tackle'], { ability: 'Sand Veil', item: 'brightpowder', evs: { hp: 252 } }),
    1, (b) => { b.field.setWeather('sandstorm', b.sides[0].active[0]); });

  // E) Hustle(3277/4096) + attacker acc +1 (float 126.66) on Tackle95:
  //    acc+1 makes 95*4/3=126.66 (FLOAT); Hustle chainModify 3277/4096 at end guard SKIPPED -> 126.66.
  await scenario('E) Hustle + Tackle95 + attacker acc+1 (float 126.66): guard SKIPS Hustle 0.8 -> 126.66',
    mon('Aerodactyl', ['tackle'], { ability: 'Hustle', evs: { spe: 252 }, nature: 'Jolly' }),
    mon('Snorlax', ['tackle'], { evs: { hp: 252 } }),
    1, (b) => setB(b, 0, 'accuracy', 1));

  // F) Sand Veil(0.8) + Compound Eyes(attacker) both chainModify, integer base Tackle95 in sand:
  //    Both accumulate into ONE event.modifier: 0.8 * 1.3 chained. 95 integer -> modify at end.
  //    chain: prevMod=1 -> CE: nextMod=tr(1.3*4096)=5324; mod=((4096*5324+2048)>>12)/4096 = 5324/4096.
  //           -> SV: nextMod=tr(0.8*4096)=3276; mod=((5324*3276+2048)>>12)/4096.
  await scenario('F) Sand Veil(0.8) + Compound Eyes(1.3) both chainModify, Tackle95 in sand (accumulate into ONE modifier)',
    mon('Butterfree', ['tackle'], { ability: 'Compound Eyes' }),
    mon('Cacturne', ['tackle'], { ability: 'Sand Veil', evs: { hp: 252 } }),
    1, (b) => { b.field.setWeather('sandstorm', b.sides[0].active[0]); });

  console.log('\n=== INTEGER-GUARD PROBE COMPLETE ===');
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
