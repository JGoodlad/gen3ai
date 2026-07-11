// probe_statusimmune_magmaarmor.js — SETTLE Magma Armor's frz-immunity draw model.
//
// MA blocks freeze via onImmunity('frz') at runStatusImmunity (BEFORE the SetStatus event),
// like the Sun-freeze gate. It should NOT add a SetStatus handler. To reach a FREEZE
// application we use Ice Beam (10% frz secondary) and sweep seeds until the freeze secondary
// FIRES on a NON-Ice target (so the ONLY block is MA). We compare, on the SAME seed:
//   (a) MA target: the freeze secondary random(100) fires, MA blocks → target UNfrozen.
//   (b) plain control (Insomnia — slp-immune, irrelevant to frz): target FROZEN.
// The SetStatus/Immunity event handler counts + the per-event draws must show MA blocks at
// Immunity with the SetStatus handler count UNCHANGED (so the draw count is identical).
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return { species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0 }, ivs: IV31, nature: 'Serious', level: 100, gender: 'N' };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function probe(fmt, targetAbility, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"${fmt}","seed":${JSON.stringify(seed)}}`);
  // Attacker Ice Beam; target a Normal-type (Snorlax) so no Ice type-immunity — the only
  // block is the ability. Target sits (Body Slam into the attacker; irrelevant).
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([mon('Regice', ['icebeam'])]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([mon('Snorlax', ['bodyslam', 'rest'], { ability: targetAbility })]) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  let cur = null;
  const events = [];
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = (...a) => { if (cur) cur.draws += 1; return realNext(...a); };
  const realRun = battle.runEvent.bind(battle);
  battle.runEvent = function (eventid, ...rest) {
    if (eventid === 'SetStatus' || eventid === 'Immunity') {
      let n = -1;
      try { n = battle.findEventHandlers(rest[0], eventid, rest[1]).length; } catch (e) { n = `err`; }
      const prev = cur; cur = { eventid, n, draws: 0 };
      const ret = realRun(eventid, ...rest);
      cur.ret = typeof ret === 'object' && ret ? (ret.id || 'obj') : ret;
      events.push(cur); cur = prev; return ret;
    }
    return realRun(eventid, ...rest);
  };
  streams.omniscient.write('>p1 move 1');
  streams.omniscient.write('>p2 move 1');
  for (let k = 0; k < 12; k++) await tick();
  return { events, tgtStatus: (battle.sides[1].active[0].status || '') };
}

(async () => {
  for (const fmt of ['gen3customgame', 'gen3ou']) {
    console.log(`\n=== ${fmt} ===`);
    // Find a seed where the control (Insomnia — frz-irrelevant) FREEZES.
    let found = null;
    for (let s = 0; s < 800; s++) {
      const seed = [s + 1, 2, 3, 4];
      const ctrl = await probe(fmt, 'Insomnia', seed);
      if (ctrl.tgtStatus === 'frz') {
        const ma = await probe(fmt, 'Magma Armor', seed);
        found = { seed, ctrl, ma };
        break;
      }
    }
    if (!found) { console.log('  no freezing seed in 800 tries'); continue; }
    const fmtEv = (r) => r.events.map((e) => `${e.eventid}(h=${e.n},draws=${e.draws},ret=${e.ret})`).join('  ');
    console.log(`  seed ${JSON.stringify(found.seed)}`);
    console.log(`    control(Insomnia): tgt='${found.ctrl.tgtStatus}'  ${fmtEv(found.ctrl)}`);
    console.log(`    MagmaArmor:        tgt='${found.ma.tgtStatus}'  ${fmtEv(found.ma)}`);
  }
})();
