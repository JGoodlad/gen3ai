// probe_statusimmune_rng.js — THE STATUS_IMMUNE DRAW-MODEL PROBE (the only oracle).
//
// For each gen3 STATUS_IMMUNE member (Limber/Insomnia/Immunity/Water Veil/Vital Spirit via
// onSetStatus, Magma Armor via onImmunity) it fires the matching status MOVE into the
// immune-ability holder and counts the EXACT PRNG draws for that decision, in BOTH formats
// (gen3ou = the 2 clause SetStatus handlers, gen3customgame = 0 clause handlers), versus a
// non-immune CONTROL (a plain-ability target that ACTUALLY gets statused).
//
// THE CRUX being settled per member:
//  - onSetStatus member: it ADDS a 3rd SetStatus-event handler in gen3ou → the tie-shuffle
//    grows from size-2 (control) to size-3 → ONE MORE draw than the control. In gen3customgame
//    it makes the ONLY handler → 1 handler → NO tie → NO shuffle → DRAW-FREE (same draw count
//    as a type-immune target: accuracy only).
//  - onImmunity member (Magma Armor): it blocks at runStatusImmunity BEFORE the SetStatus
//    event → NO SetStatus handler → gen3ou stays size-2 (SAME draws as the control), customgame
//    no shuffle (SAME as control). It is DRAW-IDENTICAL to a landed status in both formats.
//
// The two moves are drawn so the ONLY per-move draw is accuracy (never_miss=false, acc-100),
// so the difference in total draws == the SetStatus-shuffle delta.
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

// Run one decision (p1 uses `move`, p2 does nothing legal), count raw PRNG draws.
async function run(fmt, p1team, p2team, p1move, p2move, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"${fmt}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  let draws = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = (...a) => { draws += 1; return realNext(...a); };
  streams.omniscient.write(`>p1 ${p1move}`);
  streams.omniscient.write(`>p2 ${p2move}`);
  for (let k = 0; k < 12; k++) await tick();
  const tgt = battle.sides[1].active[0];
  const seedAfter = battle.prng.seed || (battle.prng.rng && battle.prng.rng.seed);
  return { draws, tgtStatus: tgt.status || '', battle };
}

// Each member: [ability, immuneMove, statusId, immuneSpecies, controlAbility].
// The immune species is chosen to NOT be naturally type-immune to the status (so the ONLY
// block is the ability). controlAbility is a plain no-op so the control ACTUALLY gets statused.
const MEMBERS = [
  ['Limber',       'thunderwave', 'par', 'Snorlax',  'Pressure'],   // onSetStatus
  ['Insomnia',     'spore',       'slp', 'Snorlax',  'Pressure'],   // onSetStatus
  ['Vital Spirit', 'spore',       'slp', 'Snorlax',  'Pressure'],   // onSetStatus
  ['Immunity',     'toxic',       'tox', 'Snorlax',  'Pressure'],   // onSetStatus (psn/tox)
  ['Water Veil',   'willowisp',   'brn', 'Snorlax',  'Pressure'],   // onSetStatus
  ['Magma Armor',  'willowisp',   'brn', 'Snorlax',  'Pressure'],   // onImmunity? NO — MA is frz.
];
// Fix Magma Armor: its immunity is FREEZE. There is no reliable single-target freeze STATUS
// move in gen3 (Ice-type moves freeze only as a SECONDARY). So probe Magma Armor via a
// secondary-freeze damaging move (Ice Beam) into a non-Ice target, comparing MA vs a plain
// control — that isolates the onImmunity('frz') block at runStatusImmunity.
MEMBERS[5] = ['Magma Armor', '__icebeam__', 'frz', 'Snorlax', 'Pressure'];

const MOVE_ACC = { thunderwave: 'thunderwave', spore: 'spore', toxic: 'toxic', willowisp: 'willowisp' };

(async () => {
  const seed = [3, 5, 7, 11];
  for (const fmt of ['gen3customgame', 'gen3ou']) {
    console.log(`\n================ FORMAT ${fmt} ================`);
    for (const [ability, mv, statusId, species, ctrlAbility] of MEMBERS) {
      if (mv === '__icebeam__') {
        // Magma Armor: Ice Beam (10% frz secondary) into MA-holder vs control. Force many
        // seeds until a seed where the freeze secondary FIRES (random(100)<10) so we compare
        // the frz-application decision. Report draws for BOTH.
        let done = false;
        for (let s = 0; s < 400 && !done; s++) {
          const sd = [s + 1, 2, 3, 4];
          // Attacker (p1) Ice Beam; target (p2) MA-Snorlax (Ice-neutral) sits with Rest.
          const immTeam = [mon('Articuno', ['icebeam'])];
          const immTgt = [mon(species, ['rest', 'bodyslam'], { ability })];
          const ctrlTgt = [mon(species, ['rest', 'bodyslam'], { ability: ctrlAbility })];
          const imm = await run(fmt, immTeam, immTgt, 'move 1', 'move 1', sd);
          const ctrl = await run(fmt, immTeam, ctrlTgt, 'move 1', 'move 1', sd);
          // We want a seed where the CONTROL froze (secondary fired) — then compare.
          if (ctrl.tgtStatus === 'frz') {
            console.log(`  ${ability.padEnd(13)} [Ice Beam frz secondary, seed ${JSON.stringify(sd)}]`);
            console.log(`      immune(${ability}):  draws=${imm.draws}  targetStatus='${imm.tgtStatus}'`);
            console.log(`      control(${ctrlAbility}): draws=${ctrl.draws}  targetStatus='${ctrl.tgtStatus}'`);
            console.log(`      => DELTA draws (immune - control) = ${imm.draws - ctrl.draws}`);
            done = true;
          }
        }
        if (!done) console.log(`  ${ability}: no seed froze the control in 400 tries (widen).`);
        continue;
      }
      // Status-move members: fire the move into the immune holder vs the plain control.
      const attacker = [mon('Blissey', [mv, 'softboiled'])];
      const immTgt = [mon(species, ['rest', 'bodyslam'], { ability })];
      const ctrlTgt = [mon(species, ['rest', 'bodyslam'], { ability: ctrlAbility })];
      const imm = await run(fmt, attacker, immTgt, 'move 1', 'move 1', seed);
      const ctrl = await run(fmt, attacker, ctrlTgt, 'move 1', 'move 1', seed);
      console.log(`  ${ability.padEnd(13)} [${mv} -> ${statusId}]`);
      console.log(`      immune(${ability}):  draws=${imm.draws}  targetStatus='${imm.tgtStatus}'`);
      console.log(`      control(${ctrlAbility}): draws=${ctrl.draws}  targetStatus='${ctrl.tgtStatus}'`);
      console.log(`      => DELTA draws (immune - control) = ${imm.draws - ctrl.draws}`);
    }
  }
})();
