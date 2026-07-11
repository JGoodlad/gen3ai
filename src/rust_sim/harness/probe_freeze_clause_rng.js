// probe_freeze_clause_rng.js — settle the gen3ou FREEZE CLAUSE MOD block bit-for-bit
// (the handler-completeness audit's second real miss).
//
// THE GAP: the engine models Sleep Clause Mod (`turn.rs::try_set_status`: the 2-clause
// SetStatus shuffle + the slp block) but NOT Freeze Clause Mod — the resolved rule:
//   onSetStatus(status, target, source) {
//     if (source?.isAlly(target)) return;
//     if (status.id === "frz") for (const pokemon of target.side.pokemon)
//       if (pokemon.status === "frz") { this.add("-message", "Freeze Clause activated."); return false; }
//   }
// Under gen3ou a SECOND foe-inflicted freeze on the same side must FAIL. This probe pins:
//   A. gen3ou: freeze B1 (ice beam secondary), switch to B2, land another would-freeze
//      → the second freeze is BLOCKED (status none) and the DRAW COUNT is IDENTICAL to
//      a landed freeze (the clause returns false INSIDE the SetStatus event whose
//      2-clause shuffle already drew; frz onStart is draw-free anyway).
//   B. the FAINTED-frozen case: does a fainted mon still count for the clause?
//      (freezeclausemod has NO `pokemon.hp &&` guard, unlike sleepclausemod — probe
//      whether a fainted mon RETAINS 'frz' so the clause still blocks.)
//   C. gen3customgame control: the second freeze LANDS (no clause).
//
// Run:  node src/rust_sim/harness/probe_freeze_clause_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31,
    nature: opts.nature || 'Hardy', level: opts.level || 100, gender: opts.gender || 'N',
  };
}

// Ice-beam turns until the ACTIVE target is frozen (or turns exhausted). Returns draws
// of the FREEZING turn (the one where the secondary landed / would land).
function freezeThenSecond(format, seed) {
  const battle = new Battle({ formatid: format, seed });
  battle.setPlayer('p1', { name: 'A', team: Teams.pack([mon('Suicune', ['icebeam', 'splash'])]) });
  battle.setPlayer('p2', {
    name: 'B',
    team: Teams.pack([
      mon('Snorlax', ['splash'], { evs: { hp: 252 } }),
      mon('Blissey', ['splash'], { evs: { hp: 252 } }),
    ]),
  });
  const backend = battle.prng.rng;
  const origNext = backend.next.bind(backend);
  let draws = 0;
  backend.next = (...a) => { draws++; return origNext(...a); };

  // Phase 1: freeze the first active (Snorlax).
  let t = 0;
  while (battle.sides[1].active[0].status !== 'frz' && t < 60 && !battle.ended) {
    battle.choose('p1', 'move icebeam');
    battle.choose('p2', 'move splash');
    t++;
  }
  if (battle.sides[1].active[0].status !== 'frz') return { ok: false };
  // Phase 2: switch B to Blissey, then ice-beam until a freeze secondary WOULD land.
  battle.choose('p1', 'move splash');
  battle.choose('p2', 'switch 2');
  let t2 = 0;
  let blocked = null;
  const logMark = battle.log.length;
  while (t2 < 80 && !battle.ended) {
    const before = battle.log.length;
    draws = 0;
    battle.choose('p1', 'move icebeam');
    battle.choose('p2', 'move splash');
    t2++;
    const lines = battle.log.slice(before);
    if (lines.some((l) => l.includes('Freeze Clause'))) {
      blocked = { draws, lines: lines.filter((l) => /-status|-message|Freeze/.test(l)) };
      break;
    }
    if (battle.sides[1].active[0].status === 'frz') {
      blocked = { landed: true, draws, lines: lines.filter((l) => /-status/.test(l)) };
      break;
    }
  }
  return { ok: true, blocked, active2: battle.sides[1].active[0].status || 'none', log: battle.log.slice(logMark) };
}

function main() {
  console.log('=== A: gen3ou — second foe freeze on the same side ===');
  for (let s = 0; s < 6; s++) {
    const seed = [s * 31 + 3, s * 7 + 1, s + 13, 2 * s + 9].map((x) => (x % 65536) || 1);
    const r = freezeThenSecond('gen3ou', seed);
    if (!r.ok) { console.log(`  seed=${seed}: first freeze never landed (skip)`); continue; }
    if (!r.blocked) { console.log(`  seed=${seed}: second freeze never attempted (skip)`); continue; }
    console.log(`  seed=${seed}: ${r.blocked.landed ? 'SECOND FREEZE LANDED (unexpected!)' : 'BLOCKED'} draws(turn)=${r.blocked.draws}  ${r.blocked.lines.join('  ')}`);
  }
  console.log('=== C: gen3customgame control — second freeze should LAND (no clause) ===');
  for (let s = 0; s < 4; s++) {
    const seed = [s * 31 + 3, s * 7 + 1, s + 13, 2 * s + 9].map((x) => (x % 65536) || 1);
    const r = freezeThenSecond('gen3customgame', seed);
    if (!r.ok || !r.blocked) { console.log(`  seed=${seed}: (skip)`); continue; }
    console.log(`  seed=${seed}: ${r.blocked.landed ? 'LANDED (expected)' : 'BLOCKED (unexpected!)'}`);
  }

  console.log('=== B: does a FAINTED mon retain frz for the clause? (gen3ou) ===');
  // Freeze Snorlax, then Superpower it to KO, then try to freeze Blissey.
  for (let s = 0; s < 8; s++) {
    const seed = [s * 17 + 5, s * 3 + 7, s + 29, 4 * s + 1].map((x) => (x % 65536) || 1);
    const battle = new Battle({ formatid: 'gen3ou', seed });
    battle.setPlayer('p1', { name: 'A', team: Teams.pack([mon('Suicune', ['icebeam', 'splash'], { evs: { spa: 252 } })]) });
    battle.setPlayer('p2', {
      name: 'B',
      team: Teams.pack([
        mon('Smeargle', ['splash']), // frail: freezes then dies to ice beams
        mon('Blissey', ['splash'], { evs: { hp: 252 } }),
      ]),
    });
    // freeze Smeargle
    let t = 0;
    while (battle.sides[1].active[0].status !== 'frz' && t < 40 && !battle.ended) {
      battle.choose('p1', 'move icebeam');
      battle.choose('p2', 'move splash');
      t++;
    }
    if (battle.sides[1].active[0].status !== 'frz') continue;
    // KO the frozen Smeargle with more ice beams
    let t3 = 0;
    while (!battle.sides[1].active[0].fainted && battle.sides[1].active[0].hp > 0 && t3 < 20 && !battle.ended) {
      battle.choose('p1', 'move icebeam');
      battle.choose('p2', 'move splash');
      t3++;
    }
    const smeargle = battle.sides[1].pokemon.find((p) => p.species.id === 'smeargle');
    if (!smeargle || smeargle.hp > 0) continue;
    console.log(`  seed=${seed}: fainted Smeargle status=${JSON.stringify(smeargle.status)} (frz retained? ${smeargle.status === 'frz'})`);
    // replacement comes in; now try to freeze Blissey
    if (battle.requestState === 'switch') battle.choose('p2', 'switch 2');
    let t4 = 0;
    let verdict = 'no second attempt';
    while (t4 < 80 && !battle.ended) {
      const before = battle.log.length;
      battle.choose('p1', 'move icebeam');
      battle.choose('p2', 'move splash');
      t4++;
      const lines = battle.log.slice(before);
      if (lines.some((l) => l.includes('Freeze Clause'))) { verdict = 'BLOCKED by clause (fainted counts)'; break; }
      if (battle.sides[1].active[0].status === 'frz') { verdict = 'LANDED (fainted does NOT count)'; break; }
    }
    console.log(`    second freeze: ${verdict}`);
    break; // one decisive sample is enough; loop only to find a cooperative seed
  }
}

main();
