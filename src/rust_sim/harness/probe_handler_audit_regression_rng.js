// probe_handler_audit_regression_rng.js — GROUND TRUTH for the handler-completeness
// audit's two regression pins (tests/regression_test.rs):
//
//   HA1 `jump_kick_miss_crashes_the_user_with_crit_and_roll_draws`
//       (`gen3_jump_kick_crash_v1`) — Hitmonlee Jump Kick MISSES Snorlax; the resolved
//       `onMoveFail` crash halves the would-be damage onto the USER, drawing crit +
//       the 16-way roll (probe_jumpkick_crash_rng.js settled the model; this probe
//       prints the exact per-turn boundary seed + HP for the pinned seed).
//
//   HA2 `freeze_clause_blocks_the_second_freeze_in_gen3ou`
//       (`gen3_freeze_clause_v1`) — gen3ou: Ice Beam freezes Snorlax turn 1; p2
//       switches to Blissey; a later Ice Beam whose freeze secondary WOULD land is
//       BLOCKED by Freeze Clause Mod (status none, draw-identical). This probe scans
//       for a compact cooperative seed and prints the scripted turn list + boundary
//       seeds + statuses.
//
// Run:  node src/rust_sim/harness/probe_handler_audit_regression_rng.js
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
const seedStr = (b) => String(b.prng.getSeed()); // gen5 seed string "a,b,c,d"

function main() {
  console.log('=== HA1: Jump Kick miss-crash (gen3customgame) ===');
  {
    // The packed-team shapes the pin uses (EV/IV/nature identical here and in Rust):
    //   Hitmonlee|||NoAbility|jumpkick,doublekick|Hardy||N||||       (EV0/IV31)
    //   Snorlax|||NoAbility|splash,tackle|Hardy|252,,,,,|N||||       (hp 252)
    const p1 = [mon('Hitmonlee', ['jumpkick', 'doublekick'])];
    const p2 = [mon('Snorlax', ['splash', 'tackle'], { evs: { hp: 252 } })];
    const seed = [164, 21, 56, 26];
    const battle = new Battle({ formatid: 'gen3customgame', seed });
    battle.setPlayer('p1', { name: 'A', team: Teams.pack(p1) });
    battle.setPlayer('p2', { name: 'B', team: Teams.pack(p2) });
    console.log(`  init seed=${seed.join(',')}  post-lead seed=${seedStr(battle)}`);
    battle.choose('p1', 'move jumpkick');
    battle.choose('p2', 'move splash');
    const lee = battle.sides[0].active[0];
    const lax = battle.sides[1].active[0];
    console.log(`  after turn 1: seed=${seedStr(battle)}  lee hp=${lee.hp}/${lee.maxhp}  lax hp=${lax.hp}/${lax.maxhp}`);
    console.log('  log: ' + battle.log.filter((l) => /\|move\||-miss|-damage/.test(l)).join('  '));
  }

  console.log('=== HA1b: Jump Kick crash THROUGH PROTECT (gen3customgame) ===');
  {
    const p1 = [mon('Hitmonlee', ['jumpkick', 'doublekick'])];
    const p2 = [mon('Snorlax', ['protect', 'splash'], { evs: { hp: 252 } })];
    const seed = [65, 7, 11, 53]; // probe D: protect blocks (no miss), crash 146
    const battle = new Battle({ formatid: 'gen3customgame', seed });
    battle.setPlayer('p1', { name: 'A', team: Teams.pack(p1) });
    battle.setPlayer('p2', { name: 'B', team: Teams.pack(p2) });
    console.log(`  init seed=${seed.join(',')}  post-lead seed=${seedStr(battle)}`);
    battle.choose('p1', 'move jumpkick');
    battle.choose('p2', 'move protect');
    const lee = battle.sides[0].active[0];
    console.log(`  after turn 1: seed=${seedStr(battle)}  lee hp=${lee.hp}/${lee.maxhp}`);
    console.log('  log: ' + battle.log.filter((l) => /\|move\||-activate|-damage|-miss/.test(l)).join('  '));
  }

  console.log('=== HA2: Freeze Clause blocks the 2nd freeze (gen3ou) ===');
  // Scripted shape: T1 icebeam/splash, T2 splash/switch-blissey, T3.. icebeam/splash.
  // Find a seed where T1 freezes Snorlax and some later ice beam is clause-BLOCKED.
  for (let s = 0; s < 3000; s++) {
    const seed = [(s * 37 + 11) % 65536 || 1, (s * 5 + 3) % 65536 || 1, (s + 7) % 65536 || 1, (2 * s + 1) % 65536 || 1];
    const battle = new Battle({ formatid: 'gen3ou', seed });
    battle.setPlayer('p1', { name: 'A', team: Teams.pack([mon('Suicune', ['icebeam', 'splash'])]) });
    battle.setPlayer('p2', {
      name: 'B',
      team: Teams.pack([
        mon('Snorlax', ['splash', 'tackle'], { evs: { hp: 252 } }),
        mon('Blissey', ['splash', 'tackle'], { evs: { hp: 252 } }),
      ]),
    });
    const postLead = seedStr(battle);
    battle.choose('p1', 'move icebeam');
    battle.choose('p2', 'move splash');
    if (battle.sides[1].active[0].status !== 'frz') continue;
    const turns = ['T1 icebeam/splash (Snorlax FROZEN)'];
    console.log(`  post-lead seed=${postLead}`);
    battle.choose('p1', 'move splash');
    battle.choose('p2', 'switch 2');
    turns.push('T2 splash/switch-blissey');
    let blocked = false;
    for (let t = 3; t <= 8 && !battle.ended; t++) {
      const before = battle.log.length;
      battle.choose('p1', 'move icebeam');
      battle.choose('p2', 'move splash');
      turns.push(`T${t} icebeam/splash`);
      if (battle.log.slice(before).some((l) => l.includes('Freeze Clause'))) { blocked = true; break; }
      if (battle.sides[1].active[0].status === 'frz') break; // landed (bad seed)
    }
    if (!blocked) continue;
    console.log(`  seed=${seed.join(',')}  turns: ${turns.join(' | ')}`);
    console.log(`  final seed=${seedStr(battle)}  blissey status=${JSON.stringify(battle.sides[1].active[0].status)} (BLOCKED)`);
    console.log(`  snorlax(bench) status=${JSON.stringify(battle.sides[1].pokemon.find((p) => p.species.id === 'snorlax').status)}`);
    console.log('  clause lines: ' + battle.log.filter((l) => /Freeze Clause/.test(l)).join('  '));
    break;
  }
}

main();
