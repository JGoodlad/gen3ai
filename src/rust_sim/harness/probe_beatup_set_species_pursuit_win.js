// probe_beatup_set_species_pursuit_win.js — ground truth for the two M6 cutover-stress classes
// closed on branch `m6-fuzz3` (2026-09-25), recorded through the REAL Showdown sim as
// standalone `ab_replay` chunks (SCEN/TEAM/FMT/INIT/DEC/END + the omniscient `L` rows).
//
//   BU — `gen3_beatup_set_species_v1`: gen-3 Beat Up reads BOTH base stats from the SET species
//        (`dex.species.get(<mon>.set.species).baseStats`, data/mods/gen3/moves.ts:48/53), so a
//        TRANSFORMED target is struck at its OWN base Def and a TRANSFORMED user's own strike is
//        at its OWN base Atk. Houndoom Beat Up into a Smeargle that Transformed into it (target
//        side: Smeargle Def 35, not Houndoom's 50), while the transformed Smeargle Beat Ups back
//        (user side: its own strike at Smeargle Atk 20, not Houndoom's 90).
//   PW — `gen3_pursuitfaint_win_stops_switch_v1`: a Pursuit KO of a switching mon whose Destiny
//        Bond takes the pursuer — the foe's LAST mon — ends the battle inside the runAction tail
//        (sim/battle.ts:2856-2857), so the re-queued gen 2-4 switch (battle.ts:2787-2794) never
//        runs: no `|switch|` before `|win|`.
//   PC — the CONTROL: the same board with a bench behind the pursuer; the battle does not end,
//        so the switch STILL brings Skarmory in (the `-hint`) and p2 then replaces.
//
// Each scenario ASSERTS the sim fact it exists for (fail-loud), then writes its chunk.
// Usage (from src/rust_sim):  node harness/probe_beatup_set_species_pursuit_win.js [outDir]
//   no outDir → assert only. The committed fixtures 81-83 in
//   tests/vectors/byte_fuzz_corpus/ were written by this script.
'use strict';
const fs = require('fs');
const path = require('path');
const e2e = require('./gen_e2e_fuzz.js');

const FMT = 'gen3ou';
const SEED = [1, 2, 3, 4];

const SCEN = {
  bu_transformed_beatup: {
    fixture: '81_beatup_reads_the_set_species_under_transform.txt',
    title: 'BEAT UP reads BOTH base stats from the SET species under Transform (gen3_beatup_set_species_v1)',
    p1: 'Smeargle||Leftovers|OwnTempo|Transform,Splash,Spikes,Protect|Jolly|,,,,,252|M||||]' +
        'Snorlax||Leftovers|ThickFat|BodySlam,Curse,Rest,SleepTalk|Adamant|252,252,,,4,|M||||]' +
        'Skarmory||Leftovers|KeenEye|DrillPeck,Spikes,Roar,Protect|Impish|252,,252,,4,|M||||',
    p2: 'Houndoom||Leftovers|EarlyBird|BeatUp,Crunch,FireBlast,Protect|Timid|,,,252,4,252|M||||]' +
        'Celebi||Leftovers|NaturalCure|Psychic,Recover,CalmMind,GigaDrain|Bold|252,,252,,4,|N||||]' +
        'Metagross||Leftovers|ClearBody|MeteorMash,Earthquake,Explosion,Agility|Adamant|252,252,,,4,|N||||',
    // t1: Houndoom Beat Up (untransformed Smeargle), Smeargle Transforms into Houndoom.
    // t2/t3: both Beat Up (slot 0 of the copied moveset).
    choices: [['m0', 'm0'], ['m0', 'm0'], ['m0', 'm0']],
    check(lines) {
      const i = lines.findIndex(l => l.startsWith('|-transform|p1a: Smeargle|p2a: Houndoom'));
      if (i < 0) throw new Error('BU: Smeargle did not Transform into Houndoom');
      const after = lines.slice(i);
      // gen3ou carries Beat Up Nicknames Mod, so the per-strike `-activate` is suppressed: read
      // the `-hitcount` lines (3 healthy party members a side = 3 strikes each).
      if (!after.includes('|-hitcount|p2a: Houndoom|3') || !after.includes('|-hitcount|p1a: Smeargle|3')) {
        throw new Error('BU: both Beat Ups must strike 3 times after the Transform');
      }
    },
  },
  pw_pursuit_destiny_bond_win: {
    fixture: '82_pursuit_faint_that_ends_the_battle_cancels_the_switch.txt',
    title: 'a Pursuit KO whose Destiny Bond takes the foe\'s LAST mon ends the battle: no |switch| before |win| (gen3_pursuitfaint_win_stops_switch_v1)',
    p1: 'Gengar||Leftovers|Levitate|DestinyBond,Thunderbolt,IcePunch,GigaDrain|Timid|252,,,4,,252|M||||]' +
        'Skarmory||Leftovers|KeenEye|DrillPeck,Spikes,Roar,Protect|Impish|252,,252,,4,|M||||',
    p2: 'Tyranitar||Leftovers|SandStream|Pursuit,Crunch,RockSlide,Earthquake|Adamant|4,252,,,,252|M||||',
    // t1: Gengar Thunderbolt, Tyranitar Crunch (chips Gengar into Pursuit-KO range).
    // t2: Gengar Destiny Bond, Tyranitar Earthquake (Levitate immune).
    // t3: Gengar switches to Skarmory, Tyranitar Pursuit (x2 on the switch) KOs it; DB takes Tyranitar.
    choices: [['m1', 'm1'], ['m0', 'm3'], ['s1', 'm0']],
    check(lines) {
      const f = lines.indexOf('|faint|p2a: Tyranitar');
      const w = lines.indexOf('|win|P1');
      if (f < 0 || w < 0) throw new Error('PW: Destiny Bond must take Tyranitar and P1 must win');
      if (!lines.includes('|-activate|p1a: Gengar|move: Destiny Bond')) throw new Error('PW: no Destiny Bond activate');
      if (lines.slice(f, w).some(l => l.startsWith('|switch|'))) throw new Error('PW: a |switch| ran before |win|');
    },
  },
  pc_pursuit_destiny_bond_continues: {
    fixture: '83_pursuit_faint_that_does_not_end_the_battle_still_switches.txt',
    title: 'CONTROL for 82 — the same Pursuit + Destiny Bond with a bench behind the pursuer: the chosen switch still brings Skarmory in',
    p1: 'Gengar||Leftovers|Levitate|DestinyBond,Thunderbolt,IcePunch,GigaDrain|Timid|252,,,4,,252|M||||]' +
        'Skarmory||Leftovers|KeenEye|DrillPeck,Spikes,Roar,Protect|Impish|252,,252,,4,|M||||',
    p2: 'Tyranitar||Leftovers|SandStream|Pursuit,Crunch,RockSlide,Earthquake|Adamant|4,252,,,,252|M||||]' +
        'Blissey||Leftovers|NaturalCure|SeismicToss,SoftBoiled,ThunderWave,IceBeam|Bold|252,,252,,4,|F||||',
    // t1-t3 as in PW; then p2's forced replacement (Blissey) and one more turn.
    choices: [['m1', 'm1'], ['m0', 'm3'], ['s1', 'm0'], ['-', 's1'], ['m0', 'm0']],
    check(lines) {
      const f = lines.indexOf('|faint|p2a: Tyranitar');
      if (f < 0) throw new Error('PC: Destiny Bond must take Tyranitar');
      if (!lines.slice(f).some(l => l.startsWith('|switch|p1a: Skarmory'))) throw new Error('PC: the switch must continue');
      if (lines.some(l => l.startsWith('|win|'))) throw new Error('PC: the battle must not end');
    },
  },
};

(async () => {
  const outDir = process.argv[2];
  let fails = 0;
  for (const [id, s] of Object.entries(SCEN)) {
    const rec = await e2e.runBattle(s.p1, s.p2, SEED, 1, 'modeled',
      { replayChoices: s.choices, format: FMT, allowHiddenPower: true });
    const lines = rec.lines || [];
    try {
      if (rec.decisions.length !== s.choices.length) {
        throw new Error(`${id}: ${rec.decisions.length} decisions replayed, script has ${s.choices.length}`);
      }
      s.check(lines);
      console.log(`OK   ${id}: ${rec.decisions.length} decisions, ended=${rec.ended} winner=${rec.winner}`);
    } catch (e) {
      fails++;
      console.log(`FAIL ${e.message}`);
      continue;
    }
    if (outDir) {
      const out = [
        `# FIXTURE ${s.fixture.slice(0, 2)} — ${s.title}.`,
        `# Source: harness/probe_beatup_set_species_pursuit_win.js scenario \`${id}\` (a CONSTRUCTED battle recorded through the real sim).`,
        '# ab_fuzz chunk — A/B differential fuzzer (real Showdown sim vs the Rust port).',
        '# Replay: target/selfcheck/ab_replay <this-file>   (one JSON verdict per battle)',
      ];
      e2e.emitBattle(out, id, s.p1, s.p2, rec, { protocol: true });
      fs.writeFileSync(path.join(outDir, s.fixture), out.join('\n') + '\n');
    }
  }
  if (fails) { console.log(`${fails} scenario(s) FAILED`); process.exit(1); }
})().catch(e => { console.error(e); process.exit(1); });
