// gen_soundproof_corpus.js — generate the byte-fuzz corpus fixture guarding the DAMAGING
// Soundproof `-immune` form (a Hyper Voice into a Soundproof holder). A one-off producer:
// constructs a Snorlax(HyperVoice/BodySlam) vs Mr.Mime(Soundproof) battle, drives it via the
// e2e recorder's runBattle (replayChoices), and emits the chunk in the corpus battle.txt format.
'use strict';
const fs = require('fs');
const path = require('path');
const { runBattle, emitBattle } = require('./gen_e2e_fuzz.js');
const { Teams } = require(path.resolve(__dirname, '../../../deps/pokemon-showdown/dist/sim'));

function pack(sets) { return Teams.pack(sets); }
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, ability, moves) {
  return { species, item: '', ability, moves, evs: EV0, ivs: IV31, nature: 'Serious', level: 100, gender: 'N' };
}

(async () => {
  const p1 = pack([mon('Snorlax', 'Immunity', ['hypervoice', 'bodyslam'])]);
  const p2 = pack([mon('Mr. Mime', 'Soundproof', ['calmmind', 'tackle'])]);
  // Turn 1: Hyper Voice (BLOCKED by Soundproof) + Calm Mind; then Body Slam to a KO (p1 wins).
  const choices = [['m0', 'm0'], ['m1', 'm0'], ['m1', 'm0'], ['m1', 'm0'], ['m1', 'm0'], ['m1', 'm0']];
  const seed = [24, 68, 137, 9];
  const rec = await runBattle(p1, p2, seed, 0, 'pool', { replayChoices: choices, format: 'gen3customgame' });
  const lines = [];
  lines.push('# ab_fuzz chunk — A/B differential fuzzer (real Showdown sim vs the Rust port).');
  lines.push('# Guards the DAMAGING Soundproof `-immune` form (gen3_ability_batch2_v1): a Hyper Voice');
  lines.push('#   (flags.sound) into a Soundproof holder → |move|…|Hyper Voice|<foe> then');
  lines.push('#   |-immune|<foe>|[from] ability: Soundproof (accuracy-only draw, no crit/damage).');
  lines.push('# Format identical to tests/vectors/e2e_fuzz_golden.txt (SCEN/TEAM/FMT/INIT/DEC/END/L).');
  emitBattle(lines, 'soundproof_dmg', p1, p2, rec, { protocol: true });
  const out = path.join(__dirname, '../tests/vectors/byte_fuzz_corpus/40_soundproof_damaging_immune.txt');
  fs.writeFileSync(out, lines.join('\n') + '\n');
  const spImmune = (rec.lines || []).filter((l) => /Soundproof/.test(l));
  console.log('wrote', out, 'decisions:', rec.decisions.length, 'ended:', rec.ended);
  console.log('soundproof lines:', JSON.stringify(spImmune));
})().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
