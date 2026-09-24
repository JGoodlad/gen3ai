// ladder_corpus_filter.js — the Showdown half of the LADDER-USAGE corpus build
// (`gen3_ladder_usage_corpus_v1`; the Python driver is `python -m utils.ladder_corpus.build`).
//
// Reads every `*_team` paste in a Metamon team directory (sorted by name) and, per team, answers
// the questions only the REAL Showdown can: does `Teams.import` read it, is it six Pokemon, is it
// `TeamValidator('gen3ou')`-legal, and what is its canonical `Teams.pack` string. It also lists the
// team's species / move / item / ability ids so the driver can ask the ENGINE (`scan_move_probe`,
// the only coverage oracle — see src/rust_sim/CLAUDE.md "The move census is RECOUNTED") whether it
// can play them, and — for INFORMATION only — the procedural generator's own coverage predicate
// (`isModeledMove(m, allowHP)` & co.), which is stricter than the engine (it has picker false
// negatives: Sleep Talk is engine-modeled but not `isModeledMove`).
//
//   node src/rust_sim/harness/ladder_corpus_filter.js <team_dir>   > rows.jsonl
//
// One JSON row per file on stdout:
//   {"file","sha","verdict":"ok"|"import_fail"|"not_six"|"validator_fail","why"?,
//    "packed"?,"species"?,"moves"?,"items"?,"abilities"?,"generator_coverage"?: null|"<reason>"}
'use strict';
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Teams, Dex, TeamValidator } = require(path.join(PS, 'dist/sim'));
const e2e = require('./gen_e2e_fuzz.js');
const ab = require('./ab_fuzz.js');

const dex3 = Dex.mod('gen3');
const toId = (s) => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
const ROOT = path.resolve(__dirname, '../../..');
const portSpecies = JSON.parse(fs.readFileSync(path.join(ROOT, 'data/pokemon/gen3_species.json'), 'utf8'));

// The belief-calibration read's predicate (designs/research_state/measurements/
// belief_calibration_2026-09-24/scripts/prep_teams.js::coverageReject), verbatim in effect.
function generatorCoverage(team) {
  for (const set of team) {
    const sid = toId(dex3.species.get(set.species || set.name).id || set.species);
    if (!portSpecies[sid]) return `species:${sid}`;
    if (e2e.REJECT_SPECIES.has(sid)) return `reject_species:${sid}`;
    for (const mv of (set.moves || [])) {
      const id = toId(mv);
      if (e2e.REJECT_MOVES.has(id)) return `reject_move:${id}`;
      if (!e2e.isModeledMove(id, true)) return `move:${id}`;
    }
    if (!e2e.MODELED_ITEMS.has(toId(set.item))) return `item:${toId(set.item)}`;
    const ok = ab.speciesAllowedAbility(sid);
    if (!ok || !ok.includes(toId(set.ability))) return `ability:${toId(set.ability)}`;
  }
  return null;
}

function main() {
  const dir = process.argv[2];
  if (!dir) { console.error('usage: ladder_corpus_filter.js <team_dir>'); process.exit(2); }
  const validator = new TeamValidator('gen3ou');
  const files = fs.readdirSync(dir).filter((f) => f.endsWith('_team')).sort();
  const out = [];
  for (const f of files) {
    const txt = fs.readFileSync(path.join(dir, f), 'utf8');
    const row = { file: f, sha: crypto.createHash('sha1').update(txt.trim()).digest('hex').slice(0, 10) };
    let team = null;
    try { team = Teams.import(txt); } catch (e) { team = null; }
    if (!team) { out.push(JSON.stringify({ ...row, verdict: 'import_fail' })); continue; }
    if (team.length !== 6) { out.push(JSON.stringify({ ...row, verdict: 'not_six', why: String(team.length) })); continue; }
    const errs = validator.validateTeam(team);
    if (errs && errs.length) {
      out.push(JSON.stringify({ ...row, verdict: 'validator_fail', why: String(errs[0]).slice(0, 120) }));
      continue;
    }
    out.push(JSON.stringify({
      ...row, verdict: 'ok', packed: Teams.pack(team),
      species: team.map((s) => toId(dex3.species.get(s.species || s.name).id || s.species)),
      moves: [...new Set(team.flatMap((s) => (s.moves || []).map(toId)))],
      items: [...new Set(team.map((s) => toId(s.item)).filter(Boolean))],
      abilities: [...new Set(team.map((s) => toId(s.ability)).filter(Boolean))],
      generator_coverage: generatorCoverage(team),
    }));
  }
  process.stdout.write(out.join('\n') + '\n');
}

main();
