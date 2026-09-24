// prep_teams.js — build the LADDER candidate list and the PROCEDURAL team list for the
// belief-calibration read (PREDICTION.md §2). Run with node from ANY cwd; PIN_ROOT names the pinned
// worktree whose harness + Showdown dist + data/ are used.
//
//   PIN_ROOT=<pin> node prep_teams.js <metamon_team_dir> <pool.json> <n_procedural> <out_dir>
//
// Filter (registered): TeamValidator('gen3ou') legal AND the procedural generator's own coverage
// predicates (port species, !REJECT_SPECIES, isModeledMove(m, allowHP=true), !REJECT_MOVES,
// MODELED_ITEMS, speciesAllowedAbility). Out-of-pool (species-set) drop is applied here too.
// Writes ladder_candidates.json, procedural.json, filter_stats.json.
'use strict';
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');

const PIN = process.env.PIN_ROOT;
if (!PIN) { console.error('PIN_ROOT unset'); process.exit(2); }
const H = path.join(PIN, 'src/rust_sim/harness');
const { Teams, Dex, TeamValidator } = require(path.join(PIN, 'deps/pokemon-showdown/dist/sim'));
const e2e = require(path.join(H, 'gen_e2e_fuzz.js'));
const ab = require(path.join(H, 'ab_fuzz.js'));
const ouRandom = require(path.join(H, 'ou_random_teams.js'));
const dex3 = Dex.mod('gen3');
const toId = (s) => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
const portSpecies = JSON.parse(fs.readFileSync(path.join(PIN, 'data/pokemon/gen3_species.json'), 'utf8'));

const [metaDir, poolJson, nProcStr, outDir] = process.argv.slice(2);
const nProc = parseInt(nProcStr, 10);
const validator = new TeamValidator('gen3ou');

function coverageReject(team) {
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
const speciesKey = (team) => team.map((st) => toId(dex3.species.get(st.species || st.name).id || st.species)).sort().join(',');

function bump(m, k) { m[k] = (m[k] || 0) + 1; }

// ── pool: species sets (for the out-of-pool drop) + the same predicates, for INFORMATION ──
const pool = JSON.parse(fs.readFileSync(poolJson, 'utf8'));
const poolKeys = new Set();
const poolStats = { total: pool.length, import_fail: 0, validator_fail: 0, coverage_fail: 0, pass: 0, reasons: {} };
for (const txt of pool) {
  const team = Teams.import(txt);
  if (!team || team.length !== 6) { poolStats.import_fail++; continue; }
  poolKeys.add(speciesKey(team));
  const errs = validator.validateTeam(team);
  if (errs && errs.length) { poolStats.validator_fail++; bump(poolStats.reasons, 'validator:' + String(errs[0]).slice(0, 60)); continue; }
  const why = coverageReject(team);
  if (why) { poolStats.coverage_fail++; bump(poolStats.reasons, why); continue; }
  poolStats.pass++;
}

// ── ladder (Metamon hl_05_26) ──
const files = fs.readdirSync(metaDir).filter((f) => f.endsWith('_team')).sort();
const ladStats = { total: files.length, import_fail: 0, not_six: 0, validator_fail: 0, coverage_fail: 0,
  in_pool_species_set: 0, dup_species_set_within_ladder: 0, pass: 0, reasons: {} };
const ladder = [];
const seenLadderKeys = new Set();
for (const f of files) {
  const txt = fs.readFileSync(path.join(metaDir, f), 'utf8');
  let team;
  try { team = Teams.import(txt); } catch (e) { team = null; }
  if (!team) { ladStats.import_fail++; continue; }
  if (team.length !== 6) { ladStats.not_six++; continue; }
  const errs = validator.validateTeam(team);
  if (errs && errs.length) { ladStats.validator_fail++; bump(ladStats.reasons, 'validator:' + String(errs[0]).replace(/^[^:]*:/, '').slice(0, 60)); continue; }
  const why = coverageReject(team);
  if (why) { ladStats.coverage_fail++; bump(ladStats.reasons, why); continue; }
  const key = speciesKey(team);
  if (poolKeys.has(key)) { ladStats.in_pool_species_set++; continue; }
  if (seenLadderKeys.has(key)) ladStats.dup_species_set_within_ladder++;   // counted, kept
  seenLadderKeys.add(key);
  ladStats.pass++;
  ladder.push({ file: f, sha: crypto.createHash('sha1').update(txt.trim()).digest('hex').slice(0, 10),
    text: Teams.export(team) });
}

// ── procedural (ou_random_teams.js, coupled, fixed seed) ──
const universe = ouRandom.buildOuUniverse({
  isModeledMove: (m) => e2e.isModeledMove(m, true),
  modeledItem: (it) => e2e.MODELED_ITEMS.has(toId(it)),
  allowedAbility: (sid, a) => { const ok = ab.speciesAllowedAbility(sid); return !!ok && ok.includes(toId(a)); },
  portSpecies,
});
let s = 20260924; const rng = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
const genStats = { setsTotal: 0, teamsKept: 0, teamsRejected: 0, genErrors: 0, rejectReasons: new Map() };
const next = ouRandom.makeOuRandomProvider(rng, genStats, universe, { coupled: true });
const proc = [];
let procInPool = 0;
while (proc.length < nProc) {
  const team = Teams.unpack(next().packed);
  if (poolKeys.has(speciesKey(team))) { procInPool++; continue; }
  const txt = Teams.export(team);
  proc.push({ idx: proc.length, sha: crypto.createHash('sha1').update(txt.trim()).digest('hex').slice(0, 10), text: txt });
}

const topReasons = (r) => Object.entries(r).sort((a, b) => b[1] - a[1]).slice(0, 12);
const stats = {
  pool: { ...poolStats, reasons: topReasons(poolStats.reasons) },
  ladder: { ...ladStats, reasons: topReasons(ladStats.reasons), share_pass: ladStats.pass / ladStats.total },
  procedural: { n: proc.length, dropped_in_pool_species_set: procInPool, coverage: ouRandom.describeCoverage(universe),
    generator: { teamsKept: genStats.teamsKept, teamsRejected: genStats.teamsRejected, genErrors: genStats.genErrors } },
};
fs.writeFileSync(path.join(outDir, 'ladder_candidates.json'), JSON.stringify(ladder));
fs.writeFileSync(path.join(outDir, 'procedural.json'), JSON.stringify(proc));
fs.writeFileSync(path.join(outDir, 'filter_stats.json'), JSON.stringify(stats, null, 1));
console.log(JSON.stringify(stats, null, 1));
