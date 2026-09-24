// ladder_corpus.js — the JS twin of `utils.ladder_corpus` (`gen3_ladder_usage_corpus_v1`): the
// LADDER-USAGE corpus as a TEAM SOURCE for the four A/B fuzzers (`--mode ladder`).
//
// The corpus is the Metamon `hl_05_26` gen3ou ladder teams, filtered to what the ENGINE can play
// (src/utils/ladder_corpus/manifest.json has the filter, its counts and every tier's stamp).
// Tiers are prefixes of one seeded permutation: `commit` (16) ⊂ `milestone` (800) ⊂ `full`
// (every team). The fuzzers draw from `milestone` by default; `--ladder-tier full` soaks all.
//
// The data file's sha256 is checked against the manifest on EVERY load, and a mismatch THROWS:
// a corpus that silently changed under a fuzzer would make its master seed replay different
// battles.
'use strict';
const path = require('path');
const fs = require('fs');
const zlib = require('zlib');
const crypto = require('crypto');

const DIR = path.resolve(__dirname, '../../utils/ladder_corpus');
const TIERS = ['commit', 'milestone', 'full'];

function manifest() {
  return JSON.parse(fs.readFileSync(path.join(DIR, 'manifest.json'), 'utf8'));
}

let cache = null;
function allRows() {
  if (cache) return cache;
  const m = manifest();
  const raw = fs.readFileSync(path.join(DIR, m.data_file));
  const got = crypto.createHash('sha256').update(raw).digest('hex');
  if (got !== m.data_sha256) {
    throw new Error(`ladder corpus ${m.data_file}: sha256 ${got} != manifest ${m.data_sha256} ` +
      '(rebuild with `python -m utils.ladder_corpus.build`)');
  }
  const rows = zlib.gunzipSync(raw).toString('utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
  if (rows.length !== m.tiers.full.n) throw new Error(`ladder corpus: ${rows.length} rows, manifest ${m.tiers.full.n}`);
  cache = { m, rows };
  return cache;
}

/** `tier`'s rows ({i, sha, file, packed}), in corpus order. */
function ladderRows(tier = 'milestone') {
  if (!TIERS.includes(tier)) throw new Error(`--ladder-tier must be ${TIERS.join('|')}, got ${tier}`);
  const { m, rows } = allRows();
  return rows.slice(0, m.tiers[tier].n);
}

/**
 * A single-team provider in `ab_fuzz.js`'s shape: each call draws one corpus team uniformly at
 * random from `tier` with the fuzzer's seeded team RNG (so a master seed replays the same draws).
 * Every corpus team already passed the ENGINE oracle; `teamFilterClean` (the JS mirror of the
 * modeled sets) is REPORTED against, never used to drop a team — a picker predicate that gates a
 * test silently shrinks it (src/rust_sim/CLAUDE.md, "The picker's own blind spots").
 */
function makeLadderProvider(rng, stats, tier = 'milestone', teamFilterClean = null) {
  const rows = ladderRows(tier);
  if (rows.length < 2) throw new Error(`ladder mode: only ${rows.length} teams in tier ${tier}`);
  let jsMirrorDisagrees = 0;
  if (teamFilterClean) for (const r of rows) if (!teamFilterClean(r.packed).ok) jsMirrorDisagrees++;
  const m = allRows().m;
  console.error(`[ladder] tier=${tier} ${rows.length} teams of ${m.tiers.full.n} ` +
    `(Metamon hl_05_26 gen3ou, engine-playable ${m.filter.counts.kept}/${m.filter.counts.total}; ` +
    `teamFilterClean disagrees on ${jsMirrorDisagrees}, kept)`);
  return function nextTeam() {
    stats.teamsKept++;
    const r = rows[Math.floor(rng() * rows.length)];
    return { packed: r.packed, genSeed: null, ladder: { i: r.i, sha: r.sha } };
  };
}

module.exports = { TIERS, manifest, ladderRows, makeLadderProvider };
