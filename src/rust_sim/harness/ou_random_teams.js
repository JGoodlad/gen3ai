// ou_random_teams.js — a gen3**ou** RANDOM team generator ("gen3ou-randbats").
//
// WHY THIS EXISTS. The fuzzers had two team sources and neither is the surface we care about:
//   * `pool`     — the 722 REAL gen3ou teams in `data/teams/`. On-surface, but a FIXED set:
//                  human-built, from a narrow meta, and the committed e2e capstone samples only
//                  220 battles from it. It cannot explore gen3ou's interaction space.
//   * `randbats` — Showdown's own gen3 random-battle sets. Diverse, but a DIFFERENT
//                  DISTRIBUTION: non-L100 levels, curated movesets, near-uniform items. This
//                  project has already measured that fixes found there do not transfer (the
//                  round-24 note: "fix bugs found on the SURFACE YOU CARE ABOUT"), and both bugs
//                  found on 2026-08-17 had literally ZERO gen3ou-pool exposure.
//
// This module is the missing third source: teams that are randomly generated but drawn from the
// REAL gen3ou distribution — so the fuzz explores far beyond 722 fixed teams while every battle
// it produces is one that could actually occur in training or on ladder.
//
// EVERY INPUT IS SMOGON-DERIVED and already committed (`data/pokemon/`):
//   `gen3_smogon_stats.json`  -> per-species `usage`      (which species appear)
//   `gen3_teammate_priors.json` -> species x species joint (coherent CORES, not 6 random mons)
//   `gen3_move_priors.json`   -> per-species move marginals
//   `gen3_item_priors.json` / `gen3_ability_priors.json` / `gen3_spread_priors.json`
// Deliberately NOT the pool's own `data/teams/gen3_species_priors.json` co-occurrence: that is
// POOL-derived, and the point of this generator is to be independent of the 722 teams.
//
// COVERAGE, MEASURED — the honest scope statement. Move sampling is renormalized over the
// ENGINE-MODELED moves, so a generated team ALWAYS plays to completion (no fail-loud, no
// truncated prefix). The cost is exactly the mass that renormalization moves: **1.33% of gen3ou
// move-slot mass** sits on the 88 engine-unmodeled moves (measured over all 216 species' priors).
// The top of that queue is `sandattack recycle confuseray safeguard conversion weatherball
// fakeout imprison present skillswap` — model those and the gap closes. (torment and eruption
// are DONE: `gen3_torment_v1` / `gen3_eruption_v1`.)
// The 1.33% is DISCLOSED by `describeCoverage()` and printed in the fuzz banner, never silent.
//
// HIDDEN POWER (the one non-obvious mechanic). HP is ~12% of gen3ou move slots, so dropping it
// would gut the generator. Gen-3 HP derives BOTH its type and its base power from IVs:
//   BP   = floor(u * 40 / 63) + 30, u = the BIT-1 sum over (hp,atk,def,spe,spa,spd)
//   TYPE = floor(t * 15 / 63),      t = the BIT-0 sum over the same stats, same weights
// BP 70 therefore requires u == 63, i.e. bit 1 set in EVERY IV — true of both 30 (11110b) and
// 31 (11111b). So restricting IVs to {30,31} pins BP at 70 automatically, and the bit-0 pattern
// (30 -> 0, 31 -> 1) alone selects the type. `hpIvsForType` brute-forces the 64 patterns once.

const path = require('path');
const ROOT = path.resolve(__dirname, '../../..');
const PS = path.join(ROOT, 'deps/pokemon-showdown');
const fs = require('fs');
const { Teams, Dex, TeamValidator } = require(path.join(PS, 'dist/sim'));

const dex3 = Dex.mod('gen3');
const toId = (s) => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
const DATA = path.join(ROOT, 'data/pokemon');
const readJson = (f) => JSON.parse(fs.readFileSync(path.join(DATA, f), 'utf8'));

// ── The gen-3 Hidden Power IV solver ────────────────────────────────────────────
const HP_TYPES = ['Fighting', 'Flying', 'Poison', 'Ground', 'Rock', 'Bug', 'Ghost', 'Steel',
  'Fire', 'Water', 'Grass', 'Electric', 'Psychic', 'Ice', 'Dragon', 'Dark'];
const IV_KEYS = ['hp', 'atk', 'def', 'spe', 'spa', 'spd']; // the ORDER the formula weights 1..32

let _hpIvCache = null;
function hpIvTable() {
  if (_hpIvCache) return _hpIvCache;
  const out = {};
  for (let mask = 0; mask < 64; mask++) {
    let t = 0;
    for (let i = 0; i < 6; i++) if (mask & (1 << i)) t += 1 << i;
    const type = HP_TYPES[Math.floor((t * 15) / 63)];
    if (out[type]) continue; // first pattern per type is enough
    const ivs = {};
    // bit0 set -> 31, else 30. Both carry bit1, so BP is pinned at 70.
    for (let i = 0; i < 6; i++) ivs[IV_KEYS[i]] = (mask & (1 << i)) ? 31 : 30;
    out[type] = ivs;
  }
  _hpIvCache = out;
  return out;
}
/** IVs that yield gen-3 Hidden Power of `type` at base power 70, or null if unrepresentable. */
function hpIvsForType(type) {
  const t = hpIvTable()[type];
  return t ? { ...t } : null;
}

// ── Priors ──────────────────────────────────────────────────────────────────────
let _priors = null;
function priors() {
  if (_priors) return _priors;
  const stats = readJson('gen3_smogon_stats.json').data;
  const usage = {};
  for (const [name, row] of Object.entries(stats)) {
    const u = typeof row.usage === 'number' ? row.usage : 0;
    if (u > 0) usage[toId(name)] = u;
  }
  _priors = {
    usage,
    moves: readJson('gen3_move_priors.json'),
    items: readJson('gen3_item_priors.json'),
    abilities: readJson('gen3_ability_priors.json'),
    spreads: readJson('gen3_spread_priors.json'),
    teammates: readJson('gen3_teammate_priors.json'),
  };
  return _priors;
}

// ── Weighted sampling helpers (seeded rng: () => [0,1)) ─────────────────────────
function weightedPick(rng, entries) {
  let total = 0;
  for (const [, w] of entries) total += w;
  if (total <= 0) return null;
  let r = rng() * total;
  for (const [k, w] of entries) { r -= w; if (r <= 0) return k; }
  return entries[entries.length - 1][0];
}
function weightedPickDistinct(rng, entries, n) {
  const pool = entries.slice();
  const out = [];
  while (out.length < n && pool.length) {
    const k = weightedPick(rng, pool);
    if (k == null) break;
    out.push(k);
    const i = pool.findIndex((e) => e[0] === k);
    if (i >= 0) pool.splice(i, 1);
  }
  return out;
}

/**
 * The eligible gen3ou species universe, each with its priors already intersected against what
 * the ENGINE can run. `isModeledMove` / `modeledItem` / `allowedAbility` are injected so this
 * module does not depend on gen_e2e_fuzz (which requires it back through ab_fuzz).
 */
function buildOuUniverse({ isModeledMove, modeledItem, allowedAbility, portSpecies }) {
  const P = priors();
  const eligible = [];
  let priorMass = 0, keptMass = 0;
  for (const [sid, moveP] of Object.entries(P.moves)) {
    const id = toId(sid);
    const sp = dex3.species.get(id);
    if (!sp || !sp.exists || sp.gen > 3) continue;
    if (portSpecies && !portSpecies[id]) continue;
    const moves = [];
    for (const [mv, p] of Object.entries(moveP)) {
      priorMass += p;
      if (!isModeledMove(mv)) continue;
      if (!dex3.moves.get(mv)?.exists) continue;
      keptMass += p;
      moves.push([mv, p]);
    }
    if (moves.length < 4) continue;
    const abilities = Object.entries(P.abilities[sid] || {}).filter(([a]) => allowedAbility(id, a));
    const items = Object.entries(P.items[sid] || {}).filter(([it]) => modeledItem(it));
    const spreads = Array.isArray(P.spreads[sid]) ? P.spreads[sid] : [];
    if (!abilities.length || !items.length || !spreads.length) continue;
    eligible.push({
      id, name: sp.name, usage: P.usage[id] || 0, moves, abilities, items, spreads,
    });
  }
  return { eligible, coverage: { priorMass, keptMass } };
}

/** A one-line, honest coverage statement for the run banner. */
function describeCoverage(universe) {
  const { priorMass, keptMass } = universe.coverage;
  const pct = priorMass > 0 ? (100 * (priorMass - keptMass)) / priorMass : 0;
  return `[ourandom] ${universe.eligible.length} gen3ou species; ` +
    `move sampling renormalized over ENGINE-MODELED moves, dropping ${pct.toFixed(2)}% of gen3ou ` +
    `move-slot prior mass (every generated battle therefore plays to COMPLETION)`;
}

/**
 * `gen3_spread_priors.json` stores a LIST of `[nature, [hp,atk,def,spa,spd,spe], weight]`
 * triples per species (the top spreads), NOT a dict. Reading it as a dict silently yields the
 * ARRAY INDEX as the nature — which Showdown then rejects with `"24" is an invalid nature`,
 * the first thing this generator did. Note the EV order is hp, atk, def, SPA, SPD, spe.
 */
function parseSpread(triple) {
  const [nature, evArr] = triple || [];
  const p = Array.isArray(evArr) ? evArr : [];
  return {
    nature: nature || 'Hardy',
    evs: { hp: p[0] || 0, atk: p[1] || 0, def: p[2] || 0, spa: p[3] || 0, spd: p[4] || 0, spe: p[5] || 0 },
  };
}

function genOuSet(entry, rng) {
  // AT MOST ONE Hidden Power. `hiddenpowerice` and `hiddenpowerflying` are distinct IDS but
  // Showdown validates them as the SAME move, so a set carrying two is rejected with
  // "<species> has multiple copies of Hidden Power <type>". Sample 4 distinct ids, then drop
  // every HP after the first and top up from the non-HP remainder.
  const isHp = (m) => m.startsWith('hiddenpower');
  let moves = weightedPickDistinct(rng, entry.moves, 4);
  if (moves.filter(isHp).length > 1) {
    const firstHp = moves.find(isHp);
    const kept = moves.filter((m) => !isHp(m) || m === firstHp);
    const spare = entry.moves.filter(([m]) => !isHp(m) && !kept.includes(m));
    moves = kept.concat(weightedPickDistinct(rng, spare, 4 - kept.length));
  }
  if (moves.length < 4) return null;
  const spreadIdx = weightedPick(rng, entry.spreads.map((t, i) => [i, t[2] || 0]));
  const { nature, evs } = parseSpread(entry.spreads[spreadIdx == null ? 0 : spreadIdx]);
  // IVs: 31s by default; a sampled Hidden Power PINS them (type + BP 70).
  let ivs = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
  const hp = moves.find((m) => m.startsWith('hiddenpower') && m !== 'hiddenpower');
  if (hp) {
    const type = HP_TYPES.find((t) => toId(t) === hp.slice('hiddenpower'.length));
    const solved = type && hpIvsForType(type);
    if (!solved) return null;
    ivs = solved;
  }
  const sp = dex3.species.get(entry.id);
  const gender = sp && sp.gender ? sp.gender : 'M'; // pin: an unset ratio makes the sim SAMPLE
  return {
    name: entry.name,
    species: entry.name,
    moves,
    ability: weightedPick(rng, entry.abilities),
    item: weightedPick(rng, entry.items),
    nature,
    evs,
    ivs,
    gender,
    level: 100, // gen3ou is always L100
  };
}

/**
 * A `{packed, genSeed}` provider with the SAME interface as `makeRandbatsProvider`, so it drops
 * into both `ab_fuzz.js` and `gen_sim_bridge_diff.js` unchanged.
 *
 * Species are drawn usage-weighted; with `coupled` the 2nd..6th are drawn from the Smogon
 * TEAMMATE joint conditioned on the lead, which produces recognisable gen3ou cores rather than
 * six unrelated mons. SPECIES CLAUSE is enforced by construction (distinct ids), and the whole
 * team is then put through the real `TeamValidator('gen3ou')` — so the banlist and every
 * team-building clause (Evasion, OHKO, the Baton Pass clauses, …) are Showdown's own verdict,
 * not a reimplementation.
 */
function makeOuRandomProvider(rng, stats, universe, opts = {}) {
  const coupled = opts.coupled !== false;
  const validator = new TeamValidator('gen3ou');
  const P = priors();
  const byId = new Map(universe.eligible.map((e) => [e.id, e]));
  const usageEntries = universe.eligible.map((e) => [e.id, e.usage]);

  function pickSpecies() {
    const lead = weightedPick(rng, usageEntries);
    const chosen = [lead];
    const mates = P.teammates[lead] || {};
    while (chosen.length < 6) {
      let next = null;
      if (coupled && Object.keys(mates).length) {
        const cand = Object.entries(mates)
          .map(([n, w]) => [toId(n), w])
          .filter(([id]) => byId.has(id) && !chosen.includes(id));
        if (cand.length) next = weightedPick(rng, cand);
      }
      if (!next) {
        const cand = usageEntries.filter(([id]) => !chosen.includes(id));
        if (!cand.length) break;
        next = weightedPick(rng, cand);
      }
      chosen.push(next);
    }
    return chosen;
  }

  return function nextTeam() {
    for (let tries = 0; tries < 300; tries++) {
      const ids = pickSpecies();
      if (ids.length < 6) { stats.genErrors++; continue; }
      const team = [];
      let bad = false;
      for (const id of ids) {
        const set = genOuSet(byId.get(id), rng);
        if (!set) { bad = true; break; }
        team.push(set);
      }
      if (bad) { stats.genErrors++; continue; }
      const packed = Teams.pack(team);
      // Showdown's own legality verdict — banlist + every team-building clause.
      const problems = validator.validateTeam(Teams.unpack(packed));
      if (problems && problems.length) {
        stats.teamsRejected++;
        const key = String(problems[0]).slice(0, 80);
        stats.rejectReasons.set(key, (stats.rejectReasons.get(key) || 0) + 1);
        continue;
      }
      stats.teamsKept++;
      stats.setsTotal += team.length;
      return { packed, genSeed: null };
    }
    throw new Error('ourandom provider: 300 consecutive rejections — check the generator');
  };
}

module.exports = {
  buildOuUniverse, makeOuRandomProvider, describeCoverage,
  hpIvsForType, hpIvTable, parseSpread, priors, HP_TYPES,
};

// ── `--selftest`: GATE-INTEGRITY for the generator itself ───────────────────────
//
// The frozen `byte_fuzz_corpus` fixtures gate the ENGINE on gen3ou-random boards, but they are
// frozen OUTPUT — break this generator and they still replay clean, so nothing fails. This is the
// generator's own gate. It is deliberately made of the two bugs that actually bit during the
// build, because those are the ones a refactor will reintroduce:
//   * the spread priors are a LIST of [nature, evs, weight] TRIPLES, not a dict (reading them as
//     a dict silently passes the ARRAY INDEX as the nature -> `"24" is an invalid nature`);
//   * all typed Hidden Powers validate as ONE move, so a set may carry at most one.
// Plus the property everything else rests on: the HP IV solver must produce the requested TYPE at
// BASE POWER 70, verified by RECOMPUTING both from the emitted IVs rather than trusting the table.
//
//   node src/rust_sim/harness/ou_random_teams.js --selftest
function selftest() {
  const e2e = require('./gen_e2e_fuzz.js');
  const ab = require('./ab_fuzz.js');
  let fail = 0;
  const check = (name, cond, extra) => {
    if (cond) { console.log(`  ok   ${name}`); } else { console.log(`  FAIL ${name}${extra ? ' — ' + extra : ''}`); fail++; }
  };

  // 1. The Hidden Power IV solver — recompute type AND base power from the emitted IVs.
  for (const want of HP_TYPES) {
    const ivs = hpIvsForType(want);
    let t = 0, u = 0;
    IV_KEYS.forEach((k, i) => { t += ((ivs[k] & 1) ? 1 : 0) << i; u += (((ivs[k] >> 1) & 1) ? 1 : 0) << i; });
    const gotType = HP_TYPES[Math.floor((t * 15) / 63)];
    const gotBp = Math.floor((u * 40) / 63) + 30;
    check(`hidden power ${want} solves to ${want} @ BP70`, gotType === want && gotBp === 70, `got ${gotType} @ BP${gotBp}`);
  }

  // 2. Spread priors really are [nature, evs, weight] triples, and parseSpread reads them.
  const P = priors();
  const anySid = Object.keys(P.spreads)[0];
  check('spread priors are a LIST (not a dict)', Array.isArray(P.spreads[anySid]));
  const sp = parseSpread(P.spreads[anySid][0]);
  check('parseSpread yields a real nature, not an array index',
    typeof sp.nature === 'string' && /^[A-Z][a-z]+$/.test(sp.nature), `got ${JSON.stringify(sp.nature)}`);
  check('parseSpread yields 6 numeric EVs summing <= 510',
    Object.values(sp.evs).every((v) => Number.isInteger(v) && v >= 0)
      && Object.values(sp.evs).reduce((a, b) => a + b, 0) <= 510, JSON.stringify(sp.evs));

  // 3. End to end: generated teams must be gen3ou-LEGAL and carry at most one Hidden Power.
  const toIdL = (s) => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
  const universe = buildOuUniverse({
    isModeledMove: (m) => e2e.isModeledMove(m, true),
    modeledItem: (it) => e2e.MODELED_ITEMS.has(toIdL(it)),
    allowedAbility: (sid, a) => { const ok = ab.speciesAllowedAbility(sid); return !!ok && ok.includes(toIdL(a)); },
    portSpecies: null,
  });
  check('universe is non-trivial (>=100 gen3ou species)', universe.eligible.length >= 100, `${universe.eligible.length}`);
  const dropped = 100 * (universe.coverage.priorMass - universe.coverage.keptMass) / universe.coverage.priorMass;
  check('renormalized-away move mass stays small (<5%)', dropped < 5, `${dropped.toFixed(2)}%`);

  const stats = { setsTotal: 0, teamsKept: 0, teamsRejected: 0, genErrors: 0, rejectReasons: new Map() };
  let s = 1234567; const rng = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
  const next = makeOuRandomProvider(rng, stats, universe);
  let multiHp = 0, badSize = 0, dupSpecies = 0;
  const N = 25;
  for (let i = 0; i < N; i++) {
    const team = Teams.unpack(next().packed);
    if (!team || team.length !== 6) { badSize++; continue; }
    const ids = team.map((t) => toIdL(t.species));
    if (new Set(ids).size !== 6) dupSpecies++;
    for (const set of team) {
      if ((set.moves || []).filter((m) => toIdL(m).startsWith('hiddenpower')).length > 1) multiHp++;
    }
  }
  check(`${N} teams all have 6 members`, badSize === 0, `${badSize} bad`);
  check('SPECIES CLAUSE holds (no duplicate species)', dupSpecies === 0, `${dupSpecies} teams`);
  check('no set carries TWO Hidden Powers', multiHp === 0, `${multiHp} sets`);
  check('every kept team passed TeamValidator(gen3ou)', stats.teamsKept === N, `kept ${stats.teamsKept}/${N}`);

  console.log(fail ? `\n[selftest] ${fail} FAILURE(S)` : '\n[selftest] all generator gate-integrity cases pass');
  process.exit(fail ? 1 : 0);
}

if (require.main === module && process.argv.includes('--selftest')) selftest();
