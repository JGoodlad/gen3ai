// dump_gen3_formats.js — the gen-3 FORMAT table, extracted from Showdown (`gen3_format_rules_v1`).
//
// WHY: the engine used to read one bit off the format id (`!id.ends_with("customgame")`) and
// run every other format as gen3ou. Showdown's formats differ in ways a battle can SEE: only
// gen3ou and gen3uubl carry Freeze Clause Mod (the gen-3 `Standard` ruleset,
// data/mods/gen3/rulesets.ts, carries Sleep Clause Mod and no Freeze Clause), so in Ubers / UU /
// RU / NU / PU / ZU / LC / Random Battle a status application gathers ONE clause `onSetStatus`
// handler, not two, and draws no speed-sort shuffle (measured: a Thunder Wave turn is 4 PRNG
// calls in OU, 3 in UU). This dump reads the RESOLVED rule tables and starts a real battle per
// format for the `|tier|` / `|rule|` framing, and writes the vector `src/format.rs` is checked
// against (`tests/format_rules_test.rs`).
//
//   node src/rust_sim/harness/dump_gen3_formats.js           -> writes tests/vectors/gen3_formats.json
//   node src/rust_sim/harness/dump_gen3_formats.js --check   -> exit 1 if the committed vector drifted
//
// The format LIST is the gen3-mod singles formats `config/formats.ts` defines (every one,
// discovered, not typed): each is SUPPORTED or REFUSED with a reason, so a new Showdown format
// fails this check until it is dispositioned.
'use strict';
const fs = require('fs');
const path = require('path');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Dex } = require(path.join(PS, 'dist/sim'));
const { Battle } = require(path.join(PS, 'dist/sim/battle'));
const OUT = path.resolve(__dirname, '../tests/vectors/gen3_formats.json');

// The formats the engine REFUSES, each with its reason. Everything else gen-3 must be supported.
const REFUSED = {
  gen31v1: 'Team Preview + Picked Team Size = 1: the engine has no team-preview phase',
};
// Trapping abilities whose unheld-species leak (`battle.ts` endTurn "canceling switches would
// leak information") a format's banlist can switch off (`ruleTable.has('-ability:<id>')`).
const TRAP_ABILITIES = ['arenatrap', 'shadowtag', 'magnetpull'];

function framing(format) {
  const battle = new Battle({ formatid: format.id, seed: [1, 2, 3, 4] });
  const set = { species: 'Snorlax', ability: 'Immunity', moves: ['splash'], evs: {}, ivs: {}, level: format.id === 'gen3lc' ? 5 : 100 };
  const { Teams } = require(path.join(PS, 'dist/sim'));
  battle.setPlayer('p1', { name: 'A', team: Teams.pack([set]) });
  battle.setPlayer('p2', { name: 'B', team: Teams.pack([set]) });
  const tier = battle.log.filter((l) => l.startsWith('|tier|')).map((l) => l.slice('|tier|'.length));
  const rules = battle.log.filter((l) => l.startsWith('|rule|')).map((l) => l.slice('|rule|'.length));
  if (tier.length !== 1) throw new Error(`${format.id}: ${tier.length} |tier| lines`);
  return { tier: tier[0], rules };
}

function rowFor(format) {
  const rt = Dex.formats.getRuleTable(format);
  // The `onSetStatus` clause handlers the SetStatus event gathers (the shuffle is a TIE between
  // two of them; one alone sorts alone and draws nothing).
  const setStatusClauses = [...rt.keys()].filter((k) => {
    const r = Dex.formats.get(k);
    return r.exists && typeof r.onSetStatus === 'function';
  }).sort();
  const f = framing(format);
  return {
    id: format.id,
    tier: f.tier,
    rules: f.rules,
    sleep_clause: rt.has('sleepclausemod'),
    freeze_clause: rt.has('freezeclausemod'),
    set_status_clause_handlers: setStatusClauses,
    species_clause: rt.has('speciesclause'),
    beatup_nicknames: rt.has('beatupnicknamesmod'),
    // battle.ts endTurn: the unheld-species trapping leak is skipped for
    // `(+hackmons || !obtainableabilities) && !format.team`.
    ability_leak: !((rt.has('+hackmons') || !rt.has('obtainableabilities')) && !format.team),
    leak_banned_abilities: TRAP_ABILITIES.filter((a) => rt.has('-ability:' + a)),
    exact_hp: !!format.debug,
    // sim/battle.ts maybeTriggerEndlessBattleClause: the staleness check past turn 100 (gen3_b10_endless_battle_v1).
    endless_battle_clause: rt.has('endlessbattleclause'),
    max_level: rt.maxLevel,
    default_level: rt.defaultLevel,
    max_team_size: rt.maxTeamSize,
    max_move_count: rt.maxMoveCount,
  };
}

function build() {
  const all = Dex.formats.all().filter((f) => f.mod === 'gen3' && f.effectType === 'Format' &&
    (f.gameType || 'singles') === 'singles').sort((a, b) => (a.id < b.id ? -1 : 1));
  const supported = [];
  const refused = [];
  for (const f of all) {
    if (REFUSED[f.id]) refused.push({ id: f.id, reason: REFUSED[f.id] });
    else supported.push(rowFor(f));
  }
  for (const id of Object.keys(REFUSED)) {
    if (!all.some((f) => f.id === id)) throw new Error(`REFUSED names ${id}, which Showdown no longer defines`);
  }
  return {
    _meta: {
      generator: 'src/rust_sim/harness/dump_gen3_formats.js',
      source: 'Dex.formats (config/formats.ts) resolved rule tables + a started Battle per format',
      out_of_scope: 'doubles formats (no doubles engine) and the gen3rs / gen3frlg / gen3colosseum mods',
    },
    supported,
    refused,
  };
}

const out = JSON.stringify(build(), null, 1) + '\n';
if (process.argv.includes('--check')) {
  const have = fs.existsSync(OUT) ? fs.readFileSync(OUT, 'utf8') : '';
  if (have !== out) {
    console.error(`gen3 format table DRIFTED from ${OUT}: regenerate (node ${path.relative(process.cwd(), __filename)}), ` +
      'then update src/format.rs to match (tests/format_rules_test.rs)');
    process.exit(1);
  }
  console.error('gen3 format table OK');
} else {
  fs.writeFileSync(OUT, out);
  console.error(`wrote ${OUT}`);
}
