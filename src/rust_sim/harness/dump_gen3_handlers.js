// dump_gen3_handlers.js — the HANDLER-COMPLETENESS AUDIT (the dispatch-bus guarantee
// as a STATIC gate).
//
// WHY: the port implements effects AT-SITE (no generic runEvent bus). The recurring bug
// class this allowed: an effect carries a handler at a hook we never enumerated or
// hand-placed at the wrong site — Immunity's onUpdate cure, Cloud Nine's onEnd
// WeatherChange, Plus/Minus's cross-field onModifySpA, the tox onSwitchIn reset living
// at runSwitch-time, sun/rain's unguarded onFieldResidual, facade's onBasePower. This
// harness enumerates EVERY handler-bearing key on EVERY effect in the port's REACHABLE
// surface from the RESOLVED `Dex.mod('gen3')` (the mod-chain law — never raw data
// files) and gates it against a committed MANIFEST in which every (effect, hook) pair
// carries an explicit, checkable DISPOSITION.
//
// THE REACHABLE SURFACE (kept in lockstep with gen_e2e_fuzz.js + the engine state
// space in src/state.rs):
//   ability:   MODELED_ABILITIES ∪ NOOP_ABILITIES        (gen_e2e_fuzz.js exports)
//   item:      MODELED_ITEMS (incl. the 22 berryEffect berries)
//   condition: every condition the engine can enter — the 6 major statuses
//              (MonState::status), the modeled volatiles (MonState::confusion/flinch/
//              protected/protect_counter+stall_duration/leech_seed/substitute/taunt/
//              disable/choice_locked_move/flash_fire/focus_energy/attract), the 4
//              weathers (Field::weather), the side condition spikes (SideState::spikes),
//              and the format clauses (sleepclausemod/freezeclausemod — BattleState::
//              sleep_clause), PLUS any `condition` sub-object carried by a surface
//              ability/item/move (auto-added; a miss fails the dump loudly).
//   move:      every isModeledMove(id) move, plus `struggle` (reachable via
//              must_struggle even though isModeledMove rejects it). Per move: every
//              resolved `on*` callback AND the effect-bearing DECLARATIVE fields
//              (secondaries shape, selfdestruct, forceSwitch, volatileStatus, status,
//              boosts, heal, thawsTarget, never-miss, priority, …).
//
// Every enumerated (effect, hook) row carries a BODY FINGERPRINT (an FNV-1a hash of the
// whitespace-normalized resolved source + a readable prefix) so a semantic change in
// the dist is DETECTED (the fingerprint drifts → the gate fails → re-probe).
//
// THE MANIFEST — tests/vectors/gen3_handler_audit.json: one row per (effect, hook) with
//   disposition: implemented | noop_justified | unreachable_justified | failloud_guarded
//   anchor:      (implemented rows) "file.rs::symbol" — the file under src/ must contain
//                the symbol (grep-verified by --audit); justified rows carry a reason +
//                the probe file where non-obvious.
// The dispositions live in harness/handler_audit_dispositions.js (curated, code-reviewed
// — the same pattern as dump_gen3_mechanics.js's curated tables).
//
// Usage:
//   node dump_gen3_handlers.js              -> regenerate the manifest JSON + the human
//                                              census MD from the dispositions table
//                                              (fails loudly on any TODO/undispositioned row)
//   node dump_gen3_handlers.js --enumerate  -> print the raw enumeration (debug/triage)
//   node dump_gen3_handlers.js --audit      -> THE GATE (no writes; exit 1): fails when
//                                              (a) a resolved handler key has NO manifest row,
//                                              (b) a manifest row's key vanished from the dist,
//                                              (c) a body fingerprint changed,
//                                              (d) an `implemented` anchor no longer greps in src/,
//                                              (e) a row carries an invalid/TODO disposition.
//                                              Wired into cargo test via tests/handler_audit_test.rs.
//
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Dex } = require(path.join(PS, 'dist/sim/dex'));

const OUT_JSON = path.resolve(__dirname, '../tests/vectors/gen3_handler_audit.json');
const OUT_MD = path.resolve(__dirname, '../tests/vectors/gen3_handler_audit.md');
const SRC_DIR = path.resolve(__dirname, '../src');

const e2e = require('./gen_e2e_fuzz.js');

const d3 = Dex.mod('gen3');

function toId(s) { return ('' + (s || '')).toLowerCase().replace(/[^a-z0-9]/g, ''); }

// ── Fingerprints ─────────────────────────────────────────────────────────────
// FNV-1a 32-bit over the whitespace-normalized body + a readable prefix. A body
// CHANGE anywhere (not just the first 120 chars) drifts the hash.
function fnv1a(s) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return ('0000000' + h.toString(16)).slice(-8);
}
function fpOf(value) {
  if (typeof value === 'function') {
    const norm = String(value).replace(/\s+/g, ' ');
    return `fn:${fnv1a(norm)}:${norm.slice(0, 120)}`;
  }
  const j = JSON.stringify(value);
  return `val:${j.length > 140 ? fnv1a(j) + ':' + j.slice(0, 120) : j}`;
}

// ── Handler-key enumeration ──────────────────────────────────────────────────
// Base-data cruft irrelevant to gen3 (Arceus plates / Genesect drives / Silvally
// memories) — same exclusion as dump_gen3_mechanics.js.
const IGNORED_HANDLERS = new Set(['onDrive', 'onMemory', 'onPlate']);

// Every `on*` key (function, number, boolean — priorities/orders/subOrders are
// draw-order-relevant and get their own rows) + the draw-relevant declarative
// condition fields.
const CONDITION_DECL_KEYS = ['duration', 'durationCallback', 'counterMax'];

function onKeys(entry, extraDecl) {
  const rows = {};
  for (const k of Object.keys(entry)) {
    if (!k.startsWith('on') || entry[k] === undefined || IGNORED_HANDLERS.has(k)) continue;
    rows[k] = entry[k];
  }
  for (const k of extraDecl || []) {
    if (entry[k] !== undefined) rows[k] = entry[k];
  }
  return rows;
}

// ── The surface ──────────────────────────────────────────────────────────────

// condition ids the ENGINE can enter (src/state.rs — see the header comment).
const ENGINE_CONDITIONS = [
  // major statuses (MonState::status)
  'brn', 'par', 'slp', 'frz', 'psn', 'tox',
  // modeled volatiles (MonState fields)
  'confusion', 'flinch', 'substitute', 'leechseed', 'protect', 'stall',
  'attract', 'taunt', 'disable', 'choicelock', 'flashfire', 'focusenergy',
  // weathers (Field::weather)
  'sandstorm', 'raindance', 'sunnyday', 'hail',
  // side conditions (SideState::spikes)
  'spikes',
  // BATCH-4c cross-turn conditions (`gen3_move_coverage_batch4c_v1`): Hyper Beam's
  // recharge lock (MonState::must_recharge), Solar Beam's charge volatile
  // (MonState::two_turn), the Doom Desire / Future Sight slot condition
  // (SideState::future_move). Standalone conditions (not `m.condition` sub-objects),
  // so they are enumerated here explicitly.
  'mustrecharge', 'twoturnmove', 'futuremove',
  // BATCH-6 trap-move volatiles (`gen3_move_coverage_batch6_v1`): the linked
  // `trapped`/`trapper` pair (MonState::trapped_by — Mean Look / Spider Web / Block).
  // Standalone conditions (the moves' `volatileStatus: 'trapped'` refers to the global
  // condition, not an `m.condition` sub-object), so enumerated explicitly.
  'trapped', 'trapper',
  // format clauses (BattleState::sleep_clause — gen3ou carries BOTH clauses)
  'sleepclausemod', 'freezeclausemod',
];

// Effect-bearing DECLARATIVE move fields (each gets its own row when present and
// non-default — the port must price it or justify it).
const MOVE_DECL_FIELDS = [
  'secondary', 'secondaries', 'self', 'selfdestruct', 'forceSwitch', 'volatileStatus',
  'sideCondition', 'slotCondition', 'status', 'boosts', 'heal', 'drain', 'recoil',
  'struggleRecoil', 'damage', 'thawsTarget', 'stallingMove', 'selfSwitch', 'weather',
  'pseudoWeather', 'terrain', 'critRatio', 'willCrit', 'multihit', 'priority',
  'sleepUsable', 'ignoreImmunity', 'breaksProtect', 'ohko', 'multiaccuracy',
  'smartTarget', 'nonGhostTarget', 'pressureTarget', 'forceSTAB', 'noPPBoosts',
];
function moveDeclRows(m) {
  const rows = {};
  for (const k of MOVE_DECL_FIELDS) {
    const v = m[k];
    if (v === undefined || v === null || v === false || v === 0 || v === '') continue;
    if (k === 'critRatio' && v === 1) continue;
    // `secondary` is the raw data field; the resolved `secondaries` list is the
    // canonical form (and covers multi-secondary moves) — one row, not two.
    if (k === 'secondary' && m.secondaries) continue;
    rows[k] = v;
  }
  // never-miss (accuracy === true) is draw-relevant: NO accuracy roll at all.
  if (m.accuracy === true) rows.neverMiss = true;
  return rows;
}

function buildSurface() {
  const rows = []; // {key, kind, id, hook, fp}
  const conditionIds = new Set(ENGINE_CONDITIONS);
  const condSource = new Map(); // id -> where it came from (for the census)
  for (const id of ENGINE_CONDITIONS) condSource.set(id, 'engine state space');

  const push = (kind, id, hook, value) =>
    rows.push({ key: `${kind}:${id}:${hook}`, kind, id, hook, fp: fpOf(value) });

  // (a) abilities: MODELED ∪ NOOP.
  const abilityIds = [...e2e.MODELED_ABILITIES, ...e2e.NOOP_ABILITIES]
    .filter((id) => id && id !== 'noability').sort();
  for (const id of abilityIds) {
    const ab = d3.abilities.get(id);
    if (!ab || !ab.exists) throw new Error(`surface ability ${id} does not resolve in gen3`);
    const inv = onKeys(ab);
    for (const [k, v] of Object.entries(inv)) push('ability', id, k, v);
    if (ab.condition) {
      conditionIds.add(toId(ab.condition.id || id));
      condSource.set(toId(ab.condition.id || id), `ability ${id} condition`);
    }
  }

  // (b) items: MODELED_ITEMS (incl. every berryEffect berry — they are members).
  const itemIds = [...e2e.MODELED_ITEMS].filter((id) => id).sort();
  for (const id of itemIds) {
    const it = d3.items.get(id);
    if (!it || !it.exists) throw new Error(`surface item ${id} does not resolve in gen3`);
    const inv = onKeys(it);
    for (const [k, v] of Object.entries(inv)) push('item', id, k, v);
    if (it.condition) {
      conditionIds.add(toId(it.condition.id || id));
      condSource.set(toId(it.condition.id || id), `item ${id} condition`);
    }
  }

  // (d) moves: the isModeledMove universe + struggle (reachable via must_struggle)
  // + sleeptalk (`gen3_move_coverage_batch5_v1` — MODELED but isModeledMove-false:
  // its e2e pickability is CARRIER-conditional via `sleepTalkPoolModeled`, so the
  // per-id predicate rejects it; the ENGINE runs it, so its handlers ARE surface).
  const moveIds = [];
  for (const mv of d3.moves.all()) {
    if (mv.gen > 3 || mv.isNonstandard) continue;
    if (e2e.isModeledMove(mv.id)) moveIds.push(mv.id);
  }
  moveIds.push('struggle');
  moveIds.push('sleeptalk');
  moveIds.sort();
  for (const id of moveIds) {
    const m = d3.moves.get(id);
    if (!m || !m.exists) throw new Error(`surface move ${id} does not resolve in gen3`);
    const inv = onKeys(m);
    for (const [k, v] of Object.entries(inv)) push('move', id, k, v);
    for (const [k, v] of Object.entries(moveDeclRows(m))) push('move', id, k, v);
    if (m.condition) {
      conditionIds.add(toId(m.condition.id || id));
      condSource.set(toId(m.condition.id || id), condSource.get(toId(m.condition.id || id)) || `move ${id} condition`);
    }
  }

  // (c) conditions: the engine state space + every auto-added `condition` sub-object.
  for (const id of [...conditionIds].sort()) {
    const c = d3.conditions.get(id);
    if (!c || c.exists === false) throw new Error(`surface condition ${id} does not resolve in gen3`);
    const inv = onKeys(c, CONDITION_DECL_KEYS);
    for (const [k, v] of Object.entries(inv)) push('condition', id, k, v);
  }

  rows.sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
  // Duplicate keys would corrupt the manifest map — fail loudly.
  for (let i = 1; i < rows.length; i++) {
    if (rows[i].key === rows[i - 1].key) throw new Error(`duplicate surface key ${rows[i].key}`);
  }
  return { rows, moveIds, abilityIds, itemIds, conditionIds: [...conditionIds].sort(), condSource };
}

// ── Anchor grep (implemented rows) ───────────────────────────────────────────
// anchor "file.rs::symbol" -> src/<file.rs> (any depth) must contain <symbol>;
// a bare "symbol" -> some file under src/ must contain it.
let _srcFiles = null;
function srcFiles() {
  if (_srcFiles) return _srcFiles;
  const out = new Map(); // relpath -> content
  const walk = (dir) => {
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      const p = path.join(dir, e.name);
      if (e.isDirectory()) walk(p);
      else if (e.name.endsWith('.rs')) out.set(path.relative(SRC_DIR, p), fs.readFileSync(p, 'utf8'));
    }
  };
  walk(SRC_DIR);
  _srcFiles = out;
  return out;
}
function anchorGreps(anchor) {
  const files = srcFiles();
  const m = anchor.match(/^([\w/]+\.rs)::(.+)$/);
  if (m) {
    // A "file.rs::symbol" anchor names a Rust MODULE. Rust lets one module live as
    // `foo.rs` PLUS a sibling `foo/` submodule directory (the turn.rs -> turn/ split),
    // so accept the symbol in `foo.rs` OR in any file under `foo/` — either form of the
    // same module. This keeps the anchor an intra-module existence check.
    const modDir = m[1].slice(0, -3) + '/'; // "turn.rs" -> "turn/"
    for (const [rel, content] of files) {
      const inModule =
        rel === m[1] || rel.endsWith('/' + m[1]) || rel.startsWith(modDir) || rel.includes('/' + modDir);
      if (inModule && content.includes(m[2])) return true;
    }
    return false;
  }
  for (const content of files.values()) if (content.includes(anchor)) return true;
  return false;
}

// ── Modes ────────────────────────────────────────────────────────────────────
const DISPOSITIONS = new Set(['implemented', 'noop_justified', 'unreachable_justified', 'failloud_guarded']);

function loadManifest() {
  return JSON.parse(fs.readFileSync(OUT_JSON, 'utf8'));
}

function buildManifest(surface) {
  const { dispositionFor } = require('./handler_audit_dispositions.js');
  const rows = {};
  const todo = [];
  for (const r of surface.rows) {
    const d = dispositionFor(r, d3);
    if (!d || !DISPOSITIONS.has(d.disposition)) { todo.push(`${r.key}  fp=${r.fp.slice(0, 80)}`); continue; }
    if (d.disposition === 'implemented' && !d.anchor) { todo.push(`${r.key}  (implemented WITHOUT an anchor)`); continue; }
    const row = { disposition: d.disposition, fp: r.fp };
    if (d.anchor) row.anchor = d.anchor;
    if (d.reason) row.reason = d.reason;
    if (d.probe) row.probe = d.probe;
    rows[r.key] = row;
  }
  if (todo.length) {
    console.error(`UNDISPOSITIONED (effect, hook) pairs (${todo.length}) — triage them in handler_audit_dispositions.js:`);
    for (const t of todo) console.error('  ' + t);
    process.exit(1);
  }
  return {
    _meta: {
      generator: 'harness/dump_gen3_handlers.js',
      gate: 'node harness/dump_gen3_handlers.js --audit (cargo: tests/handler_audit_test.rs)',
      surface: {
        abilities: surface.abilityIds.length,
        items: surface.itemIds.length,
        conditions: surface.conditionIds.length,
        moves: surface.moveIds.length,
        rows: surface.rows.length,
      },
      // Effects OUTSIDE the reachable surface BY CONSTRUCTION — deferred/fail-loud, kept
      // off the modeled path by the e2e filter + engine panics (NOT silently waved
      // through; each is a documented level-2/deferred gap, see CLAUDE.md/EDGE_CASES.md).
      // They carry no per-hook rows because the surface definition (MODELED∪NOOP
      // abilities, MODELED items, isModeledMove) excludes them; admitting one to a
      // MODELED set pulls its handlers INTO the surface and the gate then demands rows.
      excluded_deferred: {
        forecast: 'Castform forme+TYPE change under weather — deferred (probe_forecast_rng.js settled the map; the reporting surface is unprobed); the port FAIL-LOUDs in state::MonState::from_set (GIGO guard: an unmodeled ability would silently no-op then desync under weather); NOT in MODELED/NOOP + REJECT_ABILITIES/REJECT_SPECIES so the e2e filter + randbats adapter exclude its teams',
        liquidooze: 'reverses the Leech Seed drain — the port FAIL-LOUDs in apply_leech_seed; removed from NOOP_ABILITIES so its teams are off the modeled path',
        ohko_psywave_counter_family: 'fissure/horndrill/guillotine/sheercold/psywave/counter/mirrorcoat/bide/endeavor — the deferred fixed-damage family: isModeledMove rejects + MOVE_ID_BLOCKLIST + the engine fail-louds',
        protocol_lines: 'PROTOCOL PHASE 3 (gen3_protocol_phase3_v1) closed the formerly-deferred lines: the Trace |-ability| reveal, the taunt/disable residual -end lines + the Disable [miss]/[still] retro-edits, the Flash Fire -start/-immune/-end cycle, the STATUS_IMMUNE -immune [from] ability: blocks, the Synchronize→Lum interleave, the mid-battle switch-in ability lines, Leech Seed / Splash / Pay Day — all byte-verified vs the capture golden (114 battles / 16115 lines). Remaining un-emitted (uncapturable) forms are documented at their sites — see PROTOCOL_EMISSION_DESIGN.md',
      },
    },
    rows,
  };
}

function audit(surface) {
  const errors = [];
  let manifest;
  try { manifest = loadManifest(); } catch (e) {
    console.error(`AUDIT FAIL: cannot read ${OUT_JSON}: ${e.message}`);
    process.exit(1);
  }
  const mrows = manifest.rows || {};
  const live = new Map(surface.rows.map((r) => [r.key, r]));
  // (a) a resolved handler key with NO manifest row — a NEW/unnoticed handler.
  for (const r of surface.rows) {
    if (!mrows[r.key]) errors.push(`NEW HANDLER (no manifest row): ${r.key}  fp=${r.fp.slice(0, 90)}`);
  }
  for (const [key, row] of Object.entries(mrows)) {
    const l = live.get(key);
    // (b) a manifest row whose key vanished from the dist/surface — stale.
    if (!l) { errors.push(`STALE ROW (key no longer enumerated): ${key}`); continue; }
    // (c) a body fingerprint changed — the handler's semantics may have drifted.
    if (row.fp !== l.fp) {
      errors.push(`FINGERPRINT DRIFT (re-probe the handler): ${key}\n    manifest ${row.fp.slice(0, 90)}\n    resolved ${l.fp.slice(0, 90)}`);
    }
    // (e) disposition sanity.
    if (!DISPOSITIONS.has(row.disposition)) {
      errors.push(`INVALID DISPOSITION ${JSON.stringify(row.disposition)}: ${key}`);
    }
    // (d) an `implemented` anchor that no longer greps in src/.
    if (row.disposition === 'implemented') {
      if (!row.anchor) errors.push(`IMPLEMENTED WITHOUT AN ANCHOR: ${key}`);
      else if (!anchorGreps(row.anchor)) errors.push(`ANCHOR NOT FOUND in src/: ${key} -> ${row.anchor}`);
    }
  }
  if (errors.length) {
    console.error(`HANDLER AUDIT FAIL (${errors.length}):\n  ` + errors.join('\n  '));
    process.exit(1);
  }
  console.error(`handler audit OK: ${surface.rows.length} (effect, hook) rows all dispositioned + fingerprint-stable + anchors grep`);
}

// ── The human census MD ──────────────────────────────────────────────────────
function writeMd(surface, manifest) {
  const lines = [];
  lines.push('# gen3 handler-completeness audit — the DISPOSITION CENSUS');
  lines.push('');
  lines.push('GENERATED by `harness/dump_gen3_handlers.js` from the RESOLVED `Dex.mod(\'gen3\')`');
  lines.push('(the mod-chain law). The machine-checkable manifest is `gen3_handler_audit.json`;');
  lines.push('the GATE is `node src/rust_sim/harness/dump_gen3_handlers.js --audit` (wired into');
  lines.push('`cargo test` via `tests/handler_audit_test.rs`). It fails on: a NEW un-dispositioned');
  lines.push('handler, a STALE manifest row, a body-FINGERPRINT drift, a dead `implemented` anchor.');
  lines.push('');
  lines.push(`Surface: ${surface.abilityIds.length} abilities (MODELED ∪ NOOP), ${surface.itemIds.length} items,`);
  lines.push(`${surface.conditionIds.length} conditions (engine state space + attached), ${surface.moveIds.length} modeled moves`);
  lines.push(`→ **${surface.rows.length} (effect, hook) rows**.`);
  lines.push('');
  const counts = {};
  for (const row of Object.values(manifest.rows)) counts[row.disposition] = (counts[row.disposition] || 0) + 1;
  lines.push('| disposition | rows |');
  lines.push('|---|---|');
  for (const [d, n] of Object.entries(counts).sort()) lines.push(`| ${d} | ${n} |`);
  lines.push('');
  for (const kind of ['ability', 'item', 'condition', 'move']) {
    const rows = surface.rows.filter((r) => r.kind === kind);
    const perDisp = {};
    for (const r of rows) {
      const d = manifest.rows[r.key].disposition;
      perDisp[d] = (perDisp[d] || 0) + 1;
    }
    lines.push(`## ${kind} (${rows.length} rows: ${Object.entries(perDisp).sort().map(([d, n]) => `${d}=${n}`).join(', ')})`);
    lines.push('');
    lines.push('| effect | hook | disposition | anchor / reason |');
    lines.push('|---|---|---|---|');
    for (const r of rows) {
      const m = manifest.rows[r.key];
      const detail = m.anchor ? '`' + m.anchor + '`' : (m.reason || '');
      const probe = m.probe ? ` — probe: \`${m.probe}\`` : '';
      lines.push(`| ${r.id} | ${r.hook} | ${m.disposition} | ${detail}${probe} |`);
    }
    lines.push('');
  }
  fs.writeFileSync(OUT_MD, lines.join('\n') + '\n');
}

// ── main ─────────────────────────────────────────────────────────────────────
function main() {
  const surface = buildSurface();
  if (process.argv.includes('--enumerate')) {
    for (const r of surface.rows) console.log(`${r.key}\t${r.fp.slice(0, 110)}`);
    console.error(`rows: ${surface.rows.length} (abilities=${surface.abilityIds.length} items=${surface.itemIds.length} conditions=${surface.conditionIds.length} moves=${surface.moveIds.length})`);
    return;
  }
  if (process.argv.includes('--audit')) {
    audit(surface);
    return;
  }
  const manifest = buildManifest(surface);
  fs.writeFileSync(OUT_JSON, JSON.stringify(manifest, null, 1) + '\n');
  writeMd(surface, manifest);
  const counts = {};
  for (const row of Object.values(manifest.rows)) counts[row.disposition] = (counts[row.disposition] || 0) + 1;
  console.error(`rows: ${surface.rows.length}  [${Object.entries(counts).sort().map(([d, n]) => `${d}=${n}`).join(' ')}]`);
  console.error(`-> ${OUT_JSON}`);
  console.error(`-> ${OUT_MD}`);
}

main();
