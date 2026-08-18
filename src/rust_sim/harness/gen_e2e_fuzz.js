// gen_e2e_fuzz.js — THE CAPSTONE: end-to-end full-battle fuzz over REAL teams.
//
// Drives the OMNISCIENT in-process BattleStream (no server) AND records a compact
// per-decision golden that the Rust `run_full_battle` replays bit-for-bit. Unlike
// the constructed-scenario goldens (gen_fullbattle_golden.js / gen_secondary_
// golden.js — hand-picked mons + scripted choices), THIS harness:
//
//   * loads the 770 REAL Showdown-export teams under data/teams/ (sample/ + others/),
//     imports each with the real `Teams.import`, validates it under gen3ou, and packs
//     it (the EXACT bytes the Rust `team::unpack` ingests);
//   * pairs distinct teams + a battle seed from a MASTER-seeded RNG;
//   * at each decision reads the sim request and picks a RANDOM legal choice from a
//     SEPARATE seeded choice-RNG (recorded so a failing battle re-runs deterministically),
//     RESTRICTED to mechanics the Rust models (damaging fixed-BP moves with a modeled
//     secondary shape; else a SWITCH; else the battle is unmodeled-forced and dropped);
//   * runs to GAME-END, capturing the SAME per-decision record as gen_fullbattle_golden.js
//     (initSeed once; per boundary seedAfter + both actives' species/hp/maxhp/fainted/
//     status/boosts[5]/confusion + side pokemonLeft + request kind + first mover; winner)
//     PLUS the two packed teams + the recorded choice tokens.
//
// TWO OUTPUTS (the two deliverables):
//
//   (1) THE HARD GATE — tests/vectors/e2e_fuzz_golden.txt: ~FILTERED_TARGET battles
//       whose EVERY mon (both teams) uses ONLY modeled abilities/items, and where every
//       recorded choice is a modeled move or a switch. `tests/e2e_fuzz_test.rs` replays
//       these and asserts bit-for-bit to game-end (must be filtered_diverged == 0).
//
//   (2) THE COVERAGE TAXONOMY — tests/vectors/e2e_fuzz_taxonomy.txt: a separate UNFILTERED
//       sweep (real teams, NO ability/item pre-filter) that ranks coverage gaps by STATIC
//       TEAM COMPOSITION — i.e. which UNMODELED ability/item the PAIRED TEAMS CARRY, via
//       `classifyTeamsGaps(packed1, packed2)`. It is NOT an observed-first-divergence
//       classifier: it does not attribute the cause at the actual desync point, and it is
//       MOVE-LEVEL-BLIND (the sweep only ever picks damaging-or-switch choices, so
//       status moves / Spikes / Calm Mind / etc. are never chosen and so never appear as
//       a gap). It still RUNS each pair through the sim, but only to drop empty/errored
//       battles — the ranked counts come from the static team scan, not the run. So the
//       output is a "which unmodeled mechanic do real teams carry most" prioritised
//       remaining-work list, NOT a divergence-cause histogram. A ranked gap list — does
//       NOT gate cargo test.
//
// Determinism: the whole fuzz (team pairing, seeds, choices) is reproducible from the
// fixed MASTER_SEED below. A failing FILTERED battle re-runs from its recorded
// initSeed + choice tokens (the golden carries both).
//
// Run:  node src/rust_sim/harness/gen_e2e_fuzz.js
//   env knobs (optional): E2E_FILTERED_TARGET, E2E_UNFILTERED, E2E_MASTER_SEED, E2E_MAX_TRIES
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)
//
// IMPORTANT: this harness drives the SAME Rust engine the gen_fullbattle_golden.js /
// gen_secondary_golden.js goldens prove; it does not re-implement the Rust port. The
// taxonomy's divergence classification calls a tiny in-process Rust replayer ONLY via
// the committed Rust test (the harness itself is the GOLDEN producer + the Node-side
// taxonomy). The Rust gate asserts the FILTERED golden in tests/e2e_fuzz_test.rs.

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams, Dex, TeamValidator } = require(path.join(PS, 'dist/sim'));

const ROOT = path.resolve(__dirname, '../../..');
const TEAMS_DIR = path.join(ROOT, 'data/teams');
const RUST_MOVES = path.join(ROOT, 'data/pokemon/gen3_moves.json');
const OUT_GOLDEN = path.resolve(__dirname, '../tests/vectors/e2e_fuzz_golden.txt');
const OUT_TAXONOMY = path.resolve(__dirname, '../tests/vectors/e2e_fuzz_taxonomy.txt');

const FORMAT = 'gen3customgame'; // run format: arbitrary real teams, gen-3 mechanics
const VALIDATE_FORMAT = 'gen3ou'; // validate real teams as gen3ou (skip rejects)

const dex3 = Dex.forFormat(FORMAT);

// Tunables (env-overridable; defaults sized for a green gate + a real taxonomy).
// FILTERED_TARGET defaults to 220 — the size of the COMMITTED golden — so a plain
// `node gen_e2e_fuzz.js` regeneration reproduces `e2e_fuzz_golden.txt` byte-for-byte
// (the deterministic MASTER_SEED pairing makes battle e2e_N the N-th accepted battle).
const MASTER_SEED = Number(process.env.E2E_MASTER_SEED || 0x1234abcd) >>> 0;
const FILTERED_TARGET = Number(process.env.E2E_FILTERED_TARGET || 220);
const UNFILTERED_TARGET = Number(process.env.E2E_UNFILTERED || 300);
const MAX_TRIES = Number(process.env.E2E_MAX_TRIES || 6000);
const SAFETY = 1000; // max decisions per battle (real teams can grind)

function tick() { return new Promise((r) => setTimeout(r, 0)); }

// ── Deterministic choice/pairing RNG (separate from the battle PRNG) ──────────
// A simple recorded mulberry32 over a 32-bit state — Math-style but seeded, so a
// failing battle is reproducible from its recorded master-derived seed.
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randInt(rng, n) { return Math.floor(rng() * n); }

// A well-spread gen5 battle seed derived from a 32-bit state (same shape as the
// other goldens' seed pool).
function seedFrom(state) {
  let x = state >>> 0;
  const step = () => { x = (Math.imul(x, 1103515245) + 12345) >>> 0; return x & 0xffff; };
  return [step() || 1, step() || 1, step() || 1, step() || 1];
}

// ── Modeled-move predicate (the move allow/blocklist) ────────────────────────
// A move is MODELED iff it's a DAMAGING move with a FIXED dex base power and a
// secondary shape the Rust port handles. Driven by BOTH the sim move data (for the
// structural exclusions) AND the Rust gen3_moves.json (the port PANICS on >1
// secondary col except Tri Attack — so its secondaryEffects map must be ≤1 key).

const rustMoves = JSON.parse(fs.readFileSync(RUST_MOVES, 'utf8'));

// Explicit id blocklist: variable-power / fixed-damage / level / set-HP / OHKO /
// 2-turn / lock-in / item-loss / switch-trap / sleep-talk / multi etc. that pass the
// structural checks but the port does NOT model (per the capstone spec). When in
// doubt, EXCLUDE.
const MOVE_ID_BLOCKLIST = new Set([
  // variable power. NOTE: waterspout + beatup are NO LONGER blocklisted — they are MODELED
  // bit-for-bit AND e2e-ADMITTED (`gen3_move_coverage_batch4b_v1`, BATCH4B_E2E_EXCLUDED=false)
  // via MODELED_BATCH4B_MOVES in `isModeledMove`.
  // NOTE: return/frustration/flail/reversal/lowkick are NO LONGER blocklisted — they are
  // MODELED bit-for-bit AND e2e-ADMITTED (`gen3_move_coverage_batch5_v1`) via
  // MODELED_BATCH5_VARBP_MOVES in `isModeledMove`.
  'grassknot', 'magnitude', 'present', 'weatherball',
  'gyroball', 'fling', 'punishment', 'trumpcard', 'wringout', 'crushgrip',
  // hidden power: gated in `isModeledMove` by the `allowHiddenPower` param (the
  // byte fuzzer's `pool` mode admits it — engine models typed HP at fixed BP 70,
  // byte-correct for gen3ou-validated 70-BP-IV teams), NOT here (this reject runs
  // AFTER the `isHiddenPower` short-circuit, so a blocklist row would be dead).
  // fixed-damage / level / set-HP. NOTE: seismictoss/nightshade/sonicboom/dragonrage/
  // superfang were REMOVED from this list (`gen3_move_coverage_batch5_v1` housekeeping) —
  // they had been SHADOWING their own `MODELED_FIXED_DAMAGE_MOVES` admission (the
  // blocklist reject runs FIRST in `isModeledMove`), contradicting the documented
  // early-admit; they are modeled bit-for-bit (fixeddamage_test.rs + FD1-FD4).
  // counter/mirrorcoat/endeavor are likewise NO LONGER blocklisted — MODELED
  // (`gen3_move_coverage_batch5_v1`) via MODELED_BATCH5_REACTIVE_MOVES.
  'psywave', 'bide', 'finalgambit',
  // OHKO
  'fissure', 'horndrill', 'guillotine', 'sheercold',
  // switch-trap / item-swap / leaves-1 / fakeout / future / sleeptalk. NOTE:
  // knockoff/thief/covet (item REMOVAL) + rapidspin are NO LONGER blocklisted — they are
  // MODELED bit-for-bit (`gen3_move_coverage_batch1_v1`) and ADMITTED via
  // MODELED_ITEM_REMOVAL_MOVES / MODELED_RAPIDSPIN_MOVES in `isModeledMove`. `pursuit` is likewise
  // NO LONGER blocklisted — it is MODELED bit-for-bit AND e2e-ADMITTED (`gen3_move_coverage_batch4_v1`,
  // BATCH4_E2E_EXCLUDED=false) via MODELED_BATCH4_MOVES in `isModeledMove`. NOTE: `trick` (the item
  // SWAP move) is NO LONGER blocklisted — MODELED bit-for-bit (`gen3_trick_v1`) and ADMITTED via
  // MODELED_TRICK_MOVES in the Status branch. `switcheroo` (num 415) is a gen4 move — NOT gen3-legal
  // (no gen3 mon learns it) — so it stays out of the modeled universe.
  // NOTE: futuresight/doomdesire are NO LONGER blocklisted — MODELED bit-for-bit AND
  // e2e-ADMITTED (`gen3_move_coverage_batch4c_v1`, BATCH4C_E2E_EXCLUDED=false) via
  // MODELED_BATCH4C_MOVES in `isModeledMove`.
  'switcheroo', 'falseswipe',
  // NOTE: `sleeptalk` is NO LONGER blocklisted — MODELED (`gen3_move_coverage_batch5_v1`);
  // its pickability is CARRIER-conditional (see `sleepTalkPoolModeled` in pickMove: the
  // sampled pool must be all-modeled, since the CALLED move bypasses the picker). `snore`
  // (the other gen-3 sleepUsable move) stays out: its while-asleep execution + awake
  // onTry fail are unmodeled.
  // NOTE: `fakeout` is NO LONGER blocklisted — MODELED (`gen3_fakeout_v1`) and ADMITTED via
  // MODELED_FAKEOUT_MOVES. `snore` stays out (its while-asleep execution + awake onTry fail are
  // unmodeled, and run_move fail-louds on it).
  'snore',
  // NOTE: `destinybond` is NO LONGER blocklisted — MODELED bit-for-bit
  // (`gen3_move_coverage_batch6_v1`) and ADMITTED via MODELED_BATCH6_MOVES in the
  // Status branch of `isModeledMove`. `snatch` (the LAST gen-3 status move) is likewise
  // NO LONGER blocklisted — MODELED bit-for-bit (`gen3_snatch_v1`) and ADMITTED via
  // MODELED_SNATCH_MOVES in the Status branch (the port no longer FAIL-LOUDs on it).
  // NOTE: Explosion / Self-Destruct are NO LONGER blocklisted — they are FULLY modeled
  // bit-for-bit (the gen-3 self-KO that precedes the hit; `gen_explosion_golden.js` /
  // `explosion_test.rs` + the E1-E4 regression pins) and ADMITTED to the e2e capstone via
  // the `m.selfdestruct` special-case below (gated by EXPLOSION_E2E_EXCLUDED).
]);

// The MODELED standalone status-inflicting moves (category Status, bp 0) the port
// now executes bit-for-bit (the status-move layer): par (Thunder Wave / Stun Spore /
// Glare), psn (Poison Powder / Poison Gas), tox (Toxic), brn (Will-O-Wisp), slp
// (Spore / Sleep Powder / Hypnosis / Sing / Lovely Kiss / Grass Whistle). Every OTHER
// Status move (recovery / boost / phaze / hazard / Substitute / field) stays excluded
// (the port FAIL-LOUDs on them). Kept in lockstep with `modeled_status_move` in
// src/turn.rs.
const MODELED_STATUS_MOVES = new Set([
  // CONFUSE RAY (`gen3_confuse_ray_v1`) — a VOLATILE-inflicting status move (the rest of this
  // set inflicts MAJOR statuses). Admitted once the engine grew its arm; the shared
  // `add_confusion` path supplies the random(2,6) duration draw and the `-start|confusion`.
  'confuseray',
  'thunderwave', 'stunspore', 'glare',
  'poisonpowder', 'poisongas',
  'toxic',
  'willowisp',
  'spore', 'sleeppowder', 'hypnosis', 'sing', 'lovelykiss', 'grasswhistle',
]);

// The MODELED pure SELF-BOOST SETUP moves (category Status, bp 0, target self) the
// port now executes bit-for-bit (the setup-move layer): Calm Mind / Dragon Dance /
// Swords Dance / Agility / Bulk Up / Amnesia / Barrier / Acid Armor / Iron Defense /
// Cosmic Power / Tail Glow / Meditate / Sharpen / Howl / Harden / Withdraw / Growth.
// DERIVED from the Rust data (`gen3_moves.json`'s `selfBoosts`, populated only for the
// pure setup moves) so the allow-list is GIGO-PROOF in lockstep with the engine's
// `self_boost_spec` (every move the port applies a self-boost for is exactly the set
// with a non-empty `selfBoosts`; moves with an extra effect — Defense Curl/Minimize
// volatile, Double Team's evasion, Belly Drum's HP cost, Curse — carry NO `selfBoosts`
// and stay excluded → the port FAIL-LOUDs on them).
const MODELED_SETUP_MOVES = new Set(
  Object.keys(rustMoves).filter((id) => {
    const sb = rustMoves[id] && rustMoves[id].selfBoosts;
    return sb && typeof sb === 'object' && Object.keys(sb).length > 0;
  })
);

// The MODELED self-targeting HP-RECOVERY moves (category Status, bp 0, target self,
// isHeal) the port now executes bit-for-bit (the recovery-move layer): the flat-half-HP
// recovers (Recover / Soft-Boiled / Slack Off / Milk Drink → floor(maxhp/2)), the
// WEATHER-conditional heals (Moonlight / Synthesis / Morning Sun → none floor(maxhp/2) /
// sun floor(maxhp*2/3) / sand+rain+hail floor(maxhp/4)), and REST (full heal + a FIXED
// Sleep(3) self-sleep that DRAWS-then-DISCARDS the slp.onStart `random(2,6)` + a prior-
// status cure). Kept in lockstep with `recovery_heal_amount` + `run_rest` in src/turn.rs.
// DELIBERATELY EXCLUDED (the port FAIL-LOUDs → they keep the team out of the filtered
// gate): **Wish** (a DELAYED slot-keyed end-of-next-turn heal — a separate pending-heal
// model), **Heal Bell / Aromatherapy / Refresh** (team/self STATUS cure, not HP),
// **Pain Split / Leech Seed / drain / Ingrain / Aqua Ring** (other isHeal mechanics).
// `splash` (a true draw-free no-op) is ALSO modeled by the port and allowed here.
const MODELED_RECOVERY_MOVES = new Set([
  'recover', 'softboiled', 'slackoff', 'milkdrink',
  'moonlight', 'synthesis', 'morningsun',
  'rest',
  'splash',
]);

// The MODELED PROTECT / DETECT moves (category Status, bp 0, target self, priority 3,
// `stallingMove`/`volatileStatus:'protect'`) the port now executes bit-for-bit (the
// protect-move layer): the gen-3 consecutive-use STALL draw (the first protect short-
// circuits with NO draw; a consecutive one draws one randomChance(1, counter) at the
// floored 2/4/8 denominator) + the move-BLOCK (a foe move targeting the protected mon
// draws its accuracy roll then is blocked, NO crit/damage/secondary/status). Kept in
// lockstep with `run_protect` in src/turn.rs. DELIBERATELY EXCLUDED (the port FAIL-LOUDs
// → they keep the team out of the filtered gate): **Endure** (isProtect=true but
// `volatileStatus:'endure'` — a survive-at-1-HP `onDamage`, a different mechanic) +
// the gen4+ Quick Guard / Wide Guard / King's Shield / Spiky Shield (none exist in gen3).
const MODELED_PROTECT_MOVES = new Set(['protect', 'detect']);

// The MODELED ENTRY-HAZARD moves (category Status, bp 0, `sideCondition`, target foeSide)
// the port now executes bit-for-bit (the spikes layer). gen-3's ONLY entry hazard is
// **Spikes**: a never-miss `sideCondition:'spikes'` move that increments the FOE side's
// layer count (capped at 3; a 4th FAILS) draw-free, and applies the grounded switch-in
// damage (`[_,3,4,6][layers]*maxhp/24`, floored, min 1) on the gen-3 runSwitch's
// `runEvent('EntryHazard')`. Kept in lockstep with `run_status_move`'s spikes arm +
// `apply_entry_hazards` in src/turn.rs. DELIBERATELY EXCLUDED (the port FAIL-LOUDs → they
// keep the team out of the filtered gate): **Toxic Spikes** + **Stealth Rock** (NOT gen3),
// and **Rapid Spin** (the hazard-CLEAR move — a damaging move the fuzz won't pick as a
// modeled status move; hazards persist). Spikes is the only gen-3 entry hazard.
const MODELED_HAZARD_MOVES = new Set(['spikes']);

// The MODELED PHAZE moves (category Status, bp 0, `forceSwitch: true`, target normal) the
// port now executes bit-for-bit: **Roar** + **Whirlwind** — force the FOE to switch to a
// RANDOM eligible team member. The draw model (kept in lockstep with `run_status_move`'s
// phaze arm + `drag_in` in src/turn.rs): gen-3 Roar/Whirlwind resolve to `accuracy: 100`
// (NOT never-miss) so they DRAW `randomChance(100,100)`; a SUCCESSFUL phaze then draws ONE
// `sample`/`random(n)` (the random target, even for n==1) and `dragIn`s the picked mon (it
// takes Spikes via the runSwitch EntryHazard, fires its ability Start, and a Spikes-KO on
// entry chains a normal replacement); a phaze with NO eligible foe (last mon) draws ONLY the
// accuracy roll. DELIBERATELY EXCLUDED (the port FAIL-LOUDs): **Haze** (resets boosts — a
// DIFFERENT mechanic, not forceSwitch), Perish Song, Roar of Time (not gen3). Roar +
// Whirlwind are the ONLY gen-3 phaze moves (`forceSwitch`). The general `m.forceSwitch`
// reject below (the damaging-move path) never sees these — they are category Status and
// matched HERE first.
const MODELED_PHAZE_MOVES = new Set(['roar', 'whirlwind']);

// The MODELED LEECH SEED move (category Status, bp 0, type Grass, accuracy 90,
// `volatileStatus:'leechseed'`, target normal) the port now executes bit-for-bit (the
// leech-seed layer): plant the `leechseed` volatile on the FOE; each end-of-turn the seeded
// mon loses floor(maxhp/8) and the SEEDER's CURRENT active heals it (the gen4-inherited
// residual at order 10, subOrder 5 — between Leftovers sub 4 and the status DoT sub 6). The
// move DRAWS its accuracy roll (acc 90, NOT never-miss) unconditionally — even into a
// Grass-immune or already-seeded target — then plants (draw-free) on a landed hit. Kept in
// lockstep with `run_status_move`'s leechseed arm + `apply_leech_seed` in src/turn.rs.
// DELIBERATELY EXCLUDED at the residual (the port FAIL-LOUDs): a Liquid Ooze target reverses
// the drain (seeder takes damage) — rare in gen-3 OU; the MOVE-ID-BLOCKLIST keeps such teams
// out. Leech Seed is the only gen-3 drain-volatile move modeled here.
const MODELED_LEECH_MOVES = new Set(['leechseed']);

// MOVE-COVERAGE BATCH 3 (`gen3_move_coverage_batch3_v1`) — the three STATEFUL move classes the
// port now executes bit-for-bit (the batch-3 layer): CURSE (type-conditional: non-ghost self-
// boost {atk:+1,def:+1,spe:-1} drawing ONE selfDrops random(100), OR ghost floor(maxhp/2) HP +
// the `curse` volatile on the foe + the order-10-subOrder-8 residual chip), WISH (the slot-
// keyed order-7 delayed heal floor(maxhp/2) at N+1; double-Wish fails; heal-at-full silent),
// and BATON PASS (the selfSwitch:'copyvolatile' pass of the boosts + copyable volatiles
// [substitute/leech-seed/confusion/curse] to the entrant; no-bench fail). All three are
// category-Status, admitted in the Status branch (gated by BATCH3_E2E_EXCLUDED). Kept in
// lockstep with `run_status_move`'s curse/wish/batonpass arms + `apply_curse`/`apply_wish` +
// the `copyVolatileFrom` snapshot in `execute_switch` (src/turn.rs). Validated by the DEDICATED
// golden (`gen_movecoverage_batch3_golden.js` / `movecoverage_batch3_test.rs`) + the MC18-MC29
// regression pins. DEFERRED (still fail-loud / excluded): Haze (a boost-reset, not a phaze),
// Perish Song, Encore (reactive), Attract (the volatile move — Cute Charm's attract is a
// separate ability path). `whether-to-admit` is BATCH3_E2E_EXCLUDED below.
const MODELED_BATCH3_MOVES = new Set(['curse', 'wish', 'batonpass']);

// MOVE-COVERAGE BATCH 4 (`gen3_move_coverage_batch4_v1`) — the two DAMAGING moves with a
// `beforeTurnCallback`: FOCUS PUNCH (the beforeTurnMove `|-singleturn|` + the onTry cant-if-hit
// gate + the flinch-block) and PURSUIT (the beforeTurnMove lays the `pursuit` volatile; the
// switch-interrupt strikes the switching mon at ×2 BP + never-miss via `basePowerCallback` /
// `onModifyMove`). Both are damaging (bp>0), so they are ADMITTED by an EARLY special-case in
// `isModeledMove` (before the `beforeTurnCallback` / `basePowerCallback` / `onModifyMove`
// rejects that would else drop them). Kept in lockstep with `move_has_before_turn_callback` +
// the FP/Pursuit arms in src/turn.rs. Validated by the DEDICATED golden
// (`gen_movecoverage_batch4_golden.js` / `movecoverage_batch4_test.rs`) + the MC30-MC35 pins.
const MODELED_BATCH4_MOVES = new Set(['focuspunch', 'pursuit']);
// BATCH4_E2E_EXCLUDED — whether to keep FOCUS PUNCH + PURSUIT OUT of the e2e capstone's modeled
// set. Now `false` (ADMITTED, `gen3_move_coverage_batch4_v1`): STRICT `filtered_diverged == 0`
// over the 220-battle gate (242 Focus Punch / 184 Pursuit decisions). The former deferral (the
// PHAZE precedent — a pursuit-interrupt bench-order desync composed with Baton Pass) was
// ROOT-CAUSED + FIXED: the port's `execute_switch` used to fire the pursuit interrupt for ANY
// non-drag switch INCLUDING a Baton-Pass selfSwitch, striking the still-active passer — while the
// sim SUPPRESSES `BeforeSwitchOut` for a Baton Pass (`batonpass.self.onHit` sets
// `skipBeforeSwitchOutEventFlag`, moves.ts:1109). Gating the interrupt on `is_voluntary` (only a
// menu switch) fixed the spurious-strike / fainted-in-a-slot desync; admitting also surfaced +
// fixed (a) a first-mover attribution nuance (the pursuer's `|move|` emits before the switcher's
// `|switch|` → `pursuit_first_mover`) and (b) a Choice-lock-not-released-on-item-removal bug (a
// Thief'd Choice Band mon must be freed to re-pick — e2e_126). Pins MC36/MC36b/MC37/MC38.
const BATCH4_E2E_EXCLUDED = false;

// MOVE-COVERAGE BATCH 4b (`gen3_move_coverage_batch4b_v1`) — the THREE remaining MISMODELED
// single-turn damaging moves, all now MODELED bit-for-bit (the DEDICATED golden
// `gen_movecoverage_batch4b_golden.js` / `movecoverage_batch4b_test.rs` + the MC39-MC43 pins):
//   * BEAT UP — a MULTI-STRIKE stat-swap move (static dex has `basePowerCallback` + `onModifyMove`
//     setting `multihit`; the port runs the per-strike loop).
//   * THUNDER — a 120-BP Special Electric move whose id-gated `onModifyMove` rewrites the base
//     accuracy by the target's effective weather (rain never-miss / sun 50 / else 70).
//   * WATER SPOUT — a variable-BP Special Water move (`basePowerCallback` = 150·hp/maxhp).
// In the Showdown dex Beat Up + Water Spout carry `basePowerCallback` and Thunder + Beat Up carry
// `onModifyMove`, so they'd else be dropped by those rejects below — `isModeledMove` ADMITS them by
// an EARLY special-case (before those rejects), kept in lockstep with the id-gates in src/turn.rs
// (`run_beat_up` / the waterspout BP override / the thunder weather-accuracy mutation).
// `eruption` joins on `gen3_eruption_v1` — the SAME HP-scaled BP callback as waterspout,
// admitted to the same id-gate in turn/moves.rs.
const MODELED_BATCH4B_MOVES = new Set(['beatup', 'thunder', 'waterspout', 'eruption']);
const BATCH4B_E2E_EXCLUDED = false;

// MOVE-COVERAGE BATCH 7 (`gen3_move_coverage_batch7_v1`) — the GENERIC MULTI-STRIKE family: the
// FIXED-2 moves (Double Kick / Twineedle / Bonemerang) + the VARIABLE [2,5] family (Pin Missile /
// Bullet Seed / Icicle Spear / Rock Blast / Barrage / Comet Punch / Double Slap / Spike Cannon /
// Arm Thrust / Fury Attack / Fury Swipes / Bone Rush). All carry `m.multihit` (which is rejected
// below), so `isModeledMove` ADMITS them by an EARLY special-case (before the `m.multihit` reject),
// kept in lockstep with `run_multihit` in src/turn.rs. TRIPLE KICK (`triplekick`) is EXCLUDED — it
// is `multiaccuracy` (a per-strike accuracy re-roll + escalating BP), which the engine FAIL-LOUDS
// on, so the picker must never pick it. Twineedle's 20% psn is the ordinary per-strike secondary.
const MODELED_BATCH7_MULTIHIT_MOVES = new Set([
  'doublekick', 'twineedle', 'bonemerang',
  'pinmissile', 'bulletseed', 'iciclespear', 'rockblast', 'barrage', 'cometpunch', 'doubleslap',
  'spikecannon', 'armthrust', 'furyattack', 'furyswipes', 'bonerush',
]);
const BATCH7_E2E_EXCLUDED = false;

// MOVE-COVERAGE BATCH 4c (`gen3_move_coverage_batch4c_v1`) — the TURN-SPANNING classes:
// HYPER BEAM (mustrecharge — the locked single-`Recharge` request), SOLAR BEAM (the
// two-turn charge + sun skip — the locked single-move request), DOOM DESIRE + FUTURE
// SIGHT (the slot-keyed order-11 future strike). All bit-for-bit modeled
// (movecoverage_batch4c_test.rs + the MC49+ pins). `recharge` is the locked-turn
// pseudo-move id the sim's request offers — admitted so the picker submits `move 1` on a
// locked request instead of stalling. The UNMODELED siblings stay fail-loud + rejected
// (Blast Burn / Frenzy Plant / Hydro Cannon via `self.volatileStatus`+`flags.recharge`;
// Razor Wind / Sky Attack / Skull Bash / Fly / Dig / Dive / Bounce via `flags.charge`;
// no other gen3 `futuremove` exists).
const MODELED_BATCH4C_MOVES = new Set(['hyperbeam', 'solarbeam', 'doomdesire', 'futuresight', 'recharge']);
const BATCH4C_E2E_EXCLUDED = false;

// MOVE-COVERAGE BATCH 5 (`gen3_move_coverage_batch5_v1`): the REACTIVE fixed-damage family
// (COUNTER / MIRROR COAT — the beforeTurnMove volatile + the onDamage 2× recorder;
// ENDEAVOR — the target.hp − user.hp delta), the VARIABLE-BP family (RETURN / FRUSTRATION /
// FLAIL / REVERSAL / LOW KICK — engine-computed BP, draw-neutral), and SLEEP TALK (the
// move-sampler — carrier-conditional pickability, see `sleepTalkPoolModeled`). All modeled
// bit-for-bit (`gen_movecoverage_batch5_golden.js` / `movecoverage_batch5_test.rs` + the
// MC61+ pins), kept in lockstep with the src/turn.rs id-gates (`is_fixed_damage_move` /
// `variable_bp` / the sleeptalk arm).
const MODELED_BATCH5_REACTIVE_MOVES = new Set(['counter', 'mirrorcoat', 'endeavor']);
const MODELED_BATCH5_VARBP_MOVES = new Set(['return', 'frustration', 'flail', 'reversal', 'lowkick']);
const BATCH5_E2E_EXCLUDED = false;

// MOVE-COVERAGE BATCH 6 (`gen3_move_coverage_batch6_v1`) — the FINAL UNMODELED tail,
// all category-Status: ENCORE (the lock + the onOverrideAction execution override) /
// DESTINY BOND (the mutual-faint reactive volatile) / ENDURE (the protect/stall-family
// survive-at-1) / PERISH SONG (the field-wide order-12 counter) / MEAN LOOK / SPIDER
// WEB / BLOCK (the linked firm-trap volatiles) / BELLY DRUM / CHARGE / MEMENTO / MIMIC
// / PAIN SPLIT / PSYCH UP. Kept in lockstep with the src/turn.rs batch-6 arms
// (validated by gen_movecoverage_batch6_golden.js + the MC79+ pins). MIMIC's copy is
// picker-safe: the copied slot is always the target's lastMove, which the capture's
// picker only ever lets be a MODELED move (both sides pick modeled moves only) — and
// the copied slot itself re-passes `isModeledMove` at every later pick.
const MODELED_BATCH6_MOVES = new Set([
  'encore', 'destinybond', 'endure', 'perishsong',
  'meanlook', 'spiderweb', 'block',
  'bellydrum', 'charge', 'memento', 'mimic', 'painsplit', 'psychup',
]);
const BATCH6_E2E_EXCLUDED = false;

// SNATCH (`gen3_snatch_v1`) — the LAST unmodeled gen-3 status move, MODELED bit-for-bit
// (the interception + cast in `run_status_move`; the DEDICATED golden + MC100-MC104 pins).
// A snatcher STEALS the next foe self-targeted `flags.snatch` status move; the port's
// interception recurses into `run_status_move` for the stolen move, so any snatchable move
// the picker CAN pick (all in `isModeledMove`) executes bit-for-bit when stolen. Admitting
// it closes 722/722 (the last team-blocking move). The port no longer FAIL-LOUDs on snatch.
const MODELED_SNATCH_MOVES = new Set(['snatch']);
const SNATCH_E2E_EXCLUDED = false;

// HAZE (`gen3_haze_v1`) — a category-Status FIELD move (`target: all`, `accuracy: true`) that
// emits ONE `|-clearallboost` line + zeroes BOTH actives' boost stages (incl. the USER's own).
// DRAW-FREE (probe-settled). Modeled bit-for-bit in `run_status_move`'s haze arm (the DEDICATED
// golden + the HZ1/HZ2 pins). 0 team-carry (no gen3ou sample team runs Haze — the leech-seed
// situation), so admitting it is byte-neutral on the e2e sample; kept for random-battle coverage.
const MODELED_HAZE_MOVES = new Set(['haze']);
const HAZE_E2E_EXCLUDED = false;

// YAWN (`gen3_yawn_v1`) — a category-Status foe-target DELAYED-SLEEP move (`volatileStatus: 'yawn'`,
// `accuracy: true`). The CAST is DRAW-FREE; the sleep `random(2,6)` fires at the RESOLVE (the residual
// `onEnd` at order 10 subOrder 19, end of the turn AFTER cast), routed through the EXISTING
// `try_set_status('slp')` path (so the sleep counter + the gen3ou Sleep Clause + the SetStatus shuffle
// all come for free). Modeled bit-for-bit in `run_status_move`'s yawn arm + the residual `Yawn`
// handler (the DEDICATED golden + the Y1/Y2/Y3 pins). UNLIKE Haze, Yawn MAY have gen3ou team-carry —
// admitting it can grow the filter-clean pool; a plain regen reproduces the committed golden iff the
// sampled teams don't carry it.
const MODELED_YAWN_MOVES = new Set(['yawn']);
// SAFEGUARD (`gen3_safeguard_v1`) — a never-miss allySide SideCondition. The CAST is
// entirely DRAW-FREE; the block is too, and BOTH sides carrying it TIE at residual order 4
// for one extra shuffle. Blocks FOE-sourced major status + confusion; a SELF Rest passes.
const MODELED_SAFEGUARD_MOVES = new Set(['safeguard']);
const YAWN_E2E_EXCLUDED = false;

// TRICK (`gen3_trick_v1`) — a category-Status ITEM-SWAP move (`target: normal`, accuracy 100 → ONE
// `randomChance(100,100)` draw, NO `bypasssub`). ONE accuracy draw then a DRAW-FREE swap: Sticky Hold
// blocks (`-immune`), a Substitute / both-itemless / knocked-off item FAILS (`[still]`+`-fail`), else
// the two items swap. Modeled bit-for-bit in `run_status_move`'s trick arm (the DEDICATED golden +
// the TR1-TR5 pins). `switcheroo` (num 415) is gen4 — NOT gen3-legal — so Trick is the only gen-3
// item-swap move. UNLIKE Haze, Trick MAY have gen3ou team-carry, so admitting it can grow the
// filter-clean pool / change the sampled golden; a plain regen reproduces the committed golden iff the
// sampled teams don't carry it.
const MODELED_TRICK_MOVES = new Set(['trick']);
const TRICK_E2E_EXCLUDED = false;

// PARTIAL TRAP (`gen3_partial_trap_v1`) — the six DAMAGING moves whose gen-3 dex row carries
// `volatileStatus: 'partiallytrapped'`: wrap(35)/bind(20)/firespin(83)/clamp(128)/whirlpool(250)/
// sandtomb(328). Their draw model is the ORDINARY damaging chain (accuracy 70-85 → crit → damage
// roll; none has a secondary) PLUS **ONE** `random(3, 7)` at cast — the gen4-mod
// `partiallytrapped.durationCallback` fired inside `addVolatile`. The volatile then chips
// `floor(maxhp/16)` DRAW-FREE at order 10 subOrder 9 for `duration − 1` turns and FIRM-traps the
// victim. ADMITTED HERE (before the `m.volatileStatus` reject below, which would else drop all
// six). Modeled bit-for-bit in `run_move`'s partial-trap arm + the residual `PartialTrap` handler
// (`harness/probe_batch89_trap.js` + `probe_ptrap_edges{,2}.js`; the DEDICATED golden + the
// PT1-PT* pins). They MAY have gen3ou team-carry, so admitting them can change the sampled e2e
// golden — a plain regen reproduces the committed golden iff the sampled teams don't carry one.
const MODELED_PARTIALTRAP_MOVES = new Set([
  'wrap', 'bind', 'firespin', 'clamp', 'whirlpool', 'sandtomb',
]);
const PARTIALTRAP_E2E_EXCLUDED = false;

// The MODELED gen-3 FIXED-DAMAGE moves (a `damage:` / `damageCallback` move that BYPASSES
// getDamage — NO crit roll, NO 16-way damage roll) the port now executes bit-for-bit (the
// fixed-damage layer): Seismic Toss / Night Shade (damage: 'level' → the USER's level),
// Sonic Boom (20), Dragon Rage (40), Super Fang (max(floor(target.hp/2),1)). Their draw model
// is accuracy-only (Seismic Toss / Night Shade / Dragon Rage acc-100 but NOT never-miss → they
// STILL draw one; Sonic Boom / Super Fang acc-90 CAN miss), then the type-immunity short-circuit
// (accuracy-drawn-then-`-immune`), then apply the fixed amount through the sub-absorb / faint
// machinery. In the SHOWDOWN dex these carry `category:'Physical'|'Special'` with `basePower:0`
// (so they fail the `basePower > 0` gate) and a `damage`/`damageCallback` field (so they fail
// those rejects) — `isModeledMove` ADMITS them by an EARLY special-case (before those rejects),
// kept in lockstep with `is_fixed_damage_move` / `fixed_damage_amount` in src/turn.rs. The
// DEFERRED fixed-damage family (Psywave / the OHKO moves / Counter / Mirror Coat / Bide /
// Endeavor) stays out — the port FAIL-LOUDs on it, and the `basePower`/`damage`/ohko/
// `damageCallback` rejects + the blocklist keep it off the pickable path.
const MODELED_FIXED_DAMAGE_MOVES = new Set([
  'seismictoss', 'nightshade', 'sonicboom', 'dragonrage', 'superfang',
]);

// MOVE-COVERAGE BATCH 1 (`gen3_move_coverage_batch1_v1`): the DRAW-FREE post-hit effects a
// damaging move drops after a landed hit — RECOIL / DRAIN / SELF-DROP / ITEM-REMOVAL / RAPID-
// SPIN — the port now executes bit-for-bit (`gen_movecoverage_batch1_golden.js` /
// `movecoverage_batch1_test.rs` + the MC1-MC6 regression pins). Each is an EXPLICIT set (like
// MODELED_STATUS_MOVES) kept in lockstep with src/turn.rs — NOT a blanket `m.recoil`/`m.drain`
// allow — so a recoil/drain move carrying an EXTRA unmodeled mechanic (Dream Eater's
// sleep-only `onTryImmunity`) stays out.
//
// RECOIL — `recoil:[num,den]`: the USER takes max(floor(dmgDealt·num/den),1) HP; Rock Head
// negates; fires behind a sub. DRAW-FREE. The clean gen-3 recoil moves (no charge/callback/
// onModifyMove): Double-Edge / Take Down / Submission (Struggle is its own path).
// `volttackle` (`gen3_move_coverage_batch7_v1`): a Pikachu-only 120-BP recoil-1/3 move — in GEN 3
// it has NO secondary (the 10% para was added in gen4), so it is an ordinary recoil move the
// engine already prices via `recoil_fraction` (`apply_recoil`); just picker-admitted here.
const MODELED_RECOIL_MOVES = new Set(['doubleedge', 'takedown', 'submission', 'volttackle']);
// DRAIN — `drain:[num,den]`: the USER heals the fraction of the damage dealt (floor non-sub /
// ceil behind a sub); heal-at-full fails. DRAW-FREE. Liquid Ooze reverses it (fail-loud —
// excluded via the NOOP-ability filter). Dream Eater is EXCLUDED (its `onTryImmunity`
// sleep-only gate is unmodeled). Clean gen-3 drain moves: Absorb / Mega Drain / Giga Drain /
// Leech Life.
const MODELED_DRAIN_MOVES = new Set(['absorb', 'megadrain', 'gigadrain', 'leechlife']);
// SELF-DROP — the top-level `move.self.boosts` on a DAMAGING move (Overheat −2 SpA, Superpower
// −1 Atk/−1 Def): the port applies the drop AND draws the gen3 `selfDrops` random(100) (the
// `secondaryRoll`, applied unconditionally since `self.chance === undefined`) — NOT draw-free.
// `psychoboost` (`gen3_move_coverage_batch7_v1`): a Deoxys 140-BP Special move with `self.boosts
// {spa:-2}` — the SAME `selfDrops` `random(100)` path as Overheat, already engine-modeled.
const MODELED_SELFDROP_MOVES = new Set(['overheat', 'superpower', 'psychoboost']);
// ITEM REMOVAL — Knock Off (removes; gen3 no dmg boost) / Thief / Covet (steal iff the attacker
// is itemless); Sticky Hold blocks; the onAfterHit fires ONLY when the MON was damaged (not
// behind a sub). DRAW-FREE. A Liquid-Ooze-style item complication doesn't exist here.
const MODELED_ITEM_REMOVAL_MOVES = new Set(['knockoff', 'thief', 'covet']);
// RAPID SPIN — a 20-BP damaging move whose onAfterHit + onAfterSubDamage clear the USER's own
// Spikes + Leech Seed (+ partial-trap, not modeled — no partial-trap move in scope). DRAW-FREE;
// clears behind a sub too. gen3 has only Spikes among the hazards.
const MODELED_RAPIDSPIN_MOVES = new Set(['rapidspin']);

// The modeled gen-3 SUBSTITUTE move (`volatileStatus:'substitute'`, never-miss): the user
// spends floor(maxhp/4) HP to make a decoy that ABSORBS foe hits. The port models the create
// (cost + draw-free), the absorb (damage → sub HP, break at 0, no carry), the SECONDARY that
// STILL DRAWS its random(100) but applies NOTHING behind a sub (the gen-3 draw-count quirk),
// the blocked status/stat-drop, the confusion-self-hit-hits-the-mon, and the phaze-bypass — all
// bit-for-bit (`gen_substitute_golden.js` / `substitute_test.rs`).
const MODELED_SUBSTITUTE_MOVES = new Set(['substitute']);

// The modeled gen-3 move-SELECTION-RESTRICTION moves (`gen3_taunt_disable_v1`): **Taunt**
// (Dark, acc 100 — DRAWS randomChance(100,100); the `taunt` volatile is a FIXED duration:2,
// NO duration draw; blocks every Status-category move at selection + cants a QUEUED one at
// execution) and **Disable** (Normal, acc 55 — CAN miss; ONE random(2,6) durationCallback
// draw on a landed hit into a mon with a lastMove — stored +1 iff the target ALREADY moved
// [the gen4-inherited onStart `!willMove -> duration++`]; disables that one slot; onTryHit
// FAILS draw-free with no lastMove). Both protect:1 (Protect blocks, after their accuracy
// roll) + bypasssub:1 (a Substitute does NOT block). The port models the selection
// restriction (`move_usable`/`must_struggle` -> forced Struggle), the execution-time
// onBeforeMove cants, and the residual duration ticks bit-for-bit
// (`gen_taunt_disable_golden.js` / `taunt_disable_test.rs` + the TD1-TD3 regression pins).
// `torment` joins on `gen3_torment_v1` — the same selection-time family, minus the duration.
const MODELED_RESTRICTION_MOVES = new Set(['taunt', 'disable', 'torment']);
// RECYCLE (`gen3_recycle_v1`) — restores the item the mon CONSUMED ITSELF (eatItem/useItem),
// never one taken by Knock Off / Thief / Trick. Never-miss, zero draws.
const MODELED_RECYCLE_MOVES = new Set(['recycle']);
// SKILL SWAP (`gen3_skill_swap_v1`) — never-miss, draw-free, ONE gen<=4 activate line; the
// swapped-in abilities do NOT re-fire onStart, but both outgoing abilities DO fire onEnd.
const MODELED_SKILLSWAP_MOVES = new Set(['skillswap']);
// FAKE OUT (`gen3_fakeout_v1`) — priority +1 (NOT the modern +3); the flinch is a chance-100
// secondary that STILL rolls random(100); the first-turn gate counts ACTIONS, not turns.
const MODELED_FAKEOUT_MOVES = new Set(['fakeout']);

// MOVE-COVERAGE BATCH 2 (`gen3_move_coverage_batch2_v1`) — the four DRAW-friendly status-move
// classes the port now models bit-for-bit (`gen_movecoverage_batch2_golden.js` /
// `movecoverage_batch2_test.rs` + the MC9-MC17 regression pins). All are category-Status, so
// they're admitted in the Status branch below (kept in LOCKSTEP with src/turn.rs):
//   * STATUS-CURE — Refresh (self par/psn/brn), Heal Bell + Aromatherapy (whole-team major-
//     status cure incl. bench, Heal Bell skips a Soundproof ally). NEVER-MISS + DRAW-FREE.
//   * WEATHER-SET — Rain Dance / Sunny Day: a 5-turn TIMED weather (the eachEvent
//     ('WeatherChange') tie-shuffle draws only on a speed tie; setWeather fails draw-free into
//     the same weather). NEVER-MISS.
//   * STAT-DROP — Screech / Charm / Metal Sound / Feather Dance / Tickle / Fake Tears / Cotton
//     Spore / Scary Face: accuracy draw + a draw-free foe stat-drop `boost()` (Clear Body /
//     Hyper Cutter / Soundproof gated).
//   * SCREENS — Light Screen / Reflect: a 5-turn SIDE condition (halves special / physical;
//     the DAMAGE calc reads it). A physical/special hit into a side with BOTH screens up draws
//     the ModifyDamagePhase1 shuffle (the port models it). NEVER-MISS + draw-free set.
const MODELED_CURE_MOVES = new Set(['refresh', 'healbell', 'aromatherapy']);
// + hail / sandstorm (`gen3_forecast_v1`, ROUND 35): the last two C_WEATHER_SET members,
// modeled through the SAME batch-2 machinery (5-turn timed set / fail-into-same / upkeep /
// expiry; the sand+hail chips + immunities were already modeled for the ability weathers).
// Probe: `harness/probe_r35_weather_moves.js` (tied-board byte + draw parity).
const MODELED_WEATHER_MOVES = new Set(['raindance', 'sunnyday', 'hail', 'sandstorm']);
// DERIVED from `gen3_moves.json`'s `statDropBoosts` — the `MODELED_SETUP_MOVES` precedent, so the
// allow-list stays GIGO-proof and in lockstep with the engine instead of drifting from it.
// It was a HARDCODED 8-id list until `gen3_sand_attack_v1`; relaxing the extractor's accuracy/
// evasion guard then admitted `sandattack`/`smokescreen`/`kinesis`/`flash` to the DATA while the
// hardcoded list still excluded them, so the picker would never have chosen the very moves the
// change was made to unlock. Deriving it removes that failure mode entirely.
const MODELED_STATDROP_MOVES = new Set(
  Object.keys(rustMoves).filter((id) => {
    const sd = rustMoves[id] && rustMoves[id].statDropBoosts;
    return sd && typeof sd === 'object' && Object.keys(sd).length > 0;
  })
);
const MODELED_SCREEN_MOVES = new Set(['lightscreen', 'reflect']);
// BATCH2_E2E_EXCLUDED — whether to keep the batch-2 classes OUT of the e2e capstone's modeled
// allow-list (the phaze-exclusion precedent). The engine models all four classes bit-for-bit
// (`gen_movecoverage_batch2_golden.js` / `movecoverage_batch2_test.rs`, 1360 runs, + the
// MC9-MC17 regression pins), so they're PROVEN. But admitting them to the e2e surfaced ONE
// unresolved real-team-only divergence — e2e_182, a 5-HP Blissey residual-HEAL-ORDERING gap
// on a board where p2 Aromatherapy-cures its own paralysis mid-turn while p1 switches (the
// port reaches full HP ONE residual tick early; a state-only, seed-matching desync the
// dedicated-golden scenarios can't reach). Rather than let a silent desync into the STRICT
// gate, batch 2 is HONESTLY EXCLUDED here (like phaze was, `PHAZE_E2E_EXCLUDED`), keeping the
// pre-batch-2 golden byte-identical. The DEDICATED golden + the MC9-MC17 pins remain the
// batch-2 proof. Re-enable (false) once the e2e_182 residual-order interaction is root-caused.
const BATCH2_E2E_EXCLUDED = false;

// BATCH3_E2E_EXCLUDED — whether to keep the batch-3 classes (CURSE / WISH / BATON PASS) OUT of
// the e2e capstone's modeled allow-list. The engine models all three bit-for-bit (the DEDICATED
// `gen_movecoverage_batch3_golden.js` / `movecoverage_batch3_test.rs`, 16 scenarios × 80 seeds,
// + the MC18-MC29 regression pins), so they're PROVEN. Admitting them (false) grows the
// filter-clean team pool (many gen3ou teams carry Curse / Wish / Baton Pass). Set false =
// ADMITTED to the STRICT e2e gate. (`gen3_move_coverage_batch3_v1`.)
const BATCH3_E2E_EXCLUDED = false;

// PHAZE_E2E_EXCLUDED — the gen-3 phaze moves (Roar / Whirlwind) are now INCLUDED in the e2e
// capstone (flag = false), bit-for-bit, 1035 phaze-DRAG decisions across the 220-battle strict
// gate. ROOT CAUSE of the long-standing "multi-phaze `sample` desync" (FIXED 2026-07-01): it was
// NOT an eligible-list / array-order bug — Roar AND Whirlwind carry the `protect: 1` flag, so a
// Protect/Detect on the TARGET BLOCKS the phaze at `runEvent('TryHit')` (after the accuracy roll)
// → NO `forceSwitchFlag` → NO `dragIn` → NO `sample` draw. The port's phaze arm signalled the drag
// UNCONDITIONALLY (it never checked the Protect block the leechseed/status arms already do), so into
// a protected foe it dragged an EXTRA random mon; the boundary seed still matched (the extra `sample`
// was compensated downstream) while the dragged mon was wrong — which is exactly why it only surfaced
// when a phaze hit a PROTECTING foe across a long history. FIX: a `protect_blocks` check in the phaze
// arm (Substitute does NOT block — Roar/Whirlwind carry `bypasssub: 1`). Pinned by
// `regression_test.rs::phaze_blocked_by_protect_draws_no_sample_and_leaves_the_target` (P4). The
// `destinybond` a phaze-clean team can carry is in `MOVE_ID_BLOCKLIST` (fail-loud, not modeled).
// See EDGE_CASES.md "✅ FIXED — phaze multi-draw-turn `sample` desync".
const PHAZE_E2E_EXCLUDED = false;

// LEECHSEED_E2E_EXCLUDED — whether to keep Leech Seed OUT of the e2e capstone's modeled
// allow-list. DEFAULT FALSE (INCLUDED): the leech draw model (accuracy 90 + the draw-free
// residual at order 10/subOrder 5) is bit-for-bit, the residual is DRAW-FREE (so it can't
// shift the LCG the way the phaze `sample` does), and the HP composition + seed are checked
// per decision — so leech is exercised in the e2e at scale. If a residual-order desync ever
// surfaces at scale that can't be root-caused in budget, set this TRUE for the phaze-style
// HONEST EXCLUSION (keep the dedicated `leechseed_golden.js` / `leechseed_test.rs` + the L1-L3
// regression pins as the proof) and document the unresolved cause in CLAUDE.md + EDGE_CASES.md.
const LEECHSEED_E2E_EXCLUDED = false;

// SUBSTITUTE_E2E_EXCLUDED — keep Substitute OUT of the e2e capstone's modeled allow-list.
// **Now FALSE (INCLUDED, bit-for-bit).** Substitute is FULLY modeled (`gen_substitute_golden.js` /
// `substitute_test.rs`, 9 scenarios × 80 seeds, 4320 decision rows + 5 regression pins) AND the
// substitute mechanic itself is DRAW-COUNT-NEUTRAL (the absorb still draws the secondary random(100);
// the create/block are draw-free). Including it pulled REAL-TEAM battles into scope whose Suicune hit
// a STATEFUL desync the substitute is NOT the cause of — at e2e_84 dec4 (init_seed
// 52903,53571,56373,31187) p1 switches a 213-speed Tyranitar (Sand Stream) in while p2's 213-speed
// Suicune subs; the two actives TIE and the sim drew 8 PRNG calls that turn vs the port's 7. That ONE
// missing draw is the **`eachEvent('WeatherChange')` switch-in tie-shuffle** (`Field.setWeather`,
// field.ts:87): a mid-turn weather-setting switch-in into a speed tie draws one `random(0,2)` the port
// MISSED — the SAME hard class as `forced_replacement_recaches_speed_seed` /
// `para_while_active_keeps_full_cached_speed_seed`, in the SWITCHING/weather layer (NOT the substitute
// arm). **FIXED 2026-06-30** in `turn.rs` (`run_switch` reports a weather change → `turn_loop` fires the
// shuffle), pinned by `regression_test.rs::switch_into_a_tie_under_sand_draws_the_weather_change_shuffle_seed`
// (ground truth from `harness/probe_switch_tie_weather_regression_rng.js`; the +1 shuffle is shown by the
// `probe_switch_sand.js` control). With the fix, e2e_84 + all 220 filtered battles (284 substitute-MOVE /
// 320 sub-up decisions) are bit-for-bit (`filtered_diverged == 0`), so Substitute is INCLUDED here, with a
// `substitute_decisions >= 50` coverage floor in `e2e_fuzz_test.rs`. Documented in CLAUDE.md + EDGE_CASES.md.
const SUBSTITUTE_E2E_EXCLUDED = false;

// EXPLOSION_E2E_EXCLUDED — keep Explosion / Self-Destruct OUT of the e2e capstone's modeled
// allow-list. **Now TRUE (EXCLUDED) — the HONEST phaze-style exclusion.** The Explosion / Self-
// Destruct SELF-KO is itself FULLY modeled bit-for-bit (`gen_explosion_golden.js` /
// `explosion_test.rs`, 7 scenarios × 80 seeds, 3688 decision rows, 7376 FAINTED assertions, 544
// self-KO rows; + the E1-E4 `regression_test.rs` pins for user-faints-through-Protect /
// -immunity / -a-sub-break / the mutual double-faint TIE). The gen-3 self-KO faints the USER as
// part of `useMoveInner` (battle-actions.ts:501-503) BEFORE the hit resolves, is UNCONDITIONAL
// (the user faints THROUGH a Protect / a Ghost immunity / a sub / a miss) and DRAW-FREE (only
// the normal acc/crit/dmg draws fire; no trailing Quick Claw on a deciding faint).
//
// WHY EXCLUDED HERE (an HONEST exclusion — the self-KO is NOT the cause): admitting Explosion
// pulled REAL-TEAM battles into scope that hit a STATEFUL desync in a DIFFERENT layer — a
// DOUBLE-FAINT → DOUBLE-REPLACEMENT → SPIKES-CASCADE. When a last-mon-both Explosion (or any
// double faint) forces BOTH sides to replace AND one side's replacement itself faints on Spikes
// (chaining a THIRD replacement), the port mis-applies the ENTRY-HAZARD (Spikes) chip to the
// OTHER side's fresh entrant that the sim does NOT chip (e2e_9 dec43: p2's Jirachi entrant —
// full 403 in the sim — is wrongly chipped maxhp/8=50 → 353 by the port's `run_switch`
// EntryHazard during the cascading double-replacement; likewise e2e_194 dec15). The SEED stays
// bit-for-bit throughout (seed_ok=true) — it is a STATE (HP) mis-application in the
// entry-hazard × cascade-replacement machinery (`run_switch`/`execute_switch`), NOT the
// Explosion self-KO (which the 218 OTHER clean battles + the dedicated golden + the 4 regression
// pins prove exact). Explosion is merely the most common way the fuzz produces the triggering
// DOUBLE FAINT; the bug is orthogonal to it (a plain mutual-recoil / residual double-KO into the
// same Spikes cascade would desync identically). Rather than risk a broad surgery on the
// double-replacement/hazard ordering (which could regress the 218 clean battles + the spikes /
// phaze tests) inside this layer's budget, we EXCLUDE Explosion from the e2e — keeping the gate
// STRICT (no silent desync) — with the dedicated golden + the E1-E4 pins as the bit-for-bit
// proof. Documented in CLAUDE.md + EDGE_CASES.md; the cascade bug is filed there as the next
// entry-hazard-layer fix. Set false to re-include once the double-faint-cascade Spikes chip is
// fixed + this yields a clean strict pass.
const EXPLOSION_E2E_EXCLUDED = false;

// Hidden Power has 16 typed variants whose id is `hiddenpower<type>`.
function isHiddenPower(id) { return id === 'hiddenpower' || id.startsWith('hiddenpower'); }

// Does the Rust port model this move's secondary shape? (≤1 secondary col, or Tri
// Attack.) The structured boost moves collapse to a single synthetic col.
function rustSecondaryOk(id) {
  if (id === 'triattack') return true; // special-cased in the port
  const e = rustMoves[id];
  if (!e) return false;
  const sec = e.secondaryEffects || {};
  return Object.keys(sec).length <= 1;
}

// `allowHiddenPower` widens the picker to admit typed Hidden Power (which the engine
// models bit-for-bit as an ORDINARY typed damaging move, num 355-370, real BP/type —
// scan_move_coverage.js:8-13, scan_move_probe.rs). The engine models HP at a FIXED BP 70,
// which is byte-correct ONLY for the 70-BP-IV mons every gen3ou-VALIDATED team carries
// (pool / e2e). So HP is admitted ONLY when the caller vouches for BP-70 teams
// (`allowHiddenPower` — the byte fuzzer's `pool` mode); the `random` mode's RANDOM IVs
// could yield a non-70 HP BP, so it leaves HP excluded (default false). The committed e2e
// golden is produced with the default (false) so it stays byte-reproducible.
function isModeledMove(id, allowHiddenPower = false) {
  if (isHiddenPower(id)) return allowHiddenPower;
  if (MOVE_ID_BLOCKLIST.has(id)) return false;
  const m = dex3.moves.get(id);
  if (!m || !m.exists) return false;
  // FIXED-DAMAGE moves (Seismic Toss / Night Shade / Sonic Boom / Dragon Rage / Super Fang) —
  // ADMITTED HERE, before the damaging-move gates below reject them. In the Showdown dex they
  // are category Physical/Special with `basePower:0` + a `damage`/`damageCallback` field, so
  // they'd else be dropped by the `basePower > 0` / `m.damage` / `m.damageCallback` rejects.
  // The port models them bit-for-bit (`gen_fixeddamage_golden.js` / `fixeddamage_test.rs` +
  // the FD1-FD4 regression pins); the DEFERRED fixed-damage family stays out (fail-loud).
  if (MODELED_FIXED_DAMAGE_MOVES.has(id)) return true;
  // MOVE-COVERAGE BATCH 4 (`gen3_move_coverage_batch4_v1`) — FOCUS PUNCH + PURSUIT, ADMITTED
  // HERE (before the damaging-move `beforeTurnCallback` / `basePowerCallback` / `onModifyMove`
  // rejects below, which would else drop them). Both are damaging (bp>0) with no secondary, so
  // the early return is safe. Kept in lockstep with `move_has_before_turn_callback` in src/turn.rs.
  if (!BATCH4_E2E_EXCLUDED && MODELED_BATCH4_MOVES.has(id)) return true;
  // MOVE-COVERAGE BATCH 4b (`gen3_move_coverage_batch4b_v1`) — BEAT UP / THUNDER / WATER SPOUT,
  // ADMITTED HERE (before the `m.multihit` / `basePowerCallback` / `onModifyMove` rejects below,
  // which would else drop them). All three are damaging (bp>0); Beat Up carries the only secondary-
  // free multi-strike, Thunder's 30% para is the ordinary modeled secondary shape. Kept in lockstep
  // with `run_beat_up` / the waterspout+thunder id-gates in src/turn.rs.
  if (!BATCH4B_E2E_EXCLUDED && MODELED_BATCH4B_MOVES.has(id)) return true;
  // MOVE-COVERAGE BATCH 4c (`gen3_move_coverage_batch4c_v1`) — HYPER BEAM / SOLAR BEAM /
  // DOOM DESIRE / FUTURE SIGHT (+ the locked-turn `recharge` pseudo-move), ADMITTED HERE
  // (before the `flags.charge|recharge` / `self.volatileStatus` / futuremove rejects
  // below, which would else drop them). Kept in lockstep with the turn.rs id-gates.
  if (!BATCH4C_E2E_EXCLUDED && MODELED_BATCH4C_MOVES.has(id)) return true;
  // MOVE-COVERAGE BATCH 5 (`gen3_move_coverage_batch5_v1`) — the REACTIVE family (a
  // `damageCallback` + `beforeTurnCallback` shape) + the VARIABLE-BP family (a
  // `basePowerCallback` over a bp-0 data row), ADMITTED HERE (before the `basePower > 0`
  // / callback rejects below, which would else drop them). SLEEP TALK is NOT admitted
  // here — its safety is CARRIER-conditional (the sampled pool must be all-modeled), so
  // `pickMove` gates it via `sleepTalkPoolModeled`.
  if (!BATCH5_E2E_EXCLUDED
      && (MODELED_BATCH5_REACTIVE_MOVES.has(id) || MODELED_BATCH5_VARBP_MOVES.has(id))) {
    return true;
  }
  // STATUS MOVES: allow ONLY the modeled standalone status-inflicting moves (accuracy
  // + apply + sleep random(2,6)), the modeled pure SELF-BOOST setup moves (never-miss →
  // no accuracy draw, draw-free boost apply, no in-tryMoveHit Update), AND the modeled
  // self-targeting HP-RECOVERY moves (never-miss → no accuracy draw, draw-free heal; Rest
  // draws-then-discards one slp.onStart random(2,6)) — the port executes all three bit-
  // for-bit; every OTHER Status move (Wish/Heal Bell/phaze/hazard/Substitute/field/
  // Defense-Curl-volatile) stays excluded → the port FAIL-LOUDs.
  // TRANSFORM (`gen3_transform_v1`, ROUND 33) — ADMITTED HERE, before the STATUS-move
  // gate below (which allows only the enumerated standalone-status / setup / utility sets,
  // and would else drop it). Transform is `accuracy: true` (never-miss ⇒ NO accuracy draw)
  // and `transformInto` contains no `this.random` anywhere, so the whole move is DRAW-FREE
  // in every branch — success, the already-transformed fail, and the fainted-target fail
  // alike (probe `harness/probe_batch89_transform.js` vs a Splash control). Modeled in
  // `run_status_move`'s transform arm + the `clearVolatile` reverts, pinned by TF1-TF7.
  // Admitting it is what finally EXERCISES the mechanic in the byte fuzz — the whole point
  // of the round, since the picker filter is what hid the original silent no-op.
  if (id === 'transform') return true;
  if (m.category === 'Status') {
    // PHAZE (Roar / Whirlwind) is special-cased HERE — a category-Status `forceSwitch`
    // move, gated on PHAZE_E2E_EXCLUDED below. It is INCLUDED (flag = false) since the
    // Protect-blocks-phaze fix (see the PHAZE_E2E_EXCLUDED comment above) — so a
    // Roar-carrying RestTalker's Sleep-Talk pool passes `sleepTalkPoolModeled` too
    // (the CALLED Roar rides the port's force_switch_foe drag; pinned by
    // `regression_test.rs::sleep_talk_called_roar_drags_the_foe`).
    return MODELED_STATUS_MOVES.has(id) || MODELED_SETUP_MOVES.has(id) ||
      MODELED_RECOVERY_MOVES.has(id) || MODELED_PROTECT_MOVES.has(id) ||
      MODELED_HAZARD_MOVES.has(id) || MODELED_RESTRICTION_MOVES.has(id) ||
      MODELED_RECYCLE_MOVES.has(id) ||
      MODELED_SKILLSWAP_MOVES.has(id) || MODELED_FAKEOUT_MOVES.has(id) ||
      // MOVE-COVERAGE BATCH 2 (`gen3_move_coverage_batch2_v1`) — the cure / weather-set /
      // stat-drop / screen classes, all category-Status + bit-for-bit modeled. INCLUDED
      // (BATCH2_E2E_EXCLUDED = false) since the e2e_182 Pressure×allyTeam PP-deduction fix
      // (`gen3_pressure_allyteam_v1`); the DEDICATED golden + MC9-MC17 pins remain the proof.
      (BATCH2_E2E_EXCLUDED ? false : (MODELED_CURE_MOVES.has(id) || MODELED_WEATHER_MOVES.has(id) ||
        MODELED_STATDROP_MOVES.has(id) || MODELED_SCREEN_MOVES.has(id))) ||
      (LEECHSEED_E2E_EXCLUDED ? false : MODELED_LEECH_MOVES.has(id)) ||
      (SUBSTITUTE_E2E_EXCLUDED ? false : MODELED_SUBSTITUTE_MOVES.has(id)) ||
      // MOVE-COVERAGE BATCH 3 (`gen3_move_coverage_batch3_v1`) — CURSE / WISH / BATON PASS,
      // all category-Status + bit-for-bit modeled (the DEDICATED golden + MC18-MC29 pins).
      (BATCH3_E2E_EXCLUDED ? false : MODELED_BATCH3_MOVES.has(id)) ||
      // MOVE-COVERAGE BATCH 6 (`gen3_move_coverage_batch6_v1`) — the final tail, all
      // category-Status + bit-for-bit modeled (the DEDICATED golden + MC79+ pins).
      (BATCH6_E2E_EXCLUDED ? false : MODELED_BATCH6_MOVES.has(id)) ||
      // SNATCH (`gen3_snatch_v1`) — the LAST gen-3 status move. Its interception steals only
      // moves the picker also picks (all `isModeledMove`), so the steal always re-dispatches
      // a modeled arm (never a fail-loud). Closes 722/722.
      (SNATCH_E2E_EXCLUDED ? false : MODELED_SNATCH_MOVES.has(id)) ||
      // HAZE (`gen3_haze_v1`) — the boost-reset FIELD move, category Status + bit-for-bit modeled.
      (HAZE_E2E_EXCLUDED ? false : MODELED_HAZE_MOVES.has(id)) ||
      // YAWN (`gen3_yawn_v1`) — the delayed-sleep move, category Status + bit-for-bit modeled.
      (YAWN_E2E_EXCLUDED ? false : MODELED_YAWN_MOVES.has(id)) ||
      MODELED_SAFEGUARD_MOVES.has(id) ||
      // TRICK (`gen3_trick_v1`) — the item-SWAP move, category Status + bit-for-bit modeled.
      (TRICK_E2E_EXCLUDED ? false : MODELED_TRICK_MOVES.has(id)) ||
      (PHAZE_E2E_EXCLUDED ? false : MODELED_PHAZE_MOVES.has(id));
  }
  if (!(m.basePower > 0)) return false; // variable / fixed-damage carrier
  if (m.ohko) return false;
  // MOVE-COVERAGE BATCH 7 (`gen3_move_coverage_batch7_v1`) — the GENERIC MULTI-STRIKE family,
  // ADMITTED HERE (before the `m.multihit` reject), the port runs the per-strike loop bit-for-bit
  // (`run_multihit`). Triple Kick (multiaccuracy) is NOT in the set → still rejected by `m.multihit`.
  if (!BATCH7_E2E_EXCLUDED && MODELED_BATCH7_MULTIHIT_MOVES.has(id)) return true;
  if (m.multihit) return false;
  // RECOIL / DRAIN (`gen3_move_coverage_batch1_v1`) — a recoil/drain damaging move is admitted
  // ONLY if it is in the explicit modeled set (a recoil/drain move with an EXTRA unmodeled
  // mechanic — Dream Eater's sleep-only `onTryImmunity` — stays out). Otherwise reject.
  if (m.recoil && !MODELED_RECOIL_MOVES.has(id)) return false;
  if (m.drain && !MODELED_DRAIN_MOVES.has(id)) return false;
  // ITEM REMOVAL (Knock Off / Thief / Covet) + RAPID SPIN — a damaging `onAfterHit` move is
  // admitted ONLY if it is in the modeled item-removal / rapid-spin sets; every OTHER onAfterHit
  // damaging move (a future unmodeled mechanic) is rejected. (Brick Break's screen-break onHit /
  // Pay Day's coin onHit are draw-free and kept — they are NOT onAfterHit.)
  if (m.onAfterHit && !MODELED_ITEM_REMOVAL_MOVES.has(id) && !MODELED_RAPIDSPIN_MOVES.has(id)) {
    return false;
  }
  // SELF-DESTRUCT class (Explosion / Self-Destruct) — a Normal PHYSICAL damaging move that
  // faints the USER as part of the move (gen-3 self-KO, `useMoveInner`:501-503). FULLY modeled
  // bit-for-bit; ADMITTED unless re-excluded. It still must clear the remaining damaging-move
  // gates below (secondary shape etc.) — but Explosion / Self-Destruct have NO secondary, so
  // they pass. (When EXCLUDED, drop it here to keep the fuzz free of the self-KO mechanic.)
  if (m.selfdestruct && EXPLOSION_E2E_EXCLUDED) return false;
  if (m.forceSwitch) return false;
  if (m.damage) return false; // fixed/level damage
  // SIDE/SLOT-CONDITION moves are unmodeled — EXCEPT the modeled gen-3 entry hazard
  // **Spikes** (a category-Status move, already allowed in the Status branch above; this
  // belt-and-braces special-case keeps the intent explicit should a hazard ever surface
  // on the non-Status path). Every other sideCondition (Reflect/Light Screen/Toxic Spikes/
  // Stealth Rock/Safeguard/Tailwind/…) stays excluded.
  if ((m.sideCondition || m.slotCondition || m.sideConditions) && !MODELED_HAZARD_MOVES.has(id)) {
    return false;
  }
  if (m.flags && (m.flags.charge || m.flags.recharge)) return false;
  // FUTURE MOVES (`flags.futuremove`) beyond the modeled DD/FS (none exist in gen3) —
  // belt-and-braces: a futuremove not admitted above can never slip in as a plain
  // damaging move (the pre-batch-4c silent-desync class).
  if (m.flags && m.flags.futuremove) return false;
  // TOP-LEVEL move.self.boosts (SELF-DROP — Overheat/Superpower) / self.volatileStatus
  // (lockedmove — Outrage/Thrash/Petal Dance). The self-DROP is now MODELED
  // (`gen3_move_coverage_batch1_v1`): the port applies the drop AND draws the gen3 `selfDrops`
  // random(100) — admitted via MODELED_SELFDROP_MOVES. A self.volatileStatus (a locked-move) is
  // still UNMODELED → rejected. (A self.boosts move NOT in the modeled set — none in gen-3 OU —
  // stays out.)
  if (m.self && m.self.volatileStatus) return false;
  if (m.self && m.self.boosts && !MODELED_SELFDROP_MOVES.has(id)) return false;
  // PARTIAL TRAP (`gen3_partial_trap_v1`) — ADMITTED HERE, before the blanket `m.volatileStatus`
  // reject below. All six are ordinary damaging moves (bp > 0, no secondary, no callback) plus the
  // one `random(3,7)` duration draw; kept in lockstep with `is_partial_trap_move` in
  // src/turn/helpers.rs.
  if (!PARTIALTRAP_E2E_EXCLUDED && MODELED_PARTIALTRAP_MOVES.has(id)) return true;
  if (m.volatileStatus) return false; // a volatile MOVE (substitute etc.) — not damaging here
  // DRAW-ORDER / POWER callbacks the port does NOT model (each desyncs the LCG):
  //   * basePowerCallback — variable BP (Fury Cutter / Rollout / Ice Ball / Smelling
  //     Salts / Revenge); the dex BP is a placeholder.
  //   * beforeTurnCallback — FOCUS PUNCH: queues a beforeTurn `|-singleturn|` action
  //     + an `onTry` cant-if-hit gate the port doesn't run (a real seed desync).
  //   * beforeMoveCallback / priorityChargeCallback — other pre-move queue effects.
  //   * onModifyMove — accuracy/power mutation (THUNDER never-misses in rain → a
  //     DIFFERENT accuracy draw; Secret Power's terrain effect).
  //   * damageCallback — fixed/derived damage (covered by `m.damage`/blocklist, but
  //     belt-and-braces).
  // (Plain `priority` is FINE — the port reads move.priority for action order; and a
  // draw-free `onTry`/`onHit` like Brick Break screen-break / Pay Day coins is kept.)
  if (m.basePowerCallback) return false;
  if (m.beforeTurnCallback) return false;
  if (m.beforeMoveCallback) return false;
  if (m.priorityChargeCallback) return false;
  if (m.damageCallback) return false;
  if (m.onModifyMove) return false;
  // Secondary shape: 0 or 1 secondary, modeled cols only.
  const secs = m.secondaries || (m.secondary ? [m.secondary] : []);
  if (secs.length > 1 && id !== 'triattack') return false;
  for (const s of secs) {
    if (!s) continue;
    // self stat-RAISE (Meteor Mash/Ancient Power) — modeled via secondaryBoosts.
    // foe stat-DROP (Crunch/Psychic/Shadow Ball) — modeled.
    // a single status / flinch / confusion — modeled. Tri Attack — special-cased.
    const ok =
      id === 'triattack' ||
      (typeof s.status === 'string') ||
      (s.volatileStatus === 'flinch') ||
      (s.volatileStatus === 'confusion') ||
      (s.boosts && typeof s.boosts === 'object') ||
      (s.self && s.self.boosts && typeof s.self.boosts === 'object');
    if (!ok) return false;
  }
  // The port must also be able to ingest the secondary col-count.
  if (!rustSecondaryOk(id)) return false;
  return true;
}

// ── Modeled ability / item predicates (the FILTERED gate's pre-filter) ───────
// The FILTERED golden only includes battles where EVERY mon (both teams) has an
// ability that is EITHER modeled OR a provable no-op in a damaging-move-only fuzz,
// and an item in the modeled set. The taxonomy does NOT apply these.

const MODELED_ABILITIES = new Set([
  'intimidate', 'sandstream', 'drizzle', 'drought', 'levitate', 'flashfire',
  'waterabsorb', 'voltabsorb', 'thickfat', 'clearbody', 'whitesmoke',
  'hypercutter', 'keeneye', 'serenegrace', 'shielddust', 'owntempo',
  // TRAPPING (`gen3_trapping_v1`): Arena Trap (grounded foes) + Magnet Pull (Steel
  // foes) are MODELED — the port computes `is_trapped` (the switch-legality gate) and
  // the endTurn TrapPokemon/MaybeTrapPokemon tie-shuffle draws (the gen3 magnetpull
  // onAny 2-handler tie). The generator's voluntary-switch picker respects the sim's
  // `pokemon.trapped` (below), so a trapped mon always fights — mirroring the real
  // request, where the trapped mon's switches are rejected.
  'arenatrap', 'magnetpull',
  // DMG_MOD class (`gen3_item_mechanics_v1` ability side, Phase 2): the DATA-DRIVEN
  // ability damage folds (`AbilityData.dmg_mod` → resolve_atk_stat_mods /
  // resolve_def_stat_mods / resolve_bp_mods) are WIRED + validated by the dedicated
  // class-sweep golden `gen_ability_dmgmod_golden.js` → `tests/ability_dmgmod_test.rs`
  // (330 game-end battles, byte-for-byte) + the damage golden's 15 exact-roll probes +
  // the AB1-AB5 pins. The pinch family (Torrent/Blaze/Overgrow/Swarm: BP ×1.5 at hp≤⅓)
  // + Huge/Pure Power (Atk ×2) + Guts (Atk ×1.5 statused + burn-halve suppressed) +
  // Marvel Scale (Def ×1.5 while the DEFENDER is statused). These are the top
  // team-carry admission gaps (torrent=254, blaze=103, guts=50, marvelscale=35).
  //
  // ADMITTED (`gen3_move_alias_resolution_v1`): the 8 below grow the filter-clean pool to
  // 151/719 (the biggest single admission lever — the DMG_MOD gaps VANISH from the taxonomy's
  // top list). The admission was gated on ONE newly-admitted battle (e2e_86, a Swampert battle)
  // that diverged because its Gengar's packed team spells Will-O-Wisp as the ALIAS `wisp` — the
  // port's dex read only canonical ids, so the move NO-OP'd (drawing nothing while the sim ran it),
  // a draw-COUNT desync (rust 35 vs golden 41 decisions). FIXED by move-alias resolution in the dex
  // (`gen3_move_aliases.json`, §8 in EDGE_CASES). The enlarged corpus is now a clean STRICT pass.
  // (SEPARATE fix, same window: `gen3_sun_freeze_immunity_v1` — the base `sunnyday` `onImmunity('frz')`
  // blocks a freeze while the field is Sun; the port used to freeze anyway [the A/B "ice-freeze
  // cluster", 196 repros]. That is pinned by `sun_blocks_freeze_secondary_draw_free` with 0 e2e
  // decisions — no filter-clean battle has a sun+ice-move turn, so it did NOT gate this admission.)
  'torrent', 'blaze', 'overgrow', 'swarm', 'hugepower', 'purepower', 'guts', 'marvelscale',
  // ACCURACY class (`gen3_accuracy_pipeline_v1`): the DATA-DRIVEN to-hit folds
  // (`AbilityData.acc_mod` → `turn.rs::effective_accuracy`, DRAW-RELEVANT — a wrong effAcc
  // flips a hit/miss and desyncs the seed). Compound Eyes (×1.3 attacker chain), Sand Veil
  // (×0.8 defender chain in sand + its `onImmunity('sandstorm')` sand-chip immunity), Hustle
  // (acc ×0.8 on physical-type moves + its Atk ×1.5 dmgMod — now BOTH wired). Validated by
  // the class-sweep golden `gen_accuracy_golden.js` → `tests/accuracy_test.rs` (per-decision
  // STATE+HP+SEED to game-end) + the AC1-AC4 pins. Hustle is off the DATA-ONLY/deferred list.
  'compoundeyes', 'sandveil', 'hustle',
  // SWITCH_OUT class (`gen3_natural_cure_v1`): NATURAL CURE — the sole gen-3 switch-out-cure
  // ability, the #1 e2e team-carry gap (naturalcure=254, on Blissey/Starmie/Celebi/Miltank/…).
  // The holder's major status is CURED when it switches OUT (voluntary pivot OR phaze-DRAG-out),
  // DRAW-FREE (probe-settled `harness/probe_naturalcure_rng.js`: `onSwitchOut`, `onCheckShow`
  // undefined; the cure + its `[silent]` `-curestatus` reveal consume ZERO PRNG — SEED-NEUTRAL).
  // MODELED in `turn.rs::execute_switch` (status = None on an alive outgoing naturalcure holder),
  // validated by the class-sweep golden `gen_naturalcure_golden.js` → `tests/naturalcure_test.rs`
  // (280 game-end battles, per-decision STATE+STATUS+SEED, cure observable on the active-status
  // timeline) + the NC1-NC3 pins. This is the single biggest e2e-admission lever after the
  // DMG_MOD family (real gen3OU teams are saturated with Natural Cure carriers).
  'naturalcure',
  // STATUS_IMMUNE class (`gen3_status_immune_v1`): the DATA-DRIVEN status-immunity abilities —
  // Limber (par) / Insomnia + Vital Spirit (slp) / Immunity (psn,tox) / Water Veil (brn) block
  // via `onSetStatus`; Magma Armor (frz) blocks via `onImmunity` (BEFORE the SetStatus event).
  // Read from `AbilityData.status_immune` in `turn.rs::try_set_status`; PROBE-settled draw model
  // (`harness/probe_statusimmune_*.js`): DRAW-FREE in gen3customgame (the ability is the only
  // SetStatus handler → no shuffle; Magma Armor blocks before the event) — so admission is
  // SEED-CLEAN. `immunity` (=97) is the #2 team-carry gap after Natural Cure. Validated by the
  // class-sweep golden `gen_statusimmune_golden.js` → `tests/statusimmune_test.rs` (480 game-end
  // battles, the block observable on the active-status timeline) + the SI1-SI4 pins.
  // NOTE `insomnia`/`vitalspirit` MOVED here from NOOP_ABILITIES — they are genuinely MODELED
  // now (they block sleep, a modeled status move), not provable no-ops.
  'limber', 'insomnia', 'vitalspirit', 'immunity', 'waterveil', 'magmaarmor',
  // BATCH-1 DRAW-FREE / STRUCTURAL classes (`gen3_ability_batch1_v1`): four ability classes
  // WIRED + validated by the class-sweep golden `gen_ability_batch1_golden.js` →
  // `tests/ability_batch1_test.rs` (300 game-end battles, per-decision STATE+HP+SPE-BOOST+SEED,
  // byte-for-byte) + the B1-B4b pins in `regression_test.rs`:
  //   CRIT_IMMUNE  (shellarmor / battlearmor) — a hit into the holder NEVER crits (the crit
  //     roll is DRAWN then overridden false → DRAW-FREE); the seed is unchanged.
  //   WEATHER_SPEED (chlorophyll / swiftswim) — ×2 effective speed in sun / rain, folded into the
  //     cached speed the tie-shuffles read (the ×2 is modeled → the tie-shuffle order/count is
  //     seed-faithful).
  //   WEATHER_NEGATE (cloudnine / airlock) — suppresses the weather's EFFECTS (chip + speed ×2);
  //     the sun/rain eachEvent('Weather') end-of-turn shuffle is fired off the RAW weather (the
  //     STEP-1 fix `sun_rain_weather_turn_tie_draws_the_eachevent_weather_shuffle_seed`).
  //   RESIDUAL (speedboost / raindish) — Speed Boost +1 spe/active-turn + Rain Dish +maxhp/16 in
  //     rain, at residualOrder 10 subOrder 3, DRAW-FREE.
  // All four are DRAW-FREE, so admitting them is SEED-CLEAN. shellarmor is the big lever
  // (a common gen3OU defensive ability on Cloyster/Cradily/etc.).
  'shellarmor', 'battlearmor', 'chlorophyll', 'swiftswim',
  'cloudnine', 'airlock', 'speedboost', 'raindish',
  // BATCH-2 DRAW-BEARING "reactive" classes + block tail (`gen3_ability_batch2_v1`): WIRED +
  // validated by the class-sweep golden `gen_ability_batch2_golden.js` → `tests/ability_batch2_test.rs`
  // (960 game-end battles, per-decision STATE+HP+STATUS+SEED, byte-for-byte) + the B2-1..B2-7 pins
  // in `regression_test.rs`. PROBE-settled draw models (`harness/probe_contact_proc_{rng,lands}.js`,
  // `probe_effectspore_sample.js`, `probe_block_abilities_rng.js`, `probe_synchronize_rng.js`):
  //   CONTACT_PROC (static par / poisonpoint psn / flamebody brn / effectspore slp|par|psn) — an
  //     onDamagingHit that, when the holder is hit by a CONTACT move, draws `randomChance(chance)`
  //     (1/3, or Effect Spore's 1/10 + a sample(3)) and statuses the ATTACKER. The proc's
  //     randomChance draws INSIDE runEvent('DamagingHit') (gen<5) — AFTER the move's own secondary
  //     random(100). It draws even behind a Substitute + on a KO. `synchronize` is the #1 taxonomy
  //     gap, `effectspore` (=9) the #2.
  //   CONTACT recoil (roughskin) — a DRAW-FREE baseMaxhp/16 recoil to the attacker.
  //   BLOCK — soundproof (immune to a sound move: Sing / Grass Whistle / Roar — accuracy drawn then
  //     -immune, no status/drag/sample), damp (Explosion CANCELLED at TryMove — the user does NOT
  //     self-KO, the move draws NOTHING), suctioncups (a phaze into the holder draws no `sample` —
  //     the holder stays active).
  //   SYNCHRONIZE (synchronize) — reflect a foe-inflicted major status back to the SOURCE (slp/frz
  //     exempt; tox→psn), DRAW-FREE in gen3customgame (the e2e format — the reflected status draws
  //     no clause shuffle in customgame). Threaded through `turn.rs::try_set_status` (the single
  //     status choke point, source-aware).
  // The contact-proc `randomChance` / Effect Spore `sample` are the ONLY new draws (the rest are
  // draw-free-or-fewer), so admitting these exercises the draw-bearing path on real teams.
  'static', 'poisonpoint', 'flamebody', 'effectspore', 'roughskin',
  'soundproof', 'damp', 'suctioncups', 'synchronize',
  // BATCH-3 (`gen3_berry_trace_shedskin_v1`): TRACE + SHED SKIN — WIRED + validated by the
  // class-sweep golden `gen_berry_batch3_golden.js` → `tests/berry_batch3_test.rs` (1280
  // battles, per-decision STATE+STATUS+ITEM+BOOSTS+SEED) + the BR4/BR5 pins. PROBE-settled
  // draw models (`harness/probe_trace_shedskin_rng.js`):
  //   TRACE — the gen3-resolved onStart: ONE n=1 `randomFoe` sample draw (`random(1)` even
  //     for a single foe) + a LIVE copy of the foe's CURRENT ability (no copied-onStart in
  //     gen3; switch-out reverts; a lead trace's draw pre-dates the seeded start). SAFETY:
  //     the engine FAIL-LOUDS (`event.rs::TRACE_COPYABLE`, kept in LOCKSTEP with
  //     MODELED_ABILITIES ∪ NOOP_ABILITIES) on an unmodeled copy — and this filter requires
  //     BOTH teams all-modeled, so a filtered battle can never trip it.
  //   SHED SKIN — ONE `randomChance(33,100)` per STATUSED residual (order 10 subOrder 3,
  //     cure BEFORE the status DoT; the handler is gathered unconditionally for the
  //     tie-shuffle).
  'trace', 'shedskin',
  // BATCH-4 — the FINAL mechanics tail (`gen3_ability_batch4_v1`): WIRED + validated by the
  // class-sweep golden `gen_ability_batch4_golden.js` → `tests/ability_batch4_test.rs` (1260
  // game-end battles, per-decision STATE+HP+STATUS+TRAPPED+SEED, byte-for-byte) + the
  // B4-1..B4-7 pins in `regression_test.rs`. PROBE-settled draw models
  // (`harness/probe_{truant,truant_edges,innerfocus,shadowtag,cutecharm_attract,colorchange}_rng.js`):
  //   TRUANT (=4, the last ability team-carry gap) — onBeforeMove priority 9 cant iff
  //     `truantTurn` (DRAW-FREE: a loaf turn draws NOTHING, no para roll, no PP); onSwitchIn
  //     arms `turn !== 0`; the order-27 residual toggles (a Truant MIRROR tie adds ONE
  //     residual shuffle draw).
  //   INNER FOCUS (=2) — blocks the flinch volatile at the APPLY (the secondary random(100)
  //     STILL draws — draw-count-identical to a landed flinch; contrast Shield Dust's
  //     filter-the-draw).
  //   SHADOW TAG (0 sample teams — Wobbuffet is banned from gen3ou) — `onFoeTrapPokemon`
  //     traps UNCONDITIONALLY (no grounded/type gate; a MIRROR is mutually trapped),
  //     DRAW-FREE (0 extra draws — vs Magnet Pull's onAny* draws).
  //   CUTE CHARM (0 teams) + the ATTRACT volatile — a damaging CONTACT hit draws
  //     randomChance(1,3) UNCONDITIONALLY (the gender gate lives inside attract.onStart,
  //     draw-free fail); attract: onBeforeMove priority 2, `-activate` always then
  //     randomChance(1,2); cleared when the SOURCE leaves / the holder switches out.
  //   COLOR CHANGE (0 teams, Kecleon-only) — an onDamagingHit TYPE OVERRIDE
  //     (`MonState::types_override` through the ONE `mon_types` choke point: STAB/chart/
  //     status-immunity/sand-immunity), DRAW-FREE; NOT behind a sub; not on the KO hit;
  //     switch-out reverts.
  // STILL DEFERRED: forecast (a Castform forme+TYPE change under rain/sun/hail — 0 sample
  // teams; the forme-change reporting surface + the Cloud-Nine/effective-weather composition
  // are unprobed; the filter keeps every Castform-Forecast team off the modeled path).
  'truant', 'innerfocus', 'shadowtag', 'cutecharm', 'colorchange',
  // PLUS / MINUS (`gen3_plus_minus_v1`, 2026-07-10 — moved from NOOP_ABILITIES): the gen3
  // resolved `onModifySpA` scans `getAllActive()` FOES INCLUDED, so a cross-field
  // Plus↔Minus pair gives the holder SpA ×1.5 (paired ability only; special-only;
  // draw-free). The port models it as a ModifySpA chain member; probe
  // `harness/probe_plus_minus_gen3.js`; pin `minus_boosts_spa_when_the_foe_active_has_plus`.
  'plus', 'minus',
  // LIQUID OOZE (`gen3_liquid_ooze_v1`) — the drain/leech-seed heal REVERSAL, now MODELED
  // bit-for-bit (`apply_drain` / `apply_leech_seed`): a Giga/Mega/Absorb/Leech-Life drain OR a
  // Leech Seed residual into a Liquid Ooze mon turns the would-be heal into DAMAGE on the
  // healer (`|-damage|<healer>|<HP>|[from] ability: Liquid Ooze|[of] <ooze-mon>`, can KO). DRAW-FREE
  // (probe-settled). Dream Eater is EXCLUDED (not a modeled drain move). The DEDICATED golden +
  // the LO1-LO3 pins are the proof; 0 team-carry, so admitting it is byte-neutral on the e2e sample.
  'liquidooze',
  // WONDER GUARD (`gen3_wonder_guard_v1`) — Shedinja's SE-only damage gate, now MODELED bit-for-bit
  // (`turn.rs::run_move`): a DAMAGING move into a Wonder Guard holder CONNECTS only if STRICTLY
  // super-effective (`runEffectiveness > 0`) AND not type-immune; every neutral / resisted /
  // 0×-immune move is BLOCKED (`-immune [from] ability: Wonder Guard`) drawing ONLY its accuracy
  // roll. Status moves + residuals BYPASS it. The DEDICATED golden (`wonderguard_test.rs`) + the
  // WG1-WG4 pins are the proof; 0 sample teams carry Shedinja, so admitting it is byte-neutral on
  // the e2e sample (the committed golden md5 is unchanged — no filter-clean team gains Shedinja).
  'wonderguard',
  // FORECAST (`gen3_forecast_v1`, ROUND 35) — Castform's forme + TYPE swap under the
  // EFFECTIVE weather, now MODELED bit-for-bit (`turn/forecast.rs`: the WeatherChange sites
  // incl. the previously-missing UNCONDITIONAL expiry draw, the entrant onStart, the start
  // window, the silent clearVolatile revert; details/ident/request stay the BASE species).
  // The LAST fail-loud ability — its admission clears REJECT_ABILITIES/REJECT_SPECIES.
  // 0 pool teams carry Castform, so the committed e2e golden is untouched.
  'forecast',
]);
// Provably no-op in a damaging-move-only, no-PP, no-attract, no-sleep, no-OHKO,
// no-recoil, no-drain fuzz.
//
// **LIQUID OOZE moved to MODELED_ABILITIES** (`gen3_liquid_ooze_v1`) — it REVERSES the drain /
// leech-seed heal into damage on the healer, now modeled bit-for-bit (`apply_drain` /
// `apply_leech_seed`), so it is no longer a no-op NOR a fail-loud exclusion.
const NOOP_ABILITIES = new Set([
  'pressure', 'oblivious', 'runaway', 'illuminate', 'honeygather', 'pickup',
  'stench', 'sturdy', 'rockhead', 'earlybird',
  // `insomnia`/`vitalspirit` MOVED to MODELED_ABILITIES (`gen3_status_immune_v1`) — they
  // genuinely BLOCK sleep (a modeled status move), so they are modeled, not no-ops. `oblivious`
  // stays a no-op (its only immunity is to ATTRACT, which is not in the modeled move set).
  'noability',
  // BATCH-1 class-(a) NO-OPS (`gen3_ability_batch1_v1`), each PROVE-verified a true no-op in
  // the modeled move/item universe (`harness/probe_ability_batch1_noop_verify.js` — the
  // candidate ability vs an Insomnia control over full battles is BIT-IDENTICAL, STATE+SEED):
  //   lightningrod  — `onFoeRedirectTarget` (redirect Electric moves) → N/A in singles (one target).
  //   stickyhold    — `onTakeItem` (blocks Thief / Knock Off) — the item-removal moves are now
  //                   MODELED (`gen3_move_coverage_batch1_v1`), and the port models the Sticky
  //                   Hold block bit-for-bit (a `-activate` + the item unchanged), so it stays a
  //                   valid modeled ability (the block is a draw-free no-op on the seed).
  // PLUS / MINUS moved to MODELED_ABILITIES (`gen3_plus_minus_v1`, 2026-07-10): the no-op
  // verification tested them PARTNER-LESS — but the gen3 resolved `onModifySpA` scans
  // `getAllActive()` (FOES INCLUDED), so a cross-field Plus↔Minus pair is SpA ×1.5 (the A/B
  // fuzzer's thunderbolt-vs-Plusle/Minun cluster; probe `probe_plus_minus_gen3.js`). The
  // MODELED∪NOOP admission union is UNCHANGED, so the committed e2e golden is untouched.
  // FORECAST is DEFERRED (NOT a no-op): its `onWeatherChange` changes Castform's forme + TYPE in
  // rain/sun/hail (the probe DIVERGES under those weathers — `Castform-Rainy`/Water etc.), so a
  // filter-clean Castform under weather would desync. Left OUT for batch 2 (a forme-change model).
  'lightningrod', 'stickyhold',
]);
// FAIL-LOUD / DEFERRED abilities the ENGINE PANICS on (a GIGO guard) — the fuzz
// must NEVER feed one to the port. **NOW EMPTY**: FORECAST — the last member — is
// MODELED (`gen3_forecast_v1`, ROUND 35: the forme + TYPE swap at every WeatherChange
// site, the entrant onStart, the start window, the clearVolatile revert; hail/sandstorm
// modeled as weather-set moves so every forme is reachable), and `forecast` moved into
// MODELED_ABILITIES / `castform` is admitted. The sets are KEPT (empty) as the seam for
// the next ability deferral — `abilityAllowed` still hard-denies a member even if it
// were ever mistakenly added to a modeled/no-op set (the reject wins), and
// `adaptRandbatsTeam` (ab_fuzz) rejects a carrier team outright.
const REJECT_ABILITIES = new Set([]);
const REJECT_SPECIES = new Set([]);
// The MOVE analogue of REJECT_ABILITIES: a team CARRYING a move the ENGINE fail-louds on
// must never reach the port, even though `isModeledMove` already keeps the PICKER off it.
// That distinction is exactly how the Transform bug escaped: the offline fuzzers never PICKED
// Transform, but the LIVE bridge path (`gen_sim_bridge_diff.js`) drives choices off the sim's
// own request, so a randbats Ditto whose only move IS Transform reached it — node emitted
// `|-transform|p1a: Ditto|p2a: Kyogre` while the rust bridge emitted nothing (repro
// `soak_randbats/divergences/sbd_msapcesj_b22`).
//
// Both ORIGINAL members are modeled bit-for-bit and have LEFT: the wrap family in
// ROUND 32 (`gen3_partial_trap_v1`) and `transform` in ROUND 33 (`gen3_transform_v1`).
//
// **THE CURRENT MEMBERS are the ROUND-40 move-audit's 16 SILENT-DESYNC moves**
// (`gen3_unmodeled_move_failloud_v2` — see the rationale block on
// `state.rs::UNMODELED_FAILLOUD_MOVES`, kept in LOCKSTEP with it): each RAN in the engine
// with no guard while being outside the modeled universe (flat BP on a variable-BP move,
// a lock-in family with no lock, a missing first-turn/asleep-only gate), so the engine now
// fail-louds on them at CONSTRUCTION and every carrier team must be rejected here, or the
// fuzzers panic instead of skipping the team. EXPOSURE IS ZERO on both fuzzed surfaces
// (0 carriers in `data/teams/`; 0 in the ENTIRE curated gen3randombattle movepool —
// 220 species / 393 sets, exhaustive), so populating this cannot shift the e2e golden.
const REJECT_MOVES = new Set([
  'dreameater', 'falseswipe', 'furycutter', 'iceball',
  'outrage', 'petaldance', 'rage', 'revenge', 'rollout', 'secretpower',
  'smellingsalts', 'thrash', 'uproar', 'weatherball',
]);
function abilityAllowed(id) {
  const a = toId(id);
  if (REJECT_ABILITIES.has(a)) return false; // deferred / fail-loud → never admitted
  return MODELED_ABILITIES.has(a) || NOOP_ABILITIES.has(a) || a === '';
}

// The DATA-DRIVEN item classes (`gen3_item_mechanics_v1`): every entry below is priced
// by the port's generic dex-data path (`ItemData.type_boost`/`stat_mods`/`choice` →
// `resolve_atk_stat_mods`/`resolve_def_stat_mods`/`resolve_bp_mods`) — no hardcoded
// match-arm to drift from this set again (the Pink Bow / incense bug class). Class map:
// src/rust_sim/tests/vectors/gen3_mechanics_inventory.md; sweep golden:
// gen_item_mods_golden.js → tests/item_mods_test.rs.
const MODELED_ITEMS = new Set([
  '', 'leftovers', 'choiceband',
  // TYPE_BOOST, the gen3-mod STAT fold (×1.1 family + Sea Incense ×1.05)
  'seaincense',
  'charcoal', 'mysticwater', 'miracleseed', 'magnet', 'blackbelt', 'blackglasses',
  'dragonfang', 'hardstone', 'metalcoat', 'nevermeltice', 'poisonbarb',
  'sharpbeak', 'silkscarf', 'silverpowder', 'softsand', 'spelltag', 'twistedspoon',
  // TYPE_BOOST, the base-data BASE-POWER folds: the gen2 bows (DIRECT ×1.1 float)
  // + the gen4-named incenses (chainModify 4915/4096 ≈ ×1.2 — NOT ×1.1)
  'pinkbow', 'polkadotbow', 'oddincense', 'rockincense', 'roseincense', 'waveincense',
  // SPECIES_STAT (gen3-RESOLVED semantics — gen3 Light Ball is SpA-ONLY ×2;
  // DeepSeaScale/Metal Powder are DEFENDER-side folds; Soul Dew is both directions)
  'thickclub', 'lightball', 'deepseatooth', 'deepseascale', 'metalpowder', 'souldew',
  // Quick Claw (the port models its draw)
  'quickclaw',
  // ACCURACY_ITEM (`gen3_accuracy_pipeline_v1`): DEFENDER-side onModifyAccuracy DIRECT
  // multiplies — Bright Powder ×0.9 / Lax Incense ×0.95 (`ItemData.acc_mod` →
  // `turn.rs::effective_accuracy`). Draw-relevant (a hit/miss flip desyncs the seed);
  // validated by `tests/accuracy_test.rs` + the AC3 pin.
  'brightpowder', 'laxincense',
  // CRIT_ITEM (`gen3_crit_item_v1`, 0 team-carry — Farfetch'd/Chansey aren't gen3 OU;
  // admitted for completeness): the `onModifyCritRatio critRatio + N` fold (`ItemData.crit_boost`
  // → `effective_crit_ratio`). Scope Lens +1 (unconditional), Lucky Punch +2 (Chansey), Stick +2
  // (Farfetch'd). DRAW-FREE (only the crit denominator shifts — NOT the draw count), so the crit
  // outcome (crit-inclusive HP) is the only observable; validated by the CI1/CI2 pins.
  'scopelens', 'luckypunch', 'stick',
  // BOOST_RESTORE (`gen3_white_herb_v1`) — White Herb: restore all NEGATIVE boost stages to 0 +
  // consume (`ItemData.boost_restore` → `turn.rs::white_herb_restore`, at the after-move /
  // switch-in stat-drop sites). DRAW-FREE, so it never shifts the LCG; the boosts + item-held are
  // the only observable. Validated by `gen_whiteherb_golden.js` → `tests/whiteherb_test.rs` + the
  // WH1-WH5 pins. (0 sample teams carry it — the leech-seed situation — so admitting it is
  // byte-neutral on the e2e sample; a real Overheat/Superpower user could carry it.)
  'whiteherb',
  // PROC_ITEM (`gen3_ability_batch4_v1`, batch 4 — 0 team-carry, admitted for completeness):
  //   kingsrock — the appended trailing `{chance:10, flinch}` secondary for the LISTED moves
  //     (`ItemData.flinch_secondary`, execution-derived list): one extra random(100) AFTER
  //     the move's own secondary, BEFORE the foe's contact proc; Serene Grace ×2; Shield
  //     Dust filters; drawn-not-applied behind a sub; Seismic Toss/Struggle proc too.
  //   focusband — the onDamage `randomChance(1,10)` drawn FIRST on EVERY Damage event into
  //     the holder (move hits, burn/sand chips, Spikes, recoil, confusion self-hits; NOT
  //     sub-absorbed hits); survive-at-1 only on a lethal MOVE hit.
  'kingsrock', 'focusband',
  // BERRIES (`gen3_berry_trace_shedskin_v1`, batch 3): the EXACT 22 data-driven
  // `berryEffect` rows in gen3_items.json — the ONE eatItem consumption mechanism
  // (item → NONE permanently) + the four effect classes, all probe-settled
  // (`probe_berry_rng.js` / `probe_berry_sub_tie_rng.js`: the eat is DRAW-FREE; only
  // Starf's `sample` + the Figy family's nature-gated confusion `random(2,6)` draw;
  // the HEAL/PINCH residual handler shares Leftovers' order-10-subOrder-4 slot, so a
  // berry-vs-Leftovers equal-speed mirror draws IDENTICALLY) and validated by
  // `gen_berry_batch3_golden.js` → `tests/berry_batch3_test.rs` (1280 battles) + the
  // BR1-BR3/BR6 pins. `lumberry` (=64) + `salacberry` (=46) are the top remaining
  // taxonomy ITEM gaps — the batch-3 admission levers.
  //   CURE (Update-site eat BEFORE the holder's move; lum immediate-in-setStatus):
  'cheriberry', 'chestoberry', 'pechaberry', 'rawstberry', 'aspearberry',
  'persimberry', 'lumberry',
  //   HEAL (residual, `2*hp <= maxhp` exact — the BR6 boundary pin):
  'oranberry', 'sitrusberry', 'figyberry', 'wikiberry', 'magoberry',
  'aguavberry', 'iapapaberry',
  //   PINCH (residual, `4*hp <= maxhp` exact; Starf's sample; Lansat's focusenergy):
  'liechiberry', 'ganlonberry', 'salacberry', 'petayaberry', 'apicotberry',
  'lansatberry', 'starfberry',
  //   PP (leppa: +10 on the depleted slot at the Update site):
  'leppaberry',
]);
function itemAllowed(id) { return MODELED_ITEMS.has(toId(id)); }

function toId(s) {
  return ('' + (s || '')).toLowerCase().replace(/[^a-z0-9]/g, '');
}

// A team is FILTER-CLEAN iff every mon's ability + item is allowed AND it carries no
// DEFERRED/fail-loud species (Castform-Forecast — the engine panics on it, so it must
// never reach the port). Castform's only ability IS Forecast, so the ability reject
// already catches it; the explicit species reject covers a hand-hacked non-Forecast
// Castform (gen3customgame) too.
function teamFilterClean(packed) {
  const team = Teams.unpack(packed);
  for (const set of team) {
    const sid = toId(set.species || set.name);
    if (REJECT_SPECIES.has(sid)) return { ok: false, why: `ability:forecast` };
    for (const mv of (set.moves || [])) {
      if (REJECT_MOVES.has(toId(mv))) return { ok: false, why: `move:${toId(mv)}` };
    }
    if (!abilityAllowed(set.ability)) return { ok: false, why: `ability:${toId(set.ability)}` };
    if (!itemAllowed(set.item)) return { ok: false, why: `item:${toId(set.item)}` };
  }
  return { ok: true };
}

// ── Team loading ─────────────────────────────────────────────────────────────
// Load every .txt, import → validate (gen3ou) → pack. Skip rejects / import fails.
function loadTeams() {
  const files = [];
  const walk = (dir) => {
    for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
      const p = path.join(dir, ent.name);
      if (ent.isDirectory()) walk(p);
      else if (ent.name.endsWith('.txt')) files.push(p);
    }
  };
  walk(TEAMS_DIR);
  files.sort(); // deterministic order

  const validator = new TeamValidator(VALIDATE_FORMAT);
  const teams = [];
  let skipped = 0;
  for (const f of files) {
    let packed;
    try {
      const text = fs.readFileSync(f, 'utf8');
      const team = Teams.import(text);
      if (!team || team.length === 0) { skipped++; continue; }
      const errs = validator.validateTeam(team);
      if (errs && errs.length) { skipped++; continue; }
      packed = Teams.pack(team);
      // Round-trip sanity: the packed string must re-unpack to the same species set.
      const re = Teams.unpack(packed);
      if (!re || re.length !== team.length) { skipped++; continue; }
    } catch (e) { skipped++; continue; }
    teams.push({ file: path.relative(ROOT, f), packed });
  }
  return { teams, skipped, total: files.length };
}

// ── The battle driver ─────────────────────────────────────────────────────────
// Picks a RANDOM legal choice each decision from `chooseRng`, restricted by
// `opts.modeledOnly` (the FILTERED gate uses modeled moves only; the taxonomy uses
// damaging-or-switch). Returns the recorded per-decision golden + a `dropped`
// reason if the battle hit an unmodeled forced state (no legal modeled choice +
// no switch).
function statusOf(active) {
  const st = (active && active.status) || '';
  let stage = 0;
  if (st === 'tox') stage = active.statusState ? (active.statusState.stage || 0) : 0;
  if (st === 'slp') stage = active.statusState ? (active.statusState.time || 0) : 0;
  return { status: st || '-', stage };
}
function boostsOf(active) {
  const b = (active && active.boosts) || {};
  return [b.atk || 0, b.def || 0, b.spa || 0, b.spd || 0, b.spe || 0];
}
function confusionOf(active) {
  if (!active || !active.volatiles || !active.volatiles.confusion) return 0;
  return active.volatiles.confusion.time || 0;
}
// The Spikes layer count on a side (`side.sideConditions.spikes.layers`, 0 = absent) —
// the SIDE-CONDITION state the Rust e2e gate now asserts (so a switch-in onto a spiked
// side takes the right hazard chip on real Skarmory/Forretress/Cloyster spiker teams).
function spikesOf(side) {
  const sc = side.sideConditions && side.sideConditions['spikes'];
  return sc ? (sc.layers | 0) : 0;
}
function snap(side) {
  const a = side.active[0];
  if (!a) return { species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', stage: 0, left: side.pokemonLeft, boosts: [0, 0, 0, 0, 0], confusion: 0, spikes: spikesOf(side) };
  const { status, stage } = statusOf(a);
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status, stage, left: side.pokemonLeft, boosts: boostsOf(a), confusion: confusionOf(a),
    spikes: spikesOf(side),
  };
}
function forceSwitchTable(battle) {
  const out = [false, false];
  if (battle.requestState !== 'switch') return out;
  for (let i = 0; i < 2; i++) {
    const req = battle.sides[i].activeRequest;
    if (req && req.forceSwitch && req.forceSwitch[0]) out[i] = true;
  }
  return out;
}
// First mover this turn = the FIRST mon to RUN its action (the action-order sort's
// decision). A fully-paralyzed / asleep / flinched mon emits `|cant|` (not `|move|`)
// and a confusion-self-hit emits `|-activate|…|confusion` — both mean that mon RAN
// its action FIRST, matching the Rust's `first_mover` (= first sorted action). So we
// count those too (mirrors gen_secondary_golden.js — the e2e fuzz inflicts status).
function firstMoverSince(log, fromIdx) {
  for (let i = fromIdx; i < log.length; i++) {
    const parts = log[i].split('|');
    const tag = parts[1];
    const isAction =
      tag === 'move' || tag === 'switch' || tag === 'cant' ||
      (tag === '-activate' && (parts[3] || '') === 'confusion');
    if (isAction && parts.length >= 3) {
      const actor = parts[2].trim();
      if (actor.startsWith('p1a:')) return 'p1';
      if (actor.startsWith('p2a:')) return 'p2';
    }
  }
  return 'none';
}

// Encode a submitted choice into the compact golden token (matches gen_fullbattle).
function encodeChoice(c) {
  if (!c) return '-';
  const m = c.match(/^move\s+(\d+)$/);
  if (m) return `m${Number(m[1]) - 1}`;
  const s = c.match(/^switch\s+(\d+)$/);
  if (s) return `s${Number(s[1]) - 1}`;
  throw new Error(`unencodable choice ${JSON.stringify(c)}`);
}

// The exact inverse of encodeChoice — decode a recorded token back into the wire
// choice string (used by runBattle's `opts.replayChoices` path so a saved repro
// replays the RECORDED choices rather than re-picking a fresh trajectory). `-` ->
// null (no `>pN` write), `m<k>` -> `move (k+1)`, `s<n>` -> `switch (n+1)`.
function decodeChoice(tok) {
  if (tok === '-' || tok === null || tok === undefined) return null;
  const m = String(tok).match(/^m(\d+)$/);
  if (m) return `move ${Number(m[1]) + 1}`;
  const s = String(tok).match(/^s(\d+)$/);
  if (s) return `switch ${Number(s[1]) + 1}`;
  throw new Error(`undecodable choice token ${JSON.stringify(tok)}`);
}

// Pick the move-choice for a side at a `move` request: choose a random MODELED (or,
// for the taxonomy, damaging) legal move; if none, a random legal switch; else null
// (a forced-unmodeled state — the caller drops the battle from the FILTERED set, or
// records the cause for the taxonomy).
//
// `mode`: 'modeled' (FILTERED) requires isModeledMove; 'damaging' (taxonomy) accepts
// any non-status move (so it doesn't trivially desync on a chosen status move — the
// taxonomy is about ability/item gaps, not "we chose a status move").
function pickMove(battle, side, rng, mode, allowHiddenPower = false) {
  const req = battle.sides[side].activeRequest;
  if (!req || !req.active || !req.active[0]) return { choice: null, reason: 'no-active-request' };
  const moves = req.active[0].moves || [];
  const legalMoveSlots = [];
  for (let k = 0; k < moves.length; k++) {
    const mv = moves[k];
    if (mv.disabled) continue;
    const id = toId(mv.id || mv.move);
    if (mode === 'modeled') {
      // SLEEP TALK (`gen3_move_coverage_batch5_v1`): pickable ONLY when the carrier's
      // sampled POOL is all-modeled — the CALLED move bypasses the picker, so a
      // pool member the port fail-louds on (or the phaze-excluded Roar) must veto it.
      if (id === 'sleeptalk') {
        if (!BATCH5_E2E_EXCLUDED && sleepTalkPoolModeled(battle, side)) legalMoveSlots.push(k);
        continue;
      }
      // STRUGGLE is ENGINE-MODELED (`pp_struggle_test.rs` is a STATE+PP+SEED+winner
      // differential over it) but `isModeledMove` returns false for it — a picker FALSE
      // NEGATIVE, and an expensive one. When every slot is spent the sim offers ONLY
      // Struggle, so a mon that also cannot switch had NO pickable choice and the whole
      // battle was DROPPED to a prefix (`forced-unmodeled-move:struggle`). That truncates
      // exactly the PP-exhaustion endgames — the deepest, most state-laden turns, and the
      // ones gen3ou STALL teams produce most. The live per-side gate already accepts it
      // (`gen_sim_bridge_diff.js`: `id === 'struggle' || isModeledMove(id)`); the two
      // harnesses simply disagreed, and the offline one was the weaker.
      if (id === 'struggle' || isModeledMove(id, allowHiddenPower)) legalMoveSlots.push(k);
    } else {
      // taxonomy: damaging (non-status), still skip the structurally-unreplayable
      // moves that would desync on the CHOICE side (variable power/fixed/2-turn/
      // multi/recoil/drain/selfKO) — we want gaps to surface as ability/item, not
      // "we picked an unmodeled move".
      const m = dex3.moves.get(id);
      if (m && m.exists && m.category !== 'Status' && isModeledMove(id)) legalMoveSlots.push(k);
    }
  }
  // TRAPPED (`gen3_trapping_v1`): a trapped active mon's voluntary switch would be
  // REJECTED by the sim's `side.choose` ("Can't switch: The active Pokémon is
  // trapped"), stalling the capture — so the picker treats its bench as empty. This is
  // the sim's own request-legality fact (`pokemon.trapped`, set by the foe's Arena
  // Trap / Magnet Pull at endTurn), NOT a heuristic; forced replacements
  // (`pickReplacement`) stay un-gated (trapping never blocks a faint replacement).
  const active = battle.sides[side].active[0];
  // A MOVE-LOCKED request (`gen3_move_coverage_batch4c_v1` — Hyper Beam's recharge turn /
  // Solar Beam's fire turn) serializes `trapped:true` on the REQUEST (the sim's
  // lockedmove request shape) without setting `pokemon.trapped` — the picker must not
  // submit a doomed switch (the sim rejects it, stalling the capture).
  const reqTrapped = !!(req.active[0].trapped);
  const isTrapped = !!(active && active.trapped) || reqTrapped;
  const switchSlots = isTrapped ? [] : legalSwitchSlots(battle, side);
  if (legalMoveSlots.length === 0) {
    // No modeled move — prefer a switch.
    if (switchSlots.length > 0) {
      const n = switchSlots[randInt(rng, switchSlots.length)];
      return { choice: `switch ${n + 1}`, reason: 'switch-no-modeled-move' };
    }
    // No modeled move AND no switch (incl. TRAPPED with only unmodeled moves — the
    // trapped mon must fight, and the port can't replay an unmodeled move) →
    // forced-unmodeled.
    //
    // ⚠️ REPORT THE MOVES THAT ACTUALLY BLOCKED, not `moves[0]`. The old label named the
    // FIRST slot in the request regardless of why the pick failed, so a drop on a mon whose
    // first slot happened to be Substitute read as `forced-unmodeled-move:substitute` — and
    // Substitute is modeled, so that label sent a reader hunting a bug that did not exist
    // (it did, on 2026-08-17). A diagnostic that names an innocent bystander is worse than
    // one that names nothing. `undisabled` is the honest set: the slots that were OFFERED
    // and still unpickable; a fully-`disabled` request is its own distinct reason.
    const offered = moves.map((m) => ({ id: toId(m.id || m.move), disabled: !!m.disabled }));
    const undisabled = offered.filter((m) => !m.disabled).map((m) => m.id);
    const why = undisabled.length
      ? undisabled.sort().join('+')
      : `all-disabled(${offered.map((m) => m.id).sort().join('+') || 'none'})`;
    return { choice: null, reason: `forced-unmodeled-move:${why}` };
  }
  // Mostly attack; occasionally switch (to exercise the switch-phase draws) — 1/6.
  if (switchSlots.length > 0 && rng() < 1 / 6) {
    const n = switchSlots[randInt(rng, switchSlots.length)];
    return { choice: `switch ${n + 1}`, reason: 'voluntary-switch' };
  }
  const k = legalMoveSlots[randInt(rng, legalMoveSlots.length)];
  // Flag whether the picked move is a STATUS move and, more specifically, a pure
  // SELF-BOOST SETUP move (so the golden can count how many decisions exercise the new
  // status-move layer / setup-move layer respectively).
  const pickedId = toId(moves[k].id || moves[k].move);
  const pm = dex3.moves.get(pickedId);
  const isStatus = !!(pm && pm.category === 'Status');
  const isSetup = MODELED_SETUP_MOVES.has(pickedId);
  const isRecovery = MODELED_RECOVERY_MOVES.has(pickedId);
  const isProtect = MODELED_PROTECT_MOVES.has(pickedId);
  const isPhaze = MODELED_PHAZE_MOVES.has(pickedId);
  const isLeech = MODELED_LEECH_MOVES.has(pickedId);
  const isSubstitute = MODELED_SUBSTITUTE_MOVES.has(pickedId);
  const isRestriction = MODELED_RESTRICTION_MOVES.has(pickedId);
  const isExplosion = pickedId === 'explosion' || pickedId === 'selfdestruct';
  const isFixed = MODELED_FIXED_DAMAGE_MOVES.has(pickedId);
  const isBatch5 = MODELED_BATCH5_REACTIVE_MOVES.has(pickedId) ||
    MODELED_BATCH5_VARBP_MOVES.has(pickedId) || pickedId === 'sleeptalk';
  const isBatch6 = MODELED_BATCH6_MOVES.has(pickedId);
  return { choice: `move ${k + 1}`, reason: 'modeled-move', pickedId, statusMove: isStatus, setupMove: isSetup, recoveryMove: isRecovery, protectMove: isProtect, phazeMove: isPhaze, leechMove: isLeech, substituteMove: isSubstitute, restrictionMove: isRestriction, explosionMove: isExplosion, fixedMove: isFixed, batch5Move: isBatch5, batch6Move: isBatch6 };
}

// Whether a Sleep-Talk carrier's SAMPLED POOL is safe for the modeled capture
// (`gen3_move_coverage_batch5_v1`): every pool-ELIGIBLE moveslot (not
// `flags.nosleeptalk`, not `flags.charge` — the pool filter Sleep Talk itself applies,
// so a charge/nosleeptalk move can never be CALLED regardless of modeledness) must be
// `isModeledMove` — else the called move would desync/fail-loud the port. An empty
// pool is SAFE (the modeled `[still]`+`-fail` branch).
function sleepTalkPoolModeled(battle, side) {
  const active = battle.sides[side].active[0];
  if (!active) return false;
  for (const slot of active.moveSlots) {
    const mid = toId(slot.id);
    const mv = dex3.moves.get(mid);
    if (!mv || !mv.exists) return false;
    if (mv.flags && (mv.flags.nosleeptalk || mv.flags.charge)) continue; // pool-excluded
    if (!isModeledMove(mid)) return false;
  }
  return true;
}

function legalSwitchSlots(battle, side) {
  const s = battle.sides[side];
  const out = [];
  for (let k = 0; k < s.pokemon.length; k++) {
    const p = s.pokemon[k];
    if (p !== s.active[0] && !p.fainted) out.push(k);
  }
  return out;
}
function pickReplacement(battle, side, rng) {
  const slots = legalSwitchSlots(battle, side);
  if (slots.length === 0) return null;
  const n = slots[randInt(rng, slots.length)];
  return `switch ${n + 1}`;
}

// Run ONE battle to game-end (or until a drop/safety). Returns the record.
async function runBattle(p1Packed, p2Packed, seed, chooseSeed, mode, opts = {}) {
  // `opts.format` (default gen3customgame) drives the `>start` formatid — gen3ou turns
  // on the Sleep/Freeze-Clause SetStatus handler-sort shuffle + the OU framing, exactly
  // the extra draw/emission path the byte fuzzer probes (the state fuzzer stays on the
  // default). `opts.allowHiddenPower` widens the choice picker to admit typed HP (pool
  // mode only — see isModeledMove). Both default to the committed-golden behavior.
  const runFormat = opts.format || FORMAT;
  const allowHiddenPower = !!opts.allowHiddenPower;
  // `opts.replayChoices` (the decoded summary.choices array — one `[tokenP1,tokenP2]`
  // per resolved decision boundary) REPLAYS the RECORDED choices instead of re-picking
  // a fresh trajectory: when present the decision loop SKIPS pickMove/pickReplacement
  // and pops the recorded pair for the current decisionNo (decodeChoice each token:
  // `-`->null, `m<k>`->`move k+1`, `s<n>`->`switch n+1`). Everything else (the
  // >start/>player prime, the 16-tick pump, seedBefore/seedAfter capture) is
  // BYTE-IDENTICAL to the recorder, so the sim reproduces the recorded golden EXACTLY
  // — the reliable per-draw trace path (probe_repro_simtrace.js).
  const replayChoices = opts.replayChoices || null;
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  // Tee the OMNISCIENT `|...|` stream verbatim (the SAME consumer pattern
  // gen_protocol_capture.js uses), NORMALIZING `|t:|` to the shared placeholder so the
  // captured golden stays byte-stable across regens (the byte-diff filter re-normalizes
  // defensively too). This `log` doubles as the firstMoverSince source (unchanged — it
  // scans |move|/|switch|/|cant|, none of which is a |t:| line).
  (async () => {
    for await (const ch of streams.omniscient) {
      for (const l of ch.split('\n')) {
        if (l) log.push(l.startsWith('|t:|') ? '|t:|<NORMALIZED>' : l);
      }
    }
  })();

  streams.omniscient.write(`>start {"formatid":"${runFormat}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1Packed })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2Packed })}`);
  for (let i = 0; i < 12; i++) await tick();

  const rng = mulberry32(chooseSeed);
  // `movesUsed` (id -> count) is COVERAGE-ONLY metadata for the A/B fuzzer's
  // distinct-moves-exercised tally — never serialized into the golden.
  const rec = { initSeed: null, decisions: [], winner: null, ended: false, dropped: null, chooseSeed, movesUsed: {} };

  let decisionNo = 0;
  let safety = 0;
  while (!stream.battle.ended && safety < SAFETY) {
    safety++;
    const battle = stream.battle;
    const reqState = battle.requestState;
    if (reqState !== 'move' && reqState !== 'switch') { await tick(); continue; }
    const force = forceSwitchTable(battle);
    const seedBefore = battle.prng.getSeed();
    if (decisionNo === 0) {
      rec.initSeed = seedBefore;
      // `gen3_turn0_quick_claw_capture_v1` — the offline replay convention resumes from the
      // POST-CONSTRUCTION state (initSeed is captured HERE, at the first decision request,
      // deliberately skipping the turn-0 construction window). `battle.quickClawRoll` is part
      // of that state and was silently DROPPED: it is `randomChance(1,5)` rolled at every
      // COMPLETED endTurn and READ on the NEXT turn (`gen3_quick_claw_speed_v1`), so turn 1's
      // value is decided during construction. Without it the port resumes with its
      // `quick_claw_roll: false` default and mis-orders turn 1 whenever a Quick Claw leads and
      // the roll came up true — ~1 lead in 5. (Probe: `probe_turn0_quickclaw_offline.js`.)
      rec.quickClawRoll = !!battle.quickClawRoll;
    }

    let cp1 = null; let cp2 = null;
    let statusMoveThisDec = false;
    let setupMoveThisDec = false;
    let recoveryMoveThisDec = false;
    let protectMoveThisDec = false;
    let phazeMoveThisDec = false;
    let leechMoveThisDec = false;
    let substituteMoveThisDec = false;
    let restrictionMoveThisDec = false;
    let explosionMoveThisDec = false;
    let fixedMoveThisDec = false;
    let batch5MoveThisDec = false;
    let batch6MoveThisDec = false;
    if (replayChoices) {
      // REPLAY the recorded choices for this decision boundary (no re-picking) — the
      // sim reproduces the recorded golden bit-for-bit. Decode `-`->null uniformly for
      // both reqState kinds (a switch request records `-` for the non-flagged side).
      const rec_pair = replayChoices[decisionNo];
      if (!rec_pair) { rec.dropped = 'replay-exhausted'; break; }
      cp1 = decodeChoice(rec_pair[0]);
      cp2 = decodeChoice(rec_pair[1]);
    } else if (reqState === 'switch') {
      if (force[0]) { cp1 = pickReplacement(battle, 0, rng); if (!cp1) { rec.dropped = 'no-replacement-p1'; break; } }
      if (force[1]) { cp2 = pickReplacement(battle, 1, rng); if (!cp2) { rec.dropped = 'no-replacement-p2'; break; } }
    } else {
      const r1 = pickMove(battle, 0, rng, mode, allowHiddenPower);
      const r2 = pickMove(battle, 1, rng, mode, allowHiddenPower);
      if (r1.choice === null) { rec.dropped = r1.reason; break; }
      if (r2.choice === null) { rec.dropped = r2.reason; break; }
      cp1 = r1.choice; cp2 = r2.choice;
      for (const r of [r1, r2]) {
        if (r && r.pickedId && r.choice && r.choice.startsWith('move')) {
          rec.movesUsed[r.pickedId] = (rec.movesUsed[r.pickedId] || 0) + 1;
        }
      }
      statusMoveThisDec = !!(r1.statusMove || r2.statusMove);
      setupMoveThisDec = !!(r1.setupMove || r2.setupMove);
      recoveryMoveThisDec = !!(r1.recoveryMove || r2.recoveryMove);
      protectMoveThisDec = !!(r1.protectMove || r2.protectMove);
      phazeMoveThisDec = !!(r1.phazeMove || r2.phazeMove);
      leechMoveThisDec = !!(r1.leechMove || r2.leechMove);
      substituteMoveThisDec = !!(r1.substituteMove || r2.substituteMove);
      restrictionMoveThisDec = !!(r1.restrictionMove || r2.restrictionMove);
      explosionMoveThisDec = !!(r1.explosionMove || r2.explosionMove);
      fixedMoveThisDec = !!(r1.fixedMove || r2.fixedMove);
      batch5MoveThisDec = !!(r1.batch5Move || r2.batch5Move);
      batch6MoveThisDec = !!(r1.batch6Move || r2.batch6Move);
    }

    const logLenBefore = log.length;
    try { if (cp1) streams.omniscient.write(`>p1 ${cp1}`); } catch (e) {}
    try { if (cp2) streams.omniscient.write(`>p2 ${cp2}`); } catch (e) {}
    for (let i = 0; i < 16; i++) await tick();

    const seedAfter = battle.prng.getSeed();
    rec.decisions.push({
      request: reqState, force,
      choiceP1: encodeChoice(cp1), choiceP2: encodeChoice(cp2),
      seedAfter, p1: snap(battle.sides[0]), p2: snap(battle.sides[1]),
      firstMover: reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none',
      statusMove: statusMoveThisDec,
      setupMove: setupMoveThisDec,
      recoveryMove: recoveryMoveThisDec,
      protectMove: protectMoveThisDec,
      phazeMove: phazeMoveThisDec,
      leechMove: leechMoveThisDec,
      substituteMove: substituteMoveThisDec,
      restrictionMove: restrictionMoveThisDec,
      explosionMove: explosionMoveThisDec,
      fixedMove: fixedMoveThisDec,
      batch5Move: batch5MoveThisDec,
      batch6Move: batch6MoveThisDec,
    });
    decisionNo++;
  }

  rec.ended = !!stream.battle.ended;
  rec.winner = stream.battle.winner;
  // Tee the captured OMNISCIENT lines (normalized) + the run format onto the record so
  // the byte-differential path (ab_fuzz --protocol → emitBattle L/FMT rows → ab_replay)
  // can diff them; the state-only golden path ignores both (emitBattle gates L/FMT on
  // `opts.protocol`). The `[from]`/framing/emission bytes ride here verbatim.
  rec.lines = log;
  rec.format = runFormat;
  try { streams.omniscient.destroy(); } catch (e) {}
  return rec;
}

function winTok(rec) {
  if (!rec.ended) return 'none';
  if (rec.winner === 'P1') return 'p1';
  if (rec.winner === 'P2') return 'p2';
  if (rec.winner === '' ) return 'tie';
  return 'none';
}

// Serialize a battle record into the SHARED golden line format (so the Rust e2e
// test reuses the fullbattle parser shape: SCEN/TEAM/INIT/DEC/END, with DEC carrying
// boosts[5]+confusion like the secondary golden).
function emitBattle(lines, id, p1Packed, p2Packed, rec, opts = {}) {
  lines.push(`SCEN\t${id}`);
  lines.push(`TEAM\t${id}\tp1\t${p1Packed}`);
  lines.push(`TEAM\t${id}\tp2\t${p2Packed}`);
  // FMT carries the run format (gen3customgame default) so the Rust byte replayer sets
  // BattleOptions.format_id to match (gen3ou → sleep_clause ON + the OU reframe). Emitted
  // ONLY in protocol mode (state-only chunks stay format-implicit gen3customgame); it sits
  // before INIT so ab_replay's parser (which only tracks a case AFTER INIT) ignores it on
  // the state path. `gen3_omniscient_byte_fuzz_v1`.
  if (opts.protocol) lines.push(`FMT\t${id}\t${rec.format || FORMAT}`);
  // The 5th field is the turn-0 `quickClawRoll` (`gen3_turn0_quick_claw_capture_v1`). It is
  // APPENDED, and `ab_replay` treats it as OPTIONAL (absent → false), so every previously
  // saved repro dir and every committed golden stays replayable byte-for-byte.
  lines.push(['INIT', id, rec.initSeed, rec.chooseSeed, rec.quickClawRoll ? 1 : 0].join('\t'));
  rec.decisions.forEach((d, di) => {
    const sp = (s) => [s.species, s.hp, s.maxhp, s.fainted ? 1 : 0, s.status,
      s.boosts[0], s.boosts[1], s.boosts[2], s.boosts[3], s.boosts[4], s.confusion, s.left].join('\t');
    lines.push([
      'DEC', id, di, d.request, d.force[0] ? 1 : 0, d.force[1] ? 1 : 0,
      d.choiceP1, d.choiceP2, d.seedAfter, sp(d.p1), sp(d.p2), d.firstMover,
      // SPIKES layers per side (the entry-hazard SIDE CONDITION) — appended after `first`.
      d.p1.spikes, d.p2.spikes,
      // FIXED-DAMAGE / BATCH-5 usage flags (appended LAST — the Rust gate's coverage
      // FLOORS read these, so a regen that silently stops sampling the fixed-damage
      // five / the batch-5 nine FAILS the gate instead of staying green; the review's
      // "generator statistic, not a gated assertion" fix).
      d.fixedMove ? 1 : 0, d.batch5Move ? 1 : 0, d.batch6Move ? 1 : 0,
    ].join('\t'));
  });
  lines.push(['END', id, rec.ended ? 1 : 0, winTok(rec)].join('\t'));
  // The RAW captured OMNISCIENT `|...|` lines, verbatim (`|t:|` already normalized on
  // capture), in emission order — the byte-diff target (`gen3_omniscient_byte_fuzz_v1`).
  // Emitted ONLY in protocol mode. Grammar mirrors gen_protocol_capture.js:
  //   L <id> <lineNo> <RAW line — MAY contain tabs/pipes; it's the LAST field>.
  // ab_replay parses with splitn(4) so embedded tabs/pipes round-trip exactly.
  if (opts.protocol && rec.lines) {
    rec.lines.forEach((raw, li) => lines.push(['L', id, li, raw].join('\t')));
  }
}

// ── Taxonomy classification ───────────────────────────────────────────────────
// For the UNFILTERED sweep we don't re-run the Rust here; instead we classify each
// battle by the modeled-coverage of its TEAMS + the choices it actually used. The
// committed Rust taxonomy test (e2e_taxonomy_test, ignored-by-default) does the
// real divergence run; this Node side records the COVERAGE gaps (which ability/item
// each battle carried) so we have a ranked "what's blocking real teams" list even
// without the Rust replay. The Rust filtered gate is the bit-for-bit proof.
function classifyTeamsGaps(p1Packed, p2Packed) {
  const gaps = { abilities: new Set(), items: new Set() };
  for (const packed of [p1Packed, p2Packed]) {
    const team = Teams.unpack(packed);
    for (const set of team) {
      const a = toId(set.ability);
      const it = toId(set.item);
      if (!abilityAllowed(a)) gaps.abilities.add(a);
      if (!itemAllowed(it)) gaps.items.add(it);
    }
  }
  return gaps;
}

async function main() {
  const t0 = Date.now();
  const { teams, skipped, total } = loadTeams();
  console.error(`teams: loaded ${teams.length} / ${total} (.txt), skipped ${skipped} (import/validate)`);
  if (teams.length < 8) { console.error('too few valid teams loaded'); process.exit(1); }

  const pairRng = mulberry32(MASTER_SEED);

  // Precompute which loaded teams are FILTER-CLEAN (modeled abilities + items).
  const cleanIdx = [];
  for (let i = 0; i < teams.length; i++) {
    if (teamFilterClean(teams[i].packed).ok) cleanIdx.push(i);
  }
  console.error(`filter-clean teams (modeled ability+item only): ${cleanIdx.length} / ${teams.length}`);

  // ===========================================================================
  // (1) THE FILTERED GATE — pair filter-clean teams, run modeled-only choices,
  //     keep battles that reach game-end with NO drop. Bit-for-bit golden.
  // ===========================================================================
  const goldenLines = [];
  goldenLines.push('# e2e_fuzz_golden.txt — CAPSTONE filtered gate (real teams, full battles, modeled mechanics only).');
  goldenLines.push('# Per-decision-boundary STATE+SEED differential to GAME-END; the Rust replays bit-for-bit.');
  goldenLines.push(`# MASTER_SEED ${MASTER_SEED}  filter-clean-teams ${cleanIdx.length}/${teams.length}`);
  goldenLines.push('# SCEN <id>');
  goldenLines.push('# TEAM <id> <p1|p2> <packed>');
  goldenLines.push('# INIT <id> <initSeed m,n,o,p> <chooseSeed> <turn0QuickClawRoll:0|1>');
  goldenLines.push('# DEC  <id> <di> <move|switch> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  goldenLines.push('#       p1(species hp max fnt status atk def spa spd spe conf left) p2(...) first_mover \\');
  goldenLines.push('#       p1Spikes p2Spikes fixedMove batch5Move batch6Move');
  goldenLines.push('#   choice token: m<K>=move slot K (0-based) | s<N>=switch slot N (0-based) | -');
  goldenLines.push('# END  <id> <ended:0|1> <winner:p1|p2|tie|none>');

  let filteredKept = 0; let filteredDropped = 0; let tries = 0;
  let decRows = 0; let winRows = 0; let tieRows = 0; let switchRows = 0; let doubleRows = 0;
  let statusMoveDecs = 0; // how many decisions USE a status move (the prior coverage)
  let setupMoveDecs = 0; // how many decisions USE a pure self-boost setup move (the prior coverage)
  let recoveryMoveDecs = 0; // how many decisions USE a self-heal recovery move (the prior coverage)
  let protectMoveDecs = 0; // how many decisions USE a Protect/Detect move (the prior coverage)
  let phazeMoveDecs = 0; // how many decisions USE a Roar/Whirlwind phaze move (the prior coverage)
  let leechMoveDecs = 0; // how many decisions USE a Leech Seed move (the prior coverage)
  let substituteMoveDecs = 0; // how many decisions USE a Substitute move (the prior coverage)
  let restrictionMoveDecs = 0; // how many decisions USE a Taunt/Disable move (the NEW coverage)
  let explosionMoveDecs = 0; // how many decisions USE an Explosion/Self-Destruct move (the prior coverage)
  let fixedMoveDecs = 0; // how many decisions USE a FIXED-DAMAGE move (the NEW coverage)
  let batch5MoveDecs = 0; // how many decisions USE a BATCH-5 move (reactive / variable-BP / sleep talk)
  let batch6MoveDecs = 0; // how many decisions USE a BATCH-6 move (the final tail)
  const dropReasons = new Map();

  while (filteredKept < FILTERED_TARGET && tries < MAX_TRIES && cleanIdx.length >= 2) {
    tries++;
    const ia = cleanIdx[randInt(pairRng, cleanIdx.length)];
    let ib = cleanIdx[randInt(pairRng, cleanIdx.length)];
    if (ib === ia) ib = cleanIdx[(cleanIdx.indexOf(ib) + 1) % cleanIdx.length];
    if (ib === ia) continue;
    const seedState = (Math.imul(pairRng() * 4294967296, 1) ^ (tries * 2654435761)) >>> 0;
    const seed = seedFrom(seedState);
    const chooseSeed = (Math.floor(pairRng() * 4294967296) ^ 0x9e3779b9) >>> 0;
    let rec;
    try {
      rec = await runBattle(teams[ia].packed, teams[ib].packed, seed, chooseSeed, 'modeled');
    } catch (e) { filteredDropped++; dropReasons.set('exception', (dropReasons.get('exception') || 0) + 1); continue; }
    if (rec.dropped || !rec.ended || !rec.initSeed || rec.decisions.length === 0) {
      filteredDropped++;
      const r = rec.dropped || (!rec.ended ? 'not-ended' : 'empty');
      const key = r.split(':')[0];
      dropReasons.set(key, (dropReasons.get(key) || 0) + 1);
      continue;
    }
    const id = `e2e_${filteredKept}`;
    emitBattle(goldenLines, id, teams[ia].packed, teams[ib].packed, rec);
    filteredKept++;
    decRows += rec.decisions.length;
    const wt = winTok(rec);
    if (wt === 'p1' || wt === 'p2') winRows++;
    if (wt === 'tie') tieRows++;
    for (const d of rec.decisions) {
      if (d.request === 'switch') { switchRows++; if (d.force[0] && d.force[1]) doubleRows++; }
      if (d.statusMove) statusMoveDecs++;
      if (d.setupMove) setupMoveDecs++;
      if (d.recoveryMove) recoveryMoveDecs++;
      if (d.protectMove) protectMoveDecs++;
      if (d.phazeMove) phazeMoveDecs++;
      if (d.leechMove) leechMoveDecs++;
      if (d.substituteMove) substituteMoveDecs++;
      if (d.restrictionMove) restrictionMoveDecs++;
      if (d.explosionMove) explosionMoveDecs++;
      if (d.fixedMove) fixedMoveDecs++;
      if (d.batch5Move) batch5MoveDecs++;
      if (d.batch6Move) batch6MoveDecs++;
    }
  }

  fs.writeFileSync(OUT_GOLDEN, goldenLines.join('\n') + '\n');
  console.error(`FILTERED gate: kept ${filteredKept} battles (${decRows} decisions, ${winRows} wins, ${tieRows} ties, ${switchRows} forced-switch, ${doubleRows} double, ${statusMoveDecs} STATUS-MOVE decisions, ${setupMoveDecs} SETUP-MOVE decisions, ${recoveryMoveDecs} RECOVERY-MOVE decisions, ${protectMoveDecs} PROTECT-MOVE decisions, ${phazeMoveDecs} PHAZE-MOVE decisions, ${leechMoveDecs} LEECH-MOVE decisions, ${substituteMoveDecs} SUBSTITUTE-MOVE decisions, ${restrictionMoveDecs} TAUNT/DISABLE-MOVE decisions, ${explosionMoveDecs} EXPLOSION-MOVE decisions, ${fixedMoveDecs} FIXED-DAMAGE-MOVE decisions, ${batch5MoveDecs} BATCH5-MOVE decisions, ${batch6MoveDecs} BATCH6-MOVE decisions), dropped ${filteredDropped} -> ${OUT_GOLDEN}`);
  console.error('  drop reasons: ' + [...dropReasons.entries()].sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k}=${v}`).join(' '));

  if (filteredKept < 50) { console.error('FILTERED GATE: too few kept battles; loosen target or check the filter.'); process.exit(1); }

  // ===========================================================================
  // (2) THE HONEST TAXONOMY — unfiltered sweep: pair ANY teams, run damaging-or-
  //     switch choices, record the ability/item gaps each battle carried (the
  //     ranked remaining-work list). Also emits the unfiltered battles' goldens so
  //     the Rust taxonomy test can replay + classify divergence at the source.
  // ===========================================================================
  const taxRng = mulberry32(MASTER_SEED ^ 0x55555555);
  const taxLines = [];
  taxLines.push('# e2e_fuzz_taxonomy.txt — CAPSTONE honest taxonomy (real teams, UNFILTERED, damaging-or-switch choices).');
  taxLines.push('# The measured coverage + ranked remaining-work list. Does NOT gate cargo test.');
  taxLines.push(`# MASTER_SEED ${MASTER_SEED}  (taxonomy seed ${MASTER_SEED ^ 0x55555555})`);

  const abilityGapCount = new Map();
  const itemGapCount = new Map();
  let cleanBattles = 0; let gappyBattles = 0; let taxBattles = 0;

  // The UNFILTERED sweep RUNS each real-team pair through the sim (damaging-or-switch
  // choices) ONLY to drop empty/errored battles — then ranks each battle's coverage gap
  // by STATIC TEAM COMPOSITION via `classifyTeamsGaps` (which unmodeled ability/item the
  // PAIRED TEAMS CARRY), NOT by the observed first-divergence cause, and BLIND to move
  // choice (status moves / Spikes / Calm Mind are never picked, so never counted).
  // Because real gen3OU teams are saturated with Natural Cure / Torrent / Magnet Pull /
  // berries, essentially every random pair is "gappy" — so the ranked counts below ARE
  // the prioritised remaining-work list (which unmodeled mechanic the MOST real teams
  // carry). We do NOT replay these in Rust (a gappy battle can't bit-for-bit match —
  // its UNMODELED ability/item would desync); the bit-for-bit cross-engine proof is the
  // FILTERED gate (`e2e_fuzz_golden.txt` → `tests/e2e_fuzz_test.rs`).
  for (let t = 0; t < UNFILTERED_TARGET && taxBattles < UNFILTERED_TARGET * 3; t++) {
    const ia = randInt(taxRng, teams.length);
    let ib = randInt(taxRng, teams.length);
    if (ib === ia) ib = (ib + 1) % teams.length;
    const seed = seedFrom((Math.floor(taxRng() * 4294967296) ^ (t * 40503)) >>> 0);
    const chooseSeed = (Math.floor(taxRng() * 4294967296) ^ 0x1b873593) >>> 0;
    let rec;
    try {
      rec = await runBattle(teams[ia].packed, teams[ib].packed, seed, chooseSeed, 'damaging');
    } catch (e) { continue; }
    taxBattles++;
    if (!rec.initSeed || rec.decisions.length === 0) continue;

    const gaps = classifyTeamsGaps(teams[ia].packed, teams[ib].packed);
    const hasGap = gaps.abilities.size > 0 || gaps.items.size > 0;
    if (hasGap) gappyBattles++; else cleanBattles++;
    for (const a of gaps.abilities) abilityGapCount.set(a, (abilityGapCount.get(a) || 0) + 1);
    for (const it of gaps.items) itemGapCount.set(it, (itemGapCount.get(it) || 0) + 1);
  }

  // Rank the gaps by how many battles they blocked.
  const rankAbil = [...abilityGapCount.entries()].sort((a, b) => b[1] - a[1]);
  const rankItem = [...itemGapCount.entries()].sort((a, b) => b[1] - a[1]);

  taxLines.push(`# unfiltered battles run: ${taxBattles}`);
  taxLines.push(`# filter-CLEAN battles (every mon modeled ability+item): ${cleanBattles}`);
  taxLines.push(`# GAPPY battles (>=1 unmodeled ability/item somewhere): ${gappyBattles}`);
  taxLines.push('# ── ranked ABILITY gaps (battles blocked) ──');
  for (const [a, c] of rankAbil) taxLines.push(`ABILITY\t${a}\t${c}`);
  taxLines.push('# ── ranked ITEM gaps (battles blocked) ──');
  for (const [it, c] of rankItem) taxLines.push(`ITEM\t${it}\t${c}`);
  // The ENGINE-side taxonomy: among MODELED-mechanics-only battles (the filtered gate),
  // the port now has ZERO cross-engine divergence — it is bit-for-bit to game-end over
  // all FILTERED_TARGET battles. The former residual-vs-faint-under-weather ordering gap
  // is FIXED in turn.rs (per-handler faintMessages + the cached `pokemon.speed` model).
  taxLines.push('# ── ENGINE divergence cause among MODELED-mechanics battles (the gate) ──');
  taxLines.push('ENGINE_GAP\tnone\t(bit-for-bit; the prior weather residual-vs-faint gap is FIXED — see CLAUDE.md "## E2E capstone")');

  fs.writeFileSync(OUT_TAXONOMY, taxLines.join('\n') + '\n');

  console.error(`TAXONOMY: ${taxBattles} unfiltered battles (${cleanBattles} clean, ${gappyBattles} gappy)`);
  console.error('  top ability gaps: ' + rankAbil.slice(0, 12).map(([a, c]) => `${a}=${c}`).join(' '));
  console.error('  top item gaps:    ' + rankItem.slice(0, 12).map(([it, c]) => `${it}=${c}`).join(' '));
  console.error(`done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  process.exit(0);
}

// ── Module surface (the A/B fuzzer reuses these — ONE source of truth) ───────
// `harness/ab_fuzz.js` requires this file for the modeled-universe predicates
// (isModeledMove / ability / item), the battle driver (runBattle + the TAB golden
// emitter), and the team loader — so the two harnesses can never drift. Running
// this file DIRECTLY (node gen_e2e_fuzz.js) still regenerates the e2e goldens,
// byte-identically (main() only runs under require.main).
module.exports = {
  runBattle, emitBattle, winTok, encodeChoice,
  isModeledMove, isHiddenPower,
  abilityAllowed, itemAllowed, teamFilterClean, loadTeams, classifyTeamsGaps,
  MODELED_ABILITIES, NOOP_ABILITIES, REJECT_ABILITIES, REJECT_SPECIES, REJECT_MOVES, MODELED_ITEMS,
  MODELED_STATUS_MOVES, MODELED_SETUP_MOVES, MODELED_RECOVERY_MOVES,
  MODELED_PROTECT_MOVES, MODELED_HAZARD_MOVES, MODELED_PHAZE_MOVES,
  MODELED_LEECH_MOVES, MODELED_FIXED_DAMAGE_MOVES, MODELED_SUBSTITUTE_MOVES,
  MODELED_RESTRICTION_MOVES, MODELED_RECYCLE_MOVES, MODELED_SKILLSWAP_MOVES, MODELED_BATCH3_MOVES,
  MODELED_CURE_MOVES, MODELED_WEATHER_MOVES, MODELED_STATDROP_MOVES, MODELED_SCREEN_MOVES,
  MODELED_PARTIALTRAP_MOVES,
  mulberry32, randInt, seedFrom, toId,
  FORMAT, dex3,
};

if (require.main === module) {
  main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
}
