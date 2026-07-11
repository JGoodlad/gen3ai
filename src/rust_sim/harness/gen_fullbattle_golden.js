// gen_fullbattle_golden.js — Gen-3 FULL-BATTLE (move + switch + replacement →
// win/loss) differential harness.
//
// Extends harness/gen_battle_golden.js from a no-switch loop that STOPS at the
// first faint to a battle that plays to GAME-END through:
//   (1) voluntary SWITCHES (a side switches instead of moving — switch sorts
//       BEFORE the move, the two-switch speed tie draws the action-order shuffle);
//   (2) the mid-battle switch-in (the entrant's ability Start — Intimidate/weather
//       — draw-free; gen3 runSwitch draws NOTHING);
//   (3) POST-FAINT replacement (single + DOUBLE): when a mon faints the sim
//       pauses with a forceSwitch request; we submit the scripted replacement, the
//       turn resumes, and run continues PAST faints until a side is out of mons.
//
// THE PROOF (the CRUX): drive the OMNISCIENT in-process BattleStream (no server)
// a FULL battle, capturing the running PRNG seed BEFORE the first decision
// (`initSeed` for the Rust to seed once) and AFTER each DECISION BOUNDARY (each
// `move` turn AND each forced-`switch` replacement sub-step). The Rust test seeds
// a BattleState at the init seed and runs `run_full_battle` WITHOUT re-seeding —
// so the post-decision seed must match the sim's `seedAfter` at EVERY boundary,
// across the new switch-phase draw sites (the two-switch action-order shuffle, the
// around-switch eachEvent('Update') shuffles, the double-replacement insertChoice
// splice, and the win-decides-no-QuickClaw rule). An EXACT cross-decision seed
// match to game-end + the final winner is the draw-ORDER+COUNT proof over the full
// battle (a single extra/missing/mis-ordered switch-phase draw desyncs the LCG).
//
// SECONDARY-FREE MOVES ONLY (per the adversarial correction): a secondary-effect
// move (e.g. Thunderbolt's 10% paralysis) draws an EXTRA random(100) the Rust port
// DEFERS — so every move used here is secondary-free (Earthquake/Surf/Tackle/
// Body Slam? NO — Body Slam has a 30% para secondary; use Earthquake/Surf/
// Megahorn/Tackle/Hydro Pump/Explosion/Swift). Bulky defenders + strong attackers
// so battles reach a real win across the 6-mon teams.
//
// SEEDING: we capture `initSeed` at the boundary just BEFORE the first decision
// (turn 1's pre-choice seed), sidestepping the >start gender-sample + turn-1
// QuickClaw draws this bounded step omits — exactly like gen_battle_golden.js.
//
// Output: tests/vectors/fullbattle_golden.txt, TAB-delimited, std-parseable.
//
// Run:  node src/rust_sim/harness/gen_fullbattle_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/fullbattle_golden.txt');
const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };

function mon(species, moves, opts = {}) {
  return {
    species,
    item: opts.item || '',
    ability: opts.ability || 'No Ability',
    moves,
    evs: { ...EV0, ...(opts.evs || {}) },
    ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious',
    level: opts.level || 100,
    gender: opts.gender || 'N',
  };
}

function tick() { return new Promise((r) => setTimeout(r, 0)); }

// Encode a submitted choice string ('move K' / 'switch N' / null) into the compact
// golden token: 'm<K-1>' (0-based move slot) | 's<N-1>' (0-based team slot) | '-'.
function encodeChoice(c) {
  if (!c) return '-';
  const m = c.match(/^move\s+(\d+)$/);
  if (m) return `m${Number(m[1]) - 1}`;
  const s = c.match(/^switch\s+(\d+)$/);
  if (s) return `s${Number(s[1]) - 1}`;
  throw new Error(`unencodable choice ${JSON.stringify(c)}`);
}

// A well-spread deterministic gen5 seed pool (same generator the other goldens
// use so the corpus is reproducible).
function buildSeeds(n) {
  const out = [];
  let x = 0x6f2c1a3b >>> 0;
  const step = () => { x = (Math.imul(x, 1103515245) + 12345) >>> 0; return x & 0xffff; };
  for (let i = 0; i < n; i++) out.push([step() || 1, step() || 1, step() || 1, step() || 1]);
  return out;
}

// First mover this turn (the FIRST |move|/|switch| after the latest decision).
function firstMoverSince(log, fromIdx) {
  for (let i = fromIdx; i < log.length; i++) {
    const parts = log[i].split('|');
    if ((parts[1] === 'move' || parts[1] === 'switch') && parts.length >= 3) {
      const actor = parts[2].trim();
      if (actor.startsWith('p1a:')) return 'p1';
      if (actor.startsWith('p2a:')) return 'p2';
    }
  }
  return 'none';
}

function statusOf(active) {
  const st = (active && active.status) || '';
  let stage = 0;
  if (st === 'tox') stage = active.statusState ? (active.statusState.stage || 0) : 0;
  return { status: st || '-', stage };
}

// A snapshot of one side's ACTIVE mon + side pokemonLeft.
function snap(side) {
  const a = side.active[0];
  if (!a) return { species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', stage: 0, left: side.pokemonLeft };
  const { status, stage } = statusOf(a);
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status, stage, left: side.pokemonLeft,
  };
}

// The forceSwitch table for the current request, per side ('move' → [false,false]).
function forceSwitchTable(battle) {
  const out = [false, false];
  if (battle.requestState !== 'switch') return out;
  for (let i = 0; i < 2; i++) {
    const req = battle.sides[i].activeRequest;
    if (req && req.forceSwitch && req.forceSwitch[0]) out[i] = true;
  }
  return out;
}

// Run ONE scenario at one seed to game-end (or the script's end). Records each
// DECISION BOUNDARY: the request kind, the per-side choices submitted, the post
// state + seed. `sc.script(turn, battle)` returns the per-side choices for the
// CURRENT request: { p1: 'move K'|'switch N'|null, p2: ... } — null = no choice
// for that side this request (the off-side of a single forced switch).
async function runBattle(sc, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();

  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(sc.p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(sc.p2) })}`);
  for (let i = 0; i < 10; i++) await tick();

  // A FRESH script closure per battle (so the plan index resets per seed — a shared
  // closure would carry its index across seeds and start mid-plan).
  const script = sc.makeScript();

  const rec = { initSeed: null, decisions: [], winner: null, ended: false, gen: stream.battle.gen };

  let decisionNo = 0;
  let safety = 0;
  // Each loop iteration answers ONE request (move OR forced switch).
  while (!stream.battle.ended && safety < 200) {
    safety++;
    const battle = stream.battle;
    const reqState = battle.requestState; // 'move' | 'switch' | 'teampreview' | ''
    if (reqState !== 'move' && reqState !== 'switch') {
      // No open request (between commits) — let the sim settle.
      await tick();
      continue;
    }
    const force = forceSwitchTable(battle);
    const seedBefore = battle.prng.getSeed();
    if (decisionNo === 0) rec.initSeed = seedBefore;

    // Ask the scenario for the choices for THIS request.
    const choices = script(decisionNo, battle, reqState, force);
    if (!choices) break; // scenario signals "stop" (no more scripted choices)

    const logLenBefore = log.length;
    if (choices.p1) { try { streams.omniscient.write(`>p1 ${choices.p1}`); } catch (e) {} }
    if (choices.p2) { try { streams.omniscient.write(`>p2 ${choices.p2}`); } catch (e) {} }
    for (let i = 0; i < 14; i++) await tick();

    const seedAfter = battle.prng.getSeed();
    rec.decisions.push({
      request: reqState,
      force,
      // The EXACT choice submitted per side (the faithful script the Rust replays —
      // NO species-based reconstruction, so duplicate-species teams are unambiguous).
      // Format: 'm K' (move slot, 1-based) | 's N' (switch team slot, 1-based) | '-'.
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1: snap(battle.sides[0]),
      p2: snap(battle.sides[1]),
      firstMover: reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none',
    });
    decisionNo++;
  }

  rec.ended = !!stream.battle.ended;
  rec.winner = stream.battle.winner; // '' on a tie, undefined while ongoing, name on win
  try { streams.omniscient.destroy(); } catch (e) {}
  return rec;
}

// ── Scenarios ──────────────────────────────────────────────────────────────
// Each is { id, p1[], p2[], script(turn, battle, reqState, force) }. The script
// must answer BOTH a 'move' request (both sides) and any 'switch' request (the
// flagged side). All moves are SECONDARY-FREE.

function scenarios() {
  const S = [];

  // Helper: a FACTORY for a fixed per-turn move/switch plan. `plan` is an array of
  // per-MOVE-request {p1, p2}; for a forced-switch request we use `onForce(side)` to
  // pick the replacement. Returns `makeScript()` → a FRESH stateful closure (the
  // move-plan index resets per battle).
  const fromPlan = (plan, onForce) => () => {
    let i = 0;
    return (decisionNo, battle, reqState, force) => {
      if (reqState === 'switch') {
        // Forced replacement: choose for each flagged side.
        const c = { p1: null, p2: null };
        if (force[0]) c.p1 = onForce(0, battle);
        if (force[1]) c.p2 = onForce(1, battle);
        return c;
      }
      // A 'move' request: pull the next plan entry (default to "attack slot 0").
      const entry = plan[i] || { p1: 'move 1', p2: 'move 1' };
      i++;
      return entry;
    };
  };

  // Pick the first non-active, non-fainted bench slot (1-based) for a replacement.
  const firstLiveBench = (side, battle) => {
    const s = battle.sides[side];
    for (let k = 0; k < s.pokemon.length; k++) {
      const p = s.pokemon[k];
      if (p !== s.active[0] && !p.fainted) return `switch ${k + 1}`;
    }
    return 'pass';
  };

  // --- (1) BOTH SWITCH, distinct outgoing speeds, then both attack to a win. ---
  //   p1 Snorlax(spe 30)→Suicune, p2 Skarmory(spe 70)→Tyranitar. Distinct → the
  //   action-order switch sort draws nothing. Then trade attacks.
  S.push({
    id: 'both_switch_distinct',
    p1: [mon('Snorlax', ['earthquake', 'tackle'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
         mon('Suicune', ['surf', 'icebeam'], { item: 'Leftovers', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    p2: [mon('Skarmory', ['drillpeck', 'tackle'], { item: 'Leftovers', nature: 'Impish', evs: { hp: 252, def: 252 } }),
         mon('Tyranitar', ['earthquake', 'rockslide'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'switch 2', p2: 'switch 2' }, // turn 1: both switch (distinct speed)
       { p1: 'move 1', p2: 'move 1' },     // turn 2+: Suicune Surf vs Tyranitar EQ
      ],
      (side, battle) => firstLiveBench(side, battle)),
  });

  // --- (2) BOTH SWITCH at a SPEED TIE (the action-order tie-shuffle + the
  //   around-switch eachEvent shuffles fire), then a hard-hitting NON-IMMUNE MIRROR
  //   that actually KOs to a win. Two identical Snorlax leads switch to two
  //   identical Choice-Band Tyranitar (equal speed; Earthquake is NEUTRAL on
  //   Rock/Dark — real damage, no Flying immunity — so the mirror trades to a KO,
  //   then the loser replaces with its Snorlax and the grind decides a winner). ---
  S.push({
    id: 'both_switch_tie',
    p1: [mon('Snorlax', ['earthquake', 'tackle'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
         mon('Tyranitar', ['earthquake', 'crunch'], { item: 'Choice Band', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['earthquake', 'tackle'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
         mon('Tyranitar', ['earthquake', 'crunch'], { item: 'Choice Band', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'switch 2', p2: 'switch 2' }, // tie switch into the Tyranitar mirror
       { p1: 'move 1', p2: 'move 1' },     // EQ mirror (tie, NEUTRAL) — trades to a KO
      ],
      (side, battle) => firstLiveBench(side, battle)),
  });

  // --- (3) SWITCH vs MOVE: p1 switches (slow), p2 attacks (fast). The switch
  //   resolves FIRST (order 103 < 200). Earthquake is secondary-free. ---
  S.push({
    id: 'switch_vs_move',
    p1: [mon('Snorlax', ['earthquake', 'tackle'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
         mon('Skarmory', ['drillpeck', 'tackle'], { item: 'Leftovers', nature: 'Impish', evs: { hp: 252, def: 252 } })],
    p2: [mon('Jolteon', ['surf', 'swift'], { item: 'Leftovers', nature: 'Timid', evs: { hp: 4, spa: 252, spe: 252 } }),
         mon('Suicune', ['surf', 'icebeam'], { item: 'Leftovers', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'switch 2', p2: 'move 1' }, // p1 switches Snorlax→Skarmory, p2 Surf
       { p1: 'move 1', p2: 'move 1' },
      ],
      (side, battle) => firstLiveBench(side, battle)),
  });

  // --- (4) POST-FAINT single replacement → continue → WIN. A fast strong attacker
  //   OHKOs the opp lead (secondary-free Earthquake), opp replaces, repeat until
  //   the opp is out of mons (a real |win|). p1 Aerodactyl (fast) EQ sweeps two
  //   frail p2 mons. ---
  S.push({
    id: 'post_faint_sweep_win',
    p1: [mon('Aerodactyl', ['earthquake', 'rockslide'], { item: 'Choice Band', ability: 'Rock Head', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Jolteon', ['swift', 'surf'], { nature: 'Timid', evs: { spa: 252, spe: 252 } }),
         mon('Gengar', ['swift', 'surf'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 1', p2: 'move 1' }], // turn 1 (then attacks every later move req)
      (side, battle) => firstLiveBench(side, battle)),
  });

  // --- (5) POST-FAINT, the FAINTER replaces (p1's mon faints), p1 sends a fresh
  //   mon, the battle continues. p2 strong, p1 frail lead + bulky backup. ---
  S.push({
    id: 'post_faint_loser_replaces',
    p1: [mon('Jolteon', ['swift', 'surf'], { nature: 'Timid', evs: { spa: 252, spe: 252 } }),
         mon('Snorlax', ['earthquake', 'bodyslam'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Aerodactyl', ['earthquake', 'rockslide'], { item: 'Choice Band', ability: 'Rock Head', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 1', p2: 'move 1' }], // p2 OHKOs Jolteon → p1 forced-replaces
      (side, battle) => firstLiveBench(side, battle)),
  });

  // --- (6) DOUBLE FAINT replacement: two equal-speed frail mons Explosion each
  //   other (secondary-free, self-KO), BOTH replace the same turn (the
  //   insertChoice splice + the instaswitch-ordering tie shuffle), then continue. ---
  S.push({
    id: 'double_faint_replace',
    p1: [mon('Electrode', ['explosion', 'thunderbolt'], { nature: 'Hasty', evs: { atk: 252, spe: 252 } }),
         mon('Snorlax', ['earthquake', 'bodyslam'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Electrode', ['explosion', 'thunderbolt'], { nature: 'Hasty', evs: { atk: 252, spe: 252 } }),
         mon('Snorlax', ['earthquake', 'bodyslam'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 1', p2: 'move 1' }], // double Explosion KO → both replace
      (side, battle) => firstLiveBench(side, battle)),
  });

  // --- (7b) LAST-MON DOUBLE-KO → gen-3 TIE. Two single-mon teams of equal-speed
  //   Electrode that Explosion each other → both pokemonLeft hit 0 the SAME faint
  //   protocol → checkWin's `every(!pokemonLeft)` → gen3 win(null) = TIE. The
  //   deciding faint draws NO trailing Quick Claw. ---
  S.push({
    id: 'last_mon_double_ko_tie',
    p1: [mon('Electrode', ['explosion', 'thunderbolt'], { nature: 'Hasty', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Electrode', ['explosion', 'thunderbolt'], { nature: 'Hasty', evs: { atk: 252, spe: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 1', p2: 'move 1' }], // double Explosion, both last mons → tie
      (side, battle) => firstLiveBench(side, battle)),
  });

  // --- (7) A LONGER battle to win with mixed switches + faints: p1 (Suicune +
  //   Snorlax) vs p2 (Tyranitar lead → frail backup). Switch turn 1, then grind. ---
  S.push({
    id: 'mixed_switch_grind_win',
    p1: [mon('Suicune', ['surf', 'icebeam'], { item: 'Leftovers', nature: 'Modest', evs: { hp: 252, spa: 252 } }),
         mon('Snorlax', ['earthquake', 'bodyslam'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Magikarp', ['tackle', 'splash'], { nature: 'Adamant', evs: { atk: 252, spe: 252 } }),
         mon('Magikarp', ['tackle', 'splash'], { nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'switch 2', p2: 'move 1' }, // p1 pivots to Snorlax, p2 flails
       { p1: 'move 1', p2: 'move 1' },   // Snorlax EQ sweeps the two Magikarp
      ],
      (side, battle) => firstLiveBench(side, battle)),
  });

  return S;
}

const STATUS_TOKENS = { '-': 0, brn: 1, par: 2, slp: 3, frz: 4, psn: 5, tox: 6 };

async function main() {
  const seeds = buildSeeds(50);
  const lines = [];
  lines.push('# fullbattle_golden.txt — Gen-3 FULL-BATTLE (move+switch+replacement→win) golden.');
  lines.push('# Per-decision-boundary STATE+SEED differential to GAME-END.');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>   (the pre-FIRST-decision seed; the Rust seeds here)');
  lines.push('# DEC   <id>  <seed>  <decisionNo>  <request:move|switch>  <forceP1> <forceP2>  <choiceP1> <choiceP2>  <m,n,o,p seedAfter>  \\');
  lines.push('#        p1(species hp max fnt status stage left)  p2(species hp max fnt status stage left)  first_mover');
  lines.push('#   choice token: m<K> = move slot K (0-based) | s<N> = switch team slot N (0-based) | - = no choice');
  lines.push('# END   <id>  <seed>  <ended:0|1>  <winner:p1|p2|tie|none>');
  lines.push('#   request=move consumes BOTH sides; request=switch consumes the flagged side(s).');
  lines.push('#   seedAfter is asserted EXACTLY at every boundary; winner asserted at game-end.');

  const S = scenarios();
  const failures = [];
  let decRows = 0;
  let winRows = 0;
  let tieRows = 0;
  let switchReqRows = 0;
  let doubleReplaceRows = 0;

  for (const sc of S) {
    lines.push(`SCEN\t${sc.id}`);
    lines.push(`TEAM\t${sc.id}\tp1\t${Teams.pack(sc.p1)}`);
    lines.push(`TEAM\t${sc.id}\tp2\t${Teams.pack(sc.p2)}`);

    let scenDecs = 0;
    for (const seed of seeds) {
      let rec;
      try { rec = await runBattle(sc, seed); } catch (e) { failures.push(`${sc.id} seed ${seed}: ${e.message}`); continue; }
      if (rec.gen !== 3) { failures.push(`${sc.id}: expected gen 3, got ${rec.gen}`); break; }
      if (!rec.initSeed || rec.decisions.length === 0) {
        failures.push(`${sc.id} seed ${seed}: no recorded decisions`);
        continue;
      }
      const seedStr = seed.join(',');
      lines.push(['INIT', sc.id, rec.initSeed, seedStr].join('\t'));

      rec.decisions.forEach((d, di) => {
        const sp = (s) => [s.species, s.hp, s.maxhp, s.fainted ? 1 : 0, s.status, s.stage, s.left].join('\t');
        lines.push([
          'DEC', sc.id, seedStr, di, d.request, d.force[0] ? 1 : 0, d.force[1] ? 1 : 0,
          d.choiceP1, d.choiceP2, d.seedAfter,
          sp(d.p1), sp(d.p2), d.firstMover,
        ].join('\t'));
        decRows++;
        scenDecs++;
        if (d.request === 'switch') switchReqRows++;
        if (d.request === 'switch' && d.force[0] && d.force[1]) doubleReplaceRows++;
      });

      // The winner token: '' is a tie, a name is a win, undefined is "not ended".
      let winTok = 'none';
      if (rec.ended) {
        if (rec.winner === '' || rec.winner === null || rec.winner === undefined && rec.ended) {
          // gen3 tie → winner '' (win(null)); but undefined-while-ended shouldn't happen.
          winTok = (rec.winner === '' ) ? 'tie' : 'none';
        }
        if (rec.winner === 'P1') winTok = 'p1';
        else if (rec.winner === 'P2') winTok = 'p2';
        else if (rec.winner === '') winTok = 'tie';
      }
      if (winTok === 'p1' || winTok === 'p2') winRows++;
      if (winTok === 'tie') tieRows++;
      lines.push(['END', sc.id, seedStr, rec.ended ? 1 : 0, winTok].join('\t'));
    }
    if (scenDecs === 0) failures.push(`${sc.id}: produced NO decision rows`);
  }

  if (failures.length) {
    console.error('FULLBATTLE GOLDEN FAILURES:\n  ' + failures.slice(0, 30).join('\n  '));
    process.exit(1);
  }
  if (winRows < 50) { console.error(`FULLBATTLE GOLDEN: too few WIN rows (${winRows}); expected a real game-end across seeds.`); process.exit(1); }
  if (switchReqRows < 50) { console.error(`FULLBATTLE GOLDEN: too few forced-switch decision rows (${switchReqRows}).`); process.exit(1); }
  if (doubleReplaceRows < 10) { console.error(`FULLBATTLE GOLDEN: too few DOUBLE-replacement rows (${doubleReplaceRows}).`); process.exit(1); }
  void STATUS_TOKENS;

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(`fullbattle golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} win + ${tieRows} tie ends, ${switchReqRows} forced-switch reqs (${doubleReplaceRows} double) -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
