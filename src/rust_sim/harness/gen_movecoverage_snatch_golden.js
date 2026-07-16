// gen_movecoverage_snatch_golden.js — Gen-3 SNATCH differential golden (`gen3_snatch_v1`):
// the SOLE remaining unmodeled gen3 status move, which closes 722/722.
//
//   SNATCH — a Dark, category-Status, priority-+4, never-miss `target:self` move that sets
//   the `snatch` singleturn volatile (`duration:1`). While it is up, the NEXT self-targeted
//   `flags.snatch` status move used by the FOE (in gen-3 singles the only eligible victim)
//   is STOLEN: the snatcher executes it in ITS OWN context and the foe's move does nothing.
//   Probe-settled bit-for-bit (`harness/probe_snatch.js`):
//     * the CAST is DRAW-FREE (`|move|U|Snatch|U` + `|-singleturn|U|Snatch`); casting into
//       NOTHING just expires at the next turn-top.
//     * the STEAL emits the foe's `|move|` line STILL, then `|-activate|SNATCHER|move:
//       Snatch|[of] FOE`, then `|move|SNATCHER|Name|SNATCHER|[from] Snatch` + the effect —
//       the snatcher gets the boost/heal/sub/status; the VICTIM spends the stolen move's PP,
//       the SNATCHER only its Snatch PP. The stolen move's OWN native draws fire in the
//       snatcher's context (SwordsDance/Recover/Substitute = 0 extra, Rest = its sleep
//       `random(2,6)`).
//     * SNATCH INTRODUCES ZERO DRAWS OF ITS OWN — the only snatch-attributable draw is the
//       residual duration-handler tie-shuffle a MIRROR draws (both `snatch` volatiles tie at
//       NO_ORDER/subOrder-2; PROBE-VERIFIED 8 vs the both-Splash control's 7).
//     * NON-members pass through: Thunder Wave / Spikes / Wish / Snatch itself carry no
//       flags.snatch (the sim overturned the task's hypotheses).
//
// THE PROOF: drive the OMNISCIENT in-process BattleStream (no server) over CONSTRUCTED
// scenarios that each ISOLATE a branch, capturing initSeed + per-decision seedAfter, each
// active's species/hp/maxhp/fainted/status(+slp-time/tox-stage)/boosts/confusion +
// pokemon_left + CURSE + WISH + SUB-HP + FUTURE-PENDING + ENCORE + PERISH + TRAPPED + first
// mover + winner. The Rust test seeds a BattleState at initSeed and runs `run_full_battle`
// WITHOUT re-seeding. REUSES the batch-6 50-field DEC format (all snatch effects — the
// stolen boost/heal/status/sub-hp on the SNATCHER — land in the existing columns), so
// `movecoverage_snatch_test.rs` shares the batch-6 parser.
//
// Output: tests/vectors/movecoverage_snatch_golden.txt
//
// Run:  node src/rust_sim/harness/gen_movecoverage_snatch_golden.js

'use strict';
const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/movecoverage_snatch_golden.txt');
const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };

function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: opts.gender || 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

function encodeChoice(c) {
  if (!c) return '-';
  const m = c.match(/^move\s+(\d+)$/);
  if (m) return `m${Number(m[1]) - 1}`;
  const s = c.match(/^switch\s+(\d+)$/);
  if (s) return `s${Number(s[1]) - 1}`;
  throw new Error(`unencodable choice ${JSON.stringify(c)}`);
}

function buildSeeds(n) {
  const out = [];
  let x = 0x51ed270b >>> 0;
  const step = () => { x = (Math.imul(x, 1103515245) + 12345) >>> 0; return x & 0xffff; };
  for (let i = 0; i < n; i++) out.push([step() || 1, step() || 1, step() || 1, step() || 1]);
  return out;
}

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

function statusOf(active) {
  const st = (active && active.status) || '';
  let stage = 0;
  if (st === 'tox') stage = active.statusState ? (active.statusState.stage || 0) : 0;
  if (st === 'slp') stage = active.statusState ? (active.statusState.time || 0) : 0;
  return { status: st || '-', stage };
}
function boostsOf(a) {
  const b = a && a.boosts ? a.boosts : {};
  return [b.atk | 0, b.def | 0, b.spa | 0, b.spd | 0, b.spe | 0];
}
function confusionOf(a) { return a && a.volatiles && a.volatiles['confusion'] ? (a.volatiles['confusion'].time | 0) : 0; }
function curseOf(a) { return a && a.volatiles && a.volatiles['curse'] ? 1 : 0; }
function subHpOf(a) { return a && a.volatiles && a.volatiles['substitute'] ? (a.volatiles['substitute'].hp | 0) : 0; }
function encoreOf(a) { return a && a.volatiles && a.volatiles['encore'] ? (a.volatiles['encore'].duration | 0) : 0; }
function perishOf(a) { return a && a.volatiles && a.volatiles['perishsong'] ? (a.volatiles['perishsong'].duration | 0) : 0; }
function trappedOf(a) { return a && a.volatiles && a.volatiles['trapped'] ? 1 : 0; }
function wishOf(side) { const sc = side.slotConditions && side.slotConditions[0]; const w = sc && sc.wish; return w ? (w.duration | 0) : 0; }
function futureOf(side) { const sc = side.slotConditions && side.slotConditions[0]; const f = sc && sc.futuremove; return f ? (f.duration | 0) : 0; }

function snap(side) {
  const a = side.active[0];
  const wish = wishOf(side);
  const future = futureOf(side);
  if (!a) {
    return {
      species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', stage: 0, left: side.pokemonLeft,
      boosts: [0, 0, 0, 0, 0], confusion: 0, curse: 0, wish, subHp: 0, future, encore: 0, perish: 0, trapped: 0,
    };
  }
  const { status, stage } = statusOf(a);
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted, status, stage, left: side.pokemonLeft,
    boosts: boostsOf(a), confusion: confusionOf(a), curse: curseOf(a), wish, subHp: subHpOf(a), future,
    encore: encoreOf(a), perish: perishOf(a), trapped: trappedOf(a),
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

// Snatch-relevant coverage flags scanned between two decision points.
function outcomesSince(log, fromIdx) {
  const out = {
    snStolen: false,   // |-activate|SNATCHER|move: Snatch|[of] FOE
    snExec: false,     // |move|SNATCHER|Name|SNATCHER|[from] Snatch (the stolen move ran)
    snBoost: false,    // a -boost after a [from] Snatch (stolen self-boost)
    snHeal: false,     // a -heal after a [from] Snatch (stolen recover)
    snStatus: false,   // a -status after a [from] Snatch (stolen Rest → slp)
    snSub: false,      // a -start Substitute after a [from] Snatch
    snCastIdle: false, // a Snatch |-singleturn| with NO -activate this window (cast into nothing)
    twPass: false,     // a -status par NOT preceded by a Snatch -activate (Thunder Wave passed through)
    koTurn: false, tie: false,
  };
  let sawSnatchSingleturn = false;
  let sawActivate = false;
  let fromSnatch = false;
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    const tag = p[1];
    if (tag === '-singleturn' && (p[3] || '') === 'Snatch') sawSnatchSingleturn = true;
    if (tag === '-activate' && (p[3] || '') === 'move: Snatch') { out.snStolen = true; sawActivate = true; }
    if (tag === 'move') {
      const attrs = (p[5] || '') + (p[6] || '');
      fromSnatch = attrs.includes('[from] Snatch');
      if (fromSnatch) out.snExec = true;
    }
    if (fromSnatch && tag === '-boost') out.snBoost = true;
    if (fromSnatch && tag === '-heal') out.snHeal = true;
    if (fromSnatch && tag === '-status') out.snStatus = true;
    if (fromSnatch && tag === '-start' && (p[3] || '') === 'Substitute') out.snSub = true;
    if (tag === '-status' && (p[3] || '') === 'par' && !sawActivate) out.twPass = true;
    if (tag === 'faint') out.koTurn = true;
    if (tag === 'tie') out.tie = true;
  }
  if (sawSnatchSingleturn && !sawActivate) out.snCastIdle = true;
  return out;
}

function firstLiveBench(side, battle) {
  const s = battle.sides[side];
  for (let k = 0; k < s.pokemon.length; k++) if (s.pokemon[k] !== s.active[0] && !s.pokemon[k].fainted) return `switch ${k + 1}`;
  return 'pass';
}
function legalMove(side, battle, want) {
  const req = battle.sides[side].activeRequest;
  const moves = req && req.active && req.active[0] ? req.active[0].moves : null;
  if (!moves) return 'move 1';
  const usable = [];
  for (let k = 0; k < moves.length; k++) if (!moves[k].disabled) usable.push(k + 1);
  if (usable.length === 0) return 'move 1';
  return `move ${usable.includes(want) ? want : usable[0]}`;
}
function intentDriver(intent) {
  return (decisionNo, battle, reqState, force) => {
    if (reqState === 'switch') {
      const c = { p1: null, p2: null };
      const r = intent(decisionNo, battle) || {};
      if (force[0]) c.p1 = r.p1Switch ? `switch ${r.p1Switch}` : firstLiveBench(0, battle);
      if (force[1]) c.p2 = r.p2Switch ? `switch ${r.p2Switch}` : firstLiveBench(1, battle);
      return c;
    }
    const r = intent(decisionNo, battle);
    return {
      p1: r.p1Switch ? `switch ${r.p1Switch}` : legalMove(0, battle, r.p1Want),
      p2: r.p2Switch ? `switch ${r.p2Switch}` : legalMove(1, battle, r.p2Want),
    };
  };
}

async function runBattle(sc, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();

  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(sc.p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(sc.p2) })}`);
  for (let i = 0; i < 10; i++) await tick();

  if (sc.inject) {
    const battle = stream.battle;
    for (const inj of sc.inject) {
      if (inj.side !== undefined) {
        const idx = inj.slot !== undefined ? inj.slot : 0;
        const m = idx === 0 ? battle.sides[inj.side].active[0] : battle.sides[inj.side].pokemon[idx];
        if (inj.status) m.setStatus(inj.status, m, null, true);
        if (inj.hp !== undefined) m.hp = inj.hp;
        if (inj.pp) m.moveSlots[inj.pp.moveSlot].pp = inj.pp.val;
      }
    }
  }

  const script = intentDriver(sc.intent);
  const rec = { initSeed: null, decisions: [], winner: null, ended: false, gen: stream.battle.gen };
  let decisionNo = 0, safety = 0;
  while (!stream.battle.ended && safety < 600) {
    safety++;
    const battle = stream.battle;
    const reqState = battle.requestState;
    if (reqState !== 'move' && reqState !== 'switch') { await tick(); continue; }
    const force = forceSwitchTable(battle);
    const seedBefore = battle.prng.getSeed();
    if (decisionNo === 0) rec.initSeed = seedBefore;
    const choices = script(decisionNo, battle, reqState, force);
    if (!choices) break;
    const logLenBefore = log.length;
    if (choices.p1) { try { streams.omniscient.write(`>p1 ${choices.p1}`); } catch (e) {} }
    if (choices.p2) { try { streams.omniscient.write(`>p2 ${choices.p2}`); } catch (e) {} }
    for (let i = 0; i < 20; i++) await tick();
    const seedAfter = battle.prng.getSeed();
    if (seedAfter === seedBefore && log.length === logLenBefore && battle.requestState === reqState) {
      throw new Error(`STALL: choice ${JSON.stringify(choices)} did not advance (scenario ${sc.id}, decision ${decisionNo}).`);
    }
    const outcomes = outcomesSince(log, logLenBefore);
    const first = reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none';
    rec.decisions.push({
      request: reqState, force,
      choiceP1: encodeChoice(choices.p1), choiceP2: encodeChoice(choices.p2),
      seedAfter, p1: snap(battle.sides[0]), p2: snap(battle.sides[1]), firstMover: first, outcomes,
    });
    decisionNo++;
  }
  rec.ended = !!stream.battle.ended;
  rec.winner = stream.battle.winner;
  try { streams.omniscient.destroy(); } catch (e) {}
  return rec;
}

// ── Scenarios ──────────────────────────────────────────────────────────────
function scenarios() {
  const S = [];

  // (A) STEAL A SELF-BOOST (Dragon Dance): a FAST Salamence Snatches; when Tyranitar
  //     Dragon Dances the snatcher STEALS it (Salamence +1 atk / +1 spe), else the foe
  //     is unboosted. Earthquake (SE vs TTar Rock/Dark) grinds to a win. The snatcher's
  //     stolen +1 atk lands in the boost columns; a snatch turn the foe does NOT set up
  //     is a cast-into-nothing (the volatile just expires).
  S.push({
    id: 'sn_steal_dragondance',
    p1: [mon('Salamence', ['snatch', 'earthquake'], { ability: 'Intimidate', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Tyranitar', ['dragondance', 'rockslide'], { evs: { hp: 252, atk: 252 } })],
    intent: (n) => ({ p1Want: (n % 2 === 0) ? 1 : 2, p2Want: (n % 2 === 0) ? 1 : 2 }),
    require: ['snStolen', 'snBoost'],
  });

  // (B) STEAL A RECOVER (Soft-Boiled): the snatcher takes chip from the foe's attack;
  //     when it Snatches while the foe Soft-Boileds, the SNATCHER heals floor(maxhp/2)
  //     (the foe heals nothing). Jolteon Thunderbolt vs Blissey grinds to a win.
  S.push({
    id: 'sn_steal_recover',
    p1: [mon('Jolteon', ['snatch', 'thunderbolt'], { evs: { spa: 252, spe: 252 } })],
    p2: [mon('Blissey', ['softboiled', 'icebeam'], { evs: { hp: 252, def: 252 } })],
    intent: (n, battle) => {
      const a = battle.sides[0].active[0];
      const hurt = a && a.hp < a.maxhp * 0.6;
      // Snatch when hurt (steal the heal onto ourselves); else Thunderbolt.
      return { p1Want: hurt ? 1 : 2, p2Want: (n % 2 === 0) ? 1 : 2 };
    },
    require: ['snStolen', 'snHeal'],
  });

  // (C) STEAL A REST: the snatcher steals the foe's Rest → the SNATCHER sleeps (its own
  //     state; the foe stays awake). The stolen Rest draws its sleep `random(2,6)` in the
  //     snatcher's context (the draw-count teeth — the seed differential). A FAST Gengar
  //     Snatcher (so its cant / move always emits FIRST regardless of sleep → the
  //     first-mover attribution stays unambiguous), grinding Suicune with Thunderbolt. On
  //     turn 0 Gengar Snatches Suicune's Rest → Gengar sleeps; it wakes and Thunderbolts
  //     to a win. (The HEAL amount is pinned separately — MC102, injected low HP.)
  S.push({
    id: 'sn_steal_rest',
    p1: [mon('Gengar', ['snatch', 'thunderbolt'], { ability: 'Levitate', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Suicune', ['rest', 'surf'], { evs: { spa: 252, spe: 252 } })],
    // Inject both low so the turn-0 stolen Rest SUCCEEDS (Rest fails at full HP): Suicune
    // Rests, Gengar Snatches → the SNATCHER (Gengar) sleeps + heals. A FAST snatcher keeps
    // the first-mover attribution unambiguous even while asleep.
    inject: [{ side: 0, hp: 120 }, { side: 1, hp: 160 }],
    intent: (n) => ({ p1Want: (n === 0) ? 1 : 2, p2Want: (n === 0) ? 1 : 2 }),
    require: ['snStolen', 'snStatus'],
  });

  // (D) STEAL A SUBSTITUTE: the snatcher steals the foe's Substitute → the SNATCHER
  //     builds the sub (pays floor(maxhp/4) of ITS OWN hp). Snorlax Snatches when the
  //     foe subs, Body Slams else.
  S.push({
    id: 'sn_steal_substitute',
    p1: [mon('Snorlax', ['snatch', 'bodyslam'], { evs: { hp: 252, atk: 252 } })],
    p2: [mon('Gengar', ['substitute', 'shadowball'], { ability: 'Levitate', evs: { hp: 252, spa: 252, spe: 252 } })],
    intent: (n, battle) => {
      const foe = battle.sides[1].active[0];
      const hasSub = foe && foe.volatiles && foe.volatiles['substitute'];
      const mySub = battle.sides[0].active[0] && battle.sides[0].active[0].volatiles['substitute'];
      // Snatch when the foe has no sub yet (steal the sub build) and we don't have one.
      return { p1Want: (!hasSub && !mySub && n % 2 === 0) ? 1 : 2, p2Want: (!hasSub) ? 1 : 2 };
    },
    require: ['snStolen', 'snSub'],
  });

  // (E) NON-SNATCHABLE + CAST-INTO-NOTHING: Thunder Wave is NOT snatchable (no
  //     flags.snatch) → the snatcher is paralyzed normally (twPass); and on turns the
  //     foe attacks, the snatch volatile just expires (snCastIdle). The seed differential
  //     proves the port does NOT wrongly steal Thunder Wave.
  S.push({
    id: 'sn_non_snatchable_twave',
    p1: [mon('Umbreon', ['snatch', 'bite'], { evs: { hp: 252, atk: 252 } })],
    p2: [mon('Jolteon', ['thunderwave', 'thunderbolt'], { evs: { spa: 252, spe: 252 } })],
    intent: (n, battle) => {
      const a = battle.sides[0].active[0];
      const par = a && a.status === 'par';
      // Snatch early (into TWave → NOT stolen → paralyzed), then Bite grinds.
      return { p1Want: (!par && n < 2) ? 1 : 2, p2Want: (n === 0) ? 1 : 2 };
    },
    require: ['twPass', 'snCastIdle'],
  });

  // (F) THE SNATCH MIRROR (the residual-duration-handler CRUX): two EQUAL-speed Umbreon
  //     both Snatch on the same turn → both `snatch` volatiles register the residual
  //     duration handler → they TIE → ONE residual tie-shuffle draw (PROBE 8 vs 7). A
  //     wrong model (no handler) desyncs the SEED on every mirror turn. Neither steals
  //     (Snatch is not snatchable). They Bite each other to a decision on off turns.
  S.push({
    id: 'sn_mirror_residual_tie',
    p1: [mon('Umbreon', ['snatch', 'bite'], { evs: { hp: 252, atk: 252 } })],
    p2: [mon('Umbreon', ['snatch', 'bite'], { evs: { hp: 252, atk: 252 } })],
    intent: (n) => ({ p1Want: (n % 2 === 0) ? 1 : 2, p2Want: (n % 2 === 0) ? 1 : 2 }),
    require: ['snCastIdle'],
  });

  // (G) SNATCH INTO A REAL BATTLE: a fuller 2-mon battle with a switch + snatch mixed in.
  //     Salamence Snatches Metagross's Agility (steal +2 spe), pivots, and grinds to a
  //     win — the composition of the steal with switching/faints/residuals.
  S.push({
    id: 'sn_into_real_battle',
    p1: [mon('Salamence', ['snatch', 'earthquake'], { ability: 'Intimidate', evs: { atk: 252, spe: 252 } }),
         mon('Starmie', ['surf', 'recover'], { ability: 'Natural Cure', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Metagross', ['agility', 'meteormash'], { ability: 'Clear Body', evs: { hp: 252, atk: 252 } }),
         mon('Snorlax', ['bodyslam', 'rest'], { evs: { hp: 252, atk: 252 } })],
    intent: (n, battle) => {
      const me = battle.sides[0].active[0];
      const meMence = me && me.species.name === 'Salamence';
      const bench = battle.sides[0].pokemon.find((p) => p !== me && !p.fainted);
      // Turn 0: Snatch (steal Metagross's Agility if it sets up). Then EQ (SE vs
      // Metagross Steel/Psychic) to grind. Pivot to a live Starmie ONCE, when Salamence
      // is worn below half (state-based, so the decision counter's forced-switch
      // sub-boundaries never mis-fire the switch).
      if (meMence && me.hp < me.maxhp * 0.5 && bench) return { p1Switch: battle.sides[0].pokemon.indexOf(bench) + 1, p2Want: 2 };
      const p1Want = (n === 0 && meMence) ? 1 : 2;
      const p2Want = (n === 0) ? 1 : 2;
      return { p1Want, p2Want };
    },
    require: ['snStolen'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# movecoverage_snatch_golden.txt — Gen-3 SNATCH full-battle golden (`gen3_snatch_v1`).');
  lines.push('# Per-decision STATE(+status+slp-time+boosts+HP+SUB-HP+ENCORE+PERISH+TRAPPED)+SEED');
  lines.push('# +first-mover differential to GAME-END. SNATCH: the sole remaining unmodeled gen3');
  lines.push('#   status move — steals the next foe self-targeted flags.snatch status move.');
  lines.push('# REUSES the batch-6 50-field DEC format (the stolen boost/heal/status/sub land in the');
  lines.push('#   existing columns) so movecoverage_snatch_test.rs shares the batch-6 parser.');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INJECT <id>  <json array of {side?,slot?,status?,hp?,pp:{moveSlot,val}?}>  ([] if none)');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status stage left atk def spa spd spe confusion) p2(...) first \\');
  lines.push('#        p1Curse p1Wish p1SubHp  p2Curse p2Wish p2SubHp  p1Future p2Future \\');
  lines.push('#        p1Encore p1Perish p1Trapped  p2Encore p2Perish p2Trapped');
  lines.push('# END   <id>  <seed>  <ended:0|1>  <winner:p1|p2|tie|none>');

  const S = scenarios();
  const failures = [];
  let decRows = 0, winRows = 0, tieRows = 0;
  const corpus = {};
  const scenSeen = {};

  for (const sc of S) {
    lines.push(`SCEN\t${sc.id}`);
    lines.push(`TEAM\t${sc.id}\tp1\t${Teams.pack(sc.p1)}`);
    lines.push(`TEAM\t${sc.id}\tp2\t${Teams.pack(sc.p2)}`);
    lines.push(`INJECT\t${sc.id}\t${JSON.stringify(sc.inject || [])}`);
    scenSeen[sc.id] = {};

    let scenDecs = 0;
    for (const seed of seeds) {
      let rec;
      try { rec = await runBattle(sc, seed); } catch (e) { failures.push(`${sc.id} seed ${seed}: ${e.message}`); continue; }
      if (rec.gen !== 3) { failures.push(`${sc.id}: expected gen 3, got ${rec.gen}`); break; }
      if (!rec.initSeed || rec.decisions.length === 0) { failures.push(`${sc.id} seed ${seed}: no decisions`); continue; }
      const seedStr = seed.join(',');
      lines.push(['INIT', sc.id, rec.initSeed, seedStr].join('\t'));

      rec.decisions.forEach((d) => {
        for (const k of Object.keys(d.outcomes)) if (d.outcomes[k]) { scenSeen[sc.id][k] = true; corpus[k] = (corpus[k] || 0) + 1; }
      });

      rec.decisions.forEach((d) => {
        const sp = (s) => [
          s.species, s.hp, s.maxhp, s.fainted ? 1 : 0, s.status, s.stage, s.left,
          s.boosts[0], s.boosts[1], s.boosts[2], s.boosts[3], s.boosts[4], s.confusion,
        ].join('\t');
        lines.push([
          'DEC', sc.id, seedStr, d.request, d.force[0] ? 1 : 0, d.force[1] ? 1 : 0,
          d.choiceP1, d.choiceP2, d.seedAfter,
          sp(d.p1), sp(d.p2), d.firstMover,
          d.p1.curse, d.p1.wish, d.p1.subHp, d.p2.curse, d.p2.wish, d.p2.subHp,
          d.p1.future, d.p2.future,
          d.p1.encore, d.p1.perish, d.p1.trapped, d.p2.encore, d.p2.perish, d.p2.trapped,
        ].join('\t'));
        decRows++; scenDecs++;
      });

      let winTok = 'none';
      if (rec.ended) {
        if (rec.winner === 'P1') winTok = 'p1';
        else if (rec.winner === 'P2') winTok = 'p2';
        else if (rec.winner === '') winTok = 'tie';
      }
      if (winTok === 'p1' || winTok === 'p2') winRows++;
      if (winTok === 'tie') tieRows++;
      lines.push(['END', sc.id, seedStr, rec.ended ? 1 : 0, winTok].join('\t'));
    }
    if (scenDecs === 0) failures.push(`${sc.id}: produced NO decision rows`);
    for (const need of (sc.require || [])) if (!scenSeen[sc.id][need]) failures.push(`${sc.id}: REQUIRED branch ${need} never realized`);
    for (const bad of (sc.forbid || [])) if (scenSeen[sc.id][bad]) failures.push(`${sc.id}: FORBIDDEN branch ${bad} realized`);
  }

  if (failures.length) {
    console.error('SNATCH GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }

  const need = (label, key, min) => { const n = corpus[key] || 0; if (n < min) { console.error(`SNATCH GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('snatch steals (-activate)', 'snStolen', 60);
  need('stolen-move executions', 'snExec', 60);
  need('stolen self-boosts', 'snBoost', 30);
  need('stolen recovers', 'snHeal', 20);
  need('stolen rests (slp)', 'snStatus', 15);
  need('stolen substitutes', 'snSub', 15);
  need('cast-into-nothing turns', 'snCastIdle', 30);
  need('thunder-wave pass-throughs', 'twPass', 15);
  if (winRows < 30) { console.error(`SNATCH GOLDEN: too few WIN rows (${winRows} < 30)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `movecoverage snatch golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `branches: ${Object.keys(corpus).sort().map((k) => `${k}=${corpus[k]}`).join(' ')} -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
