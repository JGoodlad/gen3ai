// gen_explosion_golden.js — Gen-3 EXPLOSION / SELF-DESTRUCT differential golden.
//
// Extends harness/gen_substitute_golden.js (the per-decision STATE+STATUS+SPIKES-LAYERS+
// SUB-HP+BOOSTS+CONFUSION+SEED+winner full-battle differential) to the mechanic this
// step adds: gen-3 **Explosion / Self-Destruct** — a Normal PHYSICAL damaging move (BP
// 250 / 200 gen3) that HALVES the target's Def and faints the USER as part of the move.
//
// THE DRAW MODEL (verified bit-for-bit vs the omniscient sim's PRNG probe
// `harness/probe_explosion_rng.js`):
//
//   * Explosion is an ORDINARY damaging move: it draws ACCURACY (`randomChance(100,100)`
//     — gen-3 Explosion accuracy is 100, so it ALWAYS passes but STILL DRAWS), then CRIT
//     (`randomChance(1, critMult)`), then DAMAGE (`random(16)`) — the SAME three draws as
//     any damaging move. It has NO secondary (no trailing `random(100)`).
//   * THE SELF-KO IS DRAW-FREE and UNCONDITIONAL and happens BEFORE the hit: gen-3
//     `useMoveInner` (battle-actions.ts:501-503, `gen !== 4 && selfdestruct === 'always'`)
//     calls `this.battle.faint(pokemon)` — zeroing the user's HP + queuing its faint —
//     BEFORE `trySpreadMoveHit`. So the USER FAINTS REGARDLESS of the hit outcome:
//       - a NORMAL hit: the foe takes big def-halved damage; the user faints.
//       - into a SUBSTITUTE: the damage hits the SUB (breaks it, no carry); the user faints.
//       - into a PROTECT: the move is BLOCKED (no foe damage); the user STILL faints.
//       - into a GHOST (Normal-immune): no damage; the user STILL faints.
//       - a MISS: gen-3 Explosion accuracy is 100 → never a self-accuracy miss (but a
//         hypothetical miss would STILL faint the user — the faint precedes the hit).
//   * A MUTUAL Explosion (both last mons) is a true DOUBLE-FAINT gen-3 TIE.
//   * An Explosion that KOs the foe AND faints the user forces a DOUBLE replacement (both
//     sides pick a new mon); an Explosion into a still-alive foe forces a SINGLE
//     replacement (only the user). The deciding faint draws NO trailing Quick Claw.
//
// The self-KO changes `pokemon_left` + can end the battle / force replacements, which
// affects the REST of the turn's draws (a KO'd user's queued nothing; no Quick Claw on a
// deciding faint), but adds NO draw of its own.
//
// THE PROOF: drive the OMNISCIENT in-process BattleStream (no server) over CONSTRUCTED
// scenarios that each ISOLATE an edge, capturing the running PRNG seed BEFORE the first
// decision (`initSeed`) and AFTER each DECISION BOUNDARY, plus each active's species/hp/
// maxhp/FAINTED/status + boosts + confusion + pokemon_left + per-side SPIKES LAYERS + the
// per-side SUBSTITUTE HP (to prove the sub-break edge) + first mover + winner. The Rust
// test seeds a BattleState at the init seed and runs `run_full_battle` WITHOUT re-seeding
// — so the post-decision seed must match at EVERY boundary AND the user's FAINTED flag +
// the foe's HP/sub-HP must match. A wrong self-KO placement / a missing-or-extra draw → a
// SEED desync; a wrong faint / block / immunity → a FAINTED / HP / sub-HP desync.
//
// This golden reuses the SUBSTITUTE golden's TAB shape verbatim (SCEN/TEAM/INIT/DEC/END
// with the 50-field DEC carrying subHp), so `explosion_test.rs` reuses that parser.
//
// FAIL-LOUD: each scenario declares the EDGE it must realize (a plain self-KO, a sub-break
// self-KO, a protect-blocked self-KO, a ghost-immune self-KO, a double-faint tie, a
// KO-forces-replacement); generation aborts if the sim run did NOT realize it.
//
// Output: tests/vectors/explosion_golden.txt
//
// Run:  node src/rust_sim/harness/gen_explosion_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/explosion_golden.txt');
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
  let x = 0x6d2b79f5 >>> 0;
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

function confusionOf(a) {
  return a && a.volatiles && a.volatiles['confusion'] ? (a.volatiles['confusion'].time | 0) : 0;
}

function spikesOf(side) {
  const sc = side.sideConditions && side.sideConditions['spikes'];
  return sc ? (sc.layers | 0) : 0;
}

function subHpOf(a) {
  return a && a.volatiles && a.volatiles['substitute'] ? (a.volatiles['substitute'].hp | 0) : 0;
}

function snap(side) {
  const a = side.active[0];
  if (!a) {
    return {
      species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', stage: 0, left: side.pokemonLeft,
      boosts: [0, 0, 0, 0, 0], confusion: 0, spikes: spikesOf(side), subHp: 0,
    };
  }
  const { status, stage } = statusOf(a);
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status, stage, left: side.pokemonLeft,
    boosts: boostsOf(a), confusion: confusionOf(a), spikes: spikesOf(side),
    subHp: subHpOf(a),
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

// Scan the protocol log between two decision points for the EXPLOSION edge flags.
//   selfKO     — a `|faint|` on the user (the Explosion self-KO). We detect ANY faint of
//                the mon that used a selfdestruct move this decision.
//   subBreak   — a `|-end|...|Substitute` (Explosion broke a sub this decision).
//   subAbsorb  — a hit that HELD or BROKE a sub.
//   immune     — a `|-immune|` (Explosion into a Normal-immune Ghost).
//   protBlock  — a `|-activate|...|move: Protect` (Explosion blocked by Protect).
//   drag       — a `|drag|` (irrelevant here, kept for shape).
function outcomesSince(log, fromIdx) {
  const out = {
    p1: { fullpara: false, wake: false, thaw: false, selfhit: false, flinch: false },
    p2: { fullpara: false, wake: false, thaw: false, selfhit: false, flinch: false },
    selfKO: false, subBreak: false, subDamage: false, subAbsorb: false,
    immune: false, protBlock: false, drag: false,
  };
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    const tag = p[1];
    const who = (p[2] || '').startsWith('p1a:') ? 'p1' : (p[2] || '').startsWith('p2a:') ? 'p2' : null;
    if (tag === 'cant' && who) {
      if (p[3] === 'par') out[who].fullpara = true;
      if (p[3] === 'flinch') out[who].flinch = true;
    }
    if (tag === '-curestatus' && who) {
      if ((p[3] || '') === 'slp') out[who].wake = true;
      if ((p[3] || '') === 'frz') out[who].thaw = true;
    }
    if (tag === '-damage' && who && (p[4] || '').includes('confusion')) out[who].selfhit = true;
    // The self-KO manifests as a faint; we tag it whenever a selfdestruct move was used
    // (the caller records `usedSelfdestruct`) — here we just flag any faint for context.
    if (tag === 'faint') out.selfKO = out.selfKO || true;
    if (tag === '-end' && (p[3] || '') === 'Substitute') out.subBreak = true;
    if (tag === '-activate' && (p[3] || '') === 'Substitute' && (p[4] || '').includes('damage')) out.subDamage = true;
    if (tag === '-immune') out.immune = true;
    // A protect BLOCK: `|-activate|p2a: Blissey|Protect` (p[3] is the bare condition name).
    if (tag === '-activate' && (p[3] || '').includes('Protect')) out.protBlock = true;
    if (tag === 'drag') out.drag = true;
  }
  out.subAbsorb = out.subDamage || out.subBreak;
  // NOTE: `selfKO` is refined by the caller (which knows a selfdestruct move was picked) so a
  // foe-side faint alone does not falsely satisfy the require. We keep the raw faint flag here
  // and let the caller AND-it with `usedSelfdestruct`.
  return out;
}

function firstLiveBench(side, battle) {
  const s = battle.sides[side];
  for (let k = 0; k < s.pokemon.length; k++) {
    const p = s.pokemon[k];
    if (p !== s.active[0] && !p.fainted) return `switch ${k + 1}`;
  }
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

// Which move id does `side`'s active have in the 1-based slot `want` (clamped legal)?
function moveIdAt(side, battle, want) {
  const req = battle.sides[side].activeRequest;
  const moves = req && req.active && req.active[0] ? req.active[0].moves : null;
  if (!moves) return null;
  const legal = legalMove(side, battle, want);
  const m = legal.match(/^move\s+(\d+)$/);
  const idx = m ? Number(m[1]) - 1 : 0;
  return moves[idx] ? moves[idx].id : null;
}

function intentDriver(intent) {
  return (decisionNo, battle, reqState, force) => {
    if (reqState === 'switch') {
      const c = { p1: null, p2: null, usedSelfdestruct: false };
      if (force[0]) c.p1 = firstLiveBench(0, battle);
      if (force[1]) c.p2 = firstLiveBench(1, battle);
      return c;
    }
    const r = intent(decisionNo, battle);
    const p1want = r.p1Switch ? null : r.p1Want;
    const p2want = r.p2Switch ? null : r.p2Want;
    // Detect whether either chosen move is a selfdestruct move (for the selfKO require).
    let usedSelfdestruct = false;
    if (p1want != null) { const id = moveIdAt(0, battle, p1want); if (id === 'explosion' || id === 'selfdestruct') usedSelfdestruct = true; }
    if (p2want != null) { const id = moveIdAt(1, battle, p2want); if (id === 'explosion' || id === 'selfdestruct') usedSelfdestruct = true; }
    return {
      p1: r.p1Switch ? `switch ${r.p1Switch}` : legalMove(0, battle, r.p1Want),
      p2: r.p2Switch ? `switch ${r.p2Switch}` : legalMove(1, battle, r.p2Want),
      usedSelfdestruct,
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

  const script = intentDriver(sc.intent);
  const rec = { initSeed: null, decisions: [], winner: null, ended: false, gen: stream.battle.gen };

  const maxDecisions = sc.maxDecisions || 400;
  let decisionNo = 0;
  let safety = 0;
  while (!stream.battle.ended && safety < 400 && decisionNo < maxDecisions) {
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
    for (let i = 0; i < 18; i++) await tick();

    const seedAfter = battle.prng.getSeed();
    if (seedAfter === seedBefore && log.length === logLenBefore && battle.requestState === reqState) {
      throw new Error(`STALL: choice ${JSON.stringify(choices)} did not advance the battle ` +
        `(scenario ${sc.id}, decision ${decisionNo}) — the sim rejected it. Fix the script.`);
    }
    const outcomes = outcomesSince(log, logLenBefore);
    // Refine selfKO: a self-KO only counts when a selfdestruct move was actually used this
    // decision AND a faint fired (so a foe-only faint doesn't satisfy it).
    outcomes.selfKO = outcomes.selfKO && !!choices.usedSelfdestruct;
    const first = reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none';

    const p1 = snap(battle.sides[0]);
    const p2 = snap(battle.sides[1]);

    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1, p2,
      firstMover: first,
      outcomes,
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

  // --- (1) PLAIN Explosion into a bulky foe that SURVIVES: the USER faints (self-KO), the
  //   foe takes big def-halved damage but lives. Draws = acc+crit+dmg (no Quick Claw — the
  //   user faint pauses for a single replacement). p1 Electrode Explodes into a bulky
  //   Snorlax; p1's 2nd mon replaces. REQUIRES: selfKO. ---
  S.push({
    id: 'plain_explosion_user_faints_foe_survives',
    p1: [mon('Electrode', ['explosion', 'thunderbolt'], { ability: 'Soundproof', nature: 'Timid', evs: { spa: 252, spe: 252 } }),
         mon('Jolteon', ['thunderbolt', 'shadowball'], { ability: 'Volt Absorb', nature: 'Timid', evs: { spe: 252 } })],
    p2: [mon('Snorlax', ['splash', 'bodyslam'], { ability: 'Immunity', item: 'Leftovers', nature: 'Careful', evs: { hp: 252, def: 252, spd: 252 } })],
    // p1 Explodes (move 1) turn 1; then Jolteon Thunderbolts to a win. p2 Splashes / Body Slams.
    intent: (decisionNo) => ({ p1Want: 1, p2Want: 1 }),
    maxDecisions: 12,
    require: ['selfKO'],
  });

  // --- (2) Explosion into a SUBSTITUTE: the damage hits the SUB (breaks it, no carry to the
  //   mon); the USER STILL faints. p2 Blissey subs; p1 Electrode Explodes into the sub (which
  //   BREAKS — a max-Atk Explosion far exceeds the sub HP). The subber's mon HP is UNCHANGED
  //   (no carry). REQUIRES: selfKO + subBreak. ---
  S.push({
    id: 'explosion_into_a_substitute_breaks_it_user_faints',
    p1: [mon('Electrode', ['explosion', 'thunderbolt'], { ability: 'Soundproof', nature: 'Hasty', evs: { atk: 252, spe: 252 } }),
         mon('Jolteon', ['thunderbolt'], { ability: 'Volt Absorb', nature: 'Timid', evs: { spe: 252 } })],
    p2: [mon('Blissey', ['substitute', 'softboiled'], { ability: 'Natural Cure', item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    // dec 0: p1 Splash-equiv (Thunderbolt) while p2 Substitutes → sub up. dec 1: p1 Explodes
    // INTO the sub → breaks it, user faints. Then Jolteon replaces + Thunderbolts.
    intent: (decisionNo, battle) => {
      const p2a = battle.sides[1].active[0];
      const hasSub = p2a && p2a.volatiles && p2a.volatiles['substitute'];
      if (!hasSub && p2a && p2a.species.name === 'Blissey' && !p2a.fainted) {
        // p2 builds the sub while p1 stalls with Thunderbolt (move 2 for Electrode).
        const p1a = battle.sides[0].active[0];
        const p1Explode = p1a && p1a.species.name === 'Electrode';
        return { p1Want: p1Explode ? 2 : 1, p2Want: 1 };
      }
      // sub is up (or Blissey gone) → p1 Explodes (Electrode) / attacks (Jolteon).
      return { p1Want: 1, p2Want: 2 };
    },
    maxDecisions: 12,
    require: ['selfKO', 'subBreak'],
  });

  // --- (3) Explosion into a PROTECT: the move is BLOCKED (no foe damage) — but the USER
  //   STILL faints (the self-KO precedes the hit). p2 Blissey Protects; p1 Electrode Explodes
  //   → `-activate Protect`, no damage, the user faints. REQUIRES: selfKO + protBlock. ---
  S.push({
    id: 'explosion_into_a_protect_blocked_user_still_faints',
    p1: [mon('Electrode', ['explosion', 'thunderbolt'], { ability: 'Soundproof', nature: 'Hasty', evs: { atk: 252, spe: 252 } }),
         mon('Jolteon', ['thunderbolt'], { ability: 'Volt Absorb', nature: 'Timid', evs: { spe: 252 } })],
    p2: [mon('Blissey', ['protect', 'softboiled'], { ability: 'Natural Cure', item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    // dec 0: p1 Explodes (move 1), p2 Protects (move 1) → blocked, user faints. Then Jolteon
    // replaces + Thunderbolts (p2 keeps Protecting / Soft-Boiling — the first Protect always
    // succeeds; consecutive ones may fail but the block on dec 0 is guaranteed).
    intent: (decisionNo, battle) => {
      const p1a = battle.sides[0].active[0];
      const p1Explode = p1a && p1a.species.name === 'Electrode';
      return { p1Want: p1Explode ? 1 : 1, p2Want: 1 };
    },
    maxDecisions: 12,
    require: ['selfKO', 'protBlock'],
  });

  // --- (4) Explosion into a GHOST (Normal-immune): no damage — the USER STILL faints. p1
  //   Electrode Explodes into a Gengar (Ghost) → `-immune`, no damage, the user faints.
  //   REQUIRES: selfKO + immune. ---
  S.push({
    id: 'explosion_into_a_ghost_immune_user_still_faints',
    p1: [mon('Electrode', ['explosion', 'thunderbolt'], { ability: 'Soundproof', nature: 'Hasty', evs: { atk: 252, spe: 252 } }),
         mon('Jolteon', ['thunderbolt'], { ability: 'Volt Absorb', nature: 'Timid', evs: { spe: 252 } })],
    p2: [mon('Gengar', ['splash', 'shadowball'], { ability: 'Levitate', item: 'Leftovers', nature: 'Timid', evs: { hp: 252, spe: 252 } })],
    // dec 0: p1 Explodes into Gengar → immune, user faints. Then Jolteon replaces + attacks.
    intent: (decisionNo, battle) => {
      const p1a = battle.sides[0].active[0];
      const p1Explode = p1a && p1a.species.name === 'Electrode';
      return { p1Want: p1Explode ? 1 : 1, p2Want: 1 };
    },
    maxDecisions: 12,
    require: ['selfKO', 'immune'],
  });

  // --- (5) MUTUAL Explosion (both last mons) → a gen-3 double-faint TIE. Single-mon teams,
  //   both Explode the SAME turn (equal speed → an action-order tie-shuffle). Both faint,
  //   both pokemon_left → 0, win(null) TIE. REQUIRES: selfKO + a tie. ---
  S.push({
    id: 'mutual_explosion_double_faint_tie',
    p1: [mon('Electrode', ['explosion', 'thunderbolt'], { ability: 'Soundproof', nature: 'Timid', evs: { spe: 252 } })],
    p2: [mon('Electrode', ['explosion', 'thunderbolt'], { ability: 'Soundproof', nature: 'Timid', evs: { spe: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    maxDecisions: 3,
    require: ['selfKO'],
  });

  // --- (6) Explosion KOs a FRAIL foe AND faints the user → a DOUBLE replacement. p1 Electrode
  //   Explodes into a frail lvl-1 foe (OHKO) → BOTH faint → both sides replace. The battle then
  //   continues with the replacements. REQUIRES: selfKO. ---
  S.push({
    id: 'explosion_ko_forces_a_double_replacement',
    p1: [mon('Electrode', ['explosion', 'thunderbolt'], { ability: 'Soundproof', nature: 'Hasty', evs: { atk: 252, spe: 252 } }),
         mon('Jolteon', ['thunderbolt'], { ability: 'Volt Absorb', nature: 'Timid', evs: { spe: 252 } })],
    p2: [mon('Diglett', ['splash'], { level: 1, ability: 'Sand Veil', nature: 'Bold' }),
         mon('Snorlax', ['bodyslam', 'splash'], { ability: 'Immunity', item: 'Leftovers', nature: 'Careful', evs: { hp: 252, def: 252 } })],
    // dec 0: p1 Explodes into the lvl-1 Diglett (OHKO) → double faint → double replacement.
    // Then Jolteon vs Snorlax to game-end (or the decision cap).
    intent: (decisionNo, battle) => {
      const p1a = battle.sides[0].active[0];
      const p1Explode = p1a && p1a.species.name === 'Electrode';
      return { p1Want: p1Explode ? 1 : 1, p2Want: 1 };
    },
    maxDecisions: 14,
    require: ['selfKO'],
  });

  // --- (7) Explosion INTO A REAL BATTLE to game-end: the self-KO + replacement machinery
  //   runs inside a longer battle with a switch + faints all the way to a win. p1 leads Snorlax,
  //   SWITCHES to Electrode (dec 0), Explodes the frail lvl-1 lead (dec 1 — OHKO → double faint),
  //   then Snorlax replaces + Body Slams the last frail foe to a WIN. The frail foes carry
  //   NO-OP abilities (Oblivious) so the contact Body Slam draws NO unmodeled ability handler
  //   (Static / Cute Charm would draw a contact `random` the port doesn't model — a scenario
  //   confound, NOT an engine bug). REQUIRES: selfKO + a decided game-end. ---
  S.push({
    id: 'explosion_into_a_real_battle',
    p1: [mon('Snorlax', ['bodyslam', 'splash'], { ability: 'Immunity', item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
         mon('Electrode', ['explosion', 'thunderbolt'], { ability: 'Soundproof', nature: 'Hasty', evs: { atk: 252, spe: 252 } })],
    // Frail lvl-1 foes with NO-OP abilities (Oblivious — no contact-hit draw): Body Slam OHKOs
    // them WITHOUT any Static/Cute-Charm contact `random`, so the seed stays a clean function of
    // the modeled draws.
    p2: [mon('Diglett', ['splash'], { level: 1, ability: 'Oblivious', nature: 'Bold' }),
         mon('Igglybuff', ['splash'], { level: 1, ability: 'Oblivious', nature: 'Bold' })],
    // dec 0: p1 Snorlax → switch to Electrode (p2 Splash). dec 1: Electrode Explodes into the
    // lvl-1 Diglett (OHKO) → DOUBLE faint → double replacement. dec 2: both replace (p1 Snorlax,
    // p2 Igglybuff). dec 3+: Snorlax Body Slams Igglybuff (OHKO) → p1 WINS.
    intent: (decisionNo, battle) => {
      const p1a = battle.sides[0].active[0];
      if (decisionNo === 0 && p1a && p1a.species.name === 'Snorlax') {
        return { p1Switch: 2, p2Want: 1 }; // switch to Electrode
      }
      const p1Explode = p1a && p1a.species.name === 'Electrode';
      if (p1Explode) return { p1Want: 1, p2Want: 1 }; // Explode (OHKO the lvl-1 → double faint)
      return { p1Want: 1, p2Want: 1 };                 // Snorlax Body Slam to a win
    },
    maxDecisions: 16,
    require: ['selfKO'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# explosion_golden.txt — Gen-3 EXPLOSION / SELF-DESTRUCT full-battle golden.');
  lines.push('# Per-decision-boundary STATE(+status+spikes-layers+subHp)+BOOSTS+CONFUSION+SEED+first-mover differential to GAME-END.');
  lines.push('# (Reuses the substitute TAB format verbatim: the per-side subHp columns prove the Explosion-into-a-sub break.)');
  lines.push('# The USER faints on EVERY Explosion (self-KO precedes the hit): vs a normal hit, a sub, a Protect, a Ghost.');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status stage left atk def spa spd spe confusion) p2(...) first \\');
  lines.push('#        p1(fullpara wake thaw selfhit flinch) p2(...)  p1Spikes p2Spikes  p1SubHp p2SubHp');
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
        for (const k of ['selfKO', 'subBreak', 'subDamage', 'subAbsorb', 'immune', 'protBlock', 'drag']) {
          if (d.outcomes[k]) { scenSeen[sc.id][k] = true; corpus[k] = (corpus[k] || 0) + 1; }
        }
        if (d.p1.subHp > 0 || d.p2.subHp > 0) { corpus.subRows = (corpus.subRows || 0) + 1; }
      });

      rec.decisions.forEach((d) => {
        const sp = (s) => [
          s.species, s.hp, s.maxhp, s.fainted ? 1 : 0, s.status, s.stage, s.left,
          s.boosts[0], s.boosts[1], s.boosts[2], s.boosts[3], s.boosts[4], s.confusion,
        ].join('\t');
        const oc = (o) => [o.fullpara ? 1 : 0, o.wake ? 1 : 0, o.thaw ? 1 : 0, o.selfhit ? 1 : 0, o.flinch ? 1 : 0].join('\t');
        lines.push([
          'DEC', sc.id, seedStr, d.request, d.force[0] ? 1 : 0, d.force[1] ? 1 : 0,
          d.choiceP1, d.choiceP2, d.seedAfter,
          sp(d.p1), sp(d.p2), d.firstMover,
          oc(d.outcomes.p1), oc(d.outcomes.p2),
          d.p1.spikes, d.p2.spikes,
          d.p1.subHp, d.p2.subHp,
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

    for (const need of (sc.require || [])) {
      if (!scenSeen[sc.id][need]) failures.push(`${sc.id}: REQUIRED edge ${need} never realized across the seed sweep`);
    }
    for (const bad of (sc.forbid || [])) {
      if (scenSeen[sc.id][bad]) failures.push(`${sc.id}: FORBIDDEN edge ${bad} realized (the scenario isolation is broken)`);
    }
  }

  if (failures.length) {
    console.error('EXPLOSION GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }

  const need = (label, key, min) => { const n = corpus[key] || 0; if (n < min) { console.error(`EXPLOSION GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('SELF-KO decisions', 'selfKO', 200);
  need('SUB-BREAK-by-explosion decisions', 'subBreak', 40);
  need('GHOST-IMMUNE self-KO decisions', 'immune', 40);
  need('PROTECT-blocked self-KO decisions', 'protBlock', 20);
  if (tieRows < 40) { console.error(`EXPLOSION GOLDEN: too few TIE rows (${tieRows} < 40)`); process.exit(1); }
  if (winRows < 100) { console.error(`EXPLOSION GOLDEN: too few WIN rows (${winRows} < 100)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `explosion golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `edges: selfKO=${corpus.selfKO || 0} subBreak=${corpus.subBreak || 0} immune=${corpus.immune || 0} ` +
    `protBlock=${corpus.protBlock || 0} subRows=${corpus.subRows || 0} -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
