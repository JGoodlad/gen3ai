// gen_recovery_move_golden.js — Gen-3 SELF-HEAL / RECOVERY-MOVE differential.
//
// Extends harness/gen_setup_move_golden.js (the per-decision STATE+STATUS+BOOSTS+
// CONFUSION+SEED+winner full-battle differential) to the NEW execution path this step
// adds: the self-targeting HP-RECOVERY moves (category Status / bp 0 / target self /
// isHeal) —
//   Recover / Soft-Boiled / Slack Off / Milk Drink  (heal floor(maxhp/2)),
//   Rest                                             (full heal + FIXED self-sleep + cure),
//   Moonlight / Synthesis / Morning Sun              (WEATHER-conditional heal).
// (Wish [delayed slot-keyed heal], Heal Bell / Aromatherapy / Refresh [status cure],
//  Pain Split / Leech Seed / drain / Ingrain / Aqua Ring are NOT modeled — fail-loud.)
//
// THE DRAW MODEL (verified bit-for-bit vs the omniscient sim, `data/moves.ts` +
// `data/mods/gen4/moves.ts` onHit, the gen-3 tryMoveHit self-heal path):
//   1. ACCURACY — every recovery move is NEVER-MISS (`accuracy: true`) → NO accuracy
//      draw. (Defensive: draw `randomChance(acc,100)` iff NOT never_miss; the set is all
//      never-miss, so this never draws.)
//   2. HEAL `this.heal(amount)` on the USER — DRAW-FREE (heal() consumes NO PRNG). The
//      amounts are INTEGER truncations (gen3 maxhp == baseMaxhp):
//        * Recover/Soft-Boiled/Slack Off/Milk Drink: floor(maxhp/2)  (move.heal:[1,2]).
//        * Moonlight/Synthesis/Morning Sun (gen4-inherited onHit, PLAIN integer — NOT the
//          4096-modify): NONE → floor(maxhp/2); SUN → floor(maxhp*2/3); SAND/RAIN/HAIL →
//          floor(maxhp/4). (VERIFIED: Espeon maxhp 271 in sun heals floor(271*2/3)=180,
//          not modify(271,0.667)=181.)
//      A HEAL-AT-FULL-HP / heal-0 FAILS (`-fail`), draw-free either way.
//   3. REST — the one move with a draw subtlety: a FIXED Sleep(3) self-sleep (NO
//      random(2,6) — the draw-COUNT difference from a sleep MOVE; VERIFIED: Rest sets
//      statusState.time=3 deterministically) + a FULL heal + a prior-status CURE
//      (override) + the gen3ou-only SetStatus handler-sort shuffle (gen3customgame draws
//      nothing — the golden's battle format). A self-Rest sleep is EXEMPT from the Sleep
//      Clause CAP. The user wakes via the existing sleep counter (3 attempts).
//   4. `landed` is ALWAYS FALSE — a status moveHit returns `undefined`, so the in-
//      tryMoveHit `eachEvent('Update')` shuffle is SKIPPED. So a recovery move is DRAW-
//      FREE (the seed is a function of only the OTHER draws); only the user's HP (and for
//      Rest, status) changes.
//
// THE PROOF: drive the OMNISCIENT in-process BattleStream (no server) over CONSTRUCTED
// scenarios that each ISOLATE one branch, capturing the running PRNG seed BEFORE the
// first decision (`initSeed`) and AFTER each DECISION BOUNDARY, plus each active's
// species/hp/maxhp/fainted/status(+inner counter) + boosts + confusion + pokemon_left +
// first mover + winner. The Rust test seeds a BattleState at the init seed and runs
// `run_full_battle` WITHOUT re-seeding — so the post-decision seed must match the sim's
// at EVERY boundary, AND the per-decision HP + status must match. An EXACT cross-decision
// seed+state match to game-end is the draw-ORDER+COUNT + heal-amount proof. (A 1-HP heal
// error desyncs the STATE; a wrong draw model desyncs the SEED.)
//
// FAIL-LOUD: each scenario declares the BRANCH it must realize (a heal applied, the full-
// HP fail, a Rest sleep+wake, a Rest cure, the 3 weather-heal fractions); generation
// aborts if the sim run did NOT realize it (so a mis-built scenario can never silently
// pass an empty path). The output shares the setup_move_golden TAB format (so the Rust
// gate reuses the same parser), with a `healDelta`-style branch tally in the corpus.
//
// Output: tests/vectors/recovery_move_golden.txt
//
// Run:  node src/rust_sim/harness/gen_recovery_move_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/recovery_move_golden.txt');
// gen3customgame (NOT gen3ou) — matches the e2e capstone + setup_move format. A Rest here
// draws NO SetStatus handler-sort shuffle (the gen3ou-only draw); the Rust test passes
// `format_id: "gen3customgame"`.
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
  let x = 0x51ed2c19 >>> 0;
  const step = () => { x = (Math.imul(x, 1103515245) + 12345) >>> 0; return x & 0xffff; };
  for (let i = 0; i < n; i++) out.push([step() || 1, step() || 1, step() || 1, step() || 1]);
  return out;
}

// The first ACTION to RUN this turn (matching the Rust `first_mover`). A mon that
// full-paras / is still-asleep / frozen / flinched emits `|cant|` and runs its action
// FIRST without a `|move|` line — so count `|cant|` + the confusion `-activate` too.
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

// The inner status counter: `tox` stage / `slp` remaining-turns (Rest sleep = the slp
// time). Matches Rust Status::Toxic / Sleep.
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

function snap(side) {
  const a = side.active[0];
  if (!a) return { species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', stage: 0, left: side.pokemonLeft, boosts: [0, 0, 0, 0, 0], confusion: 0 };
  const { status, stage } = statusOf(a);
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status, stage, left: side.pokemonLeft,
    boosts: boostsOf(a), confusion: confusionOf(a),
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

// Scan the protocol log between two decision points for the per-side onBeforeMove outcome
// flags (shared format with setup_move_golden) + the RECOVERY branch flags. The recovery
// flags ride the corpus so a scenario can REQUIRE a branch realized at least once:
//   healed   — a `-heal` line from a recovery move (a self-heal applied)
//   healFail — a `-fail ...|heal` line (a full-HP Recover/Moonlight/Rest fail)
//   restSlp  — a `-status ... slp|[from] move: Rest` (a Rest self-sleep)
//   restCure — a `-curestatus` on the RESTER (the prior status overridden by Rest's
//              setStatus — distinct from the natural sleep wake; detected via the apply
//              flag in main(), so here we only flag the Rest sleep line)
//   wake     — a `-curestatus ... slp [msg]` (a sleep wake — Rest or otherwise)
function outcomesSince(log, fromIdx) {
  const out = {
    p1: { fullpara: false, wake: false, thaw: false, selfhit: false, flinch: false },
    p2: { fullpara: false, wake: false, thaw: false, selfhit: false, flinch: false },
    healed: false, healFail: false, restSlp: false, miss: false,
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
    // A recovery `-heal` line (NOT a Leftovers/item heal — those carry `[from] item:`).
    if (tag === '-heal' && !(p[4] || '').includes('[from] item') && !(p[4] || '').includes('Wish')) out.healed = true;
    // A heal FAIL (`-fail ...|heal`) — the full-HP no-op path.
    if (tag === '-fail' && (p[3] || '') === 'heal') out.healFail = true;
    // A Rest self-sleep (`-status ... slp|[from] move: Rest`).
    if (tag === '-status' && (p[3] || '') === 'slp' && (p[4] || '').includes('move: Rest')) out.restSlp = true;
    if (tag === '-miss') out.miss = true;
  }
  return out;
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

  const script = sc.makeScript();
  const rec = { initSeed: null, decisions: [], winner: null, ended: false, gen: stream.battle.gen, branchSeen: {} };

  let decisionNo = 0;
  let safety = 0;
  while (!stream.battle.ended && safety < 400) {
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
    for (let i = 0; i < 16; i++) await tick();

    const seedAfter = battle.prng.getSeed();
    // STALL GUARD: a rejected choice (no advance in seed/log/request) → fail loud.
    if (seedAfter === seedBefore && log.length === logLenBefore && battle.requestState === reqState) {
      throw new Error(`STALL: choice ${JSON.stringify(choices)} did not advance the battle ` +
        `(scenario ${sc.id}, decision ${decisionNo}) — the sim rejected it. Fix the script.`);
    }
    const outcomes = outcomesSince(log, logLenBefore);
    const first = reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none';

    for (const k of ['healed', 'healFail', 'restSlp', 'miss']) {
      if (outcomes[k]) rec.branchSeen[k] = true;
    }
    if (outcomes.p1.wake || outcomes.p2.wake) rec.branchSeen.wake = true;
    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1: snap(battle.sides[0]),
      p2: snap(battle.sides[1]),
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
// Each is { id, p1[], p2[], makeScript, require } where `require` is the set of branch
// flags the SIM run must realize at least once (across the seed sweep) — else generation
// FAILS LOUD (the scenario didn't exercise its intended branch).

function scenarios() {
  const S = [];

  const firstLiveBench = (side, battle) => {
    const s = battle.sides[side];
    for (let k = 0; k < s.pokemon.length; k++) {
      const p = s.pokemon[k];
      if (p !== s.active[0] && !p.fainted) return `switch ${k + 1}`;
    }
    return 'pass';
  };
  // A plan-driven script (one entry per MOVE decision; a forced switch picks the first
  // live bench). The plan's last entry repeats once exhausted.
  const fromPlan = (plan) => () => {
    let i = 0;
    return (decisionNo, battle, reqState, force) => {
      if (reqState === 'switch') {
        const c = { p1: null, p2: null };
        if (force[0]) c.p1 = firstLiveBench(0, battle);
        if (force[1]) c.p2 = firstLiveBench(1, battle);
        return c;
      }
      const entry = plan[Math.min(i, plan.length - 1)];
      i++;
      return entry;
    };
  };

  // --- (1) RECOVER FROM LOW HP (the floor(maxhp/2) heal, multi-turn): a bulky Suicune is
  //   chipped by a WEAK resisted hit (Pikachu's Quick Attack — Normal, 40 BP, never-miss,
  //   priority) and Recovers each turn, so HP oscillates chip→heal. The per-decision HP must
  //   match EXACTLY (a 1-HP heal error diverges the STATE). The chip is tiny + crit-safe
  //   (40 BP on a 252/252 Bold Suicune can't KO even on a crit), so the recovery loop is
  //   stable; Suicune then Surfs the frail Pikachu out. REQUIRES: a heal applied. ---
  S.push({
    id: 'recover_low_hp',
    p1: [mon('Suicune', ['recover', 'surf'], { ability: 'Pressure', item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    p2: [mon('Pikachu', ['quickattack', 'thundershock'], { ability: 'Static', nature: 'Jolly', evs: { atk: 252, spe: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 2', p2: 'move 1' }, // Surf (take a Quick Attack chip first)
      { p1: 'move 1', p2: 'move 1' }, // Recover (heals floor(maxhp/2) from the chipped HP)
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' }, // Surf toward the win (Pikachu is frail)
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]),
    require: ['healed'],
  });

  // --- (2) RECOVER AT FULL HP (the FAIL / no-op path, isolated): a Blissey at full HP uses
  //   Soft-Boiled — it FAILS (`-fail heal`), heals 0, draws nothing. The foe's Quick Attack
  //   (40 BP) chips Blissey but Leftovers tops it back to full, so the fail repeats; Blissey
  //   then Ice-Beams the frail foe out. REQUIRES: the heal-fail branch. ---
  S.push({
    id: 'softboiled_full_hp_fail',
    p1: [mon('Blissey', ['softboiled', 'icebeam'], { ability: 'Natural Cure', item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    // Diglett: very frail, uses SPLASH (does nothing — so Blissey stays at FULL HP and Soft-
    // Boiled FAILS). It dies fast to Ice Beam (Ground 4× weak). (Splash is a modeled draw-
    // free no-op; Sand Veil is N/A here — no sand.)
    p2: [mon('Diglett', ['splash'], { ability: 'Sand Veil', nature: 'Jolly', evs: { atk: 252, spe: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Soft-Boiled at full HP → FAIL
      { p1: 'move 1', p2: 'move 1' }, // FAIL again (Splash keeps Blissey at full)
      { p1: 'move 1', p2: 'move 1' }, // FAIL again
      { p1: 'move 2', p2: 'move 1' }, // Ice Beam — KO the frail Diglett (4× weak: Ground)
      { p1: 'move 2', p2: 'move 1' },
    ]),
    require: ['healFail'],
  });

  // --- (3) REST FROM LOW HP (full heal + FIXED sleep 3 + the WAKE 3 turns later, NO
  //   random(2,6)): a bulky Snorlax is chipped by a WEAK resisted hit (Vulpix's Ember —
  //   Fire, HALVED by Snorlax's Thick Fat → tiny survivable chip) below half, then RESTS
  //   (full heal + Sleep(3)). It can't move for 2 turns (slp) and wakes on the 3rd, Body
  //   Slamming the frail foe. The status counter (3→2→1→wake) must match. A wrong sleep
  //   draw (a random(2,6)) would desync the SEED on the Rest turn. Ember can NEVER KO a
  //   524-HP Thick-Fat Snorlax (even on a crit), so the sleep+wake always completes.
  //   REQUIRES: a Rest sleep + a wake. ---
  S.push({
    id: 'rest_low_hp_sleep_wake',
    p1: [mon('Snorlax', ['rest', 'bodyslam', 'splash'], { ability: 'Thick Fat', item: 'Leftovers', nature: 'Careful', evs: { hp: 252, spd: 252 } })],
    // Spinarak: frail Bug; its SWIFT (60 BP Normal, NEVER-MISS, NO secondary — so the chip
    // draws ONLY acc-free/crit/dmg, no status secondary that would complicate the Rest cure)
    // is a tiny crit-safe chip a 524-HP Snorlax shrugs off through the whole sleep; it dies
    // in a couple Body Slams. Using a no-secondary chipper keeps the Rest turn's draw model
    // clean (the foe never inflicts a status that Rest's cure would interact with).
    p2: [mon('Spinarak', ['swift'], { ability: 'Insomnia', nature: 'Modest', evs: { atk: 252, spe: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 3', p2: 'move 1' }, // Splash (take Swift chip, don't attack)
      { p1: 'move 3', p2: 'move 1' }, // Splash (more chip)
      { p1: 'move 3', p2: 'move 1' }, // Splash (more chip)
      { p1: 'move 3', p2: 'move 1' }, // Splash (more chip — Snorlax goes below half eventually)
      { p1: 'move 3', p2: 'move 1' },
      { p1: 'move 3', p2: 'move 1' },
      { p1: 'move 3', p2: 'move 1' },
      { p1: 'move 3', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' }, // REST → full heal + Sleep(3)
      { p1: 'move 1', p2: 'move 1' }, // asleep (cant) — slp 3→2
      { p1: 'move 1', p2: 'move 1' }, // asleep (cant) — slp 2→1
      { p1: 'move 2', p2: 'move 1' }, // WAKE + Body Slam (slp 1→wake) → KO the frail foe
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]),
    require: ['restSlp', 'wake'],
  });

  // --- (4) REST CURING A PRIOR STATUS (the override): a bulky Snorlax is PARALYZED by the
  //   foe's Thunder Wave (an IN-ENGINE status move — modeled), then chipped by the weak
  //   Thunder Shock, then RESTS — the prior `par` is CURED (overridden by sleep). The per-
  //   decision STATUS must show par → slp on the Rest turn. Thunder Shock (40 BP) can never
  //   KO a 524-HP Snorlax, so the cure→sleep→wake always completes; a woken Snorlax then
  //   Body Slams the frail Pichu out. REQUIRES: a Rest sleep (the cure is the par→slp flip
  //   in the per-decision STATUS). ---
  S.push({
    id: 'rest_cures_paralysis',
    p1: [mon('Snorlax', ['rest', 'earthquake', 'splash'], { ability: 'Thick Fat', item: 'Leftovers', nature: 'Careful', evs: { hp: 252, atk: 252 } })],
    // Electrode: frail Electric; Thunder Wave (par, MODELED) then Thunder Shock (40 BP)
    // chip. SOUNDPROOF is a provable NO-OP here (it only blocks sound moves — NOT the
    // contact/Static-style reactive abilities that draw RNG on a contact hit, which would
    // desync). A 524-HP Snorlax shrugs the chip and EARTHQUAKEs (non-contact, Electric 2×
    // weak) the frail Electrode out after waking. (Earthquake avoids any on-contact ability
    // reaction entirely.)
    p2: [mon('Electrode', ['thunderwave', 'thundershock'], { ability: 'Soundproof', nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 3', p2: 'move 1' }, // Splash ; foe Thunder Wave → Snorlax PARALYZED
      { p1: 'move 3', p2: 'move 2' }, // Splash ; foe Thunder Shock (chip)
      { p1: 'move 3', p2: 'move 2' }, // Splash ; foe Thunder Shock (more chip)
      { p1: 'move 3', p2: 'move 2' }, // Splash ; foe Thunder Shock (more chip)
      { p1: 'move 1', p2: 'move 2' }, // REST → full heal + Sleep(3), CURING the par
      { p1: 'move 1', p2: 'move 2' }, // asleep (slp 3→2)
      { p1: 'move 1', p2: 'move 2' }, // asleep (slp 2→1)
      { p1: 'move 2', p2: 'move 2' }, // WAKE + Earthquake (non-contact, 2× on Electric) — KO
      { p1: 'move 2', p2: 'move 2' },
    ]),
    require: ['restSlp'],
  });

  // --- (5) MOONLIGHT NO WEATHER (factor 0.5 → floor(maxhp/2)): an Umbreon with Faint Attack
  //   (never-miss Dark STAB) takes a weak Confusion chip (Psychic 50 BP — neutral on Dark
  //   Umbreon), Moonlights (no weather → floor(maxhp/2)) a few times, then Faint-Attacks the
  //   FRAIL Psychic foe (Dark 2× super-effective) out. The per-decision HP must match the
  //   no-weather heal exactly. REQUIRES: a heal applied. ---
  S.push({
    id: 'moonlight_no_weather',
    p1: [mon('Umbreon', ['moonlight', 'feintattack'], { ability: 'Synchronize', item: 'Leftovers', nature: 'Calm', evs: { hp: 252, spd: 252 } })],
    // Misdreavus: frail GHOST (2× weak to Dark Faint Attack). It chips Umbreon with QUICK
    // ATTACK (Normal — NEUTRAL on Dark, unlike a Psychic move which Dark is IMMUNE to), so
    // the no-weather Moonlight heal is exercised; it then dies in a couple Faint Attacks.
    // Levitate is MODELED (irrelevant here).
    p2: [mon('Misdreavus', ['quickattack'], { ability: 'Levitate', nature: 'Jolly', evs: { atk: 252, spe: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Moonlight (take Quick Attack chip first → heal floor(maxhp/2))
      { p1: 'move 1', p2: 'move 1' }, // Moonlight again
      { p1: 'move 1', p2: 'move 1' }, // Moonlight again
      { p1: 'move 2', p2: 'move 1' }, // Faint Attack (super-effective on Ghost) → KO the frail foe
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]),
    require: ['healed'],
  });

  // --- (6) MOONLIGHT IN SUN (factor 2/3 → floor(maxhp*2/3)): an Espeon Moonlights under a
  //   foe Groudon's DROUGHT (PERMANENT sun, set on switch-in — the only modeled sun source;
  //   Sunny Day the MOVE is not modeled), healing floor(maxhp*2/3) (the LARGER fraction,
  //   distinct from the 4096-modify). Groudon's weak Quick Attack chips Espeon so the SUN
  //   heal is exercised; the per-decision HP must match the sun-heal exactly. Espeon then
  //   grinds Groudon out with Psychic STAB. REQUIRES: a heal applied. ---
  S.push({
    id: 'moonlight_sun',
    p1: [mon('Espeon', ['moonlight', 'psychic'], { ability: 'Synchronize', item: 'Leftovers', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    // Groudon: DROUGHT sets PERMANENT sun on switch-in (MODELED). Low SpD investment so a
    // Modest Espeon's Psychic grinds it down; its Quick Attack (40 BP) is a tiny crit-safe
    // chip that lets the SUN Moonlight heal show. (Ground-type → Psychic is neutral.)
    p2: [mon('Groudon', ['quickattack'], { ability: 'Drought', nature: 'Adamant', evs: { hp: 4, atk: 252, spe: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Moonlight in SUN (take a Quick Attack chip → heal floor(maxhp*2/3))
      { p1: 'move 1', p2: 'move 1' }, // Moonlight in SUN again
      { p1: 'move 2', p2: 'move 1' }, // Psychic — grind Groudon down
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]),
    require: ['healed'],
  });

  // --- (7) SYNTHESIS IN SAND (factor 1/4 → floor(maxhp/4)): a bulky Blissey under a foe
  //   Tyranitar's Sand Stream (permanent sand) heals only floor(maxhp/4) (the SMALL fraction)
  //   AND takes the sand chip — so HP barely climbs. A weak Rock Throw chips it; the per-
  //   decision HP (heal − sand − chip) must match exactly. Blissey is NOT Rock/Ground/Steel
  //   (so it takes the sand chip), and Ice Beam (Rock 2× weak) grinds the foe out.
  //   REQUIRES: a heal applied. ---
  S.push({
    id: 'synthesis_sand',
    // Blissey (Normal — sand-chipped) max-bulk; Synthesis heals floor(maxhp/4) in sand.
    p1: [mon('Blissey', ['synthesis', 'icebeam'], { ability: 'Natural Cure', item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    // Tyranitar: Sand Stream (permanent sand, MODELED) + Rock Throw (50 BP Rock — a small
    // physical chip a 252/252 Bold Blissey survives indefinitely). It dies slowly to Ice
    // Beam (Rock 2× weak); the grind terminates well before any stall.
    p2: [mon('Tyranitar', ['rockthrow'], { ability: 'Sand Stream', nature: 'Careful', evs: { hp: 252, spd: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 2', p2: 'move 1' }, // Ice Beam (take Rock Throw chip + sand; sand already up)
      { p1: 'move 2', p2: 'move 1' }, // Ice Beam (chip the foe)
      { p1: 'move 1', p2: 'move 1' }, // Synthesis in SAND (floor(maxhp/4))
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' }, // Ice Beam — keep chipping the foe
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]),
    require: ['healed'],
  });

  // --- (8) RECOVERY INTO A REAL BATTLE TO GAME-END (the union: recovery interleaved with a
  //   switch + the full move/residual/faint machinery): p1's Suicune Recovers off a weak
  //   Quick-Attack chip behind a frail foe, VOLUNTARILY PIVOTS to a hard-hitting Salamence
  //   that sweeps the FRAIL 2-mon foe team out to a clean win. Recovery + a switch +
  //   residuals + a faint all the way to a win. (Rest's sleep+wake is exercised in scenarios
  //   3/4; here the union is Recover + a switch + a sweep, kept short + decisive so it always
  //   terminates — the chip is crit-safe so Suicune always reaches the scripted pivot.)
  //   REQUIRES: a heal applied + a win run. ---
  S.push({
    id: 'recovery_into_real_battle',
    p1: [mon('Suicune', ['recover', 'surf'], { ability: 'Pressure', item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } }),
         mon('Salamence', ['dragonclaw', 'earthquake'], { ability: 'Intimidate', item: 'Leftovers', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    // Two FRAIL chippers using ONLY Quick Attack (40 BP Normal, never-miss, no crit swing
    // that could KO a 252/252 Bold Suicune early — so the scripted pivot is always reached).
    // A max-Atk Adamant Salamence then OHKOs the frail pair with Dragon Claw. Levitate /
    // Flash Fire are MODELED.
    p2: [mon('Misdreavus', ['quickattack'], { ability: 'Levitate', nature: 'Jolly', evs: { atk: 252, spe: 252 } }),
         mon('Houndour', ['quickattack'], { ability: 'Flash Fire', nature: 'Jolly', evs: { atk: 252, spe: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 2', p2: 'move 1' }, // Suicune Surf (take a Quick Attack chip)
      { p1: 'move 1', p2: 'move 1' }, // Suicune Recover (heal back the chip)
      { p1: 'move 1', p2: 'move 1' }, // Suicune Recover again
      { p1: 'switch 2', p2: 'move 1' }, // VOLUNTARY pivot to Salamence (the switch interleave)
      { p1: 'move 1', p2: 'move 1' }, // Salamence Dragon Claw — KO the frail Misdreavus
      { p1: 'move 1', p2: 'move 1' }, // Dragon Claw the second frail foe (Houndour) out
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
    ]),
    require: ['healed'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# recovery_move_golden.txt — Gen-3 SELF-HEAL / RECOVERY-MOVE full-battle golden.');
  lines.push('# Per-decision-boundary STATE(+status+counter)+BOOSTS+SEED+first-mover differential to GAME-END.');
  lines.push('# (Shares the setup_move_golden TAB format so the Rust gate reuses the parser.)');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status stage left atk def spa spd spe confusion) p2(...) first \\');
  lines.push('#        p1(fullpara wake thaw selfhit flinch) p2(...) healed');
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

      for (const k of Object.keys(rec.branchSeen)) { scenSeen[sc.id][k] = true; corpus[k] = (corpus[k] || 0) + 1; }

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
          oc(d.outcomes.p1), oc(d.outcomes.p2), d.outcomes.healed ? 1 : 0,
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

    // FAIL-LOUD: the scenario must realize its declared branches.
    for (const need of (sc.require || [])) {
      if (!scenSeen[sc.id][need]) failures.push(`${sc.id}: REQUIRED branch ${need} never realized across the seed sweep`);
    }
    for (const bad of (sc.forbid || [])) {
      if (scenSeen[sc.id][bad]) failures.push(`${sc.id}: FORBIDDEN branch ${bad} realized (the scenario isolation is broken)`);
    }
  }

  if (failures.length) {
    console.error('RECOVERY-MOVE GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }

  // Corpus floors: every recovery branch must realize SOMEWHERE.
  const need = (label, key, min) => { const n = corpus[key] || 0; if (n < min) { console.error(`RECOVERY GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('heal-applied runs', 'healed', 20);
  need('heal-fail runs', 'healFail', 5);
  need('Rest-sleep runs', 'restSlp', 10);
  need('wake runs', 'wake', 10);
  if (winRows < 50) { console.error(`RECOVERY GOLDEN: too few WIN rows (${winRows} < 50)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `recovery-move golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `branches: healed=${corpus.healed || 0} healFail=${corpus.healFail || 0} ` +
    `restSlp=${corpus.restSlp || 0} wake=${corpus.wake || 0} -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
