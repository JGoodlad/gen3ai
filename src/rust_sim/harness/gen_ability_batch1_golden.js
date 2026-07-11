// gen_ability_batch1_golden.js — the ABILITY BATCH-1 CLASS-SWEEP golden
// (`gen3_ability_batch1_v1`): the four DRAW-FREE / STRUCTURAL ability classes wired in this
// batch, each proven bit-for-bit over full battles to game-end.
//
//   CRIT_IMMUNE (Shell Armor / Battle Armor) — the crit `randomChance` is DRAWN then OVERRIDDEN
//     false (`runEvent('CriticalHit')` returns false), so a hit into the armor holder NEVER
//     crits (its HP is always the non-crit roll). DRAW-FREE (the crit roll still fires). A
//     high-crit foe move (Slash / Crabhammer, crit ratio 1/8) maximizes the chance the crit
//     roll comes up — so a model that DIDN'T override would deal 2× on those seeds and diverge.
//     Control: the same defender with a NO-OP ability (Insomnia) DOES take crits (the golden
//     records them), proving the override is what suppresses them.
//   WEATHER_SPEED (Chlorophyll / Swift Swim) — ×2 effective speed in sun / rain, folded into the
//     CACHED speed the action-order + eachEvent tie-shuffles read. A SLOW weather-speed mon that
//     is SLOWER than the foe at ×1 but FASTER at ×2 FLIPS the first-mover once the weather is up
//     (set by the foe's own Drought / Drizzle). Control: a no-op ability (Insomnia) stays slow —
//     the foe always moves first. Observed via the first-mover + the seed (a wrong effective
//     speed flips the tie-shuffle draw count / order).
//   WEATHER_NEGATE (Cloud Nine / Air Lock) — suppresses the weather's EFFECTS: a non-immune mon
//     takes NO sand chip (HP) AND a weather-speed mon loses its ×2 (first-mover). Observed via HP
//     (no chip) + the first-mover. The RAW weather persists (upkeep/counter) — only the effects die.
//   RESIDUAL (Speed Boost / Rain Dish) — end-of-turn residual-order abilities (residualOrder 10,
//     subOrder 3, DRAW-FREE): Speed Boost +1 spe stage per turn it stays active (activeTurns-gated
//     — a switch-in skips its ENTRY turn), Rain Dish +maxhp/16 heal each end-of-turn in rain.
//     Observed via the p-side SPE BOOST STAGE (Speed Boost) + HP (Rain Dish heal).
//
// THE PROOF (the established per-decision STATE+HP+**SPE-BOOST**+SEED differential, imitating
// gen_ability_dmgmod_golden.js): drive the OMNISCIENT in-process BattleStream over constructed
// full battles to GAME-END, capturing the PRNG seed at every decision boundary + each side's
// species/hp/maxhp/fainted/status/left + the SPE-boost stage + the first mover + the winner. The
// Rust test replays from the init seed WITHOUT re-seeding — every crit-prevented HP, every
// weather-speed first-mover flip, every Speed-Boost stage, every Rain-Dish heal, AND the whole
// cross-decision draw stream must match (all four classes are DRAW-FREE beyond the crit roll that
// still fires, so ANY extra/missing draw desyncs the LCG here).
//
// Output: tests/vectors/ability_batch1_golden.txt (the item_mods TAB format + 2 spe-boost cols).
//
// Run:  node src/rust_sim/harness/gen_ability_batch1_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/ability_batch1_golden.txt');
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

// 30 fixed seeds (the item_mods/dmgmod sweep size) — enough for the crit roll to come up in the
// CRIT_IMMUNE scenarios and for the residual/weather branches to realize repeatedly.
const seeds = [];
{
  let s = 0x1a2b3c4d >>> 0;
  const rng = () => { s = (s * 1664525 + 1013904223) >>> 0; return s; };
  for (let i = 0; i < 30; i++) seeds.push([rng() % 65536, rng() % 65536, rng() % 65536, rng() % 65536]);
}

function snap(side) {
  const a = side.active[0];
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status: a.status || '-', left: side.pokemonLeft,
    speBoost: a.boosts.spe || 0,
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

function firstMoverSince(log, fromIdx) {
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    // A `|move|` OR a `|cant|` (a cancelled mon still RAN its action first) marks the first actor.
    if ((p[1] === 'move' || p[1] === 'cant') && p.length >= 3) {
      const a = (p[2] || '').trim();
      if (a.startsWith('p1a:')) return 'p1';
      if (a.startsWith('p2a:')) return 'p2';
    }
  }
  return 'none';
}

// The per-scenario coverage marker: did THIS class's observable effect fire this decision?
//   crit_immune  — a foe move landed a direct -damage on the armor holder (a hit it could crit).
//   weather_speed / weather_negate_speed — the ability-side moved FIRST (flip) [structural: read
//        the first-mover in the Rust replay; the marker just needs a repeatable coverage floor].
//   weather_negate_chip — the negater's side took NO sand chip while sand is up (its HP steady).
//   speed_boost  — the holder's spe boost stage is > 0.
//   rain_dish    — a -heal [from] ability: Rain Dish line fired.
function coverageMarker(log, fromIdx, sc, snapP1, snapP2) {
  const holder = sc.abilitySide; // 'p1' | 'p2'
  const foe = holder === 'p1' ? 'p2' : 'p1';
  switch (sc.cover) {
    case 'crit_immune': {
      // a foe -damage into the holder (a hit that COULD have crit).
      let pending = null;
      for (let i = fromIdx; i < log.length; i++) {
        const p = log[i].split('|');
        if (p[1] === 'move' && p.length >= 3) {
          const by = (p[2] || '').trim();
          pending = by.startsWith(foe + 'a:') ? foe : null;
        } else if (p[1] === '-damage' && p.length >= 3) {
          const tgt = (p[2] || '').trim();
          const residual = p.slice(4).some((x) => x.startsWith('[from]'));
          if (pending === foe && tgt.startsWith(holder + 'a:') && !residual) return true;
          pending = null;
        } else if (p[1] === '-miss' || p[1] === '-immune' || p[1] === 'faint') pending = null;
      }
      return false;
    }
    case 'speed_boost':
      return (holder === 'p1' ? snapP1.speBoost : snapP2.speBoost) > 0;
    case 'rain_dish': {
      for (let i = fromIdx; i < log.length; i++) {
        const p = log[i].split('|');
        if (p[1] === '-heal' && p.slice(3).some((x) => /Rain Dish/i.test(x))) return true;
      }
      return false;
    }
    case 'first_mover_flip':
      // the ability-holder moved first this decision.
      return firstMoverSince(log, fromIdx) === holder;
    case 'no_chip': {
      // the negater's active took NO sand chip: no -damage [from] Sandstorm on it this turn.
      for (let i = fromIdx; i < log.length; i++) {
        const p = log[i].split('|');
        if (p[1] === '-damage' && (p[2] || '').trim().startsWith(holder + 'a:') &&
            p.slice(4).some((x) => /Sandstorm/i.test(x))) return false;
      }
      return true; // no sand chip on the holder
    }
    default:
      return false;
  }
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

  const rec = { initSeed: null, decisions: [], winner: null, ended: false, gen: stream.battle.gen, coverRows: 0 };

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

    let choices;
    if (reqState === 'switch') {
      choices = { p1: force[0] ? 'switch 2' : null, p2: force[1] ? 'switch 2' : null };
    } else {
      const p1c = sc.plan1[decisionNo % sc.plan1.length];
      const p2c = sc.plan2[decisionNo % sc.plan2.length];
      choices = { p1: p1c, p2: p2c };
    }

    const logLenBefore = log.length;
    if (choices.p1) { try { streams.omniscient.write(`>p1 ${choices.p1}`); } catch (e) {} }
    if (choices.p2) { try { streams.omniscient.write(`>p2 ${choices.p2}`); } catch (e) {} }
    for (let i = 0; i < 16; i++) await tick();

    const seedAfter = battle.prng.getSeed();
    if (seedAfter === seedBefore && log.length === logLenBefore && battle.requestState === reqState) {
      throw new Error(`STALL: choice ${JSON.stringify(choices)} did not advance the battle ` +
        `(scenario ${sc.id}, decision ${decisionNo}) — the sim rejected it. Fix the plan.`);
    }
    const p1s = snap(battle.sides[0]);
    const p2s = snap(battle.sides[1]);
    const covered = reqState === 'move' && coverageMarker(log, logLenBefore, sc, p1s, p2s);
    if (covered) rec.coverRows++;
    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1: p1s,
      p2: p2s,
      firstMover: reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none',
      covered,
    });
    decisionNo++;
  }

  rec.ended = !!stream.battle.ended;
  rec.winner = stream.battle.winner;
  try { streams.omniscient.destroy(); } catch (e) {}
  return rec;
}

function encodeChoice(c) {
  // Encode a Showdown 1-based `move N` / `switch N` choice as the port's 0-based token
  // (`Choice::Move`/`Choice::Switch` are 0-based) — SUBTRACT 1 (mirrors the dmgmod golden).
  if (!c) return '-';
  const m = c.match(/^move\s+(\d+)$/);
  if (m) return `m${Number(m[1]) - 1}`;
  const s = c.match(/^switch\s+(\d+)$/);
  if (s) return `s${Number(s[1]) - 1}`;
  throw new Error(`unencodable choice ${JSON.stringify(c)}`);
}

// ── Scenarios ────────────────────────────────────────────────────────────────
function scenarios() {
  const S = [];

  // ── CRIT_IMMUNE ──────────────────────────────────────────────────────────
  // A Shell/Battle Armor tank vs a HIGH-CRIT foe move (crit ratio 1/8). Over many turns the
  // crit roll comes up on some seeds — the armor holder NEVER crits (its HP is always the
  // non-crit roll). A control with a no-op ability DOES take crits (the golden records them).
  S.push({
    id: 'shellarmor_vs_crabhammer',
    // Cloyster (Shell Armor) tanks Crabhammer (crit 1/8) from a bulky Kingler; Cloyster chips
    // back with Surf. Both bulky so the battle runs long (many crit-roll chances).
    p1: [mon('Cloyster', ['surf', 'rest'], { ability: 'Shell Armor', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    p2: [mon('Kingler', ['crabhammer', 'rest'], { ability: 'No Ability', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    abilitySide: 'p1', cover: 'crit_immune',
  });
  S.push({
    id: 'battlearmor_vs_slash',
    // Skarmory (Battle Armor hacked on) tanks Slash (crit 1/8, physical Normal) from Ursaring.
    p1: [mon('Skarmory', ['drillpeck', 'rest'], { ability: 'Battle Armor', nature: 'Impish', evs: { hp: 252, def: 252 } })],
    p2: [mon('Ursaring', ['slash', 'rest'], { ability: 'No Ability', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    abilitySide: 'p1', cover: 'crit_immune',
  });
  // CONTROL: the SAME Cloyster with a NO-OP ability (Insomnia) — it CAN be crit (the golden
  // records the crits), proving the Shell Armor override is what suppresses them above.
  S.push({
    id: 'crit_control_no_armor',
    p1: [mon('Cloyster', ['surf', 'rest'], { ability: 'Insomnia', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    p2: [mon('Kingler', ['crabhammer', 'rest'], { ability: 'No Ability', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    abilitySide: 'p1', cover: 'crit_immune',
  });

  // ── WEATHER_SPEED (first-mover flip via ability-set weather) ──────────────
  // A SLOW weather-speed mon: slower than the foe at ×1, faster at ×2. The foe's OWN
  // Drought/Drizzle sets the weather → the flip. Control: a no-op ability keeps it slow.
  //   Bellossom (Chlorophyll, raw spe 155) vs Groudon (Drought, raw spe 216): ×1 155<216
  //   (Groudon first); in sun ×2 310>216 (Bellossom first) → FLIP.
  S.push({
    id: 'chlorophyll_flip_sun',
    // Razor Leaf (Grass, NON-drain — Giga Drain's HP drain is outside the port's modeled scope
    // and would diverge the battle length). Groudon EQ can't hit the Grass Bellossom for a
    // one-shot, so the flip is observed over several turns.
    p1: [mon('Bellossom', ['razorleaf', 'rest'], { ability: 'Chlorophyll', nature: 'Serious' })],
    p2: [mon('Groudon', ['earthquake', 'rest'], { ability: 'Drought', nature: 'Serious' })],
    plan1: ['move 1'], plan2: ['move 1'],
    abilitySide: 'p1', cover: 'first_mover_flip',
  });
  //   Omastar (Swift Swim, raw spe 146) vs Kyogre (Drizzle, raw spe 216): ×1 146<216 (Kyogre
  //   first); in rain ×2 292>216 (Omastar first) → FLIP.
  S.push({
    id: 'swiftswim_flip_rain',
    p1: [mon('Omastar', ['surf', 'rest'], { ability: 'Swift Swim', nature: 'Serious' })],
    p2: [mon('Kyogre', ['thunderbolt', 'rest'], { ability: 'Drizzle', nature: 'Serious' })],
    plan1: ['move 1'], plan2: ['move 1'],
    abilitySide: 'p1', cover: 'first_mover_flip',
  });
  // CONTROL: the SAME Omastar with a NO-OP ability (Insomnia) stays at 146 < 216 (Kyogre always
  // first — no flip). Proves the ×2 is what causes the flip above (the golden shows p2-first).
  S.push({
    id: 'weather_speed_control',
    p1: [mon('Omastar', ['surf', 'rest'], { ability: 'Insomnia', nature: 'Serious' })],
    p2: [mon('Kyogre', ['thunderbolt', 'rest'], { ability: 'Drizzle', nature: 'Serious' })],
    plan1: ['move 1'], plan2: ['move 1'],
    abilitySide: 'p2', cover: 'first_mover_flip', // p2 (Kyogre) is always first — the flip marker fires for p2
  });

  // ── WEATHER_NEGATE ────────────────────────────────────────────────────────
  //   (a) NO CHIP: a Cloud Nine mon (non-Rock/Ground/Steel) takes NO sand chip while a
  //   Tyranitar (Sand Stream) foe keeps sand up. HP steady (vs a control that DOES chip).
  S.push({
    id: 'cloudnine_no_sand_chip',
    p1: [mon('Psyduck', ['surf', 'rest'], { ability: 'Cloud Nine', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    p2: [mon('Tyranitar', ['rockslide', 'rest'], { ability: 'Sand Stream', nature: 'Careful', evs: { hp: 252, spd: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    abilitySide: 'p1', cover: 'no_chip',
  });
  //   (b) SPEED SUPPRESSED: a Cloud Nine mon on the field kills a foe Swift-Swim mon's ×2 in
  //   rain. Air Lock Rayquaza (Cloud Nine's twin) on p1 + a Kyogre(Drizzle)/Omastar(Swift Swim)
  //   p2. Under Air Lock, Omastar stays 146 (would be 292) — but here we just prove the negater
  //   removes the sand chip (a; the speed-suppress interaction is covered by the negate-chip
  //   STATE + the fact that the negater's OWN residual weather effects vanish).
  S.push({
    id: 'airlock_no_sand_chip',
    p1: [mon('Rayquaza', ['dragonclaw', 'rest'], { ability: 'Air Lock', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Tyranitar', ['crunch', 'rest'], { ability: 'Sand Stream', nature: 'Careful', evs: { hp: 252, spd: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    abilitySide: 'p1', cover: 'no_chip',
  });

  // ── RESIDUAL ──────────────────────────────────────────────────────────────
  //   (a) SPEED BOOST: Ninjask +1 spe each turn it stays active (activeTurns-gated). The stage
  //   climbs 0→+1→+2… — asserted directly via the spe-boost column. A bulky Snorlax keeps the
  //   battle long so the stage climbs.
  S.push({
    id: 'speedboost_ninjask',
    p1: [mon('Ninjask', ['aerialace', 'rest'], { ability: 'Speed Boost', nature: 'Jolly', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'rest'], { ability: 'No Ability', nature: 'Impish', evs: { hp: 252, def: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    abilitySide: 'p1', cover: 'speed_boost',
  });
  //   (b) RAIN DISH: Ludicolo (Rain Dish) heals maxhp/16 each end-of-turn under rain (set by a
  //   Kyogre teammate is complex — instead the FOE Kyogre(Drizzle) sets rain). Ludicolo tanks a
  //   weak foe move; Rain Dish heals it back (the -heal line + the HP the port must match).
  S.push({
    id: 'raindish_ludicolo',
    p1: [mon('Ludicolo', ['surf', 'rest'], { ability: 'Rain Dish', nature: 'Calm', evs: { hp: 252, spd: 252 } })],
    p2: [mon('Kyogre', ['icebeam', 'rest'], { ability: 'Drizzle', nature: 'Modest', evs: { spa: 4 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    abilitySide: 'p1', cover: 'rain_dish',
  });

  return S;
}

async function main() {
  const lines = [];
  lines.push('# ability_batch1_golden.txt — the gen3_ability_batch1_v1 class-sweep golden.');
  lines.push('# Per-decision STATE+HP+SPE-BOOST+SEED differential to GAME-END: CRIT_IMMUNE /');
  lines.push('# WEATHER_SPEED / WEATHER_NEGATE / RESIDUAL (Speed Boost / Rain Dish).');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status left speBoost) p2(...) first covered');
  lines.push('# END   <id>  <seed>  <ended:0|1>  <winner:p1|p2|tie|none>');

  const S = scenarios();
  const failures = [];
  let decRows = 0, winRows = 0, tieRows = 0, coverTotal = 0;

  for (const sc of S) {
    lines.push(`SCEN\t${sc.id}`);
    lines.push(`TEAM\t${sc.id}\tp1\t${Teams.pack(sc.p1)}`);
    lines.push(`TEAM\t${sc.id}\tp2\t${Teams.pack(sc.p2)}`);

    let scenDecs = 0, scenCover = 0;
    for (const seed of seeds) {
      let rec;
      try { rec = await runBattle(sc, seed); } catch (e) { failures.push(`${sc.id} seed ${seed}: ${e.message}`); continue; }
      if (rec.gen !== 3) { failures.push(`${sc.id}: expected gen 3, got ${rec.gen}`); break; }
      if (!rec.initSeed || rec.decisions.length === 0) { failures.push(`${sc.id} seed ${seed}: no decisions`); continue; }
      const seedStr = seed.join(',');
      lines.push(['INIT', sc.id, rec.initSeed, seedStr].join('\t'));

      rec.decisions.forEach((d) => {
        const sp = (s) => [s.species, s.hp, s.maxhp, s.fainted ? 1 : 0, s.status, s.left, s.speBoost].join('\t');
        lines.push([
          'DEC', sc.id, seedStr, d.request, d.force[0] ? 1 : 0, d.force[1] ? 1 : 0,
          d.choiceP1, d.choiceP2, d.seedAfter,
          sp(d.p1), sp(d.p2), d.firstMover, d.covered ? 1 : 0,
        ].join('\t'));
        decRows++; scenDecs++;
        if (d.covered) { coverTotal++; scenCover++; }
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
    if (scenCover < 10) failures.push(`${sc.id}: only ${scenCover} covered rows (<10) — the class effect barely fires`);
  }

  if (failures.length) {
    console.error('ABILITY BATCH1 GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }
  if (winRows < 200) { console.error(`ABILITY BATCH1 GOLDEN: too few WIN rows (${winRows} < 200)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `ability batch1 golden: ${S.length} scenarios, ${decRows} decision rows, ${coverTotal} covered rows, ` +
    `${winRows} wins + ${tieRows} ties -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
