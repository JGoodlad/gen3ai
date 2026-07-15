// probe_batch5_varbp.js — ground-truth the VARIABLE-BP basePowerCallback family bit-for-bit vs
// the OMNISCIENT in-process gen3 BattleStream (no server): RETURN / FRUSTRATION (happiness-scaled),
// FLAIL / REVERSAL (HP-ratio ladder), LOW KICK (weight ladder). All carry dex basePower 0, so the
// port currently routes them Status → fail-loud. All are expected to be deterministic STATE reads
// (the Water Spout precedent — NO extra draw) — but PROBE, never assume.
//
// The mod chain is the ONLY oracle. Probe the exact:
//   1. each resolved gen3 basePowerCallback SOURCE + the exact integer math:
//      - Return  bp = f(happiness)   (floor(h*10/25)? what at h=0 — clamp to 1, or getDamage's
//        `if (!basePower) return undefined` FAIL path?)
//      - Frustration bp = f(255-h)
//      - Flail/Reversal: the 48*hp/maxhp band ladder — sweep EVERY hp for the exact thresholds
//      - Low Kick: the weight ladder — sweep weighthg for the exact kg cutoffs
//   2. where HAPPINESS comes from (the packed-team codec field; default 255; the `return102`
//      alias) and where WEIGHT comes from (dex weighthg; getWeight()).
//   3. draw-NEUTRALITY: same seed, different BP (full-vs-1-HP Flail, h255-vs-h3 Return,
//      heavy-vs-light Low Kick) → IDENTICAL post-turn seed (acc+crit+dmg+QC only).
//   4. edges: Return/Flail/Low Kick into a GHOST (immune — acc-only draws), Flail/Reversal crit
//      in gen3 (historically crit-less in gen2 — dump willCrit + sweep seeds for a real |-crit|),
//      Return at happiness 0/1/2 (bp floor→0 — fail? which draws?).
//
// Run:  node src/rust_sim/harness/probe_batch5_varbp.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
const MOVES = ['return', 'frustration', 'flail', 'reversal', 'lowkick'];

function mon(species, moves, opts = {}) {
  const s = {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: opts.gender || 'N',
  };
  if (opts.happiness !== undefined) s.happiness = opts.happiness;
  return s;
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

// ---------------------------------------------------------------------------
// 1) The RESOLVED gen3 move sources (the mod chain applied) + cross-gen diffs.
// ---------------------------------------------------------------------------
function dumpResolved() {
  const d = Dex.forFormat(FORMAT);
  console.log('=== resolved gen3 variable-BP move family ===');
  for (const id of MOVES) {
    const m = d.moves.get(id);
    console.log(`--- ${id}: cat=${m.category} bp=${m.basePower} acc=${m.accuracy} type=${m.type} ` +
      `prio=${m.priority} pp=${m.pp} critRatio=${m.critRatio} willCrit=${m.willCrit} ` +
      `secondary=${JSON.stringify(m.secondary)} secondaries=${JSON.stringify(m.secondaries)} flags=${JSON.stringify(m.flags)}`);
    if (m.basePowerCallback) console.log(`    basePowerCallback: ${m.basePowerCallback.toString().replace(/\s+/g, ' ')}`);
    for (const k of ['onModifyMove', 'onHit', 'onTry', 'onTryHit', 'onMoveFail', 'noDamageVariance']) {
      if (m[k] !== undefined) console.log(`    ${k}: ${typeof m[k] === 'function' ? m[k].toString().replace(/\s+/g, ' ') : JSON.stringify(m[k])}`);
    }
  }
  // Cross-gen: does gen3 flail/reversal crit like gen2 (never) or normally?
  for (const id of ['flail', 'reversal']) {
    for (const g of ['gen2', 'gen3', 'gen4']) {
      const m = Dex.mod(g).moves.get(id);
      console.log(`  [${g}] ${id}: willCrit=${m.willCrit} critRatio=${m.critRatio} noDamageVariance=${m.noDamageVariance} ` +
        `bpcb=${m.basePowerCallback ? m.basePowerCallback.toString().replace(/\s+/g, ' ').slice(0, 140) : '-'}`);
    }
  }
  // The alias: return102 (the codec shorthand the task names).
  const alias = d.moves.get('return102');
  console.log(`  alias return102 -> exists=${alias.exists} id=${alias.id} name=${alias.name}`);
  const fr = d.moves.get('frustration');
  console.log(`  frustration exists=${fr.exists} id=${fr.id}`);
}

// ---------------------------------------------------------------------------
// 2) EXACT BP tables by invoking the RESOLVED callbacks on live/stub objects.
//    (Still the sim's own code computing — the oracle, not a re-derivation.)
// ---------------------------------------------------------------------------
async function dumpBpTables() {
  // Need a live battle for `this` (battle.debug / clampIntRange) + a live pokemon for flail.
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { /* drain */ } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":[1,2,3,4]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([mon('Snorlax', ['flail', 'return', 'frustration', 'lowkick'])]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([mon('Skarmory', ['splash'])]) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  const d = battle.dex;
  const poke = battle.sides[0].active[0]; // live Snorlax

  // FLAIL / REVERSAL: sweep every hp, print each band edge with the raw 48*hp/maxhp.
  for (const id of ['flail', 'reversal']) {
    const cb = d.moves.get(id).basePowerCallback;
    const maxhp = poke.maxhp;
    console.log(`\n=== ${id} BP ladder (live maxhp=${maxhp}) — band edges ===`);
    let last = null;
    for (let hp = 1; hp <= maxhp; hp++) {
      poke.hp = hp;
      const bp = cb.call(battle, poke, battle.sides[1].active[0], d.getActiveMove(id));
      if (bp !== last) {
        const ratio48 = 48 * hp / maxhp;
        console.log(`  hp=${hp}/${maxhp} (48*hp/maxhp=${ratio48.toFixed(4)}) -> bp=${bp}   [edge]`);
        // also print the last hp of the PREVIOUS band for the exact boundary
        if (hp > 1) {
          const prevRatio = 48 * (hp - 1) / maxhp;
          console.log(`      (prev hp=${hp - 1} ratio=${prevRatio.toFixed(4)} bp=${last})`);
        }
        last = bp;
      }
    }
    poke.hp = poke.maxhp;
  }

  // RETURN / FRUSTRATION: sweep happiness 0..255, print edges + the named points.
  for (const id of ['return', 'frustration']) {
    const cb = d.moves.get(id).basePowerCallback;
    console.log(`\n=== ${id} BP vs happiness — edges + key points ===`);
    const key = new Set([0, 1, 2, 3, 63, 64, 102, 128, 254, 255]);
    let last = null;
    for (let h = 0; h <= 255; h++) {
      poke.happiness = h;
      const bp = cb.call(battle, poke, battle.sides[1].active[0], d.getActiveMove(id));
      if (h <= 6 || key.has(h) || bp !== last) {
        if (bp !== last || key.has(h) || h <= 6) {
          if (h <= 6 || key.has(h)) console.log(`  happiness=${h} -> bp=${bp}`);
        }
        last = bp;
      }
    }
    // min/max summary
    poke.happiness = 255;
    const bp255 = cb.call(battle, poke, null, d.getActiveMove(id));
    poke.happiness = 0;
    const bp0 = cb.call(battle, poke, null, d.getActiveMove(id));
    console.log(`  summary: h=255 -> ${bp255}, h=0 -> ${bp0}`);
  }
  poke.happiness = 255;

  // LOW KICK: the callback reads target.getWeight() — sweep weighthg over a stub for exact cutoffs.
  {
    const cb = d.moves.get('lowkick').basePowerCallback;
    console.log(`\n=== lowkick BP vs target weighthg — exact cutoffs (stub getWeight) ===`);
    let last = null;
    for (let w = 1; w <= 2100; w++) {
      const stub = { getWeight: () => w, name: 'stub' };
      const bp = cb.call(battle, poke, stub, d.getActiveMove('lowkick'));
      if (bp !== last) {
        console.log(`  weighthg=${w} (${(w / 10).toFixed(1)}kg) -> bp=${bp}   [edge; prev w=${w - 1} bp=${last}]`);
        last = bp;
      }
    }
    // Real species weights (the dex weighthg source the port needs).
    console.log('  real species getWeight/weighthg + BP:');
    for (const s of ['Pichu', 'Gastly', 'Pikachu', 'Wobbuffet', 'Breloom', 'Gengar', 'Slaking', 'Snorlax', 'Aggron', 'Groudon', 'Wailord']) {
      const sp = d.species.get(s);
      const stub = { getWeight: () => sp.weighthg, name: s };
      const bp = cb.call(battle, poke, stub, d.getActiveMove('lowkick'));
      console.log(`    ${s}: weighthg=${sp.weighthg} (${sp.weightkg}kg) -> bp=${bp}`);
    }
    // Does the LIVE Pokemon.getWeight fold anything beyond species weighthg in gen3?
    const skarm = battle.sides[1].active[0];
    console.log(`  live Skarmory getWeight()=${skarm.getWeight()} vs species weighthg=${d.species.get('Skarmory').weighthg}`);
    console.log(`  Pokemon.getWeight src: ${skarm.getWeight.toString().replace(/\s+/g, ' ')}`);
  }

  // HAPPINESS plumbing: the packed-team codec field + the Pokemon ctor default.
  {
    console.log('\n=== happiness plumbing (codec + ctor) ===');
    const t = [mon('Snorlax', ['return'], { happiness: 3 })];
    const packed = Teams.pack(t);
    console.log(`  pack(happiness:3): ${packed}`);
    const un = Teams.unpack(packed);
    console.log(`  unpack round-trip happiness=${un[0].happiness}`);
    const t2 = [mon('Snorlax', ['return'])]; // happiness omitted
    const packed2 = Teams.pack(t2);
    console.log(`  pack(happiness omitted): ${packed2}`);
    console.log(`  live default (omitted in team): pokemon.happiness=${poke.happiness}, set.happiness=${JSON.stringify(poke.set.happiness)}`);
    // happiness 0 must survive the codec (0 vs omitted-default-255 — the classic falsy trap)
    const t3 = [mon('Snorlax', ['return'], { happiness: 0 })];
    const p3 = Teams.pack(t3);
    console.log(`  pack(happiness:0): ${p3}`);
    console.log(`  unpack(happiness:0) -> ${JSON.stringify(Teams.unpack(p3)[0].happiness)}`);
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

// ---------------------------------------------------------------------------
// 3) Battle probes: draw model + edges. run() returns data so main can compare
//    post-turn seeds across paired runs (the draw-neutrality proof).
// ---------------------------------------------------------------------------
function drawLabel() {
  const st = new Error().stack.split('\n');
  const frames = [];
  for (let i = 3; i < st.length && frames.length < 4; i++) {
    const mm = st[i].match(/at ([\w.<>]+) /);
    if (mm) frames.push(mm[1]);
  }
  return frames.join('<');
}

async function run(label, p1team, p2team, plan, inject, quiet) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  const seed = (inject && inject.seed) || [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  for (const inj of ((inject && inject.acts) || [])) {
    const m = battle.sides[inj.side].active[0];
    if (!m) continue;
    if (inj.status) m.setStatus(inj.status, m, null, true);
    if (inj.hp !== undefined) m.hp = inj.hp;
    if (inj.happiness !== undefined) m.happiness = inj.happiness;
  }

  let draws = [];
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { const v = realNext(...a); draws.push(drawLabel()); return v; };

  if (!quiet) console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  const result = { label, turns: [] };
  let i = 0, safety = 0;
  while (!battle.ended && safety < 8) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    draws = [];
    const logLen0 = log.length;
    const before = battle.prng.getSeed();
    const a1b = battle.sides[1].active[0];
    const defHp0 = a1b ? a1b.hp : 0;
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'}` : '-';
    const dmgDealt = a1 ? (defHp0 - a1.hp) : defHp0;
    const lines = log.slice(logLen0).filter((l) =>
      /\|move\||-damage|-heal|-fail|-immune|-miss|-crit|-supereffective|-resisted|-activate|-end\b|-start|faint|cant/.test(l));
    result.turns.push({ before, after, draws: draws.slice(), dmgDealt, lines });
    if (!quiet) {
      console.log(`  [${rs}] ${JSON.stringify({ p1: entry.p1, p2: entry.p2 })} draws=${draws.length}  seed ${before} -> ${after}`);
      console.log(`        p1=${fmt(a0)}  p2=${fmt(a1)}  dmgDealt=${dmgDealt}`);
      draws.forEach((dl, k) => console.log(`        DRAW[${k}] ${dl}`));
      for (const l of lines) console.log(`        LINE ${l}`);
    }
    if (entry.stop) break;
  }
  if (!quiet) console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
  return result;
}

function compareSeeds(tag, a, b) {
  const sa = a.turns[a.turns.length - 1].after, sb = b.turns[b.turns.length - 1].after;
  const da = a.turns[a.turns.length - 1].draws.length, db = b.turns[b.turns.length - 1].draws.length;
  console.log(`\n*** DRAW-NEUTRALITY ${tag}: seedA=${sa} seedB=${sb} -> ${sa === sb ? 'IDENTICAL (draw-neutral ✓)' : 'DIFFERENT (NOT neutral ✗)'}  (draws ${da} vs ${db}; dmg ${a.turns[a.turns.length - 1].dmgDealt} vs ${b.turns[b.turns.length - 1].dmgDealt})`);
}

async function main() {
  dumpResolved();
  await dumpBpTables();

  const wall = () => mon('Skarmory', ['splash', 'spikes'], { evs: { hp: 252 } });
  const lax = () => mon('Snorlax', ['splash', 'bodyslam'], { evs: { hp: 252 } });

  // --- RETURN ---------------------------------------------------------------
  // baseline h=255 (bp 102) vs h=3 (bp 1): same seed, same board — the draw-neutrality pair.
  const r255 = await run('RETURN h=255 (bp 102) baseline draw model',
    [mon('Tauros', ['return', 'frustration'])], [lax()],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);
  const r3 = await run('RETURN h=3 (bp 1) — packed-team happiness, same seed',
    [mon('Tauros', ['return', 'frustration'], { happiness: 3 })], [lax()],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);
  compareSeeds('RETURN h255 vs h3', r255, r3);

  // h=0 → callback bp 0 — FAIL path? which draws?
  await run('RETURN h=0 (bp 0) — fail? draw count?',
    [mon('Tauros', ['return', 'frustration'], { happiness: 0 })], [lax()],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // Return (Normal) into a GHOST — immune: acc-only draws expected.
  await run('RETURN into GHOST (Gengar) — immune draw model',
    [mon('Tauros', ['return'])],
    [mon('Gengar', ['splash', 'shadowball'], { ability: 'Levitate', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // --- FRUSTRATION ----------------------------------------------------------
  const f0 = await run('FRUSTRATION h=0 (bp 102)',
    [mon('Tauros', ['frustration'], { happiness: 0 })], [lax()],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);
  const f252 = await run('FRUSTRATION h=252 (bp 1) — same seed',
    [mon('Tauros', ['frustration'], { happiness: 252 })], [lax()],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);
  compareSeeds('FRUSTRATION h0 vs h252', f0, f252);
  await run('FRUSTRATION h=255 (bp 0) — fail? draw count?',
    [mon('Tauros', ['frustration'], { happiness: 255 })], [lax()],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // --- FLAIL / REVERSAL -----------------------------------------------------
  const flFull = await run('FLAIL at full HP (bp 20)',
    [mon('Snorlax', ['flail'], { evs: { hp: 252 } })], [wall()],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);
  const fl1 = await run('FLAIL at 1 HP (bp 200) — same seed',
    [mon('Snorlax', ['flail'], { evs: { hp: 252 } })], [wall()],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 0, hp: 1 }] });
  compareSeeds('FLAIL full vs 1HP', flFull, fl1);

  await run('REVERSAL (Fighting) into GHOST (Sableye) — immune draw model',
    [mon('Hitmonlee', ['reversal'])],
    [mon('Sableye', ['splash', 'shadowball'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // Flail/Reversal CRIT in gen3? Sweep seeds hunting a |-crit| (gen2 famously can't crit).
  for (const id of ['flail', 'reversal']) {
    let crits = 0, hits = 0;
    for (let s = 1; s <= 40; s++) {
      const r = await run(`crit sweep`, // quiet
        [mon(id === 'flail' ? 'Snorlax' : 'Hitmonlee', [id], { evs: { hp: 252 } })],
        [mon('Slaking', ['splash'], { evs: { hp: 252 } })],
        [{ p1: 'move 1', p2: 'move 1', stop: true }],
        { seed: [s, s * 3 + 1, s * 7 + 2, s * 11 + 3] }, true);
      const t = r.turns[0];
      if (t.lines.some((l) => l.includes('|move|') && l.includes('|[miss]'))) continue;
      hits++;
      if (t.lines.some((l) => l.startsWith('|-crit|'))) crits++;
    }
    console.log(`\n*** ${id} CRIT SWEEP (40 seeds): hits=${hits} crits=${crits} -> ${crits > 0 ? 'CAN crit in gen3' : 'NO crit observed (suspicious — check willCrit)'}`);
  }

  // --- LOW KICK ---------------------------------------------------------------
  const lkHeavy = await run('LOW KICK vs Snorlax (460kg -> bp 120)',
    [mon('Machamp', ['lowkick'])], [lax()],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);
  const lkLight = await run('LOW KICK vs Wobbuffet (28.5kg -> bp 60) — same seed',
    [mon('Machamp', ['lowkick'])],
    [mon('Wobbuffet', ['splash', 'counter'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);
  compareSeeds('LOW KICK heavy vs light', lkHeavy, lkLight);

  await run('LOW KICK (Fighting) into GHOST (Sableye) — immune draw model',
    [mon('Machamp', ['lowkick'])],
    [mon('Sableye', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // --- SUBSTITUTE spot-check (variable number should hit the sub normally) ----
  await run('FLAIL at 1 HP into a SUBSTITUTE — bp 200 hits the sub',
    [mon('Snorlax', ['flail'], { evs: { hp: 252 } })],
    [mon('Slaking', ['substitute', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2', stop: true }],
    { acts: [{ side: 0, hp: 1 }] });
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
