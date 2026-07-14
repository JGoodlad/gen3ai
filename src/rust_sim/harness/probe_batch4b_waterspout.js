// probe_batch4b_waterspout.js — ground-truth WATER SPOUT (id `waterspout`) bit-for-bit vs the
// OMNISCIENT in-process BattleStream (no server). Water Spout is a VARIABLE-BP SPECIAL move:
//   - base BP 150 Water (SPECIAL), acc 100 (NOT never-miss), target allAdjacentFoes (singles = 1)
//   - basePowerCallback: `bp = move.basePower * pokemon.hp / pokemon.maxhp` (a FLOAT), then getDamage
//     does `basePower = clampIntRange(basePower, 1)` = `max(Math.floor(150*hp/maxhp), 1)`.
//   - the BP is computed INSIDE getDamage BEFORE the crit roll + BEFORE the damage random(16) roll
//     (deterministic state read — NO extra draw). No onModifyMove / no secondary.
//
// The mod chain is the ONLY oracle. Probe the exact:
//   1. the variable-BP formula at BOUNDARY HP values (full → bp 150, half → ~75, low HP where
//      150*hp/maxhp < 1 → does it FLOOR to 1 (min) or FAIL?), and whether it FLOORS the float.
//   2. that this is the ONLY change vs a normal special move — normal acc/crit/damage draws, same
//      count + order.
//   3. edge cases: at full HP, at low HP, into a Substitute (variable number hits the sub?),
//      immunity (Water Absorb), and BP computed before/after the damage-roll draw.
//
// Run:  node src/rust_sim/harness/probe_batch4b_waterspout.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));

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

function dumpResolved() {
  const d = Dex.forFormat(FORMAT);
  const m = d.moves.get('waterspout');
  console.log('=== resolved gen3 waterspout ===');
  console.log(`  cat=${m.category} bp=${m.basePower} acc=${m.accuracy} type=${m.type} target=${m.target} ` +
    `priority=${m.priority} flags=${JSON.stringify(m.flags)} never_miss=${m.accuracy === true}`);
  console.log(`  basePowerCallback src: ${m.basePowerCallback.toString().replace(/\s+/g, ' ')}`);
  console.log(`  onModifyMove: ${m.onModifyMove ? m.onModifyMove.toString().replace(/\s+/g, ' ') : '(none)'}`);
  console.log(`  secondary: ${JSON.stringify(m.secondary)}`);
  // Deterministic BP table (the port must reproduce): bp = max(floor(150*hp/maxhp), 1)
  console.log('  BP table (150*hp/maxhp floored, min 1):');
  for (const [hp, maxhp] of [[404, 404], [303, 404], [202, 404], [101, 404], [1, 404], [1, 200], [201, 301], [150, 301]]) {
    const bp = Math.max(Math.floor(150 * hp / maxhp), 1);
    console.log(`    hp=${hp}/${maxhp} -> bp=${bp}  (raw=${(150 * hp / maxhp).toFixed(4)})`);
  }
}

// A short call-site label from the PRNG-draw stack.
function drawLabel() {
  const st = new Error().stack.split('\n');
  const frames = [];
  for (let i = 3; i < st.length && frames.length < 3; i++) {
    const mm = st[i].match(/at ([\w.<>]+) /);
    if (mm) frames.push(mm[1]);
  }
  return frames.join('<');
}

async function run(label, p1team, p2team, plan, inject) {
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
    const m = inj.side === undefined ? null : battle.sides[inj.side].active[0];
    if (!m) continue;
    if (inj.status) m.setStatus(inj.status, m, null, true);
    if (inj.hp !== undefined) m.hp = inj.hp;
    if (inj.item !== undefined) m.item = inj.item;
  }

  let draws = [];
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { const v = realNext(...a); draws.push(drawLabel()); return v; };

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 6) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    draws = [];
    const logLen0 = log.length;
    const before = battle.prng.getSeed();
    const a0b = battle.sides[0].active[0];
    const a1b = battle.sides[1].active[0];
    const attHp = a0b ? `${a0b.hp}/${a0b.maxhp}` : '-';
    const defHp0 = a1b ? a1b.hp : 0;
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'} vols=[${Object.keys(m.volatiles).join(',')}]` : '-';
    const dmgDealt = a1 ? (defHp0 - a1.hp) : 0;
    console.log(`  [${rs}] ${JSON.stringify(entry)}  p1atkHP=${attHp}  draws=${draws.length}  seed ${before}->${after}`);
    console.log(`        p1=${fmt(a0)}`);
    console.log(`        p2=${fmt(a1)}  (dmgDealt=${dmgDealt})`);
    draws.forEach((dl, k) => console.log(`        DRAW[${k}] ${dl}`));
    const newLines = log.slice(logLen0).filter((l) =>
      /\|move\||-damage|-heal|-boost|-unboost|-fail|-immune|-crit|-supereffective|-resisted|-activate|-end|-start|switch|faint|debug\|BP/.test(l));
    for (const l of newLines) console.log(`        LINE ${l}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  dumpResolved();

  // 1) FULL HP — bp 150, a plain special hit: acc + crit + damage draws (the baseline draw model).
  await run('WATER SPOUT full HP (bp 150): plain special, acc+crit+dmg draws',
    [mon('Kyogre', ['waterspout', 'surf'], { evs: { spa: 252 } })],
    [mon('Snorlax', ['bodyslam'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // 2) HALF HP — bp floor(150*hp/maxhp). Inject the attacker to ~half → bp ~75. Same draw count.
  await run('WATER SPOUT half HP: bp ~75 (150*hp/maxhp floored)',
    [mon('Kyogre', ['waterspout', 'surf'], { evs: { spa: 252 } })],
    [mon('Snorlax', ['bodyslam'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 0, hp: 170 }] });  // Kyogre maxhp 341 -> ~half -> bp floor(150*170/341)=74

  // 3) LOW HP (1 HP, large maxhp) — 150*1/maxhp < 1 → clampIntRange floors to min BP 1.
  //    Does it hit for min damage (BP 1) or FAIL? Draw model unchanged (acc+crit+dmg).
  await run('WATER SPOUT at 1 HP: 150/maxhp < 1 -> BP floors to 1 (min), still hits',
    [mon('Kyogre', ['waterspout', 'surf'], { evs: { spa: 252 } })],
    [mon('Snorlax', ['bodyslam'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 0, hp: 1 }] });

  // 4) INTO A SUBSTITUTE — the variable BP hits the SUB's HP normally (breaks/absorbs). Draw count
  //    is the normal acc+crit+dmg (a damaging special move into a sub).
  await run('WATER SPOUT into a Substitute: variable BP hits the sub HP',
    [mon('Kyogre', ['waterspout', 'surf'], { evs: { spa: 252 } })],
    [mon('Snorlax', ['substitute', 'bodyslam'], { evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2', stop: true }]);

  // 5) WATER ABSORB immunity — accuracy drawn THEN absorbed (heal, no damage). Confirm draw count.
  await run('WATER SPOUT into WATER ABSORB: acc drawn then absorbed (heal, no dmg)',
    [mon('Kyogre', ['waterspout', 'surf'], { evs: { spa: 252 } })],
    [mon('Vaporeon', ['softboiled'], { ability: 'Water Absorb', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 1, hp: 100 }] });

  // 6) BP-vs-DAMAGE-ROLL ORDER: at a half-HP attacker, dump the debug BP line + the draw stack so we
  //    can see the BP is read BEFORE the random(16) damage draw (a deterministic state read).
  await run('WATER SPOUT order proof: BP read before the random(16) damage draw',
    [mon('Kyogre', ['waterspout', 'surf'], { evs: { spa: 252 } })],
    [mon('Blissey', ['softboiled'], { evs: { hp: 252, def: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 0, hp: 200 }] });
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
