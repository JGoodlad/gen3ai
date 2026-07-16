// probe_snatch.js — ground-truth SNATCH bit-for-bit vs the OMNISCIENT in-process gen3
// BattleStream (no server). SNATCH is the SOLE remaining unmodeled gen3ou move.
//
// Questions this probe settles BEHAVIORALLY (the sim is the only oracle):
//   (1) the CAST draw model — Snatch's own priority, the volatile it sets, never-miss / any
//       acc draw, and casting into NOTHING (does the volatile just expire end of turn?).
//   (2) the STEAL mechanic + draw model — when the FOE uses a snatchable self-move, HOW is it
//       intercepted, the exact announce (|-activate|snatcher|move: Snatch|[of] foe then the
//       snatcher USES the stolen move), whose PP is spent, what the stolen move then DRAWS/does.
//   (3) the EXACT snatchable set — every gen3 move with flags.snatch.
//   (4) edge cases — cast-into-nothing (volatile duration), TWO snatchers, snatched Rest/Belly
//       Drum vs the SNATCHER's own state, snatcher fainting before the steal, a NON-snatchable
//       status move (Thunder Wave), Substitute/Protect interactions.
//
// Run:  node src/rust_sim/harness/probe_snatch.js
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
  const m = d.moves.get('snatch');
  console.log(`=== resolved gen3 snatch ===`);
  console.log(`  cat=${m.category} bp=${m.basePower} acc=${JSON.stringify(m.accuracy)} type=${m.type} prio=${m.priority} pp=${m.pp} target=${m.target}`);
  console.log(`  flags=${JSON.stringify(m.flags)} volatileStatus=${JSON.stringify(m.volatileStatus)}`);
  for (const k of Object.keys(m)) {
    const v = m[k];
    if (typeof v === 'function') console.log(`  fn ${k}: ${v.toString().replace(/\s+/g, ' ')}`);
  }
  if (m.condition) {
    console.log('  --- condition:');
    for (const k of Object.keys(m.condition)) {
      const v = m.condition[k];
      if (typeof v === 'function') console.log(`      fn ${k}: ${v.toString().replace(/\s+/g, ' ')}`);
      else console.log(`      ${k}=${JSON.stringify(v)}`);
    }
  }
  for (const g of ['gen4', 'gen9']) {
    const gm = Dex.mod(g).moves.get('snatch');
    console.log(`  [${g}] prio=${gm.prio ?? gm.priority} volatileStatus=${JSON.stringify(gm.volatileStatus)}`);
    const gc = gm.condition || {};
    for (const k of Object.keys(gc)) {
      const v = gc[k];
      if (typeof v === 'function') console.log(`      [${g}] fn ${k}: ${v.toString().replace(/\s+/g, ' ')}`);
      else console.log(`      [${g}] ${k}=${JSON.stringify(v)}`);
    }
  }

  // (3) the EXACT snatchable set — every gen3 move with flags.snatch.
  console.log('\n=== gen3 moves with flags.snatch (the eligible victim set) ===');
  const snatchable = [];
  for (const mv of d.moves.all()) {
    if (mv.flags && mv.flags.snatch) snatchable.push(mv);
  }
  snatchable.sort((a, b) => a.num - b.num);
  for (const mv of snatchable) {
    console.log(`  ${String(mv.num).padStart(3)} ${mv.id.padEnd(16)} cat=${mv.category.padEnd(6)} target=${mv.target.padEnd(14)} vol=${mv.volatileStatus || '-'} boosts=${JSON.stringify(mv.boosts || mv.self && mv.self.boosts || null)}`);
  }
  console.log(`  TOTAL snatchable (gen3): ${snatchable.length}`);
  console.log(`  ids: ${JSON.stringify(snatchable.map((x) => x.id))}`);
}

function drawLabel() {
  const st = new Error().stack.split('\n');
  const frames = [];
  for (let i = 3; i < st.length && frames.length < 6; i++) {
    const mm = st[i].match(/at ([\w.<>]+) /);
    if (mm) frames.push(mm[1]);
  }
  return frames.join('<');
}
const P32 = 2 ** 32;
const rTo = (val, from, to) => Math.floor((val * (to - from)) / P32) + from;

function volStr(m) {
  if (!m) return '-';
  return Object.entries(m.volatiles).map(([k, v]) => {
    const parts = [];
    for (const f of ['duration', 'counter', 'time', 'move', 'sourceSlot', 'hp']) {
      if (v[f] !== undefined) parts.push(`${f}=${JSON.stringify(v[f])}`);
    }
    return `${k}{${parts.join(',')}}`;
  }).join(' ') || '(none)';
}
function boostStr(m) {
  if (!m) return '-';
  const nz = Object.entries(m.boosts).filter(([, v]) => v !== 0);
  return nz.length ? nz.map(([k, v]) => `${k}${v > 0 ? '+' : ''}${v}`).join(',') : '·';
}
function reqStr(side) {
  const r = side.activeRequest;
  if (!r) return 'null';
  if (r.wait) return 'wait';
  if (r.forceSwitch) return 'forceSwitch';
  if (r.active) {
    const a = r.active[0];
    const mv = (a.moves || []).map((x) => `${x.id}${x.pp !== undefined ? `(${x.pp}pp)` : ''}${x.disabled ? '[DIS]' : ''}`).join(',');
    return `moves=[${mv}]${a.trapped ? ' TRAPPED' : ''}`;
  }
  return '?';
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
  const applyActs = (acts) => {
    for (const inj of acts || []) {
      const side = battle.sides[inj.side];
      const m = inj.slot === undefined ? side.active[0] : side.pokemon[inj.slot];
      if (!m) continue;
      if (inj.status) m.setStatus(inj.status, m, null, true);
      if (inj.hp !== undefined) m.hp = inj.hp;
      if (inj.faint) { m.hp = 0; m.fainted = true; }
      if (inj.item !== undefined) m.item = inj.item;
      if (inj.boosts) Object.assign(m.boosts, inj.boosts);
      if (inj.pp) for (const [k, v] of Object.entries(inj.pp)) m.moveSlots[Number(k)].pp = v;
    }
  };
  applyActs(inject && inject.acts);

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);

  let draws = [];
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { const v = realNext(...a); draws.push({ v, l: drawLabel() }); return v; };

  let i = 0, safety = 0;
  while (!battle.ended && safety < 30) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    applyActs(entry.pre);
    draws = [];
    const logLen0 = log.length;
    const before = battle.prng.getSeed();
    const hpBefore = battle.sides.map((s) => s.active[0] ? `${s.active[0].species.name}:${s.active[0].hp}/${s.active[0].maxhp}` : '-');
    if (entry.p1) { try { streams.omniscient.write(`>p1 ${entry.p1}`); } catch (e) {} }
    if (entry.p2) { try { streams.omniscient.write(`>p2 ${entry.p2}`); } catch (e) {} }
    for (let k = 0; k < 20; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const ppStr = (m) => m ? m.moveSlots.map((s) => `${s.id}:${s.pp}/${s.maxpp}`).join(',') : '-';
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'} boosts={${boostStr(m)}} pp={${ppStr(m)}}` : '-';
    console.log(`  [${rs}] ${JSON.stringify({ p1: entry.p1, p2: entry.p2 })} draws=${draws.length}  seed ${before}->${after}`);
    console.log(`        pre : p1=${hpBefore[0]}  p2=${hpBefore[1]}  left=${battle.sides.map((s) => s.pokemonLeft)}`);
    console.log(`        post: p1=${fmt(a0)}  vols: ${volStr(a0)}`);
    console.log(`              p2=${fmt(a1)}  vols: ${volStr(a1)}`);
    console.log(`        req : p1=${reqStr(battle.sides[0])}`);
    console.log(`              p2=${reqStr(battle.sides[1])}`);
    draws.forEach((d, k) => console.log(`        DRAW[${k}] val=${d.v} r(1,2)=${rTo(d.v, 1, 2)} r(2,6)=${rTo(d.v, 2, 6)}  ${d.l}`));
    const newLines = log.slice(logLen0).filter((l) =>
      /\|move\||\|turn\||-damage|-heal|-boost|-unboost|-fail|-immune|-miss|-crit|-supereffective|cant|-activate|-hitcount|-end\b|-start|-singlemove|-singleturn|switch|drag|faint|-prepare|-nothing|-enditem|-status|-curestatus|win\|/.test(l));
    for (const l of newLines) console.log(`        LINE ${l}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner === undefined ? 'undefined' : JSON.stringify(battle.winner)}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  dumpResolved();

  // ============================================================ (1) CAST DRAW MODEL
  // SN1: cast Snatch into NOTHING (foe attacks) — read the cast's own draws (never-miss? acc
  // roll?), the volatile it sets on the snatcher, then whether it EXPIRES at end of turn.
  for (const seed of [[7, 11, 13, 17], [21, 22, 23, 24], [101, 5, 9, 3]]) {
    await run(`SN1 cast Snatch into nothing (foe attacks) seed=${JSON.stringify(seed)}`,
      [mon('Umbreon', ['snatch', 'splash'], { evs: { hp: 252 } })],
      [mon('Skarmory', ['drillpeck', 'splash'], { evs: { hp: 252 } })],
      [{ p1: 'move 1', p2: 'move 1' },                        // t1: snatch cast, foe drillpecks — volatile up?
       { p1: 'move 2', p2: 'move 2', stop: true }],           // t2: volatile gone? (splash both)
      { seed });
  }

  // ============================================================ (2) STEAL MECHANIC
  // SN2: FAST snatcher (Jolteon 130 > Skarmory 70) — snatch cast, foe uses SWORDS DANCE. The
  // snatch intercepts BEFORE the foe boosts → the SNATCHER gets +2 atk, foe boosts nothing.
  // Read the announce lines, whose PP is spent, and the draw model of the stolen move.
  for (const seed of [[7, 11, 13, 17], [21, 22, 23, 24]]) {
    await run(`SN2 steal Swords Dance (FAST snatcher) seed=${JSON.stringify(seed)}`,
      [mon('Jolteon', ['snatch', 'splash'])],
      [mon('Skarmory', ['swordsdance', 'splash'], { evs: { hp: 252 } })],
      [{ p1: 'move 1', p2: 'move 1', stop: true }],
      { seed });
  }

  // SN3: SLOW snatcher (Snorlax 30 < Skarmory 70). Does snatch STILL intercept even though the
  // foe would move first by speed? (The volatile is set this turn; the foe's SD executes AFTER
  // — snatch is a reactive interception, not a priority race.)  Actually the snatcher must set
  // the volatile BEFORE the foe's move executes → snatch's OWN +4 priority. Test both orders.
  await run('SN3 steal Swords Dance (SLOW snatcher, snatch priority)',
    [mon('Snorlax', ['snatch', 'splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['swordsdance', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // SN4: steal RECOVER — snatcher damaged, foe recovers → the SNATCHER heals, foe heals nothing.
  await run('SN4 steal Recover (snatcher heals)',
    [mon('Umbreon', ['snatch', 'splash'], { evs: { hp: 252 } })],
    [mon('Blissey', ['softboiled', 'splash'], { evs: { hp: 252 } })],
    [{ pre: [{ side: 0, hp: 100 }, { side: 1, hp: 100 }], p1: 'move 1', p2: 'move 1', stop: true }]);

  // ============================================================ (4) EDGE CASES
  // SN5: steal REST — Rest puts the SNATCHER to sleep + full heal (the snatcher's own state).
  await run('SN5 steal Rest (snatcher sleeps + heals)',
    [mon('Umbreon', ['snatch', 'splash'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['rest', 'splash'], { evs: { hp: 252 } })],
    [{ pre: [{ side: 0, hp: 100 }], p1: 'move 1', p2: 'move 1', stop: true }]);

  // SN6: steal BELLY DRUM — costs the SNATCHER's hp, boosts the SNATCHER's atk +6/+12.
  await run('SN6 steal Belly Drum (costs snatcher hp)',
    [mon('Snorlax', ['snatch', 'splash'], { evs: { hp: 252 } })],
    [mon('Poliwrath', ['bellydrum', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // SN7: a NON-snatchable status move (Thunder Wave — no flags.snatch) → NOT stolen; the foe
  // paralyzes the snatcher normally.
  await run('SN7 Thunder Wave is NOT snatchable',
    [mon('Umbreon', ['snatch', 'splash'], { evs: { hp: 252 } })],
    [mon('Jolteon', ['thunderwave', 'splash'])],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // SN8: TWO snatchers on the same side is impossible in singles, but test snatch cast then the
  // foe SWITCHES (no snatchable move) → volatile expires. Also a foe status-cure move (Heal
  // Bell / Aromatherapy) — is a TEAM-cure snatched? (flags.snatch? the snatcher cures ITS team.)
  await run('SN8 steal Aromatherapy (team-cure — snatcher cures ITS side)',
    [mon('Umbreon', ['snatch', 'splash'], { evs: { hp: 252 } }), mon('Gengar', ['splash'], { ability: 'Levitate' })],
    [mon('Blissey', ['aromatherapy', 'splash'], { evs: { hp: 252 } }), mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [{ pre: [{ side: 0, slot: 1, status: 'brn' }, { side: 1, slot: 1, status: 'brn' }], p1: 'move 1', p2: 'move 1', stop: true }]);

  // SN9: steal SUBSTITUTE — does the SNATCHER build the sub (costs the snatcher's hp)?
  await run('SN9 steal Substitute (snatcher builds sub, costs snatcher hp)',
    [mon('Snorlax', ['snatch', 'splash'], { evs: { hp: 252 } })],
    [mon('Gengar', ['substitute', 'splash'], { ability: 'Levitate', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // SN10: snatcher FAINTS before the steal — snatch cast t1; t2 snatcher at 1 hp, foe (faster)
  // KOs it, then... the snatch volatile is gone with the fainted mon (no steal). Confirm.
  await run('SN10 snatcher fainted before the steal (no steal)',
    [mon('Umbreon', ['snatch', 'splash'], { evs: { hp: 252 } }), mon('Gengar', ['splash'], { ability: 'Levitate' })],
    [mon('Jolteon', ['swordsdance', 'thunderbolt'])],
    [{ p1: 'move 1', p2: 'move 2' },                          // t1: snatch cast; jolteon tbolt (snatch NOT triggered — attack)
     { pre: [{ side: 0, hp: 1 }], p1: 'move 2', p2: 'move 1' }, // t2: jolteon (faster) SD — but snatch volatile from t1 is GONE (end of t1)
     { p1: 'switch 2', p2: 'move 2', stop: true }]);

  // SN11: steal SPIKES? — Spikes is foeSide target (NOT self) → should NOT be snatchable
  // (no flags.snatch). Confirm foe lays spikes normally.
  await run('SN11 Spikes is NOT snatchable (foeSide, not self)',
    [mon('Umbreon', ['snatch', 'splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['spikes', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // SN12: re-cast — snatch cast t1 into nothing; t2 both snatch + a snatchable move at a SPEED
  // TIE? Simpler: snatch cast, foe casts snatch too — does snatch STEAL snatch? (snatch has
  // flags.snatch? check the dump.) Mirror snatch at a tie.
  await run('SN12 snatch mirror (does snatch steal snatch?)',
    [mon('Umbreon', ['snatch', 'splash'], { evs: { hp: 252 } })],
    [mon('Umbreon', ['snatch', 'swordsdance'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },                          // both snatch
     { p1: 'move 2', p2: 'move 2', stop: true }]);            // t2: p1 splash, p2 SD — both snatch volatiles gone?

  // SN13: steal WISH — a snatched Wish → the snatcher gets the pending wish (heals the
  // snatcher's slot next turn).
  await run('SN13 steal Wish',
    [mon('Umbreon', ['snatch', 'splash'], { evs: { hp: 252 } })],
    [mon('Blissey', ['wish', 'splash'], { evs: { hp: 252 } })],
    [{ pre: [{ side: 0, hp: 100 }], p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 2', stop: true }]);            // t2: wish resolves on snatcher's side?
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });