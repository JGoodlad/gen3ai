// probe_batch4c_hyperbeam.js — ground-truth HYPER BEAM (id `hyperbeam`) bit-for-bit vs the
// OMNISCIENT in-process gen3 BattleStream (no server). Hyper Beam is the RECHARGE move:
// on a successful damaging hit the user gains `mustrecharge`; the NEXT turn is spent
// recharging (locked request), then the lock clears.
//
// The mod chain is the ONLY oracle. Probe the exact:
//   1. recharge TRIGGER conditions: hit vs MISS vs IMMUNE vs PROTECT-block vs SUB-absorb
//      vs SUB-break vs target-KO (gen3 may differ from later gens — probe each).
//   2. the recharge TURN's exact protocol lines (|cant|...|recharge| vs |move|...|recharge|)
//      + its DRAW COUNT (residual/eachEvent shuffles? Quick Claw? speed sort?).
//   3. the REQUEST on the locked turn — what moves[] contains, trapped?, canSwitch?
//      (can the user SWITCH instead of recharging in gen3? attempt it!)
//   4. statused (par / slp) user on the recharge turn — which cant wins, which draws run.
//   5. PP: does the recharge turn consume PP?
//   6. siblings in gen3: gigaimpact / rockwrecker (gen4+) vs blastburn/frenzyplant/hydrocannon.
//
// Run:  node src/rust_sim/harness/probe_batch4c_hyperbeam.js
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
  const m = d.moves.get('hyperbeam');
  console.log('=== resolved gen3 hyperbeam ===');
  console.log(`  cat=${m.category} bp=${m.basePower} acc=${m.accuracy} type=${m.type} target=${m.target} ` +
    `priority=${m.priority} pp=${m.pp} flags=${JSON.stringify(m.flags)}`);
  console.log(`  self=${JSON.stringify(m.self)} selfdestruct=${m.selfdestruct} secondary=${JSON.stringify(m.secondary)}`);
  for (const k of ['onModifyMove', 'onHit', 'onAfterHit', 'onAfterMoveSecondarySelf', 'onMoveFail', 'onAfterMove']) {
    if (m[k]) console.log(`  ${k} src: ${m[k].toString().replace(/\s+/g, ' ')}`);
  }
  // The mustrecharge + recharge condition sources as gen3 RESOLVES them.
  for (const cid of ['mustrecharge', 'recharge']) {
    const c = d.conditions.get(cid);
    console.log(`  --- condition ${cid}: exists=${c.exists} duration=${c.duration}`);
    for (const k of Object.keys(c)) {
      const v = c[k];
      if (typeof v === 'function') console.log(`      ${k}: ${v.toString().replace(/\s+/g, ' ')}`);
      else if (k !== 'name' && k !== 'exists' && k !== 'effectType' && !k.startsWith('sourceEffect')) {
        try { console.log(`      ${k}=${JSON.stringify(v)}`); } catch (e) {}
      }
    }
  }
  // The literal `recharge` MOVE id (the locked-turn pseudo-move) as gen3 resolves it.
  const rm = d.moves.get('recharge');
  console.log(`  --- move 'recharge': exists=${rm.exists} name=${rm.name} cat=${rm.category} bp=${rm.basePower} pp=${rm.pp} flags=${JSON.stringify(rm.flags)}`);
  // Siblings in gen3?
  console.log('  --- recharge-move siblings in gen3:');
  for (const id of ['gigaimpact', 'blastburn', 'frenzyplant', 'hydrocannon', 'rockwrecker', 'roaroftime']) {
    const s = Dex.mod('gen3').moves.get(id);
    console.log(`      ${id}: exists=${s.exists} gen=${s.gen} isNonstandard=${s.isNonstandard} bp=${s.basePower} cat=${s.category} self=${JSON.stringify(s.self)}`);
  }
  // Compare across gens (what does gen3 inherit / differ on?).
  for (const g of ['gen3', 'gen4', 'gen5']) {
    const dm = Dex.mod(g).moves.get('hyperbeam');
    console.log(`  [${g}] cat=${dm.category} bp=${dm.basePower} acc=${dm.accuracy} self=${JSON.stringify(dm.self)}`);
  }
}

function drawLabel() {
  const st = new Error().stack.split('\n');
  const frames = [];
  for (let i = 3; i < st.length && frames.length < 4; i++) {
    const mm = st[i].match(/at ([\w.<>]+) /);
    if (mm) frames.push(mm[1]);
  }
  return frames.join('<');
}

// plan entries: { p1, p2, p1try (illegal-choice attempt first), pre: [injections], stop }
// inject: { seed, acts: [{side, slot, status, hp, faint, item, boosts}] }
async function run(label, p1team, p2team, plan, inject) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  const p1log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  (async () => { for await (const ch of streams.p1) { for (const l of ch.split('\n')) if (l) p1log.push(l); } })();
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
      if (inj.statusData) Object.assign(m.statusState, inj.statusData);
      if (inj.hp !== undefined) m.hp = inj.hp;
      if (inj.faint) { m.hp = 0; m.fainted = true; }
      if (inj.item !== undefined) m.item = inj.item;
      if (inj.boosts) Object.assign(m.boosts, inj.boosts);
    }
  };
  applyActs(inject && inject.acts);

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);

  let draws = [];
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { const v = realNext(...a); draws.push(drawLabel()); return v; };

  const ppStr = (m) => m ? m.moveSlots.map((s) => `${s.id}:${s.pp}/${s.maxpp}`).join(',') : '-';
  let i = 0, safety = 0;
  while (!battle.ended && safety < 12) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    applyActs(entry.pre);
    // Snapshot p1's REQUEST as the sim serialized it (the bridge-visible shape).
    const req = battle.sides[0].activeRequest;
    console.log(`  [${rs}] p1 REQUEST: ${JSON.stringify(req && { active: req.active, forceSwitch: req.forceSwitch, wait: req.wait })}`);
    console.log(`        p1 volatiles=[${Object.keys(battle.sides[0].active[0] ? battle.sides[0].active[0].volatiles : {}).join(',')}] pp={${ppStr(battle.sides[0].active[0])}}`);
    draws = [];
    const logLen0 = log.length;
    const p1len0 = p1log.length;
    const before = battle.prng.getSeed();
    if (entry.p1try) {
      streams.omniscient.write(`>p1 ${entry.p1try}`);
      for (let k = 0; k < 10; k++) await tick();
      const errs = p1log.slice(p1len0).filter((l) => l.startsWith('|error|'));
      console.log(`  TRY ">p1 ${entry.p1try}" -> ${errs.length ? errs.join(' // ') : '(no error — ACCEPTED)'}`);
    }
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'} vols=[${Object.keys(m.volatiles).join(',')}]` : '-';
    console.log(`  [${rs}] ${JSON.stringify({ p1: entry.p1, p2: entry.p2 })} draws=${draws.length}  seed ${before}->${after}`);
    console.log(`        p1=${fmt(a0)}  pp={${ppStr(a0)}}`);
    console.log(`        p2=${fmt(a1)}`);
    draws.forEach((dl, k) => console.log(`        DRAW[${k}] ${dl}`));
    const newLines = log.slice(logLen0).filter((l) =>
      /\|move\||\|turn\||-damage|-heal|-boost|-unboost|-fail|-immune|-miss|-crit|-supereffective|-resisted|cant|-activate|-hitcount|-end\b|-start|switch|drag|faint|-prepare|-mustrecharge|-singleturn/.test(l));
    for (const l of newLines) console.log(`        LINE ${l}`);
    const errLines = p1log.slice(p1len0).filter((l) => l.startsWith('|error|'));
    for (const l of errLines) console.log(`        P1ERR ${l}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  dumpResolved();

  const wall = () => mon('Skarmory', ['spikes', 'protect', 'splash', 'roar'], { evs: { hp: 252 } });
  const wall2 = () => mon('Forretress', ['spikes', 'splash'], { evs: { hp: 252 } });
  // Seed [9,9,9,9]: the first turn-1 accuracy draw PASSES even a 30% check (probed) —
  // use it for every scenario that needs the turn-1 Hyper Beam to HIT. The default
  // [7,11,13,17] turn-1 acc draw FAILS 90 (a natural miss — used for the miss arm).
  const HIT = { seed: [9, 9, 9, 9] };

  // 1) BASELINE hit → recharge turn → free turn. Capture the recharge turn's protocol
  //    lines, its draw count, the locked request shape, and PP (does recharge cost PP?).
  await run('HB baseline: hit -> recharge turn -> free turn',
    [mon('Snorlax', ['hyperbeam', 'splash']), mon('Blissey', ['softboiled'])],
    [wall(), wall2()],
    [{ p1: 'move 1', p2: 'move 1' },       // hyper beam hits
     { p1: 'move 1', p2: 'move 1' },       // locked: recharge (what does 'move 1' resolve to?)
     { p1: 'move 1', p2: 'move 3', stop: true }],  // lock cleared? HB selectable again?
    HIT);

  // 2) MISS — does a missed Hyper Beam still require recharge in gen3?
  //    The default seed's turn-1 acc draw fails 90 naturally.
  await run('HB natural MISS (default seed) — recharge on miss?',
    [mon('Snorlax', ['hyperbeam', 'splash']), mon('Blissey', ['softboiled'])],
    [wall(), wall2()],
    [{ p1: 'move 1', p2: 'move 3' },
     { p1: 'move 1', p2: 'move 3', stop: true }]);

  // 3) IMMUNE — Normal Hyper Beam into a Ghost. Recharge? (note whether the acc draw
  //    still happens vs an immune target, and which of miss/immune wins the protocol)
  await run('HB into IMMUNE (Gengar) — recharge?',
    [mon('Snorlax', ['hyperbeam', 'splash']), mon('Blissey', ['softboiled'])],
    [mon('Gengar', ['splash', 'shadowball'], { ability: 'Levitate', evs: { hp: 252 } }), wall2()],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1', stop: true }],
    HIT);

  // 4) PROTECT block — recharge? (seed makes the acc draw pass so Protect actually blocks)
  await run('HB into PROTECT — recharge?',
    [mon('Snorlax', ['hyperbeam', 'splash']), mon('Blissey', ['softboiled'])],
    [wall(), wall2()],
    [{ p1: 'move 1', p2: 'move 2' },   // p2 protects
     { p1: 'move 1', p2: 'move 3', stop: true }],
    HIT);

  // 5) SUB ABSORB (sub survives) — weak user (Blissey, base Atk 10) vs Snorlax sub. Recharge?
  await run('HB into a SURVIVING SUB — recharge?',
    [mon('Blissey', ['hyperbeam', 'softboiled'])],
    [mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252 } }), wall2()],
    [{ p1: 'move 2', p2: 'move 1' },   // p2 subs
     { p1: 'move 1', p2: 'move 2' },   // HB into the sub (absorbed)
     { p1: 'move 1', p2: 'move 2', stop: true }],
    HIT);

  // 6) SUB BREAK — strong FASTER user breaks the sub. Recharge? Also: on the recharge
  //    turn the FASTER user's |cant| should print BEFORE the slower opp's move (order!).
  await run('HB BREAKS a sub (fast user; line order on the recharge turn)',
    [mon('Slaking', ['hyperbeam', 'splash'], { ability: 'No Ability' }), mon('Blissey', ['softboiled'])],
    [mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252 } }), wall2()],
    [{ p1: 'move 2', p2: 'move 1' },   // p2 subs (Slaking splashes)
     { p1: 'move 1', p2: 'move 2' },   // HB breaks the sub
     { p1: 'move 1', p2: 'move 2', stop: true }],  // recharge turn: cant BEFORE slower Snorlax?
    HIT);

  // 7) TARGET KO — HB KOs the target. Recharge still required after the replacement?
  await run('HB KOs the target — recharge next turn?',
    [mon('Snorlax', ['hyperbeam', 'splash']), mon('Blissey', ['softboiled'])],
    [wall(), wall2()],
    [{ p1: 'move 1', p2: 'move 1' },   // HB KOs (hp injected low)
     { p2: 'switch 2' },               // p2 replaces
     { p1: 'move 1', p2: 'move 2', stop: true }],  // p1 locked?
    { seed: [9, 9, 9, 9], acts: [{ side: 1, slot: 0, hp: 20 }] });

  // 8) PARALYZED user on the recharge turn — recharge cant (prio 11) vs par onBeforeMove:
  //    does the para check DRAW on the locked turn?
  await run('HB then user PARALYZED on the recharge turn',
    [mon('Snorlax', ['hyperbeam', 'splash']), mon('Blissey', ['softboiled'])],
    [wall(), wall2()],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 3', pre: [{ side: 0, slot: 0, status: 'par' }] },
     { p1: 'move 2', p2: 'move 3', stop: true }],
    HIT);

  // 9) ASLEEP user on the recharge turn — sleep cant vs recharge cant; is mustrecharge
  //    consumed by the sleep turn or does it persist?
  await run('HB then user ASLEEP on the recharge turn',
    [mon('Snorlax', ['hyperbeam', 'splash']), mon('Blissey', ['softboiled'])],
    [wall(), wall2()],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 3', pre: [{ side: 0, slot: 0, status: 'slp' }] },
     { p1: 'move 1', p2: 'move 3' },
     { p1: 'move 1', p2: 'move 3', stop: true }],
    HIT);

  // 10) SWITCH attempt on the locked turn — legal in gen3?
  await run('HB recharge turn: attempt a SWITCH (legal?)',
    [mon('Snorlax', ['hyperbeam', 'splash']), mon('Blissey', ['softboiled'])],
    [wall(), wall2()],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1try: 'switch 2', p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 3', stop: true }],
    HIT);

  // 10b) choice-string shapes on the locked turn: 'move 2' / 'move recharge' / 'move 1'.
  await run('HB recharge turn: attempt "move 2" then "move recharge"',
    [mon('Snorlax', ['hyperbeam', 'splash']), mon('Blissey', ['softboiled'])],
    [wall(), wall2()],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1try: 'move 2', p1: 'move recharge', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 3', stop: true }],
    HIT);

  // 11) QUICK CLAW on the opponent — does the locked turn still run the holder's QC draw?
  await run('HB recharge turn with opp QUICK CLAW (draws on the locked turn?)',
    [mon('Snorlax', ['hyperbeam', 'splash']), mon('Blissey', ['softboiled'])],
    [mon('Skarmory', ['spikes', 'protect', 'splash'], { item: 'Quick Claw', evs: { hp: 252 } }), wall2()],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 3' },
     { p1: 'move 1', p2: 'move 3', stop: true }],
    HIT);

  // 12) TRUANT + HB — mustrecharge.onBeforeMove also removes 'truant'. Does the recharge
  //     turn consume the loaf (i.e. can Truant Slaking act EVERY other turn with HB)?
  await run('HB by TRUANT Slaking — recharge turn vs the loaf turn',
    [mon('Slaking', ['hyperbeam', 'splash'], { ability: 'Truant' }), mon('Blissey', ['softboiled'])],
    [wall(), wall2()],
    [{ p1: 'move 1', p2: 'move 1' },   // HB hits
     { p1: 'move 1', p2: 'move 3' },   // recharge turn (truant would also loaf) — which cant?
     { p1: 'move 1', p2: 'move 3' },   // free? or loaf?
     { p1: 'move 1', p2: 'move 3', stop: true }],
    HIT);

  // 13) FLINCH on the recharge turn — faster opp Headbutt (30%). Sweep seeds for a proc;
  //     which cant wins on p1's action (recharge prio 11 vs flinch prio 8)?
  for (const seed of [[11, 22, 33, 44], [1, 1, 1, 1], [42, 42, 42, 42], [100, 200, 300, 400], [12, 34, 56, 78], [5, 4, 3, 2]]) {
    await run(`HB recharge turn vs faster HEADBUTT flinch (seed=${JSON.stringify(seed)})`,
      [mon('Snorlax', ['hyperbeam', 'splash']), mon('Blissey', ['softboiled'])],
      [mon('Jolteon', ['headbutt', 'splash'], { evs: { hp: 252, spe: 252 } }), wall2()],
      [{ p1: 'move 1', p2: 'move 2' },
       { p1: 'move 1', p2: 'move 1', stop: true }],
      { seed });
  }
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
