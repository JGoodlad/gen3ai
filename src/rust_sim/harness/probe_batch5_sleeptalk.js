// probe_batch5_sleeptalk.js — ground-truth SLEEP TALK (id `sleeptalk`) bit-for-bit vs the
// OMNISCIENT in-process gen3 BattleStream (no server). Sleep Talk: usable ONLY while asleep;
// picks ONE of the user's OTHER moves at RANDOM (a sample draw!) and executes it.
//
// The mod chain is the ONLY oracle. Probe the exact:
//   1. eligible-move pool: the gen3 exclusion list (Sleep Talk itself? Rest? charge moves?
//      Focus Punch? 0-PP moves — in or out?), and the pool ORDER (moveSlots order?).
//   2. the EXACT sample draw: one random(n) over the pool? drawn even at n=1?
//      empty pool → fail, with how many draws?
//   3. ordering vs the sleep cant: slp counter decrements first (draw-free?); the WAKE
//      turn (time hits 0 this turn) — does Sleep Talk still run or fail?
//   4. PP: Sleep Talk's own PP consumed; the picked move's PP NOT? failed-use PP?
//   5. the picked move's execution: full normal draw model? a picked Solar Beam —
//      charges? a picked Rest? a Choice Band lock interaction?
//   6. protocol lines: |move|..|Sleep Talk| then |move|..|<picked>|..|[from]move: Sleep Talk?
//
// Run:  node src/rust_sim/harness/probe_batch5_sleeptalk.js
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
  const m = d.moves.get('sleeptalk');
  console.log('=== resolved gen3 sleeptalk ===');
  console.log(`  cat=${m.category} bp=${m.basePower} acc=${m.accuracy} type=${m.type} target=${m.target} ` +
    `priority=${m.priority} pp=${m.pp} flags=${JSON.stringify(m.flags)} sleepUsable=${m.sleepUsable}`);
  for (const k of Object.keys(m)) {
    const v = m[k];
    if (typeof v === 'function') console.log(`  ${k} src: ${v.toString().replace(/\s+/g, ' ')}`);
  }
  console.log(`  noSleepTalk(field)=${JSON.stringify(m.noSleepTalk)}`);
  // How do the candidate moves resolve in gen3 (flags that could exclude them)?
  console.log('  --- candidate-move flags in gen3:');
  for (const id of ['rest', 'solarbeam', 'skullbash', 'focuspunch', 'bide', 'uproar', 'metronome',
                    'mirrormove', 'assist', 'sketch', 'mimic', 'thrash', 'splash', 'bodyslam',
                    'doubleteam', 'curse', 'hyperbeam', 'fly', 'dig', 'razorwind']) {
    const s = d.moves.get(id);
    console.log(`      ${id}: exists=${s.exists} gen=${s.gen} cat=${s.category} bp=${s.basePower} ` +
      `flags=${JSON.stringify(s.flags)} sleepUsable=${s.sleepUsable}`);
  }
  // Cross-gen comparison of the sleeptalk callback (gen3 inherits through the mod chain).
  for (const g of ['gen3', 'gen4', 'gen5']) {
    const dm = Dex.mod(g).moves.get('sleeptalk');
    const src = (dm.onHit || dm.onTryHit || '').toString().replace(/\s+/g, ' ');
    console.log(`  [${g}] onTry=${(dm.onTry || '').toString().replace(/\s+/g, ' ')}`);
    console.log(`  [${g}] onTryHit/onHit: ${src}`);
  }
  // The slp condition as gen3 resolves it (the cant / decrement / sleepUsable path).
  const slp = d.conditions.get('slp');
  console.log('  --- condition slp (gen3 resolved):');
  for (const k of Object.keys(slp)) {
    const v = slp[k];
    if (typeof v === 'function') console.log(`      ${k}: ${v.toString().replace(/\s+/g, ' ')}`);
  }
}

function drawLabel() {
  const st = new Error().stack.split('\n');
  const frames = [];
  for (let i = 3; i < st.length && frames.length < 5; i++) {
    const mm = st[i].match(/at ([\w.<>]+) /);
    if (mm) frames.push(mm[1]);
  }
  return frames.join('<');
}

// plan entries: { p1, p2, p1try, pre: [injections], stop }
// inject: { seed, acts: [{side, slot, status, statusData, hp, faint, item, boosts, movepp}] }
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
      if (inj.movepp) {
        for (const [mid, pp] of Object.entries(inj.movepp)) {
          const s = m.moveSlots.find((x) => x.id === mid);
          if (s) s.pp = pp;
        }
      }
    }
  };
  applyActs(inject && inject.acts);

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);

  let draws = [];
  let rnds = [];
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { const v = realNext(...a); draws.push(drawLabel()); return v; };
  // Semantic layer: capture every prng.random(...) call's args + result (sample routes here).
  const prng = battle.prng;
  const realRandom = prng.random.bind(prng);
  prng.random = function (...a) { const v = realRandom(...a); rnds.push(`random(${a.map(String).join(',')})=${v}`); return v; };

  const ppStr = (m) => m ? m.moveSlots.map((s) => `${s.id}:${s.pp}/${s.maxpp}`).join(',') : '-';
  const stStr = (m) => m ? `${m.status || '-'}(time=${m.statusState && m.statusState.time})` : '-';
  let i = 0, safety = 0;
  while (!battle.ended && safety < 14) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    applyActs(entry.pre);
    const req = battle.sides[0].activeRequest;
    console.log(`  [${rs}] p1 REQUEST: ${JSON.stringify(req && { active: req.active, forceSwitch: req.forceSwitch, wait: req.wait })}`);
    console.log(`        p1 pre: ${stStr(battle.sides[0].active[0])} pp={${ppStr(battle.sides[0].active[0])}} vols=[${Object.keys(battle.sides[0].active[0] ? battle.sides[0].active[0].volatiles : {}).join(',')}]`);
    draws = []; rnds = [];
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
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${stStr(m)} vols=[${Object.keys(m.volatiles).join(',')}]` : '-';
    console.log(`  [${rs}] ${JSON.stringify({ p1: entry.p1, p2: entry.p2 })} draws=${draws.length}  seed ${before}->${after}`);
    console.log(`        p1=${fmt(a0)}  pp={${ppStr(a0)}}`);
    console.log(`        p2=${fmt(a1)}`);
    draws.forEach((dl, k) => console.log(`        DRAW[${k}] ${dl}`));
    rnds.forEach((rl, k) => console.log(`        RND[${k}] ${rl}`));
    const newLines = log.slice(logLen0).filter((l) =>
      /\|move\||\|turn\||-damage|-heal|-boost|-unboost|-fail|-immune|-miss|-crit|-supereffective|-resisted|cant|-activate|-hitcount|-end\b|-start|switch|drag|faint|-prepare|-mustrecharge|-singleturn|-status|-curestatus|-anim/.test(l));
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

  const wall = () => mon('Skarmory', ['splash', 'spikes'], { evs: { hp: 252 } });
  const SLP5 = { side: 0, slot: 0, status: 'slp', statusData: { time: 5, startTime: 5 } };

  // 1) BASELINE pool + sample draw, multi-seed. Pool = 3 status/no-draw moves (splash,
  //    curse, doubleteam) so any turn draw beyond the sample itself is visible. Which
  //    move fires per seed → the random(n)→index mapping over moveSlots order.
  for (const s of [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [7, 11, 13, 17]]) {
    await run(`BASELINE pick (seed ${JSON.stringify(s)}): pool=splash,curse,doubleteam`,
      [mon('Snorlax', ['sleeptalk', 'splash', 'curse', 'doubleteam'])],
      [wall()],
      [{ p1: 'move 1', p2: 'move 1' },
       { p1: 'move 1', p2: 'move 1', stop: true }],
      { seed: s, acts: [SLP5] });
  }

  // 2) n=1 pool — is the sample draw made when only ONE move is eligible?
  await run('n=1 pool: [sleeptalk, splash] — draw at n=1?',
    [mon('Snorlax', ['sleeptalk', 'splash'])],
    [wall()],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [SLP5] });

  // 3) EMPTY pool — Sleep Talk as the ONLY move. Fail? Draw count? PP consumed?
  await run('EMPTY pool: [sleeptalk] only',
    [mon('Snorlax', ['sleeptalk'])],
    [wall()],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [SLP5] });

  // 4) REST in the pool — eligible in gen3? Low HP so a picked Rest visibly heals +
  //    resets the sleep counter. Multi-seed to see both picks.
  for (const s of [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]) {
    await run(`REST eligibility (seed ${JSON.stringify(s)}): [sleeptalk, rest] hp=100`,
      [mon('Snorlax', ['sleeptalk', 'rest'])],
      [wall()],
      [{ p1: 'move 1', p2: 'move 1', stop: true }],
      { seed: s, acts: [{ ...SLP5, hp: 100 }] });
  }

  // 5) CHARGE move in the pool — [sleeptalk, solarbeam]: excluded? If picked: does it
  //    fire in ONE turn or |-prepare| charge? What happens next turn?
  for (const s of [[1, 1, 1, 1], [2, 2, 2, 2]]) {
    await run(`SOLARBEAM (seed ${JSON.stringify(s)}): [sleeptalk, solarbeam]`,
      [mon('Snorlax', ['sleeptalk', 'solarbeam'])],
      [wall()],
      [{ p1: 'move 1', p2: 'move 1' },
       { p1: 'move 1', p2: 'move 1', stop: true }],
      { seed: s, acts: [SLP5] });
  }

  // 6) FOCUS PUNCH in the pool — excluded in gen3?
  await run('FOCUSPUNCH: [sleeptalk, focuspunch]',
    [mon('Snorlax', ['sleeptalk', 'focuspunch'])],
    [wall()],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [SLP5] });

  // 7) 0-PP move in the pool — [sleeptalk, splash(pp=0), curse]: is splash still
  //    sampleable? Multi-seed; watch the random(n) arg (n=2 vs n=3).
  for (const s of [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [5, 5, 5, 5]]) {
    await run(`0-PP pool member (seed ${JSON.stringify(s)}): splash pp=0`,
      [mon('Snorlax', ['sleeptalk', 'splash', 'curse'])],
      [wall()],
      [{ p1: 'move 1', p2: 'move 1', stop: true }],
      { seed: s, acts: [{ ...SLP5, movepp: { splash: 0 } }] });
  }

  // 8) DAMAGING pick — the full draw model of the picked move (acc? crit? damage roll?
  //    secondary?). Pool = [sleeptalk, bodyslam] (n=1 → always bodyslam).
  await run('DAMAGING pick: [sleeptalk, bodyslam] — full draw model of the picked move',
    [mon('Snorlax', ['sleeptalk', 'bodyslam'])],
    [mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [SLP5] });

  // 9) WAKE turn — slp time=1: the decrement hits 0 THIS turn. Does the user wake and
  //    Sleep Talk then FAIL (awake), or does it still execute?
  await run('WAKE turn: slp time=1, use Sleep Talk',
    [mon('Snorlax', ['sleeptalk', 'splash', 'curse', 'doubleteam'])],
    [wall()],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 0, slot: 0, status: 'slp', statusData: { time: 1, startTime: 3 } }] });

  // 10) AWAKE user (no status) — Sleep Talk fails? Draw count? PP consumed on the fail?
  await run('AWAKE user: Sleep Talk with no status',
    [mon('Snorlax', ['sleeptalk', 'splash', 'curse', 'doubleteam'])],
    [wall()],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // 11) CHOICE BAND — asleep CB user picks via Sleep Talk: does the CALLED move set the
  //     choice lock, or Sleep Talk itself? Inspect the NEXT request.
  await run('CHOICE BAND lock: [sleeptalk, splash, bodyslam] + Choice Band',
    [mon('Snorlax', ['sleeptalk', 'splash', 'bodyslam'], { item: 'choiceband' })],
    [wall()],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [SLP5] });

  // 12) STILL-ASLEEP ordering — pool of one status move; is there a |cant| line BEFORE
  //     |move|Sleep Talk, and does the slp time decrement draw-free? (covered by every
  //     scenario's LINE dump — this one pins a 3-turn sequence: asleep, asleep, wake.)
  await run('3-turn sequence: slp time=3 → talk, talk, wake-turn',
    [mon('Snorlax', ['sleeptalk', 'curse'])],
    [wall()],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 0, slot: 0, status: 'slp', statusData: { time: 3, startTime: 3 } }] });
}

main().catch((e) => { console.error(e); process.exit(1); });
