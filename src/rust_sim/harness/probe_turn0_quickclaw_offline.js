// probe_turn0_quickclaw_offline.js — settle WHY the offline `ab_replay` path diverges at
// DECISION 0 on a Quick-Claw lead, against the OMNISCIENT in-process BattleStream.
//
// WHY: the `--mode random --protocol --master-seed 100125 --battles 1000` benchmark leaves 3
// divergences, ALL at dec=0. The `firstmover` one (rmsc7a22r_ab_10_12) reads:
//     expected p2 / got Some(0)   "wrong first mover on a move turn (SEED MATCHES)"
// p1 leads Charizard (base spe 100), p2 leads Dusclops (base spe 25) holding a QUICK CLAW. So
// the sim's turn-1 Quick Claw fired and the port's did not — while the recorded dec-0 seed
// AGREED, which is the tell: the port is starting from the right seed but the WRONG boolean.
//
// THE HYPOTHESIS. `Battle.quickClawRoll` is `randomChance(1,5)` rolled at every COMPLETED
// `endTurn` and READ on the NEXT turn (`gen3_quick_claw_speed_v1`). Turn 1's value is therefore
// rolled during the TURN-0 CONSTRUCTION WINDOW — the gender samples + the two `runSwitch`
// actions + `endTurn` that run inside `>player`, BEFORE the first decision request. But the
// offline recorder captures `initSeed` AT the first decision request (gen_e2e_fuzz.js:1448,
// `if (decisionNo === 0) rec.initSeed = seedBefore`) — i.e. POST-construction — and the port's
// offline entry `start_with_switchins` is deliberately DRAW-FREE. So the port resumes with the
// correct seed and `quick_claw_roll: false` (its `BattleState::start` default), losing a bit the
// sim had already decided. A Quick-Claw lead is then wrong ~1 time in 5.
//
// This is the KNOWN, DOCUMENTED turn-0 construction deferral on the offline path (the BRIDGE
// already models it in full via `start_with_turn0_construction` / `gen3_turn0_construction_v1`);
// what is NOT established is that it is what these repros actually are. Assert, don't assume.
//
// WHAT THIS PRINTS, per repro, from the RAW `>start` seed in its summary.json:
//   * the construction draw COUNT and the post-construction seed, CHECKED against the repro's
//     recorded INIT seed (if these disagree the whole theory is wrong, so it is a hard gate);
//   * `battle.quickClawRoll` as the sim holds it at the first decision request;
//   * which side actually moved first on turn 1, and whether a Quick Claw holder is leading.
// The prediction under the hypothesis: quickClawRoll === true on ab_10_12, p2 leads holding a
// Quick Claw, and the two OTHER (kind=seed) repros are a DIFFERENT construction shape — the
// count of construction draws, not the boolean.
//
// Run:  node src/rust_sim/harness/probe_turn0_quickclaw_offline.js
'use strict';
const path = require('path');
const fs = require('fs');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));

const REPROS = path.join(__dirname, 'ab_fuzz_out/tail1/divergences');
const tick = () => new Promise((r) => setTimeout(r, 0));

async function probe(dir) {
  const sum = JSON.parse(fs.readFileSync(path.join(REPROS, dir, 'summary.json'), 'utf8'));
  const [p1Packed, p2Packed] = [sum.packed_teams.p1, sum.packed_teams.p2];

  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();

  // Count EVERY construction draw by hooking the live prng's single underlying primitive
  // BEFORE `>player` (which is where the construction window runs).
  streams.omniscient.write(`>start {"formatid":"${sum.fmt || 'gen3customgame'}","seed":${JSON.stringify(sum.battle_seed)}}`);
  for (let i = 0; i < 4; i++) await tick();
  const battle = stream.battle;
  let draws = 0;
  const origNext = battle.prng.rng.next.bind(battle.prng.rng);
  battle.prng.rng.next = (...a) => { draws++; return origNext(...a); };

  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1Packed })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2Packed })}`);
  for (let i = 0; i < 14; i++) await tick();

  const constructionDraws = draws;
  // `getSeed()` returns the backend-native form (a string on the gen5 backend, an array on
  // older ones) — normalize to the comma form the repro's INIT line uses.
  const rawSeed = battle.prng.getSeed();
  const postSeed = (Array.isArray(rawSeed) ? rawSeed.join(',') : String(rawSeed)).replace(/^gen5,/, '');
  const qcr = battle.quickClawRoll;
  const lead = (s) => {
    const a = battle.sides[s].active[0];
    return { name: a.name, item: a.item || '(none)', spe: a.getStat('spe') };
  };
  const l1 = lead(0), l2 = lead(1);

  // Play ONLY decision 0 with the repro's own recorded choices, then read who moved first.
  const c0 = sum.choices && sum.choices[0];
  if (c0) {
    if (c0.p1) streams.omniscient.write(`>p1 ${c0.p1}`);
    if (c0.p2) streams.omniscient.write(`>p2 ${c0.p2}`);
    for (let k = 0; k < 20; k++) await tick();
  }
  const firstMove = log.find((l) => l.startsWith('|move|'));
  const firstMover = firstMove ? firstMove.split('|')[2].slice(0, 3) : '(none)';

  const initMatches = postSeed === sum.init_seed;
  console.log(`\n=== ${dir}  (${sum.first_divergence.kind}) ===`);
  console.log(`    raw >start seed        : ${sum.battle_seed.join(',')}`);
  console.log(`    construction draws     : ${constructionDraws}`);
  console.log(`    post-construction seed : ${postSeed}`);
  console.log(`    repro INIT seed        : ${sum.init_seed}   ${initMatches ? 'MATCH — the recorder captures POST-construction' : '*** MISMATCH — theory broken ***'}`);
  console.log(`    battle.quickClawRoll   : ${qcr}   <-- turn 1 reads THIS, rolled during construction`);
  console.log(`    p1 lead: ${l1.name} item=${l1.item} spe=${l1.spe}`);
  console.log(`    p2 lead: ${l2.name} item=${l2.item} spe=${l2.spe}`);
  console.log(`    turn-1 first mover     : ${firstMover}  (repro expected ${sum.first_divergence.expected})`);
  const qcHolder = [l1, l2].some((l) => l.item.toLowerCase().replace(/[^a-z]/g, '') === 'quickclaw');
  console.log(`    a Quick Claw is leading: ${qcHolder}`);
  return { dir, kind: sum.first_divergence.kind, constructionDraws, initMatches, qcr, qcHolder };
}

(async () => {
  const dirs = fs.readdirSync(REPROS).filter((d) => fs.existsSync(path.join(REPROS, d, 'summary.json')));
  const out = [];
  for (const d of dirs) out.push(await probe(d));

  console.log('\n--- READ ---');
  for (const r of out) {
    console.log(`    ${r.dir}  kind=${r.kind}  constructionDraws=${r.constructionDraws}`
      + `  quickClawRoll=${r.qcr}  quickClawLead=${r.qcHolder}  initSeedMatches=${r.initMatches}`);
  }
  console.log('');
  console.log('    The offline port entry `start_with_switchins` is DRAW-FREE and defaults');
  console.log('    `quick_claw_roll: false`, so ANY repro with quickClawRoll=true AND a Quick');
  console.log('    Claw lead is mis-ordered on turn 1 by construction, not by a draw bug.');
  console.log('    A construction draw COUNT > the modeled window explains the kind=seed pair.');
})();
