// probe_residual_order_rng.js — GROUND-TRUTH for the residual-ordering engine bugs
// (#3a faintMessages+if(ended)return, #6 gather order) by INJECTING the exact
// post-move board (burn + low HP) into the omniscient sim, running ONE residual
// turn, and reading the post-turn HP + PRNG seed. Both Gengar use Tackle (Normal →
// Ghost-immune, 0 damage) so the residual is the SOLE HP change. The Rust regression
// test reproduces the identical injected board + asserts the same post-turn HP/seed.
//
//   #3a — a FAST burned mon at 1 HP self-KOs via burn DoT (status DoT sub 6, but
//         SPEED outranks subOrder) and, being its side's last mon, ENDS the game →
//         the SLOWER foe's Leftovers (sub 4) never ticks. Assert the foe's HP is
//         UNCHANGED (under the bug — all handlers then one faint pass — the foe heals).
//   #6  — a 2-mon burn+Leftovers SPEED TIE: the residual handler-sort's Fisher-Yates
//         tie-shuffle permutes handlers in their PRE-SORT (gather) order, so the
//         status DoT must be gathered BEFORE Leftovers per mon (findPokemonEventHandlers
//         gathers status→item). One Gengar is at low HP so a wrong gather order runs
//         OUR DoT before the foe's game-ending DoT → we get chipped when we should
//         stay clean. We capture the exact post-turn HP + seed at the injected board.
//
// Run:  node src/rust_sim/harness/probe_residual_order_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

// runMulti: like run() but plays a multi-decision PLAN (each entry {p1,p2}) after an
// optional one-time inject, logging seedBefore/After + the active speeds per decision.
// Used for the #7 forced-replacement scenario (inject p1 lead to low HP so p2 KOs it).
async function runMulti(label, seed, p1team, p2team, inject, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  for (const inj of inject) {
    const m = battle.sides[inj.side].active[0];
    if (inj.status) m.setStatus(inj.status, m, null, true);
    if (inj.hp !== undefined) m.hp = inj.hp;
  }
  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };
  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 40) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const before = battle.prng.getSeed();
    const dc0 = drawCount;
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'}${m.fainted ? ' FNT' : ''} spe=${m.getStat('spe')}` : '-';
    console.log(`  [${rs}] ${JSON.stringify(entry)} draws=${drawCount - dc0}  seedBefore=${before} seedAfter=${after}`);
    console.log(`        p1=${fmt(a0)} | p2=${fmt(a1)}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

// inject: a list of {side, hp, status} applied to each side's active BEFORE the turn.
async function run(label, seed, p1team, p2team, inject, choices) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;

  // Inject the post-move board directly onto the sim's mon objects.
  for (const inj of inject) {
    const m = battle.sides[inj.side].active[0];
    if (inj.status) m.setStatus(inj.status, m, null, true);
    if (inj.hp !== undefined) m.hp = inj.hp;
  }

  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  const before = battle.prng.getSeed();
  const m0 = battle.sides[0].active[0], m1 = battle.sides[1].active[0];
  console.log(`\n=== ${label} ===`);
  console.log(`  initSeed=${before}`);
  console.log(`  injected: p1=${m0.species.name} ${m0.hp}/${m0.maxhp} ${m0.status || '-'} spe=${m0.getStat('spe')} | ` +
              `p2=${m1.species.name} ${m1.hp}/${m1.maxhp} ${m1.status || '-'} spe=${m1.getStat('spe')}`);

  if (choices.p1) streams.omniscient.write(`>p1 ${choices.p1}`);
  if (choices.p2) streams.omniscient.write(`>p2 ${choices.p2}`);
  for (let k = 0; k < 24; k++) await tick();

  const after = battle.prng.getSeed();
  const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
  const fmt = (m) => `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'}${m.fainted ? ' FNT' : ''}`;
  console.log(`  draws=${drawCount}`);
  console.log(`  seedAfter=${after}`);
  console.log(`  p1=${fmt(a0)}\n  p2=${fmt(a1)}`);
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // #3a — FAST burned Gengar at 1 HP (its side's only mon) self-KOs via burn DoT;
  // the game ends → the SLOWER foe Gengar's Leftovers never ticks. Foe is GHOST too so
  // Tackle is immune BOTH ways (the residual is the SOLE HP change). Distinct speed
  // (p1 Timid 252 spe vs p2 Brave 0 spe) → no tie-shuffle (clean). Foe pre-chipped so a
  // Leftovers tick (had it wrongly run) would be observable.
  await run('#3a burn self-KO ends before foe Leftovers',
    [2, 3, 5, 7],
    [mon('Gengar', ['tackle'], { item: 'Leftovers', evs: { spe: 252 }, nature: 'Timid' })],
    [mon('Gengar', ['tackle'], { item: 'Leftovers', evs: {}, nature: 'Brave' })],
    [{ side: 0, hp: 1, status: 'brn' }, { side: 1, hp: 200 }],
    { p1: 'move 1', p2: 'move 1' });

  // #6 — the residual GATHER-ORDER tie. A Gengar mirror under SAND (Tyranitar-free:
  // injected weather), both burned + Leftovers at IDENTICAL speed (tie). The FOE (p2,
  // its side's last mon) is at a burn-DoT-lethal HP so its DoT ENDS the game; OUR mon
  // (p1) is at full HP. The residual handler-sort ties the two mons' DoT handlers; the
  // Fisher-Yates shuffle permutes them IN GATHER ORDER, deciding whether the foe's
  // game-ending DoT runs before our DoT (so we stay un-chipped) or after (we get
  // chipped first). Tackle is Ghost-immune both ways → the residual is the only change.
  await run('#6 residual gather order: sand+burn 2-mon tie, foe DoT ends game',
    [2, 3, 5, 7],
    [mon('Gengar', ['tackle'], { item: 'Leftovers', ability: 'Sand Stream', evs: { hp: 252, spe: 252 }, nature: 'Timid' })],
    [mon('Gengar', ['tackle'], { item: 'Leftovers', evs: { hp: 252, spe: 252 }, nature: 'Timid' })],
    [{ side: 0, hp: 324, status: 'brn' }, { side: 1, hp: 40, status: 'brn' }],
    { p1: 'move 1', p2: 'move 1' });

  // #7 — forced-replacement updateSpeed. p1 lead Jirachi (faster, Timid) is injected to
  // a low HP and Thunder-Waves p2 Jirachi THIS turn (para'd mid-turn); p2 Psychic KOs
  // the low-HP p1 lead → p1 forced to replace with a fresh full-speed Jirachi-B. The
  // resumed tail + next turn's `eachEvent('Update')` shuffles compare p2 (para'd this
  // turn: live 82, but STALE full 328 if not re-cached) with B (328). The fix re-caches
  // p2 to 82 on the forced commit → no tie; the bug leaves it at 328 → a phantom tie →
  // an extra shuffle draw → a divergent seed. We capture every boundary's seed.
  for (const hp0 of [8, 12, 16, 20, 25, 30]) {
    await runMulti(`#7 forced-replacement updateSpeed (inject p1 lead hp=${hp0})`,
      [4, 8, 15, 16],
      [mon('Jirachi', ['thunderwave', 'swift'], { evs: { hp: 252, spe: 252 }, nature: 'Timid' }),
       mon('Jirachi', ['swift'], { evs: { hp: 252, spe: 252 }, nature: 'Timid' })],
      [mon('Jirachi', ['swift', 'thunderwave'], { evs: { hp: 252, spe: 252 }, nature: 'Timid' })], // STALE full 328 == B
      [{ side: 0, hp: hp0 }],
      [
        { p1: 'move 1', p2: 'move 1' }, // p1-A TWaves p2 (para mid-turn) ; p2 Swift KOs the low-HP A → p1 replace
        { p1: 'switch 2', p2: 'move 1' }, // forced: p1 brings Jirachi-B (328) vs para'd p2 (stale 328 / live 82)
        { p1: 'move 1', p2: 'move 1' }, // both Swift — tie count B(328) vs p2(para 82)
        { p1: 'move 1', p2: 'move 1' },
      ]);
  }
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
