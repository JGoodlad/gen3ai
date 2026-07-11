// probe_naturalcure_isolate.js — TIGHT draw-count isolation for the NC cure + the
// sleep-counter / tox-stage reset detail. THE PROBE IS THE ONLY ORACLE.
//
// The main probe used a DOUBLE-switch turn (both sides switch → the 2nd runSwitch draws
// the order-101 splice). To PROVE the cure itself is draw-free with ZERO other switch
// machinery, this uses a SINGLE p1 switch while p2 uses a NON-drawing move against a
// switch (a Recover into full HP is a self-move → draws nothing for the switch turn; the
// switch itself is draw-free — the runSwitch is strictly ordered before the leftover move,
// no tie). Then NC vs non-NC draw counts must be EXACTLY equal AND (ideally) 0.
//
// Run:  node src/rust_sim/harness/probe_naturalcure_isolate.js
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

async function run(p1team, p2team, plan, opts = {}) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  const seed = opts.seed || [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  for (const inj of (opts.inject || [])) {
    const m = battle.sides[inj.side].active[0];
    if (!m) continue;
    if (inj.status) m.setStatus(inj.status, m, null, true);
    if (inj.tox !== undefined && m.statusState) m.statusState.stage = inj.tox;
  }
  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = (...a) => { drawCount += 1; return realNext(...a); };
  const steps = [];
  for (const step of plan) {
    const before = drawCount;
    if (step.p1) streams.omniscient.write(`>p1 ${step.p1}`);
    if (step.p2) streams.omniscient.write(`>p2 ${step.p2}`);
    for (let k = 0; k < 12; k++) await tick();
    steps.push(drawCount - before);
  }
  return { battle, steps };
}

(async () => {
  console.log('=== TIGHT single-switch draw isolation (p1 switches, p2 self-Recovers) ===');
  // p2 uses Recover (self-move, no accuracy/damage draw); p1 switches its statused mon out.
  // A single switch's runSwitch is strictly ordered before the leftover move → NO splice.
  // So the WHOLE turn should draw only whatever the moves draw. We pick a status where the
  // switch turn is otherwise draw-clean.
  const STATUSES = ['brn', 'par', 'psn', 'tox', 'slp', 'frz'];
  for (const st of STATUSES) {
    const nc = [mon('Starmie', ['recover', 'surf'], { ability: 'Natural Cure' }),
      mon('Skarmory', ['spikes'], { ability: 'Keen Eye' })];
    const nonNc = [mon('Starmie', ['recover', 'surf'], { ability: 'Illuminate' }),
      mon('Skarmory', ['spikes'], { ability: 'Keen Eye' })];
    // p2 slower so its Recover is second; give p2 Recover so it self-targets (no draw vs switch).
    const p2 = [mon('Blissey', ['softboiled'], { ability: 'Serene Grace' })];
    const tox = st === 'tox' ? 6 : undefined;
    const a = await run(nc, p2, [{ p1: 'switch 2', p2: 'move 1', note: 'NC switch out' }],
      { inject: [{ side: 0, status: st, tox }], seed: [3, 5, 7, 9] });
    const b = await run(nonNc, p2, [{ p1: 'switch 2', p2: 'move 1', note: 'non-NC switch out' }],
      { inject: [{ side: 0, status: st, tox }], seed: [3, 5, 7, 9] });
    const findStar = (bt) => bt.sides[0].pokemon.find((p) => p.species.name === 'Starmie');
    const ncS = findStar(a.battle);
    const bS = findStar(b.battle);
    // Also assert the post-switch PRNG SEED is byte-identical between NC and non-NC.
    const seedEq = JSON.stringify(a.battle.prng.seed) === JSON.stringify(b.battle.prng.seed);
    console.log(`  ${st.toUpperCase()}: NC draws=${a.steps[0]} status='${ncS.status || ''}'  |  nonNC draws=${b.steps[0]} status='${bS.status || ''}'  seedEqual=${seedEq}  postSeed=${JSON.stringify(a.battle.prng.seed)}`);
  }

  console.log('\n=== SLEEP COUNTER + TOX STAGE reset detail (setStatus("") wipes statusState) ===');
  // Inject sleep with a specific counter, switch out (NC), switch back in, dump the counter.
  {
    const nc = [mon('Starmie', ['recover', 'surf'], { ability: 'Natural Cure' }),
      mon('Skarmory', ['spikes'], { ability: 'Keen Eye' })];
    const p2 = [mon('Blissey', ['softboiled'], { ability: 'Serene Grace' })];
    const a = await run(nc, p2,
      [{ p1: 'switch 2', p2: 'move 1', note: 'sleeping NC switches out' },
        { p1: 'switch 2', p2: 'move 1', note: 'NC switches back in' }],
      { inject: [{ side: 0, status: 'slp' }], seed: [3, 5, 7, 9] });
    const star = a.battle.sides[0].pokemon.find((p) => p.species.name === 'Starmie');
    const ss = star.statusState || {};
    console.log(`  after switch-out+back: Starmie status='${star.status || ''}' active=${star.isActive} statusState=${JSON.stringify(ss)}  (cured ⇒ '' and empty statusState)`);
  }

  console.log('\n=== -curestatus emission on the OUTGOING mon (is it [silent]?) ===');
  // Capture the protocol lines around the switch to confirm the -curestatus [silent] shape.
  {
    const stream = new BattleStream();
    const streams = getPlayerStreams(stream);
    const lines = [];
    (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l.startsWith('|')) lines.push(l); } })();
    streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":[3,5,7,9]}`);
    streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([mon('Starmie', ['recover', 'surf'], { ability: 'Natural Cure' }), mon('Skarmory', ['spikes'], { ability: 'Keen Eye' })]) })}`);
    streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([mon('Blissey', ['softboiled'], { ability: 'Serene Grace' })]) })}`);
    for (let i = 0; i < 12; i++) await tick();
    const b = stream.battle;
    b.sides[0].active[0].setStatus('brn', b.sides[0].active[0], null, true);
    streams.omniscient.write('>p1 switch 2');
    streams.omniscient.write('>p2 move 1');
    for (let k = 0; k < 12; k++) await tick();
    console.log('  protocol lines around the NC switch:');
    for (const l of lines) if (l.includes('curestatus') || l.includes('switch') || l.includes('-heal') || l.includes('|move|')) console.log('    ', l);
  }
})();
