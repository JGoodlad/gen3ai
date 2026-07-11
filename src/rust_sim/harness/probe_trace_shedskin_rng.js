// probe_trace_shedskin_rng.js — settle TRACE + SHED SKIN against the resolved gen3 sim
// (Mandate 1 — the sim is the only oracle).
//
// TRACE cruxes:
//   (T1) the n=1 DRAW: gen3 trace.onStart runs `pokemon.side.randomFoe()` — does the
//        `battle.sample` DRAW with a single foe (the phaze-n=1 gotcha)?
//   (T2) copy semantics: `setAbility` in gen3 (`battle.gen > 3` gate) — the copied
//        ability's onStart must NOT fire (a Traced Intimidate/Drought does nothing on
//        copy); the copied ability's PASSIVE effects apply from then on.
//   (T3) untraceable: gen3 trace has NO notrace/noability guard (flags: {}) — it
//        copies ANYTHING (even another Trace / No Ability)?
//   (T4) switch-out reverts to the base ability; a re-switch-in traces AGAIN (draw).
//   (T5) draw POSITION: within the switch-in ability-start event (vs Intimidate order).
//   (T6) copied-ability LIVENESS: a traced Flash Fire / Levitate actually gates damage.
// SHED SKIN cruxes:
//   (S1) residual order 10 subOrder 3 (BEFORE Leftovers 4 / DoT 6): a same-mon cure
//        happens BEFORE the status DoT would chip → a cured brn/psn does NOT chip.
//   (S2) the DRAW: randomChance(33,100) ONCE per statused residual; NO draw unstatused.
//   (S3) cure scope: major status only (confusion stays).
//   (S4) a shed-skin holder vs a Leftovers foe at an equal-speed... (subOrder 3 vs 4 —
//        NO tie, no shuffle between them).
// Run: node src/rust_sim/harness/probe_trace_shedskin_rng.js

'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

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

async function boot(p1team, p2team, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of String(ch).split('\n')) lines.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 14; i++) await tick();
  const battle = stream.battle;
  const prng = battle.prng;
  const calls = [];
  for (const name of ['random', 'randomChance', 'sample']) {
    const orig = prng[name].bind(prng);
    prng[name] = (...a) => {
      const r = orig(...a);
      const stack = String(new Error().stack).split('\n').slice(2, 6).join(' ');
      const m = stack.match(/at (?:Battle|BattleActions|Pokemon|Side|Field)\.?(\w+)/);
      calls.push(`${name}(${a.map((x) => (Array.isArray(x) ? `[${x.map((y) => (typeof y === 'object' ? '<o>' : JSON.stringify(y))).join(',')}]` : JSON.stringify(x))).join(',')})@${m ? m[1] : '?'}=>${typeof r === 'object' ? '<o>' : JSON.stringify(r)}`);
      return r;
    };
  }
  return { battle, streams, lines, calls };
}
async function turn(ctx, c1, c2) {
  const before = ctx.calls.length; const lb = ctx.lines.length;
  if (c1) ctx.streams.omniscient.write(`>p1 ${c1}`);
  if (c2) ctx.streams.omniscient.write(`>p2 ${c2}`);
  for (let k = 0; k < 12; k++) await tick();
  return { calls: ctx.calls.slice(before), lines: ctx.lines.slice(lb) };
}

(async () => {
  // (T1)+(T5) Trace lead vs a single passive-ability foe. Compare the STARTUP draw
  // count vs a control (Limber lead — draw-free onStart... none). Instrument from >start:
  // boot() wraps only after startup, so run the lead-trace via a MID-BATTLE switch-in.
  {
    for (const [lbl, ab] of [['TRACE', 'Trace'], ['CONTROL Limber', 'Limber']]) {
      const ctx = await boot(
        [mon('Machamp', ['splash'], { ability: 'Guts' }), mon('Gardevoir', ['splash'], { ability: ab })],
        [mon('Snorlax', ['splash'], { ability: 'Immunity' })],
        [101, 102, 103, 104]);
      const t = await turn(ctx, 'switch 2', 'move 1'); // Gardevoir switches in on Snorlax(Immunity)
      console.log(`\n=== (T1) mid-battle switch-in ${lbl} vs one foe ===`);
      console.log(`  calls: ${t.calls.join('  ')}`);
      console.log(`  lines: ${JSON.stringify(t.lines.filter((l) => /switch|-ability|move/.test(l)))}`);
      const g = ctx.battle.p1.active[0];
      console.log(`  gardevoir.ability=${g.ability} baseAbility=${g.baseAbility}`);
    }
  }

  // (T2) Traced Intimidate must NOT fire (no foe Atk drop) in gen3.
  {
    const ctx = await boot(
      [mon('Machamp', ['splash'], { ability: 'Guts' }), mon('Gardevoir', ['splash'], { ability: 'Trace' })],
      [mon('Gyarados', ['splash'], { ability: 'Intimidate' })],
      [111, 112, 113, 114]);
    const t = await turn(ctx, 'switch 2', 'move 1');
    const g = ctx.battle.p1.active[0]; const foe = ctx.battle.p2.active[0];
    console.log('\n=== (T2) Trace copies Intimidate — copied onStart must NOT fire ===');
    console.log(`  lines: ${JSON.stringify(t.lines.filter((l) => /-ability|unboost|switch/.test(l)))}`);
    console.log(`  gardevoir.ability=${g.ability}  foe atk boost=${foe.boosts.atk} (must be 0)  OUR atk boost=${g.boosts.atk}`);
  }

  // (T2b) Traced Drought must NOT set sun on copy.
  {
    const ctx = await boot(
      [mon('Machamp', ['splash'], { ability: 'Guts' }), mon('Gardevoir', ['splash'], { ability: 'Trace' })],
      [mon('Snorlax', ['splash'], { ability: 'Drought' })],
      [115, 116, 117, 118]);
    // Snorlax's own Drought set sun at startup; let it expire? gen3 Drought is permanent — use
    // a different check: weather is ALREADY sun from the foe's own start. Instead check the
    // -weather line count after the trace switch-in (no NEW -weather from the copy).
    const t = await turn(ctx, 'switch 2', 'move 1');
    console.log('\n=== (T2b) Trace copies Drought — no new -weather from the copy ===');
    console.log(`  lines: ${JSON.stringify(t.lines.filter((l) => /-weather|-ability/.test(l)))}`);
  }

  // (T3) untraceable? Trace vs a No Ability foe + vs a Trace foe.
  {
    for (const [lbl, foeAb] of [['No Ability', 'No Ability'], ['Trace (mirror)', 'Trace'], ['Wonder Guard', 'Wonder Guard']]) {
      const ctx = await boot(
        [mon('Machamp', ['splash'], { ability: 'Guts' }), mon('Gardevoir', ['splash'], { ability: 'Trace' })],
        [mon('Snorlax', ['splash'], { ability: foeAb })],
        [121, 122, 123, 124]);
      const t = await turn(ctx, 'switch 2', 'move 1');
      const g = ctx.battle.p1.active[0];
      console.log(`\n=== (T3) Trace vs foe ${lbl} ===`);
      console.log(`  calls: ${t.calls.join('  ')}`);
      console.log(`  gardevoir.ability=${g.ability}; -ability lines: ${JSON.stringify(t.lines.filter((l) => /-ability/.test(l)))}`);
    }
  }

  // (T4) switch-out reverts; re-entry re-traces (a second draw).
  {
    const ctx = await boot(
      [mon('Machamp', ['splash'], { ability: 'Guts' }), mon('Gardevoir', ['splash'], { ability: 'Trace' })],
      [mon('Snorlax', ['splash'], { ability: 'Immunity' })],
      [131, 132, 133, 134]);
    await turn(ctx, 'switch 2', 'move 1'); // trace Immunity
    const g = ctx.battle.p1.pokemon.find((p) => p.species.id === 'gardevoir');
    const midAbility = g.ability;
    await turn(ctx, 'switch 2', 'move 1'); // Gardevoir out, Machamp in
    const outAbility = g.ability;
    const t3 = await turn(ctx, 'switch 2', 'move 1'); // Gardevoir back in → re-trace
    console.log('\n=== (T4) trace → switch out (revert?) → re-trace ===');
    console.log(`  after trace: ${midAbility}; after switch-out: ${outAbility}; re-entry calls: ${t3.calls.join('  ')}`);
    console.log(`  re-entry -ability: ${JSON.stringify(t3.lines.filter((l) => /-ability/.test(l)))}`);
  }

  // (T6) copied-ability LIVENESS: trace Flash Fire → a Fire move into us activates FF.
  {
    const ctx = await boot(
      [mon('Machamp', ['splash'], { ability: 'Guts' }), mon('Gardevoir', ['splash', 'splash'], { ability: 'Trace' })],
      [mon('Houndoom', ['flamethrower'], { ability: 'Flash Fire' })],
      [141, 142, 143, 144]);
    const t1 = await turn(ctx, 'switch 2', 'move 1'); // trace FF; foe flamethrower into us
    const g = ctx.battle.p1.active[0];
    console.log('\n=== (T6) traced Flash Fire is LIVE (fire immunity volatile) ===');
    console.log(`  gardevoir.ability=${g.ability} hp=${g.hp}/${g.maxhp} volatiles=${Object.keys(g.volatiles).join('|') || '-'}`);
    console.log(`  lines: ${JSON.stringify(t1.lines.filter((l) => /-immune|-start|damage|-ability/.test(l)))}`);
  }

  // ── SHED SKIN ──
  // (S2) draw model: statused → ONE randomChance(33,100)/residual; unstatused → none.
  // (S1) a success cures BEFORE the DoT chips (sub 3 < 6): brn holder keeps full hp
  //      on the cure turn.
  {
    const ctx = await boot(
      [mon('Jolteon', ['willowisp'], { evs: { spe: 252 } })],
      [mon('Arbok', ['splash'], { ability: 'Shed Skin' })],
      [151, 152, 153, 154]);
    const holder = ctx.battle.p2.active[0];
    console.log('\n=== (S1/S2) Shed Skin burned — per-residual randomChance(33,100) until cure; cure turn = NO chip ===');
    for (let i = 0; i < 6; i++) {
      const t = await turn(ctx, 'move 1', 'move 1');
      console.log(`  turn${i + 1}: status=${holder.status || '-'} hp=${holder.hp}/${holder.maxhp} calls=[${t.calls.join('  ')}]`);
      console.log(`          lines=${JSON.stringify(t.lines.filter((l) => /-activate|curestatus|damage.*brn|-status/.test(l)))}`);
      if (!holder.status && i > 0) break;
    }
  }

  // (S3) cure scope: confusion is NOT cured.
  {
    const ctx = await boot(
      [mon('Jolteon', ['confuseray'], { evs: { spe: 252 } })],
      [mon('Arbok', ['splash'], { ability: 'Shed Skin' })],
      [161, 162, 163, 164]);
    const holder = ctx.battle.p2.active[0];
    for (let i = 0; i < 3; i++) {
      const t = await turn(ctx, 'move 1', 'move 1');
      console.log(`  (S3) turn${i + 1}: conf=${!!holder.volatiles['confusion']} calls=[${t.calls.join('  ')}]`);
    }
  }
})().catch((e) => { console.error(e); process.exit(1); });
