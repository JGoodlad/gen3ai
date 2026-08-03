// probe_ptrap_edges.js — the ROUND-32 EDGE probe for the gen3 PARTIAL-TRAP family
// (wrap/bind/firespin/clamp/whirlpool/sandtomb) vs the OMNISCIENT gen3 BattleStream.
//
// `probe_batch89_trap.js` settled the HAPPY PATH (one random(3,7) at cast, maxhp/16
// draw-free chip, release forms, the firm switch-block). This probe settles the EDGES
// the implementation has to get right, each of which is a distinct byte/draw question:
//
//   A. SUBSTITUTE on the victim      — is the volatile applied? is the duration DRAWN?
//   B. RE-CAST on an already-trapped mon — addVolatile returns false: fail line? draw?
//   C. The cast KOs the target       — volatile on a corpse? duration drawn?
//   D. TRAPPER FAINTS               — release form + which residual turn
//   E. VICTIM faints to the CHIP     — emission + faint ordering
//   F. RAPID SPIN by the trapped mon — the `-end` form (silent or not) + draws
//   G. TYPE IMMUNITY at cast         — Sand Tomb→Flying, Wrap→Ghost: no trap, no draw
//   H. PROTECT blocks the cast       — no trap, no duration draw
//   I. MUTUAL wrap, speed-tied       — the (10,9) residual tie-shuffle draw
//   J. BATON PASS by a trapped mon   — is it legal, and does the volatile COPY?
//   K. PHAZE (Roar) the victim out   — volatile gone; is the trapper's link dead?
//   L. NON-WRAP byte forms           — Fire Spin / Sand Tomb `-activate`/`-damage`/`-end`
//   M. The per-side |request| shape  — trapped:true on the FIRST request?
//
// Run: node harness/probe_ptrap_edges.js
'use strict';
const path = require('path');
const { mon, run, fmtCalls } = require('./probe_batch4_lib');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const KEEP = (l) => l && !l.startsWith('|t:|') && l !== '|' && !l.startsWith('|upkeep');

function show(tag, r, n = 99) {
  r.perDecision.slice(0, n).forEach((d, i) => {
    console.log(`  ${tag} t${i + 1}: draws=${d.nexts} calls=[${fmtCalls(d.calls)}]`);
    console.log(`        lines=${JSON.stringify(d.lines.filter(KEEP))}`);
    if (r.states[i] !== undefined) console.log(`        state=${JSON.stringify(r.states[i])}`);
  });
}
const durDraw = (d) => d.calls.filter((c) => c.kind === 'random' && c.args[0] === 3 && c.args[1] === 7).length;
const st = (b) => {
  const o = {};
  for (const s of [0, 1]) {
    const m = b.sides[s].active[0];
    if (!m) { o[`p${s + 1}`] = '-'; continue; }
    const v = m.volatiles.partiallytrapped;
    o[`p${s + 1}`] = `${m.species.id} ${m.hp}/${m.maxhp}${v ? ` PT(dur=${v.duration},src=${v.source && v.source.species.id})` : ''}${m.trapped ? ' TRAPPED' : ''}${m.volatiles.substitute ? ' SUB' : ''}`;
  }
  return o;
};

// Seeds to sweep until the cast LANDS (these moves are 70-85 acc).
const SEEDS = [];
for (let s = 0; s < 40; s++) SEEDS.push([s * 7 + 1, s * 3 + 2, s * 5 + 4, s * 11 + 3]);

async function findLanding(mkTeams, choices, pred, opts = {}) {
  for (const seed of SEEDS) {
    const r = await run(mkTeams(), seed, choices, { onBoundary: st, ...opts });
    if (pred(r)) return { seed, r };
  }
  return null;
}
const landed = (name) => (r) => r.perDecision[0].lines.some((l) => l.includes('-activate') && l.includes(`move: ${name}`));

async function main() {
  // ── A. SUBSTITUTE on the victim ────────────────────────────────────────────
  console.log('############ A. SUBSTITUTE on the victim ############');
  {
    const mk = () => [
      [mon('Dragonite', ['wrap', 'splash'], { ability: 'Inner Focus' })],
      [mon('Snorlax', ['substitute', 'splash'], { ability: 'Own Tempo' }), mon('Blissey', ['splash'])],
    ];
    for (const seed of SEEDS.slice(0, 8)) {
      const r = await run(mk(), seed, [['move 2', 'move 1'], ['move 1', 'move 2'], ['move 1', 'move 2']], { onBoundary: st });
      const cast = r.perDecision[1];
      if (!cast.lines.some((l) => l.includes('|move|p1a: Dragonite|Wrap'))) continue;
      console.log(`=== seed ${JSON.stringify(seed)} (t1 Sub, t2 Wrap into the sub) ===`);
      show('SUB', r);
      console.log(`  >>> duration draws on the wrap turn = ${durDraw(cast)}`);
      break;
    }
  }

  // ── B. RE-CAST on an already-trapped mon ───────────────────────────────────
  console.log('\n############ B. RE-CAST wrap on an already-trapped mon ############');
  {
    const mk = () => [
      [mon('Dragonite', ['wrap', 'splash'], { ability: 'Inner Focus' })],
      [mon('Snorlax', ['splash'], { ability: 'Own Tempo' })],
    ];
    const f = await findLanding(mk, [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']], landed('Wrap'));
    if (f) { console.log(`=== seed ${JSON.stringify(f.seed)} ===`); show('RECAST', f.r); console.log(`  >>> duration draws t2 = ${durDraw(f.r.perDecision[1])}`); }
  }

  // ── C. the cast KOs the target ─────────────────────────────────────────────
  console.log('\n############ C. the cast KOs the target (volatile on a corpse?) ############');
  {
    const mk = () => [
      [mon('Dragonite', ['wrap', 'splash'], { ability: 'Inner Focus', evs: { atk: 252 } })],
      [mon('Shedinja', ['splash'], { ability: 'Wonder Guard' }), mon('Blissey', ['splash'])],
    ];
    // Shedinja has 1 HP but Wonder Guard blocks neutral hits; use a 1-HP-ish target instead:
    const mk2 = () => [
      [mon('Dragonite', ['wrap', 'splash'], { ability: 'Inner Focus', evs: { atk: 252 } })],
      [mon('Diglett', ['splash'], { ability: 'Sand Veil', level: 1 }), mon('Blissey', ['splash'])],
    ];
    for (const mkX of [mk2, mk]) {
      const f = await findLanding(mkX, [['move 1', 'move 1'], [null, 'switch 2']], (r) =>
        r.perDecision[0].lines.some((l) => l.startsWith('|faint|p2a')));
      if (f) {
        console.log(`=== seed ${JSON.stringify(f.seed)} ===`); show('KO', f.r);
        console.log(`  >>> duration draws on the KO turn = ${durDraw(f.r.perDecision[0])}`);
        break;
      }
    }
  }

  // ── D. TRAPPER FAINTS ──────────────────────────────────────────────────────
  console.log('\n############ D. the TRAPPER faints (release form + turn) ############');
  {
    const mk = () => [
      [mon('Diglett', ['wrap', 'splash'], { ability: 'Sand Veil', level: 5 }), mon('Gengar', ['splash'], { ability: 'Levitate' })],
      [mon('Snorlax', ['return', 'splash'], { ability: 'Own Tempo', evs: { atk: 252 } })],
    ];
    const f = await findLanding(mk, [['move 1', 'move 2'], ['move 2', 'move 1'], [null, null], ['move 1', 'move 1']], landed('Wrap'));
    if (f) { console.log(`=== seed ${JSON.stringify(f.seed)} ===`); show('TRAPPERFNT', f.r); }
  }

  // ── E. VICTIM faints to the CHIP ───────────────────────────────────────────
  console.log('\n############ E. the VICTIM faints to the residual chip ############');
  {
    // A low-level victim whose maxhp/16 chip finishes it after some Wrap damage.
    const mk = () => [
      [mon('Dragonite', ['wrap', 'splash'], { ability: 'Inner Focus', evs: { atk: 252 } })],
      [mon('Blissey', ['bellydrum', 'splash'], { ability: 'Natural Cure' }), mon('Snorlax', ['splash'])],
    ];
    const f = await findLanding(mk, [
      ['move 1', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'],
      ['move 2', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'],
    ], (r) => r.perDecision.some((d) => d.lines.some((l) => l.startsWith('|faint|p2a'))));
    if (f) { console.log(`=== seed ${JSON.stringify(f.seed)} ===`); show('CHIPKO', f.r); }
    else console.log('  (no chip-KO board found in the seed sweep)');
  }

  // ── F. RAPID SPIN by the trapped mon ───────────────────────────────────────
  console.log('\n############ F. RAPID SPIN by the trapped mon ############');
  {
    const mk = () => [
      [mon('Dragonite', ['wrap', 'splash'], { ability: 'Inner Focus' })],
      [mon('Starmie', ['rapidspin', 'splash'], { ability: 'Natural Cure' })],
    ];
    const f = await findLanding(mk, [['move 1', 'move 2'], ['move 2', 'move 1'], ['move 2', 'move 2']], landed('Wrap'));
    if (f) { console.log(`=== seed ${JSON.stringify(f.seed)} ===`); show('SPIN', f.r); }
  }

  // ── G. TYPE IMMUNITY at cast ───────────────────────────────────────────────
  console.log('\n############ G. type immunity at cast (no trap, no duration draw) ############');
  {
    const mk = () => [
      [mon('Dragonite', ['wrap', 'splash'], { ability: 'Inner Focus' })],
      [mon('Gengar', ['splash'], { ability: 'Levitate' })],
    ];
    const r = await run(mk(), [5, 4, 3, 2], [['move 1', 'move 1']], { onBoundary: st });
    show('GHOST', r);
    const mk2 = () => [
      [mon('Dugtrio', ['sandtomb', 'splash'], { ability: 'Arena Trap' })],
      [mon('Skarmory', ['splash'], { ability: 'Keen Eye' })],
    ];
    const r2 = await run(mk2(), [5, 4, 3, 2], [['move 1', 'move 1']], { onBoundary: st });
    show('FLYING', r2);
  }

  // ── H. PROTECT blocks the cast ─────────────────────────────────────────────
  console.log('\n############ H. PROTECT blocks the cast ############');
  {
    const mk = () => [
      [mon('Dragonite', ['wrap', 'splash'], { ability: 'Inner Focus' })],
      [mon('Snorlax', ['protect', 'splash'], { ability: 'Own Tempo' })],
    ];
    const r = await run(mk(), [5, 4, 3, 2], [['move 1', 'move 1'], ['move 1', 'move 2']], { onBoundary: st });
    show('PROTECT', r);
  }

  // ── I. MUTUAL wrap (both sides trapped, speed-TIED) ────────────────────────
  console.log('\n############ I. MUTUAL wrap, speed-tied (the (10,9) tie-shuffle) ############');
  {
    const mk = () => [
      [mon('Dragonite', ['wrap', 'splash'], { ability: 'Inner Focus' })],
      [mon('Dragonite', ['wrap', 'splash'], { ability: 'Inner Focus' })],
    ];
    const f = await findLanding(mk, [['move 1', 'move 1'], ['move 2', 'move 2'], ['move 2', 'move 2']],
      (r) => r.perDecision[0].lines.filter((l) => l.includes('move: Wrap')).length >= 2);
    if (f) { console.log(`=== seed ${JSON.stringify(f.seed)} ===`); show('MUTUAL', f.r); }
    else console.log('  (no double-landing seed found)');
  }

  // ── J. BATON PASS by a trapped mon ─────────────────────────────────────────
  console.log('\n############ J. BATON PASS by a trapped mon (legal? volatile copied?) ############');
  {
    const mk = () => [
      [mon('Dragonite', ['wrap', 'splash'], { ability: 'Inner Focus' })],
      [mon('Snorlax', ['batonpass', 'splash'], { ability: 'Own Tempo' }), mon('Blissey', ['splash'], { ability: 'Natural Cure' })],
    ];
    const f = await findLanding(mk, [['move 1', 'move 2'], ['move 2', 'move 1'], [null, 'switch 2'], ['move 2', 'move 1'], ['move 2', 'move 1']], landed('Wrap'));
    if (f) { console.log(`=== seed ${JSON.stringify(f.seed)} ===`); show('BP', f.r); }
  }

  // ── K. PHAZE the victim out ────────────────────────────────────────────────
  console.log('\n############ K. ROAR the trapped victim out ############');
  {
    const mk = () => [
      [mon('Dragonite', ['wrap', 'roar'], { ability: 'Inner Focus' })],
      [mon('Snorlax', ['splash'], { ability: 'Own Tempo' }), mon('Blissey', ['splash'], { ability: 'Natural Cure' })],
    ];
    const f = await findLanding(mk, [['move 1', 'move 1'], ['move 2', 'move 1'], ['move 1', 'move 1']], landed('Wrap'));
    if (f) { console.log(`=== seed ${JSON.stringify(f.seed)} ===`); show('ROAR', f.r); }
  }

  // ── L. NON-WRAP byte forms ─────────────────────────────────────────────────
  console.log('\n############ L. Fire Spin / Sand Tomb / Whirlpool / Clamp / Bind byte forms ############');
  for (const [sp, mv, name] of [
    ['Houndoom', 'firespin', 'Fire Spin'], ['Dugtrio', 'sandtomb', 'Sand Tomb'],
    ['Kingdra', 'whirlpool', 'Whirlpool'], ['Cloyster', 'clamp', 'Clamp'], ['Machamp', 'bind', 'Bind'],
  ]) {
    const mk = () => [
      [mon(sp, [mv, 'splash'], { ability: 'No Ability' })],
      [mon('Blissey', ['splash'], { ability: 'Natural Cure' })],
    ];
    const f = await findLanding(mk, [['move 1', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'],
      ['move 2', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1']], landed(name));
    if (!f) { console.log(`  ${name}: no landing seed`); continue; }
    const all = f.r.perDecision.flatMap((d) => d.lines.filter(KEEP))
      .filter((l) => l.includes('partiallytrapped') || l.includes(`move: ${name}`) || (l.includes('-end') && l.includes(name)));
    console.log(`  ${name} (seed ${JSON.stringify(f.seed)}): ${JSON.stringify(all)}`);
  }

  // ── M. the per-side |request| shape ────────────────────────────────────────
  console.log('\n############ M. per-side |request| trapped flag ############');
  {
    const teams = [
      [mon('Dragonite', ['wrap', 'splash'], { ability: 'Inner Focus' })],
      [mon('Snorlax', ['splash'], { ability: 'Own Tempo' }), mon('Blissey', ['splash'], { ability: 'Natural Cure' })],
    ];
    for (const seed of SEEDS.slice(0, 12)) {
      const stream = new BattleStream();
      const streams = getPlayerStreams(stream);
      const p2lines = [];
      const omni = [];
      (async () => { for await (const ch of streams.p2) for (const l of String(ch).split('\n')) p2lines.push(l); })();
      (async () => { for await (const ch of streams.omniscient) for (const l of String(ch).split('\n')) omni.push(l); })();
      streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
      streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(teams[0]) })}`);
      streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(teams[1]) })}`);
      for (let i = 0; i < 14; i++) await new Promise((r) => setTimeout(r, 0));
      streams.omniscient.write('>p1 move 1'); streams.omniscient.write('>p2 move 1');
      for (let i = 0; i < 14; i++) await new Promise((r) => setTimeout(r, 0));
      if (!omni.some((l) => l.includes('move: Wrap'))) continue;
      const reqs = p2lines.filter((l) => l.startsWith('|request|'));
      console.log(`=== seed ${JSON.stringify(seed)} ===`);
      reqs.forEach((l, i) => console.log(`  p2 request[${i}]: ${l.slice(0, 400)}`));
      // now try to switch: is it rejected, and is a NEW request sent?
      const before = p2lines.length;
      streams.omniscient.write('>p2 switch 2');
      for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0));
      console.log(`  after '>p2 switch 2': new p2 lines = ${JSON.stringify(p2lines.slice(before))}`);
      break;
    }
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
