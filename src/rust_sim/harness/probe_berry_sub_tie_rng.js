// probe_berry_sub_tie_rng.js — three follow-up berry cruxes probe_berry_rng.js left open:
//   (A) SUBSTITUTE no-trigger, CLEAN: holder ABOVE threshold with a sub up takes
//       sub-absorbed hits → pokemon.hp untouched → berry stays uneaten; then real
//       hp set below threshold WITH the sub still up → eats behind the sub.
//   (B) THE RESIDUAL TIE DRAW-COUNT: an equal-speed mirror where p2 holds sitrus
//       (below 1/2) vs a control where p2 holds Leftovers (damaged) — the berry
//       handler must occupy the SAME order-10 subOrder-4 slot ⇒ IDENTICAL total
//       draw count + the same residual tie-shuffle.
//   (C) CURE-ON-SWITCH-IN timing: a para'd cheri holder switches IN mid-battle —
//       the cure fires at the first eachEvent('Update') after the switch action
//       (line position probed), draw-free.
// Run: node src/rust_sim/harness/probe_berry_sub_tie_rng.js

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
    prng[name] = (...a) => { const r = orig(...a); calls.push(`${name}(${a.map((x) => (Array.isArray(x) ? `[${x.length}]` : JSON.stringify(x))).join(',')})=>${JSON.stringify(r)}`); return r; };
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
  // (A) SUB no-trigger, clean.
  {
    const ctx = await boot(
      [mon('Blissey', ['seismictoss'], { evs: { spe: 252 } })],
      [mon('Snorlax', ['substitute', 'splash'], { item: 'salacberry' })],
      [71, 72, 73, 74]);
    const holder = ctx.battle.p2.active[0];
    await turn(ctx, 'move 1', 'move 1'); // blissey toss (hits mon pre-sub? blissey faster: toss lands on the bare mon), snorlax subs
    holder.hp = Math.floor(holder.maxhp / 4) + 40; // ABOVE the pinch threshold, sub up
    const t2 = await turn(ctx, 'move 1', 'move 2'); // toss absorbed by the sub
    console.log('=== (A) sub absorbs, holder ABOVE threshold → NO eat ===');
    console.log(`  sub=${JSON.stringify(!!holder.volatiles['substitute'])} hp=${holder.hp}/${holder.maxhp} item=${holder.item || 'NONE'} spe_boost=${holder.boosts.spe}`);
    console.log(`  lines: ${JSON.stringify(t2.lines.filter((l) => /enditem|boost|activate/.test(l)))}`);
    if (holder.volatiles['substitute']) {
      holder.hp = Math.floor(holder.maxhp / 4) - 3; // below threshold, sub STILL up
      const t3 = await turn(ctx, 'move 1', 'move 2');
      console.log('=== (A2) real hp BELOW threshold behind a LIVE sub → eats ===');
      console.log(`  sub=${JSON.stringify(!!holder.volatiles['substitute'])} hp=${holder.hp}/${holder.maxhp} item=${holder.item || 'NONE'} spe_boost=${holder.boosts.spe}`);
      console.log(`  lines: ${JSON.stringify(t3.lines.filter((l) => /enditem|boost/.test(l)))}`);
    }
  }

  // (B) residual tie draw-count: berry-vs-Leftovers mirror == Leftovers-vs-Leftovers mirror.
  {
    const counts = {};
    for (const [lbl, item] of [['sitrus', 'sitrusberry'], ['leftovers-control', 'leftovers']]) {
      const ctx = await boot(
        [mon('Snorlax', ['splash'], { item: 'leftovers' })],
        [mon('Snorlax', ['splash'], { item })],
        [9, 9, 9, 9]);
      const a = ctx.battle.p1.active[0]; const b = ctx.battle.p2.active[0];
      a.hp = a.maxhp - 60; b.hp = Math.floor(b.maxhp / 2) - 10;
      const t = await turn(ctx, 'move 1', 'move 1');
      counts[lbl] = t.calls.length;
      console.log(`=== (B) ${lbl}: draws=${t.calls.length} ===`);
      console.log(`  calls: ${t.calls.join('  ')}`);
      console.log(`  heals: ${JSON.stringify(t.lines.filter((l) => /-heal|enditem/.test(l)))}`);
    }
    console.log(`  (B) VERDICT: identical draw count = ${counts['sitrus'] === counts['leftovers-control']}`);
  }

  // (C) cure-on-switch-in timing: para'd cheri holder pivots out and back in.
  {
    const ctx = await boot(
      [mon('Jolteon', ['thunderwave', 'splash'], { evs: { spe: 252 } })],
      [mon('Snorlax', ['splash'], { item: 'cheriberry' }), mon('Slakoth', ['splash'])],
      [81, 82, 83, 84]);
    const snorlax = ctx.battle.p2.pokemon.find((p) => p.species.id === 'snorlax');
    snorlax.setStatus('par'); // pre-status WITHOUT an Update following (constructed)
    snorlax.item = 'cheriberry';
    // wait: setStatus fires AfterSetStatus (lum would eat; cheri does NOT) — verify cheri intact:
    console.log(`=== (C) pre-status'd cheri holder: item=${snorlax.item} status=${snorlax.status} ===`);
    const t1 = await turn(ctx, 'move 2', 'switch 2'); // switch snorlax OUT (slakoth in)
    const t2 = await turn(ctx, 'move 2', 'switch 2'); // switch snorlax back IN
    console.log(`  switch-in turn lines: ${JSON.stringify(t2.lines.filter((l) => /switch|enditem|curestatus|move|upkeep/.test(l)))}`);
    console.log(`  post: item=${snorlax.item || 'NONE'} status=${snorlax.status || '-'}`);
    console.log(`  calls: ${t2.calls.join('  ')}`);
  }
})().catch((e) => { console.error(e); process.exit(1); });
