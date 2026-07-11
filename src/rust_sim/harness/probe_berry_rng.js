// probe_berry_rng.js — settle the EXACT gen3 BERRY consumption model against the
// resolved `Dex.mod('gen3')` sim's PRNG + state (Mandate 1 — the sim is the only oracle).
//
// The cruxes this probe nails (per the batch-3 build mandate):
//   (1) eatItem DRAW MODEL: is the eat itself draw-free (runEvent UseItem/TryEatItem/
//       EatItem have no other handlers)? Which onEat effects DRAW (starf sample; the
//       figy-family confusion volatile's random(2,6))?
//   (2) HEAL/PINCH TRIGGER SITE + THRESHOLD: residual order 10 subOrder 4 ONLY (the
//       gen3 mod deletes onUpdate) — hp <= maxhp/2 (heal) / hp <= maxhp/4 (pinch),
//       exact float boundary == 2*hp <= maxhp / 4*hp <= maxhp.
//   (3) THE LEFTOVERS TIE: a berry residual handler is order 10 subOrder 4 — the SAME
//       key as Leftovers — so a 2-mon equal-speed berry-vs-Leftovers board TIE-SHUFFLES
//       (one random(0,2)); distinct speeds draw nothing.
//   (4) CURE TIMING: lum eats IMMEDIATELY inside setStatus (onAfterSetStatus priority
//       -1); the single-status berries (cheri/chesto/...) eat at the NEXT
//       eachEvent('Update') site — BEFORE the holder's own move, so the holder never
//       rolls full-para that turn (a draw-count difference).
//   (5) SUBSTITUTE: a sub-absorbed hit leaves pokemon.hp untouched → NO trigger.
//   (6) KO: a fainted holder never eats (residual excludes fainted; eatItem gates !hp).
//   (7) ITEM-BECOMES-NONE: after the eat the item is gone for the battle (no second
//       eat on a later crossing / a later status).
//   (8) LEPPA: onUpdate when a moveSlot hits pp 0 → +10 PP on that slot.
//   (9) SYNCHRONIZE + LUM on the same mon: AfterSetStatus handler order (sync prio 0
//       vs lum prio -1).
//
// Run: node src/rust_sim/harness/probe_berry_rng.js

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

function siteLabel(stack) {
  const lines = String(stack).split('\n').slice(2);
  for (const l of lines) {
    const m = l.match(/at (?:Battle|BattleActions|Pokemon|Side|Field)\.?(\w+)/);
    if (m && !['random', 'randomChance', 'sample', 'shuffle'].includes(m[1])) return m[1];
  }
  return '?';
}

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
  const wrap = (name) => {
    const orig = prng[name].bind(prng);
    prng[name] = (...a) => {
      const r = orig(...a);
      calls.push({ kind: name, args: a, site: siteLabel(new Error().stack), ret: r });
      return r;
    };
  };
  wrap('random'); wrap('randomChance'); wrap('sample'); wrap('shuffle');
  return { battle, streams, lines, calls };
}

function fmtCalls(cs) {
  return cs.map((c) => `${c.kind}(${JSON.stringify(c.args)})@${c.site}=>${JSON.stringify(c.ret)}`).join('  ');
}
async function turn(ctx, c1, c2) {
  const before = ctx.calls.length;
  const lbefore = ctx.lines.length;
  if (c1) ctx.streams.omniscient.write(`>p1 ${c1}`);
  if (c2) ctx.streams.omniscient.write(`>p2 ${c2}`);
  for (let k = 0; k < 12; k++) await tick();
  return { calls: ctx.calls.slice(before), lines: ctx.lines.slice(lbefore) };
}
function monLine(p) {
  return `${p.species.id} hp=${p.hp}/${p.maxhp} status=${p.status || '-'} item=${p.item || 'NONE'} boosts=${JSON.stringify(p.boosts)} pp=${p.moveSlots.map((m) => m.pp).join(',')} vol=${Object.keys(p.volatiles).join('|') || '-'}`;
}

(async () => {
  // ───────────────────────────────────────────────────────────────────────────
  // (2) HEAL/PINCH threshold boundary + trigger site (residual-only) + (1) draws.
  // Constructed HP (direct mutation — the omniscient probe's exact-boundary tool).
  // ───────────────────────────────────────────────────────────────────────────
  for (const [item, num, den, label] of [
    ['sitrusberry', 1, 2, 'HEAL sitrus (thr 1/2, heal 30)'],
    ['oranberry', 1, 2, 'HEAL oran (thr 1/2, heal 10)'],
    ['figyberry', 1, 2, 'HEAL figy (thr 1/2, heal maxhp/3 + minus-atk confusion)'],
    ['salacberry', 1, 4, 'PINCH salac (thr 1/4, +1 spe)'],
    ['liechiberry', 1, 4, 'PINCH liechi (thr 1/4, +1 atk)'],
    ['starfberry', 1, 4, 'PINCH starf (thr 1/4, sample +2)'],
    ['lansatberry', 1, 4, 'PINCH lansat (thr 1/4, focusenergy)'],
    ['leppaberry', 1, 4, 'PP leppa via constructed hp (should NOT eat on hp)'],
  ]) {
    // Adamant Snorlax to give figy (minus=spa? adamant is +atk -spa) — use NAUGHTY
    // (+atk -spd) for aguav check instead; figy confuses on minus ATK → use MODEST.
    const nature = item === 'figyberry' ? 'Modest' : 'Serious';
    const ctx = await boot(
      [mon('Blissey', ['splash', 'seismictoss'])],
      [mon('Snorlax', ['splash', 'curse'], { item, nature })],
      [11, 22, 33, 44]);
    const holder = ctx.battle.p2.active[0];
    const maxhp = holder.maxhp;
    const above = Math.floor(maxhp * num / den) + 1; // strictly above → no eat
    const at = Math.floor(maxhp * num / den);        // at/below → eat (if 2*hp<=maxhp exact)
    holder.hp = above;
    const t1 = await turn(ctx, 'move 1', 'move 1');
    const afterAbove = monLine(holder);
    holder.hp = at;
    const t2 = await turn(ctx, 'move 1', 'move 1');
    console.log(`\n=== ${label} maxhp=${maxhp} ===`);
    console.log(`  hp=${above} (above thr): calls=[${fmtCalls(t1.calls)}]`);
    console.log(`    state: ${afterAbove}`);
    console.log(`  hp=${at} (at thr):    calls=[${fmtCalls(t2.calls)}]`);
    console.log(`    state: ${monLine(holder)}`);
    console.log(`    lines: ${JSON.stringify(t2.lines.filter((l) => /enditem|heal|boost|start|activate|-item/.test(l)))}`);
    // (7) second crossing after the eat → nothing.
    holder.hp = Math.max(1, at - 5);
    const t3 = await turn(ctx, 'move 1', 'move 1');
    console.log(`  re-cross after eat:  calls=[${fmtCalls(t3.calls)}]  state: ${monLine(holder)}`);
  }

  // The eat happens at the RESIDUAL, not right after damage: drop below threshold
  // MID-TURN with the holder acting AFTER the damage — the holder's own move happens
  // pre-eat (hp still low), the eat only at end of turn. Witness via line order.
  {
    const ctx = await boot(
      [mon('Blissey', ['seismictoss'], { evs: { spe: 252 } })],
      [mon('Snorlax', ['splash'], { item: 'sitrusberry' })],
      [5, 6, 7, 8]);
    const holder = ctx.battle.p2.active[0];
    holder.hp = Math.floor(holder.maxhp / 2) + 50; // one toss (100) crosses the threshold
    const t = await turn(ctx, 'move 1', 'move 1');
    console.log('\n=== HEAL trigger SITE (mid-turn cross → eat only at residual) ===');
    console.log(`  lines: ${JSON.stringify(t.lines.filter((l) => /move|damage|enditem|heal|upkeep|turn/.test(l)))}`);
  }

  // ───────────────────────────────────────────────────────────────────────────
  // (3) The LEFTOVERS TIE: equal-speed Snorlax mirror, p1 Leftovers (full hp? no —
  // Leftovers fires only when damaged; give it damage too), p2 sitrus below 1/2.
  // Both handlers order 10 subOrder 4 → tie-shuffle ONE random. Control: p2 slower.
  // ───────────────────────────────────────────────────────────────────────────
  {
    for (const [lbl, p2spe] of [['TIE (equal speed)', {}], ['CONTROL (p2 slower ivs spe 0)', { ivs: { ...IV31, spe: 0 } }]]) {
      const ctx = await boot(
        [mon('Snorlax', ['splash'], { item: 'leftovers' })],
        [mon('Snorlax', ['splash'], { item: 'sitrusberry', ...p2spe })],
        [9, 9, 9, 9]);
      const a = ctx.battle.p1.active[0]; const b = ctx.battle.p2.active[0];
      a.hp = a.maxhp - 60; b.hp = Math.floor(b.maxhp / 2) - 10;
      const t = await turn(ctx, 'move 1', 'move 1');
      console.log(`\n=== LEFTOVERS-vs-BERRY residual ${lbl} ===`);
      console.log(`  calls=[${fmtCalls(t.calls)}]`);
      console.log(`  heal-line order: ${JSON.stringify(t.lines.filter((l) => /-heal|enditem/.test(l)))}`);
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // (4) CURE berries: lum immediate (inside setStatus) vs cheri at the next Update.
  //     Faster TWave para's the SLOWER holder → does the holder move un-para'd with
  //     ZERO full-para roll? chesto+rest; persim+confusion; pecha vs tox.
  // ───────────────────────────────────────────────────────────────────────────
  for (const [item, holderMoves, foeMoves, label] of [
    ['lumberry', ['splash'], ['thunderwave'], 'LUM immediate (onAfterSetStatus) on TWave'],
    ['cheriberry', ['splash'], ['thunderwave'], 'CHERI cure at next Update after TWave'],
    ['chestoberry', ['rest', 'splash'], ['seismictoss'], 'CHESTO + Rest (cure at the post-Rest Update)'],
    ['persimberry', ['splash'], ['confuseray'], 'PERSIM cures confusion at Update'],
    ['pechaberry', ['splash'], ['toxic'], 'PECHA cures TOX too'],
    ['aspearberry', ['splash'], ['icebeam'], 'ASPEAR frz (may need luck — informational)'],
  ]) {
    const ctx = await boot(
      [mon('Jolteon', foeMoves, { evs: { spe: 252 } })],
      [mon('Snorlax', holderMoves, { item })],
      [3, 14, 15, 92]);
    const holder = ctx.battle.p2.active[0];
    if (item === 'chestoberry') holder.hp = 150; // make Rest legal + meaningful
    const t1 = await turn(ctx, 'move 1', 'move 1');
    console.log(`\n=== ${label} ===`);
    console.log(`  turn1 calls=[${fmtCalls(t1.calls)}]`);
    console.log(`  turn1 lines: ${JSON.stringify(t1.lines.filter((l) => /move|-status|curestatus|enditem|cant|-start|-end |heal/.test(l)))}`);
    console.log(`  state: ${monLine(holder)}`);
    // (7) a SECOND status after the eat → sticks (no berry).
    const t2 = await turn(ctx, 'move 1', item === 'chestoberry' ? 'move 2' : 'move 1');
    console.log(`  turn2 calls=[${fmtCalls(t2.calls)}]`);
    console.log(`  turn2 state: ${monLine(holder)}`);
  }

  // (9) SYNCHRONIZE + LUM on the same mon (AfterSetStatus order: sync prio 0 > lum -1?).
  {
    const ctx = await boot(
      [mon('Jolteon', ['thunderwave'], { evs: { spe: 252 } })],
      [mon('Espeon', ['splash'], { item: 'lumberry', ability: 'Synchronize' })],
      [21, 22, 23, 24]);
    const t = await turn(ctx, 'move 1', 'move 1');
    console.log('\n=== SYNCHRONIZE + LUM same mon (TWave in) ===');
    console.log(`  calls=[${fmtCalls(t.calls)}]`);
    console.log(`  lines: ${JSON.stringify(t.lines.filter((l) => /-status|curestatus|enditem|activate/.test(l)))}`);
    console.log(`  p1: ${monLine(ctx.battle.p1.active[0])}`);
    console.log(`  p2: ${monLine(ctx.battle.p2.active[0])}`);
  }

  // ───────────────────────────────────────────────────────────────────────────
  // (5) SUBSTITUTE: a sub-absorbed hit leaves hp untouched → no pinch/heal trigger.
  // ───────────────────────────────────────────────────────────────────────────
  {
    const ctx = await boot(
      [mon('Blissey', ['seismictoss'])],
      [mon('Snorlax', ['substitute', 'splash'], { item: 'salacberry' })],
      [31, 32, 33, 34]);
    const holder = ctx.battle.p2.active[0];
    holder.hp = Math.floor(holder.maxhp / 4) + Math.floor(holder.maxhp / 4) + 30; // sub cost keeps him ABOVE 1/4
    const t1 = await turn(ctx, 'move 1', 'move 1'); // sub up (cost maxhp/4), toss absorbed? (toss goes first? Blissey faster)
    console.log('\n=== SUBSTITUTE × pinch berry ===');
    console.log(`  turn1 (sub up): calls=[${fmtCalls(t1.calls)}]`);
    console.log(`  state: ${monLine(holder)}`);
    const t2 = await turn(ctx, 'move 1', 'move 2'); // toss into the sub — real hp unchanged
    console.log(`  turn2 (toss into sub): lines=${JSON.stringify(t2.lines.filter((l) => /activate|enditem|damage|boost|end/.test(l)))}`);
    console.log(`  state: ${monLine(holder)}  ← hp must be unchanged, berry uneaten`);
    // now set real hp below the pinch threshold WHILE the sub is up: residual reads
    // pokemon.hp → eats even behind a sub.
    if (holder.volatiles['substitute']) {
      holder.hp = Math.floor(holder.maxhp / 4) - 5;
      const t3 = await turn(ctx, 'move 1', 'move 2');
      console.log(`  turn3 (real hp < 1/4 behind the sub): lines=${JSON.stringify(t3.lines.filter((l) => /enditem|boost/.test(l)))}`);
      console.log(`  state: ${monLine(holder)}  ← eats behind the sub`);
    }
  }

  // (6) KO: the holder is KO'd this turn → no eat.
  {
    const ctx = await boot(
      [mon('Blissey', ['seismictoss'], { evs: { spe: 252 } })],
      [mon('Snorlax', ['splash'], { item: 'sitrusberry' }), mon('Slakoth', ['splash'])],
      [41, 42, 43, 44]);
    const holder = ctx.battle.p2.active[0];
    holder.hp = 90; // one toss KOs
    const t = await turn(ctx, 'move 1', 'move 1');
    console.log('\n=== KO: no eat on a fainted holder ===');
    console.log(`  lines: ${JSON.stringify(t.lines.filter((l) => /faint|enditem|heal/.test(l)))}`);
    console.log(`  holder: ${monLine(holder)}`);
  }

  // (8) LEPPA: run a 5-pp move (belly drum? use a 5pp move — 'megakick' 5pp? use
  // 'petaldance'? keep it simple: SPLASH has 40. Use 'swordsdance' 20... take BLIZZARD
  // 5 pp) to 0 → the next Update eats + restores 10.
  {
    const ctx = await boot(
      [mon('Blissey', ['splash'])],
      [mon('Snorlax', ['blizzard', 'splash'], { item: 'leppaberry' })],
      [51, 52, 53, 54]);
    const holder = ctx.battle.p2.active[0];
    holder.moveSlots[0].pp = 1;
    const t = await turn(ctx, 'move 1', 'move 1');
    console.log('\n=== LEPPA: pp 1 → 0 → eat at Update, +10 pp ===');
    console.log(`  calls=[${fmtCalls(t.calls)}]`);
    console.log(`  lines: ${JSON.stringify(t.lines.filter((l) => /enditem|activate|move/.test(l)))}`);
    console.log(`  state: ${monLine(holder)}`);
  }

  // (1)+(2) STARF sample semantics under a BOOST CAP: +6 atk already → the sample pool
  // shrinks (atk excluded). Also the figy confusion draw is covered above.
  {
    const ctx = await boot(
      [mon('Blissey', ['splash'])],
      [mon('Snorlax', ['splash'], { item: 'starfberry' })],
      [61, 62, 63, 64]);
    const holder = ctx.battle.p2.active[0];
    holder.boosts.atk = 6; holder.boosts.def = 6; holder.boosts.spa = 6; holder.boosts.spd = 6;
    holder.hp = Math.floor(holder.maxhp / 4) - 1;
    const t = await turn(ctx, 'move 1', 'move 1');
    console.log('\n=== STARF with atk/def/spa/spd capped (+6): sample pool = [spe] only ===');
    console.log(`  calls=[${fmtCalls(t.calls)}]`);
    console.log(`  state: ${monLine(holder)}`);
  }
})().catch((e) => { console.error(e); process.exit(1); });
