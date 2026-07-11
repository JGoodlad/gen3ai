// probe_leechseed_rng.js — instrument the gen3 Leech Seed draw model bit-for-bit.
//
// Verifies, against the OMNISCIENT in-process BattleStream (no server):
//   1. THE MOVE: gen3 Leech Seed is accuracy 90 → it DRAWS randomChance(90,100) (can
//      MISS). On a LANDED hit it adds the `leechseed` volatile to the foe (draw-free).
//   2. THE GRASS IMMUNITY: a Grass target — does the accuracy roll STILL draw? (gen3
//      tryMoveHit sets naturalImmunity at onTryImmunity but reports `-immune` only AFTER
//      the accuracy roll — so accuracy IS drawn, then `-immune`, no volatile.)
//   3. ALREADY-SEEDED: a 2nd Leech Seed on an already-seeded mon FAILS (addVolatile
//      returns false). Does it draw accuracy? (Yes — accuracy then `-fail`.)
//   4. THE LEECH RESIDUAL: each end-of-turn the seeded mon loses floor(maxhp/8) and the
//      SEEDER's ACTIVE heals that (clamped). DRAW-FREE. residualOrder/subOrder?
//   5. ORDER INTERACTION: leech + Leftovers + weather chip + status DoT on the SAME mon
//      — the residual sequence (weather8 → Leftovers sub4 → LEECH sub5 → DoT sub6).
//   6. A seeder that FAINTED (no heal); a leech drain that KOs the seeded mon.
//
// We wrap battle.prng.next to count raw draws per decision window, and ALSO dump the
// residual handler-sort order (the gathered onResidual handlers + their order/subOrder)
// by hooking fieldEvent so we can read the exact gen3 residualOrder/subOrder for leech.
//
// Run:  node src/rust_sim/harness/probe_leechseed_rng.js
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
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

// First, print the resolved gen3 leechseed condition's residualOrder/subOrder from the dex.
function dumpResolvedOrder() {
  const dex3 = Dex.forFormat(FORMAT);
  const ls = dex3.moves.get('leechseed');
  console.log('=== resolved gen3 leechseed move ===');
  console.log(`  accuracy=${ls.accuracy} type=${ls.type} volatileStatus=${ls.volatileStatus} ` +
    `category=${ls.category}`);
  const cond = ls.condition || {};
  console.log(`  condition.onResidualOrder=${cond.onResidualOrder} ` +
    `condition.onResidualSubOrder=${cond.onResidualSubOrder}`);
  // Leftovers + the status conditions for comparison.
  const lefto = dex3.items.get('leftovers');
  console.log(`  leftovers.onResidualOrder=${lefto.onResidualOrder} subOrder=${lefto.onResidualSubOrder}`);
  for (const id of ['brn', 'psn', 'tox']) {
    const c = dex3.conditions.get(id);
    console.log(`  ${id}.onResidualOrder=${c.onResidualOrder} subOrder=${c.onResidualSubOrder}`);
  }
}

async function run(label, p1team, p2team, plan, inject) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  const seed = [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  for (const inj of (inject || [])) {
    if (inj.weather) {
      battle.field.setWeather(inj.weather, battle.sides[0].active[0]);
      battle.field.weatherState.duration = 0;
    }
    const m = inj.side === undefined ? null : battle.sides[inj.side].active[0];
    if (!m) continue;
    if (inj.status) m.setStatus(inj.status, m, null, true);
    if (inj.hp !== undefined) m.hp = inj.hp;
  }

  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  // Hook fieldEvent('Residual') to dump the residual handler order ONCE per residual.
  const realFieldEvent = battle.fieldEvent.bind(battle);
  battle.fieldEvent = function (eventid, ...rest) {
    if (eventid === 'Residual') {
      const getKey = 'duration';
      let handlers = battle.findFieldEventHandlers(battle.field, `onField${eventid}`, getKey);
      for (const side of battle.sides) {
        handlers = handlers.concat(battle.findSideEventHandlers(side, `onSide${eventid}`, getKey));
        for (const active of side.active) {
          if (!active) continue;
          handlers = handlers.concat(battle.findPokemonEventHandlers(active, `on${eventid}`, getKey));
          handlers = handlers.concat(battle.findSideEventHandlers(side, `on${eventid}`, undefined, active));
          handlers = handlers.concat(battle.findFieldEventHandlers(battle.field, `on${eventid}`, undefined, active));
        }
      }
      battle.speedSort(handlers);
      const desc = handlers.map((h) => {
        const eff = h.effect || {};
        const holder = h.effectHolder && h.effectHolder.species ? h.effectHolder.species.name : '?';
        return `${eff.id || eff.name}[ord=${h.order},sub=${h.subOrder},spd=${h.speed},holder=${holder}]`;
      });
      console.log(`        RESIDUAL handlers: ${desc.join(' ')}`);
    }
    return realFieldEvent(eventid, ...rest);
  };

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 50) {
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
    const seededOf = (m) => (m && m.volatiles && m.volatiles['leechseed'])
      ? `SEEDED(src=${m.volatiles['leechseed'].sourceSlot})` : '';
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'}${m.fainted ? ' FNT' : ''} ${seededOf(m)}` : '-';
    console.log(`  [${rs}] ${JSON.stringify(entry)} draws=${drawCount - dc0}  seedBefore=${before} seedAfter=${after}`);
    console.log(`        p1=${fmt(a0)} | p2=${fmt(a1)}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  dumpResolvedOrder();

  // (A) Basic seed → drain + seeder heal each turn. p1 Meganium seeds p2 Snorlax.
  //     p1 Meganium also at reduced HP so we can see the heal. Both use Leech Seed/filler.
  await run('seed lands → drain + seeder heal each turn',
    [mon('Meganium', ['leechseed', 'synthesis'], { evs: { hp: 252, def: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // Meganium Leech Seed → Snorlax ; Snorlax Splash
      { p1: 'move 2', p2: 'move 1' }, // Meganium Synthesis (heal so we see leech heal too); Splash
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ],
    [{ side: 0, hp: 100 }]); // injure Meganium so the seeder heal is visible

  // (B) GRASS-immune target — does accuracy still draw? p1 seeds a Grass Sceptile.
  await run('GRASS target immune (accuracy still drawn?)',
    [mon('Snorlax', ['leechseed', 'splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [mon('Sceptile', ['splash'], { evs: { hp: 252 } })], // Grass → immune
    [
      { p1: 'move 1', p2: 'move 1' }, // Leech Seed → Sceptile (Grass) → -immune
      { p1: 'move 2', p2: 'move 1' },
    ]);

  // (C) ALREADY-SEEDED fail. Seed once, then seed again — 2nd draws accuracy then -fail.
  await run('already-seeded 2nd Leech Seed FAILS',
    [mon('Snorlax', ['leechseed', 'splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [mon('Blissey', ['splash'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // seed Blissey
      { p1: 'move 1', p2: 'move 1' }, // seed AGAIN → already seeded → fail (acc still drawn?)
      { p1: 'move 2', p2: 'move 1' },
    ]);

  // (D) ORDER INTERACTION: leech + Leftovers + SANDSTORM chip + status DoT on the SAME
  //     seeded mon. p1 Tyranitar (Sand Stream) seeds nothing; the seeded mon is a p2
  //     Leftovers + burned mon under sand. We seed p2 with p1's lead? No — leech needs a
  //     Grass move user. Use a p1 Meganium seeder; p1 ALSO has Tyranitar? Single active.
  //     Simpler: p1 Cacturne (Sand Veil? no) — use Tyranitar lead on p2 for sand, p1
  //     Meganium seeds the sand-setting Tyranitar (Rock/Dark → sand-immune) — but we want
  //     the seeded mon to ALSO take sand+burn+leftovers. So seed a NON-sand-immune mon.
  //     p1 Meganium (Leech Seed) vs p2 Gengar (Leftovers, burn-injected) under sand from
  //     a p2 Tyranitar that we DON'T use — instead inject sand via setWeather.
  await run('leech + Leftovers + sand + burn on the SAME seeded mon (residual order)',
    [mon('Meganium', ['leechseed', 'synthesis'], { evs: { hp: 252, def: 252 } })],
    [mon('Gengar', ['splash'], { item: 'Leftovers', evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // seed Gengar
      { p1: 'move 2', p2: 'move 1' }, // residual: sand chip(8) + Leftovers(sub4) + LEECH(sub5) + burn(sub6)
      { p1: 'move 2', p2: 'move 1' },
    ],
    // inject AFTER start: sand weather + burn Gengar + chip its HP.
    [{ weather: 'sandstorm' }, { side: 1, status: 'brn', hp: 200 }]);

  // (E) seeder FAINTED (no heal): the leech drain still hits the seeded mon, but the
  //     seeder's active is fainted → no heal (getAtSlot returns a fainted mon → return).
  //     Hard to construct: seeder faints same turn. Instead: seed, then seeder switches
  //     out to a fainted... can't. Use the sourceSlot semantics: if the seeder's active
  //     faints, leech finds `target.fainted` → "Nothing to leech into" → drain SKIPPED
  //     entirely (the whole onResidual returns before this.damage!). VERIFY that.
  await run('seeder active fainted → leech does NOTHING (drain skipped)',
    [mon('Meganium', ['leechseed', 'splash'], { evs: { hp: 252 } }),
     mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [mon('Gengar', ['shadowball', 'splash'], { evs: { spa: 252, spe: 252 } })],
    [
      { p1: 'move 1', p2: 'move 2' }, // seed Gengar ; Gengar Splash
      // now KO Meganium: inject low HP + Gengar Shadow Ball. Meganium faints → forced switch.
      { p1: 'move 2', p2: 'move 1', stopAfter: true },
    ],
    [{ side: 0, hp: 30 }]);

  // (F) leech drain KOs the seeded mon. Seed a low-HP mon; the drain (maxhp/8) KOs it.
  await run('leech drain KOs the seeded mon',
    [mon('Meganium', ['leechseed', 'splash'], { evs: { hp: 252 } })],
    [mon('Gengar', ['splash'], { evs: { hp: 252 } }),
     mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // seed Gengar (inject Gengar to maxhp/8-1 so drain KOs)
      { p1: 'move 2', p2: 'move 1' }, // residual leech → KO Gengar
    ],
    [{ side: 1, hp: 20 }]); // Gengar low so leech drain KOs
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
