// probe_substitute_rng.js — instrument the gen3 SUBSTITUTE draw model bit-for-bit.
//
// Verifies, against the OMNISCIENT in-process BattleStream (no server), the EXACT
// PRNG draw count + the resulting STATE for each Substitute branch — the CRUX being
// the secondary-block draw-COUNT:
//
//   (1) Substitute CREATE: never-miss (no accuracy draw), spends floor(maxhp/4),
//       creates a `substitute` volatile with hp=floor(maxhp/4). DRAW-FREE create.
//   (2) Substitute FAIL: hp <= floor(maxhp/4) (can't afford) OR already-subbed → FAIL,
//       draw-free.
//   (3) A DAMAGING move into a sub: draws acc+crit+damage (the sub absorbs the HP); the
//       SECONDARY random(100) is NOT drawn (the sub-block short-circuits it). THE PROOF:
//       a secondary move vs a SUB draws ONE FEWER random(100) than vs a BARE mon.
//   (4) The sub BREAKS when a hit >= sub HP (excess does NOT carry in gen3).
//   (5) A STATUS move (Thunder Wave / Toxic) into a sub is BLOCKED (no status). Accuracy?
//   (6) A stat-DROP move (e.g. an Intimidate-like / a -SpD secondary) is blocked by sub.
//   (7) A CONFUSION self-hit hits the SUBSTITUTE's HP, not the mon.
//   (8) PHAZE (Roar) BYPASSES the sub — the user is dragged anyway.
//
// We wrap battle.prng.next to count raw draws per decision window, and snapshot the
// `substitute` volatile HP each turn.
//
// Run:  node src/rust_sim/harness/probe_substitute_rng.js
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

function dumpResolved() {
  const dex3 = Dex.forFormat(FORMAT);
  const sub = dex3.moves.get('substitute');
  console.log('=== resolved gen3 substitute move ===');
  console.log(`  accuracy=${sub.accuracy} type=${sub.type} category=${sub.category} ` +
    `volatileStatus=${sub.volatileStatus} target=${sub.target}`);
  const cond = sub.condition || {};
  console.log(`  condition.onTryPrimaryHitPriority=${cond.onTryPrimaryHitPriority}`);
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
    if (inj.confusion) m.addVolatile('confusion');
  }

  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 50) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    if (i >= plan.length) break; // plan exhausted → stop (no infinite repeat)
    const before = battle.prng.getSeed();
    const dc0 = drawCount;
    const entry = plan[i]; i++;
    // Mid-battle injection BEFORE submitting this turn's choices (e.g. confuse a
    // mon once its sub is up, or chip a mon to a precise HP).
    if (entry.injectBefore) {
      for (const inj of entry.injectBefore) {
        const m = battle.sides[inj.side].active[0];
        if (inj.hp !== undefined) m.hp = inj.hp;
        if (inj.confusion) m.addVolatile('confusion');
        if (inj.status) m.setStatus(inj.status, m, null, true);
      }
    }
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const subOf = (m) => (m && m.volatiles && m.volatiles['substitute'])
      ? `SUB(${m.volatiles['substitute'].hp})` : '';
    const confOf = (m) => (m && m.volatiles && m.volatiles['confusion']) ? 'CONF' : '';
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'}${m.fainted ? ' FNT' : ''} ${subOf(m)}${confOf(m)}` : '-';
    console.log(`  [${rs}] ${JSON.stringify(entry)} draws=${drawCount - dc0}  seedBefore=${before} seedAfter=${after}`);
    console.log(`        p1=${fmt(a0)} | p2=${fmt(a1)}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  dumpResolved();

  // (A) CREATE: Snorlax subs. Cost floor(maxhp/4), sub.hp=floor(maxhp/4). Both never-miss.
  //     Baseline: a Splash/Splash turn draws only Quick Claw (1). A Substitute/Splash turn?
  await run('CREATE substitute (cost + sub hp, draw count)',
    [mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252 } })],
    [mon('Blissey', ['splash'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [
      { p1: 'move 2', p2: 'move 1' }, // Splash/Splash baseline → draws = ?
      { p1: 'move 1', p2: 'move 1' }, // Snorlax Substitute → cost + create
      { p1: 'move 1', p2: 'move 1' }, // Substitute AGAIN → already-subbed FAIL (draw-free?)
    ]);

  // (B) FAIL at low HP: hp <= floor(maxhp/4) → can't afford. Snorlax maxhp=524,
  //     floor(524/4)=131. Inject hp=131 (== threshold → FAIL) then hp=132 (> threshold → OK).
  await run('FAIL at low HP (hp <= floor(maxhp/4)) then OK at +1',
    [mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252 } })],
    [mon('Blissey', ['splash'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1', injectBefore: [{ side: 0, hp: 131 }] }, // hp==131 → FAIL
      { p1: 'move 1', p2: 'move 1', injectBefore: [{ side: 0, hp: 132 }] }, // hp==132 → OK (cost 131)
    ]);

  // (C) THE SECONDARY-BLOCK DRAW-COUNT PROOF. A secondary move (Body Slam, par30) into
  //     a BARE Blissey vs a SUBBED Blissey. The bare hit draws acc+crit+dmg+SECONDARY(100).
  //     The sub hit draws acc+crit+dmg and NO secondary(100) → one fewer.
  //     We make Blissey sub first, then Snorlax Body Slams it.
  await run('SECONDARY blocked by sub: vs BARE Blissey (draws the secondary 100)',
    [mon('Snorlax', ['bodyslam', 'splash'], { evs: { atk: 252 } })],
    [mon('Blissey', ['splash', 'softboiled'], { ability: 'Natural Cure', evs: { hp: 252, def: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // Body Slam BARE Blissey → acc+crit+dmg+sec(100)+QC
      { p1: 'move 1', p2: 'move 1' },
    ]);
  await run('SECONDARY blocked by sub: vs SUBBED Blissey (NO secondary 100 → 1 fewer)',
    [mon('Snorlax', ['bodyslam', 'splash'], { evs: { atk: 252 } })],
    [mon('Blissey', ['substitute', 'softboiled'], { ability: 'Natural Cure', evs: { hp: 252, def: 252 } })],
    [
      { p1: 'move 2', p2: 'move 1' }, // p1 Splash, p2 Blissey Substitute → sub up
      { p1: 'move 1', p2: 'move 2' }, // p1 Body Slam INTO the sub ; p2 Soft-Boiled → no sec(100)
      { p1: 'move 1', p2: 'move 2' },
    ]);

  // (D) BREAK: a hit >= sub HP breaks it (excess does NOT carry to the mon in gen3).
  //     Snorlax big Body Slam into a low-HP-sub Blissey. We need the hit > sub.hp.
  await run('BREAK the sub (excess does not carry)',
    [mon('Snorlax', ['bodyslam', 'splash'], { evs: { atk: 252 }, nature: 'Adamant' })],
    [mon('Gengar', ['substitute', 'splash'], { evs: { hp: 0 } })], // low maxhp Gengar → small sub
    [
      { p1: 'move 2', p2: 'move 1' }, // p1 Splash, p2 Gengar Substitute → sub.hp=floor(maxhp/4)
      { p1: 'move 1', p2: 'move 2' }, // p1 Body Slam (big) into sub → BREAK, mon HP unchanged
      { p1: 'move 1', p2: 'move 2' }, // now sub gone → next hit hits the mon
    ]);

  // (E) STATUS move into a sub is BLOCKED. Thunder Wave into a subbed mon. Accuracy drawn?
  await run('STATUS move (Thunder Wave) blocked by sub',
    [mon('Jolteon', ['thunderwave', 'splash'], { evs: { spe: 252 } })],
    [mon('Blissey', ['substitute', 'softboiled'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [
      { p1: 'move 2', p2: 'move 1' }, // p1 Splash, p2 Blissey Substitute
      { p1: 'move 1', p2: 'move 2' }, // p1 Thunder Wave INTO sub → blocked, no par; Soft-Boiled
      { p1: 'move 1', p2: 'move 2' },
    ]);

  // (F) STAT-DROP secondary blocked by sub: Crunch (-1 SpD secondary) into a subbed mon.
  await run('STAT-DROP secondary (Crunch -1 SpD) blocked by sub',
    [mon('Tyranitar', ['crunch', 'splash'], { evs: { atk: 252 } })],
    [mon('Gengar', ['substitute', 'splash'], { evs: { hp: 252 } })],
    [
      { p1: 'move 2', p2: 'move 1' }, // p1 Splash, p2 Gengar Substitute
      { p1: 'move 1', p2: 'move 2' }, // p1 Crunch INTO sub → no -SpD (the secondary blocked)
      { p1: 'move 1', p2: 'move 2' },
    ]);

  // (G) CONFUSION self-hit hits the SUB. Sub up + confused → the typeless 40-BP self-hit
  //     hits the sub's HP, not the mon. We inject confusion onto a subbed mon.
  await run('CONFUSION self-hit hits the SUB (not the mon)',
    [mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252, atk: 252 } })],
    [mon('Blissey', ['splash'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // Snorlax Substitute (sub up)
      // confuse Snorlax now (sub is up); Splash so the only action is the confusion check
      { p1: 'move 2', p2: 'move 1', injectBefore: [{ side: 0, confusion: true }] },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]);

  // (H) PHAZE bypasses the sub: Roar drags the subbed mon out anyway.
  await run('PHAZE (Roar) bypasses the sub — user dragged',
    [mon('Suicune', ['roar', 'splash'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252 } }),
     mon('Gengar', ['splash'], { evs: { hp: 252 } })],
    [
      { p1: 'move 2', p2: 'move 1' }, // p1 Splash, p2 Snorlax Substitute (sub up)
      { p1: 'move 1', p2: 'move 2' }, // p1 Roar → drag p2 (bypasses sub); p2 Splash
    ]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
