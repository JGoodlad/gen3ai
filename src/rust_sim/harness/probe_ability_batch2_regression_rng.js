// probe_ability_batch2_regression_rng.js — GROUND TRUTH for the `gen3_ability_batch2_v1`
// regression pins (B2-1 CONTACT_PROC Static / B2-2 Effect Spore sample / B2-3 Rough Skin /
// B2-4 Damp / B2-5 Soundproof / B2-6 Suction Cups / B2-7 Synchronize) in tests/regression_test.rs.
//
// Each scenario drives the OMNISCIENT in-process BattleStream over a CONSTRUCTED gen3customgame
// board whose EXACT packed teams + seed the Rust regression test replays. It RESEEDS the sim's
// prng to the RAW seed right before the first decision (so the decision draws line up with the
// Rust's draw-free `start_with_switchins`), then prints the per-decision seedAfter + observable
// STATE — copied verbatim into the pins as the real-Showdown ground truth. Each pin FAILS if its
// class's engine wiring is reverted.
//
// For the CONTACT_PROC scenarios it SCANS a seed range to find one where the proc PASSES on the
// first contact hit (so the ATTACKER is statused + the proc's randomChance is exercised).
//
// Run:  node src/rust_sim/harness/probe_ability_batch2_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));

function tick() { return new Promise((r) => setTimeout(r, 0)); }

// Drive one constructed battle, reseeding to `rawSeed` at the first decision, and print each
// decision's seedAfter + both actives' species/hp/status + the first mover. Returns the log.
async function run(label, p1, p2, rawSeed, plan, opts = {}) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[1,2,3,4]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2 })}`);
  for (let i = 0; i < 12; i++) await tick();
  const b = stream.battle;
  b.prng = new PRNG(rawSeed.slice());

  const a0 = () => b.sides[0].active[0], a1 = () => b.sides[1].active[0];
  if (!opts.quiet) console.log(`\n=== ${label} (raw seed ${rawSeed.join(',')}) ===`);

  let i = 0, safety = 0;
  const decs = [];
  while (!b.ended && safety < 60 && i < plan.length) {
    safety++;
    const rs = b.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const before = log.length;
    const entry = plan[i]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const first = (() => {
      for (let j = before; j < log.length; j++) {
        const p = log[j].split('|');
        if ((p[1] === 'move' || p[1] === 'cant') && p.length >= 3) {
          const a = (p[2] || '').trim();
          if (a.startsWith('p1a:')) return 'p1';
          if (a.startsWith('p2a:')) return 'p2';
        }
      }
      return 'none';
    })();
    const fmt = (m) => `${m.species.name} ${m.hp}/${m.maxhp}${m.status ? ' ' + m.status : ''}`;
    const seedAfter = b.prng.getSeed();
    decs.push({ seedAfter, p1: fmt(a0()), p2: fmt(a1()), first, lines: log.slice(before) });
    if (!opts.quiet) {
      console.log(`  dec ${i - 1} [${rs}] ${JSON.stringify(entry)} first=${first}`);
      console.log(`    seedAfter=${seedAfter}`);
      console.log(`    p1=${fmt(a0())} | p2=${fmt(a1())}`);
      const notable = log.slice(before).filter((l) => /-status|ability:|Synchronize|cant|-immune|-activate|-damage.*ability/.test(l));
      if (notable.length) console.log(`    lines: ${JSON.stringify(notable)}`);
    }
  }
  try { streams.omniscient.destroy(); } catch (e) {}
  return { decs, ended: b.ended, log };
}

// Scan raw seeds to find one where the first contact-proc `randomChance` PASSES (the attacker is
// statused by the ability). Returns the first passing raw seed (or null).
async function findProcSeed(p1, p2, statusRe, base) {
  let s = base >>> 0;
  const rng = () => { s = (s * 1664525 + 1013904223) >>> 0; return s; };
  for (let n = 0; n < 400; n++) {
    const seed = [rng() % 65536, rng() % 65536, rng() % 65536, rng() % 65536];
    const r = await run('scan', p1, p2, seed, [{ p1: 'move 1', p2: 'move 1' }], { quiet: true });
    if (r.decs[0] && statusRe.test(r.decs[0].lines.join('\n'))) return seed;
  }
  return null;
}

async function main() {
  // ── B2-1 CONTACT_PROC Static: Body Slam (contact) into a Static holder → the ATTACKER is
  //    paralyzed. Scan for a seed where the proc PASSES on the first hit.
  const attacker = 'Snorlax|||NoAbility|bodyslam,earthquake|Adamant|252,252,,,,|N||||';
  const static_holder = 'Electabuzz|||Static|thunderbolt,thunderbolt|Modest|,,,,252,252|N||||';
  const staticControl = 'Electabuzz|||Insomnia|thunderbolt,thunderbolt|Modest|,,,,252,252|N||||';
  const staticSeed = await findProcSeed(attacker, static_holder, /-status\|p1a: Snorlax\|par\|\[from\] ability: Static/, 0x11223344);
  console.log('B2-1 Static proc seed:', staticSeed);
  if (staticSeed) {
    await run('B2-1 CONTACT_PROC Static (Body Slam paras the attacker)', attacker, static_holder, staticSeed, [{ p1: 'move 1', p2: 'move 1' }]);
    await run('B2-1 CONTROL (Insomnia — no proc, attacker un-statused)', attacker, staticControl, staticSeed, [{ p1: 'move 1', p2: 'move 1' }]);
  }

  // ── B2-2 Effect Spore: Body Slam into an Effect Spore holder → randomChance(1,10) then
  //    sample(slp/par/psn). Scan for a seed where the sample fires (the attacker gets a status).
  const es_holder = 'Vileplume|||EffectSpore|sludgebomb,sludgebomb|Modest|,,,,252,252|N||||';
  const esSeed = await findProcSeed(attacker, es_holder, /-status\|p1a: Snorlax\|(slp|par|psn)\|\[from\] ability: Effect Spore/, 0x55667788);
  console.log('B2-2 Effect Spore proc seed:', esSeed);
  if (esSeed) {
    await run('B2-2 Effect Spore (sample a status onto the attacker)', attacker, es_holder, esSeed, [{ p1: 'move 1', p2: 'move 1' }]);
  }

  // ── B2-3 Rough Skin: Body Slam into a Rough Skin holder → the attacker loses maxhp/16 recoil
  //    (DRAW-FREE). Any seed works (deterministic recoil); the CONTROL (no-op) has NO recoil.
  const rs_holder = 'Sharpedo|||RoughSkin|surf,surf|Modest|,,,,252,252|N||||';
  const rsControl = 'Sharpedo|||Insomnia|surf,surf|Modest|,,,,252,252|N||||';
  const rsSeed = [40012, 7781, 55230, 19004];
  await run('B2-3 Rough Skin (attacker takes maxhp/16 recoil, draw-free)', attacker, rs_holder, rsSeed, [{ p1: 'move 1', p2: 'move 1' }]);
  await run('B2-3 CONTROL (Insomnia — NO recoil, same seed)', attacker, rsControl, rsSeed, [{ p1: 'move 1', p2: 'move 1' }]);

  // ── B2-4 Damp: p1 Snorlax uses Explosion while p2 Golduck (Damp) is active → CANCELLED, Snorlax
  //    does NOT self-KO, the move draws NOTHING. Control: a no-op Golduck → Explosion self-KOs.
  const boomer = 'Snorlax|||NoAbility|explosion,bodyslam|Adamant|,252,,,,|N||||';
  const damp_holder = 'Golduck|||Damp|surf,surf|Modest|,,,,252,252|N||||';
  const dampControl = 'Golduck|||Insomnia|surf,surf|Modest|,,,,252,252|N||||';
  const dampSeed = [12345, 54321, 11111, 22222];
  await run('B2-4 Damp (Explosion cancelled, Snorlax does NOT self-KO)', boomer, damp_holder, dampSeed, [{ p1: 'move 1', p2: 'move 1' }]);
  await run('B2-4 CONTROL (no-op — Explosion self-KOs Snorlax)', boomer, dampControl, dampSeed, [{ p1: 'move 1', p2: 'move 1' }]);

  // ── B2-5 Soundproof: p1 Jynx Sings (sound) into p2 Electrode (Soundproof) → immune, no sleep.
  //    Control: a no-op Electrode → Jynx's Sing puts it to sleep (accuracy 55; scan for a land).
  const singer = 'Jynx|||NoAbility|sing,icebeam|Modest|,,252,,,|N||||';
  const sp_holder = 'Electrode|||Soundproof|thunderbolt,thunderbolt|Timid|,,,,252,252|N||||';
  const spControl = 'Electrode|||Insomnia|thunderbolt,thunderbolt|Timid|,,,,252,252|N||||';
  // Insomnia BLOCKS sleep too, so use a different no-op for the control that CAN sleep: Static.
  const spSleepControl = 'Electrode|||Static|thunderbolt,thunderbolt|Timid|,,,,252,252|N||||';
  const spSeed = [30982, 33910, 19571, 50263];
  await run('B2-5 Soundproof (Sing immune, Electrode NOT asleep)', singer, sp_holder, spSeed, [{ p1: 'move 1', p2: 'move 1' }]);
  // find a seed where Sing LANDS on the Static control (proving Soundproof is what blocks it)
  const spLand = await findProcSeed(singer, spSleepControl, /-status\|p2a: Electrode\|slp/, 0x99aabbcc);
  console.log('B2-5 Sing-lands control seed:', spLand);
  if (spLand) {
    await run('B2-5 CONTROL (Static — Sing LANDS, Electrode asleep)', singer, spSleepControl, spLand, [{ p1: 'move 1', p2: 'move 1' }]);
    await run('B2-5 Soundproof at the LAND seed (still immune)', singer, sp_holder, spLand, [{ p1: 'move 1', p2: 'move 1' }]);
  }

  // ── B2-6 Suction Cups: p2 Suicune Roars (priority -6) into p1 Cradily (Suction Cups) + a bench
  //    → the drag is BLOCKED (no sample); Cradily stays active. Control: a no-op Cradily is dragged.
  const sc_holder = 'Cradily|||SuctionCups|surf,rest|Bold|252,,252,,,|N||||';
  const scControl = 'Cradily|||Insomnia|surf,rest|Bold|252,,252,,,|N||||';
  const cradilyBench = 'Snorlax|||NoAbility|bodyslam,rest|Impish|252,,252,,,|N||||';
  const roarer = 'Suicune|||NoAbility|roar,surf|Bold|252,,252,,,|N||||';
  const scSeed = [13127, 45333, 18295, 15391];
  await run('B2-6 Suction Cups (Roar blocked, Cradily STAYS, no sample)',
    sc_holder + ']' + cradilyBench, roarer, scSeed, [{ p1: 'move 1', p2: 'move 1' }]);
  await run('B2-6 CONTROL (Insomnia — Cradily is DRAGGED, sample drawn)',
    scControl + ']' + cradilyBench, roarer, scSeed, [{ p1: 'move 1', p2: 'move 1' }]);

  // ── B2-7 Synchronize: p1 Jolteon Thunder Waves p2 Alakazam (Synchronize) → Alakazam para'd AND
  //    Jolteon (the source) para'd too. Control: a no-op Alakazam → only Alakazam para'd.
  const twaver = 'Jolteon|||NoAbility|thunderwave,thunderbolt|Timid|,,,,252,252|N||||';
  const sync_holder = 'Alakazam|||Synchronize|psychic,recover|Timid|,,252,,252,|N||||';
  const syncControl = 'Alakazam|||Insomnia|psychic,recover|Timid|,,252,,252,|N||||';
  const syncSeed = [42782, 54377, 52057, 58231];
  await run('B2-7 Synchronize (Thunder Wave reflects par to the caster)', twaver, sync_holder, syncSeed, [{ p1: 'move 1', p2: 'move 1' }]);
  await run('B2-7 CONTROL (Insomnia — no reflect, Jolteon un-statused)', twaver, syncControl, syncSeed, [{ p1: 'move 1', p2: 'move 1' }]);

  // ── B2-8 CONTACT_PROC behind a SUBSTITUTE (the review-caught bug fix): a Static holder subs, then a
  //    WEAK contact hit (Chansey Tackle) is ABSORBED by the SURVIVING sub → the proc does NOT fire
  //    (onDamagingHit is on the MON, not the sub — the `!absorbed` gate) → the ATTACKER (Chansey) is
  //    UN-statused, DRAW-FREE (no contact-proc randomChance). Reverting `!absorbed` fires a phantom
  //    randomChance → shifts the seed (and can paralyze Chansey). p2 Electabuzz is faster → subs first.
  const weakAtk = 'Chansey|||NoAbility|tackle,softboiled|Bold|252,,252,,,|N||||';
  const subStatic = 'Electabuzz|||Static|substitute,thunderbolt|Timid|,,,,252,252|N||||';
  const subSeed = [9137, 21044, 5510, 43902];
  await run('B2-8 CONTACT_PROC behind a surviving sub (Static does NOT proc, attacker un-statused, draw-free)',
    weakAtk, subStatic, subSeed, [{ p1: 'move 1', p2: 'move 1' }]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
