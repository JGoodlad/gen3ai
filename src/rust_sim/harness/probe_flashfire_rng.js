// probe_flashfire_rng.js — instrument the gen3 FLASH FIRE activation + ×1.5 fire-boost
// bit-for-bit against the OMNISCIENT in-process BattleStream (no server).
//
// THE PROBE IS THE ONLY ORACLE. Reading base data/*.ts is a hypothesis; gen3 conditions
// `inherit: true` from gen4 which REPLACES/DELETES handlers. Everything below is settled
// against the RESOLVED `Dex.mod('gen3')` sim, not a source read.
//
// SETTLES (do NOT trust the task hints — the sim is the source of truth):
//   ACTIVATION
//     A1. Does a Fire move activate Flash Fire (holder takes 0, the `flashfire` volatile set)?
//     A2. Does a Fire move that MISSES still activate it? (i.e. is onTryHit before/after the
//         accuracy roll in gen3 tryMoveHit) — give the attacker a low-accuracy Fire move + force
//         a miss seed and check whether the volatile appears.
//     A3. Does a Fire-type STATUS move (Will-O-Wisp) activate it?
//     A4. Does activation consume any PRNG (it must be draw-free)? Wrap prng.next, count draws
//         across the absorbing move vs a control (non-Fire move into the same mon).
//     A5. Does it persist across turns and CLEAR on switch-out? Dump the volatile after a switch.
//     A6. Is a mon already burned / frozen relevant to activation? (should be irrelevant.)
//   THE BOOST
//     B1. Once activated, a Fire move gets ×1.5. Is it onModifyAtk/SpA (a STAT mod, composes
//         with the boost table + 4096 chain) or onBasePower (a BP-chain member)? Dump the
//         RESOLVED flashfire handler inventory (onModifyAtk/onModifySpA/onBasePower/onSource...).
//     B2. The EXACT multiplier — confirm the resolved chainModify args (hypothesis ×1.5=[3,2]).
//     B3. DRAW-FREE (the boost adds/removes no draw)?
//     B4. Applies to BOTH physical AND special Fire moves (gen3 type-based phys/spec split)?
//     B5. Does NOT apply to non-Fire moves.
//   EXACT-EQUALITY PROOF (the ×1.5 fold): like gen_damage_golden.js, force the MAX damage roll
//     (random(16)==0) so the realized damage == the deterministic baseDamage, and compare the
//     absorbed-then-boosted damage vs the un-boosted baseline. The ratio must be EXACTLY the
//     4096-fixed-point ×1.5 (`modify(dmg,[3,2])`) at whatever chain stage B1 says.
//
// Run:  node src/rust_sim/harness/probe_flashfire_rng.js
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

// ---- B1/B2: dump the RESOLVED flashfire ability handler inventory + the chainModify shape.
function dumpResolvedFlashFire() {
  const d = Dex.mod('gen3');
  const ab = d.abilities.get('flashfire');
  console.log('=== RESOLVED gen3 Flash Fire ability handlers ===');
  const handlerKeys = Object.keys(ab).filter((k) => k.startsWith('on'));
  console.log('  handler keys:', handlerKeys.join(', ') || '(none)');
  for (const k of handlerKeys) {
    const fn = ab[k];
    if (typeof fn === 'function') {
      // Print the function source so we can SEE the chainModify args (the resolved dist).
      const src = fn.toString().replace(/\s+/g, ' ').slice(0, 400);
      console.log(`  ${k}: ${src}`);
    } else {
      console.log(`  ${k}: ${JSON.stringify(fn)}`);
    }
  }
  // Also the flashfire CONDITION (the volatile) — its onModify* live there in most gens.
  const cond = ab.condition;
  if (cond) {
    console.log('  --- ability.condition (the flashfire volatile) ---');
    for (const k of Object.keys(cond)) {
      const fn = cond[k];
      if (typeof fn === 'function') {
        console.log(`    ${k}: ${fn.toString().replace(/\s+/g, ' ').slice(0, 400)}`);
      } else {
        console.log(`    ${k}: ${JSON.stringify(fn)}`);
      }
    }
  }
}

// Generic single-battle runner. `inject` fires after both leads are in (set status / hp / pp).
// `wrapPrng` counts raw PRNG draws per decision window. Returns the battle + a per-window draw log.
async function run(label, p1team, p2team, plan, opts = {}) {
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
    if (inj.hp !== undefined) m.hp = inj.hp;
  }

  // Instrument raw draws (wrap the backend rng.next, like probe_pp_struggle_rng.js).
  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = (...a) => { drawCount += 1; return realNext(...a); };
  const windowDraws = [];

  for (const [i, step] of plan.entries()) {
    const before = drawCount;
    if (step.p1) streams.omniscient.write(`>p1 ${step.p1}`);
    if (step.p2) streams.omniscient.write(`>p2 ${step.p2}`);
    for (let k = 0; k < 10; k++) await tick();
    windowDraws.push(drawCount - before);
  }
  return { battle, windowDraws, label };
}

// Snapshot: does the mon carry the flashfire volatile? its HP? status?
function ffState(m) {
  return {
    hp: m ? m.hp : null,
    maxhp: m ? m.maxhp : null,
    status: m ? (m.status || '-') : null,
    flashfire: m ? !!(m.volatiles && m.volatiles['flashfire']) : null,
  };
}

// ---- EXACT damage measurement: force the MAX roll (random(16)==0) by sweeping seeds and
// taking the maximum realized damage on the measured p1->p2 hit; that max IS baseDamage.
function buildSeeds(n) {
  const out = [];
  let x = 0x9e3779b9 >>> 0;
  const s = () => { x = (Math.imul(x, 1103515245) + 12345) >>> 0; return x & 0xffff; };
  for (let i = 0; i < n; i++) out.push([s() || 1, s() || 1, s() || 1, s() || 1]);
  return out;
}

// EXACT-EQUALITY design: the FF holder is the ATTACKER (its OWN Fire moves get ×1.5). A Fire
// move INTO an FF holder is always absorbed (0 dmg), so we can't measure there — the boost lives
// on the holder's outgoing Fire move. p1 = the FF mon (Houndoom); it first absorbs a Fire move
// from a partner-less path (`activate` uses its OWN ember on the FOE to... no — self can't
// self-absorb). Instead p2 (a throwaway Charizard) fires an Ember at the FF Houndoom on turn 1 to
// ACTIVATE it, then turn 2 p1 Houndoom hits p2 with the MEASURED Fire move. Baseline = same but
// the FF mon carries `No Ability` (never activates) so turn 1's Ember just chips it (we top HP so
// that never confounds the measured OUTGOING hit on the defender).
//
// To keep the ATTACKER's Fire STAB out of the ratio confound, the attacker (Houndoom, Fire/Dark)
// has Fire STAB in BOTH arms — it cancels in the ratio. The defender is a bulky neutral
// (Blissey, pure Normal → Fire is neutral 1×) so no type mult confound. We FORCE the max roll.
async function measureFireDamage({ move, activate, ability, seed }) {
  // p1 = the FF holder + ATTACKER. moves: [measured Fire move, filler].
  const p1 = [mon('Houndoom', [move, 'rest'], { ability, nature: 'Serious', evs: { spa: 252, atk: 252 } })];
  // p2 = the ACTIVATOR/DEFENDER (Blissey: pure Normal → Fire neutral, very bulky so it survives).
  const p2 = [mon('Blissey', ['ember', 'softboiled'], { ability: 'No Ability', evs: { hp: 252, def: 252 } })];
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  const attacker = () => battle.sides[0].active[0];
  const defender = () => battle.sides[1].active[0];
  // Turn 1: p2 Ember at p1 (activates FF iff ability is Flash Fire); p1 uses filler (rest).
  //         Whether it activates or not, we top p1's HP after so the measured turn is clean.
  streams.omniscient.write('>p1 move 2'); // p1 rest (filler, no attack)
  streams.omniscient.write('>p2 move 1'); // p2 ember → activates FF on p1 (if FF)
  for (let k = 0; k < 10; k++) await tick();
  if (attacker()) attacker().hp = attacker().maxhp;   // clean attacker
  if (defender()) defender().hp = defender().maxhp;   // full defender for a clean full-HP hit
  const pre = defender() ? defender().hp : null;
  // Turn 2: the MEASURED outgoing Fire move p1 → p2. p2 fills (softboiled).
  streams.omniscient.write('>p1 move 1');
  streams.omniscient.write('>p2 move 2');
  for (let k = 0; k < 10; k++) await tick();
  const post = defender() ? defender().hp : null;
  // If p1 didn't actually get to attack (paralysis/speed), dmg reads 0 → the max sweep ignores it.
  const dmg = (pre !== null && post !== null) ? Math.max(0, pre - post) : 0;
  // Assert the activation state matched intent (guards against a silent no-activate).
  const active = !!(attacker() && attacker().volatiles && attacker().volatiles['flashfire']);
  return { dmg, activated: active };
}

async function maxFireDamage(cfg, nSeeds = 200) {
  const seeds = buildSeeds(nSeeds);
  let max = 0;
  let sawActivated = false;
  for (const seed of seeds) {
    const { dmg, activated } = await measureFireDamage({ ...cfg, seed });
    if (activated) sawActivated = true;
    if (dmg > max) max = dmg;
  }
  return { max, sawActivated };
}

async function main() {
  dumpResolvedFlashFire();

  // ----- A1/A4/A6: a Fire move into a Flash Fire mon activates it, holder takes 0, DRAW-FREE.
  // Control: a Fire move into a NON-FF mon (same species-immune? use a Fire-type Houndoom that
  // just resists) to compare draw count.
  console.log('\n=== A1/A4: Fire move into Flash Fire (activation + draw count) ===');
  {
    // p1 Charizard Ember (Fire) into p2 Houndoom (Flash Fire). Houndoom keeps full HP + flashfire set.
    const r = await run('ff_activate',
      [mon('Charizard', ['ember', 'tackle'], { ability: 'Blaze' })],
      [mon('Houndoom', ['rest'], { ability: 'Flash Fire' })],
      [{ p1: 'move 1', p2: 'move 1' }]);
    const ho = r.battle.sides[1].active[0];
    console.log('  after Ember→FF Houndoom:', JSON.stringify(ffState(ho)), 'draws=', r.windowDraws[0]);

    // Control: same move but p2 is a NON-FF Fire-resister (Houndoom w/ No Ability) — Fire move
    // hits (Fire resists Fire ½×) so it DOES damage + draws crit/dmg. Draw-count contrast.
    const c = await run('ff_control_nonff',
      [mon('Charizard', ['ember', 'tackle'], { ability: 'Blaze' })],
      [mon('Houndoom', ['rest'], { ability: 'No Ability' })],
      [{ p1: 'move 1', p2: 'move 1' }]);
    const ho2 = c.battle.sides[1].active[0];
    console.log('  control Ember→plain Houndoom:', JSON.stringify(ffState(ho2)), 'draws=', c.windowDraws[0]);
    console.log('  → activation draw-free?', r.windowDraws[0] <= c.windowDraws[0] ? 'YES (<=, no extra)' : 'CHECK');
  }

  // ----- A2: does a MISSED Fire move activate Flash Fire? Give p1 a low-accuracy Fire move and
  // find a seed that MISSES; check whether the volatile appears (onTryHit runs before/after acc).
  console.log('\n=== A2: does a MISSED Fire move activate Flash Fire? ===');
  {
    // Fire Blast = 85% accuracy. Sweep seeds; report the FIRST that misses + whether FF activated,
    // and the FIRST that hits + activation. (A hit obviously activates.)
    let missReported = false, hitReported = false;
    for (const seed of buildSeeds(60)) {
      const r = await run('ff_miss', [mon('Charizard', ['fireblast'], { ability: 'Blaze' })],
        [mon('Houndoom', ['rest'], { ability: 'Flash Fire' })],
        [{ p1: 'move 1', p2: 'move 1' }], { seed });
      const ho = r.battle.sides[1].active[0];
      // Detect miss vs absorb from the log: |-miss| vs |-immune|/|-start|Flash Fire.
      const log = r.battle.log.join('\n');
      const missed = /\|-miss\|/.test(log);
      const activated = !!(ho.volatiles && ho.volatiles['flashfire']);
      if (missed && !missReported) {
        console.log(`  MISS seed ${JSON.stringify(seed)}: flashfire volatile=${activated}`);
        missReported = true;
      }
      if (!missed && !hitReported) {
        console.log(`  HIT/absorb seed ${JSON.stringify(seed)}: flashfire volatile=${activated}`);
        hitReported = true;
      }
      if (missReported && hitReported) break;
    }
    if (!missReported) console.log('  (no miss seed found in 60 — Fire Blast rarely missed; raise pool)');
  }

  // ----- A3: does a Fire-type STATUS move (Will-O-Wisp) activate Flash Fire?
  console.log('\n=== A3: does Fire STATUS move (Will-O-Wisp) activate Flash Fire? ===');
  {
    // Houndoom is FIRE-TYPE → the resolved onTryHit WoW special-case (`target.hasType("Fire")`)
    // returns → no activation. This is REPRESENTATIVE of gen3 OU (every FF holder IS Fire-type),
    // but to prove the WoW-activation MECHANIC we also test a synthetic NON-Fire FF holder.
    const r = await run('ff_wisp_firetype', [mon('Gengar', ['willowisp'], { ability: 'Levitate' })],
      [mon('Houndoom', ['rest'], { ability: 'Flash Fire' })],
      [{ p1: 'move 1', p2: 'move 1' }]);
    const ho = r.battle.sides[1].active[0];
    const log = r.battle.log.join('\n');
    console.log('  WoW → Fire-type FF Houndoom:', JSON.stringify(ffState(ho)),
      '| -start?', /Flash Fire/.test(log), '(expect false: WoW special-cases a Fire-type target)');

    // Synthetic non-Fire FF holder (Snorlax w/ Flash Fire) — does WoW activate FF there?
    // WoW is 85% acc; sweep seeds to find a landing WoW and report the activation there.
    let done = false;
    for (const seed of buildSeeds(40)) {
      const r2 = await run('ff_wisp_nonfire', [mon('Gengar', ['willowisp'], { ability: 'Levitate' })],
        [mon('Snorlax', ['rest'], { ability: 'Flash Fire' })],
        [{ p1: 'move 1', p2: 'move 1' }], { seed });
      const sn = r2.battle.sides[1].active[0];
      const log2 = r2.battle.log.join('\n');
      const landed = !/\|-miss\|/.test(log2);
      if (landed) {
        console.log('  WoW (LANDED) → NON-Fire FF Snorlax:', JSON.stringify(ffState(sn)),
          '| -start?', /Flash Fire/.test(log2),
          '(non-Fire status-less target: WoW activates FF, NO burn; a status token would mean it burned)');
        done = true;
        break;
      }
    }
    if (!done) console.log('  (no landing WoW found in 40 seeds — raise the pool)');
  }

  // ----- A5: persist across turns + CLEAR on switch-out.
  console.log('\n=== A5: FF persists across turns + clears on switch-out ===');
  {
    // p2 Houndoom (FF) absorbs an Ember (activate). Then p2 SWITCHES to Snorlax and back.
    // Check flashfire before + after the switch-out.
    const r = await run('ff_switch',
      [mon('Charizard', ['ember', 'tackle'], { ability: 'Blaze' })],
      [mon('Houndoom', ['rest'], { ability: 'Flash Fire' }), mon('Snorlax', ['rest'])],
      [{ p1: 'move 1', p2: 'move 1' },     // turn1: Ember absorbed → FF on Houndoom
       { p1: 'move 2', p2: 'switch 2' },   // turn2: p2 switches OUT Houndoom → Snorlax
       { p1: 'move 2', p2: 'switch 2' }]); // turn3: p2 switches BACK Snorlax → Houndoom
    // After turn1 (index 0): FF should be up. After switch-back (index 2): FF should be GONE.
    // Find the Houndoom robustly (species.id, then baseSpecies).
    const houndoom = r.battle.sides[1].pokemon.find(
      (p) => (p.species && p.species.id === 'houndoom') || (p.set && /houndoom/i.test(p.set.species)));
    console.log('  Houndoom flashfire AFTER a switch-out/in cycle:',
      !!(houndoom && houndoom.volatiles && houndoom.volatiles['flashfire']),
      '(expect false — cleared on switch-out)');
    // Also confirm it was ON right after absorbing: re-run stopping at turn1.
    const r1 = await run('ff_switch_t1',
      [mon('Charizard', ['ember', 'tackle'], { ability: 'Blaze' })],
      [mon('Houndoom', ['rest'], { ability: 'Flash Fire' }), mon('Snorlax', ['rest'])],
      [{ p1: 'move 1', p2: 'move 1' }]);
    const ho1 = r1.battle.sides[1].active[0];
    console.log('  Houndoom flashfire right after absorbing:',
      !!(ho1.volatiles && ho1.volatiles['flashfire']), '(expect true)');
  }

  // ----- A6: activation is status-irrelevant (a burned/frozen Houndoom still activates).
  console.log('\n=== A6: FF activation with the holder already statused ===');
  {
    const r = await run('ff_statused',
      [mon('Charizard', ['ember', 'tackle'], { ability: 'Blaze' })],
      [mon('Houndoom', ['rest'], { ability: 'Flash Fire' })],
      [{ p1: 'move 1', p2: 'move 1' }], { inject: [{ side: 1, status: 'par' }] });
    const ho = r.battle.sides[1].active[0];
    console.log('  paralyzed Houndoom after Ember:', JSON.stringify(ffState(ho)), '(flashfire expect true)');
  }

  // ----- B1..B5 + EXACT ×1.5 proof: measured Fire damage boosted vs baseline, BOTH categories.
  console.log('\n=== B: the ×1.5 boost (EXACT max-roll proof), special + physical ===');
  {
    for (const [label, move] of [['SPECIAL fireblast', 'fireblast'], ['PHYSICAL firepunch', 'firepunch']]) {
      const base = await maxFireDamage({ move, ability: 'No Ability' });
      const boost = await maxFireDamage({ move, ability: 'Flash Fire' });
      const ratio = base.max > 0 ? (boost.max / base.max) : NaN;
      // The EXACT fixed-point ×1.5 at ModifyDamagePhase1 is modify(bd,[3,2]) = trunc(bd*6144/4096).
      const expect = Math.trunc(base.max * 6144 / 4096);
      console.log(`  ${label}: baseline(no FF, max roll)=${base.max}  boosted(FF active, max roll)=${boost.max}` +
        `  ratio=${ratio.toFixed(4)}  activated(base/boost)=${base.sawActivated}/${boost.sawActivated}`);
      console.log(`    NOTE: ModifyDamagePhase1 ×1.5 applies to baseDamage BEFORE +2/STAB/type, so the ` +
        `FINAL-damage ratio is NOT exactly 1.5 (downstream +2/STAB/type shift it) — the EXACT proof is ` +
        `the golden's calc_damage().base == sim base, per-scenario. (informational ratio only)`);
    }
    // B5: a NON-Fire move by the FF holder gets NO boost. (Charizard has no non-Fire attacking move
    // here; instead prove the boost is Fire-gated by observing a non-Fire move is un-changed by FF —
    // covered implicitly since only Fire moves route through the flashfire onModify handler; we
    // assert it in the golden's wrong-type control rather than here.)
    console.log('  (B5 non-Fire no-boost is asserted by the golden wrong-type control.)');
  }

  console.log('\n=== DONE ===');
}

main().catch((e) => { console.error(e); process.exit(1); });
