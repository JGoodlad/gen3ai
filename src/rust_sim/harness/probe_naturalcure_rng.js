// probe_naturalcure_rng.js — settle the gen3 NATURAL CURE switch-out cure bit-for-bit
// against the OMNISCIENT in-process BattleStream (no server). THE PROBE IS THE ONLY ORACLE.
//
// The dump probe (probe_naturalcure_dump.js) already established from the RESOLVED
// `Dex.mod('gen3')`:
//   - naturalcure has `onCheckShow: undefined` and a real `onSwitchOut`:
//       onSwitchOut(pokemon) { if (!pokemon.status || pokemon.status==='fnt') return;
//         this.add('-curestatus', ...); pokemon.clearStatus(); }
//   - `switchIn` fires `runEvent('SwitchOut', oldActive)` on BOTH voluntary + drag
//     (only `BeforeSwitchOut` is `!isDrag`-gated), BEFORE `clearVolatile()`.
//   - `clearStatus() → setStatus('')` wipes status + statusState (tox stage / sleep counter).
//
// THIS probe NAILS the DRAW MODEL + timing + which-statuses + phaze-drag empirically:
//   D1. THE DRAW-COUNT CRUX. Compare the sim's raw PRNG draw COUNT on a switch-out for:
//        (a) a NATURAL-CURE mon switching out while STATUSED,
//        (b) a NON-NC mon switching out while STATUSED (same board),
//        (c) an UNSTATUSED NC mon switching out.
//       If (a)==(b)==(c) then the cure (and the '[silent]' -curestatus reveal) is DRAW-FREE
//       ⇒ admitting NC is seed-neutral for the pre-existing suites.
//   D2. WHICH STATUSES are cured: brn / psn / tox / par / slp / frz — dump each mon's
//       post-switch-out (bench) status. Confirm the tox STAGE + sleep COUNTER reset.
//   D3. VOLUNTARY switch cures (the base case).
//   D4. PHAZE-DRAG-OUT cures: a foe Roars the statused NC mon OUT — is the dragged-out
//       (now-bench) mon cured? (runEvent SwitchOut fires on isDrag too.)
//   D5. FAINT is a no-op: a fainted NC mon has `status==='fnt'`/no status → the onSwitchOut
//       early-returns; nothing to cure. (Confirm the guard by KO'ing a statused NC mon.)
//
// Run:  node src/rust_sim/harness/probe_naturalcure_rng.js
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

// Generic runner. `inject` sets status/hp on either active before the plan runs.
// Wraps rng.next to count raw draws per plan step. Returns per-step draw counts + a
// state-dump callback hook.
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
    if (inj.tox !== undefined && m.statusState) m.statusState.stage = inj.tox;
    if (inj.hp !== undefined) m.hp = inj.hp;
  }

  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = (...a) => { drawCount += 1; return realNext(...a); };
  const steps = [];

  for (const [i, step] of plan.entries()) {
    const before = drawCount;
    if (step.p1) streams.omniscient.write(`>p1 ${step.p1}`);
    if (step.p2) streams.omniscient.write(`>p2 ${step.p2}`);
    for (let k = 0; k < 12; k++) await tick();
    steps.push({ i, draws: drawCount - before, note: step.note || '' });
  }
  return { battle, steps, label };
}

// Dump every mon (both sides) species + status + statusState detail.
function dumpTeams(battle, tag) {
  for (const side of battle.sides) {
    for (const p of side.pokemon) {
      const ss = p.statusState || {};
      const extra = [];
      if (ss.stage !== undefined) extra.push(`toxstage=${ss.stage}`);
      if (ss.time !== undefined) extra.push(`slptime=${ss.time}`);
      if (ss.startTime !== undefined) extra.push(`slpstart=${ss.startTime}`);
      console.log(`   [${tag}] ${side.id} ${p.species.name.padEnd(12)} status='${p.status || ''}' hp=${p.hp}/${p.maxhp} active=${p.isActive} ${extra.join(' ')}`);
    }
  }
}

(async () => {
  // ---------------------------------------------------------------------------
  // D1 + D2 + D3: VOLUNTARY switch-out draw count + which-status cure, NC vs non-NC.
  // Board: a slow p2 mon just sits (Splash-like: use a legal do-nothing — Recover into
  // full HP is fine, or a switch). We make p1's active a statused mon and SWITCH it out.
  // To isolate the switch-out draws with NOTHING else drawing, p2 also switches (a
  // double-switch turn draws ONLY the switch machinery, no move/accuracy/damage).
  // ---------------------------------------------------------------------------
  console.log('=== D1/D2/D3: voluntary switch-out — draw count + status cure (NC vs non-NC) ===');
  const STATUSES = ['brn', 'par', 'psn', 'tox', 'slp', 'frz'];
  for (const st of STATUSES) {
    // p1: [statused active Blissey (NC), bench Skarmory]; p2: [Snorlax, bench Gengar].
    // Both sides double-switch on the SAME turn (draw-isolated to switch machinery).
    const ncTeam = [mon('Blissey', ['softboiled', 'seismictoss'], { ability: 'Natural Cure' }),
      mon('Skarmory', ['spikes', 'roar'], { ability: 'Keen Eye' })];
    const nonNcTeam = [mon('Blissey', ['softboiled', 'seismictoss'], { ability: 'Serene Grace' }),
      mon('Skarmory', ['spikes', 'roar'], { ability: 'Keen Eye' })];
    const p2Team = [mon('Snorlax', ['bodyslam', 'rest'], { ability: 'Immunity' }),
      mon('Gengar', ['shadowball', 'nightshade'], { ability: 'Levitate' })];
    const tox = st === 'tox' ? 5 : undefined; // ramp tox to a high stage to prove reset

    const a = await run(`NC-${st}`, ncTeam, p2Team,
      [{ p1: 'switch 2', p2: 'switch 2', note: `NC ${st} switches out` }],
      { inject: [{ side: 0, status: st, tox }], seed: [1, 2, 3, 4] });
    const b = await run(`nonNC-${st}`, nonNcTeam, p2Team,
      [{ p1: 'switch 2', p2: 'switch 2', note: `non-NC ${st} switches out` }],
      { inject: [{ side: 0, status: st, tox }], seed: [1, 2, 3, 4] });
    // The switched-out mon is now on the bench (index 1 after the swap; find by species).
    const findBliss = (bt) => bt.sides[0].pokemon.find((p) => p.species.name === 'Blissey');
    const ncBliss = findBliss(a.battle);
    const nonNcBliss = findBliss(b.battle);
    const ncStage = ncBliss.statusState ? ncBliss.statusState.stage : undefined;
    console.log(`  ${st.toUpperCase()}: NC draws=${a.steps[0].draws} -> Blissey status='${ncBliss.status || ''}'${ncStage !== undefined ? ` toxstage=${ncStage}` : ''}   |   nonNC draws=${b.steps[0].draws} -> Blissey status='${nonNcBliss.status || ''}'   [seed-neutral: ${a.steps[0].draws === b.steps[0].draws ? 'YES' : 'NO'}]`);
  }

  // D1c: UNSTATUSED NC switch-out draw count (must equal the non-NC / statused counts).
  {
    const ncTeam = [mon('Blissey', ['softboiled'], { ability: 'Natural Cure' }),
      mon('Skarmory', ['spikes'], { ability: 'Keen Eye' })];
    const p2Team = [mon('Snorlax', ['bodyslam'], { ability: 'Immunity' }),
      mon('Gengar', ['shadowball'], { ability: 'Levitate' })];
    const c = await run('NC-unstatused', ncTeam, p2Team,
      [{ p1: 'switch 2', p2: 'switch 2', note: 'unstatused NC switches out' }], { seed: [1, 2, 3, 4] });
    console.log(`  UNSTATUSED: NC draws=${c.steps[0].draws} (should equal the statused NC/non-NC draw count)`);
  }

  // ---------------------------------------------------------------------------
  // D4: PHAZE-DRAG-OUT cure. p2 Roars the statused NC Blissey OUT. Does the dragged-out
  // (now-bench) mon get cured? Blissey must be SLOWER so Roar (prio -6) drags it; give
  // p1 a bench so the drag has a target.
  // ---------------------------------------------------------------------------
  console.log('\n=== D4: PHAZE-DRAG-OUT — a Roar drags the statused NC mon out, is it cured? ===');
  for (const st of ['brn', 'tox', 'par']) {
    const ncTeam = [mon('Blissey', ['softboiled', 'seismictoss'], { ability: 'Natural Cure', evs: { spe: 0 } }),
      mon('Skarmory', ['spikes'], { ability: 'Keen Eye' })];
    const p2Team = [mon('Aerodactyl', ['roar', 'earthquake'], { ability: 'Pressure' }), // fast, Roars
      mon('Snorlax', ['bodyslam'], { ability: 'Immunity' })];
    const tox = st === 'tox' ? 4 : undefined;
    const d = await run(`drag-${st}`, ncTeam, p2Team,
      [{ p1: 'move 1', p2: 'move 1', note: `p2 Aerodactyl Roars the ${st} NC Blissey out` }],
      { inject: [{ side: 0, status: st, tox }], seed: [5, 6, 7, 8] });
    const bliss = d.battle.sides[0].pokemon.find((p) => p.species.name === 'Blissey');
    console.log(`  ${st.toUpperCase()} drag: Blissey active=${bliss.isActive} status='${bliss.status || ''}' draws=${d.steps[0].draws}  [dragged-out ${bliss.isActive ? 'NO (Roar failed?)' : 'YES'} -> cured: ${!bliss.status ? 'YES' : 'NO'}]`);
    dumpTeams(d.battle, `drag-${st}`);
  }

  // ---------------------------------------------------------------------------
  // D5: FAINT is a no-op. KO a statused NC mon — onSwitchOut early-returns (status 'fnt'
  // or cleared on faint). The forced replacement is a normal switch; nothing to cure.
  // p2 Earthquakes the low-HP statused NC Blissey to death; a replacement comes in.
  // ---------------------------------------------------------------------------
  console.log('\n=== D5: FAINT is a no-op (a fainted NC mon has nothing to cure) ===');
  {
    const ncTeam = [mon('Blissey', ['softboiled'], { ability: 'Natural Cure' }),
      mon('Skarmory', ['spikes'], { ability: 'Keen Eye' })];
    const p2Team = [mon('Tyranitar', ['earthquake'], { ability: 'Sand Stream' }),
      mon('Snorlax', ['bodyslam'], { ability: 'Immunity' })];
    const e = await run('faint-nc', ncTeam, p2Team,
      [{ p1: 'move 1', p2: 'move 1', note: 'p2 EQs the 1-HP burned NC Blissey to death' },
        { p1: 'switch 2', note: 'p1 sends in Skarmory (forced replacement)' }],
      { inject: [{ side: 0, status: 'brn', hp: 1 }], seed: [9, 10, 11, 12] });
    const bliss = e.battle.sides[0].pokemon.find((p) => p.species.name === 'Blissey');
    console.log(`  FAINT: Blissey fainted=${bliss.fainted} status='${bliss.status || ''}' (a fainted NC mon's status is '' — no cure needed) draws=[${e.steps.map((s) => s.draws).join(',')}]`);
  }

  console.log('\n=== SUMMARY ===');
  console.log('  Trigger: onSwitchOut (onCheckShow undefined). Fires on voluntary + drag (isDrag).');
  console.log('  Timing:  runEvent("SwitchOut") BEFORE clearVolatile, on an alive outgoing mon.');
  console.log('  Draw model: SEE the D1 [seed-neutral] column (expected DRAW-FREE — no tie-shuffle).');
  console.log('  Statuses: SEE D2 (expected all of brn/par/psn/tox/slp/frz; tox stage + slp counter reset).');
})();
