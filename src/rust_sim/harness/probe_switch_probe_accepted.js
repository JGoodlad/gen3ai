// probe_switch_probe_accepted.js — the DETERMINISTIC regression gate for
// `gen3_simbridge_probe_accepted_v1` (the differ's switch-probe contract).
//
// THE BUG IT PINS. `gen_sim_bridge_diff.js` probes a `trapped`/`maybeTrapped` request with a
// switch, and used to assume that probe is ALWAYS rejected: it drained with
// `requireRequest:false` and then sent a SECOND, "real" choice for the same side. But the sim
// sets `maybeTrapped` for a SPECULATIVE maybe too — `battle.ts:1730` runs `FoeMaybeTrapPokemon`
// for every ability the foe's SPECIES could have — and in that case the mon is NOT trapped, so
// `Side.chooseSwitch` (`sim/side.ts:981`) ACCEPTS the switch and the sim emits NOTHING. Then:
//   * the drain burned its FULL 30 s cap waiting for output that cannot come (x up to
//     SAFETY=600 decisions in one battle — the same stall shape ROUND 27 fixed on the other
//     two drain call sites, left unfixed on this one), and
//   * the follow-up choice was refused with `[Invalid choice] ... sent more choices than
//     unfainted Pokemon` and NO re-request, while `cmds` recorded a choice that never
//     committed — so the saved repro no longer replays the battle.
//
// THE FIXED MATCHUP. p2 is a **Sand Veil** Dugtrio: the species CAN have Arena Trap, so the sim
// flags p1 `maybeTrapped:true` defensively, but p1 is free to switch. This is **gen3ou** on
// purpose — `battle.ts:1742` skips the speculative loop in `gen3customgame` (no
// `obtainableabilities`), which is why every default-format soak missed this.
//
// WHAT FAILS ON A REVERT (MEASURED): restoring the old probe block (`if (false)` on the
// accepted branch + the 30 s drain cap) takes this battle from **11.5 s to 181 s** — assertion
// 1 fails, exit 1. Assertion 2 did NOT fire in that revert (the extra choice happened to land
// on the NEXT boundary and was legal, silently skewing `cmds` instead of erroring); it is kept
// as a guard for the loud variant of the same corruption, but assertion 1 is the load-bearing
// one.
//
// HERMETIC: both children are the NODE bridge (the differ is agnostic about what the second
// child is), so this gate needs no `cargo build` and pins the HARNESS contract, not the port.
//
// USAGE: node src/rust_sim/harness/probe_switch_probe_accepted.js
'use strict';

const path = require('path');
const { BridgeChild, diffOneBattle, NODE_BRIDGE } = require('./gen_sim_bridge_diff.js');
const ROOT = path.resolve(__dirname, '../../..');
const { Teams } = require(path.join(ROOT, 'deps/pokemon-showdown/dist/sim'));

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
const set = (species, moves, ability) => ({
  name: species, species, item: 'Leftovers', ability, moves,
  evs: EV0, ivs: IV31, nature: 'Serious', level: 100, gender: 'M',
});

// A probe costs one full drain cap on the old code (30 s, or SBD_DRAIN_MAX_MS). The fixed path
// concludes "accepted" from the CONTENT of an empty drain inside PROBE_MAX_MS (750 ms), and
// this battle issues several probes — measured 7.0 s fixed vs 181 s reverted, so the bound
// separates the two by more than an order of magnitude even on a loaded box.
const WALL_BUDGET_MS = Number(process.env.PROBE_WALL_BUDGET_MS || 25000);

async function main() {
  const p1 = Teams.pack([
    set('Snorlax', ['bodyslam', 'rest'], 'Immunity'),
    set('Blissey', ['softboiled', 'seismictoss'], 'Natural Cure'),
  ]);
  const p2 = Teams.pack([
    set('Dugtrio', ['earthquake', 'rockslide'], 'Sand Veil'),   // NOT Arena Trap
    set('Skarmory', ['drillpeck', 'spikes'], 'Keen Eye'),
  ]);
  const startJson = JSON.stringify({
    formatid: 'gen3ou', seed: [1, 2, 3, 4],
    p1: { name: 'P1', team: p1 }, p2: { name: 'P2', team: p2 },
  });

  const a = new BridgeChild('node', [NODE_BRIDGE]);
  const b = new BridgeChild('node', [NODE_BRIDGE]);
  let result, elapsed;
  try {
    const t0 = Date.now();
    // trapProb 1.0 => probe EVERY maybeTrapped request; the chooseSeed is fixed.
    result = await diffOneBattle(a, b, startJson, 0xC0FFEE, 1.0, false);
    elapsed = Date.now() - t0;
  } finally { a.kill(); b.kill(); }

  const flat = [];
  for (const side of ['p1', 'p2']) for (const ch of ((result.nodeSide || {})[side] || [])) flat.push(...ch);
  const sawMaybeTrapped = flat.some((l) => l.includes('"maybeTrapped":true'));
  const overChoice = flat.filter((l) => l.includes('sent more choices'));

  let fail = 0;
  const check = (label, ok, detail) => {
    if (ok) console.error(`  ok   ${label}`);
    else { fail++; console.error(`  FAIL ${label}${detail ? ` — ${detail}` : ''}`); }
  };

  console.error('[probe_switch_probe_accepted] gen3ou / Sand Veil Dugtrio (speculative maybeTrapped)');
  // 0 — NOT VACUOUS: the case must actually be reached, or the other checks prove nothing.
  check('the speculative maybeTrapped request was reached', sawMaybeTrapped,
    'no |request| carried "maybeTrapped":true — the matchup no longer exercises the path');
  // 1 — the STALL.
  check(`the battle ran within its wall-clock bound (${elapsed} ms < ${WALL_BUDGET_MS} ms)`,
    elapsed < WALL_BUDGET_MS, 'an accepted probe is burning a full drain cap again');
  // 2 — the CORRUPTION: a second choice for an already-committed side.
  check('no choice was sent for an already-committed side', overChoice.length === 0,
    JSON.stringify(overChoice.slice(0, 2)));
  // 3 — the battle still plays out (the fix must not strand a boundary).
  check('the battle reached an end or a clean error', !!(result.ended || result.err),
    JSON.stringify(result.err || '(neither ended nor errored)'));
  if (result.err) console.error(`  note: battle error = ${result.err}`);

  console.error(fail
    ? `\n[probe_switch_probe_accepted] ${fail} FAILURE(S) — gen3_simbridge_probe_accepted_v1 REGRESSED`
    : '\n[probe_switch_probe_accepted] PASS — the accepted switch probe is handled as a committed choice');
  process.exit(fail ? 1 : 0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
