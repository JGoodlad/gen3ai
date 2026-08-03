// gen_sim_bridge_diff.js — the STDIN/STDOUT DROP-IN A/B DIFFERENTIAL for `sim_bridge`.
//
// Spawns BOTH the real Node bridge (`src/utils/bridge/local_sim_bridge.js`) AND the Rust
// `sim_bridge` binary as SUBPROCESSES, feeds them the IDENTICAL stdin command stream over
// random teams, and asserts BYTE-IDENTICAL stdout: the per-side `pN <base64(chunk)>` frames
// (decoded + `|t:|` normalized + `|debug|`/blank dropped) in per-side order, plus `__END__`.
// This is the drop-in gate: if the Rust binary's per-side chunk stream matches the Node
// bridge's for the same commands, `node local_sim_bridge.js` can be swapped for `sim_bridge`
// behind the Python bridge with zero protocol change.
//
// The command stream is DRIVEN BY READING THE NODE BRIDGE's `|request|` frames off its stdout
// (the real drop-in consumer's loop): at each move / forceSwitch request we pick a random
// LEGAL + MODELED choice from a seeded choice-RNG, and — with `--trap-prob` — issue a REJECTED
// switch probe when the active is trapped (the maybeTrapped→trapped machine). The SAME command
// stream is teed to the Rust binary. Team generation REUSES bridge_ab_fuzz.js/ab_fuzz.js.
//
// `__RECON__` is EXCLUDED from the diff (the drop-in binary defers it — the Node bridge dumps
// the sim's internal inputLog, a separate concern; the Python side degrades gracefully when no
// `__RECON__` arrives). This is stated in the run banner + the per-repro note.
//
// CROSS-SIDE INTERLEAVING is NOT asserted (only PER-SIDE chunk sequences): the p1-vs-p2
// ordering on the wire is a Node microtask-scheduler artifact the Python demux (which routes
// each `pN` line to its own client) does not depend on. The invariant that matters for the
// drop-in is that EACH side's chunk sequence is byte-identical — which is what this asserts.
//
// USAGE
//   node src/rust_sim/harness/gen_sim_bridge_diff.js
//        [--mode trapping|randbats|random|pool]   (default randbats)
//        [--format gen3customgame|gen3ou]         (default gen3customgame)
//        [--battles N]                            (default 150)
//        [--master-seed S]                        (default from time; ALWAYS printed)
//        [--persistent]                           (reuse ONE child pair across all battles — the
//                                                   persistent-mode multi-battle-per-child path)
//        [--trap-prob P]                          (default 0.5)
//        [--out DIR]                              (default harness/sim_bridge_diff_out/)
//        [--verbose]
//        [--selftest]                             (allowlist gate-integrity assertions, then exit)
//
// THE GREEN GATE (`gen3_simbridge_diff_allowlist_v1`). Exit 0 = no NON-allowlisted
// divergence; 1 = a real divergence (repro saved under `<out>/divergences/`). A documented
// request-DISPLAY residual is counted SEPARATELY, filed under `<out>/allowlisted/`, and does
// NOT fail the run — without this the gate is unusable (a 10-battle randbats smoke reported
// 8/10 "diverged", ALL of them the same `return102` alias). The allowlist RECONCILES then
// compares the FULL lines, so any residual byte difference still FAILS; `--selftest` pins
// that property (10 cases, incl. the load-bearing negatives: a different move, an alias PLUS
// a pp difference, and a LEVEL/GENDER *value* divergence).
//
// ⚠️ WHY THIS HARNESS IS THE STRONGEST GATE WE HAVE — it is the EXTERNAL-CONSISTENCY test.
// `ab_replay` replays a FIXED recorded decision list, so it must assume both engines segment
// decisions identically; when they don't it reports a `kind=seed` artifact and cannot tell
// segmentation from a real legality bug. This harness DISCOVERS boundaries live (it reads a
// request, then chooses), so that artifact CANNOT occur by construction, and a genuine extra
// request shows up as an unambiguous request-frame mismatch. It also compares what poke-env
// ACTUALLY consumes — the per-side stream AND the `|request|` JSON (requests are per-side
// frames, so the byte diff covers them) — rather than the omniscient log poke-env never sees.
//
// THROUGHPUT: ~600 battles/hr with `--persistent`, ~80/hr without (each battle otherwise
// respawns a Node child that reloads the whole Showdown dist). ~96% of wall time is the
// per-write quiescence settle, not CPU (`user` ≈ 2s per 60s wall) — so ALWAYS use
// `--persistent` for a soak, and the settle is the lever if more speed is needed.

'use strict';

const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const e2e = require('./gen_e2e_fuzz.js');
const { isModeledMove, mulberry32, randInt, seedFrom, toId, dex3 } = e2e;
const ab = require('./ab_fuzz.js');
// NOTE: bridge_ab_fuzz.js has NO require.main guard (requiring it runs its fuzzer), so the
// trapping matchup provider is inlined below rather than imported.

const ROOT = path.resolve(__dirname, '../../..');
const NODE_BRIDGE = path.join(ROOT, 'src/utils/bridge/local_sim_bridge.js');
const CRATE = path.resolve(__dirname, '..');
// ISOLATED target dir — never rebuild the shared target/ (the live ab_replay fuzzer).
// (overridable so two concurrent investigations never share one target dir / build lock)
const SIMBRIDGE_TARGET = process.env.POKESIM_SIMBRIDGE_TARGET || '/tmp/pokesim_target_simbridge';
const RUST_BRIDGE = path.join(SIMBRIDGE_TARGET, 'release/sim_bridge');

const { Teams } = require(path.join(ROOT, 'deps/pokemon-showdown/dist/sim'));

// ── Flags ─────────────────────────────────────────────────────────────────────
function parseFlags(argv) {
  const f = {
    mode: 'randbats', format: 'gen3customgame', battles: 150, masterSeed: null,
    persistent: false, trapProb: 0.5, out: path.join(__dirname, 'sim_bridge_diff_out'),
    verbose: false, extraTrappers: false,
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i]; const next = () => argv[++i];
    if (a === '--mode') f.mode = next();
    else if (a === '--format') f.format = next();
    else if (a === '--battles') f.battles = Number(next());
    else if (a === '--master-seed') f.masterSeed = Number(next()) >>> 0;
    else if (a === '--persistent') f.persistent = true;
    else if (a === '--trap-prob') f.trapProb = Number(next());
    else if (a === '--out') f.out = path.resolve(next());
    else if (a === '--repro') f.repro = path.resolve(next()); // replay ONE saved repro dir
    else if (a === '--verbose') f.verbose = true;
    else if (a === '--extra-trappers') f.extraTrappers = true; // Magnet Pull + Shadow Tag (seed-gap)
    else if (a === '--selftest') f.selftest = true;   // allowlist gate-integrity assertions
    else { console.error(`unknown flag ${a}`); process.exit(2); }
  }
  if (!['trapping', 'randbats', 'random', 'pool'].includes(f.mode)) {
    console.error(`--mode must be trapping|randbats|random|pool`); process.exit(2);
  }
  if (!['gen3customgame', 'gen3ou'].includes(f.format)) {
    console.error(`--format must be gen3customgame|gen3ou`); process.exit(2);
  }
  if (f.masterSeed === null) f.masterSeed = (Date.now() ^ (process.pid * 2654435761)) >>> 0;
  return f;
}

// ── A line-buffered subprocess wrapper for a bridge child ───────────────────────
class BridgeChild {
  constructor(cmd, args) {
    this.proc = spawn(cmd, args, { stdio: ['pipe', 'pipe', 'pipe'] });
    this.buf = '';
    this.queue = [];       // undelivered decoded stdout events (drained by drainQuiescent)
    this.exited = false;
    this.stderr = '';
    this.proc.stdout.on('data', (d) => this._onData(d));
    this.proc.stderr.on('data', (d) => { this.stderr += d.toString(); });
    this.proc.on('exit', () => { this.exited = true; this.queue.push({ type: 'exit' }); });
    // A dead child's stdin write raises EPIPE — swallow it (we kill children eagerly).
    this.proc.stdin.on('error', () => {});
    this.proc.on('error', () => {});
  }
  _onData(d) {
    this.buf += d.toString();
    let nl;
    while ((nl = this.buf.indexOf('\n')) !== -1) {
      const line = this.buf.slice(0, nl); this.buf = this.buf.slice(nl + 1);
      this.queue.push(this._decode(line));
    }
  }
  _decode(line) {
    if (line.startsWith('p1 ') || line.startsWith('p2 ')) {
      const side = line.slice(0, 2);
      const chunk = Buffer.from(line.slice(3), 'base64').toString('utf8');
      return { type: 'chunk', side, chunk };
    }
    if (line === '__END__') return { type: 'end' };
    if (line.startsWith('__ERR__')) {
      return { type: 'err', msg: Buffer.from(line.slice('__ERR__ '.length), 'base64').toString('utf8') };
    }
    if (line.startsWith('__RECON__')) return { type: 'recon' };
    return { type: 'other', line };
  }
  write(s) { try { if (!this.exited && this.proc.stdin.writable) this.proc.stdin.write(s + '\n'); } catch (e) {} }
  kill() {
    try { if (!this.exited && this.proc.stdin.writable) this.proc.stdin.write('END\n'); } catch (e) {}
    try { this.proc.kill(); } catch (e) {}
  }
}

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

// Collect the child's stdout events for ONE write until the bridge is WAITING FOR INPUT
// again — the natural terminal signal is a `|request|` frame (every boundary ends with the
// per-side requests) OR `__END__`/`__ERR__`/exit. This is deterministic (no timing race):
// after START or a completed CHOOSE, the bridge always flushes its batch(es) then the
// request(s); after a NON-completing CHOOSE (one side of a move boundary, or a trapped
// reject) it flushes only the error/re-request/nothing, so a short quiescence settle after
// the pull is the fallback. We pull until: (a) a request frame appeared AND a settle
// elapses with nothing new, or (b) end/err/exit.
async function drainQuiescent(child, opts = {}) {
  const requireRequest = opts.requireRequest !== false; // default: wait for a request frame
  const pollMs = opts.pollMs || 5;      // queue poll granularity
  const settleMs = opts.settleMs || 40; // quiet window after a terminal signal
  const maxMs = opts.maxMs || Number(process.env.SBD_DRAIN_MAX_MS || 30000); // hard cap (a slow Node child flushes in ~0.5s)
  const events = [];
  let sawRequest = false;
  let sawTerminal = false;
  const start = Date.now();
  // Poll the queue (no leaked next() waiters). Break once the terminal signal appeared AND
  // a quiet settle elapsed (so trailing chunks after the request are collected too).
  let quietSince = null;
  for (;;) {
    let drained = false;
    while (child.queue.length) {
      const ev = child.queue.shift();
      events.push(ev);
      drained = true;
      if (ev.type === 'chunk' && ev.chunk.includes('|request|')) sawRequest = true;
      if (ev.type === 'end' || ev.type === 'err' || ev.type === 'exit') sawTerminal = true;
    }
    if (sawTerminal) break;
    if (drained) { quietSince = null; }
    // Terminal-quiescent: the bridge is now waiting for input.
    //  - completing write / START: a `|request|` frame appeared.
    //  - reject drain (`requireRequest:false`): a firm trap flushes ONLY `|error|` (no
    //    request) → break once we've collected something.
    const ready = requireRequest ? sawRequest : events.length > 0;
    if (ready) {
      if (quietSince === null) quietSince = Date.now();
      if (Date.now() - quietSince >= settleMs) break;
    }
    if (Date.now() - start > maxMs) { events.timedOut = true; break; }
    await sleep(pollMs);
  }
  events.waitedMs = Date.now() - start;
  // `silenceOk`: this call site treats "nothing arrived" as a legitimate ANSWER (the accepted
  // switch probe), so its cap is not a stall and must not pollute the `drain_timeouts` canary.
  // A child that genuinely died there still surfaces — the boundary is then incomplete and the
  // NEXT (checked) drain reports it.
  if (events.timedOut && !opts.silenceOk) {
    DRAIN_TIMEOUTS.push({ tag: opts.tag || '?', requireRequest, events: events.length });
    if (process.env.SBD_TRACE_DRAINS) {
      console.error(`[drain-timeout] tag=${opts.tag || '?'} requireRequest=${requireRequest} events=${events.length}`);
    }
  }
  return events;
}
// Every drain that hit its cap this run, tagged by call site — so a stall shows up in the
// SUMMARY even on the paths that (legitimately) do not abort the battle.
const DRAIN_TIMEOUTS = [];

// A drain that hit its `maxMs` cap without the terminal signal means a child stopped
// talking mid-battle. Treat it as a HARD ERROR, not a shrug (`gen3_simbridge_drain_timeout_v1`).
//
// WHY THIS MATTERS — it cost a whole soak. `drainQuiescent` used to time out SILENTLY, return
// whatever it had, and let the loop continue; `diffOneBattle` allows SAFETY=600 iterations. So a
// single boundary that never emits a `|request|` frame burns 30 s and the battle keeps trying —
// up to 600 x 30 s = ~5 HOURS on ONE battle. The 1200-battle randbats soak parked on battle 59
// for 2 h looking exactly like a deadlock: driver alive at ~2% CPU (the 5 ms poll), both child
// bridges idle in `ep_poll`/`unix_stream_data_wait` having said everything they were going to.
// Diagnosing that cost more than the bugs the soak found. A stalled child is now reported
// immediately, with the side and phase, so it becomes a FINDING instead of an invisible hang.
function assertDrained(events, node, rust, where, ctx) {
  if (!events || !events.timedOut) return null;
  // Dump the CONTEXT a stall needs to be diagnosed without a re-run: the last commands we
  // wrote and the tail of what each side last emitted. Without this the error names the
  // decision but not the SHAPE, and you have to reproduce to learn anything.
  const tail = (perSide, n) => {
    if (!perSide) return '(none)';
    const flat = [];
    for (const side of ['p1', 'p2']) for (const ch of (perSide[side] || [])) flat.push(...ch);
    return flat.slice(-n).join(' ¦ ') || '(none)';
  };
  const lastCmds = ctx && ctx.cmds ? ctx.cmds.slice(-6).map((c) => c.join(' ')).join(' ¦ ') : '(none)';
  return `drain timeout after 30s at ${where} — a child stopped emitting (no |request| frame).\n`
    + `    last cmds: ${lastCmds}\n`
    + `    node tail: ${tail(ctx && ctx.nodeSide, 6)}\n`
    + `    rust tail: ${tail(ctx && ctx.rustSide, 6)}\n`
    + `    node stderr: ${(node.stderr || '').slice(-200) || '(none)'} | `
    + `rust stderr: ${(rust.stderr || '').slice(-200) || '(none)'}`;
}

// ── THE SWITCH-PROBE CONTRACT (`gen3_simbridge_probe_accepted_v1`) ──────────────
// The differ probes a `trapped`/`maybeTrapped` request with a switch to exercise the
// maybeTrapped -> trapped machine, and USED to assume that probe is always REJECTED — it
// drained with `requireRequest:false` (break on the first event) and then sent a SECOND,
// "real" choice for the same side.
//
// That assumption is FALSE. `sim/pokemon.ts::getRequestData` emits `maybeTrapped` in TWO
// distinct situations (it is set only when `trapped !== true`):
//   (a) a HIDDEN trap — `tryTrap(true)` set `trapped = 'hidden'`; `Side.chooseSwitch`
//       (`sim/side.ts:968`) REJECTS it -> `|error|` + a re-request.  <- the assumed case
//   (b) a SPECULATIVE maybe — `battle.ts:1730` runs `FoeMaybeTrapPokemon` for every ability
//       the foe's SPECIES could have (so cancelling can't leak whether it traps). The mon is
//       NOT trapped, so the switch is ACCEPTED (`side.ts:981` only sets `cantUndo`) and the
//       sim emits NOTHING.
// PROBE-SETTLED by `probe_maybetrapped_probe_accepted.js`: a Sand Veil Dugtrio (species CAN
// have Arena Trap) flags the foe `maybeTrapped:true` in **gen3ou** and the probe is accepted
// silently; in **gen3customgame** the speculative loop is skipped (`battle.ts:1742`, no
// `obtainableabilities`), which is why the default-format soaks never hit it.
//
// Two consequences, both fixed here:
//   1. THE STALL — a drain waiting for output that will never come burned its FULL 30 s cap,
//      up to `SAFETY`=600 times in one battle. Same shape as the drain-timeout bug ROUND 27
//      fixed on the OTHER two call sites; this call site was left unchecked.
//   2. THE CORRUPTION — the accepted switch is the side's COMMITTED choice, so the follow-up
//      choice is refused (`[Invalid choice] ... sent more choices than unfainted Pokemon`, NO
//      re-request) and `cmds` records a choice that never committed => the saved repro no
//      longer replays the battle.
// The discriminator is CONTENT, not timing: a rejected probe always answers with `|error|`.
// `PROBE_MAX_MS` only bounds how long we wait to conclude "nothing is coming" — the sim
// answers a refusal synchronously inside its `write()` (~1 ms; a few ms through the child's
// pipe), so 750 ms is ~100x margin, and it is paid on EVERY accepted probe (the measured
// regression gate spends ~7 of them per toy battle). A false "accepted" would leave the
// boundary incomplete, which the NEXT drain reports as a stall with a repro — loud, not
// silent — so the tight bound is safe.
const PROBE_MAX_MS = Number(process.env.SBD_PROBE_MAX_MS || 750);
function probeWasAccepted(events) {
  for (const ev of events || []) {
    if (ev.type === 'chunk' && ev.chunk.includes('|error|')) return false;
  }
  return true;
}
let probesAccepted = 0;

// A whole-battle WALL-CLOCK budget (`gen3_simbridge_battle_budget_v1`) — the BELT-AND-BRACES
// the ROUND-27 note asked for, on top of the root-caused fixes: `drainQuiescent`'s `maxMs`
// caps a SINGLE drain, never the outer loop, so `SAFETY`=600 decisions x a capped drain is
// still hours on ONE battle. This turns any residual no-progress shape into a REPORTED error
// (with a repro) in minutes instead of an invisible park. It is deliberately NOT a substitute
// for a diagnosis: every stall it catches still prints its side/phase and saves a repro.
const BATTLE_BUDGET_MS = Number(process.env.SBD_BATTLE_BUDGET_MS || 300000);

// ── One battle: drive BOTH children off the Node bridge's request frames ────────
// Returns { nodePerSide, rustPerSide, cmds, ended, err } where *PerSide is
// { p1: [chunkLines...], p2: [...] } (each chunk a normalized line array).
async function diffOneBattle(node, rust, startJson, chooseSeed, trapProb, verbose) {
  // Per-side collected chunks (normalized), for BOTH children.
  const nodeSide = { p1: [], p2: [] };
  const rustSide = { p1: [], p2: [] };
  const cmds = [];
  let ended = false;

  // Feed START to both. START flushes the framing + the initial request(s) → drain-to-request.
  node.write('START ' + startJson);
  rust.write('START ' + startJson);
  {
    const en = await drainQuiescent(node, { tag: 'start-node' }); collect(en, nodeSide);
    const er = await drainQuiescent(rust, { tag: 'start-rust' }); collect(er, rustSide);
    const ctx = { cmds, nodeSide, rustSide };
    const stall = assertDrained(en, node, rust, 'START (node)', ctx)
               || assertDrained(er, node, rust, 'START (rust)', ctx);
    if (stall) return { err: stall, cmds, nodeSide, rustSide };
  }

  // A SPEED-TIED lead matchup is now VALIDATED (was skipped). The sim's turn-0 CONSTRUCTION
  // window — the `start` action's insertChoice tie-break + the eachEvent('Update'/'WeatherChange')
  // shuffles + Magnet Pull's onAny trap shuffles when the two leads share Spe — is modeled
  // bit-for-bit by `gen3_turn0_construction_v1` (`BridgeSession::new_construct_turn0` runs the
  // real turn-0 through the driver machinery from the RAW seed). So a speed-TIED lead no longer
  // desyncs the turn-1 first-mover, and the per-mon gender `sample(['M','F'])` is modeled too
  // (unspecified-gender teams are byte-correct). Gated by `turn0_construction_test.rs` vs the
  // omniscient sim; this harness re-proves it end-to-end on the LIVE `sim_bridge` binary.

  const rng = mulberry32(chooseSeed);
  const SAFETY = 600;
  let safety = 0;
  const deadline = Date.now() + BATTLE_BUDGET_MS;
  // Drive choices by reading the NODE bridge's LATEST per-side request frames.
  while (!ended && safety < SAFETY) {
    safety++;
    if (Date.now() > deadline) {
      return { err: `battle wall-clock budget exceeded (${BATTLE_BUDGET_MS} ms) at decision `
        + `${safety} — no progress to an end (gen3_simbridge_battle_budget_v1)`,
        cmds, nodeSide, rustSide };
    }
    // True when BOTH sides' choices were committed by an ACCEPTED switch probe, so the
    // boundary already flushed during the probe drains and no further write is coming.
    let probeCompletedBoundary = false;
    const reqs = latestRequests(nodeSide);
    const decideMove = reqs.p1 && reqs.p1.active && reqs.p2 && reqs.p2.active
      && !isWait(reqs.p1) && !isWait(reqs.p2);
    const forceP1 = reqs.p1 && reqs.p1.forceSwitch;
    const forceP2 = reqs.p2 && reqs.p2.forceSwitch;

    if (forceP1 || forceP2) {
      // Forced replacement(s): each flagged side writes; the LAST completing write flushes
      // the switch batch + the next request(s).
      const sides = [];
      for (const [side, req] of [['p1', reqs.p1], ['p2', reqs.p2]]) {
        if (req && req.forceSwitch && req.forceSwitch[0]) {
          const c = pickReplacementReq(req, rng);
          if (!c) return { err: `no legal replacement for ${side}`, cmds };
          sides.push([side, c]);
        }
      }
      writeBoundary(node, rust, sides, cmds);
    } else if (decideMove) {
      // Move request: BOTH sides pick (+ optional trapped-switch reject probe first).
      const sides = [];
      for (const [side, req] of [['p1', reqs.p1], ['p2', reqs.p2]]) {
        const isTrapped = !!(req.active && req.active[0] && (req.active[0].trapped || req.active[0].maybeTrapped));
        const benchSlots = benchFromReq(req);
        if (isTrapped && benchSlots.length > 0 && rng() < trapProb) {
          // A switch probe against a `trapped`/`maybeTrapped` request. It is USUALLY rejected
          // (the |error| + re-request flush on THIS side) — but NOT always; see
          // `probeWasAccepted` (`gen3_simbridge_probe_accepted_v1`).
          const rejSlot = benchSlots[randInt(rng, benchSlots.length)];
          const rej = `switch ${rejSlot + 1}`;
          cmds.push([side, rej]);
          node.write(`CHOOSE ${side} ${rej}`);
          rust.write(`CHOOSE ${side} ${rej}`);
          const pn = await drainQuiescent(node, { requireRequest: false, maxMs: PROBE_MAX_MS, silenceOk: true, tag: 'probe-node' });
          const pr = await drainQuiescent(rust, { requireRequest: false, maxMs: PROBE_MAX_MS, silenceOk: true, tag: 'probe-rust' });
          collect(pn, nodeSide); collect(pr, rustSide);
          if (probeWasAccepted(pn)) {
            // ACCEPTED — the switch IS this side's committed choice. Sending a second choice
            // here is what the old code did: the sim refuses it with `[Invalid choice] ... sent
            // more choices than unfainted Pokémon` (NO re-request), and `cmds` then records a
            // choice that never committed, so the saved repro no longer replays the battle.
            probesAccepted++;
            continue;
          }
        }
        const c = pickModeledLegalReq(req, rng, isTrapped);
        if (!c) return { err: `no legal choice for ${side}`, cmds };
        sides.push([side, c]);
      }
      writeBoundary(node, rust, sides, cmds);
      probeCompletedBoundary = sides.length === 0;
    } else {
      break; // no move/switch boundary → battle ended (an __END__ was drained)
    }
    // The completing write flushed the turn/switch batch + the next request(s) → drain both.
    {
      const requireRequest = !probeCompletedBoundary;
      const en = await drainQuiescent(node, { tag: 'decision-node', requireRequest }); collect(en, nodeSide);
      const er = await drainQuiescent(rust, { tag: 'decision-rust', requireRequest }); collect(er, rustSide);
      const ctx = { cmds, nodeSide, rustSide };
      const stall = assertDrained(en, node, rust, `decision ${safety} (node)`, ctx)
                 || assertDrained(er, node, rust, `decision ${safety} (rust)`, ctx);
      if (stall) return { err: stall, cmds, nodeSide, rustSide };
    }
    if (nodeSide._ended || rustSide._ended) { ended = true; break; }
  }
  if (safety >= SAFETY) return { err: 'battle did not advance to end (safety cap)', cmds };

  return { nodeSide, rustSide, cmds, ended: nodeSide._ended || rustSide._ended };
}

// Write the choices that COMPLETE a boundary to both children (no drain between — only the
// final write flushes the batch; the caller drains after).
function writeBoundary(node, rust, sides, cmds) {
  for (const [side, choice] of sides) {
    cmds.push([side, choice]);
    node.write(`CHOOSE ${side} ${choice}`);
    rust.write(`CHOOSE ${side} ${choice}`);
  }
}

// Collect drained events into per-side normalized chunk lists (mutates `dst`).
function collect(events, dst) {
  for (const ev of events) {
    if (ev.type === 'chunk') {
      const lines = normalizeChunk(ev.chunk);
      if (lines.length) dst[ev.side].push(lines);
    } else if (ev.type === 'end') {
      dst._ended = true;
    } else if (ev.type === 'err') {
      dst._err = ev.msg;
    } else if (ev.type === 'recon') {
      // EXCLUDED from the diff (drop-in binary defers __RECON__).
      dst._recon = true;
    }
  }
}

// Normalize a chunk to a line array: `|t:|` → `|t:|<NORMALIZED>`, drop `|debug|`/blank
// (poke-env-ignored free-form sim lines the bridge emitter never produces).
function normalizeChunk(chunk) {
  const out = [];
  for (const raw of chunk.split('\n')) {
    if (raw.startsWith('|debug|') || raw === '') continue;
    out.push(raw.startsWith('|t:|') ? '|t:|<NORMALIZED>' : raw);
  }
  return out;
}

// The latest per-side request objects from the collected node chunks (scan backwards).
function latestRequests(nodeSide) {
  const out = { p1: null, p2: null };
  for (const side of ['p1', 'p2']) {
    outer:
    for (let ci = nodeSide[side].length - 1; ci >= 0; ci--) {
      const lines = nodeSide[side][ci];
      for (const l of lines) {
        if (l.startsWith('|request|')) {
          const payload = l.slice('|request|'.length);
          if (!payload || payload === 'null') continue;
          try { out[side] = JSON.parse(payload); } catch (e) { continue; }
          break outer;
        }
      }
    }
  }
  return out;
}
function isWait(req) { return !!(req && req.wait); }
function benchFromReq(req) {
  // Legal switch slots = non-active, non-fainted mons in the request's side.pokemon.
  const out = [];
  const mons = (req.side && req.side.pokemon) || [];
  for (let k = 0; k < mons.length; k++) {
    const m = mons[k];
    if (m.active) continue;
    if (typeof m.condition === 'string' && m.condition.endsWith(' fnt')) continue;
    out.push(k);
  }
  return out;
}
function pickModeledLegalReq(req, rng, isTrapped) {
  const moves = (req.active && req.active[0] && req.active[0].moves) || [];
  const modeledSlots = [];
  const legalSlots = [];   // legal but not necessarily MODELED — the safe last resort
  for (let k = 0; k < moves.length; k++) {
    // A slot is UNUSABLE if the sim marked it `disabled` OR it is out of PP. Checking only
    // `disabled` was the bug that killed a 1200-battle soak (`gen3_simbridge_picker_pp_v1`):
    // Showdown escalates a repeat bad choice from `[Unavailable choice]` (which RE-REQUESTS,
    // so the harness recovers) to `[Invalid choice]` (which does NOT), after which the drain
    // waits forever for a `|request|` that will never come. gen3customgame has NO turn cap and
    // NO endless-battle clause, so nothing ends the battle either — the run just parks.
    if (moves[k].disabled) continue;
    if (moves[k].pp !== undefined && moves[k].pp <= 0) continue;
    legalSlots.push(k);
    const id = toId(moves[k].id || moves[k].move);
    if (id === 'struggle' || isModeledMove(id)) modeledSlots.push(k);
  }
  const benchSlots = isTrapped ? [] : benchFromReq(req);
  if (modeledSlots.length === 0) {
    if (benchSlots.length > 0) return `switch ${benchSlots[randInt(rng, benchSlots.length)] + 1}`;
    // Prefer ANY usable slot over the old blind `move 1`, which happily re-picked a disabled /
    // 0-PP move forever. If literally nothing is usable the sim will substitute Struggle, and
    // `move 1` is then the correct thing to send.
    if (legalSlots.length > 0) return `move ${legalSlots[randInt(rng, legalSlots.length)] + 1}`;
    return 'move 1';
  }
  if (benchSlots.length > 0 && rng() < 1 / 6) {
    return `switch ${benchSlots[randInt(rng, benchSlots.length)] + 1}`;
  }
  return `move ${modeledSlots[randInt(rng, modeledSlots.length)] + 1}`;
}
function pickReplacementReq(req, rng) {
  const slots = benchFromReq(req);
  if (slots.length === 0) return null;
  return `switch ${slots[randInt(rng, slots.length)] + 1}`;
}

// ── Per-side byte diff ──────────────────────────────────────────────────────────
// Compare per-side chunk sequences byte-for-byte (chunk boundaries + line content).
function diffPerSide(nodeSide, rustSide) {
  for (const side of ['p1', 'p2']) {
    const a = nodeSide[side], b = rustSide[side];
    const n = Math.min(a.length, b.length);
    for (let ci = 0; ci < n; ci++) {
      const la = a[ci], lb = b[ci];
      const m = Math.min(la.length, lb.length);
      for (let li = 0; li < m; li++) {
        if (la[li] !== lb[li]) {
          return { side, chunk: ci, line: li, kind: classify(la[li], lb[li]), expected: la[li], got: lb[li] };
        }
      }
      if (la.length !== lb.length) {
        const longer = la.length > lb.length ? 'node' : 'rust';
        const extra = (la.length > lb.length ? la : lb)[m] || '';
        return { side, chunk: ci, line: m, kind: 'line_count',
          expected: la.length > lb.length ? extra : '', got: lb.length > la.length ? extra : '',
          detail: `chunk ${ci}: node ${la.length} lines vs rust ${lb.length} (${longer} longer)` };
      }
    }
    if (a.length !== b.length) {
      const longer = a.length > b.length ? 'node' : 'rust';
      const extra = (a.length > b.length ? a : b)[n] || [];
      return { side, chunk: n, line: 0, kind: 'chunk_count',
        expected: a.length > b.length ? (extra[0] || '') : '', got: b.length > a.length ? (extra[0] || '') : '',
        detail: `node ${a.length} chunks vs rust ${b.length} (${longer} longer) on ${side}` };
    }
  }
  return null;
}
function classify(exp, got) {
  const is = (l, p) => l.startsWith(p);
  if (is(exp, '|request|') || is(got, '|request|')) return 'request';
  if (is(exp, '|error|') || is(got, '|error|')) return 'error';
  if (is(exp, '|switch|') || is(exp, '|drag|') || is(exp, '|-damage|') || is(exp, '|-heal|') || is(exp, '|-sethp|')) return 'privacy';
  if (is(exp, '|player|') || is(exp, '|tier|') || is(exp, '|rule|') || is(exp, '|teamsize|') || is(exp, '|start') || is(exp, '|gametype|') || is(exp, '|gen|')) return 'preamble';
  return 'perside';
}

// ── KNOWN-RESIDUAL ALLOWLIST (`gen3_simbridge_diff_allowlist_v1`) ───────────────
// The GREEN-GATE discipline this harness was missing: a documented request-DISPLAY
// residual is counted SEPARATELY (and its repro filed under `allowlisted/`) instead of
// failing the run, while ANY un-cataloged byte difference still fails. Without this the
// gate is unusable — a 10-battle randbats smoke reported 8/10 "diverged", ALL of them the
// same `return102` alias.
//
// The keys mirror `src/bin/bridge_replay.rs::classify_known_perside_residual` EXACTLY, so
// the two harnesses agree on what is benign (they gate the same `|request|` serializer):
//
//   * `return102-numeric-alias` — Showdown renders the numeric-BP alias INCONSISTENTLY
//     (roster `return102`, active `id` bare `return`, active display `Return 102` WITH a
//     space); the port normalizes to the bare form. poke-env resolves both to the same move.
//   * `curse-nonghost-target-self-vs-normal` — a non-Ghost Curse's runtime-effective target
//     is `self` (`nonGhostTarget`) while the static dex says `normal`.
//   * `gender-level-details-construction-draw` — a `, L<n>` / `, <gender>` suffix in a
//     `details` field (inactive on pinned-gender L100 pools; kept for randbats/random).
//
// SAFETY (the load-bearing property, same as bridge_replay): this RECONCILES then compares
// the FULL lines — the pair is allowlisted ONLY IF the two are byte-EQUAL after applying the
// documented transforms. A co-occurring pair is allowed as a UNION, but ANY residual
// difference leaves them unequal → returns null → the divergence FAILS the gate. So this can
// never swallow a real bug that merely happens to share a line with a known form.
// Collapse a happiness-scaled move's numeric-BP alias to its bare token, for ANY base power
// (`gen3_happiness_bp_alias_any_digits_v1`), recording every removed digit run so the caller
// can compare them PAIRWISE. Anchored at a non-alphanumeric boundary, so a longer move whose
// name merely CONTAINS the token is never truncated; digits are the ONLY thing removed.
function stripBpAlias(s, bps) {
  const take = (name, d) => { (bps[name] = bps[name] || []).push(d); };
  return s
    .replace(/(^|[^A-Za-z0-9])(return|frustration)(\d+)/g,
      (_m, pre, name, d) => { take(name, d); return pre + name; })
    .replace(/(^|[^A-Za-z0-9])(Return|Frustration) (\d+)/g,
      (_m, pre, name, d) => { take(name.toLowerCase(), d); return pre + name; });
}

const ALLOWLIST_TRANSFORMS = [
  // Collapse every alias form to the bare token, in BOTH directions, so the inconsistent
  // sim rendering (roster vs active id vs display) normalizes the same way on both sides.
  //
  // ⚠️ The numeric suffix is the move's COMPUTED BASE POWER, so it VARIES: Return is
  // `max(1, floor(happiness * 2 / 5))` and Frustration the mirror, so at the DEFAULT
  // happiness 255 the pair renders `return102` and — the raw product being 0, clamped to 1 —
  // **`frustration1`, never `frustration102`**. Hardcoding `102` on both (as this did) means
  // the Frustration arm can never match a real battle: the pair fails to reconcile, and a
  // co-occurring correctly-handled `return102` FAILS THE GATE with it. Measured on the rust
  // sibling: 14 of 25 `--mode random` battles, all Frustration boards.
  //
  // PAIRWISE, per this file's own rule below: the deferral is presence-vs-ABSENCE of the
  // suffix, so if BOTH sides render a numeric BP and the values DISAGREE (a mis-parsed
  // happiness) that is a real divergence and the whole reconcile refuses (`null`).
  { key: 'return102-numeric-alias', applyPair: (a, b) => {
      const ab = {}; const bb = {};
      const a2 = stripBpAlias(a, ab); const b2 = stripBpAlias(b, bb);
      const uniq = (v) => [...new Set(v || [])].sort().join(',');
      for (const name of ['return', 'frustration']) {
        const ua = uniq(ab[name]); const ub = uniq(bb[name]);
        if (ua && ub && ua !== ub) return null;   // a BASE-POWER divergence, not a display form
      }
      return [a2, b2];
    } },
  // The non-Ghost Curse target.
  { key: 'curse-nonghost-target-self-vs-normal', apply: (s) => s
      .replace(/("id":"curse","pp":\d+,"maxpp":\d+,"target":)"normal"/g, '$1"self"') },
  // NOTE — the `gender-level-details-construction-draw` key from bridge_replay is
  // DELIBERATELY NOT PORTED as a blanket strip. Removing a `, L<n>` / `, <gender>` suffix
  // from BOTH sides would reconcile a genuine VALUE divergence (node `L84` vs rust `L83`,
  // or M vs F) into a false pass — the exact way an allowlist turns a gate VACUOUS. The
  // other two keys are safe because they map an ALIAS to a CANONICAL form: a real value
  // difference (`return` vs `tackle`) still fails to reconcile. If a genuine
  // presence-vs-absence details residual ever shows up here, implement it PAIRWISE (strip
  // only from the side that HAS the suffix when the other LACKS it) — never symmetrically.
];

// Returns the matching reason key(s) joined by '+', or null if the pair is NOT fully
// reconciled by the documented transforms (⇒ a genuine divergence).
function classifyKnownResidual(exp, got) {
  if (exp === got) return null;                       // not a divergence at all
  if (!exp.startsWith('|request|') || !got.startsWith('|request|')) return null;
  const used = [];
  let a = exp, b = got;
  for (const t of ALLOWLIST_TRANSFORMS) {
    let a2, b2;
    if (t.applyPair) {
      // A PAIRWISE transform may REFUSE (null) — it inspected both sides and found a genuine
      // value difference hiding under the alias form. Refusing fails the whole gate, by design.
      const pair = t.applyPair(a, b);
      if (pair === null) return null;
      [a2, b2] = pair;
    } else {
      a2 = t.apply(a); b2 = t.apply(b);
    }
    if (a2 !== a || b2 !== b) used.push(t.key);
    a = a2; b = b2;
  }
  // FULL reconciliation required — any residual byte difference fails the gate.
  return a === b && used.length ? used.join('+') : null;
}

// ── ALLOWLIST GATE-INTEGRITY SELF-TEST (`node gen_sim_bridge_diff.js --selftest`) ──
// The project standard for any allowlist: PROVE it cannot swallow a real bug (the rust
// side ships `a1_allowlist_tests` / `perside_construction_mirror_flip_tests` for exactly
// this). Every NEGATIVE case below is load-bearing — it asserts a genuine difference is
// still REPORTED. Run it after touching ALLOWLIST_TRANSFORMS.
function selftest() {
  const REQ = (moves, extra = '') =>
    `|request|{"active":[{"moves":${JSON.stringify(moves)}}]${extra},"side":{"name":"P1"}}`;
  let fail = 0;
  const check = (label, got, want) => {
    const ok = want === null ? got === null : got === want;
    if (!ok) { fail++; console.error(`  FAIL ${label}: got ${JSON.stringify(got)} want ${JSON.stringify(want)}`); }
    else console.error(`  ok   ${label}`);
  };

  // POSITIVE — the documented alias, in each of the three renderings the sim uses.
  check('roster alias reconciles',
    classifyKnownResidual(REQ(['return102']), REQ(['return'])), 'return102-numeric-alias');
  check('display alias reconciles',
    classifyKnownResidual(REQ([{ move: 'Return 102', id: 'return' }]), REQ([{ move: 'Return', id: 'return' }])),
    'return102-numeric-alias');
  check('frustration alias reconciles',
    classifyKnownResidual(REQ(['frustration102']), REQ(['frustration'])), 'return102-numeric-alias');
  // `gen3_happiness_bp_alias_any_digits_v1` — the REAL rendering at the default happiness 255
  // (BP clamped to 1). The pre-fix hardcoded `102` could not match this, so every Frustration
  // board read as a divergence and dragged its co-occurring `return102` down with it.
  check('frustration alias at the clamped BP of 1 reconciles',
    classifyKnownResidual(REQ(['frustration1']), REQ(['frustration'])), 'return102-numeric-alias');
  check('a non-default happiness BP reconciles',
    classifyKnownResidual(REQ([{ move: 'Return 84', id: 'return' }]),
                          REQ([{ move: 'Return', id: 'return' }])), 'return102-numeric-alias');

  // NEGATIVE — a genuine difference must NOT be allowlisted.
  // The PAIRWISE guard: both sides numeric but DISAGREEING is a base-power divergence (a
  // mis-parsed happiness), not a display form — collapsing both would be a false pass.
  check('a differing numeric BASE POWER on both sides is not allowlisted',
    classifyKnownResidual(REQ(['return102']), REQ(['return84'])), null);
  check('a differing frustration BASE POWER on both sides is not allowlisted',
    classifyKnownResidual(REQ([{ move: 'Frustration 1', id: 'frustration' }]),
                          REQ([{ move: 'Frustration 102', id: 'frustration' }])), null);
  check('a DIFFERENT move is not allowlisted',
    classifyKnownResidual(REQ(['return102']), REQ(['tackle'])), null);
  check('an alias PLUS a residual pp difference is not allowlisted',
    classifyKnownResidual(REQ([{ move: 'Return 102', id: 'return', pp: 32 }]),
                          REQ([{ move: 'Return', id: 'return', pp: 31 }])), null);
  // The one that killed the symmetric details strip: a LEVEL/GENDER VALUE divergence.
  check('a LEVEL value difference is not allowlisted',
    classifyKnownResidual(REQ(['return102'], ',"details":"Ninetales, L84, M"'),
                          REQ(['return'], ',"details":"Ninetales, L83, M"')), null);
  check('a GENDER value difference is not allowlisted',
    classifyKnownResidual(REQ(['return102'], ',"details":"Ninetales, L84, M"'),
                          REQ(['return'], ',"details":"Ninetales, L84, F"')), null);
  check('a missing move is not allowlisted',
    classifyKnownResidual(REQ(['return102', 'surf']), REQ(['return'])), null);
  // Non-request lines and equal lines are never allowlisted.
  check('a non-request line is never allowlisted',
    classifyKnownResidual('|-damage|p1a: X|100/200', '|-damage|p1a: X|99/200'), null);
  check('identical lines yield null',
    classifyKnownResidual(REQ(['return']), REQ(['return'])), null);

  // ── SWITCH-PROBE CONTRACT (`gen3_simbridge_probe_accepted_v1`) ──────────────────
  // The discriminator must be CONTENT (an `|error|` came back), never "did anything at all
  // arrive" — a probe that COMPLETES the boundary answers with the turn batch and no error,
  // and that is an ACCEPTED probe, not a rejected one.
  const chunk = (c) => [{ type: 'chunk', side: 'p1', chunk: c }];
  check('silence means the probe was ACCEPTED', probeWasAccepted([]), true);
  check('an |error| means the probe was REJECTED',
    probeWasAccepted(chunk("|error|[Unavailable choice] Can't switch: The active Pokemon is trapped")), false);
  check('the FIRM-trap |error| form is also REJECTED',
    probeWasAccepted(chunk("|error|[Invalid choice] Can't switch: The active Pokemon is trapped")), false);
  check('a turn batch with no error means ACCEPTED (the probe completed the boundary)',
    probeWasAccepted(chunk('|move|p1a: Snorlax|Body Slam|p2a: Dugtrio\n|-damage|p2a: Dugtrio|10/100')), true);

  console.error(fail ? `\n[selftest] ${fail} FAILURE(S)` : '\n[selftest] all allowlist gate-integrity cases pass');
  process.exit(fail ? 1 : 0);
}

// ── The trapping team provider (inlined from bridge_ab_fuzz.js) ─────────────────
// CORE = ARENA TRAP only. Its `onFoe` trap handler adds ZERO PRNG draws (1 handler per
// event → no tie-shuffle), so the sim's turn-0 CONSTRUCTION window is just the Quick Claw
// (1 draw) that `advance_seed_for_construction` models exactly → the seeded byte-diff over
// Arena Trap isolates the BRIDGE layer and is byte-perfect.
//
// MAGNET PULL + SHADOW TAG are behind `--extra-trappers`. Their FULL-BATTLE draw model is
// bit-for-bit (trapping_test.rs for MP; probe_shadowtag_rng.js for ST), but each surfaces a
// SEED-CONVENTION gap the bridge binary does not model in `advance_seed_for_construction`:
//   * MAGNET PULL is `onAnyTrapPokemon` — its turn-0 endTurn trap events draw a tie-shuffle
//     when the actives are speed-TIED (a Magneton-vs-Skarmory speed tie draws 4 EXTRA
//     construction draws → 5 total, not 1). The binary advances the raw seed by only the
//     Quick Claw, so a speed-TIED Magnet Pull battle desyncs the first-mover (probe-settled;
//     see the task report). NOT a bridge defect — the port's `run_full_battle` deliberately
//     does not model the turn-0 endTurn (the "pre-first-decision seed convention"), and the
//     construction draw COUNT is team-dependent (untied MP is fine; only the tie draws).
//   * SHADOW TAG surfaced a degenerate-endless-battle mirror first-mover flip (same class).
// So MP/ST are OFF by default; `--extra-trappers` includes them to exercise the requests +
// the trapped machine (the bridge-layer bytes ARE correct there — only the seeded first-mover
// on a tie desyncs).
const TRAPPERS_CORE = [
  { species: 'Dugtrio', ability: 'Arena Trap', moves: ['earthquake', 'splash'], gender: 'M' },
  { species: 'Diglett', ability: 'Arena Trap', moves: ['earthquake', 'splash'], gender: 'M' },
];
const TRAPPERS_EXTRA = [
  { species: 'Magneton', ability: 'Magnet Pull', moves: ['thunderbolt', 'splash'], gender: 'N' },
  { species: 'Nosepass', ability: 'Magnet Pull', moves: ['rockslide', 'splash'], gender: 'M' },
  { species: 'Wobbuffet', ability: 'Shadow Tag', moves: ['splash'], gender: 'M' },
  { species: 'Wynaut', ability: 'Shadow Tag', moves: ['splash'], gender: 'M' },
];
const TRAP_FOES = [
  { species: 'Snorlax', moves: ['bodyslam', 'splash'], gender: 'M' },
  { species: 'Zapdos', moves: ['thunderbolt', 'splash'], gender: 'N' },
  { species: 'Gengar', moves: ['icebeam', 'splash'], ability: 'Levitate', gender: 'M' },
  { species: 'Skarmory', moves: ['drillpeck', 'splash'], ability: 'Keen Eye', gender: 'M' },
  { species: 'Banette', moves: ['shadowball', 'splash'], gender: 'M' },
  { species: 'Regice', moves: ['icebeam', 'splash'], gender: 'N' },
  { species: 'Metagross', moves: ['meteormash', 'splash'], ability: 'Clear Body', gender: 'N' },
  { species: 'Salamence', moves: ['rockslide', 'splash'], ability: 'Intimidate', gender: 'M' },
  { species: 'Jirachi', moves: ['bodyslam', 'splash'], ability: 'Serene Grace', gender: 'N' },
  { species: 'Suicune', moves: ['surf', 'splash'], ability: 'Pressure', gender: 'N' },
];
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mkSet(spec) {
  return {
    name: spec.species, species: spec.species, item: spec.item || 'Leftovers',
    ability: spec.ability || 'No Ability', moves: spec.moves,
    evs: EV0, ivs: IV31, nature: 'Serious', level: 100, gender: spec.gender || 'N',
  };
}
function makeTrappingProvider(rng, trappers) {
  return function nextPair() {
    const trapper = trappers[randInt(rng, trappers.length)];
    const mirror = rng() < 0.25;
    const nFoes = 1 + randInt(rng, 4);
    const chosen = ab.sampleDistinct(rng, TRAP_FOES, nFoes);
    const p1 = mirror
      ? [mkSet(trapper), ...chosen.slice(0, Math.max(1, nFoes - 1)).map(mkSet)]
      : [mkSet(trapper)];
    const p2 = mirror
      ? [mkSet(trappers[randInt(rng, trappers.length)]), ...chosen.slice(0, Math.max(1, nFoes - 1)).map(mkSet)]
      : chosen.map(mkSet);
    const dedup = (team) => {
      const seen = new Set();
      return team.filter((s) => { const id = toId(s.species); if (seen.has(id)) return false; seen.add(id); return true; });
    };
    const t1 = dedup(p1), t2 = dedup(p2);
    return {
      p1: Teams.pack(t1.length ? t1 : [mkSet(trapper)]),
      p2: Teams.pack(t2.length ? t2 : [mkSet(TRAP_FOES[0])]),
    };
  };
}

// ── Team pair provider (reuse ab_fuzz.js) ───────────────────────────────────────
function buildPairProvider(flags, teamRng, genStats) {
  if (flags.mode === 'trapping') {
    const trappers = flags.extraTrappers ? TRAPPERS_CORE.concat(TRAPPERS_EXTRA) : TRAPPERS_CORE;
    return makeTrappingProvider(teamRng, trappers);
  }
  let single;
  if (flags.mode === 'randbats') single = ab.makeRandbatsProvider(teamRng, genStats);
  else if (flags.mode === 'pool') single = ab.makePoolProvider(teamRng, genStats);
  else {
    const universe = ab.buildRandomUniverse();
    single = ab.makeRandomProvider(teamRng, universe, genStats);
  }
  // Pin explicit genders (the sim draws a gender `sample` for an unspecified ratio, an
  // unmodeled construction draw the port doesn't advance for — see
  // `advance_seed_for_construction`'s gap note) AND LEVEL 100 (the port targets L100 in the
  // `|switch|`/`details` LEVEL display — a randbats L<100> set would show `, L79` on the
  // Node side but be omitted by the port; a pre-existing omniscient-stream gap orthogonal to
  // the bridge layer). Both make the seeded byte-diff isolate the BRIDGE, not those two
  // documented ENGINE gaps.
  const pinGendersAndLevel = (packed) => {
    const team = Teams.unpack(packed);
    if (!team) return packed;
    let touched = false;
    for (const set of team) {
      if (!set.gender) {
        const sid = toId(set.species || set.name);
        const sp = dex3.species.get(sid);
        if (sp && sp.exists) { set.gender = sp.gender === '' ? 'M' : sp.gender; touched = true; }
      }
      if (set.level && set.level !== 100) { set.level = 100; touched = true; }
    }
    return touched ? Teams.pack(team) : packed;
  };
  return () => ({ p1: pinGendersAndLevel(single().packed), p2: pinGendersAndLevel(single().packed) });
}

// ── Replay a SAVED repro deterministically (`--repro <dir>`) ────────────────────
// The differ discovers boundaries LIVE off node's request frames, which is exactly what makes
// it immune to the decision-segmentation artifact that dogs the offline `ab_replay` path. The
// cost was that a saved divergence could only be re-checked by re-running the whole sweep to
// the same battle index. This replays ONE saved repro's recorded `start_json` + `cmds` through
// BOTH real bridges — the SAME entry points production uses (node `local_sim_bridge.js`; rust
// `sim_bridge` -> `BridgeSession::new_construct_turn0`, which runs the real turn-0 window from
// the RAW seed) — so the seeding convention is correct BY CONSTRUCTION.
//
// That last point is not academic. A hand-built repro seeded with the RAW seed while the sim
// spends its first draws on turn-0 construction will silently replay those construction draws
// as its own move-phase draws and MANUFACTURE a divergence that does not exist. Replaying the
// recorded input through the real bridge removes that whole class of self-inflicted error.
async function replayRepro(dir) {
  const summary = JSON.parse(fs.readFileSync(path.join(dir, 'summary.json'), 'utf8'));
  console.error(`[repro] ${path.basename(dir)} battle=${summary.battle_index} format=${summary.format} cmds=${summary.cmds.length}`);
  if (summary.first_divergence) {
    const d = summary.first_divergence;
    console.error(`[repro] RECORDED: side=${d.side} chunk=${d.chunk} line=${d.line} kind=${d.kind}`);
  }
  buildRust();
  const node = new BridgeChild('node', [NODE_BRIDGE]);
  const rust = new BridgeChild(RUST_BRIDGE, []);
  const nodeSide = { p1: [], p2: [] }, rustSide = { p1: [], p2: [] };
  try {
    node.write('START ' + summary.start_json);
    rust.write('START ' + summary.start_json);
    collect(await drainQuiescent(node), nodeSide);
    collect(await drainQuiescent(rust), rustSide);
    // Feed in BOUNDARY groups, not one cmd at a time. A two-sided move boundary only flushes
    // once BOTH sides have answered, so draining after the first half waits the full quiescence
    // timeout for output that cannot arrive (147 cmds x one stalled drain each is a ~1h hang —
    // observed). The recorder wrote both sides of a boundary consecutively, so pair a cmd with
    // the next one when it is for the OTHER side; a lone cmd (forced switch) stays a group of 1.
    let fed = 0;
    for (let i = 0; i < summary.cmds.length; ) {
      if (nodeSide._ended || rustSide._ended) break;
      const group = [summary.cmds[i]];
      if (i + 1 < summary.cmds.length && summary.cmds[i + 1][0] !== summary.cmds[i][0]) {
        group.push(summary.cmds[i + 1]);
      }
      for (const [side, choice] of group) {
        node.write(`CHOOSE ${side} ${choice}`);
        rust.write(`CHOOSE ${side} ${choice}`);
      }
      collect(await drainQuiescent(node), nodeSide);
      collect(await drainQuiescent(rust), rustSide);
      i += group.length; fed += group.length;
    }
    let div = diffPerSide(nodeSide, rustSide);
    // Apply the SAME known-residual allowlist the main path uses, or this replay reports a
    // documented request-DISPLAY artifact (e.g. `return102-numeric-alias`) as a real divergence.
    // Reconcile-then-compare: any residual byte difference still fails.
    let allowed = null;
    if (div) {
      allowed = classifyKnownResidual(div.expected, div.got);
      if (allowed) console.error(`[repro] first diff is ALLOWLISTED (${allowed}) — scanning past it is not supported here; treat as clean for this line.`);
    }
    console.error(`[repro] fed ${fed}/${summary.cmds.length} cmds; ended node=${!!nodeSide._ended} rust=${!!rustSide._ended}`);
    if (!div) console.error('[repro] NO DIVERGENCE — this repro replays byte-clean.');
    else if (allowed) console.error(`[repro] ALLOWLISTED-ONLY (${allowed}) — no unexplained divergence.`);
    else {
      console.error(`[repro] DIVERGED side=${div.side} chunk=${div.chunk} line=${div.line} kind=${div.kind}`);
      // Show the DIFFERING REGION, not a fixed prefix: these lines are often long |request|
      // JSON whose first 200 chars are identical, which makes a head-slice useless.
      const ea = String(div.expected), ga = String(div.got);
      let k = 0; while (k < ea.length && k < ga.length && ea[k] === ga[k]) k++;
      const lo = Math.max(0, k - 110);
      console.error(`  first differing char: ${k} (lenA=${ea.length} lenB=${ga.length})`);
      console.error(`  expected: ...${ea.slice(lo, k + 110)}`);
      console.error(`  got:      ...${ga.slice(lo, k + 110)}`);
    }
  } finally { node.kill(); rust.kill(); }
}

// Build the Rust bridge into the isolated target dir (shared by main + --repro).
function buildRust() {
  const { spawnSync } = require('child_process');
  const env = { ...process.env, PATH: `${process.env.HOME}/.cargo/bin:${process.env.PATH}`, CARGO_TARGET_DIR: SIMBRIDGE_TARGET };
  const r = spawnSync('cargo', ['build', '--release', '--bin', 'sim_bridge'], { cwd: CRATE, env, stdio: 'inherit' });
  if (r.status !== 0) { console.error('[sim_bridge_diff] cargo build failed'); process.exit(1); }
  if (!fs.existsSync(RUST_BRIDGE)) { console.error(`[sim_bridge_diff] rust binary missing: ${RUST_BRIDGE}`); process.exit(1); }
}

// ── Main ────────────────────────────────────────────────────────────────────────
async function main() {
  if (parseFlags(process.argv).selftest) return selftest();
  const flags = parseFlags(process.argv);
  if (flags.repro) return replayRepro(flags.repro);
  const runId = `sbd_${Date.now().toString(36)}`;
  fs.mkdirSync(path.join(flags.out, 'divergences'), { recursive: true });

  console.error(`[sim_bridge_diff] run_id=${runId} mode=${flags.mode} format=${flags.format} ` +
    `battles=${flags.battles} master_seed=${flags.masterSeed} persistent=${flags.persistent} ` +
    `trap_prob=${flags.trapProb} out=${flags.out}`);
  console.error(`[sim_bridge_diff] reproduce: node src/rust_sim/harness/gen_sim_bridge_diff.js ` +
    `--mode ${flags.mode} --format ${flags.format} --battles ${flags.battles} --master-seed ${flags.masterSeed}` +
    (flags.persistent ? ' --persistent' : ''));
  console.error('[sim_bridge_diff] __RECON__ is EXCLUDED from the diff (drop-in binary defers it; ' +
    'the Python side degrades gracefully). Cross-side p1/p2 interleaving is not asserted (scheduler ' +
    'artifact; the diff compares PER-SIDE chunk sequences).');

  // Build the Rust binary ONCE into the isolated target dir.
  {
    const { spawnSync } = require('child_process');
    const env = { ...process.env, PATH: `${process.env.HOME}/.cargo/bin:${process.env.PATH}`, CARGO_TARGET_DIR: SIMBRIDGE_TARGET };
    const r = spawnSync('cargo', ['build', '--release', '--bin', 'sim_bridge'], { cwd: CRATE, env, stdio: 'inherit' });
    if (r.status !== 0) { console.error('[sim_bridge_diff] cargo build failed'); process.exit(1); }
  }
  if (!fs.existsSync(RUST_BRIDGE)) { console.error(`[sim_bridge_diff] rust binary missing: ${RUST_BRIDGE}`); process.exit(1); }

  const teamRng = mulberry32(flags.masterSeed);
  const battleRng = mulberry32((flags.masterSeed ^ 0xabcdef01) >>> 0);
  const genStats = { setsTotal: 0, setsAdjusted: 0, naturesNormalized: 0, teamsKept: 0, teamsRejected: 0, genErrors: 0, rejectReasons: new Map() };
  const pairProvider = buildPairProvider(flags, teamRng, genStats);

  const cov = { battles: 0, ok: 0, diverged: 0, err: 0, ended: 0, trapped: 0, forceSwitch: 0, persistentBattles: 0, kinds: new Map(),
    allowlisted: 0, allowReasons: new Map(), startedAt: Date.now(), elapsedMs: 0 };

  // Persistent mode: ONE child pair reused across ALL battles (the multi-battle-per-child path).
  let sharedNode = null, sharedRust = null;
  if (flags.persistent) {
    sharedNode = new BridgeChild('node', [NODE_BRIDGE]);
    sharedRust = new BridgeChild(RUST_BRIDGE, []);
  }

  for (let bi = 0; bi < flags.battles; bi++) {
    const pair = pairProvider();
    const battleSeed = seedFrom((Math.floor(battleRng() * 4294967296)) >>> 0);
    const chooseSeed = (Math.floor(battleRng() * 4294967296) ^ 0x9e3779b9) >>> 0;
    const start = { formatid: flags.format, p1: { name: 'P1', team: pair.p1 }, p2: { name: 'P2', team: pair.p2 } };
    if (battleSeed) start.seed = battleSeed;
    if (flags.persistent) start.persistent = true;
    const startJson = JSON.stringify(start);

    const node = flags.persistent ? sharedNode : new BridgeChild('node', [NODE_BRIDGE]);
    const rust = flags.persistent ? sharedRust : new BridgeChild(RUST_BRIDGE, []);
    if (flags.persistent) cov.persistentBattles++;

    let result;
    try {
      result = await diffOneBattle(node, rust, startJson, chooseSeed, flags.trapProb, flags.verbose);
    } catch (e) {
      result = { err: String(e && e.message || e) };
    }
    if (result.skip) {
      cov.skipped = (cov.skipped || 0) + 1;
      cov.battles++;
      // Non-persistent: throw the children away (the battle wasn't played out). Persistent:
      // the shared children are now mid-battle with no clean reset; recreate them so the next
      // START starts fresh (a skip is rare with distinct-speed leads).
      if (!flags.persistent) { node.kill(); rust.kill(); }
      else {
        sharedNode.kill(); sharedRust.kill();
        sharedNode = new BridgeChild('node', [NODE_BRIDGE]);
        sharedRust = new BridgeChild(RUST_BRIDGE, []);
      }
      continue;
    }

    cov.battles++;

    if (result.err) {
      cov.err++;
      // ALWAYS print — a battle error is rare and load-bearing (a stalled child, a fail-loud
      // panic in either bridge). Hiding it behind --verbose is what made the wedged soak look
      // like an unexplained hang instead of a reportable finding.
      console.error(`[ERROR] battle ${bi}: ${result.err}`);
      // SAVE A REPRO for an error too (`gen3_simbridge_error_repro_v1`). Previously only
      // divergences and allowlisted residuals were filed, so a STALL — the thing that killed a
      // whole soak — left nothing to analyse and could only be studied by re-running the seed
      // and hoping. An error is the case that MOST needs a repro.
      try {
        const dir = saveRepro(flags, runId, bi, pair, startJson,
          { nodeSide: result.nodeSide || {}, rustSide: result.rustSide || {}, cmds: result.cmds || [] },
          { side: '-', chunk: -1, line: -1, kind: 'error', expected: '', got: '', detail: result.err },
          chooseSeed, 'errors');
        console.error(`  repro → ${dir}`);
      } catch (e) { console.error(`  (repro save failed: ${e && e.message})`); }
      // RECOVER, don't abort (`gen3_simbridge_error_recovery_v1`). The children ARE
      // unrecoverable mid-battle — but the fix is to REPLACE them, exactly as the `skip` path
      // above already does, not to end the run.
      //
      // WHY: this one line cost the entire 1200-battle soak. It stopped at battle **59** — not
      // because anything was wrong with the port, but because ONE battle errored and persistent
      // mode `break`s. The battle in question was a 106-turn ENDLESS STALL (p2 down to a single
      // full-HP Leftovers Tropius spamming Swords Dance, its damaging moves at 0 PP) — an
      // artifact of RANDOM play, not a parity finding, and exactly the kind of battle a long
      // fuzz run will keep producing. Aborting on it threw away the other 1141 battles.
      if (!flags.persistent) { node.kill(); rust.kill(); }
      else {
        sharedNode.kill(); sharedRust.kill();
        sharedNode = new BridgeChild('node', [NODE_BRIDGE]);
        sharedRust = new BridgeChild(RUST_BRIDGE, []);
      }
      continue;
    }

    if (result.ended) cov.ended++;
    // Coverage: count trapped-reject probes + forced switches from the cmd/node stream.
    for (const [, c] of result.cmds) if (c.startsWith('switch')) cov.forceSwitch++;
    for (const side of ['p1', 'p2']) {
      for (const ch of result.nodeSide[side]) {
        for (const l of ch) { if (l.startsWith('|request|') && l.includes('"trapped":true')) cov.trapped++; }
      }
    }

    const div = diffPerSide(result.nodeSide, result.rustSide);
    if (!div) {
      cov.ok++;
    } else {
      // GREEN GATE: a documented request-DISPLAY residual is counted separately (repro filed
      // under `allowlisted/`) and does NOT fail the run; anything un-cataloged still does.
      const reason = classifyKnownResidual(div.expected, div.got);
      if (reason) {
        cov.allowlisted++;
        cov.allowReasons.set(reason, (cov.allowReasons.get(reason) || 0) + 1);
        saveRepro(flags, runId, bi, pair, startJson, result, div, chooseSeed, 'allowlisted');
      } else {
        cov.diverged++;
        cov.kinds.set(div.kind, (cov.kinds.get(div.kind) || 0) + 1);
        const dir = saveRepro(flags, runId, bi, pair, startJson, result, div, chooseSeed);
        console.error(`[DIVERGED] battle ${bi} side=${div.side} chunk=${div.chunk} line=${div.line} kind=${div.kind}\n` +
          `  expected: ${JSON.stringify(div.expected)}\n  got:      ${JSON.stringify(div.got)}\n  detail: ${div.detail || ''}\n  repro → ${dir}`);
      }
    }

    if (!flags.persistent) { node.kill(); rust.kill(); }
  }
  if (flags.persistent) { sharedNode.kill(); sharedRust.kill(); }

  cov.elapsedMs = Date.now() - cov.startedAt;
  const kindsStr = [...cov.kinds.entries()].map(([k, c]) => `${k}=${c}`).join(',') || '-';
  const allowObj = Object.fromEntries(cov.allowReasons);
  console.error('\n[sim_bridge_diff] SUMMARY');
  console.error(JSON.stringify({
    run_id: runId, mode: flags.mode, format: flags.format, master_seed: flags.masterSeed,
    battles: cov.battles, ok: cov.ok, diverged: cov.diverged, errored: cov.err,
    allowlisted: cov.allowlisted, allowlisted_by_reason: allowObj,
    skipped_lead_speed_tie: cov.skipped || 0,
    ended_battles: cov.ended, trapped_true_frames: cov.trapped, switch_cmds: cov.forceSwitch,
    persistent_battles: cov.persistentBattles, divergence_kinds: kindsStr,
    // Stall telemetry (`gen3_simbridge_probe_accepted_v1`): `drain_timeouts` must stay 0 —
    // ANY non-zero value means a child went quiet somewhere, even on a path that recovers.
    switch_probes_accepted: probesAccepted,
    drain_timeouts: DRAIN_TIMEOUTS.length,
    drain_timeouts_by_tag: DRAIN_TIMEOUTS.reduce((m, d) => { m[d.tag] = (m[d.tag] || 0) + 1; return m; }, {}),
    battles_per_hour: cov.elapsedMs ? Math.round(cov.battles / (cov.elapsedMs / 3.6e6)) : null,
  }, null, 2));
  if (cov.diverged > 0) {
    console.error(`[sim_bridge_diff] GREEN-GATE FAIL: ${cov.diverged} NON-allowlisted divergence(s)` +
      ` (allowlisted=${cov.allowlisted} by reason ${JSON.stringify(allowObj)})`);
  }

  // Exit non-zero ONLY on a NON-allowlisted divergence (the ab_fuzz green-gate convention).
  process.exit(cov.diverged > 0 ? 1 : 0);
}

function saveRepro(flags, runId, bi, pair, startJson, result, div, chooseSeed, bucket) {
  // `bucket` selects `divergences/` (a real find) or `allowlisted/` (a documented
  // request-DISPLAY residual, kept for audit + as a corpus-fixture source).
  const dir = path.join(flags.out, bucket || 'divergences', `${runId}_b${bi}`);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'summary.json'), JSON.stringify({
    run_id: runId, battle_index: bi, mode: flags.mode, format: flags.format, master_seed: flags.masterSeed,
    start_json: startJson, choose_seed: chooseSeed, packed_teams: pair,
    cmds: result.cmds, first_divergence: div,
    note: '__RECON__ excluded; per-side chunk sequences compared (cross-side interleaving is a scheduler artifact).',
  }, null, 2) + '\n');
  // Dump both per-side streams for eyeballing.
  const dump = (name, side) => {
    const out = [];
    for (const s of ['p1', 'p2']) {
      side[s].forEach((ch, ci) => { out.push(`=== ${s} chunk ${ci} ===`); ch.forEach((l) => out.push('  ' + l)); });
    }
    fs.writeFileSync(path.join(dir, name), out.join('\n') + '\n');
  };
  dump('node_stream.txt', result.nodeSide);
  dump('rust_stream.txt', result.rustSide);
  return dir;
}

// Exported so the DETERMINISTIC regression gate (`probe_switch_probe_accepted.js`) can drive
// `diffOneBattle` on a FIXED matchup instead of re-running a whole random sweep.
module.exports = {
  BridgeChild, diffOneBattle, drainQuiescent, probeWasAccepted, diffPerSide,
  classifyKnownResidual, NODE_BRIDGE, RUST_BRIDGE, DRAIN_TIMEOUTS,
};

if (require.main === module) {
  main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
}
