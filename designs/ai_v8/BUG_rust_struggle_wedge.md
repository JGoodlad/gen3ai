# BUG HANDOFF — rust sim wedges (infinite loop) on Struggle / full PP exhaustion

**Severity:** HIGH — blocked `--use-bridge=rust` as a training/eval transport.
**Status:** **RESOLVED 2026-07-22 (`gen3_bridge_struggle_resolve_v1`, a parallel thread — fix not yet
on this commit's `main`).** The repro + evidence below stand; only the ROOT CAUSE differed from this
doc's first hypothesis (see the banner).
**Found:** 2026-07-22, running the belt-and-braces multi-battle rust smoke that was deferred when the
Tier-2 incremental fix (`93f05fb`) shipped.

> **RESOLUTION UPDATE.** The actual root cause was **NOT** a Struggle-EXECUTION defect (the "The fix"
> section's hypothesis). The engine runs Struggle bit-for-bit (validated by the `pp_struggle` pins +
> e2e). The bug was the BRIDGE **rejecting the Struggle CHOICE**: `bridge::resolve_choice` maps a
> poke-env `move <id>` against the mon's REAL 4-move set, but a must-Struggle mon's request offers the
> synthetic `{"id":"struggle"}`, which is never in the real moveset → `resolve_choice` returned `None`
> → the out-of-range fallback `Move(moves.len())` → `choice_is_legal` rejected it → the boundary never
> committed → the bridge re-issued the SAME request forever (hence the `serialize_mon`/`build_request`
> loop the gdb caught). Fix: one branch in `resolve_choice` (mirroring the `move_locked` single-entry
> mapping) — `if want == "struggle" { return Some(Choice::Move(0)) }` — then the driver substitutes
> Struggle via its `must_struggle` exception. The `gen_sim_bridge_diff.js` long/PP-stall corpus
> extension (below) is still the right regression net. The symptom analysis in this doc (node 5271 vs
> rust 1M+ lines, the deterministic repro, the gdb trace) was all correct; only the fix LOCATION moved
> from the move engine to the choice resolver.

---

## One-line

When a Pokémon has 0 PP on **every** move and must use **Struggle**, the rust sim (`src/rust_sim`,
`--use-bridge=rust`) emits `|-activate|<mon>|move: Struggle` and re-issues a `|request|` to both
players **without ever executing Struggle** (no typeless damage, no ¼ recoil, no turn advance). The
battle state never changes → an endless no-progress bridge↔Python exchange. **The node bridge resolves
Struggle and the game ends normally.**

## Symptom / measured impact

On the identical deterministic battle (same seed + teams + action sequence):

| bridge | protocol lines | outcome |
|---|---|---|
| node (`local_sim_bridge.js`) | **5,271** | completes |
| rust (`sim_bridge`) | **1,048,107** | infinite loop (killed at timeout) |

- At **concurrency=1** (the production mode — one bridge child per training env; eval defaults to
  `--eval-concurrency-per-worker 1`), **2–5 of 16 random pool matchups wedge; node wedges 0/16.**
- Trigger is **PP exhaustion**, which random play and PP-stall strategies reach routinely → ~30% of
  random-play battles wedge. In training this stalls the `SubprocVecEnv` barrier (one wedged env hangs
  the whole run) → a silent 0-FPS hang, then a launcher restart at best.
- This is a **SECOND, DISTINCT bug** from the O(N³) replay-from-genesis slowness fixed in `93f05fb`.
  That fix made per-decision replay flat-fast (verified); this one is a **Struggle-EXECUTION defect**.
  Turns advance normally until a mon hits Struggle, then freeze.

## Deterministic repro (self-contained)

Requires the rust binary built (`cd src/rust_sim && cargo build --release --bin sim_bridge`, or set
`POKESIM_SIM_BRIDGE_BIN`). Run from repo root with the project interpreter + `PYTHONPATH=src`.

```python
# repro.py — node completes (~5k lines), rust wedges (>1M lines, turn freezes)
import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
import asyncio, hashlib, random, subprocess
import torch as th
from poke_env.player import RandomPlayer
from poke_env.ps_client import LocalhostServerConfiguration, AccountConfiguration
from utils.teambuilder import Gen3Teambuilder
from utils.team_loader import TeamLoader
from utils.bridge.local_battle_runner import run_local_battles
th.set_num_threads(1)
teams = TeamLoader().get_all_teams()
sha = lambda t: hashlib.sha1(t.strip().encode()).hexdigest()[:10]
bysha = {sha(t): t for t in teams}
t1, t2 = bysha["21022d30fb"], bysha["3495ef83ef"]   # Skarmory (p2) Struggles under random play
n = [0]
def acct():
    n[0] += 1; return AccountConfiguration(f"BG{n[0]:02d}", "pw")
async def run(impl, budget):
    random.seed(0)                                   # deterministic RandomPlayer action sequence
    p1 = RandomPlayer(team=Gen3Teambuilder([t1]), battle_format="gen3ou",
                      server_configuration=LocalhostServerConfiguration,
                      account_configuration=acct(), start_listening=False)
    p2 = RandomPlayer(team=Gen3Teambuilder([t2]), battle_format="gen3ou",
                      server_configuration=LocalhostServerConfiguration,
                      account_configuration=acct(), start_listening=False)
    try:
        await asyncio.wait_for(run_local_battles(p1, p2, 1, concurrency=1, impl=impl,
                                                 seed=[0, 0, 0, 0]), budget)
        print(f"{impl}: COMPLETED  finished={p1.n_finished_battles}")
    except asyncio.TimeoutError:
        print(f"{impl}: WEDGED (>{budget}s, no completion)")
    subprocess.run(["pkill", "-9", "-x", "sim_bridge"], check=False)
asyncio.run(run("node", 60)); asyncio.run(run("rust", 45))
```

Expected: `node: COMPLETED finished=1` / `rust: WEDGED`. Also available (scratch, in the main
checkout's `tmp/`, not committed): `rust_hang_isolate.py` (hang-rate over many pairs + bridge cpu/stat
sampling), `rust_hang_progress.py` (turn-freeze vs advance probe), `proto_diff.py` (node-vs-rust
protocol-tail diff — the script that named the bug). Copy their logic from here if `tmp/` is gone.

## Root cause (evidence)

- **gdb on the wedged `sim_bridge`:** steady-state blocked in `read()` on stdin
  (`std::io::BufReader<StdinRaw>::read_until`), caught once mid
  `bridge::BridgeSession::advance → emit_boundary_request_chunks → build_request → serialize_mon →
  String::clone`. So the bridge finishes emitting a request, then blocks for a reply, gets one, emits
  the next identical request… the loop is a real bridge↔Python round-trip that makes **no state
  progress.** (Note: `ps %cpu ≈18%` is the process LIFETIME average — the fast early game — NOT a spin;
  do not misread it.)
- **Protocol tail (rust), repeating forever:**
  ```
  |-activate|p2a: Skarmory|move: Struggle
  |request| {"active":[{"moves":[{"move":"Spikes","pp":2,...}]}]}      (p1)
  |request| {"active":[{"moves":[{"move":"Struggle",...}]}]}           (p2)
  |-activate|p2a: Skarmory|move: Struggle
  ...
  ```
  Struggle is **announced** (`-activate`) but its **effect never runs** — no `|move|`, no damage, no
  `[from] Recoil`, no turn increment. Node’s stream for the same state shows the full Struggle
  resolution and the game ending.

## The fix

- **Where:** `src/rust_sim/src/turn/driver.rs` — the `QAction::Move { struggle: true }` path (`run_move`
  / the move engine). The Struggle *substitution* logic (`must_struggle`, the `struggle` flag) already
  works — the mon is correctly forced to Struggle; what fails is **executing** it.
- **What Struggle must do in gen 3 (match node byte-for-byte):** a typeless physical attack (BP 50,
  never immune — hits through type chart), then **recoil to the user of ¼ of its MAX HP** (gen 3
  Struggle recoil is maxhp/4, floor 1), then advance the turn normally (faint checks, residuals, next
  turn). Confirm the exact `|move|`/`|-damage|`/`[from] Recoil|[of]` line shapes against the node
  stream — `src/rust_sim/src/protocol.rs:545` already documents the gen-3 Struggle-recoil `[from]`/`[of]`
  convention, so the emission format is known; the gap is the execution/advance.

## Acceptance / verification (do all three)

1. The repro above prints `rust: COMPLETED` and the rust line count ≈ node’s (± the normalized
   timestamp / account-name cosmetic diffs).
2. **A named deterministic regression test** in `src/rust_sim` (fixed seed) that drives a battle to PP
   exhaustion and asserts it COMPLETES (and that Struggle dealt damage + recoil) — must FAIL if the fix
   is reverted. (Project rule: every sim bug → a named regression test.)
3. **Extend `src/rust_sim/harness/gen_sim_bridge_diff.js`** to include long PP-stall / Struggle battles
   and re-run node-vs-rust byte parity → 0 diverged. Its current 40 SHORT battles never reach Struggle,
   which is exactly why this escaped.

## Why it escaped (prevention)

The rust port is validated three ways — the golden md5, the rust-internal parity test
(`bridge_incremental_matches_genesis_replay`), and `gen_sim_bridge_diff.js` (40 battles vs node). **None
of the three drove a battle to full PP exhaustion**, so Struggle’s execution was never differenced
against node. The durable fix to the *class* is a committed long-battle / PP-stall corpus in the
byte-differential harness (see memory `feedback_validate_observable_bytes` — "validate observable BYTES
continuously, not just state+seed"). Add that corpus alongside the point fix.

## Context / links

- Memory: `project_rust_bridge_incremental` (updated 2026-07-22 with this finding),
  `project_rust_sim_port`, `feedback_validate_observable_bytes`, `feedback_edge_case_regression_tests`,
  `feedback_determinism_means_fixable` (the node byte stream is the oracle — this IS fixable).
- Until fixed: train/eval on `--use-bridge=node` or websocket. Nothing else depends on rust.
