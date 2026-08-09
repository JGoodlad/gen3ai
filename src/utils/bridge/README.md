# Pokémon Showdown Bridge

This directory contains the bridge logic used to access the Pokémon Showdown simulation library directly from Python, without requiring a running server.

## Overview
By bridging Python to Node.js, we can utilize the actual `pokemon-showdown` codebase (located in `deps/`) as a library. This allows us to perform complex operations like **Team Validation** using the official Smogon rules and logic, but with the performance of a local function call.

## Components

### Team validation (request/response)
1.  **`validate_team.js`**: A Node.js script that imports the Showdown `TeamValidator` and `Teams` modules. It reads a team and format from `stdin` and outputs a validation result as JSON.
2.  **`team_validator.py`**: The Python wrapper that manages the subprocess communication with the Node bridge.

### Local battle (streaming) — run whole battles with no Showdown server
A second bridge runs *entire battles* in-process via Showdown's `BattleStream`, so a
poke-env `Player` can play with **no websocket server, no port, no usernames, no
matchmaking**. Unlike the team validator, it relays the **protocol text** each side sees,
so it feeds poke-env's parser exactly as the live server would — the parsing/encoder
pipeline runs byte-for-byte the same; only the transport changes (poke-env issue #907).

1.  **`local_sim_bridge.js`**: A Node streaming relay. `START`/`CHOOSE`/`FORCELOSE`/`END`
    on `stdin`; per-side, base64-framed protocol chunks (`p1 <b64>` / `p2 <b64>`) on
    `stdout`. One battle per process. Uses `getPlayerStreams()` so Showdown does the
    per-side channel demux. Accepts an optional fixed PRNG `seed` for reproducible battles.
2.  **`battle_stream_client.py`**: `BattleStreamClient(PSClient)` — a poke-env transport
    that subclasses the (vendored) `PSClient` **from outside** `src/poke_env/` (it modifies
    no poke_env file). No websocket; translates poke-env's `/choose …` onto the bridge and
    no-ops websocket ceremony (`/utm`, `/timer on`, `/leave`, `/challenge`, …).
3.  **`local_battle_runner.py`**: `run_local_battles(player1, player2, n_battles, *,
    seed=None)` — a drop-in for `player1.battle_against(player2, n_battles=…)`. It spawns one
    bridge per battle, fabricates the `>battle-…`/`|init|` room header the sim does not emit,
    and routes each side's protocol to the right player's client. Build the players with
    `start_listening=False` (no websocket opens); the runner swaps in the bridge transport.

This powers the `*_fuzz_test.py` suite. (A few timing-sensitive checks stay on the live
server as `*_fuzz_e2e_test.py` — e.g. `effectiveness_fuzz_e2e_test`.)

#### Node vs Rust sim bridge (`--use-bridge {off,node,rust}`)

The bridge child that speaks the `local_sim_bridge.js` stdin/stdout protocol has **two
implementations**, selected by `--use-bridge`:

- **`node`** (`local_sim_bridge.js`) — the default bridge impl, a relay over the real Showdown
  `BattleStream`. Handles the full gen3 move/ability set and produces the `__RECON__`
  reconstruction record + honors `resumeReseed` (the forensic / search / counterfactual layers).
  **Every exit path is drain-aware** (`gen3_bridge_flush_on_exit_v1`, `exitWhenDrained`): a bare
  `process.exit()` after `out()` discards Node's un-drained async pipe writes, which TRUNCATED
  large `__RECON__` lines whenever the reader drained slowly (224 `Incorrect padding` capture
  failures in run_20260807_135637's final eval at `--eval-concurrency 100`; the persistent
  training path never exits, so it never truncated — and the rust bridge was never affected,
  its `LineWriter` blocks on the kernel pipe per line). Gate:
  `bridge_flush_on_exit_integration_test.py` (deterministic — a 1 MB `__ERR__` payload with a
  deliberately lazy reader; fails at ~48 KB on a pre-fix bridge).
- **`rust`** — the std-only `src/rust_sim/src/bin/sim_bridge.rs` binary, a byte-for-byte
  protocol-compatible drop-in (validated at the chunk/stdout level by
  `src/rust_sim/harness/gen_sim_bridge_diff.js`). No Node needed for battle stepping.

`sim_bridge_bin.py::bridge_spawn_argv(impl)` turns the impl into the spawn argv both transport
seams (`bridge_session.py`, `local_battle_runner.py`) exec:
`node` → `["node", local_sim_bridge.js]`; `rust` → `[<resolved sim_bridge binary>]`.
`resolve_sim_bridge_bin()` honors `$POKESIM_SIM_BRIDGE_BIN` (absolute-path override) first, else
runs `cargo build --release --bin sim_bridge` in `src/rust_sim` and caches the resulting
`target/release/sim_bridge`; it raises a clear, actionable error (never a silent fall-back to
node) if cargo/crate/binary is unavailable.

**Honest scope of `rust` (re-audited 2026-08-04) — what works, and the ONE real gap.**
`__RECON__` (`gen3_bridge_recon_record_v1`) and `resumeReseed` (`gen3_bridge_resume_reseed_v1`) BOTH
shipped in `2b826d4`; the older "no `__RECON__` / no `resumeReseed`" text here was stale. A
**seeded** rust battle passes the whole forensic stack — `reconstruction_fuzz_test`,
`reroll_many_parity_fuzz_test` and `search_clone_parity_fuzz_test` all PASS on rust-recorded
records, with the Node replay/clone drivers reproducing the rust-recorded winner, turn and next obs
bit-for-bit (a cross-impl parity result worth more than the fuzzes' nominal subject). The record's
`>start`/`>player` lines are exact — the only part `replay_kernels.js` reads; its committed-choice
`input_log` lines are re-rendered from the engine's own script (`>p1 move 1` slot form vs Node's
`>p1 move icebeam`), replay-equivalent but not byte-identical, and consumed by nothing.

**⚠️ The three SEED gaps once listed here are FIXED** (`bc00d4d`,
`gen3_bridge_seedless_fixed_seed_v1` + `gen3_bridge_seed_forms_v1`; re-verified against the live
tree and the built binary 2026-08-04). Do not plan from the struck-through text:
- ~~The SEEDLESS path emits nothing and is not random.~~ **FIXED.** A seedless `START` MINTS a fresh
  `sodium,<hex>` (`Prng::generate_seed`) instead of falling through to `DEFAULT_CONSTRUCT_SEED`, and
  emits `__RECON__` carrying that resolved seed. So a `rust` run DOES write per-trace
  `*_reconstruction.json` (prober `falsify` / `better-line` / `replay-counterfactual` work) and every
  training episode draws its own dice. Gate:
  `bridge_impl_parity_test::test_seedless_rust_battles_are_distinct_and_recorded`.
- ~~A STRING `seed` is silently dropped.~~ **FIXED.** One shared `parse_seed_field` accepts every
  form Node accepts (`[a,b,c,d]`, `"m,n,o,p"`, `"gen5,<hex16>"`, `"sodium,<hex>"`); a
  present-but-unparseable seed is a LOUD `__ERR__`, never a silent fall-through. Gate:
  `test_seed_forms_reproduce_the_same_battle_on_rust_and_node`.
- ~~`resumeReseed` accepts only the array form.~~ **FIXED** — same shared parser, so the
  counterfactual Monte-Carlo works on rust.

**THE COVERAGE-HOLE LESSON (the durable part).** Every gate on that path — `sim_bridge_bin_test`,
`gen_sim_bridge_diff.js`, and the parity test's own win-rate check — was inherently SEEDED or
compared only aggregates, so the **production SEEDLESS branch was never exercised**. A "default"
branch that no test takes is untested however green the suite looks.

Still genuinely deferred:
- **No rust clone-and-branch search** (`Battle::serialize`/`deserialize` are `todo!()` in
  `src/rust_sim/src/battle.rs`), and `teacher/generate.py` calls `run_local_battles` with no `impl`
  (→ node). `train_rl_agent` hard-`parser.error`s on `--search-teacher` / `--teacher-persistent` +
  `rust` — correct, but for the `input_log` reason above (replay-EQUIVALENT, not byte-identical),
  not for any seed reason.

**Coverage (MEASURED).** The port fail-louds (`__ERR__ … is not modeled` → `RuntimeError` → env
crash → launcher restart; the child itself survives via `catch_unwind`) rather than desync. On the
**training pool this is a non-issue**: 719/719 `data/teams/` teams construct and 1500 random-play
rust battles produced **zero** coverage errors. **⚠️ The randbats figures once quoted here (14.0% of
teams / 27.0% of battles) are STALE — every blocker they named is now CLOSED**, so do not plan from
them: the Deoxys/Unown forme DATA gap is fixed (`gen3_species_formes_v1`, ROUND 38 — 419 species rows
incl. the 33 gen-3 formes), `transform` is modeled (`gen3_transform_v1`, ROUND 33), the wrap family
too (`gen3_partial_trap_v1`, ROUND 32), and **Forecast/Castform — the last construction blocker — is
modeled** (`gen3_forecast_v1`, ROUND 35). Arbitrary ladder `gen3ou` remains ~5% of battles (INFERRED
from Smogon usage weights, not measured).

The old counter-hazard — unmodeled items/moves running as **silent no-ops** — is likewise CLOSED by
the ROUND 39/40 silent-no-op audits: the 5 genuinely-effectful unmodeled items
(`gen3_unmodeled_item_failloud_v1`) and the 16 silent-desync moves (`fakeout`, `rollout`, the lock-in
family, …, `gen3_unmodeled_move_failloud_v2`) now FAIL LOUD at construction. Full-universe census,
re-run 2026-08-04: **369 gen3-legal moves → 281 modeled, 88 fail-loud, 0 MISMODELED**, checkable via
`SCAN_UNIVERSE=1 node src/rust_sim/harness/scan_move_coverage.js` (exits non-zero if any silent
desync reappears). Measured exposure of the guarded sets is ZERO on both surfaces — they are
latent-hazard guards, not live failures. (An earlier note here named *Aromatherapy* and *Wish* as
unmodeled examples — both are in fact MODELED; the pool carries nothing unmodeled at all.)
- **Turn-0 construction — MODELED (`gen3_turn0_construction_v1`).** The bridge builds via
  `BridgeSession::new_construct_turn0`, which runs the sim's full turn-0 construction window from the
  RAW `>start` seed — the per-mon gender `sample(['M','F'])` + the speed-tie insertChoice/eachEvent
  shuffles (incl. Magnet Pull's `onAny` trap shuffles + a weather-setter's WeatherChange) + the Quick
  Claw — so a *seeded* rust battle is byte-for-byte with node even on a **speed-tied lead** or an
  **unspecified-gender** mon. (Formerly `advance_seed_for_construction` modeled only the Quick Claw and
  the diff harness skipped speed ties; both are gone.) The bridge runs that window with logging OFF
  and re-emits the leads' switch-in ability lines afterwards, so it also RECORDS the order the two
  `runSwitch` actions resolved to (`gen3_turn0_construction_mirror_order_v1`) — at a raw-Speed TIE
  that order is the `insertChoice` PRNG draw, p2-first half the time, and re-deriving it as
  "faster-first, tie = side order" permuted the Intimidate / weather-setter block on a tied lead
  (the seed and the board stayed correct; only the emitted lines were out of order). Gated by
  `src/rust_sim/tests/turn0_construction_test.rs` + `harness/gen_sim_bridge_diff.js`.

**Forfeit parity (`gen3_bridge_forfeit_win_v1`) — was the training-wedge bug.** A `FORCELOSE
<side>` must end the battle the way Node does: Node writes `>forcelose` INTO the sim, so Showdown
runs a real `win(otherSide)` and BOTH players receive `|` + `|win|<name>` before `__END__`. The
Rust bridge used to emit a bare `__END__` with no win line, which left poke-env's `Battle.finished`
False forever — the env's next `reset()` then waited on a result that could never arrive. Because
the training seam forfeits whenever `reset()` lands mid-battle, *every* episode boundary could
wedge, which is what stalled the multi-env `--use-bridge=rust` runs (they logged episodes but
never completed a single PPO iteration). Fixed by `BridgeSession::forfeit`; pinned by
`src/rust_sim/src/bridge.rs::a_forfeit_emits_the_win_line_to_both_sides_not_a_bare_end` and gated
end-to-end by `bridge_session_fuzz_test.py --impl rust` (whose every-9th-episode forfeit-reset is
the reproducer).

**Illegal-choice parity — the two impls fail DIFFERENTLY (know this before debugging a hang).**
Showdown's `Side.emitChoiceError` (`sim/side.ts:510`) branches on whether the refusal actually
CHANGED the request: `[Unavailable choice]` comes with a **re-request** (the client recovers),
while `[Invalid choice]` comes with **nothing at all** — the client is expected to re-pick from the
request it already holds. So a node child that goes quiet mid-battle is usually **not** wedged: it
refused an illegal choice and is waiting for a legal one (`probe_illegal_choice_park.js` drives
this deterministically: `move 4` on a 2-move mon → `[Invalid choice]`, 0 requests; a legal retry
then resolves the turn immediately). Two consequences:
- **BOTH impls now bound it** (`gen3_node_bridge_reject_bound_v1`): `REJECT_STREAK_CAP = 8`
  consecutive refusals at one boundary turns the spin into a loud `__ERR__` — rust in
  `BridgeSession`, node in `local_sim_bridge.js`. The counter resets **only on a COMMITTED
  decision** (node reads `battle.inputLog.length`, the one place an ACCEPTED choice lands);
  resetting on a received re-request instead would make the cap unreachable, which is the
  subtle way to get this wrong. The cap is generous on purpose — a handful of refusals is
  legitimate (the maybe-trapped probe is a normal two-exchange round; the max streak measured
  in normal play is **1**) — it exists to make an unbounded spin diagnosable, not to police
  ordinary refusals. Gate: `node_reject_bound_integration_test.py` (4 tests: the wedge pin,
  which HANGS pre-fix; an over-eager-cap guard; the reset-condition pin; and a cap-constant
  lockstep check), revert-verified — disabling the bound leaves the wedge pin spinning out its
  full 25 s budget with no `__ERR__` while the other three still pass.
  **MEASURED not-a-regression** (the real risk of a bound like this is a FALSE trip, not a missed
  one): `bridge_session_fuzz_test --impl node` 40 episodes clean; the bridge test package 91
  passed / 6 skipped; the python unit suite 3900 passed; and a 100-battle node-vs-rust
  `gen_sim_bridge_diff --mode randbats --persistent` soak came back **100/100 ended, 0
  divergences, 0 drain timeouts** while carrying **128 `trapped:true` frames** — i.e. the
  legitimate refusal round (the maybe-trapped probe) was exercised heavily and never tripped the
  cap. The max streak seen in normal play remains 1.
  **SCOPE — the bound makes the failure BOUNDED; the wake below makes it IMMEDIATE.**
  `BridgeSession` latches the `__ERR__` into `_child_error` and raises it at the next `reset()`
  (so the run dies with the refusal text, the side, and the offending choice).
  `run_local_battles` (the eval driver) is stricter still: it raises on the `__ERR__` frame
  directly.

**A dead child WAKES an in-flight `step()`** (`gen3_bridge_child_error_wakes_step_v1`) — the
former "known gap", now closed. Latching alone only covered the NEXT `reset()`: a `step()` already
parked in `battle_queue.race_get` waited for a request the now-dead reader could never deliver, so
it sat out poke-env's watchdog first. The gap became materially more reachable once the reject
bound above made `__ERR__` a real outcome on BOTH transports.

It is closed by REUSING the mechanism poke-env already has rather than inventing a second one:
`ps_client.listen` sets `_disconnected` on an unrequested websocket close, and
`_AsyncQueue._get` / `race_get` race their gets against it, raising `ShowdownException` instead of
hanging. A dead bridge child is the SAME event — "the transport can no longer deliver" — so the
reader now fires that signal from its two fatal exits (child EOF, dispatch failure).

**The load-bearing detail, and why the obvious implementation is wrong:** `_EnvPlayer` binds its
queues to `ps_client._disconnected` at CONSTRUCTION, and `attach()` then REPLACES `ps_client` with
a `BattleStreamClient` carrying its own fresh event. Signalling the NEW client's event wakes
NOTHING — the queues still hold the ORIGINAL object. The session therefore captures the events off
the QUEUES themselves.
`bridge_session_test.py::test_child_error_wakes_a_blocked_queue_get_instead_of_hanging` pins that
by IDENTITY and is revert-verified against the plausible-but-wrong client-signalling version,
which it fails. Terminal by design and safe: the signal fires only on no-in-place-recovery paths,
while a routine `_recycle_child` CANCELS the reader and so reaches neither — confirmed by
`bridge_session_fuzz_test --impl {node,rust}`, 40 episodes each (including the every-9th-episode
forfeit-reset), both clean.
- **The impls still differ on move-INDEX validation.** Rust does not validate the move index
  against the request, so an out-of-range `move N` is accepted and the turn advances where node
  refuses it — benign in training (poke-env only sends choices drawn from the request) but it is
  why the two children diverge on a malformed driver, and why the node side is the one that could
  spin.
**Diagnostic rule: an idle bridge child means "waiting for a legal choice", so look at the last
choice the DRIVER sent, not at the child.** (Full diagnosis: `src/rust_sim/CLAUDE.md` → ROUND 30.)

The move-name/switch-species transport parity (poke-env serializes choices by move-id + species
name, e.g. `move hiddenpowerice` / `switch Salamence` — not slot numbers) is exercised by
`bridge_impl_parity_test.py` (rust integration smoke + rust-vs-node win-rate parity at `seed=None`).

**Throughput (`bridge_impl_throughput_benchmark.py`, 16-core box, idle):** rust is FASTER than
node at every scale measured, and its child is an order of magnitude smaller:

| workers | node | rust | rust/node | node RSS/child | rust RSS/child |
|---|---|---|---|---|---|
| 8  | 1852 steps/s | 2182 steps/s | **1.18×** | 225 MB | 9 MB |
| 48 (production `--n-envs`) | 1942 steps/s | 2729 steps/s | **1.41×** | 223 MB | 9 MB |

The ~25× smaller child is the bigger operational win: at `--n-envs 48` the bridge children cost
~10.7 GB under node vs ~0.4 GB under rust. (An older note recording node 798 vs rust 427 fps at 8
envs was taken on a CPU-saturated box and is superseded — trust the ratio from a same-invocation
A/B on an idle machine.)

### Single-turn damage oracle (`damage_probe.js`) — exact ground truth, no poke-env

A third bridge mode drives the **OMNISCIENT** (referee) BattleStream directly — *not* the
per-side protocol poke-env consumes. The omniscient stream reports **EXACT both-side HP**
(`|-damage|p2a: X|389/461`, not the percent a player sees), and the live `battle` object exposes
the sim's **OWN computed stats** (`storedStats`), boosts, status, item, ability, types, weather, and
side conditions. So we can construct a battle with fully-specified teams, force a move sequence, and
read the **exact** damage a hit dealt — the clean ground truth for validating the differentiable
`DamageOperator`'s gen3 physics with **zero measurement confounds** (no percent-rounding, no
stale-HP, no overkill caps that plague scraping damage from random games).

- **`damage_probe.js`**: batch request/response over stdio (like `validate_team.js`). `stdin` = one
  JSON `{scenarios:[{id, formatid, seed?, p1:[sets], p2:[sets], choices:[["p1","move 1"],…]}]}`; it
  packs each team (`Teams.pack`), runs a fresh `BattleStream` per scenario, writes the choices to the
  omniscient stream, and emits one JSON line per scenario: `{id, weather, log:[omniscient lines],
  p1:<snap>, p2:<snap>}` where each snap carries `{species, maxhp, hp, stats, boosts, status, item,
  ability, types, sideConditions}`.
- Consumer: **`agents/training/poke_env_gaps/damage_op_probe_fuzz_test.py`** — the **authoritative**
  damage-op physics gate. It stages one modifier per scenario (type/STAB/super-effective/resisted/4×/
  type+ability immunity/Thick Fat/Choice Band/type-boost item/+Atk/+SpA boosts/burn/Reflect/Light
  Screen/rain/sun/defender +Def), measures p1's final hit on p2 with exact HP, and asserts the sim's
  damage lands inside the op's band (computed from the SIM's exact stats). The random-game
  `damage_op_fuzz_test.py` is a looser broad-coverage net by comparison (its per-side percent HP is
  inherently confounded — adjudicate any real physics question in the probe).

### RL-training transport (`BridgeSession`) — the bridge as a gym env transport

`run_local_battles` is a *synchronous driver*: it owns the battle loop and pulls each decision
from a self-contained `choose_move`. That shape can't host SB3, whose action arrives from
*outside* the env via `step()`. `BridgeSession` (`bridge_session.py`) closes that gap: it makes
the bridge a **drop-in transport for poke-env's `PokeEnv`**, so the *exact same* obs / reward /
mask / wrapper stack trains with no websocket, no server, no port, no `/challenge` connection
storm.

`PokeEnv` already inverts control with its two `_EnvPlayer` agents and their
`battle_queue` / `order_queue` handshake — the websocket is only the byte transport underneath.
`attach_bridge_transport(env, battle_format=…)` swaps that transport on a freshly-built env
(built `start_listening=False`):

1.  Reassigns both `_EnvPlayer.ps_client` to a `BattleStreamClient` (sides p1 / p2).
2.  Intercepts the one battle-start seam — `agent1.battle_against` (the `/challenge` handshake
    `PokeEnv.reset()` calls) — with a coroutine that spawns the bridge subprocess, sends `START`
    with both packed teams, and launches a **background reader**.
3.  The reader mirrors poke-env's own websocket `listen()`: it reads one protocol chunk, frames
    it with the room header, and **fires** the feed as a task (never `await`s it). That is
    mandatory — `_EnvPlayer._choose_move` blocks on `order_queue` awaiting SB3's action, so
    awaiting the feed inline would stall the reader and deadlock the *other* side. It is safe
    because `_handle_message` serializes same-battle handling under a per-battle `asyncio.Lock`,
    so chunks for one battle stay strictly ordered, and the two sides use independent clients.

Two child-lifecycle modes (`attach_bridge_transport(persistent=…)`): **persistent** (default)
reuses ONE long-lived Node child per env across every episode (a fresh `START` rebuilds a clean
`BattleStream`); **spawn-per-battle** spawns a fresh child per battle. Persistent is the win — a
single-env transport-latency A/B (`bridge_vs_websocket_latency_benchmark.py`, RandomPlayer
opponent, no GPU) measured **~13.0 ms/step websocket → ~6.1 ms/step persistent bridge (~2.1×)**,
while spawn-per-battle was only ~11.3 ms/step: re-loading the Showdown sim into a fresh Node
process every episode eats nearly all the savings, so reusing the child is what unlocks the gain
(matching issue #907's "reset/startup overhead is the bottleneck"). It is **flag-guarded**:
`python -m main.launcher … --use-showdown-bridge` (or `train_rl_agent.py --use-showdown-bridge`),
default off (websocket). Guarded by `bridge_session_test.py` (transport-swap contract, no server)
and `bridge_session_integration_test.py` (a real `Gen3Env` plays full episodes over the bridge).

**Persistent-child lifecycle (two rules, both guarded):**
- **A dead child CRASHES the env, no in-place recovery.** If the Node child exits mid-run, lost /
  inconsistent battle state means resuming could feed PPO a corrupted transition — so the reader
  latches `_child_crashed` on stdout EOF and the next `reset()` raises (the launcher restarts from
  checkpoint). Same crash-over-corruption rule as the trainee's stale-decision path.
- **A healthy child is RECYCLED every `recycle_every` battles — a backstop, not a routine need.**
  `bridge_heap_growth_benchmark.py` measured a child's RSS **flat**: ~189 MB fresh → a one-time
  ~+36 MB V8 warmup → **~229 MB with ~0 growth over thousands of battles** (V8 GC reclaims the
  per-battle `BattleStream`). At production scale a child plays only ~2150 battles in the
  launcher's 3h restart window, so the default `recycle_every=5000` **never fires under the
  launcher** (the 3h restart owns the lifecycle); it only caps marathon / no-launcher direct runs.

**Eval rides the same flag, via the *synchronous* driver, not `BridgeSession`.** Eval is a pure
synchronous-decision matchup (a greedy trainee vs a bot/sentinel — no SB3-supplied action), so it
doesn't need the inversion-of-control machinery: the eval worker's `_play_unit` just calls
`run_local_battles` instead of `battle_against` when `use_showdown_bridge` is set (players built
`start_listening=False`). The flag threads as a `use_showdown_bridge` config key through
`PerOpponentEvalCallback` / `SelfPlayCallback` → `eval_worker`, plus the end-of-training
`evaluate_model_random`. So `--use-showdown-bridge` makes a whole run — training **and** eval —
need no Showdown server.

### Battle reconstruction (capture + offline replay / re-roll)

Every bridge battle is **fully reconstructable**. At battle end the child emits a
`__RECON__ <b64 json>` frame (just before `__END__`) carrying the full-information
reconstruction record: the **resolved** PRNG seed (the sim mints a fresh random one per battle
when none is passed — eval is reproducible-after-the-fact without pinning a seed), both packed
teams, the sim's own `inputLog`, and the raw `commands` the child processed. The raw command log
is kept *in addition to* `inputLog` because the sim logs only **committed** choices — a refused
`[Unavailable choice]` maybe-trapped probe never reaches `inputLog`, but its `|error|` +
re-request round *is* part of the protocol the agent saw, and replaying the raw commands
regenerates it exactly.

`reconstruction.py` owns the layer:

- **Capture join** — the `__RECON__` frame arrives *after* the `|win|` chunks (when the eval
  forensic trace is already written), so the two sides meet in a bounded registry keyed by battle
  tag: the demux calls `offer_record`, the forensic writer calls `register_trace_prefix`, and
  whichever lands second writes `<prefix>_reconstruction.json` next to the trace. (`BridgeSession`
  instead keeps a single-slot `last_recon` — training persists no traces.)
- **`replay_battle(record)`** — re-runs the battle verbatim (`replay_driver.js`, batch
  JSON-over-stdio, no server) and returns the regenerated per-side protocol chunks +
  the final omniscient outcome. Byte-identical to the live streams modulo `|t:|` wall-clock
  lines (state/obs-invisible; in poke-env's `MESSAGES_TO_IGNORE`).
- **`reroll_turn(record, t, seeds=…)`** — reconstructs to the start of turn `t`, then resolves
  that one turn under each fresh seed by swapping `battle.prng` in place (every die routes
  through it). Each side's start-of-turn action source is independently `recorded` / `random` /
  an explicit choice string; mid-turn follow-ups in re-rolled timelines (forced switches that the
  original timeline may not contain) use a configurable `followup` policy.
- **`reroll_many(record, t, arms)`** — the BATCHED form: resolves N independent ARMS (each its own
  `{p1_action, p2_action, seed, label}`, with the EXACT per-side action semantics of `reroll_turn`)
  of turn `t` in **one Node process**. The dominant per-re-roll cost is the **~677 ms Node-spawn /
  pokemon-showdown `require`**, NOT the in-process `buildToTurn` (~26 ms warm), so a candidate sweep
  (the prober's one-ply lookahead) pays it ONCE instead of once per candidate — measured **~9×** on a
  full 9-legal-action sweep (5.5 s → 0.62 s). Each arm runs in its own fresh session, so its suffix
  chunks are **byte-identical** to the same single `reroll_turn` (modulo `|t:|`) and the materialized
  successor obs is **bit-for-bit identical** — pinned by `reroll_many_parity_fuzz_test.py`. (This is
  why `State.serializeBattle`/`deserializeBattle` is NOT wired here: the data shows the spawn dominates,
  not the rebuild, so cloning a battle snapshot wouldn't move the needle for these probes — batching
  does. `serializeBattle` is the lever for an in-process *tree* search, e.g. MCTS, which is out of scope.)

**The one-sided / omniscient wall (hard rule).** The record holds the opponent's team and the
dice — referee-view data. It exists only at this bridge layer and in the separate
`*_reconstruction.json` artifact; the obs pipeline never reads it. Offline obs come from
`agents.training.obs_materializer`, which is fed **only the per-side chunks** these primitives
regenerate and replays them through the real encoder (rebuilding tracker state). The round-trip
guarantee — materialized obs == the live `states.npz` rows **bit-for-bit** — is enforced by
`agents/training/obs_roundtrip_fuzz_test.py`; replay/re-roll invariants by
`reconstruction_fuzz_test.py`; the registry by `reconstruction_test.py`.

### Counterfactual replay-to-end (`counterfactual.py`)

Where `reroll_turn` re-rolls a SINGLE turn, **`replay_counterfactual`** picks up a recorded battle at
turn T, substitutes a different move for one side, and **plays the rest LIVE to a win/loss** — the
prober's "could it have won if it hadn't choked this turn?" (Feature 2). It reuses the
`run_local_battles` driver wholesale: both players are real poke-env players whose `choose_move` is
**scripted** (`install_scripted_prefix`) to replay the recorded commands until the divergence, then
handed back to the live policy. Faithful prefix: `START` uses the record's resolved seed + both packed
teams (turns 1..T-1 reproduce the real board), and each scripted `Gen3Player` decision runs
`embed_battle` + `tracker.advance(recorded_idx)` — the recorded index recovered by inverting the
recorded choice string through the real action mapper — so the **post-divergence turn-history stays
faithful**. At turn T our side plays the substitute and goes live; the opponent plays its recorded
turn-T move (it couldn't have reacted on the same turn) and goes live from T+1. The caller builds the
players (a greedy trainee + the RELOADED real opponent — a reproducible bot, a sentinel/stable
checkpoint, or a flagged self-model fallback; orchestrated by `src/main/prober/replay.py`).
`divergence_turn=None` scripts the whole game (the full-replay correctness oracle).
**Monte-Carlo**: `post_t_seed` (threaded into `START` as `resumeReseed: {turn, seed}`) swaps the sim
PRNG at the START of the divergence turn (mirroring `replay_driver.js`'s swap, but inside the live
`local_sim_bridge.js`), so the prefix keeps the recorded dice while each rollout resamples the
post-divergence dice → a win-rate ± CI. Faithfulness is proven by `counterfactual_fuzz_test.py`: a
full scripted replay reproduces the recorded **winner** AND the recorded one-sided obs **bit-for-bit**
(the `obs_roundtrip` guarantee carried through the live `run_local_battles` path), and the reseed keeps
the prefix fixed while varying the continuation. (`run_local_battles(..., start_extra=…)` is the generic
seam that merges extra `START` fields like `resumeReseed`.) For a human-readable **play-by-play**,
`run_local_battles(..., chunk_sink=[])` accumulates every `(side, chunk)` the bridge emits, and
`counterfactual.summarize_trajectory(side, sink)` parses OUR one-sided protocol into a per-turn
`{turn, events}` log (moves / switches / damage / faints / crits / status / win) — so a recovered
counterfactual win reads as an actual move-by-move line (`replay_counterfactual(..., capture_trajectory=True)`
→ the prober's `--narrate` / TUI `C`).

### Warm clone-and-branch search-server (`search_driver.js` + `search_session.py`)

Where `reroll_turn` / `reroll_many` rebuild the battle from turn 1 per call (O(commands-to-T)), a
multi-ply SEARCH (the prober's `better_line` beam) must branch a TREE from any explored node — so
re-replaying the prefix per node is infeasible at depth. The search-server clones a **mid-battle state
in-process** via Showdown's `State.serializeBattle` / `deserializeBattle` (`dist/sim/state.js`): a
verified round-trip of a battle paused with both move requests open (deserialize rebuilds the open
requests via `getRequests`; PRNG continuity restored from the live counter), at **~1.7 ms/clone —
~16× cheaper than the ~26 ms warm `buildToTurn`, and CONSTANT in depth**.

- **`search_driver.js`** — a WARM, persistent Node process (vs `replay_driver.js`'s one-shot batch)
  holding a node-snapshot cache. `open_root {record, turn}` reconstructs to turn T (via the shared
  `replay_kernels.js`), serializes the root, and returns the request/team-complete prefix chunks.
  `expand_many {arms}` clones a parent node, applies one joint turn (our action + the opponent's, via
  the same `resolveTurn`/`resolveTurnExact` kernels), and re-serializes the child. A deserialized
  battle can't re-emit the historical `|request|` lines (requests are out-of-band, not in `battle.log`),
  so each expand returns this ply's one-sided **suffix** (flushed-prefix baseline + the new turn) and
  the Python caller composes `root-prefix + each ply's suffix` — the same `(prefix + suffix)` shape
  `reroll_many` produces. `recorded_exact` (root only) reproduces the realized turn for the
  `value_crn` anchor. The per-side chunk splitting is REUSED verbatim (a deserialized battle is
  re-attached to a fresh `BattleStream` and `restart()`-ed with the same `send` wiring).
- **`search_session.py`** (`SearchSession`) — the Python wrapper: one process per `better_line` call
  (context-managed), a synchronous request→one-line-response protocol over a background-drained queue
  (a wedged child fails ONE call, never hangs the prober). `open_root` / `expand_many` / `close`.
- **`replay_kernels.js`** — the shared sim kernels (`buildSession` / `buildToTurn` / `resolveTurn` /
  `resolveTurnExact` / `recordedQueues` / `randomChoice` / `outcomeOf` / …) lifted out of
  `replay_driver.js` so the replay/re-roll path and the search-server use ONE implementation (no
  drift; the trusted reroll path is byte-for-byte unchanged).

**Faithfulness (a NEW path, so proven not asserted):** `search_clone_parity_fuzz_test.py` (real
bridge, no server) asserts over many battles that a depth-1 clone's successor obs equals the
`reroll_many` obs **bit-for-bit**, that the `recorded_exact` clone reproduces the recorded `states.npz`
next obs (the `value_crn` anchor), and that a depth-2 clone composes a valid chain. The omniscient
clone/outcome/teams/dice drive only the opponent + the dice — never the obs encoder (the one-sided
wall). Consumed by `main.prober.better_line`; see `src/main/prober/CLAUDE.md`.

**Reading the teams for review** — `record.team_details(side)` / `decode_packed_team(packed)`
is THE one decode home (moves, EVs, IVs, nature, item, ability, level; omission-defaults
applied; ids resolved through the sim's alias table — a pool export can say `wisp`, everything
downstream speaks `willowisp`). It delegates to poke-env's `parse_packed_team`, never a
reimplementation, and is validated **field-by-field against the sim's own `Teams.unpack` over
the ENTIRE team pool** by `packed_team_decode_integration_test.py` (719 teams / 4314 mons —
which caught the alias case on its first run). Replay/re-roll never decode at all: the packed
strings go back to the sim verbatim, so counterfactuals always run with the true hidden details
(exact Hidden Power types, speed stats, damage ranges).

## Why use a Bridge?
- **Serverless**: No need to start or manage a Pokémon Showdown server process.
- **No contention**: Each call/battle is fully isolated — no shared server lifecycle, no
  username collisions, no port. Multiple fuzz tests can run at once without fighting.
  (This means no *lifecycle* contention. It does NOT mean no *CPU* contention — see
  **Timeouts and a busy box** below.)
- **Speed**: Local function calls beat websocket round-trips (the fuzz suite is *faster*
  on the bridge than on the server).
- **Accuracy**: It uses the *exact* same Showdown code as the live server (same `deps/` files).
- **Reproducible**: the battle bridge accepts a fixed PRNG seed.

## Timeouts and a busy box (`gen3_contention_robust_timeouts_v1`)

The bridge removes the *server*, not the *CPU*. This box normally carries a production training
run, so every wall-clock bound here is partly a measurement of the load average — a healthy battle
that takes ~2 s idle takes ~6 s beside a 48-env run, for reasons that have nothing to do with the
sim.

All three bounds on this path are therefore scaled by measured contention
(`utils.contention.scale_timeout`, read at CALL time so the factor tracks load as it develops;
factor = `max(1, loadavg/cpus)` clamped to 12x, so an **idle box is exactly 1.0 and nothing
changes**):

| bound | baseline | where |
|---|---|---|
| per-battle | `_PER_BATTLE_TIMEOUT` 180 s (parity test overrides to 20 s) | `local_battle_runner._per_battle_timeout()` |
| previous battle's `__END__` before child reuse | `_BATTLE_END_TIMEOUT` 180 s | `bridge_session` |
| silent-stall watchdog | `_RACE_GET_TIMEOUT_S` 120 s (`GEN3_RACE_GET_TIMEOUT_S`) | `poke_env.environment.env._race_get_timeout()` |

Two rules this encodes, both learned the hard way:

1. **A timeout is never a semantic outcome.** `bridge_impl_parity_test` used to fold a per-battle
   timeout into its "unmodeled move" SKIP bucket; beside a live trainer that turned 39/40 starved
   battles into a clean-looking pass that blamed the Rust port's move coverage for the box's load.
   Timeouts now have their own counter, and >25% timed out is INCONCLUSIVE, not a verdict.
2. **Bound the IDLE gap, not the total duration,** wherever progress is observable
   (`contention.ProgressDeadline`). Contention stretches how long a battle takes; only a genuine
   wedge stops the protocol lines arriving. A total-duration cap cannot tell those apart, so the
   only way to stop it flaking is to raise it until it stops catching the real bug too.

Every timeout raised on this path appends `describe_contention()` — the load average plus the
`ps -eo pcpu,pid,args --sort=-pcpu | head` command — so a starved failure says so itself.
`GEN3AI_TIMEOUT_SCALE=N` forces the factor when you already know the regime.

## Usage

### Team validation
```python
from src.utils.bridge.team_validator import validate_team_locally

result = validate_team_locally("gen3ou", team_text)
if result["valid"]:
    print("Team is valid!")
else:
    print(f"Errors: {result['errors']}")
```

### Local battle (used by the `*_fuzz_test.py` suite)
```python
from utils.bridge.local_battle_runner import run_local_battles

# Players must be built with start_listening=False so no websocket is opened.
await run_local_battles(my_player, opponent, n_battles=40)   # no `npm run showdown`
```

### RL-training transport (used by `train_rl_agent.py --use-bridge {node,rust}`)
```python
from utils.bridge.bridge_session import attach_bridge_transport

# Build the env with start_listening=False, then swap in the bridge transport.
env = Gen3Env(mappings, battle_format="gen3ou", team=teambuilder, start_listening=False)
attach_bridge_transport(env, battle_format="gen3ou", impl="node")  # or impl="rust"
# env now trains with no websocket / server — everything above the transport is unchanged.
```
`train_rl_agent.py --use-bridge {off,node,rust}` (default `off` = websocket) selects the transport
for BOTH training and eval; `--use-showdown-bridge` is a DEPRECATED back-compat alias for
`--use-bridge=node` (the two must agree if both are passed). `run_local_battles(..., impl=…)` takes
the same impl for the eval driver.
