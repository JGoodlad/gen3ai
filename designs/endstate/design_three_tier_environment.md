# Design — The three-tier environment: a Rust battle core, one inference tier, and the retirement of poke-env

**Status: END STATE, being built (updated 2026-09-27).** Authored 2026-09-22 at the owner's request,
after a month in which every search-cost crossing (the Rust bridge, the one-sided view, the
materializer flip, the one-fork-per-decision landing) bought a large per-op speedup and a small
end-to-end one. The owner licensed the build on 2026-09-23 as the RUST CORE PROGRAM, on unification
grounds; [`program_rust_core.md`](program_rust_core.md) is the plan, and this document stays the
statement of the end state and its "why". 🚨 **ALWAYS-CURRENT (owner, 2026-09-27)**: see
[`README.md`](README.md). Where the build overrode a part of this design, the part is marked
**SUPERSEDED** in place, with its pointer, and its reasoning is kept.

**Companion:** [`design_model_management.md`](design_model_management.md) — the model-side end state
(the registry, the pool, the inference tier's catalogue). This document owns the environment side.

**Where it stands (2026-09-27).** Detail and evidence: `program_rust_core.md` §2 and the ledger.

| piece | status |
|---|---|
| Tier 1a–1c: typed events at the source (M1), `BattleVersion` + `present()` + legality (M2), trackers + `TurnDelta` + the win-indicator reward on the version (M3), the whole observation in Rust (M4) | **BUILT** 2026-09-23 → 09-24, each at zero parity divergences (ledger 2026-09-23 *M1 BUILT*, *M2 LANDED*; 2026-09-24 *M3 CLOSED*, *M4 CLOSED*) |
| The cutover (M6): training reads the Rust core's observation (`--obs-source core`, the default on the rust bridge) | **DONE** `ac0b6469`, 2026-09-25 (owner); labels, reward, the Python tracker fold and the mapper still run in Python |
| Deletion pass, part 1: search is core-only; the port's one-sided view projection deleted; the shaped reward path deleted | **DONE** `43712881`, `97a30387`, `2c7e6acb`, `e3ef16db` (2026-09-26). Five manifest rows blocked on M5 / M7 (TASK_BACKLOG T10) |
| M5 — N envs per process (§3.6) | **Phase A DONE** 2026-09-26: ONE env core with TWO thin front ends (FFI and a shared-memory process), the default per consumer set by crash isolation, 12 build lanes. Lane 0 scheduled (TASK_BACKLOG T1) |
| Tier 2 — the inference service (§4) | **scheduled with Lane 0** (TASK_BACKLOG T2); shape re-stated in §4.5 |
| Memory-triggered restarts; allocator discipline | **proposed / in Lane 0** — §3.7 |
| M7 — the ladder client on the core (§6's `play.py` row) | **proposed** (TASK_BACKLOG T9) |

---

## 0. The one-paragraph version

Today a battle state reaches the network by a chain of re-derivations: the Rust simulator produces
protocol text; poke-env parses it back into objects; the event-sourced layer folds those objects into
an event log; read-models fold the log; per-decision trackers fold the read-models; the encoder reads
the trackers. Every link was the cheapest next Lego at the time, and every link re-derives a fact the
simulator already had. The end state has **three tiers**: (1) a **Rust battle core** that owns the
omniscient state, the per-side view, the typed event history and the observation encoder, as one
persistent (structurally shared) structure buildable either from the simulator's own transitions or
from a parsed protocol log; (2) an **inference tier**, one process owning every loaded network, taking
observation rows from any caller into a bucketed, padded, timer-flushed batch and returning actions or
values; (3) the **learner**, which is PPO as it is today, receiving rollouts. Search becomes an
in-process tree over version handles with no serialization on the hot path. The ladder client becomes
"parse the log into the same structure, encode, act" — one path, no skew. poke-env retires from the
production path and survives as a parity oracle (until T27 P6, 2026-10-08, when the oracle was retired too: nothing of ours imports poke-env and the Python battle layer is deleted).

---

## 1. What was true on 2026-09-22, with the numbers that motivated this

Measured facts, each with its record, as of authoring (kept as the argument; the "Where it stands"
table above is the present). §2 onward is the design.

| fact | number | record |
|---|---|---|
| A training step is gradient-bound | ~89 % gradient, ~11 % rollout at the production shape | `designs/ops/training_runbook.md` (compiled trainer) |
| The rollout is latency-bound, not compute-bound | stock vec-env is a barrier; GPU ~86 % idle during collection; `--async-rollout` +14–20 % | runbook (async rollout) |
| Each env is a process holding its own opponent | opponent forward compiled, **B = 1, CPU**, 6.4 → 1.0 ms; ~4 % of a step after the compile | runbook (compiled opponents) |
| The per-op Rust speedup does not reach the decision | driver per-op 7–20×, end-to-end `better_line` **1.89×** | memory `project_rust_search_driver` |
| The view's raw encode speedup does not reach the decision | encode **47×** (B = 1); search-level **1.35×** (B = 33), 0.98× (B = 1) | ledger 2026-09-20 · *THE MATERIALIZER FLIP LANDED* |
| Where a searched decision's wall goes on the view road (exclusive) | `expand_many` + JSON 15.7 % · tracker fork 10.9 % · D10 fallback 9.8 % · encode 9.5 % · `open_root` 7.8 % · prefix replay 7.4 % · read-models 6.7 % · Python glue 12.1 % | `measurements/search_profile_2026-09-22/` |
| After one-fork-per-decision + lazy tokens | view road **1.33×** wide / **1.66×** B = 1 (interleaved A/B) | same record |
| The rollout-leaf battery that would decide search-as-teacher | **1,222 CPU-h** at R = 4 for 400 pairs — unaffordable | ledger 2026-09-19 · *ROLLOUT LEAF* |
| Half of the read-model is not sim state | `LiveView`'s `moves` (sighting counts), `volatiles` (a fold with drop rules), `status_counter`, `protect_counter`, `revealed`, team order, `item`, `consumed_item`, `ability` — each a poke-env PRESENTATION rule | `designs/rust_sim/one_sided_view.md` §2 |
| The event log is load-bearing | reward (`TurnDelta` fold), intent labels, belief encoders, trackers, the prober all read it; the diff-based detective was retired for it | `src/agents/battle/CLAUDE.md` |
| A third party's parser can be wrong where ours is right | Metamon's upstream poke-env drops Baton Pass boosts (our fork fixed it 2026-08-23) | `measurements/metamon_obs_faithfulness_2026-09-22/` |

The pattern across the first six rows is the signature of a bottom-up-assembled path: each crossing
moves the cost to the next Lego in line rather than removing it. The last three rows are why the
rewrite is not a simple port — the **rules** poke-env applies are the observation's definition, and
the reward was trained on poke-env's reading, not on the simulator's truth.

---

## 2. The three tiers

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  TIER 3 · LEARNER  (unchanged: MaskablePPO, dual-head policy, the callbacks)  │
│     receives (obs, mask, action, logp, value, reward) columns per env         │
└───────────────▲──────────────────────────────────────────────────────────────┘
                │ rollout buffers (shared memory, one column per env)
┌───────────────┴──────────────────────────────────────────────────────────────┐
│  TIER 2 · INFERENCE  (one process; owns every loaded network)                 │
│     score(model_id, rows[B, obs_dim], masks[B, 11]) -> actions / logits / V   │
│     bucketed batch · padded to bucket · flush on full-or-timer · GPU          │
│     callers: trainee rollout, self-play pool, sentinels, bots-as-models,      │
│              search leaf, cf label factory, eval, prober, anchors, ladder     │
└───────────────▲──────────────────────────────────────────────────────────────┘
                │ obs rows + masks (in); actions / values (out)
┌───────────────┴──────────────────────────────────────────────────────────────┐
│  TIER 1 · BATTLE CORE  (Rust; N envs per process, no per-env process)         │
│     BattleVersion: omniscient board · per-side OneSidedView · typed EventLog  │
│       persistent: fork = handle clone; history = parent chain                 │
│     built FROM the sim's own transitions  OR  FROM a parsed protocol log      │
│     methods: legal_actions(side) · encode(side, &mut [f32; 2501]) ·          │
│              trackers(side) · turn_delta(side) · successors(side, k)          │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Tier boundaries are data, not calls.** Tier 1 → Tier 2 is a float row and a mask; Tier 2 → Tier 1
is an action index. Tier 1 → Tier 3 is a rollout column. No tier imports another's types beyond
those three shapes. That is what makes each tier replaceable and each tier testable alone.

---

## 3. Tier 1 — the Rust battle core

### 3.1 The structure: `BattleVersion`, persistent

A battle is a chain of immutable versions. Each version holds:

- **the omniscient board** — the simulator's `BattleState` after this transition (the port already
  has it; today it is discarded after the protocol is emitted);
- **`parent: Option<Arc<BattleVersion>>`** — the previous version;
- **`events: Vec<BattleEvent>`** — the typed events that produced this version from its parent,
  emitted BY THE SIMULATOR at the transition, with attribution known at the source (which side's
  move is resolving, whether damage is residual/hazard/ability, whether a switch is a drag, whether
  Sleep Talk delegated) — not inferred from protocol line order;
- **`view: [OnceCell<OneSidedView>; 2]`** — the per-side presentation, computed on demand and
  memoized, because search asks for it once per node and training once per decision;
- **`turn`, `request: [Option<RawRequest>; 2]`** — the raw `|request|` facts per side, which is
  what legality is derived from (the port already sends raw request bytes rather than parsed
  legality, by contract).

**Fork is `Arc::clone` on the handle.** A K-world search forks K children from one version without
copying the board. A prefix is a pointer to a version; undo holds the older pointer; redo the newer.
The 10.5 ms tracker deep-copy the profile found (now 0.5 ms by a pinned-pickle thaw) goes to a
pointer copy. The prefix replay (7.4 % exclusive, 13.7 % inclusive, 53.5 % at B = 1) goes away
entirely because the prefix is never replayed — it is the parent chain.

**History is the parent chain with per-side cursors.** "What happened on turn t − 3" is a walk up
three parents, not a fold over a log. `TurnView`'s per-turn facts (`move_order`, `we_moved_first`,
`both_attacked`, `someone_fainted`, `damage_on(species, side)`) become methods over one version's
`events`. `TurnDelta.build_from_events` becomes `version.turn_delta(side)`.

**Why persistent and not an LSM tree.** An LSM tree is a write-optimized on-disk key-value store; it
solves a different problem. What search needs is *structural sharing across versions with O(1)
fork*, which is what persistent (immutable, functional) data structures provide. The battle board is
small enough that even a naive struct copy is cheap in Rust; the persistent structure earns its keep
on the **history**, where the spine is shared and a fork does not duplicate the past.

### 3.2 Two entry points, one state — and the skew-proof

`BattleVersion` is buildable from two sources:

1. **The simulator's own transitions** (`step(version, choices) -> version`) — training, search,
   eval, offline meters.
2. **A parsed protocol log** (`parse(lines) -> Vec<BattleVersion>`) — the ladder client, replay
   analysis, the prober, anchors when a third party drives the server.

Both produce the same type. **The gate is that (2) reproduces (1) on the simulator's own log**:
emit protocol from `step`, parse it with `parse`, and assert version-by-version equality of the
board, the events and both views. That is a stronger and simpler gate than today's, where two
codebases (the port's emitter and poke-env's parser) agree only because a differential fuzzer says
so. The ladder client then cannot skew from training by construction — it is the same struct.

> **DECIDED since (owner, 2026-09-23; `program_rust_core.md` §6c):** the OBSERVATION always comes
> through entry point 2, in training too: per-side protocol text → `parse` → the reading → the view →
> encode, so the ladder's parser runs on every training decision. Only search's successors keep the
> typed-at-source shortcut, licensed by M1's `parse(emit(step)) == step` gate. Cost measured at
> 21.6–30.8 µs per decision.

### 3.3 The presentation rules — the part that is NOT a port

`designs/rust_sim/one_sided_view.md` §2 is the standing finding: half of `LivePokemon` is not sim
state, it is poke-env's fold of the protocol. The contract that document forced — *the port emits
sim facts and raw protocol facts; every poke-env presentation rule is applied in Python* — was the
right split for a crossing. **For the end state the rules cross too, and they cross as RULES, not
as a re-derivation:**

- `OneSidedView` is computed from the omniscient board + this side's `events` by an explicit
  `present(board, events, side) -> OneSidedView` function *(**SUPERSEDED in its signature** by M2,
  2026-09-23: `present()` is built from ONE side's stream with NO board parameter, which is what makes
  it usable on a third party's server; the board is used only by the core's audit)* whose every rule is named, tested
  individually, and pinned to the poke-env line it mirrors (the doc already lists them: sighting
  counts doubled against Pressure; the volatile fold's drop rules; the status-counter transitions;
  the protect-counter as a stall-move count; `revealed` set by the `|switch|` line; team order from
  the first `|request|`; `item` only from `|-item|`/`|-enditem|`/`[from] item:`; `consumed_item`'s
  both-sides clearing; the two-slot `ability` with Trace semantics).
- **The reward was trained on poke-env's reading, so the port must reproduce the reading, not the
  truth**, until a model is retrained on the truth. `present()` is where that choice lives, one rule
  at a time, each behind a flag that defaults to "poke-env's reading" and can be flipped to "the
  sim's truth" for a retrain. Today the choice is implicit and unflippable.
  *(**What happened instead**, Phase 0 and M2, 2026-09-23/24: the core's event is a SUPERSET, the
  truth fields plus the reading projection, with the truth read by nothing until a retrain chooses it.
  Where the reading was simply WRONG, the fix went into the fork rather than behind a flag: three
  poke-env reading bugs, R1–R3, fixed at the training-input boundary `c97358e8`, `gen3_pe_reading_fixes_v1`.)*
- **Third-party parsers are the reason to own this.** Metamon's poke-env drops Baton Pass boosts;
  ours does not; a future opponent's will differ somewhere else. When our view is `present()` over
  the sim's truth, a third party's divergence is a diff against a known-correct reference, not a
  mystery between two parsers.

### 3.4 The event log — kept as the spine, emitted at the source

The log is load-bearing (§1) and stays the primary structure, not a derived one. The change is
WHERE it is produced: formerly (before T27 P6) `Gen3Battle.parse_message` classified each protocol line and appends a
`BattleEvent` with attribution resolved before poke-env mutates state; in the end state the
simulator appends the event at the transition, where attribution is a fact rather than an
inference. The `EventKind` / `MESSAGE_POLICY` / `EVENT_VALUE_KEYS` / `EVENT_OPTIONAL_KEYS` schema
(`gen3_event_value_schema_v1`) is preserved verbatim — consumers read absent keys as absent and that
contract survives the crossing — and the conservation invariant (no line silently dropped) becomes
"every sim transition emits ≥ 1 event or is declared silent".

**Hazard, stated:** the log parser (§3.2 entry 2) must produce the SAME attribution from protocol
text that the simulator produces from its transition. Where Showdown's protocol is ambiguous (the
`[of]` redirection under Damp, Sleep Talk delegation, the `[miss]` suffix synthetics — all named in
the contract doc) the parser reproduces poke-env's disambiguation rule, and the §3.2 gate is what
proves it. This is the single hardest piece of the rewrite and it is where the parity oracle earns
its keep.

### 3.5 Encoder, legality, trackers — methods on a version

- `encode(side, out: &mut [f32; OBS_DIM])` writes the 2,501-dim vector into a caller-provided
  buffer (a rollout column, a search batch row). Layout is `Gen3ObservationEncoder.get_layout()`,
  generated into Rust from the same constants (`agents/observation/constants.py` is the single
  source; a build step renders the Rust table and `arch_tables_test`-style pins keep them equal).
  **The encoder is the LAST thing to cross** (§7) because it carries the most rules and the
  byte-parity burden is largest.
- `legal_actions(side)` derives from the raw `|request|` exactly as `LegalActions` does today; the
  mask is the 11-dim action mask.
- `trackers(side)` — the per-decision trackers (`EpisodeTracker`'s `record_context` /
  `advance_window`, the choice-band, sleep and wish beliefs) become fields on the version, folded
  from `events` at construction; a fork inherits them by sharing. This is what removes the tracker
  fork from the search profile.
- `successors(side, k)` — materialize k successors: fork, step, present, encode. This is
  `expand_many` + `view_successor` + the materializer as one in-process call, with no JSON.

### 3.6 N envs per process

An env is a `BattleVersion` handle plus a team pair plus an opponent id — a struct, not a process.
One Rust process steps hundreds in a tight loop, writes each one's obs row into the rollout buffer,
and hands the batch to Tier 2. *(**Refined by M5 Phase A**, 2026-09-26, owner + benchmark: ONE env
core stepping N battles on its own worker threads into caller-owned columns, with TWO thin front ends
over one `core::dispatch`: an FFI `cdylib` and a separate env process over `/dev/shm`. They measured
equal at N = 48 (proc/FFI 0.994 [0.943, 1.026]) and byte-identical, so the default is set per
consumer by crash isolation: PROCESS where the Python host holds state a core abort must not destroy
(training rollout, in-trainer eval), FFI where the host is disposable (search, probes, meters).
Rust never runs a network. `program_rust_core.md` §2 M5.)* The forkserver, the per-env bridge child, `SubprocVecEnv` and the
async vec-env all retire; the async scheduler's *idea* (forward whichever envs are ready) survives
as "the batch is whatever is ready at the flush".

### 3.7 Long-lived processes: memory-triggered restarts and allocator discipline (added 2026-09-27)

N envs in one long-lived process make memory growth a first-class failure mode. Two rows carry it
(TASK_BACKLOG, 2026-09-27):
- **Allocator discipline** (in M5 Lane 0, T1): a global allocator (mimalloc or jemalloc) and per-env
  buffers REUSED across episodes rather than reallocated. *(Rationale not recorded beyond the row.)*
- **Memory-triggered restarts** (T7, proposed): the launcher restarts on RSS over a baseline, with a
  ~24 h backstop, in place of restarts on a fixed clock; an env child respawns without the learner.
  The recorded reason is fewer killed games (a restart ends the games in flight). The measurement
  behind it: the core-obs burn-in `ai_v13_33_core_burnin` held memory flat over five restarts across
  31M+ steps (ledger 2026-09-26 *NEW LINEAGE … GO*).

---

## 4. Tier 2 — the inference tier

### 4.1 Why it exists

Six consumers of "run a network on a board" were built at the point each was needed, on whatever
batch shape was in hand: trainee rollout (GPU, N ready envs), self-play opponents (CPU, B = 1, one
per env process), sentinels and bots (CPU, one game at a time), the search leaf (its own
materializer + forward), the cf label factory (its own batches), anchors/ladder/prober (B = 1,
client-shaped). None can share a GPU call because none shares a queue. Each was measured alone and
each is individually fine; the sum is the box's CPU budget, which is exactly what limits envs in
flight, agents in parallel and battery size.

### 4.2 The service

One process. It owns a **catalogue** of loaded networks keyed by model id (the trainee's current
weights, every pool snapshot, every sentinel, every baseline the registry names — see the companion
doc for how the catalogue is populated and evicted). It exposes one call:

```
score(model_id, rows: [B, OBS_DIM] f32, masks: [B, 11] bool, mode: {sample, greedy, value})
  -> actions [B] | logits [B, 11] | values [B]
```

Requests arrive from any caller over shared memory (obs rows are written in place; the request is
an index range). The service **buckets** by model id, **pads** each bucket to a fixed batch size
from a small set (e.g. 32 / 128 / 512 — compiled graphs want fixed shapes), **masks** padded rows out
of sampling, and **flushes** a bucket on *full* or on *oldest-request-age > T*. T is a few
milliseconds in training (nobody waits but a rollout buffer) and sub-millisecond in search.

**Padding is standard and near-free.** A GPU forward on this network at B = 40 costs about what it
costs at B = 64; the padded rows' share of the forward is the overhead, and it is small when the
batch is large and the network is small. Serious serving systems do exactly this (bucketed shapes,
padded batches, a timer-bounded flush). What padding cannot fix is a straggler whose row has not
arrived — the flush timer bounds that tail, and Tier 1's N-per-process stepping shrinks it.

### 4.3 What it changes, in order of size

1. ~~**CPU inference goes to near zero across all six consumers at once** — sixty-four opponents
   stop costing sixty-four cores of dispatch-bound B = 1 forwards.~~ **SUPERSEDED by Phase 0
   (2026-09-23, `program_rust_core.md` §0):** a compiled B = 1 opponent forward costs 2.4–4.6 ms, which
   at the live arm's rate is **0.4–2.1 cores of 16**, not sixty-four. Tier 2's case is unification, the
   batched search leaf, and that N envs per process cannot exist without it.
2. **Batch size becomes a throughput knob with latency governed by one timer**, instead of six
   accidental settings.
3. **The env simplifies to a simulator plus an encoder**, which is the shape Tier 1 wants to be.
4. **Search inherits a batched leaf for free**: successors of one decision are a batch by
   construction, and a search-as-teacher's confirm rollouts are a batch across candidates.

### 4.4 What it does NOT change

Throughput of the gradient step (89 % of a training step). This tier is the enabler for Tier 1's
scale and for search; on its own it moves steps/hour by a few percent and frees cores.

### 4.5 The service as scheduled (2026-09-27, TASK_BACKLOG T2)

The shape §4.2 sketched, made concrete for the build:
- **Fixed GPU weight SLOTS**: same-architecture networks (trainee, pool opponents, sentinels) held as
  stacked weights in fixed slots, and every request **tagged with its slot**, so one forward can
  serve many models. This replaces §4.2's catalogue keyed by an open set of model ids.
- **Fixed-width buckets** (or a compiled graph with dynamic shapes), in place of a menu of bucket sizes.
- **Priority classes**: the training rollout first; **eval runs as background filler** in the gaps
  (which is what lets eval run continuously, TASK_BACKLOG T6).
- **A GPU-over-CPU bias**: forwards go to the GPU by default.
- **Judged on the learner's END-TO-END throughput**, not on the service's own calls per second.

*(The reasons for these choices are not recorded beyond the row. Its training use rides M5;
bf16 inference for opponents is a separate fidelity experiment, EXPERIMENT_BACKLOG X19.)*

---

## 5. Tier 3 — the learner, unchanged

`MaskablePPO` with the dual-head policy, the callbacks, the reward manager reading `TurnDelta`, the
pool's promotion logic. The only change is the source of its rollout columns (Tier 1 writes them
directly) and where its opponents' forwards run (Tier 2). PPO stays exactly on-policy: Tier 2 freezes
the trainee's weights for a collection window exactly as `collect_rollouts_async` does today.

---

## 6. What retires, what survives, what is the oracle

| today | end state |
|---|---|
| `Gen3Battle(Battle)` over poke-env | retired from production, then **DELETED (T27 P6 slice 6d-2, 2026-10-08)** together with poke-env; the parity ORACLE for §3.2's gate and `present()`'s rules was retired with it (the core's checks are `core_corpus_test.py`, the obs golden, `core_present_golden_test.py` and the cargo tests) |
| `BattleEvent` schema | **survives verbatim**; emitted by the sim, parsed from logs |
| `LiveView` / `TurnView` / `LegalActions` / `StrictBattleView` | become `OneSidedView` / version methods; the strict-API lock's *idea* (non-battle code reads only through read-models) survives as the Rust type boundary |
| `TurnDelta.build_from_events` | `version.turn_delta(side)` |
| `EpisodeTracker`, belief trackers | fields on the version |
| `Gen3ObservationEncoder` (Python) | its encode path is **DELETED (T27 P6 slice 6d-2)**; the class keeps the layout / dimension / `describe_vector` for the model and the prober, and the Rust encoder's byte gate is the obs golden (`agents/training/golden_obs_core.py`) |
| `obs_materializer` / `view_successor` / `view_adapter` / `event_fold` / `expand_many` JSON | `version.successors()` in-process |
| forkserver, bridge child per env, `SubprocVecEnv`, `AsyncSubprocVecEnv` | one Rust env process, N envs |
| per-env compiled opponent (`--compile-opponents`, preload) | Tier 2 |
| `ws_frontend` (the Rust websocket front end) | survives as-is: it is already the "third party drives the server" transport; its client side becomes §3.2 entry 2 |
| `play.py` (ladder client) | parse → version → encode → Tier 2 → act; one path with training |
| the differential fuzzers and the e2e capstone | **survive unchanged**; they gate the simulator, which does not change |

*(Status of each row, as of 2026-09-27: the deletion manifest in `program_rust_core.md` §4. The
shaped reward path, not in this table, was also deleted, `e3ef16db`. The scripted bots, also not in
the table, are to be PORTED to the core, owner decision recorded in TASK_BACKLOG T3, Lane F.)*

---

## 7. Ordering, gates, and when to decide

> **SUPERSEDED as the plan (2026-09-23).** The owner licensed the program on unification grounds
> and re-cut this ordering in `program_rust_core.md`: build the whole core ALONGSIDE, one continuous
> parity gate, ONE cutover, one deletion pass; and the cutover (M6) BEFORE N-envs-per-process (M5).
> The decision point below read LICENSED on its own terms (Python glue + trackers + folds 59% of a
> searched decision, Phase 0) but was recorded only. The table is kept as the original reasoning.

**The recommendation is incremental replacement behind byte gates, encoder last, with a decision
point after the current crossings.** A rewrite started on speculation would run for weeks with two
paths coexisting and every measurement having to say which one it ran on; the profile is what
licenses each step.

| step | what crosses | gate | status |
|---|---|---|---|
| 0 | the searched-decision profile | `search_decision_benchmark.py`, interleaved A/B | landed 2026-09-22 |
| 1 | one fork per decision; lazy tokens | fork-sharing parity (bytes) | landed (1.33× / 1.66×) |
| 2 | D10 — the port emits the intermediate-request view | two-seed parity sweep; tracker-fed byte gate with D10 arms included | in flight |
| 3 | `expand_many` payload / transport | byte-parity on `LiveView.from_view_json`; two-seed sweep | in flight |
| **decision point** | re-profile after 2–3. If Python glue + trackers + folds are still > ~50 % of the decision, the rewrite is licensed; if < 25 %, the in-process tree alone closes it | — | after 2–3 |
| 4 | **Tier 2** — the inference service, opponents first | actions byte-identical to the per-env compiled path on a fixed obs set; rollout FPS non-regression at `--n-envs 64`; the 5.07e-07 max-Δ bar the compile met | Python + torch, no parity oracle needed — can start any time the GPU is not mid-read |
| 5 | **Tier 1a** — `BattleVersion` + emitted events + parsed events, NO encoder yet; Python encoder reads the Rust view through the existing adapter | §3.2's parse-reproduces-step gate; `event_fold_parity_fuzz_test` at zero divergences; every `present()` rule with its own unit test | rust; the hardest step |
| 6 | **Tier 1b** — trackers and legality as version fields/methods | tracker-fed byte gate | rust |
| 7 | **Tier 1c** — the encoder | `obs_build_benchmark` + a byte gate against `Gen3ObservationEncoder` on the fuzz corpus and on every golden | rust; last, by design |
| 8 | N envs per process; retire the forkserver path | training FPS non-regression; a full `--debug` smoke and the first two minutes of a real launch (the only test of the preload layer) | — |
| 9 | the ladder client on entry point 2 | `ladder_drift_scan` + a Metamon/Foul Play anchor cell byte-identical to the Python client's | — |

**Not before a research read completes.** *(Still the rule; the cutover landed between reads.)* Steps 4–9 each change what a measurement runs on; each
lands between reads with its own entry stating which path every subsequent number used.

**Sized honestly.** *(As built, M1–M4 landed within two calendar days, 2026-09-23 → 09-24,
with agents in parallel; M5 alone is re-sized at ≈ 24–33 agent-days, `program_rust_core.md` §2.)* Steps 5–7 are several agent-weeks of grinding with gates, most of it in
`present()`'s rules and the parser's attribution. The one-sided view (a fraction of step 5) took two
agent-days and found four contract changes on real boards that no code reading predicted; budget for
that rate.

---

## 8. What this buys, stated as the measurements it would unlock

- The rollout-leaf battery at 400 pairs (1,222 CPU-h today) becomes affordable — the read that
  decides whether search-as-teacher is worth reopening.
- A search-based teacher's confirm rollouts run as batched leaves on Tier 2 — the owner's named
  interest (2026-09-17: "a search based exploiter teacher … a fine tune, search useful model for
  ladder teams").
- Hundreds of envs in flight at no per-env process cost, which is what makes batching a free
  throughput knob rather than a latency trade.
- One path for training, eval, anchors and the ladder — a third party's parser divergence becomes a
  diff against a known-correct reference (the Metamon Baton Pass finding is the first instance).
- The prober walks versions instead of re-parsing traces; "what happened on turn t" is an index.

---

## 9. Open questions this document does not settle

1. **Reading vs truth.** Which `present()` rules should flip to the sim's truth at the next retrain,
   and what that costs the reward — a registered experiment, not a design choice.
2. **Tier 2's placement for search at inference on a ladder box** *(still open; M5 Phase A's
   per-consumer rule puts offline search on the FFI front end)* — same process as the tree, or a
   sidecar; decided by the flush-timer latency the tree can tolerate.
3. **Whether the Python encoder survives as an oracle forever** *(CLOSED, T27 P6, 2026-10-08: it was retired — its encode path is deleted and the Rust encoder has its own goldens)*.
4. **The generated-layout mechanism** (Python constants → Rust table) — a build step or a checked-in
   table with a pin; the latter is simpler and matches `arch_tables`.

---

## Decision record

Owner decisions are marked **(owner)**. Ledger entries are cited by date and title; `L…` is the
line in `ledger_index.md`'s numbering.

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-09-22 | The environment's end state | Three tiers: a Rust battle core owning board, per-side view, typed events and encoder; one inference tier; PPO unchanged | Continuing crossing by crossing (each moved the cost to the next link: per-op 7–47×, end-to-end 1.33–1.89×) | §1; `measurements/search_profile_2026-09-22/` |
| 2026-09-22 | History structure | Persistent versions (fork = handle clone, history = parent chain) | An LSM tree (solves write-heavy on-disk storage, not O(1) fork) | §3.1 |
| 2026-09-22 | Crossing order | Encoder LAST (most rules, largest byte-parity burden) | — | §7; kept by the program (M4 last of the Tier 1 slices) |
| 2026-09-23 | License to build **(owner)** | The RUST CORE PROGRAM, licensed on UNIFICATION ("a low tech debt, robust, performant and unified approach") | Waiting for §7's re-profile decision point (it read LICENSED, 59% glue, but was recorded only) | ledger 2026-09-23 *PHASE 0* (L21259); `program_rust_core.md` §0 |
| 2026-09-23 | Operating model **(owner)** | Build alongside, one continuous parity gate, ONE cutover, one deletion pass | A flag / flip / delete per milestone | `program_rust_core.md` §1 |
| 2026-09-23 | Order **(owner)** | The cutover (M6) BEFORE N envs per process (M5): change one thing at a time | §7's order (N envs per process before the ladder client, one path) | `program_rust_core.md` §2 |
| 2026-09-23 | Observation path **(owner)** | The observation ALWAYS comes through the parser, training included; search keeps the typed shortcut | Training reading typed-at-source events | `program_rust_core.md` §6c; `parse` 21.6–30.8 µs/decision |
| 2026-09-23 | Reading vs truth | The core's event is a SUPERSET (truth + reading projection); reading BUGS fixed in the fork | §3.3's per-rule flags defaulting to the reading | Phase 0 (8 reading rules, 0 residual on 259,782 comparisons); R1–R3 fixed `65f22334`, boundary `c97358e8` (ledger 2026-09-23 *THE TRUTH AUDIT*, 2026-09-24 *TRAINING-INPUT BOUNDARY*) |
| 2026-09-23 | `present()` signature | Built from one side's stream, no board parameter | `present(board, events, side)` | ledger 2026-09-23 *RUST CORE M2 LANDED* (L21293) |
| 2026-09-23 | Tier 2's justification | Unification, the batched search leaf, and N-per-process needs it | "Frees sixty-four cores" (measured: 0.4–2.1 of 16) | `program_rust_core.md` §0 |
| 2026-09-23 | The reward on the core | Win indicator only; shaping NOT ported | Porting PBRS / bias terms | `c49ef704`; then deleted, below |
| 2026-09-25 | The cutover **(owner)** | `--obs-source core` default on the rust bridge; a registered DEVIATION: switched at ~60–70% of the stress counts, the stress continuing as confirmation and any CUTOVER-class divergence reverting the default | Waiting for 100% of the counts | `ac0b6469`; ledger 2026-09-25 *THE RUST CORE CUTOVER* (L21395) |
| 2026-09-26 | On a core frame refusal in a live run **(owner)** | Save the repro, then resume on `--obs-source python` OR idle the GPU while fixing, the orchestrator's call per incident | Always waiting for the fix (the registration's recommendation, withdrawn) | ledger 2026-09-26 *NEW LINEAGE … GO* (L21415) |
| 2026-09-26 | Deletion pass, part 1 | Search core-only; the port's one-sided view deleted; five rows SKIPPED with verified blockers | Deleting on the manifest's premise that Python was already only an oracle (false: opponents, eval, labels, reward, clock) | `43712881`, `97a30387`, `2c7e6acb`; ledger L21403 |
| 2026-09-26 | Shaped reward **(owner)** | Deleted; a resume or fork of a shaped checkpoint REFUSES (`ShapedRewardCheckpointError`) | A silent switch to terminal-only | `e3ef16db`; ledger L21417 |
| 2026-09-26 | M5 transport **(owner + benchmark)** | BOTH front ends (FFI and a `/dev/shm` process) over ONE core; default per consumer by crash isolation; 12 lanes | Picking one transport (they measured equal: 0.994 [0.943, 1.026]) | `measurements/rust_core_m5_transport_2026-09-26/`; ledger L21421 |
| 2026-09-27 | Scripted bots **(owner)** | Port them to the core (Lane F) | Keeping bot battles on the old path until M7 | TASK_BACKLOG T3 (`93745a66`); rationale not recorded |
| 2026-09-27 | Tier 2 shape (TASK_BACKLOG row; whose decision is not recorded) | Fixed weight slots, slot-tagged requests, fixed-width or compiled buckets, priority classes with eval as background filler, GPU-over-CPU, judged on the learner's end-to-end throughput | §4.2's open catalogue with a menu of bucket sizes | TASK_BACKLOG T2 (`93745a66`); rationale not recorded |
| 2026-09-27 | Long-lived processes (TASK_BACKLOG rows) | Allocator discipline in Lane 0; memory-triggered restarts (proposed) | Restarts on a fixed clock only | TASK_BACKLOG T1, T7 (`93745a66`) |
| 2026-09-27 | Ladder client | M7 after the cutover (TASK_BACKLOG T9) | Moving the ladder client early (would ADD a path) | `program_rust_core.md` §1 rule 5 |
| 2026-10-08 | poke-env and the Python oracles (T27 P6, owner's single-stack direction) | Retired ENTIRELY: the vendored fork, the Python battle layer, trackers, encoder encode path and action stack are deleted; the Rust core's own goldens are the parity references | Keeping the Python encoder / `Gen3Battle` as a standing oracle (§6 and open question 3) | `designs/ops/deletion_pass_manifest.md` §8.3–§8.6; `designs/deleted_flags.md` "Deleted PATHS" |
