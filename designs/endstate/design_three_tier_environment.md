# Design — The three-tier environment: a Rust battle core, one inference tier, and the retirement of poke-env

**Status: DESIGN, not scheduled.** Authored 2026-09-22 at the owner's request, after a month in which
every search-cost crossing (the Rust bridge, the one-sided view, the materializer flip, the
one-fork-per-decision landing) bought a large per-op speedup and a small end-to-end one. This document
states the end state those crossings are approaching, so the next profile has something to decide
against. Nothing in it is a commitment; the ordering in §7 is the recommendation.

**Companion:** [`design_model_management.md`](design_model_management.md) — the model-side end state
(the registry, the pool, the inference tier's catalogue). This document owns the environment side.

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
production path and survives as a parity oracle.

---

## 1. What is true now, with the numbers that motivate this

Measured facts, each with its record. This section is the argument; §2 onward is the design.

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

### 3.3 The presentation rules — the part that is NOT a port

`designs/rust_sim/one_sided_view.md` §2 is the standing finding: half of `LivePokemon` is not sim
state, it is poke-env's fold of the protocol. The contract that document forced — *the port emits
sim facts and raw protocol facts; every poke-env presentation rule is applied in Python* — was the
right split for a crossing. **For the end state the rules cross too, and they cross as RULES, not
as a re-derivation:**

- `OneSidedView` is computed from the omniscient board + this side's `events` by an explicit
  `present(board, events, side) -> OneSidedView` function whose every rule is named, tested
  individually, and pinned to the poke-env line it mirrors (the doc already lists them: sighting
  counts doubled against Pressure; the volatile fold's drop rules; the status-counter transitions;
  the protect-counter as a stall-move count; `revealed` set by the `|switch|` line; team order from
  the first `|request|`; `item` only from `|-item|`/`|-enditem|`/`[from] item:`; `consumed_item`'s
  both-sides clearing; the two-slot `ability` with Trace semantics).
- **The reward was trained on poke-env's reading, so the port must reproduce the reading, not the
  truth**, until a model is retrained on the truth. `present()` is where that choice lives, one rule
  at a time, each behind a flag that defaults to "poke-env's reading" and can be flipped to "the
  sim's truth" for a retrain. Today the choice is implicit and unflippable.
- **Third-party parsers are the reason to own this.** Metamon's poke-env drops Baton Pass boosts;
  ours does not; a future opponent's will differ somewhere else. When our view is `present()` over
  the sim's truth, a third party's divergence is a diff against a known-correct reference, not a
  mystery between two parsers.

### 3.4 The event log — kept as the spine, emitted at the source

The log is load-bearing (§1) and stays the primary structure, not a derived one. The change is
WHERE it is produced: today `Gen3Battle.parse_message` classifies each protocol line and appends a
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
and hands the batch to Tier 2. The forkserver, the per-env bridge child, `SubprocVecEnv` and the
async vec-env all retire; the async scheduler's *idea* (forward whichever envs are ready) survives
as "the batch is whatever is ready at the flush".

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

1. **CPU inference goes to near zero across all six consumers at once** — sixty-four opponents
   stop costing sixty-four cores of dispatch-bound B = 1 forwards. That CPU is what constrains envs
   in flight and agents in parallel today.
2. **Batch size becomes a throughput knob with latency governed by one timer**, instead of six
   accidental settings.
3. **The env simplifies to a simulator plus an encoder**, which is the shape Tier 1 wants to be.
4. **Search inherits a batched leaf for free**: successors of one decision are a batch by
   construction, and a search-as-teacher's confirm rollouts are a batch across candidates.

### 4.4 What it does NOT change

Throughput of the gradient step (89 % of a training step). This tier is the enabler for Tier 1's
scale and for search; on its own it moves steps/hour by a few percent and frees cores.

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
| `Gen3Battle(Battle)` over poke-env | retires from production; **survives as the parity ORACLE** for §3.2's gate and for `present()`'s rules |
| `BattleEvent` schema | **survives verbatim**; emitted by the sim, parsed from logs |
| `LiveView` / `TurnView` / `LegalActions` / `StrictBattleView` | become `OneSidedView` / version methods; the strict-API lock's *idea* (non-battle code reads only through read-models) survives as the Rust type boundary |
| `TurnDelta.build_from_events` | `version.turn_delta(side)` |
| `EpisodeTracker`, belief trackers | fields on the version |
| `Gen3ObservationEncoder` (Python) | **survives as the oracle** for the Rust encoder's byte gate |
| `obs_materializer` / `view_successor` / `view_adapter` / `event_fold` / `expand_many` JSON | `version.successors()` in-process |
| forkserver, bridge child per env, `SubprocVecEnv`, `AsyncSubprocVecEnv` | one Rust env process, N envs |
| per-env compiled opponent (`--compile-opponents`, preload) | Tier 2 |
| `ws_frontend` (the Rust websocket front end) | survives as-is: it is already the "third party drives the server" transport; its client side becomes §3.2 entry 2 |
| `play.py` (ladder client) | parse → version → encode → Tier 2 → act; one path with training |
| the differential fuzzers and the e2e capstone | **survive unchanged**; they gate the simulator, which does not change |

---

## 7. Ordering, gates, and when to decide

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

**Not before a research read completes.** Steps 4–9 each change what a measurement runs on; each
lands between reads with its own entry stating which path every subsequent number used.

**Sized honestly.** Steps 5–7 are several agent-weeks of grinding with gates, most of it in
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
2. **Tier 2's placement for search at inference on a ladder box** — same process as the tree, or a
   sidecar; decided by the flush-timer latency the tree can tolerate.
3. **Whether the Python encoder survives as an oracle forever** or is retired once the Rust one has
   its own goldens; the one-sided view's experience says keep the oracle until two full seeds of
   fresh boards have been byte-clean.
4. **The generated-layout mechanism** (Python constants → Rust table) — a build step or a checked-in
   table with a pin; the latter is simpler and matches `arch_tables`.
