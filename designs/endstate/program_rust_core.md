# Program — the Rust core: milestones, gates, and the one cutover

**Status: PLAN, authored 2026-09-23 at the owner's request (Phase 0 of the RUST CORE PROGRAM).
M1 and M2 BUILT; the per-emission EMISSION SELF-CHECK (§3) BUILT, ON in every test and fuzzer build,
compiled out of production.**
Explicit-only, like every document in `endstate/`: update it on the owner's word. It implements
[`design_three_tier_environment.md`](design_three_tier_environment.md) (the end state; its §7 is the
ordering this plan re-cuts) and is licensed on **unification** grounds — the owner's goal is "a low
tech debt, robust, performant and unified approach; search will be on the table in our true end
state". The before-numbers every milestone reports against, the event-attribution spike and the
tech-debt inventory are in
[`../research_state/measurements/rust_core_phase0_2026-09-23/`](../research_state/measurements/rust_core_phase0_2026-09-23/README.md).

---

## 0. What Phase 0 changed about the plan

| Phase-0 result | what it changes |
|---|---|
| **The §3.4 hazard is not a parser problem.** Source-typed events for 15 of the 35 event kinds (16 of the 51 EVENT keywords; **82% of all event lines by volume**) equal `Gen3Battle`'s reading on 297 real bridge battles × 2 viewers — **259,782 per-viewer event comparisons, 0 residual, 0 unmatched** — once **8 named poke-env READING rules** are applied. Every rule is text-derivable, and the sim's own truth for the one attribution field the protocol does not print (which move owns an outcome line) is recovered from line order by a 4-token reset rule on **12,850 / 12,850** outcome lines | Tier 1a is no longer "the hardest step" on the parser side. The work is the READING RULES (how many, and which flip to the truth at a retrain) — so the plan front-loads a rule inventory and makes the core emit BOTH the truth and the reading |
| **The event schema is lossier than the source.** The reading drops the `[of]` of 2,608 Recoil / Leech Seed / drain lines, the `[from] ability:` of every immune-by-ability, gives a `-miss` the USER as its target, a `[still]` move the FOE as its target, and writes one miss as TWO MISS events | The core's event is a SUPERSET: truth fields + the reading projection. `gen3_event_value_schema_v1` is kept verbatim on the projection; the truth fields are additive and read by nothing until a retrain chooses them |
| **The searched decision is ~87% not-the-engine.** View road, wide B: rust engine ≈ 4% (+ an unsplit 8.5% `open_root`), transport/JSON 10%, **Python glue + trackers + folds 59% (66% with the event/prefix folds)** | The design's §7 decision point reads LICENSED (> 50%) on its own terms, as a record only — the owner licensed the program on unification |
| **Tier 2 does not free the CPU the design says it does.** A compiled B = 1 opponent forward is 2.4–4.6 ms on this box at load 30–35 (0.98 ms on record at a quieter box); at the live arm's 445 steps/s, and assuming EVERY opponent is a network, that is **0.4–2.1 cores of 16**, not "sixty-four cores" | Tier 2's case is unification, the batched search leaf and — decisively — that **N envs per process cannot exist without it** (48 per-env B = 1 compiled graphs inside one Rust env process is not a thing). Its TRAINING use therefore rides the one cutover with Tier 1; its non-training users may adopt early |
| Two latent defects surfaced (a `ViewEventFolder` that never resets the move owner at `\|turn\|`, reachable at the default search depth 3; the port PANICS at 1,000 committed turns where the pinned Showdown TIES) | Both become prerequisites of the milestone whose gate would otherwise trip on them (§5) |

---

## 1. The operating model — build alongside, one cutover, one deletion pass

*(Owner amendment, 2026-09-23. It replaces the flag / flip / delete-per-milestone discipline.)*

1. **BUILD ALONGSIDE.** The Rust core is built to completion as a path TRAINING DOES NOT USE. While
   it is under construction the Python path is simply the product; two paths cost us only when
   both are IN USE. **No default flips in the training loop mid-program.**
2. **CONTINUOUS PARITY GATE from the first milestone.** Every milestone adds its SLICE (events,
   views + legality, trackers, obs, env) to ONE byte-parity harness against the Python path, on a
   fixed, reproducible corpus. The cheap tier lives in the routine gate, so a research change to
   the Python side (a new obs feature, a flag, a tracker fix) breaks parity THE DAY IT LANDS and is
   mirrored while it is small (§3).
3. **ONE CUTOVER.** When the core is complete, the CUTOVER tier (§3) is the gate. Training switches
   ONCE, between research reads, with a ledger entry naming the commit.
4. **ONE DELETION PASS** right after the cutover removes the old production path — the per-milestone
   lists in §4 are that pass's MANIFEST. **Named oracles survive:** poke-env + `Gen3Battle` + the
   read-models (event and view parity), the Python encoder until two full seeds are byte-clean
   post-cutover, and node Showdown + the differential fuzzers (they gate the SIMULATOR, which this
   program does not change).
5. **Early adopters.** A non-training consumer MAY move to the core before the cutover **iff** it is
   clearly lower-risk AND it REMOVES a duplicate path rather than adding one. Exactly one qualifies:
   **search** (§2, M2), whose materializer today runs TWO roads plus a fallback between them. The
   prober, anchors and the ladder client do NOT: each reads the same Python path training reads,
   so moving one early would ADD a path — they move after the cutover (M7).

---

## 2. The milestones

> **ORDER CHANGED (owner, 2026-09-23): the CUTOVER comes BEFORE M5.** Sequence: M1 → M2 → M3 → M4
> → **M6, the cutover, in TODAY'S env shape** (process-per-env; training's observation built by the
> core through `parse` per §6c) → the deletion pass → **M5 (N envs per process) only after that**,
> as its own later change with its own gate. Why: the switch to Rust for training is derisked by
> changing ONE thing at a time. The cutover swaps what builds the observation while the process
> model stays fixed; M5 then changes the process model on a Rust-built observation that is already
> proven. **M5 is not started until the owner says so.** T2 (the inference tier) is M5's prerequisite,
> so it waits with M5 unless a non-training consumer adopts it earlier.

Sizes are **agent-days**, calibrated to two measured rates: the one-sided view (a fraction of
Tier 1a) took **2 agent-days and found 4 contract changes** no reading predicted; the Phase-0 spike
typed 15 event kinds and built its harness in **~½ agent-day and found 8 reading rules + 2 latent
defects**. Every estimate below carries a 1.5× surprise allowance at that rate.

### M1 — Typed events at the source, all kinds, + the parity harness (Tier 1a, part 1)

**Status (2026-09-23): BUILT, not used by training** — every omniscient line is a typed `Line` whose
text is its rendering (the spike feature deleted; production bytes proven identical by the port's
gates), one side's stream → `CoreEvent`s carrying `Gen3Battle`'s reading (11 named rules),
`parse(lines)` with `parse(emit(step)) == step`, the record `gen3_core_event_v1`, and slice E of the
harness at COMMIT + MILESTONE tiers, 0 residual. Both prerequisites closed (the turn-limit tie,
`gen3_turn_limit_tie_v1`; the `ViewEventFolder` reset, `gen3_view_fold_turn_reset_v1`). Contract:
`designs/rust_sim/core_events.md`; measurements: `research_state/measurements/rust_core_m1_2026-09-23/`.

**What crosses.** Every omniscient line is emitted by a TYPED `ProtocolBuilder` method (today 17
`push_raw` call sites bypass them; 3 are in the spike's subset and were typed there). The core
holds, per line, a `CoreEvent` = the source facts (actor, target, the `[from]`/`[of]` cause, exact
HP, the ENGINE's action scope) + its **reading projection** (the `BattleEvent` a given viewer's
`Gen3Battle` would build — `gen3_event_value_schema_v1` verbatim, including the move-suffix
synthetic MISS/FAIL). `parse(lines) -> Vec<CoreEvent>` (the design's entry 2) lands in the same
milestone, because the spike showed it is cheap and it is what the ladder client and the prober
will stand on. Conservation moves to the source: **one record per committed line, refused otherwise**
(the spike's binary already refuses a mismatch).

**Prerequisites.** (a) Showdown's turn-limit TIE in the port (`sim/battle.ts:1836` ties at turn >
1000 and warns from 500; the port panics at 1,000 committed turns — **3 of 300** random-player
battles hit it in Phase 0, so a 70k-battle cutover corpus would hit it ~700 times). (b) The
`ViewEventFolder` move-owner reset (§5 F2), so the Python oracle the gate compares against is right.

**Gate.** Slice E of the parity harness at COMMIT + MILESTONE tiers (§3): per viewer, per event,
`seq · turn · kind · side · actor · target · value · raw` equal to `Gen3Battle`'s; plus the
Rust-internal `parse(emit(step)) == step` on the same corpus; plus the 22-scenario protocol corpus
(`src/rust_sim/tests/protocol_test.rs`'s goldens) because random battles never exercised four
ambiguity-prone shapes — Damp's `[of]` cant, a slot-less future-move `-miss`, a Heal Bell bench
`-curestatus`, `[from] lockedmove` (0 occurrences in 297 battles). **Size: 3–4 agent-days** (35 of the 51
keywords remain, the fiddly half — ITEM/ENDITEM/ABILITY/VOLATILE/ACTIVATE `[from]`/`[of]` merges,
CLEARBOOST's seven-keyword `op`, SIDE refs — plus the harness infrastructure). **Unlocks:** the
typed stream every later slice folds; nothing in production changes.

**The persisted RECORD (added 2026-09-23, owner request).** The log has two lives: as STRUCTURE (in
memory, needed wherever the encoder runs, search included, because the observation carries a
32-event window and every tracker is a fold over events) and as a RECORD (written out and re-read
by evals, the prober, replay and online-play archives; never by search). M1 fixes the record's
format, because it is where silent drift would enter AFTER the cutover:
- **What is stored:** the per-side protocol TEXT as received or emitted (the authority; §6b) plus
  the typed `CoreEvent` stream derived from it, with a header carrying `event_schema` (e.g.
  `gen3_core_event_v1`), the core commit, the producing path (`step` or `parse`), the viewer side,
  and the Showdown version, when a real server produced the text. Storing both the text and the
  typed stream means any future parser can re-derive, and any disagreement is a diff rather than a
  mystery.
- **How it is read:** only through the core (`parse` for the text; a versioned deserialiser for the
  typed stream). A reader facing an unknown `event_schema` REFUSES; it never guesses.
- **How it evolves:** a schema change bumps `event_schema` and ships a MIGRATION that re-derives the
  typed stream from the stored text. That is the reason the text is kept: old records are re-parsed,
  never hand-edited.
- **Gate:** a committed golden corpus of records (the commit-tier battles plus the 22 protocol
  scenarios) must round-trip byte-identically (write → read → write) and must re-parse from its
  stored text to the stored typed stream, on every commit. A new schema must pass a
  migrate-then-compare on the whole golden corpus.
- **Scope:** only persistence points pay for it (the end of an eval game, an online game, a prober
  export). A search successor is never serialised.

### M2 — `BattleVersion` + `present()` + legality (Tier 1a, part 2) — search adopts

**Status (2026-09-23): M2 BUILT and LANDED** (`gen3_core_version_v1`, `gen3_core_present_v1`,
`gen3_core_search_v1`; contract [`designs/rust_sim/present.md`](../rust_sim/present.md)).
`BattleVersion` (persistent, `Arc` parent, per-transition typed events, memoized per-side view, raw
request; built by step or by parse, gated `parse == step` version by version on every corpus
battle); `present(reading)` over a `BoardReading` — **built from ONE SIDE'S STREAM, no board parameter**, the board a
REFEREE (`check_view`); `legal_actions()` + the 11-dim mask. **`present()` is the TRUE reading**
(owner directive): where poke-env is wrong about a sim fact the disagreement is a registered
FINDING (`agents/battle/poke_env_findings.py`) — PE-V10, PE-R1b, PE-V16, all three reaching the
obs. **All three FIXED in the fork at `c97358e8` (`gen3_pe_reading_fixes_v1`, owner 2026-09-24; a
TRAINING-INPUT boundary), and the registry is now EMPTY.** Slice V's core column (present + legality + mask + the board audit)
is green at COMMIT and MILESTONE with no allowlist but the findings (numbers in the measurement
record below). **Search adopted it** — `materializer=core` is the default; the three gates run it
as a third road with the INTEGRITY check on every arm; every battery row is stamped
`materializer` / `core_path` / `integrity`. The protocol and view roads are NOT deleted (§4 lists
what now is). Findings: the core road makes a D10 leaf the version AT the intermediate decision (a
forced switch is non-branchable there, where the old roads expanded it from end-of-turn with the
intermediate tokens — a depth ≥ 2 semantic difference); `recorded_exact` is refused on core; on
procedural teams slice V found two VIEW-ROAD projection defects the core does not have (a Trick
`volatile`, a Mimic-copied move) — the road is on the deletion list. And the training transport no
longer folds the one-sided view (`gen3_view_fold_opt_in_v1`): post-fix `sim_bridge` CPU is **0.909×
pre-M1** [0.876, 0.932] with byte-identical output. Measurements:
[`research_state/measurements/rust_core_m2_2026-09-23/`](../research_state/measurements/rust_core_m2_2026-09-23/README.md).

*Before M2, slice V ran AHEAD of it against the view road's projection* (the truth audit,
`gen3_core_parity_views_v1`): it CLOSED the three read-model findings (`one_sided_view.md` §4b)
with deferral D6 and eleven more projection classes, and found poke-env READING defects R1–R4 —
R1–R3 FIXED in the fork, a TRAINING-INPUT change (`designs/CHANGELOG.md` 2026-09-23).

**What crosses.** The persistent version (omniscient board + events + `parent`), per-side
`OneSidedView` computed by `present(board, events, side)` whose every poke-env rule is NAMED, tested
alone and pinned to the line it mirrors (the nine of `designs/rust_sim/one_sided_view.md` §2, the
D6–D9 deferrals, and the THREE open read-model findings of its §4b — the own-side sleep-counter
drift, the opponent ability disclosed by a `[from] ability:` clause, the missing `substitute` — which
must CLOSE here because the gate has no allowlist; CLOSED 2026-09-23 on today's projection, see the
status above). `legal_actions(side)` from the raw `|request|`.
Each reading-vs-truth choice is a flag defaulting to the READING.

**Gate.** Slice V (the whole `LiveView` graph + `LegalActions` per decision, both viewers) at
COMMIT + MILESTONE; `parse` reproduces board + events + both views version-by-version.
**Size: 5–7 agent-days.**

**Early adoption — SEARCH.** `SearchConfig.materializer` gains `core`; a searched decision's
successors come from versions. This DELETES a duplicate path rather than adding one: the protocol
road, the view road's D10 fallback and its depth-≥2 fallback (a D10 leaf carries `fork=None` today
because the port has no board AT the intermediate decision; a version does) all become deletable.
Condition: no live arm runs the search teacher (`SearchTeacherCallback` is inside the trainer);
the adoption is its own ledger entry and every search number after it is stamped `materializer=core`.
Gate for the adoption: the existing `materializer_parity_integration_test`,
`fork_sharing_parity_integration_test` and `one_sided_view_parity_fuzz_test` run with `core` as a
third road, identical decisions and obs bytes.

### M3 — Trackers and `TurnDelta` as version methods (Tier 1b)

**What crosses.** `EpisodeTracker.record_context` / `advance_window` (recency, pair history, the
32-row event window, the progress clock), the choice-band, Hidden-Power, sleep and wish beliefs,
`TurnDelta.build_from_events`; fields on the version, folded at construction, shared by a fork (the
pinned-pickle tracker thaw — 16.9% of the searched decision today — becomes a pointer copy).

**Status (2026-09-24): the trackers CROSSED** (`gen3_core_trackers_v1`, contract
[`designs/rust_sim/trackers.md`](../rust_sim/trackers.md)): slots, the Hidden-Power belief, the
progress clock's obs half, recency, pair history, the 32-row event window, the wish and sleep folds,
the α/β label and the win-indicator reward, folded on the version from one side's stream and shared
by a fork (`Arc`, copied on the fork's own decision). Slice T green at COMMIT (0 divergences, 1,838
decisions) and MILESTONE (5/5 pass, E + V + T, at `493db48f`) and on fresh battles (the
fold-equivalence fuzz re-pointed at the core, `rust_core_trackers_fuzz_test.py`: 176 battles, 39,110
decisions, 0 on E, V and T). The
NATIVE window record (`gen3_core_window_record_v1`: ordered actions + attributed effects, ACTION
DENIAL incl. the gen-3 TURN CUT — any faint cancels every remaining queued action, so a faster
self-KO denies the survivor — and the opponent's denied choice unrepresentable by type, checked at
the slice level too) is gated by 24 constructed fixtures.
**The tracker fork:** the Rust fold per successor is 0.089 ms, 1.8 % of a wide searched decision
[0.7, 2.4] — 0.17× [0.07, 0.30] the pinned-pickle thaw ALONE (9.9 % [9.3, 10.3]); search still
thaws until M4's encoder reads the version's trackers (`SearchConfig.core_trackers` stays OFF).
**The loss catalogue** (§5 of [`research_state/measurements/rust_core_m3_2026-09-24/`](../research_state/measurements/rust_core_m3_2026-09-24/README.md);
820 / 820 MILESTONE battles, 142,410 viewer decisions, slice T 0): per 1,000 decisions a turn actor
is denied by fainting first 24.9, by the gen-3 TURN CUT 4.5, refused 99.1 — and every opponent
denial's α/β label is MASKED. GIGO it surfaced (none fixed; training inputs): the 22-column event
window records every stat DROP as a rise (58.4), a Protect block as a hit (17.5; it also kills the
progress clock's exogenous-freeze branch), a Rapid Spin clear as a hazard set (8.7), Trick as two
reveals, Perish Song / Destiny Bond faints as `attack`; the label names a DRAGGED mon as the
opponent's switch (7.5), a replacement straddling a window as a chosen switch (8.4), the called
move of a Sleep Talk (0.3); the clock's "our move dealt damage" clause credits any chip to a
status / failed move (spurious 24.2, decisive 1.3). One poke-env finding: Assist and Nature Power
crash the parse exactly as Metronome does.

**Gate.** Slice T: the tracker state (every tracker above, every event-window row), the α/β
opponent-intent LABEL per decision, and the REWARD (the win indicator) equal per decision, both
viewers. **`TurnDelta` is NOT gated field by field and is NOT ported as a structure** (owner direction,
2026-09-24: drop the turn summary where an entity-aligned substitute exists — the event window is
that substitute and is in production): its obs frames were deleted (`gen3_frame_deletion_v1`), so
slice T gates its CONSUMERS — the label, the progress clock — through a projection of the fields they
read. The core's native record of a window is an ORDERED per-action / per-effect list with full
attribution (`record::Window`), of which the frozen `TurnDelta` and the 22-column event window are
lossy projections. The existing `turn_delta_fold_equivalence_fuzz_test` shape re-pointed at the
core. **Size: 4–5 agent-days** (2,985 LOC of trackers).

**The REWARD in slice T is the WIN INDICATOR alone (decided 2026-09-23, orchestrator; owner consulted).**
Every win-prob-era run trains on `1 TERMINAL + 0 PBRS + 0 BIAS` (`terminal_indicator`, `victory_value`
1.0; the 11 shaped flags recorded `inert_reward_flags`), so the shaped terms (`reward_potentials.py`,
`reward_bias_terms.py`, the shaped half of `reward_manager.py`) are **NOT ported**. The evidence is
enough to not carry shaping forward, NOT enough to say shaping is no better: the flywheel pair read
strength NOT DETECTED (+17.5 Elo to win-prob, one run per arm), and the one contrary CANDIDATE — arm S's untaught
+8.3 pp — is confounded by a 1.44× realized dose. `TurnDelta` does NOT cross as a structure (see
**Gate**): its consumers do, through a projection, and the Python `TurnDelta` survives only as the
label's oracle until the deletion pass (§4's M3 owner row). Any future shaping is model-derived (the held frozen-φ rung, a
T2 value as the potential), never env code.

**Before M3 starts (M2's hand-off, 2026-09-23):** *(1) DONE 2026-09-24 (`gen3_core_engine_split_v1`:
`engine::Engine` + the `BridgeSession` transport around it; `sim_bridge` byte-identical on the
41-battle transcript; a version's fork is 0.950× [0.921, 0.952] the pre-split clone, the transport
history 3.9 % of it — [`research_state/measurements/rust_core_m3_2026-09-24/`](../research_state/measurements/rust_core_m3_2026-09-24/README.md)).*
(1) split `BridgeSession` into an ENGINE (`Battle`,
`FullBattleDriver`, the open boundary and requests as typed values), which `BattleVersion` owns, and
a TRANSPORT (chunks, the text reframe, request JSON strings, `cmd_buf`, `script` / `request_seeds`),
which `sim_bridge` wraps around the engine. Today every fork clones `script` and `request_seeds`, and
both grow with battle length (UNMEASURED); M5 needs the engine without the transport anyway.
*(2) DONE 2026-09-24 (`gen3_core_error_v1`: `core_error::CoreError` — `Refusal{PyExc}` / `Malformed` / `Fault`, across `core_events`, `present`, `version` and the engine's fatal; the refusal class compared against poke-env by `rust_core_present_test.py`).* (2) Replace `Result<T, String>` with a `CoreError` enum that separates refusals mirroring a poke-env
exception class (which the parity gate compares), malformed input, and engine faults. Today a
refusal and a bug are indistinguishable strings.

### M4 — The encoder (Tier 1c) — last, by design

**Status (2026-09-24): the ENCODER BUILT, not used by training** (`gen3_core_encoder_v1`, contract
[`designs/rust_sim/encoder.md`](../rust_sim/encoder.md)): `BattleVersion::encode(side, &mut [f32;
2501])` over the side's reading, view, legality and M3 trackers; the layout GENERATED from
`agents/observation/constants.py` (`gen3_core_obs_layout_v1`, pinned by `rust_core_obs_layout_test.py`);
the dex / prior tables read from `data/`; NaN-prefilled in test / fuzz builds, zero-filled in release;
the row on the wire as a `<f4` frame wrapped with `np.frombuffer`, a wrong dtype / shape / length /
contiguity REFUSED (`gen3_core_obs_wire_v1`). **Slice O green at COMMIT**: 2,081 decisions, both
viewers, every row BYTE-equal, and the obs golden reproduced hash for hash. `obs_build_benchmark.py`
has its core row. **Search takes rows** (`expand_many`'s `rows`: each core arm's leaf is encoded on
its version, shipped with its mask and its choice tokens — `present::choice_tokens`, compared against
the real mapper by slice O — so no view JSON, Python tracker or Python encoder touches a search
successor; the three search gates pass on it, `one_sided_view_parity_fuzz_test`'s new CORE ROW road
byte-equal to the protocol road). **The typed shortcut and its integrity mode are DELETED** (§4).
Measurements:
[`research_state/measurements/rust_core_m4_2026-09-24/`](../research_state/measurements/rust_core_m4_2026-09-24/README.md).

**What crosses.** `encode(side, &mut [f32; 2501])` with the layout GENERATED from
`agents/observation/constants.py` into a checked-in Rust table pinned by an `arch_tables`-style test
(the design's §9 Q4 — the checked-in table is simpler and matches `arch_tables`). The dex/belief
tables come from `data/` exactly as the Python facade reads them.

**Gate.** Slice O: the 2501-dim row `np.array_equal` per decision, both viewers, at COMMIT +
MILESTONE, plus every obs golden; `obs_build_benchmark.py` gains a core row. **Size: 6–9
agent-days** (5,481 LOC, 20 sub-encoders, the largest byte-parity burden). **The M4 final gate includes the
LADDER-USAGE corpus (owner, 2026-09-24):** the filtered Metamon `hl_05_26` gen3ou teams (real
teams from outside our pool, the surface where the Heal Bell crash hid) join the pool and procedural
corpora in slices E/V/O and the fuzzers. The corpus lands IN PARALLEL with M3, so it is standing
before M4's gate runs. Search now takes rows
from `successors()`; the `expand_many` JSON, the view JSON and `view_adapter` leave its path.

**Transport at M4 (decided 2026-09-23): keep the process and the pipe protocol.** The obs row, the
11-dim mask and the training-only label keys (ARCHITECTURE.md §7, taken from the board the Rust side
holds) travel in the reply; Python wraps them with `np.frombuffer`. At M4 the pipe is not the cost:
a production `Gen3Env.step` is ~3–4 ms, and the Python that M2–M4 remove dominates it, while a
10 KB row adds ~1 µs to a round trip that already exists (`m1_transport_throughput_2026-09-23`).
A shared mmap would save only that copy, not the round trip. The pipe also keeps crash isolation,
exact binary pinning (`POKESIM_SIM_BRIDGE_BIN`) and a replayable transcript. **Encoder
invariants:** test and fuzz builds NaN-fill the row before `encode`, so a skipped slot is visible;
release builds zero-fill. A wrong dtype, shape or contiguity is REFUSED, never silently converted.

### T2 — The inference tier (independent track; interleaves anywhere after M1)

**What crosses.** One process owning every loaded network; `score(model_id, rows, masks, mode)`;
bucketed, padded, timer-flushed. Non-training consumers adopt as each is gated — the search leaf,
`main.anchors`, the prober, the cf label factory, the offline meters — and each consumer's OLD
per-consumer load+forward code is deleted in its adoption commit (that is what makes early adoption
remove a path). Its TRAINING use (the pool opponents) rides the M6 cutover: it is a prerequisite of
M5, not a separate flip.

**Gate.** Greedy actions byte-identical and a max|Δ| bar on a NAMED tensor (the runbook's
5.07e-07 does not say which; Phase 0 measured 1.91e-06 on the logits) against the per-env compiled path on
a fixed banked obs set; a service throughput benchmark. **Size: 4–6 agent-days.** **Measured
price it is judged against** (Phase 0 (b), one torch thread, loads 30–35): compiled B = 1 CPU
2.44–4.64 ms/call (6.5–7.3× over eager), a B = 48 CPU batch 0.92–1.26 ms/row — so even on CPU,
batching is 2.6–3.7× per row (4.5× at four threads).

### M5 — N envs per process, successors, the Rust env (Tier 1, env shape)

**What crosses.** A Rust env process stepping N battles, writing obs rows + masks into shared
rollout columns, opponents on T2. `successors(side, k)` = fork, step, present, encode, **plus an
optional leaf hook** (§6). Built alongside: selected only by the parity and throughput harnesses,
never by a research arm.

**Gate.** Slice N (env level): obs rows, masks, rewards and dones equal the Python `Gen3Env` on
recorded battles with both sides scripted from the recording; a depth-3 successor slice (search's
default depth) equal to the protocol road; training throughput at `--n-envs 48` measured as an
interleaved A/B against the current path. **Size: 5–8 agent-days.**

**Transport at M5 is decided BY BENCHMARK:** in-process FFI, EnvPool-style (the Rust library steps N
envs on its own threads with the GIL released and fills the learner's NumPy arrays, so there is no
IPC and no shared-memory segment), against a separate Rust env process that writes into shared memory
and signals once per BATCH. Once Python leaves the per-decision loop, a per-env round trip becomes
a real fraction of the step. Prerequisites: `trainer_turn_benchmark.py` must default to the Rust
bridge first (TECH_DEBT_BACKLOG P2); if FFI wins, every build stamps the module with its commit and
source hash, and Python REFUSES a mismatch at import. Python imports whichever `.so` comes first on
`sys.path`, which is the 09-09 rust-target incident's class.

### M6 — THE CUTOVER, then the DELETION PASS

The CUTOVER tier (§3) green; the `--debug` smoke; **the first two minutes of a real launch** on a
throwaway run dir (the only test of the preload layer, and after M5 of the Rust env's startup);
training throughput non-regression. Then training switches once, between research reads, with a
ledger entry; `metadata.json` records the env core and its commit. The deletion pass (§4) follows in
the next commit series. **Size: 2–3 agent-days** (the campaign is wall time, not agent time), plus
the deletion pass ~2 agent-days.

### M7 — The ladder client, the prober and anchors on the core (after the cutover)

`play.py` becomes parse → version → encode → T2 → act (design §7 step 9), gated by
`ladder_drift_scan` and a Metamon / Foul Play anchor cell whose actions are byte-identical to the
Python client's; the prober walks versions instead of re-parsing traces. Removes the last
production use of poke-env. **Size: 3–4 agent-days.**

**Program total: ≈ 38–56 agent-days** of build, gating and deletion, wall time dominated by the
gates. The order M1 → M2 → M3 → M4 → M5 → M6 is forced (each slice folds the previous one's
output); T2 interleaves after M1.

---

## 3. The parity gate — three tiers, cost proportional to the decision it guards

*(Owner amendment 2, 2026-09-23.)* ONE harness, `rust_core_parity` (proposed:
`src/agents/battle/rust_core_parity_test.py` + a corpus module), with one SLICE per milestone
(E events · V views + legality · T trackers + `TurnDelta` + reward · O obs · N env + successors).
Every slice compares the core against the Python path run under **`designs/production_config.json`**
(the production surface, read via `agents.training.baselines.production_config()`), both viewers,
every decision, **no allowlist** — a divergence fails and prints its census.

**Measured basis for the times below** (Phase 0, this box): the spike's play + replay + two-viewer
event diff ran **297 battles in ~49 s single-process at load ≈ 25 (0.16 s/battle)**; a Python obs
build is 0.52–0.66 ms cold per decision (`obs_build_benchmark`); a random-player battle has ≈ 90
decisions a side; a compiled B = 1 policy forward is 1–4.6 ms. "Loaded" = the usual training arm
sharing the box (load 20–35 on 16 cpus).

| tier | runs | corpus | reproducibility | expected wall (idle / loaded) | command (proposed) |
|---|---|---|---|---|---|
| **COMMIT** | inside the ROUTINE gate, every change | **8 battles**: 6 seeded-random pairings over 6 distinct pool teams + 2 production-policy battles; RECORDED (the `__RECON__` input log: seed, packed teams, command list), committed as a gzip fixture (~50 KB) | nothing is re-played by players: both paths re-derive from the recorded input log, so no player RNG is involved; the fixture carries the pool-team hashes and the gate REFUSES if the core or the pool no longer reproduces the recorded chunks (the spike's byte check) | **~5–8 s / ~10–15 s** (all slices; E alone ≈ 2 s) | `python3 -m pytest src/agents/battle/rust_core_parity_test.py -q` (unmarked) |
| **MILESTONE** | marked `slow`; verdict merged into `designs/ops/slow_tier_status.json` so a recorded FAIL turns the routine gate red. Run when a Rust milestone lands AND when a Python change touches a covered area (`agents/battle`, `agents/observation`, the tracker files, `agents/action`) | **2 seeds × 200 seeded-random battles** (key ranges 0–199 and 5000–5199, the Phase-0 recipe) **+ 2 × 50 production-policy battles** (the `production` baseline checkpoint, `gen3_policy_sample_rng_v1`-seeded sampling) **+ the 22-scenario protocol corpus × 2 seeds** | the key recipe: teams by pool index `key`, `key+1`; player RNG 1000+key / 2000+key; sim seed `[11+key, 22+key, 33+key, 44+key]`; concurrency 1; the forfeit at `StallConfig().threshold` so no battle reaches the 1,000-turn limit. A manifest (pool content hash, checkpoint sha256, key ranges) is committed; the gate REFUSES on a manifest mismatch — a pool change regenerates the manifest in the same commit | **~3–5 min / ~8–12 min** at `-n 2` (E+V+T+O); ≤ 30 min once slice N adds depth-3 successors | `python3 -m pytest src/agents/battle/rust_core_parity_test.py -m slow -q -n 2` |
| **CUTOVER** | ONCE, before training switches, box otherwise idle, all cores | **the full team pool**: every one of the 719 teams × 25 seeded opponents × **both seeds** × two policy families (seeded-random; the production checkpoint at T = 1) = **71,900 battles**, whole-battle byte identity of events, views, legality, trackers, `TurnDelta`/reward and the 2501-dim obs at every decision, both viewers; depth-3 successors on 2% of decisions; then the `--debug` smoke and the **first two minutes of a real launch** | the same key recipe over the full pool; the manifest + every `__RECON__` record banked in the cutover record dir, so any failing battle is re-runnable alone | ≈ 25 core-h → **~2 h on 14 idle cores**; 24 h budgeted so the first failure can be fixed and the WHOLE campaign re-run, not the failing slice | `python3 -m main.rust_core_cutover --manifest <dir>/manifest.json --jobs 14 --out <dir>` |

**The per-emission complement — the EMISSION SELF-CHECK** (`gen3_core_emission_selfcheck_v1`,
BUILT 2026-09-23; [`designs/rust_sim/emission_selfcheck.md`](../rust_sim/emission_selfcheck.md)).
The tiers above check a battle after it ends. The self-check checks each line AT ITS EMISSION: the
omniscient line round-trips through the typed `Line`, each viewer's render is the typed fold that
viewer is owed, no secret reaches the other viewer, and a failure panics on the exact line with both
renders. It is ON in `cargo test` and in the `selfcheck` build every fuzzer, every pytest session and
every fuzz script runs, so every tier and every fuzz run above also runs it; it is compiled out of
the `--release` build training uses (no symbol, byte-identical output, measured cost in its record).

### Keeping the COMMIT tier cheap

* **No players.** Both paths re-derive from a recorded input log: the core replays it in-process;
  the Python reference replays the per-side protocol into `Gen3Battle` + the read-models + the
  trackers + the encoder offline — the materializer's shape, without poke-env's async Player loop.
* **One core call per battle** (a binary like the spike's `event_spike`, JSON out) until M5 decides
  the in-process binding; the core side of a fixture can be cached keyed on (core binary hash,
  fixture hash) — the Python side is never cached, because it is the side research changes.
* **Eight battles is enough for "the day it lands"** because the tier's job is to catch a changed
  DEFAULT behaviour, which touches every decision; the MILESTONE tier is where rare shapes live.

### How a deliberate Python change propagates

1. **The gate compares under the production config.** A research change that only adds a flag,
   OFF in `production_config.json`, keeps parity by construction — research is not blocked.
2. **A change to default behaviour fails COMMIT the same day.** The author mirrors it in Rust in the
   same change when it is small (the expected case — an encoder cell, a tracker rule), or lands it
   behind a flag OFF by default. Either way the harness's **flag-coverage table** (every
   obs/tracker/event-affecting flag in `agents/model/flag_registry.py` + the training parser)
   marks each flag `core: implemented` or lists it in a declared **`RUST_CORE_PENDING`** set with a
   date. This is a set of FLAGS, never of divergences: the tiers run the production config and
   print the pending set on every run; **the CUTOVER tier refuses unless the set is empty** (every
   pending flag implemented in Rust or deleted).
3. **Layout changes propagate mechanically.** The Rust layout table is generated from
   `agents/observation/constants.py` and pinned by a routine test, so a new obs block fails that pin
   before it fails a byte comparison.
4. **Data changes propagate through `data/`.** The core's dex and belief tables read the same files
   the Python facade reads; a `tools/` regeneration is picked up by both.

---

## 4. The cutover's DELETION MANIFEST (one pass, after M6)

Each row names the milestone whose slice made it deletable. LOC from Phase 0 (d).

| from | deleted | LOC | notes |
|---|---|---|---|
| M1 | `agents/battle/event_fold.py` (`ViewEventFolder`) | 494 | the core's events serve every successor |
| M2 | `agents/battle/view_adapter.py` (`LiveView.from_view_json`, `ViewBattle`); `agents/training/view_successor.py`; the `view_pN` / `view_pN_at` / `pN_chunks` JSON of `search_driver` (`view.rs`'s JSON render); the `view_fallback_*` counters; `search_impl_parity.py`'s view allowlist entries | 617 + 500 | the view road |
| M2 | the PROTOCOL road: `obs_materializer.materialize_branches*`, `open_branch_fork`, `_PlayerSnapshot`, `_ReplayObsPlayer`; `--materializer` values `protocol` / `view` | most of 1,020 | the replay-for-counterfactual half of `obs_materializer` that the prober still uses moves to M7 |
| M2 | the view road's RUST half: `view.rs` (`one_sided_view`, `SideObservation`, the reveal fold with `BridgeChunks::observed` / `enable_view_fold`), `tests/one_sided_view_test.rs`, `search::Capture::views` + `Resolved::views_at`, `core_events --views`'s `views` / `truth` payload; `view_adapter.py`'s rules (`ViewBattle` survives — `core_successor` builds on it — though since M4 search's core road reads rows, not `core_successor`); `ViewEventFolder` use in `view_successor.py` (the fold's M1 row stands); the view half of `one_sided_view_parity_fuzz_test.py` and `rust_core_parity_views.py`'s PROJECTION column (the core column stays); `view_materialize_benchmark.py` | ~1,100 Rust + ~700 Python | made deletable by `materializer=core` (M2 adoption) |
| M2 | search's FALLBACKS: `search.py`'s `view_fallback_intermediate` / `view_fallback_no_payload` and the depth-≥2 protocol fallback; `RealizedWidths`' view counters | ~120 | a core leaf is a version, so there is nothing to fall back from |
| DONE (M4, 2026-09-24) | the TYPED SHORTCUT: `CorePath::Typed`, `BridgeSession::typed_side_lines`, `session_from_record_core` in search, `SearchConfig.core_path` (`--core-path`), and with them the INTEGRITY mode (`SearchConfig.integrity`, `--search-integrity`, `expand_many`'s `integrity`, `streams_equal`, `CoreIntegrityError`, `text_view`) | ~250 | measured to save nothing (§6); every version folds `parse(render)`, §6c's one observation path. KEPT: a CORE session's per-line engine SCOPE (`BridgeSession::side_scopes`) — no line, only the owner truth the native record's grouping gate and the parse-reproduces-step gate read |
| DONE `c97358e8` | the matching `agents/battle/poke_env_findings.py` entry (PE-V10 / PE-R1b / PE-V16), all deleted, registry empty | 1 entry each | a TRAINING-INPUT change, the owner's call; deleting the entry TIGHTENS slice V |
| M3 | `agents/training/clone_pins.py`; `ViewSuccessorFactory._clone_tracker`; `training/turn_delta_legacy.py` (test-only today) | 149 + 327 | the tracker fork becomes a pointer copy |
| M3 (owner, 2026-09-24) | `agents/training/turn_delta.py` (`TurnDelta` + `build_from_events`), `battle/turn_view.py`'s `TurnDelta`-only reads, `episode_tracker.build_delta*`, and the Python trackers of `episode_tracker.py` (`RecencyTracker`, `PairHistoryTracker`, `EventWindowTracker`, `EpisodeTracker.record_context` / `advance_window`), `progress_clock.py`'s obs half, `hidden_power_tracker.py`, `wish_belief.build_wish_pending`, `sleep_belief.build_sleep_sources`, `opp_intent_labels.build_opp_intent_label`; the unported `choice_band_tracker.py` (no production reader) | 603 + ~1,900 + 240 | slice T (the core's trackers, label, reward); `TurnDelta` survives until then only as the label's ORACLE. Joins the pass WITH the shaped reward path (its other reader) |
| M3 (decided 2026-09-23) | the SHAPED reward path: `reward_potentials.py`, `reward_bias_terms.py`, the shaped branches of `reward_manager.py` / `reward_composition.py`, and the inert shaped flags (into `designs/deleted_flags.md`), with their tests | ~1,100 + tests | the Rust reward is the win indicator (§2 M3). Until this pass, arm S stays re-runnable as a comparator |
| after M7 | RESHAPE `BoardReading`: its fields mirror poke-env's `Battle` (`_player_username` …) because slice V compares field by field; design the reading for the view once poke-env has no production user | — | a reshape, not a deletion |
| M4 | the Python pipeline's PERF layers — `observation/assembler.py` (incremental cache), the `live_view()` memo (`_state_epoch`, the request-change door), the `live_view_build_micros` memos | ~600 (assembler 498) | the Python encoder SURVIVES as the oracle; its perf scaffolding does not (a simpler oracle is a better oracle) |
| M4 | `utils/bridge/search_session.py`'s JSON protocol (`open_root` / `expand_many`), `search_driver`'s search verbs, `driver_timing.rs`; node `search_driver.js` / `replay_driver.js` / `replay_kernels.js` once nothing diffs against them | 405 + 785 (node) + 128 + the driver verbs | search runs in-process on versions. **Status (M4, 2026-09-24): OFF search's per-arm path** — a core arm's reply is the leaf's ENCODED ROW + mask + choice tokens (`expand_many`'s `rows`), so the view / legality / events JSON, `CoreSuccessorFactory`, `core_view` / `view_adapter.ViewBattle`, the Python trackers and the Python encoder no longer touch a search successor, and the core road opens no Python prefix fork. The protocol verbs themselves remain (the in-process `successors()` is M5's transport decision); the JSON core payload and `core_successor.py` stay, unused by search, until the pass |
| M5 / T2 | `--compile-opponents`, `--compile-opponents-preload`, `--compile-opponents-strict` (+ their `--no-` forms); `agents/model/compile_opponents.py`, `compile_preload.py`, `compile_prewarm.py`; the per-env model cache in `snapshot_pool.py` | 673 + the cache | opponents are T2 catalogue entries |
| M5 | `SubprocVecEnv` construction in `main/train/env_factory.py`; `agents/training/async_vec_env.py` and `--async-rollout` ("the batch is whatever is ready at the flush" survives as T2's timer); the per-env bridge child for training (`utils/bridge/bridge_session.py`, `battle_stream_client.py`); `--use-bridge` (`node`/`off` for training) | 287 + 810 + the factory half | the forkserver and the per-env process retire |
| M5 | `agents/training/gen3_env.py` + `wrappers.py`'s per-env opponent plumbing | most of 1,711 | the env is a struct in the Rust env process |
| M7 | the Python ladder client path in `play.py`; the prober's trace re-parse | — | after M7 poke-env has no production user |

**Survivors (named oracles, not debt):** `src/poke_env/` + `agents/battle/gen3_battle.py` +
`battle_event.py` + `live_view.py` / `turn_view.py` (event and view parity); the Python encoder until
two full seeds are byte-clean post-cutover (design §9 Q3 then decides); node Showdown,
`local_sim_bridge.js` and the differential fuzzers + e2e capstone (they gate the simulator);
`ws_frontend` (the third-party transport).

---

## 5. Findings Phase 0 hands to the plan

| id | finding | where it lands |
|---|---|---|
| F1 | The port PANICS at 1,000 committed turns (`turn/driver.rs::BATTLE_TURN_CAP`); the pinned Showdown TIES at turn > 1000 and emits `\|bigerror\|` auto-tie warnings from turn 500 (`sim/battle.ts:1836-1848`). 3 of 300 random-player battles reached it | M1 prerequisite (port fix, byte-gated by a scenario golden) |
| F2 | `ViewEventFolder._apply("turn")` does not reset `_current_move_user_side`, which poke-env's `end_turn` does (`abstract_battle.py:1634`). Reproduced on a 7-line protocol: a start-of-turn Intimidate→Clear Body `-fail` reads `side=ours` on the fold vs `None` live. Reachable at depth ≥ 2 (`branch()` after a folded ply) — `SearchConfig.max_depth` defaults to 3, and `event_fold_parity_fuzz_test` seeds at depth 1 only | M1 prerequisite (two-line Python fix + a depth-2 case in the fold gate) — or a tech-debt row now |
| F3 | The event schema drops source facts (§0 row 2) | M1 — the core carries truth + reading |
| F4 | Tier 2's CPU saving is 0.4–2.1 cores at the live rate (§0 row 4) | T2's case is restated: unification, search leaf, prerequisite of M5 |
| F5 | The design's §1 table says "~89 % gradient, ~11 % rollout" while the runbook's compile measurement moved end-to-end FPS **+33 %** from the opponent forward alone — both cannot describe the same shape | measure at M5 before claiming a throughput number |

---

## 6. Search in the end state — what `successors()` must leave room for

Search is IN the end state and is not optimised for today (it is not providing value now because
of low discrimination between states). The interface decisions that keep a leaf pluggable later:

* `successors(side, k, leaf: Option<&dyn Leaf>)` returns version HANDLES plus, when a leaf is given,
  the leaf's scores — so a tree is versions + a batch, and no successor is ever serialised.
* `Leaf` is a trait whose one implementation today is "encode rows → T2 `score(model_id, rows,
  masks, mode=value)`"; a rollout leaf (the 1,222 CPU-h battery the design prices) is a second
  implementation stepping versions to terminal, batching its forwards through the same T2 call.
* Determinization stays in Python until a milestone measures it: a world is a version built from a
  different opponent team, which `BattleVersion` supports by construction (the view of `side` does
  not depend on the unrevealed opponent mons).
* Every search number is stamped with the road it ran on (`materializer`), so the M2 adoption is
  visible in every table after it — and, since M2, with `core_path` and `integrity`.

**M2 RECORDS (2026-09-23, `research_state/measurements/rust_core_m2_2026-09-23/`).**

* **Fork / succession cost, core vs view road** (interleaved, one road per process, the same 10
  banked decisions of `ai_v12_02_winprob_critic`). Wide B (684 arms, honest arm, m_opp 3,
  k_worlds 4; load1 10.4–24.6): the core road's Rust `expand_many` costs **1.41× the view road's per
  successor** [1.36, 1.48] — the version's stream fold, ≈ 0.08 ms/arm — and the whole searched
  DECISION is **0.926×** [0.920, 0.944]: Python no longer re-derives the view or re-parses the ply.
  B = 1 (10 arms; load1 15.8–31.1): 1.44× per successor [1.24, 1.69], 1.116× per decision
  [0.956, 1.276] — not resolved.
* **B1 — what the TYPED SHORTCUT saves: nothing measurable. A DECISION INPUT.** Typed at the source
  vs the full text path (render → parse → the stream-only fold), same method: the per-successor
  fold typed/text **1.019** [0.983, 1.316] at wide B and 0.912 [0.819, 1.087] at B = 1; the whole
  Rust `expand_many` per arm **1.089** [1.048, 1.229] at wide B — the typed road is DEARER, its
  session recording a source record for every line; per decision 1.015 [1.004, 1.421] (wide) and
  0.989 [0.556, 1.220] (B = 1); the shortcut's share of the decision wall **−0.1 %** [−1.1, +0.1].
  Why: a successor's cost is the board-reading clone + the fold, identical on both paths; `Line::parse`
  of a ply's ~20–30 lines is a few µs. **Recommendation: DELETE the shortcut** — make
  `core_path=text` the only path (it is §6c's observation path everywhere else), and with it
  `CorePath::Typed`, `typed_side_lines`, the search session's source recording and the integrity
  mode (§4). Not done in M2: the default stays `typed` until the owner rules.
* **B2 — the INTEGRITY mode.** `SearchConfig.integrity = N` (`--search-integrity N`,
  `expand_many`'s `integrity`): every Nth core arm is also folded from its text; Rust asserts the two
  VERSIONS equal (`streams_equal`: board reading, events, view) and Python the encoded obs bytes and
  mask (`CoreIntegrityError`, naming the decision, the depth and the first differing obs block).
  **ON (N = 1) in `materializer_parity_integration_test`, `one_sided_view_parity_fuzz_test` and
  `core_successor_test`; OFF (0) by default in production search**; a number from a sampled run is
  stamped "integrity-sampled at 1/N". Mismatches found: **0** (teeth:
  `version_test::the_typed_shortcut_and_the_text_path_fold_the_same_version`'s different-successor
  case and `core_successor_test`'s tampered `text_view` each fail).

---

## 6a. Worlds: building a board for search from ONE side's view, and the ORACLE that grades it (added 2026-09-23, owner)

> **DEFERRED (owner, 2026-09-23): "kick search determinization down the road."** Nothing in this
> section is scheduled: no world sampler, no construct entry point, no oracle battery. It is recorded
> so the core leaves room for it. M2's search work is unaffected: search INTEGRITY (strict mode,
> `parse(emit(step)) == step`) and the shortcut-vs-text measurement are correctness of the search we
> have, not determinization. **The learned-sampler option, recorded for when this returns:** the
> production model already predicts every hidden team fact as a MARGINAL (`BeliefHead` species,
> `MoveBelief`, `HPTypeBelief`, `ItemBelief`, `SpreadBelief`; ARCHITECTURE.md §2, §7), supervised by
> the true opponent team. A determinizer would add (i) a JOINT, slot-by-slot sampler using those
> heads as the proposal, filtered for consistency as in step 2; (ii) the non-team hidden state
> (counters, exact HP given a spread, PP) by RULE, never learned; (iii) a fresh RNG seed per world.
> Its known hazard: the labels come from the 719-team pool, so the heads can MEMORISE pool teams.
> That bias is sanctioned for the policy, but it would make every search world a pool team. It must
> be graded on held-out LADDER replays (e.g. Metamon's `hl_05_26` gen3ou set) and by the oracle
> yardstick below, against the Smogon-prior sampler.

Search needs a full board, but a side only has its view. Today's search (`search_dividend/determinize.py`)
builds worlds by **swap-and-replay**: never-revealed opponent slots are replaced with pool-consistent
donors (gender-matched so the PRNG draw count holds), the battle is REPLAYED from turn 0 with the same
seed and choices, and a world is kept only if our side's protocol comes out byte-identical. This is
exact, but it needs the battle's SEED and its input log, and "uniform over consistent pool teams" is
the true posterior only because eval draws opponents from our pool. **Neither holds on the ladder.**
It also carries a measured, negligible leak: a revealed mon keeps its TRUE EVs/nature/item/ability.

**The ladder path, CONSTRUCT-FROM-VIEW (not yet planned; the named prerequisite for any ladder search):**
1. The parse-built partial version (M2) holds every known fact, including our whole side from `|request|`.
2. A **world sampler** fills the unknowns from SMOGON-derived priors (the owner's rule), conditioned on
   the reveals (teammate joint priors, as `ou_random_teams.js` already uses; moves, items and spreads;
   optionally sharpened by the model's belief heads), then filters by CONSISTENCY with the evidence: an
   observed damage number the sampled spread cannot produce, a speed order it contradicts, an HP% that
   does not round from the sampled exact HP. Hidden counters are sampled by the game's rules (sleep
   turns left given turns slept; confusion, Encore, Disable, Taunt; a pending Wish or Future Sight).
3. A **mid-battle CONSTRUCT entry point in the port**, where every HP, status counter, boost, volatile,
   PP count and field condition is SETTABLE, with a fresh seed. The port builds only from turn 0 today.
4. K worlds searched and combined at the root, belief-weighted, with strategy fusion named as the known
   weakness (information-set search is the mitigation).
Cross-checks, in our own battles where the truth exists: **construct(the TRUE full state) must
reproduce the real version exactly**, which proves the entry point sets every piece of state; and
construct-from-view worlds are compared in distribution against swap-and-replay worlds.

**The ORACLE yardstick (first-class, never on the ladder).** For any searched decision in our own
battles, run the same search on the TRUE board (one world) beside the determinized one. Today's
arms already include `oracle` and `playoff` in `TRUE_WORLD_ARMS`; the core makes this a standing
measurement, not an arm choice. Report three numbers:
- **decision agreement:** honest vs oracle, on the same decision;
- **value regret:** the true-world value of the oracle's choice minus the true-world value of the
  honest choice;
- **the dividend gap in games:** oracle arm vs honest arm at matched budget.
That gap IS the price of hidden information plus our determinization's error, so it measures the
world sampler directly (and separates it from the irreducible hidden-information floor, probed
2026-08-22). **The ladder cannot use it BY TYPE:** the oracle needs a step-built version holding
the board, and a parse-built (ladder) version has none, so an oracle search is unconstructible
there. Every search number that used the true world is stamped `world=oracle`, which includes any
`playoff`-arm number, since that arm inherits the oracle's world.

## 6b. Which source is authoritative — and playing against a REAL Showdown server

**The truth is "what happened in the battle".** It has two representations: typed events (the
structured form) and protocol text (the serialised form). Which one is AUTHORITATIVE depends on who
runs the battle:
- **Our simulator runs it** (training, self-play eval, search): the transition is the authority.
  The typed events are emitted AT the transition; the text is a rendering of them and is produced
  only when something needs it (a record, a gate, a websocket client).
- **Someone else's server runs it** (the ladder, a third-party anchor): the server is the
  authority, and all we receive is ONE SIDE's text stream. The core `parse`s it into the same typed
  events. There is no omniscient board: the opponent's unrevealed mons and exact HP (Showdown sends
  the foe's HP as a percentage) are unknown. So a parse-built version carries the per-side view
  and the events, and its omniscient board is partial by construction. That is enough, because
  the encoder reads only the side's view.

**Why the two cannot disagree: a chain of two proven equalities.**
1. *Our emitted text == Showdown's text for the same battle.* This is the port's existing contract,
   proven byte-for-byte against real Node Showdown by the protocol-emission phases, the e2e capstone
   and the differential fuzzers.
2. *`parse(our emitted text) == the typed events we emitted.*` This is M1's round-trip gate, at all
   three tiers.

Together: parsing Showdown's text yields exactly what our simulator would have typed for that
battle, so a model acting on the ladder sees the same events it trained on. **The link the chain
does not cover is server VERSION drift**: the public server runs Showdown master, while we pin a
commit. That is covered by three guards: `ladder_drift_scan` before every live session; the parser
REFUSING an unknown keyword (by design; a silent skip would be worse than a lost game); and the
Showdown version stamped into every online record, so a later re-parse knows which dialect it reads.

## 6c. DECIDED (owner, 2026-09-23): the OBSERVATION always comes through the parser

Training, evaluation and online play build the observation by the same path: **per-side protocol
text → `parse` → the reading → the view → encode**. Search's successors keep the typed-at-source
shortcut (it needs the omniscient board to step, and it is the hot path), and M1's
`parse(emit(step)) == step` gate is what licenses that shortcut. Why: one observation path
everywhere means the ladder's parser is exercised on every training decision, not only on the
parity corpus. Cost: M1 measured `parse` at 21.6–30.8 µs per decision, ≈ 0.03 % of the live arm's
step, and 60–70 % of that is the `|request|` JSON, which the current path decodes on every decision
anyway, so the marginal cost is smaller still. It binds M2–M5: the core's env shape (M5) produces
the training observation by parsing its own per-side stream.

## 7. Which path a research number ran on, milestone by milestone

| landing | training numbers | search numbers | offline meters / anchors |
|---|---|---|---|
| M1, M3, M4 (built alongside) | unchanged — Python path | unchanged | unchanged |
| M2 search adoption (ledger entry) | unchanged | `materializer=core` from that entry | unchanged |
| T2 non-training adoptions (one entry each) | unchanged | the leaf on T2 | each meter stamps `inference: service`; greedy actions proven byte-identical, so the numbers are value-neutral |
| **M6 CUTOVER** (ledger entry, between reads) | **every number after it runs on the core**; `metadata.json` records the env core + commit; a resume pinned to a pre-cutover commit stays on Python (the launcher's pin), so no run spans the cutover without a re-pin | core | core for training-side evals |
| M7 | — | — | anchors / ladder stamp `client: core` |
