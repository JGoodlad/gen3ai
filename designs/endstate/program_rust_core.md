# Program — the Rust core: milestones, gates, and the one cutover

🚨 **ALWAYS-CURRENT WHILE BEING IMPLEMENTED (owner, 2026-09-27).** Any build that differs from this program (a milestone's scope, a lane plan, a gate, the deletion manifest) updates this doc in the same commit, saying what changed and why. The Q-head / opponent-pointer spec is its sibling, [`design_q_head.md`](design_q_head.md).

**Status: PLAN, authored 2026-09-23 at the owner's request (Phase 0 of the RUST CORE PROGRAM).
M1–M4 BUILT; M6 CUTOVER DONE (`ac0b6469`) and deletion pass part 1 DONE; M5 Phase A DONE (`17757d09`), the M5 build GO'd by the owner for Mon 2026-09-28 ~22:00; M7 after. The per-emission EMISSION SELF-CHECK (§3) BUILT, ON in every test and fuzzer build, compiled out of production.**
ALWAYS-CURRENT (see the banner above; owner, 2026-09-27 — this replaced the old "explicit-only" rule). It implements
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
> proven. **M5 was not to start until the owner said so — the owner GO'd the M5 build on 2026-09-27 ("launch M5 on Monday if we have quota", then "bump launch to Monday night, say 10pm").** T2 (the inference tier) is M5's prerequisite,
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
contiguity REFUSED (`gen3_core_obs_wire_v1`). **Slice O green at COMMIT** (2,081 decisions, both
viewers, every row BYTE-equal; the obs golden reproduced hash for hash) **and MILESTONE** (at
`34da7225`: 194,304 decisions over pool random, `production` policy and the LADDER-USAGE corpus, every
row byte-equal, 1.15 M choice tokens equal; RE-RUN on the new training-input boundary `b6dfd7e8` (the
M3 GIGO fix, its event-window change mirrored in the encoder): 10 / 10, 194,606 decisions byte-equal,
the regenerated obs golden reproduced; fresh battles incl. PROCEDURAL teams: 77,475 + 66,818
decisions, 0).
`obs_build_benchmark.py` has its core row (8–12× the Python production encode). **Search takes rows** (`expand_many`'s `rows`: each core arm's leaf is encoded on
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

**T2 DESIGN (M5 Lane T2, 2026-09-29; progress and resume point:
[`research_state/measurements/m5_t2/PROGRESS.md`](../research_state/measurements/m5_t2/PROGRESS.md)).**
The owner's 2026-09-27 decision (ledger L21431 (2)) makes T2 UNIFIED inference: the trainee, the
policy opponents and eval all forward through one service of fixed GPU weight SLOTS, fixed-shape
BUCKETS and PRIORITY classes. Package `src/agents/inference/service/`; standalone — nothing in
training calls it until Lane G. Measured facts it rests on (RTX 3080 Ti, torch 2.5.1+cu121, idle GPU,
`ai_v14_06_lbat_ctrl_fix` final, the 64-row real-obs fixture, 2026-09-29): the production policy
has **3.07M parameters (12.3 MB) + 0.97M buffer elements** — a slot is ~16 MB, so slots are cheap;
an EAGER decision forward costs **~28–31 ms per call at ANY batch 2–128** (flat: launch-bound, ~11k
ops), while the Inductor-compiled forward replayed as a CUDA graph costs **1.06 / 1.23 / 1.97 /
2.94 ms at B = 2 / 8 / 48 / 128** (23 µs/row at 128; 14× eager at 48, 26× at 2), within 1e-5 of
eager on legal log-probs and 7e-7 on V. Because a replay is nearly flat in B, padding is cheap and
the bucket set should be COARSE.

* **The decision forward** (`decision.py`). One pure callable per policy, `DecisionModule(obs [B,D]
  f32, mask [B,A] bool) → (logp [B,A] f32, V [B] f32)`: the policy's own extractor, towers, pointer
  head and `_critic_value` (every critic mode), with sb3's two-step mask normalisation written out
  in the same order, so it is **BIT-IDENTICAL** to `MaskableCategorical`'s log-probs on the same
  device (measured, CPU). Illegal entries are `-inf`. Greedy = `argmax(logp)`. Sampling is the
  CALLER's, from `logp`, with its own RNG. An architecture whose extractor reads a Dict key beyond
  `observation` (`extra_obs_keys`, today `--value-true-team`) is REFUSED at declaration.
* **Slots.** A SLOT GROUP is one architecture (the state-dict signature: keys × shapes × dtypes)
  with `n_slots` declared at startup. Each slot is a private replica of the group's template
  policy whose persistent parameters and buffers are VIEWS into one STACKED `[n_slots, …]` tensor
  per state-dict entry, allocated once at startup. `load(slot, state_dict, model_id)` is an
  in-place `copy_` into those views — no allocation, no recompile, and a captured CUDA graph (which
  holds the storage pointers) serves the new weights on its next replay. A signature mismatch is
  a typed `SlotArchMismatch`. Every load is parity-verified at one bucket against the eager
  reference of the SAME weights (a miscompile can be weight-dependent: the 2026-09-28 defect showed
  on real weights), so every weight set served has passed the gate. A single MULTI-SLOT forward over
  the stack (`torch.func.vmap`) is **NO-GO today** (measured): the extractor writes in place into
  tensors it creates inside the forward (`t0_species.species_team_prior_logits`'s
  `zeros(...).scatter_(src)` is the first of many such sites), which vmap refuses; making it
  vmappable is a model-wide out-of-place rewrite, not a T2 change. Several small slots share the
  GPU through LANES instead (below).
* **Buckets.** A declared ascending tuple of batch sizes, **minimum 2** (torch 2.8 cannot lower a
  batch-1 CUDA graph of this extractor — K1 finding; and a batch dim of 1 is a degenerate
  broadcast). A slot's pending rows are packed into chunks of the largest bucket and a remainder
  padded up to the smallest bucket that holds it; pad rows repeat a REAL row of the batch and
  their outputs are discarded. Default `(8, 48, 128)`; a service declares its own. Rows are
  independent in this forward up to float rounding (measured on CPU: 8 rows alone equal the same
  rows inside 64 bit-for-bit, while a padded batch differs from an unpadded one by ≤ 4.8e-7 on
  log-probs), which the parity gate re-proves per bucket with a partially-filled batch.
* **API** (what Lane E's opponent routing and Lane H's eval call). `svc = InferenceService(spec)`
  → `svc.startup()` (build, warm, capture, parity-gate, FREEZE) → `slot = svc.slot(group, i)`;
  `svc.load(slot, state_dict, model_id)`; `ticket = svc.submit(slot, obs, mask, priority)` (numpy
  or tensor rows; any count); `svc.flush()`; `ticket.result() → Decision(logp, value, greedy)`;
  `svc.score(...)` = submit + flush for a synchronous caller; `ticket.host()` → the same three as
  NumPy views into a PINNED host output arena. Results are VIEWS into declared arenas, valid until
  the next flush (the caller copies what it keeps), so a flush allocates nothing. `submit` COPIES
  the rows: a caller's rows are typically a view into a buffer its env core rewrites on the next
  step (Lanes A/B's column mappings). Requests are SLOT-TAGGED; a flush packs every admitted row
  grouped by slot.
* **Staging (unit 4).** A flush packs its rows into one of TWO pinned host input arenas
  (alternating per flush), issues ONE asynchronous host-to-device copy of the whole flush, and ends
  with ONE device-to-host copy of every output; per bucket chunk the engine only does device-side
  copies into the static inputs, the replay, and a device-side copy out. Before re-packing an arena
  the engine waits on the event recorded after that arena's last copy, so the host never rewrites
  rows a queued copy has not read yet (the flush itself never blocks the host on the GPU).
* **Lanes (unit 3).** `lanes` (declared, default 1) CUDA streams, each with its OWN graph memory
  pool and static inputs; slot `s` runs on lane `s % lanes`, so graphs that share a pool never run
  concurrently. Each lane's graphs are WARMED AND CAPTURED ON THAT LANE'S STREAM: torch's default
  capture stream is one class-level stream, cuBLAS keys its workspace by stream, and graphs
  captured there share one workspace — two lanes replaying together then served max|Δ logp|
  **0.048** (measured; `service_cuda_test`'s regression FAILS on the reverted code). Measured
  (2.5.1, idle, 8 slots × bucket 8, 48 rows = Lane E's shape, `service_benchmark.py`): **9.89 ms
  per flush on 1 lane, 3.95 ms on 4 lanes**; the raw replay probe reads 9.81 / 6.79 / 4.26 / 2.58 ms
  for 1 / 2 / 4 / 8 lanes (8 slots), parity unchanged (7.2e-7), graph memory 212 → 399 MiB. Small
  batches leave most SMs idle, so concurrent lanes are nearly free throughput.
  v1 is an in-process object driven by the caller's loop (the M5 core is in-process over FFI); a
  timer-flushed thread or process front end is added only when a consumer needs one.
* **Priority.** `ROLLOUT` > `EVAL` > `FILLER`. `flush()` serves EVERY pending `ROLLOUT` row, then at
  most `filler_batches_per_flush` (declared, default 1) bucket batches of the lower classes, oldest
  first — eval rides along as background filler and never delays a rollout step by more than one
  batch; `drain()` serves everything (eval's own end-of-cycle).
* **The lifecycle** (the M5 DESIGN PRINCIPLE). `DECLARED → startup() → FROZEN`, and `POISONED` on a
  violation (every later call raises `LifecycleViolation`). Startup acquires everything: the slot
  replicas and stacked storage, the two pinned host input arenas, the device input and output
  arenas and the pinned host output arena, per lane its stream, graph pool and static bucket
  inputs, one compiled forward per bucket (or, `aot`, one package per group × bucket), one CUDA
  graph per slot × bucket in its lane's pool, and runs the parity gate; then it freezes. After the freeze a flush only copies in, replays and
  copies out. Counters that must stay 0: `compiles_after_freeze` (dynamo graph count),
  `captures_after_freeze`, `cuda_segments_after_freeze` (`torch.cuda.memory_stats`
  `segment.all.allocated` across the service's own calls); a non-zero one is a typed
  `LifecycleViolation` at the point it happens. Nothing is lazily compiled: an undeclared bucket
  cannot be reached (rows are packed into declared buckets), and a new slot cannot be added.
* **Validation — the same bars as the compile gate.** At startup, for EVERY slot × bucket, the
  committed real-obs fixture (`compile_parity_obs.npz`; full bucket AND a partially-filled one)
  through the backend vs the policy's own eager sb3 path on the same weights and device
  (`decision.policy_reference`), judged by `compile_trainer.decision_verdicts`: legal log-probs
  1e-3, V 1e-4 at fp32 (the TF32 rule against an fp32 reference otherwise); illegal entries
  exactly `-inf`; greedy equal on every row whose eager top-2 legal margin exceeds the log-prob bar
  (near-ties are counted and reported, never silently passed — a fresh or real policy has EXACT
  ties, measured). The same check runs on every `load` and as the K6 IN-RUN CANARY
  (`svc.canary(slot)`, the caller's cadence). With more than one lane, startup and a full canary
  also run a CONCURRENT gate: one flush carrying every slot's rows, so every lane replays at once,
  each slot judged against its own eager reference (the per-slot gate runs one slot per flush and
  cannot see lanes interfere). A failure is a typed `ParityFailure` that names the slot, bucket
  and quantity.
* **Backends — decided by measurement on 2.5.1 (the production env).** `eager` (the reference;
  CPU and CUDA) and `graph`: `torch.compile` (Inductor, `dynamic=False`, one compile per bucket,
  reused by every slot of the group — measured: a second module instance of the same class does
  not recompile) captured into a `torch.cuda.CUDAGraph` per slot × bucket. **AOT
  (`torch.export` → AOTInductor) is BLOCKED on 2.5.1**, three causes, all measured: (1) an AOT
  artifact is ONE graph, so the `6521f420` trunk split (a dynamo graph break) cannot exist in it,
  and the unsplit graph reproduces the 2.5.1 Inductor miscompile — max|Δ legal log-prob| **0.68**,
  |ΔV| 0.012 on the real checkpoint at B = 8 (0.21 on V on a fresh one); (2) the box has no CUDA
  toolkit (AOTInductor's C++ build needs `CUDA_HOME`; a shim from the pip/Triton headers works);
  (3) `x[...] = 0.0` in `extractor_ctx.slice_pokemon_categoricals` exports as `fill.Tensor` of a
  lifted constant, which 2.5.1 AOTInductor cannot lower (an FX rewrite to `fill.Scalar` fixes it).
  With (2) and (3) worked around AOT is 1.86 ms at B = 8 — no faster than the graph backend's 1.23.
  **On torch 2.8.0+cu126** (`gen3ai_torch28`, owner 2026-09-29: install the headers) AOT WORKS
  once the CUDA 12.6 headers are in that env (`cuda-cccl`, `cuda-cudart-dev`,
  `cuda-crt-dev_linux-64`; `environment_torch28.yml`), and no FX rewrite is needed. It is the
  backend `aot` (≥ 2.8 only; refused on 2.5.1 with the reason): one package per slot group ×
  bucket with the WEIGHTS AS INPUTS, because a package that embeds its weights also bakes
  weight-derived folded constants that `load_constants` does not reach (swapping a perturbed policy
  into a real checkpoint's package served max|Δ logp| 0.076 / |ΔV| 0.026; runtime constant folding
  failed to load). Measured, idle GPU, real checkpoint, 1 slot: parity **1.9e-5 / 7.7e-7**
  (a perturbed fresh policy through the SAME package: 8.3e-7 / 4.5e-7); **1.91 / 2.31 / 3.31 ms**
  per flush at B = 8 / 48 / 128 vs the graph backend's **1.25 / 1.81 / 2.76** on the same torch;
  build ~77 s per bucket (graph ~51 s); package ~15 MB; load 0.01–0.09 s. It is slower because an
  AOT call cannot be captured as a CUDA graph (its model container synchronises on its own
  run-finished event: "operation not permitted when stream is capturing"), so it stays
  launch-bound. What it buys is a **PYTHON-FREE artifact**: a 60-line C++ program linked against
  libtorch only (no libpython) loads the package in 131 ms and reproduces eager at 6e-6 / 1.8e-7
  (`measurements/m5_t2/probes/cpp/`) — what a Rust caller of T2 would load. So `graph` stays the
  default on BOTH torches (faster), and `aot` is the ≥ 2.8 path for a Python-free consumer. The K6
  lock is structural for both: after the freeze dynamo is never
  entered (a replay is not a dynamo call). **Two capture blockers**, measured: the EAGER forward
  makes **24 host syncs** (Python-scalar `index_put`s and host-built constants in `extractor_ctx`,
  `damage_op*`, `team_transformer`), so eager cannot be captured; and the compiled forward had ONE
  (`team_transformer._event_reference_cells` built a constant on the CPU and copied it each call —
  also what made Inductor's own `reduce-overhead` mode skip cudagraphs and run at 2.1 s/call). That
  one is fixed at its source (built on the device; eager bit-identical). Dynamic-shape compile
  (`mark_dynamic`, one compile for all buckets) FAILS on 2.5.1 (a sympy assertion), so startup
  pays ~75 s per bucket; the K3 per-run cache is what makes a restart cheap.

### M5 — N envs per process, successors, the Rust env (Tier 1, env shape)

**Status (2026-09-26): PHASE A DONE — the transport is decided and the build is planned. Nothing
is built in `src/`, and no production byte changed.** The owner's direction the same day: do not
pick one transport. Carry BOTH front ends over ONE env core, because the consumers differ. The
measurement record and the prototype (a std-only crate + two thin front ends + the parity and bench
scripts) are in
[`research_state/measurements/rust_core_m5_transport_2026-09-26/`](../research_state/measurements/rust_core_m5_transport_2026-09-26/README.md).
The prototype is the SEED of Lane 0 below. It stays in the measurement directory, and Lane 0 moves
what it keeps into `src/`.

**What crosses.** ONE Rust env core stepping N battles on its own worker threads. It writes the obs,
mask, need, reward, done and label COLUMNS into caller-provided buffers. Policy opponents are on T2:
their rows go out in the same batch, and Rust never runs a network. `successors(side, k)` means
fork, step, present and encode, **plus an optional leaf hook** (§6). The core has TWO thin front
ends: a C-ABI `cdylib` (FFI, loaded with `ctypes`) and a separate env process over `/dev/shm` (one
opcode byte per batch). **Neither front end holds battle logic.** Both forward to one
`core::dispatch(op, cols)`. Built alongside, the core is selected only by the parity and throughput
harnesses, never by a research arm, until M5's gate.

**THE TRANSPORT BENCHMARK (registered above; measured 2026-09-26 on a quiet box).** Contention
factor 1.0; the lineage `ai_v14_01_base` had not launched. Interleaved A B / B A pairs, 95 %
bootstrap CIs, the rule "warn, never stretch". Both front ends ran the same core on real battles:
the bridge corpus's 46 gen3ou teams, the production parse chain and encoder, and seeded random
policies.

| consumer shape | FFI | process | proc / FFI |
|---|---|---|---|
| training, N = 48, T = 8, both sides' rows out | 12.14 µs/decision [11.60, 12.41], 82 k/s | 11.96 [11.61, 12.31] | **0.994 [0.943, 1.026]**, not detected |
| eval, N = 16, T = 4 | 19.92 [19.82, 19.95] | 20.10 [20.02, 20.21] | 1.012 [1.006, 1.014] |
| search, `successors(64)` | 8.72 ms/call (136 µs/successor) | 8.46 | 0.952 [0.932, 0.965] |
| one-off: start-up to the first row / N = 1 latency | 24.1 ms / 64.5 µs | 26.1 ms / 62.3 µs | +2.0 ms / 0.972 |

- **Transport per batch:** 6.9 µs (FFI) and 14.1 µs (process), against 1.05 ms of core work at
  N = 48. At most 1.4 % either way. **The transport does not decide M5.**
- **Parity:** the two front ends, T = 1 vs 4, and a rerun are **byte-identical** on every column at
  every step, and so are `successors` rows.
- **Scaling:** the core scales 5.8× at T = 8.
- **Against today's path:** 69 µs of single-thread CPU per decision against today's 1.07–1.19 ms
  full cycle (cutover §6). That is ~15–17× less CPU per decision, and it is an upper bound: the
  prototype has no labels and no stall bookkeeping.
- **GIL:** both front ends release it; both pay the same GIL re-take on return.

**Defaults per consumer (the rule: PROCESS when the Python host holds state a core fault must not
destroy; FFI when the host is disposable):**
- **Training rollout: process.** The learner's optimizer, rollout buffer and GPU context survive a
  core abort, and a dead child is a typed error plus a respawn.
- **Eval: process** inside the trainer; **FFI** inside a disposable eval worker.
- **Offline search, probes, meters, reruns: FFI.** 2 ms faster to the first row, no child or shm to
  reap, one process under a debugger, version handles as pointers.

**The build stamp covers BOTH front ends.** The 09-09 incident was a stale binary; a stale `.so` is
the same class. The stamp is the commit plus a hash over the `(path, git-blob-id)` listing of every
source that reaches the build. Both front ends REFUSE a mismatch at load, and the teeth were proven
(2.7 ms at import). **Carrying two front ends instead of one costs +1.5–2 agent-days** (Lane B and
the FFI == process gate, ≈ 6–8 % of M5). One dispatch entry keeps the running cost near zero.

**THE INVENTORY — what must move for Python to leave the per-decision loop** (Python LOC today):

| item | today (Python) | depends on | Rust status | parity gate |
|---|---|---|---|---|
| trainee obs row | `sim_bridge` `core_obs` → `Gen3Env._core_row` (Python still encodes the non-decision embeds) | — | **DONE** (M4 / M6) | slice O + slice N (existing) |
| mask + mapper (action index → choice token) | `action/mapper.py` 252, `mask_generator.py` 98, `serialize.py` 99, `LegalActions` | the reading | **DONE in the core** (`present::legal_actions` / `mask` / `choice_tokens`, compared token for token by slice O); the env takes action INDICES | slice O tokens + slice N |
| tracker fold (recency, pair history, 32-row window, progress clock, HP / wish / sleep beliefs, α/β label, `TurnDelta` consumers) | `episode_tracker.py` 706, `turn_delta.py` 637, `progress_clock.py` 526, `hidden_power_tracker.py` 280, … (~2,985) | the event stream | **DONE** (M3, slice T), per side on the version | slice T + slice N |
| reward (win indicator) + episode end | `reward_manager.py` 178 (the shaped path DELETED, `e3ef16db`, v122), `stall.py` 62 (forfeit at `StallConfig().threshold`), `Gen3Env.step` / `reset` bookkeeping | the engine's winner, the turn count | **DONE in the env core** (M5 Lane D, `src/rust_env/src/episode.rs`): the terminal reward per the spec's `terminal`, `terminated` / `truncated` (raw `calc_term_trunc`), the stall forfeit before any feed, ties; the terminal observation is DECLARED ABSENT (production never consumes it) | slice N reward / `terminated` / `truncated` — PASS both front ends (`rust_env_episode_parity_test.py`) |
| **training labels** (production keys) | `observation/belief_labels.py` 353 + `Gen3Env` label methods ~300 (`_belief_labels`, `_spread_labels`, `_nature_ev_map`, `_hp_type_labels`, `_item_labels`, `_merge_training_keys`) + `belief_tables.true_nature_ev_label` | the OPPONENT's TRUE team (the Rust env holds both packed teams and the engine board); `species_known` read from the row (revealed-first slot packing); the species / move / item NUM tables; `win_margin` reads `reward_manager._last_material_margin` (`material_margin.py` 50, a by-product of the live view, not a reward term) | NOT BUILT (the INVENTORY is: 30 keys `Gen3Env` can emit, **21 in production** — `src/utils/rust_env/label_inventory.py`, pinned against `Gen3Env` by a routine test, doc `designs/rust_sim/env_labels.md`). 18 CORE-built: `belief_species` / `belief_moves` / `known_moves`, spread / nature / EV (+ masks), `hp_type_label` / `item_label` (+ masks), `win_margin` (port `material_margin.py`), and the four **intent** labels (`opp_action_kind` / `_num` / `opp_switch_slot` / `_species` — ON in production: `opp_intent` true ⇒ `--arch production` sets `opp_intent_coef` 0.05; built on the port's `trackers::IntentLabel`). HOST-filled: `win_target` / `win_mask` (zero placeholders, back-filled per ROLLOUT by `WinProbLabelCallback`, which stays Python) and `opp_class` (per episode, from the host's routing state). Off the production surface: `win_row_w` / `fork_pg_m` host constants; defensive / bait opportunity, distill mask, true team, dense aux a typed REFUSAL in the Rust env until ported | slice N: every key the production config emits, per decision, equal to `Gen3Env` |
| policy-opponent encoding + forward (T2) | `RLPlayer.embed_battle` (`inference/player.py` 770) with the assembler, `compile_opponents.py` / `compile_preload.py` / `compile_prewarm.py` 673, `snapshot_pool.py` 505's per-env model cache | T2 (the inference service); an opponent's arch MUST equal the core's obs layout (`MIGRATION_FLOOR`) | the opponent row is DONE (the core encodes either side). T2 BUILT; the ROUTING BUILT (M5 Lane E): the core writes `opp_slot` per env, the host submits p2's rows grouped by slot, one flush, greedy or today's sampler, actions back through `action[:, 1]` | T2's gate + **slice N with a policy opponent — PASS** (Lane E paragraph) |
| per-episode opponent + team choice | `snapshot_pool.py`, `fixed_opponent_pool.py` 558, `team_pfsp_callback.py` 214, `pool_seed.py` 349, the teambuilder | per-EPISODE, not per-decision | **the CHOICE STAYS PYTHON** (M5 Lane E: `rust_env_opponents.EpisodeOpponentSampler`, draw for draw the wrapper's rules), STAGED per env as `ep_opp` (a route of the spec's declared `opponents` table) with the teams and seed; the core consumes it at every start and echoes `opp_route` / `opp_slot` | the sampler pinned against `_select_episode_opponent`; slice N replays the staged inputs |
| scripted bots (training floor roster, exploiter keep-bots, eval roster, final eval, anchors `bot:<name>`) | ten pooled classes (`bot_inventory.py`): `poke_env/player/baselines.py` (`RandomPlayer`, `SimpleHeuristicsPlayer`), `agents/opponents.py` (eight `Gen3*Player`s), `agents/baitbot.py` (`Gen3BaitBotPlayer`, only with `--bait-bot-share`) — all read a poke-env `Battle` | the reading (`BoardReading` is the port of that `Battle`) | **BUILT in the core AND ROUTED** (M5 Lane F, `src/rust_env/src/bots/`: all ten, reading the core's own per-side reading through `bots::view::View`; RNG = CPython's stream. Lane E: a `bot` route plays them inside the core at real decisions only, never exposed, streams seeded by a declared rule) | per-bot action equality on a banked decision corpus — **PASS** (COMMIT 1,976 decisions routine, MILESTONE 20,229 `slow`, 0 mismatches) |
| eval | `eval_callback.py` 1,826 + `eval_sharding/` + trace recording (the trace quota, `eval_manifest.json`) | T2, bots, the greedy regime (`eval_sentinel_greedy`), the persisted record (`gen3_core_event_v1`) | NOT BUILT | the same seed set played on both paths: equal results in the greedy regime; traces readable by the prober |
| the env surface callbacks read | 71 `env_method` / `get_attr` sites in `agents/training` + `main/train` (non-test) | — | NOT BUILT | each site mapped to a column or an info field, or deleted |
| search's transport | `utils/bridge/search_session.py` 405 (JSON `open_root` / `expand_many`), `search_driver`'s verbs, `driver_timing.rs` 128, the node drivers still diffed against | `successors()` in-process | **BUILT, not yet the default** (M5 Lane I: `utils/rust_env/successors.py` over `src/rust_env/src/search/`; `SearchConfig(search_impl="inproc")`); the JSON road stays search's default until a ledger entry switches it | depth-3 successor slice equal to `search_driver`'s rows + the three search gates — PASS (Lane I paragraph) |

**THE LANE PLAN** (agent-days include the §2 1.5× surprise allowance; **H** = opus-high,
**M** = opus-medium). Each lane OWNS its files; nothing outside them changes without a hand-off
line in this section.

The Rust env is a NEW crate, `src/rust_env/` (package `pokesim_env`, std-only). It path-depends on
the port, so the port crate, its `release` build and `sim_bridge` (what training execs today) are
untouched until M5's gate. The Python side is `src/utils/rust_env/`.

| lane | owns | gate it must pass | depends on | size |
|---|---|---|---|---|
| **0 — the shared core boundary (FIRST)** | `src/rust_env/{Cargo.toml, build.rs, src/lib.rs, src/core/{mod, spec, columns, pool, dispatch, refusal}.rs}`; `src/utils/rust_env/{protocol.py, columns.py, stamp.py}` | ① rows byte-equal to `sim_bridge`'s `__OBS__` / `core_events --obs` on the same input log (F-M5-5); ② determinism: seed → bytes, thread-count-invariant; ③ the column schema GENERATED from one table and pinned by a routine test; ④ the refusal policy (quarantine + banked input log + typed `CoreError` class); ⑤ the stamp's teeth | — | **H, 2.5–3.5** |
| A — FFI front end | `src/rust_env/src/ffi.rs`, `src/utils/rust_env/ffi.py` (signatures GENERATED, absolute-path load, stamp refusal) | the Lane-0 corpus through FFI byte-equal to the core's in-Rust run; a panic → typed error, not a crash | 0 | **M, 1** |
| B — process front end | `src/rust_env/src/bin/rust_env_proc.rs`, `src/rust_env/src/shm.rs`, `src/utils/rust_env/proc.py` (respawn on death, `/dev/shm` hygiene) | **FFI == process byte-identical** on recorded battles (routine tier); SIGKILL → typed error + respawn; no leaked segment | 0 | **M, 1.5** |
| C — training labels | `src/rust_env/src/labels/`, `src/agents/training/rust_env_labels_parity_test.py`; the inventory `src/utils/rust_env/label_inventory.py` + `src/agents/training/rust_env_label_inventory_test.py` + `designs/rust_sim/env_labels.md`; its rows of `columns.py` / `protocol.py` (hand-off: lane-C-owned rows only) | slice N label columns == `Gen3Env`'s production keys per decision (COMMIT in the routine gate, MILESTONE slow) | 0 | **H, 3–4** |
| D — episode + reward (**BUILT 2026-09-29**) | `src/rust_env/src/episode.rs`, `src/utils/rust_env/episode.py`, `src/rust_env/tests/episode_test.rs`, `src/agents/training/rust_env_episode_parity_test.py`; its rows of `columns.py` (`reward`, `terminated`, `truncated`) and the spec's `terminal` key (hand-off) | slice N `reward` / `terminated` / `truncated`, the stall forfeit at `StallConfig().threshold`, ties, the terminal observation | 0 | **M, 1–1.5** |
| T2 — the inference service (already in the program) | `src/agents/inference/service/` | T2's own gate | — (parallel with everything) | **H, 4–6** |
| E — opponent routing (**BUILT 2026-09-29**) | `src/rust_env/src/opponents.rs`, `src/rust_env/tests/opponents_test.rs`, `src/agents/training/rust_env_opponents{,_parity,_benchmark}.py` + tests (per-episode route in; opponent rows out to T2; bots played in the core); hand-offs: its rows of `columns.py` (`ep_opp`, `opp_route`, `opp_slot`) + the spec key `opponents` (`protocol.py`, `spec.rs`), `pool.rs` (`route`, `bots`, `feed_token`, the start's bot loop, a bot's p2 row not encoded), `episode.rs` (the bot feed, the policy p2 stall rule, `need` hiding), `SnapshotPool.sample(rng=)`, `SingleAgentWrapper.step` (`gen3_no_phantom_opponent_poll_v1`), Lane F's COMMIT bot bank re-recorded (`tests/fixtures/bots/commit_corpus.json.gz`: the fix removes the phantom draws) | slice N with a POLICY opponent: its actions through T2 equal the per-env compiled path's (greedy byte-identical) | 0, T2 | **H, 1.5–2** |
| F — scripted bots (owner 2026-09-27: PORT them — "happy to rewrite the bots as needed"; **BUILT 2026-09-29**) | `src/rust_env/src/bots/`, `src/rust_env/tests/bots_gate_test.rs` + `tests/fixtures/bots/`, `src/utils/rust_env/{bot_inventory, bot_view, bot_corpus, bot_tables}.py` + their tests; hand-offs: `PMove.last_used` in `src/rust_sim/src/present/mon.rs`, `Game::reading` in `search/game.rs`, `lib.rs`, one `core_cargo_test.py` name | per-bot action equality on a banked decision corpus | 0 | **H, 3–4** |
| G — training integration (**BUILDING from 2026-09-29**; the Lane G paragraph below) | `src/agents/training/rust_rollout/` (the collector package), `src/agents/training/keyed_draw.py`, `src/agents/training/rust_vec_env.py`, `src/main/train/rust_env_setup.py` + `src/main/train/parser/env_core.py` (a new `--env-core` flag, python the DEFAULT), `metadata.json`'s env-core stamp; the env surface (measured 2026-09-29: 15 `env_method` call sites, 14 names, in the training sources — `rust_vec_env.SURFACE` serves 9, 5 are reachable only under a flag the Rust core refuses) | slice N at the ROLLOUT level (the buffer the learner sees equal on scripted recorded battles); the `--debug` smoke on the Rust env; **the first two minutes of a real launch**; throughput at `--n-envs 48` interleaved against today's path (`trainer_turn_benchmark.py` defaulting to the rust bridge first — TECH_DEBT P2); F-M5-3's GIL constraint | A or B, C, D, E | **H, 3–4** |
| H — eval on the core | `src/agents/training/eval_callback.py` + `eval_sharding/` (behind the same flag), traces as `gen3_core_event_v1` records | the same seed set on both paths → equal greedy results; traces load in the prober | G, T2, F (BUILT) | **H, 3–4** |
| I — search on `successors()` in-process (**BUILT 2026-09-29**) | `src/rust_env/src/search.rs` (+ `src/rust_env/src/search/{tree,game,playout,ffi_imp}.rs`), `src/utils/rust_env/successors.py` (replaces `search_session.py`'s JSON for the in-process road; + `successors_parity.py`, `successors_benchmark.py`, `successors_integration_test.py`, `src/rust_env/tests/search_game_test.rs`); its rows of `ffi.FUNCTIONS` (hand-off) | the depth-3 successor slice equal to `search_driver`'s rows; the three search gates (the M2 trio was deleted with its roads — their in-process successors, Lane I paragraph); the PLAYOUT primitive Lane S's ④ calls | 0, A | **H, 2–3** |
| J — the M5 gate harness (**BUILT 2026-09-29**) | `src/main/rust_core_m5/` (the lane registry + verdict table, slice N env level, the depth-3 slice, the throughput A/B + its hooks, the composed verdict; `python -m main.rust_core_m5`) | it IS the gate: every lane's own gate delegated to as ONE declared row (a lane the doc marks BUILT with no row fails a routine test), slice N at N = 48, the depth-3 slice, the A/B at `--n-envs 48`; grows as lanes land | 0 | **H, 2–3** |
| **S — the POLICY-SPECTRUM instrument (owner, 2026-09-29)** | a fixed TURN BANK + a reader: `src/main/policy_spectrum/` (bank builder, reader, report) | ① the bank is stored as RE-ENCODABLE battle inputs (core input logs + decision index, via Lane 0's replay path), never as today's obs vectors — so every future architecture (M5, the discrete-token boundary X5) is read on the SAME turns; re-encoding a banked turn reproduces the recorded obs byte-equal at the recording commit; ② ≥ 10,000 decisions, stratified and stamped (phase, forced switch vs free choice, opponent class incl. exploiters and bots, our team, move categories: attack / status / setup / hazard / recovery / switch); ③ the reader evaluates ANY checkpoint on the whole bank (forward passes only, no games) and reports the RANK-MASS SPECTRUM — mean probability on the policy's own 1st, 2nd, … nth choice — plus entropy, per stratum; ④ a GROUND-TRUTH subset (~1–2k turns): every legal action branched to the end with common random numbers (successors(), Lane I; greedy continuation) ⇒ each action's TRUE value; on it: mass on NEAR-BEST actions (within ε of the best true value) vs mass on DOMINATED actions, and the STARVATION rate = turns where a near-best action gets < 1% mass, per move category — starvation measured over the whole move space, not a hand-picked Toxic case; ⑤ trend over training: read every snapshot of a lineage (healthy sharpening removes DOMINATED mass; starvation also removes NEAR-BEST mass; at genuine guess turns near-best mass should stay split — the Nash view) | 0 (replay/re-encode), I for ④ | **H, 2–3** (④'s branch compute is CPU hours) |
| **K — the learner pipeline (owner, 2026-09-28: "scope all of this into the M5 overhaul")** | K1 **torch ≥ 2.8 in a NEW conda env** (never mutate `gen3ai_stable` under live or pinned runs; switch via `$GEN3AI_PYTHON` at a run boundary), then drop the `6521f420` CUDA graph split; K2 **diagnostics on a CADENCE**: the noise-scale probe (7.1 s = 12% of the update, measured by the learner benchmark 2026-09-28) and the other optional telemetry every N updates (flag-set), never every update; moving the ~240 host reads per micro-batch on-device is HYGIENE only (measured 1.0 s = 1.7%, GPU 99% busy — the host-sync hypothesis was refuted); K2b **fewer, larger kernels**: a larger micro-batch at the same effective batch (benchmark config first), and a look at loss assembly (12%); K3 a **HERMETIC per-run compile cache** (owner, 2026-09-29: "always initialize from a fresh setup — who knows how many silent errors a shared cache allows"; was: a persistent cache across restarts): today the learner's Inductor/Triton caches are UNMANAGED — one shared `/tmp/torchinductor_<user>` (12 GB on tmpfs, i.e. RAM, on 2026-09-29) used by every run, pin, torch env, benchmark and test, with the opponents' CPU compile in a second shared dir (`/tmp/gen3ai_inductor_cache`) — and the 2026-09-29 K1b fault proved a cache key can omit a setting that changes the artifact (donated buffers). Rule: `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` point inside the RUN's own directory (or its archive), created EMPTY at a FRESH launch, reused only by that run's own restarts, and guarded by a stamp (pin sha + torch version + `compile_control`'s config-row hash) — a mismatch wipes it; tests and benchmarks always get a fresh temp dir; nothing is ever shared across runs, pins or envs; the real-obs parity gate still runs at EVERY start (it verifies outputs, whatever was loaded). Cost: one cold compile per run (minutes), not per restart; K4 **CPU-lane hygiene**: offline reads, anchors and their peers (Foul Play / Metamon) under a cgroup `cpu.max` or nice 19 inherited by every child, so a GPU arm keeps its rollout CPU (L95 lost 35% fps to X22 on 09-28); K5 the epochs/TF32 defaults from the learner battery's verdicts (a decision, no build); K6 **a DECLARED compile lifecycle** (owner, 2026-09-28: "pre-validate, reset, explicitly compile every model we want, freeze; any other compile request is wrong"): every compiled callable × input signature (shape bucket, dtype, grad/train mode, contiguity) is DECLARED in one table, compiled at startup, then the cache is frozen BEFORE the first real iteration — an undeclared signature is a typed FATAL that names the failing guard, never absorbed by a warm-up iteration (today's `8fc297a2` locks after iteration 1 and absorbed one unidentified signature — find it and declare it); mode flags passed as arguments rather than read as module attributes (fewer implicit guards); an **IN-RUN PARITY CANARY**: every N updates, the compiled path and the eager module run the committed real-obs fixture and must agree at the startup gate's bars (legal log-probs, V; the train graph's gradient cosine at a slower cadence), a disagreement is a typed FATAL — the startup gate proves the graph at t=0, the canary proves it is still the same function at t=N (owner, 2026-09-28: "I don't review the code or the logs, I trust agents — stopping GIGO is so powerful"); after K1 `fullgraph=True`; for INFERENCE (T2's slots × buckets) ahead-of-time artifacts (AOTInductor / `torch.export`), which have no runtime guards at all — the training step stays on dynamo unless K7 says otherwise; K8 **a GRAPH-BREAK BUDGET, pinned by a routine test** — measured 2026-09-28 (`torch._dynamo.explain`, CPU, C's final, 32 fixture rows): only the feature extractor is compiled today, and it is **3 graphs / 2 breaks** (+1 deliberate CUDA split ⇒ 4 on GPU) over 7,063 ops — the breaks are our own `forward_guard` (a `WeakKeyDictionary` lookup) and a rate-limited logger calling `time.time()` inside the forward (`utils/logging/rate_limiter.py:14`); the heads, the `MaskableCategorical` masking and the loss run EAGER (the whole policy forward traces to 10 graphs / 9 breaks, sb3's distribution `__dict__.pop` ×4+) — and loss assembly is 12% of the update. Target: extractor = 1 graph after K1, heads + masking + loss compiled, the guard and logging hoisted out of the traced region (the guard KEPT, made trace-safe); a routine test runs `explain()` and FAILS if `graph_count` exceeds the pinned budget — a new break is a failing test, never a silent de-optimisation. Fewer breaks mean bigger fused regions, i.e. more miscompile surface — K6's parity gate + in-run canary are what make that safe; K9 **learner-side GIGO gates** (2026-09-28): (a) a **LEARNER GOLDEN** — beside `reward_golden` and the obs golden, a pinned checkpoint + a pinned small rollout buffer + one eager fp32 update ⇒ hashed post-update parameters and losses; any change to what the update computes fails the routine gate until deliberately re-recorded (a refactor that silently changes the loss is today invisible); (b) **behaviour-policy consistency** — on the FIRST micro-batch of every update, before any optimizer step, the learner's recomputed log π(a|s) must equal the rollout's stored log-prob (max |Δ| < 1e-4, typed FATAL): it catches stale rollout weights, eval-vs-train-mode differences and rollout/learner obs mismatches (it would NOT have caught the 09-28 miscompile — rollout and learner shared the wrong function — which is why K6's eager canary exists); (c) an audit that every non-finite loss/gradient path is FAIL-CLOSED (typed FATAL, no silent `nan_to_num` on a trained quantity); K10 **the recipe surface and its prerequisites** (from `design_learner_recipe.md`, 2026-09-29): (a) a RECIPE block in `designs/production_config.json` applied by `--arch production` and diffed by `checkargs` — today five parser defaults (n_envs, batch_size, n_epochs, ent_coef, clip_range_vf) differ from the live recipe and `--arch production` covers none of them, so a fresh argv that omits them silently trains a different recipe (32 / 4096 / 5 / 0.02 / 0.5); (b) ~~the DELAYED-LABEL BUFFER~~ SUBSUMED by order constraint 6 (Lane G's complete-game collector, 2026-09-29) — at λ = 1 the win-prob critic drops the rows of games still running at the rollout's window edge (≈ L/(2·n_steps) of rows: ~18–35% at n_steps = 64, biased to long games), so NO short-window / dynamic-n_steps arm may run before rows are back-filled when their game ends; (c) `main.dose` counts a ragged final minibatch as a partial step (1.5 steps/epoch at 98,304 / 65,536) though it is a full-weight optimizer step (2/epoch) — doses across shapes are off by 4/3; fix the count and require rollouts that divide evenly into micro × K; K7 **an AOT TRAINING-STEP SPIKE** (after K1, ~1 agent-day, go / no-go / go-with-refactors): `torch.export` the extractor + heads + loss forward AND backward on torch ≥ 2.8 — pass = one graph with no breaks, the real-obs parity gate (legal log-probs, V, gradient cosine ≥ 0.9999 vs eager), update time vs dynamo on the learner benchmark, build/load time and artifact size, and coverage of mem-efficient attention backward, masking, probe hooks and the aux losses; a failure names the blocking constructs | the learner-only benchmark (`2d39b130`) is the gate: update time per phase before/after; K1 the real-obs parity gate green at fp32 and TF32 on the eval AND train graphs with the split OFF, plus a speed A/B; K2 **learning bit-identical** with diagnostics on vs off at the same seed (loss/params after one update), and the cadence's skipped updates log nothing rather than a stale value | K1 and K2 none (dispatch in the first wave); K3 with G's restart policy | **M–H, 3–4.5** |

**Order constraints.**
1. **Lane 0 first.** Every lane reads its column contract and its dispatch.
2. **After Lane 0:** A, B, C, D, F, I and J run in parallel (disjoint files), and T2 runs from day 1.
3. E needs T2. G needs (A or B) + C + D + E. H needs G + T2 (the bots are ported: Lane F).
4. **M5's gate** is J's slice N at MILESTONE, plus the depth-3 slice, plus the throughput A/B at
   `--n-envs 48`.
6. **LANE G's COLLECTOR — COMPLETE-GAME BUFFER + SAMPLE-COUNT TRIGGER (owner, 2026-09-29; replaces fixed n_steps windows and K10(b)'s "delayed-label buffer").** Today a rollout is a fixed window of n_steps decisions per env; games simply continue across windows, the POLICY's GAE bootstraps from V at the window edge (fine), but the win-prob CRITIC (λ = 1, outcome-trained) has no outcome for a game still running at the edge and DROPS those rows — ≈ L/(2·n_steps) of rows (~1% at n_steps = 2048, ~35% at 64), biased toward long games. The M5 collector instead:
   - runs every env continuously; each finished GAME's full trajectory (obs, action, the behaviour log-prob μ(a|s) at play time, the policy VERSION that played each step, labels) enters the buffer at game end — every row has its real outcome; GAE runs over whole games (no edge truncation);
   - fires an update when the buffer holds the target SAMPLE COUNT, which the adaptive-batch / rollout controller sets (so "n_steps" disappears as a knob — rollout size is dynamic by construction); games still in progress carry over to a later update;
   - corrects staleness with the stored μ through PPO's own per-row ratio π/μ (with its existing clip) — for EVERY row, whatever its age. **No row is ever dropped for age** (owner, 2026-09-29: dropping biases the data — it would systematically remove long/stall games); **no extra age cap or truncation by default.** The per-row ratio is the unbiased correction, not a bias (it down-weights an action exactly as much as the current policy now plays it less); the only biases are PPO's existing clip (a truncation it already applies every epoch) and the state-distribution mismatch every PPO variant ignores — both grow with staleness, so they are MEASURED, not pre-empted: every update logs the version-age distribution and the ratio / clip-fraction / KL per age bucket. Only if old-age rows show ratios well outside the clip band does a registered change add a remedy — preferably PER-GAME VERSION PINNING (a game is played start-to-finish by the policy version it began with, held in a T2 slot, so each game is single-policy data) before any truncation;
   - K10(b) is SUBSUMED (the buffer is the delayed-label fix); K10(a) (the recipe block) and K10(c) (dose ragged step) stand.
   Gate: slice N at the ROLLOUT level (the buffer equal to today's path on scripted recorded battles when the trigger is set to reproduce today's window, so the new collector is proven before it changes the schedule), then the staleness distribution and learning-per-sample read in the sizing study (constraint 5).
5. **SIZING — decided by measurement, not picked (added 2026-09-29, owner question: "where do we decide how many envs? … the right number of weight slots?").** Nothing in the plan chose N before this. After M5's gate (parity at N = 48), Lane J runs a registered SIZING STUDY, and G/T2 expose every size as a DECLARED startup input (declared lifecycle — nothing resizes after freeze):
   - **Envs and rollout shape.** Sweep N ∈ {48, 256, 1024, 2048} holding the ROLLOUT size (N × n_steps ≈ 98k–131k samples) roughly fixed, so n_steps shrinks as N grows (2048 → 64 at N = 2048; a game is ~45 decisions, so short windows bootstrap from V — fine at the adopted λ = 0.80, and the critic trains λ = 1 to outcomes). Read (a) samples/s end to end, split into env / trainee inference / opponent inference / update, and (b) learning per SAMPLE (untaught-8 and the SmallRL guard at matched samples, one short fork each), plus the gradient-noise-scale meter to choose the update batch (N0's meter read the policy as OVER-batched, noise_scale_ratio 0.44). Pick the smallest N that saturates the GPU without losing learning per sample; the rollout staleness (policy lag within a rollout) is the thing to watch as N grows.
   - **Weight SLOTS are derived, not tuned.** The trainee is ONE slot with a big batch (N rows); slots exist for DISTINCT weight sets used at the same time. Opponent slots = the number of distinct policy opponents that can be live at once = the declared pool window (self-play snapshots) + stable opponents + exploiter targets (bots run in the Rust core, not in slots); eval filler gets its own slot group. At ~16 MB a slot, even 64 slots is ~1 GB — memory is not the limit; LATENCY is (rows per slot ≈ N × P(policy opponent) / slots, padded to buckets). So startup computes the slot count from the pool config, picks BUCKETS from the rows-per-slot distribution (measured waste < a declared bound), and lanes = min(slots, 8) (T2 measured: 48 rows over 8 slots = 9.9 ms on 1 lane, 3.95 on 4, 2.6 on 8). A pool refresh between rollouts is a declared LOAD into an existing slot, never a new slot.
   - **The UPDATE batch is DYNAMIC, not swept (owner 2026-09-29: "long term target ~1.1× the critical batch — how do we avoid making this static?").** The existing `--adaptive-batch policy` controller (`adaptive_batch_callback.py`, gen3_adaptive_batch_v1) already does it without recompiling: the compiled MICRO-batch shape stays FIXED (one learner graph) and the controller moves the number of accumulated micro-batches K per optimizer step to hold `noise_scale_ratio_policy` = B_noise / (micro × K) at a target; 1.1× the critical batch ⇒ `--adaptive-batch-target ≈ 0.91`. Two consequences to design for: (i) the rollout must stay a few optimizer steps deep at the LARGEST batch the controller may reach, so the rollout size (N × n_steps) is sized from the expected B_noise range — N is DECLARED at startup, n_steps may vary per rollout inside a DECLARED maximum (buffers pre-allocated at the max; the declared lifecycle holds); (ii) a moving batch moves the DOSE (lr × epochs / batch) — the controller's interaction with the LR/KL controller is documented in `src/agents/training/CLAUDE.md` and must be re-read before adopting it.
   - **INFERENCE BATCH SIZES (buckets) — one compiled graph per bucket, so pick few and pick well.** Compile is ~50–75 s per bucket on 2.5.1 (~42 s on 2.8), paid once per run with K3's per-run cache; capture per lane is cheap. Padding cost depends on the regime: at small batches a call is overhead-bound (8 → 48 rows costs only 1.25 → 1.81 ms), so padding is nearly FREE there and a few coarse buckets suffice; at large batches it is compute-bound and padding costs in proportion. So: the TRAINEE's batch is exactly N every step (all envs step together; a parked env pads) ⇒ ONE bucket = N; OPPONENT rows per slot vary with the random opponent draw (≈ multinomial around N × P(policy opp) / slots) ⇒ 2–4 buckets chosen from that distribution's quantiles to bound padding waste; the LEARNER's micro-batch is fixed ⇒ one graph; EVAL filler uses the opponents' buckets.
   - Output: the production N, the n_steps maximum, the adaptive-batch target and K bounds, slots, buckets and lanes, recorded in the Decision record with the numbers, before the first serious training run on the new infrastructure.
   - **Re-grounded in [`design_learner_recipe.md`](design_learner_recipe.md) (2026-09-29, PROPOSED there, not decided here):** at λ = 1 the win-prob critic DROPS the rows of episodes unfinished at the window edge (no back-fill), about L/(2·n_steps) of all rows — ~18–35 % at n_steps = 64 — so the delayed-label buffer lands before any short-window arm; rollout shapes should divide into micro × K (the live 98,304 / 65,536 shape takes a half-size second step every epoch and the dose undercounts steps by 4/3); and under epoch reuse K cannot remove ROLLOUT noise, so n_steps gets its own controller (D_eff ≥ k·B_noise).

**M5 DESIGN PRINCIPLE — a DECLARED LIFECYCLE (owner, 2026-09-28: "declare every resource the server could use before starting anything").** A training process is a long-lived server, not a notebook. Every run has two phases: **STARTUP** declares and acquires everything the steady state will ever use — compiled artifacts/graphs (K6), GPU buffers (rollout buffer, micro-batch staging, inference slots, a fixed caching-allocator pool), env/worker pools, ports, files — validates it (parity gates, stamps), then FREEZES; **STEADY STATE** acquires nothing new. Anything that would lazily appear mid-run — a recompile, an undeclared signature, a growing CUDA segment count, a new process or thread, a new port — is a counted, typed failure at the point it happens, never a silent adaptation. Each lane states what it acquires at startup and exposes a `*_after_freeze` counter that must stay 0 (compiles, CUDA segments via `torch.cuda.memory_stats`, pool size). Lazy initialization is allowed only in offline/research tools, never in a production run.

**Mechanical enough for opus-medium:** A, B, D, K4. **opus-high:** 0, C, E, F, G, H, I, J, K1–K3, T2.

**Lane K1 and the compile sentinel (2026-09-28, `gen3_compile_sentinel_v1`):** `--compile-trainer` is now guarded by `src/agents/model/compile_control.py` (gate → reset → prewarm → lock after the first update; a late recompile or a cache-limit hit is `FATAL_CONFIG`), built on torch 2.5.1's `error_on_recompile` + a cache-limit log detector.

**Lane K1 BUILT (2026-09-28):** a new conda env **`gen3ai_torch28`** (torch 2.8.0+cu126, torchvision 0.23.0, torchaudio 2.8.0, triton 3.4.0; `environment_torch28.yml` — a sibling file, because `bootstrap.sh` re-applies `environment.yml` to `gen3ai_stable` on any hash change — from the main checkout, or a worktree given `--update-shared-env`; a plain worktree bootstrap refuses since 2026-09-29); `gen3ai_stable` untouched. The code runs on BOTH: the `6521f420` split is keyed on the torch version (ON for 2.5.1, OFF for 2.8.0+cu126), and `compile_control` has a row per version — 2.8 locks with `set_stance("fail_on_recompile")`, and a `wrap_compiled` sticky counter replaces what 2.8's moved start callback no longer sees. Verified split OFF on 2.8 (`ai_v14_06_lbat_ctrl_fix` final): the real-obs gate PASSES at fp32 and TF32 on the eval and train graphs (train gradient cosine 1.000000 fp32; TF32 1−cos 6.1e-5 vs eager's 4.6e-5), 3,840 trace rows argmax 1.0000 fp32; the extractor is 1 graph / 0 breaks under `explain` on CUDA. Detail: `designs/training/compile_flags.md` "Lane K1". **Lane K1b (2026-09-29, `gen3_donated_buffer_off_v1`):** the first 2.8 A/B CRASHED on the first real update — torch ≥ 2.6 donates the compiled backward's saved activations by default, and the read-only probes' `retain_graph=True` backwards (grad-balance, per-term noise, distill projection) cannot run on a donating graph; and the Inductor cache key is blind to donation, so a donating backward cached by one process was served to a donation-off compile. `compile_control._COMPILE_CONFIG` now pins `donated_buffer=False` and (2.8) a cache-key tag at `install()`; a full compiled `train()` with every first-update probe is now tested (`compiled_train_probes_test`, CPU + CUDA). A/B: `designs/research_state/measurements/m5_k1/`. **Open:** a batch-1 CUDA eval graph does not lower on 2.8 (Triton `CompilationError`; `--critic shaped` prewarm and the in-process final eval reach it), the interpreter switch for new runs at a run boundary (orchestrator), and `fullgraph=True` (K6).

**Lane K2 BUILT (2026-09-29, `gen3_diagnostics_cadence_v1`, config v124):** `--diagnostics-every N` (fresh default 10) runs the optional probes (per-term noise scale, `grad/*`, `rank/*`, `edge/*`, `cell/*`) on every Nth update and on each process's first update (the compile lock follows it, so the probes' signatures are declared before the freeze); a skipped update writes none of their tags. `--rank-tripwire` keeps `rank/*` every update and `--adaptive-batch policy` keeps the per-term probe every update. Learning bit-identical ON vs SKIPPED on the production extractor surface (params, AdamW state, losses, RNG — `diagnostics_cadence_test.py`, mutation-checked). Recorded + inherited; pre-v124 runs inherit 1. The on-device `.item()` hygiene is NOT done (1.7%, measured). **Open:** the GPU speed read (`learner_benchmark` `diag_skipped` on an idle GPU). Detail: `designs/training/ppo_step.md`.

**Lane K first wave (2026-09-28):** K1 (torch) and K2 (sync-free update) dispatch with Lane 0 on M5's first night — disjoint files (`environment.yml` + `team_transformer.py`'s split; `instrumented_ppo/`), and the learner benchmark already exists as their gate.

**Lane 0 (BUILDING from 2026-09-28; progress and resume point: [`research_state/measurements/m5_lane0/PROGRESS.md`](../research_state/measurements/m5_lane0/PROGRESS.md)).** The crate `src/rust_env/` (package `pokesim_env`, `rlib` + `cdylib`, its own `target/`) exists; its column contract is GENERATED from ONE table (`src/utils/rust_env/columns.py` + `protocol.py` → `src/rust_env/src/core/columns.rs`, pinned by the routine `columns_test.py`), gate ③. What the contract adds to the prototype's, and why: the episode inputs are CALLER-STAGED columns (`ep_team` = two indices into a team table declared at startup, `ep_seed` = the Showdown seed), because per-episode team and opponent choice stays Python (the inventory above) and an auto-reset inside STEP must already hold the next episode's inputs; `refused` (a quarantine is told apart from a tie by a column, not a counter); `episode`, `dec_n` (sim_bridge's `__OBS__` `n`, the alignment key gate ① compares on) and `turn`; and a pool-level `counters` column carrying the DECLARED-LIFECYCLE `*_AFTER_FREEZE` counters. The prototype's in-Rust seeded random opponent is NOT carried: an in-Rust opponent is Lane F's (bots), and Lane 0 serves both sides' rows to the caller (the T2 shape). Every status is a typed Python class (`protocol.py`: `CoreFault`, `CallerError`, `CorePanic`, `LifecycleViolation`, `RefusalBudgetExceeded`). **Unit 2 (the core):** `spec.rs` (the STARTUP DECLARATION — one JSON object, every key required, keys GENERATED from `protocol.SPEC_KEYS`), `pool.rs` (N envs on T workers; each env a live `BridgeSession` + one parse chain per side folded write-by-write with `sim_bridge` `core_obs`'s alignment rules, the row encoded straight into the caller's column), `dispatch.rs` (`Core::new` → `freeze(cols)` → `dispatch(op, cols)`), `refusal.rs`. The LIFECYCLE is enforced, not advised: no op before freeze, one binding, a rebind refused and counted, STEP before RESET refused, and every team of the table VALIDATED BY USE at startup (unpacked, a battle constructed, a parse chain rooted), so a bad team fails at startup rather than as a mid-run quarantine. **The refusal policy as built (differs from the prototype's "a refusal quarantines"):** EVERY error the port returns for one battle quarantines — `Refusal`, `Malformed` AND `Fault` — banked with its kind and Python class, because it is confined to that battle's discarded state and is counted, banked replayable and budgeted; why widen it: the one real refusal on record (F-M5-1's HP-tracker elimination) is typed `CoreError::Fault` by the port, so a Refusal-only rule would have halted a training batch on it. The ENV CORE's own invariants (alignment, a stuck env), caller errors and panics fail the batch and POISON the pool (every later op is `LIFECYCLE`). Gate ② (determinism: 12 envs × 250 steps, byte-identical at T = 1 / 4 / 5 / 12 and a rerun, a different seed differs) and gate ④ (a real quarantine replayed from the Phase-A bank) run in the routine tier through `src/utils/rust_env/core_cargo_test.py`. **Unit 3 (the stamp, gate ⑤):** `build.rs` stamps `stamp=v1;commit;src;nfiles;nan_poison;schema;data` — `src` over every `.rs` + `Cargo.toml` of the port and the env crate + `build.rs` (git blob ids, FNV-1a-64; `src/utils/rust_env/stamp.py` is the Python twin), and two fields the prototype's stamp lacked: `nan_poison` (a self-check and a release build of the SAME sources write different bytes into an unwritten cell, so a caller can demand the one it means) and `data` (the port reads `data/pokemon` from its COMPILE-TIME path, so a build from another checkout reads another checkout's data — the 09-09 class for data); `schema` is the table's `SCHEMA_ID`. Teeth on a temporary copy of the sources (an edit and an added file each refuse; restore passes), and the Rust build's own stamp is checked by the Python recompute in the routine tier. **Unit 4 (gate ①):** see F-M5-5 below. All five Lane-0 gates are BUILT and routine (`core_cargo_test.py` builds the port's self-check `sim_bridge` in the same checkout and runs the crate's suite, ~10 s warm). **Units 5–6:** gate ① also runs over the Metamon LADDER corpus (owner 2026-09-24) — the COMMIT tier's 16 teams in the routine gate (9,198 frames) and the MILESTONE tier's 800 teams as a `slow` test (16 envs × 4,000 steps: **121,463 frames byte-equal over 717 input logs, 0 quarantined**, ~90 s), and the other two team sources as `slow` tests — the 719 TRAINING-POOL teams packed as training packs them (90,556 frames over 641 logs) and 200 PROCEDURAL teams (90,844 frames over 557 logs), all byte-equal, 0 quarantined; a battle the core QUARANTINES is compared too (`sim_bridge` must refuse the same battle). Gate ② now includes quarantines (6 envs replaying F-M5-1's battle: columns and bank identical at 1 / 3 / 6 threads). A decision the core's own op opens and then closes (a quarantine on the other side's fold, or the placeholder turn-limit forfeit in the same op) is never exposed to the caller; `sim_bridge` ships its frame — the gate names these "unexposed tail frames" (10 in the milestone run) and allows at most one per side at a log's end. **Cost, UNVERIFIED on a quiet box:** release build, N = 48, both sides' rows: 23.5–23.9 µs per row at T = 8 and 126–252 µs at T = 1, read at load1 19–32 on 16 cores with a production run training (the prototype read 12.1 / 69.0 on a quiet box; no A/B was run, so no ratio is claimed — `tests/bench_test.rs`, ignored).

**Lane A BUILT (2026-09-29; progress and resume point: [`research_state/measurements/m5_laneA/PROGRESS.md`](../research_state/measurements/m5_laneA/PROGRESS.md)).** `src/rust_env/src/ffi.rs` + `src/utils/rust_env/ffi.py` (`FfiCore`): a C ABI over `Core::new(Spec::from_json)` → `freeze(addrs)` → `dispatch(op, addrs)`, the columns allocated from `columns.py` with N and `OBS_DIM` read FROM THE LIBRARY, the address array bound once. **Signatures GENERATED** from one table (`ffi.FUNCTIONS`): the `extern "C"` wrappers are a generated region of `ffi.rs` that call hand-written `imp::<name>` with the same arguments (a drifted type does not compile), the ctypes `argtypes` come from the same rows, and the table's id (covering the column schema id) is compiled in and compared at load; `ffi_test.py` (routine) pins the region. **Load** by ABSOLUTE path only; before anything is returned: every symbol, the table id, the schema id + column count, then `stamp.check_stamp`. **Panics:** every export runs inside `catch_unwind` (unwinding out of `extern "C"` aborts); a panic is `CorePanic`, a panic inside a handle's locked section POISONS the handle, a concurrent call on one handle is refused (`LIFECYCLE`). **Gates (all PASS, `ffi_integration_test.py`, routine):** ① Lane 0's gate-① runs RECORDED in Rust at T = 1 (`tests/ffi_reference_test.rs`) and REPLAYED through ctypes at T = 3 — every output column after every op byte-equal, plus the deterministic counters and the bank: the bridge corpus (6 × 400), the ladder COMMIT tier (8 × 600), and F-M5-1's banked refusal + 60 steps (a real quarantine); teeth: one flipped byte fails; ② a panic (string / non-string payload / locked section) → `CorePanic` then `LifecycleViolation`, the process exits 0 (run in a subprocess); ③ every `*_AFTER_FREEZE` counter read through the FFI is 0 after each run, and a rebind is refused AND counted; ④ **throughput (descriptor, 2026-09-29, load1 2.7–5.0 on 16 cores, no run live):** transport ≈ 5 µs per dispatch (≈ 0.3 % of a T = 8, N = 48 STEP); FFI 17.1–17.9 µs/row at T = 8 vs in-Rust 15.9–17.8 (different battles; the gap is inside the core, not the boundary — no ratio claimed). Findings for Lanes B / E / G / J (poisoning rules, the wall-clock counters, `dlopen` path caching, the untested worker-thread panic path, the probe export) are PROGRESS's F-LA-1…8.

**Lane F BUILT (2026-09-29; progress and resume point: [`research_state/measurements/m5_laneF/PROGRESS.md`](../research_state/measurements/m5_laneF/PROGRESS.md)).** The scripted bots are ported: all ten pooled classes, as `bots::Kind` in `src/rust_env/src/bots/`.

- **The inventory** is `src/utils/rust_env/bot_inventory.py`. Its routine test derives every roster from the code (the training floor roster, which is also the exploiter keep-bots mix; the eval roster, which anchors and the prober reach; the final eval; the warm-start smoke; the conditional BaitBot). It fails on an unlisted pooled bot, a stale `used_by`, an unlisted bot class, or a "ported" row without its `Kind`. `MaxBasePowerPlayer` is defined and plays in no pool.
- **What a bot reads (decided): the core's own per-side `BoardReading`**, the port of the poke-env `Battle` the Python bot reads (`env.battle2`), through `bots::view::View`. There is no second tracker. Equality is proven at every banked decision: `bot_view.py` and `View::render` write the same canonical string (every attribute any bot reads; floats as IEEE bits), and the gate compares hashes before actions. One hand-off made it possible: `PMove.last_used` (poke-env's `_is_last_used`, which `staller_v2` reads).
- **Randomness (decided): the SAME stream, not explicit inputs.** `bots::rng::PyRandom` is CPython 3.11's MT19937 (`seed`, `random`, `getrandbits`, `_randbelow`, `choice`; CPython goldens). A seeded Python bot and the Rust bot then draw the same numbers, which Lane H's "same seeds on both paths" needs. The corpus banks each stream's offset before and after every decision, so a wrong draw COUNT fails even when the action agrees.
- **The gate** (`src/utils/rust_env/bot_corpus.py` records in a production-surface `Gen3Env` on the rust bridge with the real bot as p2; `bots::gate` replays the banked battle on Lane I's `Game` and, at every p2 decision, compares the view hash, the sent action's token and 11-dim index, and every stream's offset; the battle must replay):
  - COMMIT (routine): the committed 55-episode bank, 0 mismatches (re-banked after the F-LF-1 fix and again when non-Ghost Curse became a setup move; it carries a chosen `heuristic2` battle where the setup step fires and three battles where a setup bot Curses). The bank also re-records byte for byte (in two parts), so a Python bot change fails first.
  - MILESTONE (`slow`): 520 fresh episodes over pool / ladder / procedural teams, 20,229 decisions, 0 mismatches.
  - Teeth: a moved action, view hash or stream offset, and a relabelled bot, each fail on their own counter.
  - Branch coverage is reported per `logic.rs` return site. Every site is reached except the no-active-mon guards and fallbacks that are unreachable at a real decision.
- **Not built here:** wiring a bot into `pool.rs` — **DONE by Lane E** (its paragraph below: `bot` routes; the seed rule `opponents::stream_seed`). It was Lane E's routing (F-LF-4), with `Bot::decide(reading, tokens) -> Decision { order, token, index }` as the API, and the per-episode stream SEED rule (F-LF-3: the Python production bots are unseeded — the global `random`, and OS entropy for BaitBot — so the Rust env's explicit seeding is a declared change of stream, not of distribution).
- **Findings** (PROGRESS F-LF-1…5):
  - **F-LF-1 — FIXED 2026-09-29 (decision changed: "port as-is" → fix, ordered by the orchestrator as GIGO while no training ran):** the setup branch NEVER fired in `SimpleHeuristicsPlayer`, `Gen3HeuristicV2Player`, `Gen3SetupSweepPlayer` and `Gen3SetupSweepV2Player` — each compared poke-env's `Target` enum to the string `"self"`. Python and Rust now both test `Target.SELF`, and (owner 2026-09-29, "allow Curse") a non-Ghost's Curse counts as setup with its real +1 Atk / +1 Def / −1 Spe while a Ghost's never does (`self_setup_boosts`); the bank was re-recorded; `src/poke_env_enum_str_compare_gate_test.py` gates the class. Every earlier eval row against these bots is an ERA BOUNDARY (ledger `2026-09-29 · GIGO FIX · F-LF-1`).
  - **F-LF-2 — CLOSED by Lane E** (`gen3_no_phantom_opponent_poll_v1`): the training wrapper's PHANTOM polls (`choose_move` on steps whose p2 order is never sent) consumed bot draws on the Python path only; the wrapper no longer polls there, and the COMMIT bot bank was re-recorded (0 phantom polls).

**Lane E BUILT (2026-09-29; progress and resume point: [`research_state/measurements/m5_laneE/PROGRESS.md`](../research_state/measurements/m5_laneE/PROGRESS.md)).** Per-episode opponents in the core, POLICY opponents served by T2, Lane F's bots played in the core.

- **The core** (`src/rust_env/src/opponents.rs`). A ROUTE TABLE is part of the startup declaration (the spec's new `opponents` key): `external` (the caller answers p2 — the harnesses, and `spec_json`'s default), `policy` with a T2 slot (one-to-one), or `bot` with a DECLARED seed (+ BaitBot's `p_bait`, the run's `--bait-bot-p`). The caller stages `ep_opp` (a route index) WITH `ep_team` / `ep_seed`; the core consumes it at every start (auto-reset, quarantine restart, a parked env's start) and writes `opp_route` / `opp_slot` for the episode the other columns describe. Out-of-table = `CallerError`. A BOT route is played inside the core by Lane F's `Bot::decide` at real decisions only (never a phantom poll, F-LF-2; never at p1's forfeit, F-LF-5) and its p2 decision is never exposed; each env holds one bot per route for the pool's life, stream k seeded `random.Random(opponents::stream_seed(seed, env, k))` (F-LF-3: today's bots are unseeded — a declared change of stream, not distribution). A POLICY route's p2 decision at turn >= the stall threshold is never exposed (today's `RLPlayer` stall check precedes any forward), and p2 forfeits if no p1 decision is open.
- **The host** (`src/agents/training/rust_env_opponents.py`). `OpponentPlan` builds the route table from the SAME inputs `env_factory` uses (pool window + spare, stable opponents, the exploiter — two slots under `--exploiter-ladder` —, the floor roster). `EpisodeOpponentSampler` is `MaskableAgentWrapper._select_episode_opponent` DRAW FOR DRAW (pinned under every production branch, mutation-checked), with the pool draw through `SnapshotPool.sample(rng=)` (hand-off: an optional per-env stream; default byte-identical). `SlotFamily` makes a pool refresh a declared T2 LOAD into a FREE slot — one no current or staged episode names — and a full family is a typed `SlotCapacityExceeded`. `PolicyOpponentServer` submits p2's rows grouped by `opp_slot`, one flush, then greedy or `sample_actions` — today's `torch.multinomial(Categorical(logits=x/T).probs, 1, True, generator=g)` BIT FOR BIT (one CPU generator per env per opponent player).
- **What differs from today, declared:** a draw happens one episode earlier (staged for the auto-reset); the pool is scanned once per generation by the host; every stream is seeded.
- **THE GATE — PASS** (`rust_env_opponents_parity.py`: record through core + T2, replay in a CUDA-less child through a production-surface `Gen3Env` on the rust bridge against the PER-ENV `RLPlayer` path — the same snapshot via `SnapshotPool.load_model`, `--compile-opponents`' strict compile, its generator restored to the recorded state; every sent p2 decision: row bytes, mask, greedy / sample, |Δ legal logp|, the outcome and the decision counts). COMMIT (routine, eager vs eager): greedy, and sampled at a low stall threshold with a mid-run pool refresh that must be a LOAD and must be played — 0 divergences; teeth: one moved action fails. MILESTONE (GPU, T2 `graph` + 8 lanes vs the COMPILED CPU per-env path, `ai_v14_06_lbat_ctrl_fix` pool of 6 + 1 refreshed, 16 envs): **sampled 2,052 decisions, greedy 1,725, 0 divergences (7 near-ties, all equal), max |Δ legal logp| 2.7e-5 / 4.5e-5**; every T2 and core `*_after_freeze` 0 across the refresh.
- **Throughput (descriptor, Lane S computing on the same GPU):** 48 envs, pool of 20 in 24 slots, production mix with the 8 roster bots in the core: opponent serve 5.9 ms / step, of which the T2 flush (~40 rows over ~18 slots) 1.15 ms and SAMPLING 5.1 ms (the per-row generator loop — F-LE-8, Lane G's call); core step 1.65 ms; rows per slot per flush 1–4 in 90 % of cases (buckets 8 / 16 nearly idle — for the SIZING study).
- **Finding fixed at the source (F-LE-1, `gen3_no_phantom_opponent_poll_v1`, a TRAINING-INPUT change):** `SingleAgentWrapper.step` polled the opponent on steps whose order is never sent; a self-play `RLPlayer` then recorded a phantom decision (its progress clock one step high at the next decisions) and drew a sample, a bot drew from its RNG. It now polls only when `agent2_to_move`. F-LE-2…9 (Lane G's staging rules, pinned teams, the exploiter ladder, mixed-arch slot groups, sampling cost) are in PROGRESS.

**Lane G (BUILDING from 2026-09-29; progress and resume point: [`research_state/measurements/m5_laneG/PROGRESS.md`](../research_state/measurements/m5_laneG/PROGRESS.md); design: [`designs/training/rust_collector.md`](../training/rust_collector.md)).** The PPO trainer's rollout on the Rust env core, built as order constraint 6 says.

- **Unit 1 — the COLLECTOR + BUFFER (BUILT).** One host loop over ONE env core: p2's policy rows and the trainee's rows go to T2 in ONE flush (Lane E's server split into `submit` / `complete`); the trainee's action is the KEYED DRAW (`gen3_keyed_draw_v1`: key = (run seed, stream, env, episode, `dec_n`) → splitmix64 → inverse CDF on the served log-probs), its behaviour log-prob, V and the POLICY VERSION T2 served are stored per row in a preallocated ARENA; a game's rows wait there until it ENDS, then get their COMPLETE-GAME GAE (`game_gae` — sb3's arithmetic operation for operation, pinned BIT-IDENTICAL to sb3's window GAE on the same game) and their own outcome as the win label (`win_mask` 1 on every row — K10(b) subsumed) and join a completion-ordered FIFO; the update fires at the declared TARGET sample count and consumes exactly it (default n_steps × N; a multiple of lcm(micro-batch, N); an adaptive-batch `set_target` hook inside a declared band); a game straddling the target is split, its tail trained at the next update — no row dropped or down-weighted for age; a CUT game (quarantine / respawn) is released and counted. The WINDOW fill reproduces today's schedule (sb3's GAE with the next row's V as bootstrap; the callback's own back-fill, now a shared function) for the parity gate. Teams / seeds staged per env from seeded teambuilder copies into a startup team table (pinned stable / exploiter teams included, F-LE-5); re-staging by the `episode` column moving (F-LE-4); version pinning a declared option, OFF. `RustVecEnv` is the model's VecEnv: spaces + N + the env surface (`SURFACE`; an unmapped call is a typed refusal). Tests: the store (GAE bits, FIFO split, both fills, the no-drop accounting), the trigger, and the collector on the real core through BOTH front ends (every row accounted for, rows of two versions after an update, the trainee's keyed draw replayed from its key, every lifecycle counter 0).
- **Units 2–4:** the trainer wiring behind `--env-core rust`, the parity gates (rollout level + learner level), the opponents' sampling change (F-LE-8) with the bucket choice (F-LE-9) — PROGRESS.

**Lane C (BUILDING from 2026-09-29; progress and resume point: [`research_state/measurements/m5_laneC/PROGRESS.md`](../research_state/measurements/m5_laneC/PROGRESS.md)).** Unit 1, the INVENTORY: `Gen3Env` can emit 30 label keys, and the production surface emits 21 in eight families. Those are belief 3, spread / nature / EV 6, HP type 2, item 2, win-prob placeholders 2, material margin 1, opponent class 1, and opponent intent 4. The row above said the intent labels were off in production; they are ON, and ARCHITECTURE.md §7 said the same wrong thing. Both are corrected. The table of record is `src/utils/rust_env/label_inventory.py`. The routine `rust_env_label_inventory_test.py` constructs `Gen3Env` and fails on any of these:
- an unlisted key;
- a dtype or shape drift;
- a gate that does not emit its key;
- a production surface different from the rows marked production;
- a consumer that stops naming its key;
- ARCHITECTURE.md §7's ✅/❌ disagreeing with the table.

It found one latent bug: an intent-only `Gen3Env` raised `UnboundLocalError` at construction. That bug was unreachable in production. The fix is a hand-off line in `gen3_env.py`: `_imax` is hoisted out of the belief block.

Unit 2, the GENERATED label columns. `columns.py` builds one output column per `core` row of the inventory, named by its `Gen3Env` key, with its dtype and `(N, SIDES, *shape)`: 18 columns. The table gains an `i64` dtype, so the int64 labels cross byte-equal. `columns.rs` gains `labels::FAMILIES` (family → column indices) and `labels::NOT_CORE`. A label column is written iff `need` = 1 AND its family is declared in the spec's new `labels` key (a hand-off: one `SPEC_KEYS` row + `spec.rs`). The declaration is checked at STARTUP by `src/rust_env/src/labels/mod.rs`, and each wrong kind is refused by name: a `host_*` family ("filled by the HOST"), a `refused` one ("OFF the production surface … not ported"), an unknown one, and a `core` family whose producer is not built yet. So no column is ever silently stale. The pin is `src/utils/rust_env/label_columns_test.py` (routine): columns == the inventory's core rows, the family map partitions them, and the rendering and the schema id cover them.

Unit 3, family `belief` (`belief_species`, `belief_moves`, `known_moves`) and the LABEL SLICE N (`src/agents/training/rust_env_labels_parity_test.py`).

- **How it works.** The core records every p1 decision's row + label columns and both sides' action indices, through the FFI. A production-surface `Gen3Env` then replays each episode on the rust bridge: same teams, same seed, the trainee's indices, and p2's indices through the real mapper on `battle2`. At every trainee decision the row must equal the core's, and every built label key must be byte-equal.
- **COMMIT tier (routine):** 8 pool episodes, 843 decisions.
- **MILESTONE tier (`slow`):** 200 pool + 200 ladder episodes, 31,564 decisions, 94,692 key compares. **0 divergences.**
- **Teeth:** one moved cell of `belief_moves` fails 73 of 73 decisions.
- **What changed in the core (hand-off in `pool.rs`).** The core labels in `pool.rs` `advance`, which now folds EVERY side's chain for a write before encoding or labelling any decision. A label reads the OTHER side's own team, the like-for-like source of `battle2.team`. The encode reads only its own chain, so the rows are unchanged and gate ① stays green.

Unit 4, families `hp_type` and `item` (`labels/per_slot.rs`), in the same slice at 7 keys per decision.

- **Result:** 220,948 key compares over the milestone's 400 episodes, 0 divergences.
- **Teeth, per family:** one moved cell fails every decision.
- **The truth side's timing (F-LC-4) is decided.** The item label changed 65 times in 55 of those episodes (61 of the changes to "nothing"), and every decision is equal on both paths.

Unit 5, family `spread` (`labels/spread.rs`). It is the derived stats plus the nature / EV label, with the nature table read from the `data/pokemon` directory the build stamp names. It was built as a port of the IV-31 stat inversion, cached per episode per side; since 2026-09-29 (`gen3_true_spread_labels_v1`, both findings below fixed, see the Decision record) the label is the truth mon's DECLARED spread, guarded against its stats and read per decision.

- **Result:** 13 keys per decision and 410,332 key compares over the 400 milestone episodes, 0 divergences.
- **Two findings for the owner, both in PROGRESS:**
  - F-LC-5: on the TRAINING POOL, 54.6 % of revealed-slot decisions carry no nature / EV label, because the inversion assumes IV 31. On the ladder it is 1.2 %.
  - F-LC-6: the Python per-battle cache is keyed by the species set alone, so a same-species opponent team with other spreads reuses a stale inversion. The Rust env recomputes per episode.
  - **Both FIXED 2026-09-29 (`gen3_true_spread_labels_v1`):** the nature / EV label is the declared set in Python and Rust together, no cache on either side; the milestone's nature / EV coverage equals its spread coverage on every tier (pool 83,905 / 83,905 slot-decisions), still 714,978 key compares with 0 divergences.

Unit 6, family `intent` (`labels/intent.rs`). It is the num step over the port's `trackers::IntentLabel`, which slice T already holds equal in ids. A bare Hidden Power resolves to the attacker's TRUE typed num.

- **Result:** 17 keys per decision and 543,643 key compares over the 400 milestone episodes (31,979 decisions), 0 divergences. The slice now runs the core with Lane D's production stall forfeit, so the per-side decision counts must match EXACTLY.
- **Every branch exercised:**
  - 15,163 moves, including 1,088 typed-HP resolutions;
  - 9,322 switches, 1,628 of them to a hidden mon;
  - 7,494 unknowns.

Unit 7, family `margin` (`labels/margin.rs`). It computes `material_margin` on the side's `present()` view; the RESET decision reads 0.0, the reward manager's reset value. **Every `core` label key is now BUILT: all 18 production keys the core computes.**

- **The milestone** gains 100 PROCEDURAL episodes. Over 500 episodes it makes 39,721 decisions and 714,978 key compares, with 0 divergences.
- **Cost, a descriptor:** +5.4 % per row on the self-check build at N = 16, T = 1.

Left for other lanes:
- **Lane G:** the host-filled keys (`win_target` / `win_mask` 0.0, `opp_class` per episode), under the inventory's `rust` / `const` contract.
- **Lanes B / J:** re-record the FFI == process run with every family declared (F-LC-7).

Left for Lane C: a policy-driven stream and a release-build cost read.

**Lane B BUILT (2026-09-29; progress and resume point: [`research_state/measurements/m5_laneB/PROGRESS.md`](../research_state/measurements/m5_laneB/PROGRESS.md)).** `src/rust_env/src/bin/rust_env_proc.rs` + `src/rust_env/src/shm.rs` + `src/utils/rust_env/proc.py` (`ProcCore`, the same surface as `FfiCore`): the core in a CHILD over ONE anonymous shared mapping — a `memfd` handed over by fd, never a `/dev/shm` name, so no exit path (not even a SIGKILL of the PARENT) can leak one (Decision record) — laid out from `columns.py` (a 4 KiB header + every column 64-aligned); one request byte per op, one reply frame `[status][len][payload]` per op (a failure's payload is `DispatchError::json`); every non-control byte is forwarded to `Core::dispatch` unread. **The wire is GENERATED** (control bytes, header words) into a marked region of `shm.rs`, its id compiled in. **The handshake is the child's first output, before it reads a byte** (wire id, obs_dim, column count, schema id, stamp): a foreign child is refused and sent NOTHING. **Death** = EOF on the reply pipe → `CoreProcessDied` (returncode + signal); with `auto_respawn` a fresh child on the SAME mapping, re-stamped, outputs zeroed, before the error is raised; a respawn is COUNTED (`PROC_SPAWNS_AFTER_FREEZE`); `op_timeout` (default none) kills by PID → `CoreProcessTimeout`. **Gates (all PASS, `proc_integration_test.py`, routine, ~7 s warm):** ① **FFI == PROCESS byte-identical** — Lane A's recordings replayed through BOTH front ends side by side: every output column after every op equal to the FFI's and the recording, the counters (`CORE_NS_*` masked), the bank, and the typed error channel (class, message, env, input log); bridge corpus 6 × 400, ladder COMMIT 8 × 600, F-M5-1's quarantine; ② SIGKILL between and during ops, an abort, a stopped child past `op_timeout` → a typed error in bounded time + a working fresh core; a binary swapped before the respawn is refused (`StampMismatch` as the cause); ③ no `/dev/shm` entry, no memfd fd or mapping left in any process of the user, no child left — after a close, a startup error, a poisoned core, a child SIGKILL and a PARENT SIGKILL; ④ a wrong build kind and a forged child (wrong sources / wrong wire) refused before a byte is sent; ⑤ every `*_AFTER_FREEZE` = 0 through the process, lifecycle refusals do not poison (a rebind is not expressible). Teeth mutation-checked. ⑥ **throughput (descriptor, 2026-09-29, load1 2.2–3.0 on 16 cores, no run live, the SAME battles on both, paired):** transport 15.4 vs 4.9 µs per dispatch at T = 8 (≈ 0.7 % of an N = 48 STEP); proc / FFI wall **1.020 [1.007, 1.034]** at T = 8 and **0.967 [0.958, 0.976]** at T = 1 — core-side differences, not transport (Phase A: 0.994). Findings for Lanes E / G (a respawn is a counted steady-state event that needs a budget policy and a RESET; poisoning = F-LA-1 with `respawn()` as the cheap recovery; parent death by stdin EOF, not PDEATHSIG; `op_timeout` is G's to choose) are PROGRESS's F-LB-1…8.

**Lane D BUILT (2026-09-29; progress and resume point: [`research_state/measurements/m5_laneD/PROGRESS.md`](../research_state/measurements/m5_laneD/PROGRESS.md)).** `src/rust_env/src/episode.rs` owns the env's RESET / STEP / QUARANTINE and every end-of-episode rule; `pool.rs` keeps the battle mechanics (`start` / `advance` / `feed`). The semantics equal `Gen3Env`'s: **the reward** is the terminal alone, from the spec's new `terminal` object (`victory_value`, `terminal_indicator`, `draw_penalty`, `timeout_turn_cap`; `utils/rust_env/episode.py` builds it from a `RewardConfig`; `spec_json` defaults to the production terminal, pinned against `production_config()`); **two new columns** `terminated` / `truncated` are `PokeEnv.calc_term_trunc` on p1's reading, RAW (the winprob re-label, `resolve_episode_end`, stays the host's — Lane G); **the stall forfeit** keeps the spec key `turn_limit` but now means `StallConfig().threshold`: at a p1 decision whose turn is `>=` it p1 FORCELOSEs instead of acting and p2 is not fed (`PokeEnv.step`), decided BEFORE any feed — **F-L0-6 closed** (the placeholder checked `>` after the feeds, one decision late, encoding rows it then discarded; gate ①'s forfeit tail-frame tolerance is removed); **ties** are truncated, 0 under the indicator; **the terminal observation is NOT produced** (Decision: `Gen3Env`'s is a Python-encoded row of the finished battle whose only consumer is SB3's truncation bootstrap, which production never takes because `resolve_episode_end` makes every winprob truncation terminal; porting a finished-battle encode would be a new parity surface with no consumer). **F-L0-2 closed:** an ended episode's `reward` / `terminated` / `truncated` / `done` are written BEFORE the auto-reset, and a quarantine-class refusal of a START (any start: auto-reset, RESET, after a quarantine) banks it (`refused` = 1) and PARKS the env (`need` = 0 0, no battle) until the next op starts it from the then-staged inputs — before, the outcome was never written and the deterministic in-op retry failed the whole batch. **Gates (all PASS):** Rust unit + `tests/episode_test.rs` (F-L0-6 pin: the forfeit op is exactly the one whose exposed p1 decision had `turn >= limit`, feeds no CHOOSE, and every encoded row is exposed; F-L0-2 pin: a refused start keeps the ended reward and parks; both mutation-checked); **slice N episode parity** `src/agents/training/rust_env_episode_parity_test.py` — recorded through the FFI AND the process front end (identical), replayed through a production-surface `Gen3Env` with the same threshold and terminal: COMMIT (routine) natural endings, stall forfeits at a low threshold, a signed-terminal mix (+30 / −30 / −35 at the cap), a deterministic TIE (Explosion vs a last mon), teeth; MILESTONE (`slow`) 60 pool + 60 ladder episodes, production terminal and threshold.

**Lane I BUILT (2026-09-29; progress and resume point: [`research_state/measurements/m5_laneI/PROGRESS.md`](../research_state/measurements/m5_laneI/PROGRESS.md)).** Two shapes over ONE FFI handle (`rust_env_search_new`, a `SearchSpec`: the clock flags + the DECLARED `max_nodes` / `max_branches`, refused by name when exceeded, never grown):
- **The search TREE** (`search/tree.rs`): `search_driver`'s CORE road in process — `open_root` (a reconstruction record at a turn, `core: "text"`, trackers) and `expand_many` with rows, statement for statement over the port's own kernels (`pokesim::search`: clone, reseed, aux RNG, follow-up policy, D10 capture) and step-built `BattleVersion`s; each leaf row is ENCODED straight into a NumPy buffer. `utils/rust_env/successors.py::Successors` is a drop-in for `SearchSession` on that road (same `RootView` / `ExpandedNode`, same node ids); `SearchConfig(search_impl="inproc")` / `--search-impl inproc` selects it (hand-off in `search_dividend/search.py`). What it does not serve is refused by name (a non-core root, `rows` false, `recorded_exact`, `integrity`).
- **PLAY OUT TO THE END** (`search/game.rs` + `search/playout.rs`; `successors.play_out`), Lane S's primitive: from a decision of a banked core INPUT LOG, branch every legal action of one side (or a list) × a set of dice seeds — each branch's engine RESEEDED at the branch point, so siblings sharing a seed share the dice stream (COMMON RANDOM NUMBERS; a seed `null` keeps the battle's own) — and play every branch to the end under a CALLER-SUPPLIED policy: the host loop reads every pending decision of every live branch in ONE batch (rows, masks, `who = 2·branch + side`), runs its scorer (`greedy(scorer)`; T2's `score()` later) and feeds the actions back; Rust never runs a network. Ends: a winner, a tie, training's STALL FORFEIT for the searched side (`StallConfig().threshold`, `crate::episode`'s rule), or `max_turns` (< 1,000, the port's panic) as truncated. Rows come from the TRAINING observation path (§6c): a `Game` is the env core's battle (the parse fold + `sim_bridge`'s alignment rules) without the columns, branchable by cloning the engine into a fresh transport and the two side streams — a step-built version chain was rejected because each child keeps its parent (a 100-decision playout would hold 100 engines per branch).
**Gates (all PASS; routine COMMIT + `slow` MILESTONE):** the playout battle IS the core's row path (`tests/search_game_test.rs`: the core's own finished episodes replayed through a `Game`, every decision's row bytes / mask / ordinal equal; a branch at every 9th command fed the rest equals the linear replay; playouts CRN-deterministic; the recorded continuation replays the banked battle command for command); **① the depth-3 successor slice** (`successors_parity.py`: the tree and the `search_driver` binary in lockstep, every root field and every arm field + ROW BYTES to depth 3, no allowlist — COMMIT 6,384 arms / 7,872 rows; MILESTONE ladder 12 + pool 6 + procedural 6 battles: **47,856 arms, 53,736 rows byte-equal, 4,422 D10 leaves**; teeth); **the three search gates** — the M2 trio named above (`materializer_parity`, `fork_sharing_parity`, `one_sided_view_parity`) was DELETED with the roads it compared (`43712881`), so Lane I's are their in-process successors: ② clone independence (an arm re-expanded after its siblings is byte-identical; a stale node id is refused), ③ DECISION equality (the real `SearchEngine`, oracle arm, pinned widths, `max_depth` 2, a sum-of-row scorer: per-action scores bit-equal, same action, fallback and widths on `rust` vs `inproc`), and the row-level oracle (`core_row_parity_fuzz_test`: the binary's rows == the poke-env replay) reaching the in-process rows transitively through ①. **Throughput (descriptor, release, one thread, load1 ≈ 1, 2026-09-29):** a depth-1 search ply **232 vs 279 µs per successor, in-process / JSON 0.803 [0.793, 0.825]**; playouts **~170 per s, ~16.7 k policy decisions per s** (60 µs / decision with a stand-in scorer), **4.4× at 8 handles on 8 Python threads** (the GIL is released in the core). **Finding F-LI-1 (search, both roads, NOT fixed — the port's `search.rs`):** `resolve_turn_sourced_with` feeds the second side a FOLLOW-UP choice into the NEXT turn when the first side's replacement switch ends the turn, so the child node's engine disagrees with its own leaf; a deeper ply then fails the whole `expand_many` batch (`bridge fatal: unresolvable choice`) — 25 batches in the milestone slice — or (**UNVERIFIED**: an arm the engine accepts there was not checked against its row) scores a board the row does not describe. Every search that deepens past depth 1 is exposed; the fix moves `search_driver`'s bytes (the node driver changes with it or its parity retires). F-LI-2…8 in PROGRESS.

**Lane S ①–③ BUILT (2026-09-29; progress, the baseline read and the ④ / ⑤ spec: [`research_state/measurements/m5_laneS/PROGRESS.md`](../research_state/measurements/m5_laneS/PROGRESS.md)).** `src/main/policy_spectrum/` (`python -m main.policy_spectrum build|gate|read|report`). **The bank** (`research_state/measurements/m5_laneS/bank_v1/`, written once, byte-deterministic): 20,712 decisions in 580 battles drawn from the lineage's own EVAL TRACES (N0 36M–74M, C_fix, K2, K3; a 2 win / 2 loss / 1 draw quota per cycle × roster opponent against the loss-enriched trace quota) plus K2 final's side of the round-0 exploiter A′'s games; stored as the sim INPUT LOGS (seed, both packed teams, the command log) + a decision index, never obs vectors — the recording is kept as each row's sha256, the action and 11 raw logits. Stamped: free / forced switch, phase, legal count, opponent class (bot / pool snapshot / exploiter) and name, our team, outcome, source, and each legal action's category (attack / status / setup / hazard / recovery / switch; status subtypes). **Gate ① PASS:** all 21,087 recorded decisions of the traced sides (four recording commits) re-encode BYTE-EQUAL through `core_events --obs` at HEAD, masks and actions equal; the replay path is `core_events`, not the env core (which stages episodes by team index + numeric seed and cannot take an eval trace's log — F-LS-5). **Gate ② PASS** (the committed bank, routine). **Gate ③ PASS:** any checkpoint `.zip`, CPU, forward passes only (~11 s per checkpoint after a ~45 s re-encode); reading an eval snapshot reproduces its own recorded probabilities (max |Δp| 4.1e-6); deterministic. **Baseline read** (13 checkpoints; N0 trained under the compile miscompile): N0 sharpens by 9.5M (rank-1 mass 0.474 → 0.586) and then stays flat to 75M (0.559); the one large step is N0 75M → C_fix (+0.057 [+0.052, +0.062], entropy −0.155), then K2 0.629, K3 0.621 — and on the fixed learner the share of free turns where a LEGAL setup / status / recovery move gets < 1 % roughly doubles while their mean mass barely moves. **Gate ④ at scale** (`truth.py` + `truth_report.py`; 1,600 turns over-sampling setup / status / recovery / hazard, every legal action × 64 shared dice seeds to the end, under THREE greedy continuations — K2 final, N0 final, C_fix final — 0 refused): from N0 75M to K3 the mass on DOMINATED actions falls a little (−0.017 to −0.019) while the share of decisive turns where a NEAR-BEST action gets < 1 % rises from 0.31–0.35 to 0.48–0.49 (+0.14 to +0.16) — STARVATION, concentrated at the N0 → C_fix step, largest on switches, detected on attacks and status moves in 3/3 continuations and on setup moves in 2/3; no verdict flips sign across continuations, and 16 → 64 seeds raises rather than lowers it. Detail: Lane S PROGRESS.

**Lane J BUILT (2026-09-29; progress, the readings and the resume point: [`research_state/measurements/m5_laneJ/PROGRESS.md`](../research_state/measurements/m5_laneJ/PROGRESS.md)).** `src/main/rust_core_m5/` — `python -m main.rust_core_m5 gates|slice-n|depth3|throughput|verdict`; every component writes `results/<component>_<tier>.json` there and `verdict` composes the M5 gate from them (a component never run reads NOT RUN, never PASS).
- **The lane registry** (`lanes.py`): each lane's parity gate is ONE declared row — its own pytest files / node ids, delegated to, never re-implemented; GPU node ids apart. The fold: NOT BUILT never runs and never passes; any fail ⇒ FAIL; a timeout ⇒ INCONCLUSIVE; nothing passed ⇒ INCONCLUSIVE; a GPU part not run reads NOT RUN beside the row. `gates --tier commit|milestone` runs the union of the BUILT rows' tests in ONE pytest session at the tier's markers; `gates --from-status` reads the MILESTONE verdicts the slow tier banked, with their commits. `lanes_test.py` (routine) FAILS when a lane this doc marks BUILT (a lane-table cell or a `**Lane X … BUILT` paragraph) has no BUILT row, a lane-table lane has no row, a declared test does not exist, or a row claims BUILT where this doc does not without saying why — rows 0, C and T2 carry such a note today (F-LJ-3: their paragraphs still read BUILDING / unmarked with every gate built and green).
- **Slice N at the env level** (`slice_n.py`): Lane C's and Lane D's record-in-Rust / replay-in-Python machinery JOINED and run at N — one core of N envs on T threads, the production declaration (every label family, production terminal, the stall threshold), episodes chained through the AUTO-RESET (F-LE-4 staging), recorded through the FFI at T and the process front end at T′ (identical), replayed through the production-surface `Gen3Env`: `dec_n`, row bytes, BOTH masks the learner reads (`obs["action_mask"]`, `env.action_masks()`), every label key, zero non-final rewards, the final reward / `terminated` / `truncated`, exact decision counts. **MILESTONE PASS** (N = 48, T = 8 / 5): pool 240 + ladder 240 + procedural 96 episodes at the production threshold + 48 ladder at threshold 12 — **46,353 decisions, 834,354 label and 92,706 mask compares, 0 divergences**, 847 auto-resets, 11 stall forfeits AT THE PRODUCTION THRESHOLD (closes F-LD-6's UNVERIFIED) and 7 ties. COMMIT (routine): N = 4, T = 3 / 2, 16 pool episodes incl. 8 low-threshold forfeits. Teeth: a moved mask / row / label / outcome / p2 count each fail on their own key. Out of scope, declared: the host-filled keys and the ROLLOUT level (Lane G), in-core opponents (never exposed — Lanes E / F).
- **The depth-3 slice** (`depth3.py`, Lane I's `successors_parity` at its registered milestone sources): **PASS — 47,856 arms, 53,736 rows byte-equal, 4,422 D10 leaves, rows at every depth; 25 batches refused by BOTH roads** (F-LI-1). THE READING (Decision record): "search's default depth" = `SearchConfig().max_depth`, ASSERTED = 3; a batch both roads refuse identically is equality, not coverage, so PASS means "equal wherever either road serves an arm" and the refused count prints beside every verdict until F-LI-1 is fixed.
- **The throughput A/B** (`throughput.py` + `hooks.py`, a DESCRIPTOR — no bar is registered): today's path (N `SubprocVecEnv` forkserver workers, production-surface `Gen3Env` on the rust `sim_bridge`, `MaskableAgentWrapper` + `Monitor`) vs the Rust env core (process front end, release, every label family, production terminal and threshold), each built ONCE and timed in interleaved blocks, CPU over the whole process tree from `/proc`. Typed HOOKS: `TraineeInference` (BUILT: `RandomLegal`, `T2Inference`), `OpponentMix` (BUILT: `UniformRandom` = `RandomPlayer` / the in-core bot `random`; DECLARED NOT BUILT: `ProductionMix`, Lane G over Lane E), `Collector` (BUILT: `StepCollector`; DECLARED NOT BUILT: `CompleteGameCollector`, Lane G, order constraint 6) — a NOT-BUILT hook raises `HookNotBuilt` naming its inputs and never falls back. A CUDA hook re-executes under the GPU lock. **CPU read (2026-09-29, N = 48, T = 8, release, random trainee + random opponent, 4 interleaved pairs × 25 s, load1 0.8 at start and self-driven to 24 by the arms themselves): the Rust env 34,154 trainee decisions/s (1.41 ms per vec step) vs today's 3,194 (15.1 ms) — 10.7× [10.2, 11.3]; CPU per decision 176 µs vs 3,519 µs — 0.050× [0.0496, 0.0504]**; startup 1.0 s vs 17.1 s; every core `*_AFTER_FREEZE` 0. **With the trainee's forward in the loop** (T2 `graph` on CUDA under the GPU lock, `ai_v14_06_lbat_ctrl_fix` final, buckets 8 / 48, same shape): **11,611 vs 2,154 decisions/s — 5.4× [4.9, 5.8]; CPU per decision 0.072× [0.071, 0.074]**; the Rust path's step is then HALF inference (2.1 of 4.1 ms per 48-row step, serial — overlapping the forward with the env step is Lane G's lever and an input to the SIZING study); T2 startup 163 s, every T2 and core `*_after_freeze` 0. The env step read 1.4 ms without the GPU process and 2.0 ms beside it (cause UNVERIFIED). Neither read has a learner, an SB3 buffer or policy opponents (Lane G / E).
- **The M5 verdict today: NOT MET — only because Lanes G and H are NOT BUILT.** Every built lane's gate PASSES at the milestone tier except Lane E's compiled test (F-LJ-6), slice N and the depth-3 slice PASS, the A/B is measured. **F-LJ-6 (RED, not banked — the orchestrator's call):** Lane E's `test_slow_compiled_per_env_path` FAILS with one `greedy_neartie` (compiled CPU vs eager argmax at a ≤ 2 × bar log-prob gap) whenever it runs after other tests (3 of 3 sessions, including its own file alone) and passed once run completely alone — ORDER-DEPENDENT; its assertion treats the near-tie class its own harness names as fatal, and its banked PASS hides it from the routine gate. E's GPU milestone and T2's CUDA suite PASS under the GPU lock. **F-LJ-5:** a milestone run without `GEN3AI_TEST_ALLOW_GPU` banks SKIP over a banked GPU PASS in `slow_tier_status.json` (its merge replaces rows); the harness banks through a scratch copy and never lets a skip replace a verdict (`gates.bank_without_skip_clobber`). F-LJ-1 (`bootstrap.sh` updates the SHARED conda env from every fresh worktree) and the rest are in PROGRESS.

**Total: ≈ 24–33 agent-days, plus Lane K (3–4.5, added 2026-09-28) ⇒ ≈ 27–37.5** (F excluded: 21–29), including T2's 4–6 (F-M5-2: the §2 estimate of
5–8 assumed labels, reward, opponents and eval were already off Python). **Critical path:**
0 → C / D → G → H plus T2, about 12–16 agent-days. With four lanes in flight that is ~2 calendar
weeks.

**What M5 then unblocks in §4:**
- `SubprocVecEnv` + `async_vec_env.py` + the per-env bridge child for training;
- `gen3_env.py` + `wrappers.py`'s opponent plumbing;
- `--compile-opponents*` + the per-env model cache;
- the Python trackers / `TurnDelta` (after C + D + E);
- the assembler and the `live_view()` memos (after E + H);
- `search_session.py`'s JSON protocol (after I — I is BUILT; the default switch to `inproc` is the step before the deletion, and gate ① retires with the JSON road it compares against).

**Gate.** Slice N (env level): obs rows, masks, labels, rewards and dones equal the Python
`Gen3Env` on recorded battles with both sides scripted from the recording. A depth-3 successor slice
(search's default depth) equals `search_driver`'s rows. The two front ends are byte-identical.
Training throughput at `--n-envs 48` is measured as an interleaved A/B against the current path.
**Size: see the lane plan (≈ 24–33 agent-days, T2 included).**

**Transport at M5 (DECIDED 2026-09-26, owner + benchmark): BOTH front ends, one core.** The
benchmark above measured them equal on speed, so the default per consumer follows the
crash-isolation rule. Prerequisites that stand:
- `trainer_turn_benchmark.py` defaults to the Rust bridge before G's A/B (TECH_DEBT_BACKLOG P2).
- Every build stamps the core with its commit and source hash, and BOTH front ends refuse a mismatch
  at load.
- The FFI module loads from an absolute path, never from whichever `.so` comes first on `sys.path`
  (the 09-09 class).

**Findings Phase A hands to the build** (detail in the record's §5):
- **F-M5-1:** a shared READING-class Hidden-Power refusal. A transformed mon KO'd by Hidden Power
  loses its temporary types at switch-out, and both trackers re-read the live board. It is not
  reachable from `data/teams/`, but it is reachable from 20 ladder-corpus teams, so M7 is exposed.
  The fix is proposed, not applied (a training-input change).
- **F-M5-2:** the M5 size above.
- **F-M5-3:** the GIL re-take bounds a rollout thread that shares an interpreter with a busy Python
  thread, on both front ends.
- **F-M5-4:** the core ran 2.8–4.8 % faster out of process at N = 1. Cause unverified.
- **F-M5-5:** CLOSED by Lane 0's gate ① (2026-09-28): the env core's rows, masks and tokens equal
  THIS checkout's `sim_bridge` `__OBS__` frames byte for byte on the core's own input logs — 4,545
  frames over 34 logs (28 finished episodes, bridge-corpus teams, seeded random policy), with a teeth
  test (the same logs replayed under the other `decision_tense` move 14 of 97 rows).
- **F-M5-6:** `ctypes` over a C ABI keeps the crate std-only.

### M6 — THE CUTOVER, then the DELETION PASS

**Status (2026-09-25): the SWITCH IS MADE — `--obs-source core` is the production default on the
rust bridge (owner decision, 2026-09-25); the DELETION PASS ran 2026-09-26** (`43712881`, `97a30387`: search's non-core roads and the port's one-sided view; the rows still blocked are named in §4). 🚨 **A registered DEVIATION:**
training switched BEFORE the CUTOVER tier's registered counts were met — at the switch the stress
stood at ~60–70% of every count (the ladder full tier 7,960 + 7,920 of 11,407 + 11,407, the pool 6,000
of 8,628, slice N 2,100 / 700 / 700 / 350 of 3,000 / 1,000 / 1,000 / 500, the fuzzers ~70%), with ZERO
open CUTOVER-class divergences, targets 5, 9 and 10 met, and every other target clean so far. The
stress KEEPS RUNNING to its registered counts as POST-SWITCH confirmation; **any CUTOVER-class
divergence it finds REVERTS the default to `python`** (a one-line change) until it is fixed. Before
the switch (the prep record, kept):
`--obs-source core` (`gen3_core_obs_source_v1`) — the rust `sim_bridge` ships the trainee's row as
an `__OBS__` frame built through the PARSER (§6c) and `Gen3Env` takes it, refusing a frame of
another battle / decision / turn, a NaN cell or a disagreeing mask; labels, reward, the tracker fold
and the mapper stay Python. `core_events --obs` now also requires the parse-chain row to equal the
step-chain row at every decision. Slice N (env level, two envs in lockstep) is pinned at COMMIT and
MILESTONE (`main/rust_core_cutover/slice_n_test.py`), zero differences, no allowlist. Its first
find, **F1** — the live env recorded a trainee decision on a PHANTOM step (the trainee not asked to
move: a `wait` request, or poke-env's re-embed of an answered request), 5.0% of steps — is FIXED on
the Python side (`gen3_no_phantom_decision_v1`, a TRAINING-INPUT change folded into the boundary; a
`wait` request reaching the record raises). The
stress (`python -m main.rust_core_cutover`, the CUTOVER subsection of §3) runs concurrently with the
training queue under its governor. Measurements:
[`research_state/measurements/rust_core_cutover_2026-09-24/`](../research_state/measurements/rust_core_cutover_2026-09-24/README.md).

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

**M7 RUST-ONLY READINESS (owner, 2026-09-28: "make sure everything will work once we are rust only for parsing").** M7 is not done when the three named consumers move; it is done when NOTHING parses through Python. Required: (1) an INVENTORY, generated not hand-written, of every importer of `poke_env`, `agents.battle` (Gen3Battle / LiveView / TurnView / TurnDelta) and the Python trackers — production code, tools, meters, the prober and its web views, AND tests/fixtures — each mapped to its core replacement or to deletion; (2) a routine-tier IMPORT GATE that fails if any non-allowlisted module imports them after M7 (the allowlist shrinks to empty with the deletion pass, as the size gate's did); (3) every consumer's parity on the SAME recorded battles before its Python path is deleted (the prober's views on core-walked traces equal to today's; anchors byte-identical actions; the ladder client via `ladder_drift_scan`); (4) the test tiers re-run with poke-env UNINSTALLABLE (a clean env without the vendored fork on the path) — a test that only passes because the Python parse is present is found there, not in production.

**Program total: ≈ 38–56 agent-days** (as planned 2026-09-23; M5 alone re-sized 2026-09-26 to ≈ 24–33 incl. T2 — the M5 lane plan) of build, gating and deletion, wall time dominated by the
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
| **CUTOVER** | ONCE, before training switches — **reformulated 2026-09-24 (owner option 2): run CONCURRENTLY with the training queue at `nice 19`, gated on COVERAGE and COUNT, not hours**; the pre-registered targets are the subsection below | every full-tier LADDER team from both slots, the whole pool × 12 partners per slot, procedural teams, the `production` policy on pool / ladder / procedural, slice N env-level episodes, the four A/B fuzzers, a soak | every battle re-runnable alone from its row (key, team indices, source) and every divergent battle's input log banked | wall time set by the shared box (units are minutes long, resumable) | `python -m main.rust_core_cutover run --out ~/gen3ai_archive/cutover_stress_2026-09-24 --detach` |

### The CUTOVER tier — PRE-REGISTERED targets (2026-09-24, written before any stress result was read)

The owner chose option 2: the cutover stress runs CONCURRENTLY with the live training queue, at low
priority, "and make it so we feel very good about it". Because the cores are shared, the gate is
COVERAGE and COUNT — every target below met with **zero unexplained divergences** — never a number
of hours. The driver is `src/main/rust_core_cutover/` (`gen3_core_cutover_stress_v1`); its
`plan.py` IS this table (the registration on disk refuses an edit), and it runs from a `git
archive` PIN with its own self-check binaries (never a worktree). Progress:
`python -m main.rust_core_cutover status --out ~/gen3ai_archive/cutover_stress_2026-09-24`.

**What a divergence means** (`main/rust_core_cutover/verdict.py`): **CUTOVER** — the core against
the Python path training reads (every slice E / T / O / N field, slice V's `[core]` column and
mask, every `[ALIGN]`, every refusal); **READING** — the reading (poke-env's, which the core mirrors,
so both paths hold it) against the SIMULATOR (`[BOARD]`, `[TRUTH]`, a bare `[SIM-FACT]`);
**VIEW-ROAD** — the legacy view road's projection (a bare `[PRESENTATION/…]`; §4 deletes it). The
gate is **CUTOVER = 0**; every READING and VIEW-ROAD class is NAMED with its rate in the readiness
report (READING classes are its "poke-env reading findings"), and an unrecognised key counts as
CUTOVER. (The taxonomy was written after a 40-battle procedural smoke showed three V divergences
of the view-road and R3-residue classes already on record in M2 / M3; no target was set after it.)

| # | target | count (registered) | gate |
|---|---|---|---|
| 1 | slices **E / V / T / O**, both viewers, every decision, live == offline, the self-check build: **the LADDER full tier from both slots** (pass A: key `k` plays teams `2k` / `2k+1` — the MILESTONE recipe, so its three NAMED known divergences replay byte for byte; pass B: the same pairs SWAPPED) | 11,407 + 11,407 battles; **all 22,813 teams in each pass** | CUTOVER 0; refused 0; errored 0; A's three named keys fire in exactly their named keys |
| 2 | the same slices on the **pool**: every team × 12 partners (offsets 1 … 233) from EACH slot, seeded-random | 8,628 battles | as 1 |
| 3 | the same slices, **the `production` policy** at T = 1: pool (every team × 4 partners), the ladder MILESTONE tier from both slots, procedural | 2,876 + 800 + 500 battles | as 1 |
| 4 | the same slices on **fresh PROCEDURAL** teams, seeded-random | 4,000 battles (8,000 teams) | as 1 |
| 5 | slice E on the 22-scenario protocol corpus × 2 + every byte-fuzz fixture | 1 unit | as 1 |
| 6 | **slice N** (env level, M6's slice): two `Gen3Env`s in LOCKSTEP on the same seed, teams and actions, one `--obs-source python`, one `--obs-source core`; per decision the obs row (bytes), the mask, the reward, `terminated` / `truncated` and EVERY training-only label key the production config emits (ARCHITECTURE.md §7) equal | 3,000 pool seeded-random + 1,000 pool `production` policy + 1,000 ladder + 500 procedural episodes | any difference = CUTOVER; errored 0 |
| 7 | the four Rust-vs-Node **A/B fuzzers**, the self-check build, each by its own green-gate definition: `ab_fuzz.js` (omniscient STATE), `ab_fuzz.js --protocol --format gen3ou` (bytes), `bridge_ab_fuzz.js --format gen3ou` (per-side + request), `gen_sim_bridge_diff.js --format gen3ou --persistent` (external consistency) | STATE and BYTES: 10,000 ladder-full + 3,000 `ourandom` + 1,000 pool each; BRIDGE: 5,000 ladder + 2,000 pool + 1,000 trapping; SIM-BRIDGE: 2,000 ladder + 1,000 pool (**amendment 1**, 2026-09-24: these two fuzzers have no `ourandom` mode — the first registration named it, its two units errored on the flag, and `pool` replaced it at the same counts; recorded in the registration's `amendments`) | 0 non-allowlisted diverged / panic / parse_error (and 0 `errored`, 0 `drain_timeouts`) |
| 8 | **SOAK**: one persistent Rust bridge child per unit behind a real `Gen3Env` (the training transport), RSS of the child and of the env process sampled every 250 episodes | 6 children × 10,000 episodes (python obs) + 4 × 10,000 (`--obs-source core`) — each ≈ 4.6× a production child's 3-h life | 0 errors, 0 child replacements (a crash), and no unit's child or env RSS above 1.10 × its episode-250 sample at its last sample |
| 9 | the `--debug` smoke and **the first two minutes of a real launch** under `--obs-source core` on a throwaway run dir (the forkserver preload, the compile and warm-start layers) | 1 launch | reaches its first PPO iteration; no `FATAL` / `Traceback` / `[ModelVersion] FATAL`; the launch line stamps the obs source |
| 10 | **throughput non-regression**, `trainer_turn_benchmark.py --pin-battles` over the RUST bridge, core vs python obs source, interleaved | ≥ 6 interleaved pairs | the upper end of the 95% CI of (core − python) / python per-turn time ≤ +3% (the equivalence rule: the delta's own CI inside the bar) |

Not in the tier: depth-3 successors (the original row's) — successors are search's, gated by M2's
three search gates and M4's row road, and become a training concern only at M5.

**The stress's cost to training** is itself pre-registered (`main/rust_core_cutover/governor.py`):
the live arm's MARGINAL fps per PPO iteration, read-only from its `launcher_child.log`; a 15-min
OFF window before the stress's first unit, a 20-min OFF window every 4 h and whenever a new run
appears (an arm switch); if the median of 5 ON iterations falls below 0.85 × the OFF baseline, the
worker cap drops by 2 (floor 2; start 6 of 16 cores), and the governor never raises it.

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
| DONE `43712881` (M1, 2026-09-26) | `agents/battle/event_fold.py` (`ViewEventFolder`) + its unit test and parity fuzz | 494 (−405 module, −~1,000 with tests) | the core's events serve every successor |
| DONE `43712881` + `97a30387` (M2, 2026-09-26) | `agents/battle/view_adapter.py` (`LiveView.from_view_json`, `ViewBattle`); `agents/training/view_successor.py`; the `view_pN` / `view_pN_at` / `pN_chunks` JSON of `search_driver` (`view.rs`'s JSON render); the `view_fallback_*` counters; `search_impl_parity.py`'s view allowlist entries | 617 + 500 | the view road |
| DONE IN SEARCH `43712881` (M2, 2026-09-26); the rest → M7 | the PROTOCOL road: `obs_materializer.materialize_branches*`, `open_branch_fork`, `_PlayerSnapshot`, `_ReplayObsPlayer`; `--materializer` values `protocol` / `view` | most of 1,020 | DONE: search's `_materialize` / `_root_fork` / `_branch_fork` and fork caches, `open_view_fork`, `SearchConfig.materializer`, `--materializer` (deleted_flags.md), `--search-impl node`, `TreeNode.chunks`. **KEPT until M7:** `materialize_branches` (+ its `open_branch_fork` / `materialize_branches_from` halves), `_PlayerSnapshot`, `_ReplayObsPlayer` — the prober's `lookahead` calls `materialize_branches`, and `materialize_decisions` / `materialize_from_record` (cf producer, prober, `ab_racing`) run on `_ReplayObsPlayer` |
| DONE `97a30387` (M2, 2026-09-26) | the view road's RUST half: `view.rs` (`one_sided_view`, `SideObservation`, the reveal fold with `BridgeChunks::observed` / `enable_view_fold`), `tests/one_sided_view_test.rs`, `search::Capture::views` + `Resolved::views_at`, `core_events --views`'s `views` / `truth` payload; `view_adapter.py`'s rules (`ViewBattle` survives — `core_successor` builds on it — though since M4 search's core road reads rows, not `core_successor`); `ViewEventFolder` use in `view_successor.py` (the fold's M1 row stands); the view half of `one_sided_view_parity_fuzz_test.py` and `rust_core_parity_views.py`'s PROJECTION column (the core column stays); `view_materialize_benchmark.py` | ~1,100 Rust + ~700 Python | made deletable by `materializer=core` (M2 adoption). Also: `tests/view_fold_opt_in_test.rs` (its subject is gone), `driver_timing`'s `view` row, slice V's reading-vs-engine TRUTH checks (they read the projection's engine-sourced `ours` block; the core column + the core's board audit carry the Baton-Pass / Spikes teeth), the three projection-only `LADDER_KNOWN_DIVERGENCES`. `one_sided_view_parity_fuzz_test.py` → `core_row_parity_fuzz_test.py` (search's core row vs the poke-env replay, D10 included) |
| DONE `43712881` (M2, 2026-09-26) | search's FALLBACKS: `search.py`'s `view_fallback_intermediate` / `view_fallback_no_payload` and the depth-≥2 protocol fallback; `RealizedWidths`' view counters | ~120 | a core leaf is a version, so there is nothing to fall back from |
| DONE (M4, 2026-09-24) | the TYPED SHORTCUT: `CorePath::Typed`, `BridgeSession::typed_side_lines`, `session_from_record_core` in search, `SearchConfig.core_path` (`--core-path`), and with them the INTEGRITY mode (`SearchConfig.integrity`, `--search-integrity`, `expand_many`'s `integrity`, `streams_equal`, `CoreIntegrityError`, `text_view`) | ~250 | measured to save nothing (§6); every version folds `parse(render)`, §6c's one observation path. KEPT: a CORE session's per-line engine SCOPE (`BridgeSession::side_scopes`) — no line, only the owner truth the native record's grouping gate and the parse-reproduces-step gate read |
| DONE `c97358e8` | the matching `agents/battle/poke_env_findings.py` entry (PE-V10 / PE-R1b / PE-V16), all deleted, registry empty | 1 entry each | a TRAINING-INPUT change, the owner's call; deleting the entry TIGHTENS slice V |
| PART DONE `43712881`; rest SKIPPED 2026-09-26 | `agents/training/clone_pins.py`; `ViewSuccessorFactory._clone_tracker`; `training/turn_delta_legacy.py` (test-only today) | 149 + 327 | the tracker fork becomes a pointer copy. DONE: `_clone_tracker` (with `view_successor.py`). **`clone_pins.py` → M7**: `obs_materializer._PlayerSnapshot` (the prober's `materialize_branches`) still pins through it. **`turn_delta_legacy.py` SKIPPED**: its four test consumers (`battle_context_test.py` 36 call sites, `move_attribution_test.py`, the two `poke_env_gaps` fuzzers) use it as the harness that checks `BattleContext`'s snapshot-derived flags — deleting it is a test REWIRE onto the event fold, not a deletion |
| SKIPPED 2026-09-26 — still LIVE in training | `agents/training/turn_delta.py` (`TurnDelta` + `build_from_events`), `battle/turn_view.py`'s `TurnDelta`-only reads, `episode_tracker.build_delta*`, and the Python trackers of `episode_tracker.py` (`RecencyTracker`, `PairHistoryTracker`, `EventWindowTracker`, `EpisodeTracker.record_context` / `advance_window`), `progress_clock.py`'s obs half, `hidden_power_tracker.py`, `wish_belief.build_wish_pending`, `sleep_belief.build_sleep_sources`, `opp_intent_labels.build_opp_intent_label`; the unported `choice_band_tracker.py` (no production reader) | 603 + ~1,900 + 240 | slice T (the core's trackers, label, reward); `TurnDelta` survives until then only as the label's ORACLE. Joins the pass WITH the shaped reward path (its other reader). **Blocker (verified 2026-09-26):** under `--obs-source core` the env still runs `EpisodeTracker.record` / `update_progress_clock` (→ `build_delta_from` → `TurnDelta`) every trainee step for the α/β LABEL, the progress clock and the REWARD (`calc_reward` reuses the folded delta), and every policy OPPONENT (`RLPlayer.track_decision`) folds them too — the cutover moved only the trainee's ROW. They leave when labels + reward + opponents come off the Python path (M5 / T2) |
| DONE `e3ef16db` (M3, 2026-09-26; owner call: a shaped checkpoint REFUSES) | the SHAPED reward path: `reward_potentials.py`, `reward_bias_terms.py`, `reward_verify.py`, the shaped branches of `reward_manager.py` / `reward_composition.py` / `reward_config.py` / `reward_weights.py`, the clock's CHARGE (`last_penalty`), the eval-side reward clock, and the 14 shaped flags (into `designs/deleted_flags.md`), with their tests | ~1,100 + ~6,000 tests | the Rust reward is the win indicator (§2 M3). **Parity:** `reward_golden_test` (v2) recorded at `029cee83` passes unchanged — production reward and `win_margin` byte-identical. **Resume/fork (owner, 2026-09-26):** a checkpoint that trained shaped REFUSES LOUDLY (`model_version.shaped_reward`, in `resolve_config` + `checkargs`), never switches silently to the indicator — run it pinned to ≤ `029cee83`. 🚨 **Arm S is no longer re-runnable on HEAD: its comparator runs PINNED** (≤ `029cee83`). `MODEL_CONFIG_VERSION` 122, floor unchanged |
| after M7 | RESHAPE `BoardReading`: its fields mirror poke-env's `Battle` (`_player_username` …) because slice V compares field by field; design the reading for the view once poke-env has no production user | — | a reshape, not a deletion |
| SKIPPED 2026-09-26 — premise false until M5 / T2 | the Python pipeline's PERF layers — `observation/assembler.py` (incremental cache), the `live_view()` memo (`_state_epoch`, the request-change door), the `live_view_build_micros` memos | ~600 (assembler 498) | the Python encoder SURVIVES as the oracle; its perf scaffolding does not (a simpler oracle is a better oracle). **Blocker (verified 2026-09-26):** the Python encoder is NOT oracle-only yet — every policy OPPONENT in training (the pool / self-play / exploiter `RLPlayer`s of `wrappers.py`) encodes through `RLPlayer.embed_battle` WITH the assembler, eval does too, and the `live_view()` memo serves the trainee's own per-step `tracker.record` / mask / progress clock / reward reads (the memo's own docstring: five builds per decision). Deleting them is a training-throughput regression; they leave when opponents move to T2 |
| SKIPPED 2026-09-26 — blocked on M5 | `utils/bridge/search_session.py`'s JSON protocol (`open_root` / `expand_many`), `search_driver`'s search verbs, `driver_timing.rs`; node `search_driver.js` / `replay_driver.js` / `replay_kernels.js` once nothing diffs against them | 405 + 785 (node) + 128 + the driver verbs | search runs in-process on versions. **Status (M4, 2026-09-24): OFF search's per-arm path** — a core arm's reply is the leaf's ENCODED ROW + mask + choice tokens (`expand_many`'s `rows`), so the view / legality / events JSON, `CoreSuccessorFactory`, `core_view` / `view_adapter.ViewBattle`, the Python trackers and the Python encoder no longer touch a search successor, and the core road opens no Python prefix fork. The protocol verbs themselves remain (the in-process `successors()` is M5's transport decision); the JSON core payload and `core_successor.py` stay, unused by search, until the pass. **2026-09-26:** `core_successor.py` DELETED `43712881`; the JSON protocol is still search's (and the prober's counterfactual's) only transport — the in-process `successors()` is M5 — and the node drivers are still diffed against: `harness/search_impl_parity.py`, `harness/replay_impl_parity.py`, `utils/bridge/search_clone_parity_fuzz_test.py` (and `local_sim_bridge.js` names `replay_driver.js`). `driver_timing.rs` stays with the protocol (its `view` row went in `97a30387`) |
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
| **M6 CUTOVER — DONE 2026-09-25** (ledger entry, between reads; the stress continues as post-switch confirmation, §2 M6) | **every number after it runs on the core**; `metadata.json` records the env core + commit; a resume pinned to a pre-cutover commit stays on Python (the launcher's pin), so no run spans the cutover without a re-pin | core | core for training-side evals |
| M7 | — | — | anchors / ladder stamp `client: core` |

---

## Decision record

Owner decisions are marked **(owner)**. `L…` is the ledger line as `ledger_index.md` lists it.

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-09-23 | Build the core **(owner)** | The RUST CORE PROGRAM, licensed on unification ("a low tech debt, robust, performant and unified approach; search will be on the table") | Licensing on the design's §7 re-profile (it read LICENSED at 59% glue; recorded only) | §0; Phase 0 record `measurements/rust_core_phase0_2026-09-23/`; ledger L21259 |
| 2026-09-23 | Operating model **(owner)** | Build alongside, continuous parity gate, ONE cutover, ONE deletion pass; search the only early adopter | A flag / flip / delete per milestone; moving the prober, anchors or ladder client early (each would ADD a path) | §1 |
| 2026-09-23 | Parity gate **(owner)** | One harness, three tiers (COMMIT / MILESTONE / CUTOVER), cost proportional to the decision guarded | — | §3 (owner amendment 2) |
| 2026-09-23 | Order **(owner)** | M1 → M4, then the CUTOVER (M6) in today's process-per-env shape, then the deletion pass, then M5; M5 not started until the owner says so | M5 before the cutover (two changes at once) | §2 |
| 2026-09-23 | Events **(Phase 0)** | The core emits a SUPERSET: truth fields + the reading projection; `gen3_event_value_schema_v1` kept verbatim on the projection | — | §0 |
| 2026-09-23 | Observation path **(owner)** | Always through the parser, training included; search keeps the typed shortcut | Typed-at-source observation in training | §6c |
| 2026-09-23 | Search determinization **(owner)** | DEFERRED ("kick search determinization down the road") | Building worlds now | §6a |
| 2026-09-23 | Reward in slice T **(orchestrator; owner consulted)** | Win indicator alone; shaping not ported, the shaped path into the deletion manifest | Porting PBRS / bias terms | §2 M3; `c49ef704` |
| 2026-09-24 | Poke-env reading bugs **(owner)** | Fixed in the fork (R1–R3), a training-input change; the findings registry emptied | Carrying them as allowlisted divergences | `c97358e8`; ledger L21305 |
| 2026-09-24 | M3 gate **(owner)** | Slice T gates the CONSUMERS; `TurnDelta` not ported as a structure | Field-by-field `TurnDelta` parity | §2 M3; ledger L21313 |
| 2026-09-24 | Corpus **(owner)** | The Metamon ladder-usage corpus joins the M4 gate and every fuzz / parity gate | Pool-only corpora | `886ae28f`; ledger L21319 |
| 2026-09-24 | Emission self-check | Every emitted line checked at emission; ON in tests and fuzzers, compiled OUT of release | — | `239ebe3e`; ledger L21297 |
| 2026-09-24 | CUTOVER stress **(owner, option 2)** | Run CONCURRENTLY with the training queue at low priority, gated on coverage and count, targets pre-registered | option 1 (not recorded in this doc) | §3 |
| 2026-09-25 | The switch **(owner)** | `--obs-source core` default; registered DEVIATION at ~60–70% of the counts, the stress continuing as confirmation, a CUTOVER-class divergence reverting the default | Waiting for the full counts | `ac0b6469`; ledger L21395 |
| 2026-09-26 | Deletion pass part 1 | Delete what is only an oracle; SKIP five rows with verified blockers | Deleting on the manifest's premise (Python was not yet only an oracle) | §4; ledger L21403 |
| 2026-09-26 | Shaped checkpoints **(owner)** | Refuse a resume or fork of a shaped checkpoint | A silent switch | `e3ef16db`; ledger L21417 |
| 2026-09-26 | M5 transport **(owner + benchmark)** | BOTH front ends over ONE core; default per consumer by crash isolation; 12 lanes, ≈ 24–33 agent-days incl. T2 | One transport; the registered 5–8 agent-day size | §2 M5; ledger L21421 |
| 2026-09-27 | Scripted bots **(owner)** | Port them (Lane F) | Keeping bot battles on the old path until M7 | TASK_BACKLOG T3 ("owner yes"). BUILT 2026-09-29 (Lane F paragraph, §2): as-is, bugs included (F-LF-1 is the owner's call); a bot reads the core's own reading; its randomness is the same CPython stream, not explicit inputs |
| 2026-09-27 | Doc status **(owner)** | ALWAYS-CURRENT while implemented; the Q-head spec is its sibling | Explicit-only updates | `556eb4f8`, `51e36c09` |
| 2026-09-28 | Learner pipeline **(owner)** | Lane K in M5: torch ≥ 2.8 in a new env (drop the miscompile split), a sync-free update with diagnostics on a cadence, a persistent compile cache, CPU-lane hygiene; K1+K2 in the first wave | Leaving the learner out of M5 (env-only); upgrading torch in place under live pinned runs | ledger 2026-09-28 (compile miscompile, `6521f420`); learner benchmark `2d39b130`; C's TB (`train_ms` 56.9 s, noise probe 5.35 s) |
| 2026-09-28 | Compile lifecycle **(owner)** | DECLARED signatures compiled at startup, cache frozen before the first iteration, undeclared = FATAL naming the guard (K6); AOT artifacts for inference slots | Warm up one iteration and lock (`8fc297a2`, the interim); lazy compilation | this doc Lane K; `8fc297a2` finding 2 (one unidentified iteration-1 signature) |
| 2026-09-28 | Declared lifecycle **(owner)** | M5 design principle: startup declares + acquires every resource, steady state acquires nothing, `*_after_freeze` counters must stay 0; K7 AOT training-step spike decides dynamo vs AOT for training | Lazy initialization in production runs; assuming training export is immature without measuring | owner 2026-09-28 |
| 2026-09-29 | Entropy / policy sharpness **(owner)** | NO direct adoption of a lower entropy bonus; the entropy-anneal arm (X23) runs only after the architecture is stable (M5 + the discrete-token boundary), judged by Lane S's policy-spectrum instrument on a fixed turn bank (rank-mass spectrum; near-best vs dominated mass and starvation over the whole move space, from branch ground truth) | Adopting a lower `ent_coef` on X23's strength read alone; a contrived single-move (Toxic) starvation probe | owner 2026-09-29 |
| 2026-09-29 | Sizing (N envs, rollout shape, slots, buckets, lanes) | A registered SIZING STUDY after M5's parity gate (order constraint 5): N swept at a fixed rollout size, throughput split by component + learning per sample + the noise-scale meter; slots DERIVED from the declared pool, buckets from rows-per-slot, lanes = min(slots, 8) | Picking N ("a couple thousand") or a slot count up front; tuning slots by hand | owner question 2026-09-29; T2 lane measurements |
| 2026-09-29 | Lane G collector **(owner)** | Complete-game buffer + sample-count update trigger; behaviour log-prob + policy version stored per row; no row dropped for age and no extra truncation by default — PPO's ratio corrects every row, age-bucketed ratio/clip/KL logged, per-game version pinning as the first remedy if needed (owner: dropping/down-weighting by age biases the data); n_steps retired; K10(b) subsumed | Fixed n_steps windows with dropped critic rows; a bolted-on delayed-label buffer; an age cap that drops stale rows; fully async actor-learner (more staleness, more machinery) | owner 2026-09-29; design_learner_recipe.md (dropped-row finding) |
| 2026-09-29 | FFI signatures (Lane A) | GENERATED from one Python table into a marked region of `ffi.rs` (wrappers call `imp::<name>` with the same arguments) + ctypes argtypes from the same rows + a compiled-in table id compared at load | Hand-written ctypes declarations (the prototype); a C header + cbindgen (a new dependency, and it would generate from Rust, not from the table both languages already read) | `m5_laneA/PROGRESS.md`; `ffi_test.py` |
| 2026-09-29 | T2 AOT on torch 2.8 **(owner: install the headers; Lane T2, measured)** | CUDA 12.6 headers in `gen3ai_torch28` only; backend `aot` (≥ 2.8, weights as inputs, parity-gated like `graph`) ADDED as the path for a Python-free (C++/Rust) caller; `graph` stays the DEFAULT on both torches | Making `aot` the ≥ 2.8 default (the owner's conditional, pushed back on: it passes parity but is 1.3–1.5× SLOWER — 1.91/2.31/3.31 vs 1.25/1.81/2.76 ms at B = 8/48/128 — because an AOT call cannot be graph-captured); embedded-weight packages (weight swap wrong: 0.076 logp) | this doc's T2 DESIGN "Backends"; `research_state/measurements/m5_t2/window_2026-09-29/` |
| 2026-09-29 | T2 multi-slot throughput **(Lane T2, measured)** | LANES: per-lane CUDA streams + graph pools, graphs captured on their lane's stream; Lane E's shape 9.89 → 3.95 ms per flush at 4 lanes | One vmap forward over stacked weights (NO-GO: in-place writes into forward-created tensors across the extractor) | this doc's T2 DESIGN "Lanes" |
| 2026-09-29 | T2 inference backend **(Lane T2, measured)** | `torch.compile` per bucket + one CUDA graph per slot × bucket (shared pool), weights in stacked per-group slot storage loaded by in-place copy, every load and startup slot × bucket parity-gated at the compile gate's decision bars | AOTInductor (blocked on 2.5.1: the unsplit graph miscompiles, 0.68 legal log-prob; blocked on 2.8: no CUDA toolkit on the box; not faster where it ran); Inductor `reduce-overhead` (2.1 s/call: a host-built constant skipped cudagraphs); dynamic-shape compile (fails on 2.5.1); eager CUDA graphs (24 host syncs in the eager forward) | this doc's T2 DESIGN paragraph; `research_state/measurements/m5_t2/` |
| 2026-09-29 | Search in process (Lane I) | The search TREE as step-built versions (byte-equal to `search_driver`'s core road); PLAYOUTS on a linear, branchable training-path `Game` (the parse fold) reseeded at the branch point (CRN); the policy is a HOST loop over batched pending rows (no Rust → Python callback); the three deleted M2 search gates replaced by clone-independence + decision-equality + the transitive row oracle | Playouts on version chains (a child keeps its parent: 100 engines per 100-decision branch); a ctypes callback from Rust into Python per decision (one GIL crossing per decision, no batching across branches); an in-handle worker pool (a per-call thread spawn breaks the declared lifecycle — callers hold one handle per thread instead) | `m5_laneI/PROGRESS.md`; `successors_integration_test.py`, `tests/search_game_test.rs` |
| 2026-09-29 | Nature / EV training label (`gen3_true_spread_labels_v1`; Lane C F-LC-5 / F-LC-6) | The label is the truth mon's DECLARED nature and stat-effective EVs, which both readings already hold (poke-env's own-team spread backfill, mirrored by the Rust reading), with a THROWING guard that the declared set at its true IVs reproduces the request's stats; no cache, Python and Rust together | Keeping the IV-31 stat inversion and inverting with the true IVs instead (the inversion is not identifiable: its Σ ≤ 510 omits HP EVs, so it named a wrong nature for 2.5 % of the ladder's IV-31 mons); keying the cache by the full team (the read is a few lookups, so the cache buys nothing) | `research_state/measurements/label_coverage_2026-09-29/`; the parity gate's milestone coverage |
| 2026-09-29 | Process front end transport (Lane B) | ONE anonymous `memfd` passed to the child by fd, laid out from the column table, the child writing a header the host compares; the child's handshake (wire + stamp) BEFORE it reads a byte; death = EOF on the reply pipe; respawn on the same mapping, COUNTED | A named `/dev/shm` segment + unlink on exit (the prototype: a SIGKILL of the parent before the unlink leaks it — hygiene by cleanup, not by construction); unlink-after-handshake (still a window); `PR_SET_PDEATHSIG` for orphans (it fires on the spawning THREAD's death) | `m5_laneB/PROGRESS.md`; `proc_integration_test.py` gate ③ |
| 2026-09-29 | Opponent routing (Lane E) | A DECLARED route table (`external` / T2 `policy` slot / core `bot`) in the spec; the per-episode route CHOSEN in Python by the wrapper's rules, STAGED as `ep_opp` with the teams, echoed as `opp_route` / `opp_slot`; bots played in the core with a declared seed rule; a pool refresh = a T2 LOAD into a free slot | Routing by model id per step (host bookkeeping the core already knows); porting the selection rules to Rust (per-episode, not hot); reusing a slot still played by a staged episode | this doc's Lane E paragraph; `m5_laneE/PROGRESS.md` |
| 2026-09-29 | Lane G collector build (Lane G, unit 1) | The complete-game buffer as ONE preallocated row ARENA + a completion-ordered FIFO; complete-game GAE computed at game end with sb3's own arithmetic (bit-identical to sb3's window GAE on the same game); the update consumes EXACTLY the target (a multiple of lcm(micro-batch, N)), splitting at most the one game that straddles it (its tail trained next update); rows laid column-major into the model's own `[D / N, N]` buffer so `train()` is untouched; a cut game's rows released and counted; trainee sampling = the keyed draw | Consuming every completed row at the trigger (a variable D: a ragged micro-batch = an undeclared compiled signature); a separate learner buffer class (every buffer reader re-checked); computing GAE on the filled buffer (a split game would bootstrap at the column edge, the thing order constraint 6 retires) | `designs/training/rust_collector.md`; `store_test.py`, `collector_integration_test.py` |
| 2026-09-29 | T2 buckets for training (F-LE-9, Lane G) | `(8, N)`: 8 serves an opponent slot's rows (every per-slot count in Lane E's 600-step read at 48 envs / pool 20 was 1–6, 94 % were 1–4; padding is nearly free at small batches — 1.06 → 1.23 ms from 2 → 8 rows), N the trainee's batch and any opponent slot above 8 rows (a small pool early in a run) | Lane E's `(2, 4, 8, 16)` — 16 served nothing and 2 / 4 each cost a compile (~75 s on 2.5.1) and a capture per slot for a sub-0.2 ms saving; one bucket per slot count (no measurement supports it). The SIZING study re-reads it at its N | Lane E `bench_48env_pool20.json` (rows per slot); T2 DESIGN (per-bucket replay times) |
| 2026-09-29 | The opponent is polled only when its order is sent **(Lane E finding, training-input change)** | `gen3_no_phantom_opponent_poll_v1` in `SingleAgentWrapper.step` | Reproducing the phantom poll in the core (it corrupts the opponent's progress clock and consumes draws) | F-LE-1: 5 of 281 opponent rows off by one clock step, each right after a phantom |
| 2026-09-29 | The M5 gate harness's shape (Lane J) | ONE registry row per lane naming its OWN gate tests (delegated, not re-implemented), folded into PASS / FAIL / NOT BUILT / INCONCLUSIVE / NOT RUN; a routine test fails when this doc marks a lane BUILT and it has no row; the verdict composed from component JSONs, a missing component = NOT RUN | Re-implementing each lane's comparison inside J (a second, drifting copy of every gate); inferring lanes from test filenames (a tier is declared, never inferred) | `lanes_test.py`; the milestone gate table in `m5_laneJ/PROGRESS.md` |
| 2026-09-29 | Slice N "at the env level" (Lane J) | Lanes C's and D's record / replay JOINED at N = 48, T = 8 / 5, episodes chained through the auto-reset, both front ends recorded and compared, both learner-read masks compared | Running each lane's N = 1 slice and calling the conjunction slice N (never exercises N envs, the auto-reset staging, or thread counts > 1 against `Gen3Env`) | 46,353 decisions, 0 divergences (Lane J paragraph) |
| 2026-09-29 | The depth-3 slice's reading (Lane J) | Lane I's gate ① at its milestone sources; the depth ASSERTED = `SearchConfig().max_depth`; batches refused identically by both roads counted and printed beside PASS, not treated as coverage | Counting a both-roads refusal as a failure (it is F-LI-1, a defect of both roads, not a parity gap); silently ignoring it | 25 refused batches of 53,736 rows (F-LI-1) |
