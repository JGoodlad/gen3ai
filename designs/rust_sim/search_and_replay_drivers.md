# The offline drivers — the SEARCH server and the REPLAY family

<!-- Lifted verbatim out of `src/rust_sim/CLAUDE.md` on 2026-09-08 (the leaf split: 2,659 -> a command card). This tree is ALWAYS-CURRENT: state the truth, never narrate a change. The leaf keeps the rule, the command and the hazard; this file keeps the detail. -->

`src/bin/search_driver.rs` over `src/search.rs` is the drop-in replacement for BOTH node offline
drivers (`search_driver.js`'s persistent clone-and-branch server and `replay_driver.js`'s one-shot
`replay`/`reroll`/`reroll_many` verbs). This document holds the protocol, the kernel reuse, the two
gates per family, the honest scope of `pre_state`, and the clone-branch gate. The leaf keeps the
snapshot API contract, the invocation and the hazards.

---

## The clone-branch gate (`tests/bridge_clone_branch_test.rs`)

**Gate: `tests/bridge_clone_branch_test.rs`** (7 tests) on a real seeded gen3 battle — two
snapshots advanced with the SAME choices emit byte-identical chunks (and so does the original);
two advanced with DIFFERENT choices leave the parent's turn / chunk stream / outstanding request
/ PRNG seed bit-for-bit unchanged; a `reseed` on a clone leaves the parent matching an
independently-built control; `clear_chunks` yields exactly the tail the un-cleared branch
appends; and the boundary accessors track a partially-answered request, a REJECTED switch, and
game-end. Every test carries a non-vacuity guard (the fixture is a real mid-battle `move`
boundary, the two branch choices genuinely diverge, the reseed genuinely changed the dice) —
without those a snapshot of a finished battle would pass while proving nothing. Its lead pair is
a bulky Blissey MIRROR on purpose: a lead fainting inside the prefix would turn the pause into a
forced replacement and silently stop testing the branch point. FAULT-INJECTION PROVEN (a
`clear_chunks` that also reset the log cursor, and a reject that failed to record its re-issued
request, each fail exactly one named test).


## The SEARCH SERVER: `src/search.rs` + `src/bin/search_driver.rs` (`gen3_rust_search_driver_v1`)

The Rust half of the clone-and-branch search layer: a byte-compatible drop-in for
`src/utils/bridge/search_driver.js`, so `utils/bridge/search_session.py` (the prober's
`better_line` beam) can swap `node search_driver.js` for this binary
with ZERO protocol change.

**The same binary ALSO serves the offline REPLAY family** — see
"## The REPLAY family" below (`gen3_rust_replay_driver_v1`). The two protocols share this
binary and every kernel; they differ only in dispatch (`mode` = one-shot, `cmd` = persistent).

- **`src/search.rs`** — the port of `src/utils/bridge/replay_kernels.js`'s search-relevant
  kernels. Pure helpers, no stdin/stdout, no node map: `aux_rng_from_seed` / `pick_uniform` /
  `random_choice` / `followup_choice`, `Record::parse` + `session_from_record` + `at_turn_start`
  + `build_to_turn` + `recorded_turn_choices`, `resolve_turn_exact` / `resolve_turn`, and the
  omniscient `outcome_of` / `pre_state` renderers. (Plus the replay half's `recorded_queues` /
  `TurnSource` / `resolve_turn_sourced` / `turn_log` / `log_len` / `write_cmd`.)
- **`src/bin/search_driver.rs`** — the persistent stdin/stdout JSON server (`open_root` /
  `expand_many` / `close`), the node-snapshot map, and the monotonic `n0, n1, …` ids. A panic is
  caught and returned as `{id, ok:false, error}` (the `sim_bridge` pattern) — a search server
  that dies mid-beam would strand the caller.

**WHY THE PORT IS STRUCTURALLY SIMPLER.** Node has to clone with `State.serializeBattle`, then
per arm `deserializeBattle` → `restart()` a fresh `BattleStream` with the same `send` wiring →
`sendUpdates()` to flush the whole re-emitted historical log → mark a BASELINE index so only this
ply's suffix is returned (the re-emitted prefix is useless: requests are sent out of band and
never stored in `battle.log`, so a materializer parsing it would never see the team). The port
does `snapshot()` + `clear_chunks()` and the branch's chunks ARE its suffix by construction. Same
wire contract, no byte format, no baseline.

**THE ONE STRUCTURAL DIFFERENCE, and why it is unobservable.** Node writes BOTH sides' choices
then ticks; `feed_cmd` advances synchronously per call. Showdown's `BattleStream._write` is itself
synchronous (`_writeLine` then `battle.sendUpdates()` inline) and the `await tick()` only drains
the async chunk collectors, so both engines resolve at the moment the LAST needed side answers; a
boundary needing both sides does not resolve on the first feed; and a REJECT re-issues the request
and leaves the boundary open in both (`is_choice_done` stays false). Verified end-to-end by the
golden below.

**TWO port-specific decisions worth knowing:**
- **`"default"`** is resolved by the ONE `Side.autoChoose()` port, `bridge::resolve_auto_choice`
  (a forced switch → the first non-fainted bench slot; a move request → the first non-disabled
  request slot; `Move(0)` under Struggle **or a move-lock**). `search.rs::resolve_default` only
  renders that back to a wire token and records the literal `"default"` in `choices_used`,
  matching Node. Exercised by the golden (2 arms).
  **It used to hold its own copy of the logic, and that copy was wrong for a MOVE-LOCKED mon** —
  it scanned the four real moveslots and could answer `move 2`, where the sim's single-entry
  locked request means slot 1 and `choice_is_legal` accepts only index 0. `default` is no longer
  search-only either: `parse_choice` accepts it (and `auto`/`pass`/`skip`) on the production
  wire, because poke-env really sends those tokens — see the bridge row's CHOOSE-path note.
- **`RESOLVE_GUARD`** mirrors Node's `if (guard++ > 40)` — the value BEFORE the increment, so 41
  iterations run. A naive `guard += 1; if guard > 40` cuts one short and can flip a `stuck`
  verdict; the count is pinned exactly (`used[1].len() == 41`).

**GATE 1 — `tests/search_driver_test.rs`** (5 tests, no node needed): the aux-RNG stream against
four draw tables produced by `replay_kernels.js`'s OWN `auxRngFromSeed` under node (EXACT f64
equality — `u32 / 2^32` is representable, so a tolerance would only hide a divergence) plus
`pickUniform`'s index and its one-draw cost; `random_choice` determinism per seed, non-determinism
across seeds, and legality of every pick against an independently-derived option set; CLONE
INDEPENDENCE (two arms diverge, the parent's turn / chunk stream / outstanding request / PRNG seed
are bit-for-bit unchanged, and re-expanding reproduces an arm byte-for-byte); and the `stuck`
guard's exact iteration count. Every test carries a non-vacuity guard — the fixture must really be
a mid-battle `move` boundary with a live bench, the two arms must really diverge, the wedge must
really have wedged.

> ⚠️ **P6 slice 6c (2026-10-08): the golden's GENERATORS are deleted.** `harness/search_golden.py` and `harness/gen_search_golden.py` drove real gen3ou battles through poke-env `RandomPlayer`s to capture `tmp/search_golden_node.json` (gitignored scratch — no golden is committed, and none existed on a fresh checkout), so `search_impl_parity.py`, `replay_impl_parity.py` and `rust_record_replay_check.py` below can no longer be fed one from this tree; they are kept (REPORTED, not deleted) until the owner decides whether the node-vs-rust search A/B is rebuilt on the Rust env core's recorded battles. The cargo `tests/search_driver_test.rs` never needed the golden.

**GATE 2 — `src/rust_sim/harness/search_impl_parity.py`** (scratch, needs node + a captured record): replays
`tmp/search_golden_node.json` — the NODE `search_driver.js` wire output over 6 decision points
across 2 real gen3ou battles, 54 arms (12 exercising multi-round forced-switch follow-ups) plus
depth-2 expansions — through the built binary over raw stdin/stdout JSON, and diffs every field.
**RESULT: PASS, 18873 leaf fields matched.** It normalizes ONLY `|t:|<epoch>` (the port's one
documented emission exception) and prints its allowlist + coverage notes on every run.
(**Re-measured 2026-08-23 on SEVEN freshly generated goldens: PASS on every one**, 12 cases /
120 arms / ~37.6k leaf fields each — roughly 2× the recorded golden, because the generator now
samples turn 1 as well. The live allowlist takes **0 hits** on all seven.)
FAULT-INJECTION PROVEN: a one-line off-by-one in `pick_uniform` produces 2716 divergences (and the
illegal-move allowlist below correctly REFUSES to reconcile, hits 0 → 1 → 0).

⚠️ **STATUS 2026-08-23 — three corrections, and that PASS is not reproducible as written.**
1. **Both parity harnesses were UN-RUNNABLE.** `ROOT = parents[1]` was correct while they lived in
   `<root>/tmp/`; the move to `src/rust_sim/harness/` (`ede4c79`) left the index behind, so every
   default path resolved under `src/rust_sim/` and did not exist. Same defect in
   `search_golden.py`'s `DRIVER`. All three now use `parents[3]`. A gate nobody can start is
   indistinguishable from a gate that passes.
2. **The golden now samples TURN 1** (`search_golden.py`, `turns = {1, 2, 5, 9}`). It sampled only
   2/5/9, which is exactly why the cross-impl gate never saw `gen3_search_turn1_open_v1`. On a
   turn-1 golden the pre-fix binary reports **157** divergences and the fixed one **1**.
3. ~~On a FRESHLY generated golden both gates currently report pre-existing divergences.~~
   **CLOSED 2026-08-23 (`gen3_fresh_golden_parity_triage_v1`) — BOTH gates are green on freshly
   generated goldens, and the count that reaches this table is now 0.** Do not re-derive a plan
   from the old wording, which guessed at TWO candidates and named neither correctly. The
   divergences were real, pre-existing, and reproduced on the pre-fix binary; SEVEN fresh goldens
   (21 real gen3ou battles, `search_golden.py 3` each) resolved them into **four classes, all of
   them rust BUGS against the node oracle, all now fixed**:

   | class | instances | who was wrong | reach | fix |
   |---|---|---|---|---|
   | Return / Frustration numeric-BP alias missing from the `\|request\|` | 8 fields / golden E | RUST | **the BRIDGE too** — node and the live server both emit it, so `--use-bridge=rust` was the odd transport out (poke-env's `Move.retrieve_id` collapses it, which is why nothing downstream ever noticed) | `gen3_happiness_bp_request_alias_v1` |
   | `pre_state.*.volatiles[len]` — the `substitutebroken` volatile unmodeled | 6 / golden D | RUST | offline drivers only (`pre_state` has no consumer) | `gen3_substitute_broken_volatile_v1` |
   | a SINGLE-ENTRY request (forced Struggle / move lock) silently ACCEPTED `move 2` | 6 / golden G | RUST | **the BRIDGE too** — an `\|error\|` frame and a different committed choice | `gen3_single_entry_request_slot_reject_v1` |
   | `outcome.pN.active_status` — a fire KO thawed the corpse | 1 / golden A | RUST | a STATE divergence in the bridge too, though `0 fnt` hides it on the wire | `gen3_fire_thaw_ko_keeps_status_v1` |

   The **typed-Hidden-Power `|error|` display name** the old text guessed at is a REAL fifth bug
   (`Side.chooseMove` names `dex.moves.get(moveid).name`, i.e. the BARE id — never the request's
   suffixed display), but it did **not** appear in any of the seven goldens: it needs a *disabled*
   typed HP fed as an explicit arm. It is fixed and unit-gated anyway
   (`gen3_reject_message_bare_move_name_v1`), on the training-path argument — poke-env sees
   `|error|` frames. **The "29" and the "1" were never stable counts**; a golden is three random
   battles, and the per-golden divergence count ran **1 / 0 / 0 / 6 / 8 / 0 / 6** across A-G — three
   of the seven would have read as a green gate, and the Struggle class turned up only on the
   SEVENTH. **Run each gate on at least two fresh seeds before calling it green.**

**THE CHOICE-REJECT DIVERGENCE IS CLOSED — the only live allowlist entry is `.error` TEXT**
(Node returns a JS `e.stack`, the port a plain message; the ok/`ok:false` VERDICT is still compared
strictly). When an arm feeds an explicit move the request marks `disabled` (the golden hits a
Choice-locked Aerodactyl), Node emits `|error|[Unavailable choice] Can't move: X's Y is disabled` +
a re-request **to that side only**, whose disabled slot gains `"disabledSource":""`
(`Side.chooseMove`'s `updateRequestForPokemon`, which is also what makes the error `[Unavailable]`
rather than `[Invalid]`) — and `bridge.rs` now emits exactly that
(`gen3_choice_reject_framing_v1`: `serialize_active_with_disabled_source`, the
`[Invalid]`/`[Unavailable]` split on whether there is a request to re-issue, gated by
`tests/bridge_choice_reject_test.rs`).

⚠️ **The old entry here claimed the port "emits no `|error|` … on a path poke-env never takes".
BOTH halves were false, and it is retracted — do not re-derive a plan from it.** The framing half
was closed by `gen3_choice_reject_framing_v1` and *this note survived its own fix*. The "never
takes" half was falsified by poke-env taking it and killing **two production launches at ~8
minutes** — root-caused as `gen3_locked_choice_never_rejected_v1` (a move-LOCKED mon gets a
single-entry `trapped:true` request, and `classify_reject` never consulted `move_locked()`, so the
port could REFUSE the only move offered — it contradicted ITSELF; fixed with a `move_locked()`
early-out beside `must_struggle`, gated by
`bridge_choice_reject_test::a_move_locked_mon_is_never_rejected_for_its_only_offered_move`). The
existing fuzz could not catch it by construction: it drives masked-LEGAL tokens, and there the
token IS masked-legal because the mask is built FROM the request. **The durable lesson is that an
allowlist entry can outlive its own fix and then mislead every reader after** — see the root
`CLAUDE.md`. Verify an allowlist claim against the harness (`src/rust_sim/harness/search_impl_parity.py::ALLOWLIST`)
and the code, never against this prose alone.

**HONEST SCOPE — `pre_state`.** It mirrors Node's `preState` and has NO consumer today. `turn`,
`weather`, the per-mon species/hp/maxhp/status/fainted/position, boosts, item, ability, the
bare-`hiddenpower` moveSlot ids + PP, and the three gen-3 side conditions are faithful reads.
Two fields are RECONSTRUCTIONS: `pseudoWeather` is derived from `BattleState::sleep_clause` (the
port has no pseudo-weather map; gen-3 has no pseudo-weather MOVE, so the clause rules are the only
real entries — VERIFIED, 6/6 golden cases), and `volatiles` is rebuilt from the port's typed
volatile fields. The golden gives that mapping exactly ONE piece of coverage, and it was a real
find: the port's **`duration: 1` single-turn fields** (`focuspunch`, `pursuit`, `protect`,
`flinch`, `beatup`, the Counter/Mirror-Coat record, `endure`, `snatch`) are cleared at the NEXT
turn's TOP (`turn::helpers::clear_flinch`) where Showdown removes them at the residual, so they
are STALE-SET at a move-request boundary — reporting `focuspunch` diverged 2 of the golden's 12
`pre_state`s, and the group is excluded.

**Exactly ONE volatile NAME is positively verified: `substitutebroken`** — `gen3_substitute_broken_volatile_v1`,
and it took a FRESHLY generated `replay_impl_parity` golden to find it, because the recorded
golden's twelve `pre_state`s are all EMPTY. gen3 inherits gen4's Substitute, which pairs the
removal with `addVolatile('substitutebroken')`; the condition has no duration and no gen3 reader,
so it sits on the mon until `clearVolatile`. It is mechanically INERT in gen 3 and is modeled
purely so this readout can name it. **Every other name (`choicelock`, `perishsong`, `twoturnmove`,
…) is still UNVERIFIED** — do not read a green parity run as evidence that one is spelled or timed
right; `replay_impl_parity` prints a `pre_state:nonempty-volatiles` count on every run precisely so
an all-empty record set cannot be mistaken for coverage.

**The port-only one-sided `view_p1` / `view_p2` (and D10's `view_pN_at`) payloads are DELETED**
(Rust Core deletion pass, program §4 M2); a CORE root's arms carry `core_pN` instead (the leaf
version's reading, or with `rows` its encoded row — [`present.md`](present.md) §4). History:
[`one_sided_view.md`](one_sided_view.md).

### THE REPLY IS SHIPPED, NOT JUST RENDERED — `side` ELIDES THE HALF NOBODY READS

`gen3_expand_many_side_elision_v1`. An `expand_many` request may carry a top-level
**`side`: `"p1"` | `"p2"`**, and the driver then OMITS the other side's one-sided payload
(`pN_chunks`, and `core_pN` on a core arm) from every arm of that batch. A search runs for one side
and reads nothing of the other — so on 864 banked arms (measured when the view payload still
shipped) the discarded copy was **43.0% of the reply bytes**, rendered, quoted, piped
and `json.loads`-ed to be dropped (the split, and the three candidates it killed:
[`../research_state/measurements/expand_many_2026-09-22/README.md`](../research_state/measurements/expand_many_2026-09-22/README.md)).
`requests` is NOT in that class and is never elided: BOTH sides are read (the opponent's legal
choices at interior plies, `branchable` for ours).

| | |
|---|---|
| **default** | `SideWant::Both` — **byte-identical to the historical body**, same fields in the same ORDER. A request without `side` cannot tell the feature exists, which is what keeps `search_impl_parity`'s node comparison (whose golden requests carry no such key) valid field-for-field |
| **an unknown `side`** | an ERROR, never a fall-back to Both. A typo that returned everything would read as a working elision that saved nothing, and the only symptom would be a benchmark that refused to move |
| **the elided slot, in Python** | `search_session.ElidedSide` — falsy, so the existing `payload or {}` guards take their COUNTED fallback, but RAISING on every way of reading a value out. An empty dict would have ENCODED: into a well-formed observation of a battle nobody played |
| **under `impl="node"`** | nothing is elided. `search_driver.js` ignores the key and returns both sides, and the sentinel keys on the field being ABSENT, never on the request |
| **wall gate** | `tests/search_side_elision_test.rs` — the same arm from the same root in the same process, with and without `side`, asserting the surviving payload is **byte-identical**, the other is gone, a `side`-less arm keeps the historical field order, and an unknown side is refused without killing the session |
| **decision gate** | `src/main/search_dividend/side_elision_parity_integration_test.py` (deleted with the battery, P6 slice 6d-1; the sentinel's contract now lives in `utils/bridge/search_session_test.py`) — the real `SearchEngine` over one seeded decision, elision on vs off, comparing every successor's obs BYTES in order, the scores, the chosen action, the widths **and both fallback counters** (an elided payload read as empty would quietly move arms to the protocol road and still score plausibly) |

🚨 **The driver can also report its OWN per-phase wall** — `POKESIM_SEARCH_TIMING=1` adds a
`timing_us` object (`sim` / `view` / `chunks` / `render` / `total`, micros summed over the batch)
to an `expand_many` reply. It is read ONCE per process and renders **the empty string** when off,
so an un-set build is byte-identical; that is what lets the cross-impl parity harness go on
comparing this driver to node's, which has no such field. `pokesim::driver_timing`.

**IT ALREADY REPLACES node in `better_line`.** This paragraph used to list three things as STILL
NEEDED and every one of them had already been done — *a note that outlived its own fix*, the exact
class this file polices elsewhere (corrected 2026-09-07, verified against the code named here):

- `utils/bridge/search_session.py` **has** the impl switch — `SearchSession(..., impl="node")`
  selects the child through `search_driver_spawn_argv(impl)`, and `"rust"` spawns this binary.
- `search_clone_parity_fuzz_test` **took** `--impl rust`, documented in its own header (deleted in P6 slice 6d-1), plus an
  independent `--record-impl` so a record made by one engine is replayed on the other.
- The search-teacher `input_log` blocker was **gone, and its stated reason was FALSE** — nothing reads
  the record's committed-choice lines; the only readers (`replay_kernels.js::writeStart`,
  `ReconstructionRecord.start_options()` / `.players()`) touch the `>start` / `>player` lines, which the
  rust record renders exactly. The search-teacher consumer itself (and its `gen3_search_teacher_composition_rust_v1`
  composition gate, which ran >= 2 teacher cycles on `--use-bridge rust`) is DELETED (deletion pass L3,
  2026-10-02; [`../training/search_teacher.md`](../training/search_teacher.md)). The remaining legs of
  that composition — better_line node≡rust candidate values bit-identical · `search_clone_parity` · the
  counterfactual confirm leg — are untouched and still gated.
  Porting a teacher onto this driver is a deferred build, to be scheduled with X15.

### THE CORE ROAD — a tree of `BattleVersion`s (`gen3_core_search_v1`, the Rust Core Program's M2)

`open_root` takes an optional **`core: "typed" | "text"`** and **`side: "p1" | "p2"`**. With
`core`, the root is a `BattleVersion` built over a core-recording session
(`session_from_record_core`), folding only `side`'s stream, and every node is
`NodeState::Core(version, path)`; without it, the node is the historical engine session and the
reply is byte-for-byte the old one. On a core node, `expand_many` steps a child VERSION per arm and
returns, per wanted side, **`core_pN = {view, legal, request, events, mid, text_view?}`** — the
version's `present()` view, `legal_actions()`, the raw `|request|`, the READINGS of the ply's
events, and `mid`: the same four AT each intermediate decision the ply resolved itself (D10), built
from the engine snapshots `resolve_turn_capturing(Capture { sessions: true })` takes
there. The child node of a D10 arm IS that intermediate version (a forced replacement is
non-branchable on this road). `recorded_exact` is REFUSED on a core node.

`POKESIM_SEARCH_TIMING=1` adds `core` (the fold from text) and `core_render` (the leaf JSON) to
`timing_us` (the typed shortcut and its `integrity` twin are deleted, program §4 M4).

**Capture is explicit.** `resolve_turn` / `resolve_turn_sourced` capture NOTHING (`Capture::NONE`);
the core road asks for `Capture { sessions: true }`, and a plain root's arms and the replay verbs
for none. Contract of the version and the reading:
[`present.md`](present.md).

### THE IN-PROCESS TWIN — the core road without the child (M5 Lane I)

`src/rust_env/src/search/tree.rs` serves this driver's CORE road (`open_root` with `core: "text"` +
trackers, `expand_many` with `rows`) IN PROCESS over the env core's FFI
(`utils/rust_env/successors.py::Successors`, a drop-in for `SearchSession` on that road;
`SearchConfig(search_impl="inproc")`). It calls the same kernels statement for statement, so its
replies are BYTE-EQUAL to this binary's — root fields, node ids, every arm field and every leaf row
to depth 3 (`src/utils/rust_env/successors_parity.py`, routine COMMIT + `slow` MILESTONE). A change
to the core road here must land in `tree.rs` too, or that gate fails. Contract, gates and numbers:
`designs/endstate/program_rust_core.md` §2 M5 (Lane I).

🚨 **A known kernel defect both roads share (F-LI-1, not fixed):** `resolve_turn_sourced_with` feeds
the second side inside the same loop iteration after the first side's feed, without re-checking the
boundary — when side 1's REPLACEMENT switch ends the turn, side 2's fresh next-turn move request is
answered by the FOLLOW-UP policy inside this arm. The child node's engine then disagrees with its
own leaf, and a deeper arm fed the leaf's own token fails the whole `expand_many`
(`bridge fatal: unresolvable choice`). Node's `resolveTurn` has the same shape (this port is
faithful), so a fix moves both drivers' bytes. Repro and counts:
`designs/research_state/measurements/m5_laneI/PROGRESS.md` F-LI-1.

## The REPLAY family: the one-shot `replay` / `reroll` / `reroll_many` verbs (`gen3_rust_replay_driver_v1`)

The SECOND half of the offline layer, served by the SAME `src/bin/search_driver.rs` binary. It
closes the gap the search half opened: `utils/bridge/reconstruction.py::_run_driver` routes BOTH
verb families through `sim_bridge_bin.search_driver_spawn_argv`, so under `impl="rust"` a
`replay_battle` / `reroll_turn` / `reroll_many` call reached a binary that answered
`unknown cmd` — which broke `better_line` (it calls `replay_battle`), `lookahead` and `falsify`
even though the search half worked.

**PROTOCOL.** `{mode: "replay"|"reroll"|"reroll_many", …}` on stdin → ONE bare JSON object on
stdout with **no trailing newline** → exit **0**; on failure `{"error": …}` → exit **1**. That is
`replay_driver.js`'s whole lifecycle, and the EXIT CODE is the load-bearing half: `_run_driver`
branches on it BEFORE it parses stdout, so the message is informational. Dispatch is on the `mode`
KEY; a `cmd` request still runs the persistent search loop, in the same process, unchanged.

**KERNEL REUSE, not a second copy.** The only kernel the search half never needed is
`recorded_queues` — the replay family's `"recorded"` action source is a QUEUE, not the search
path's single-shot latch, because a live turn's command stream can contain a REFUSED choice
followed by its correction (the maybe-trapped probe). Replaying that faithfully under a fresh seed
requires pulling the NEXT recorded command on a refusal; a single-shot source would fall to the
follow-up POLICY and invent a pick the original battle never made. `resolve_turn` was refactored
onto a shared `resolve_turn_sourced` over a `TurnSource` enum (`Silent` / `Random` / `Explicit` /
`Queue`) so both families run ONE resolution loop — the guard arithmetic and the routing invariant
exist once. `resolve_turn`'s own behaviour is byte-identical (`ActionSpec::Recorded` lowers to
`Silent`).

**`turn_log` — the one field the search half does not produce.** It is defined as Node's
`b.log.slice(logStart)`, and Showdown's `battle.log` is NOT the port's line list: an `addSplit`
effect occupies THREE entries there (`|split|pN`, the SECRET line, the SHARED line), and EVERY
HP-bearing line is one, because `Pokemon.getHealth()` returns a `{side, secret, shared}` object
for the `Battle.add` split path — including `0 fnt` and including a `reportExactHP` format, where
`shared` merely equals `secret`. `bridge.rs::split_log_lines` reconstructs the triples from the
port's single omniscient line using EXACTLY the predicate set the production per-side fold
(`derive_side`) uses, so the two can never disagree about which lines are split. The owner-only
pair (`|-ability|…|[silent]`, the Intimidate `|-hint|`) gets an EMPTY shared entry, matching
`addSplit(side, secret)` with no `shared`. VERIFIED byte-for-byte on 133 arms.

**ONE REAL PORT BUG the widened coverage caught: the `fnt` status token.** `Battle.checkFainted`
writes `status = 'fnt'` for a FAINTED ACTIVE mon; `BattleActions.switchIn` clears it again when
the corpse is swapped off (`if (oldActive.fainted) oldActive.status = ''`); and `runAction` runs
`faintMessages(); if (this.ended) return true;` BEFORE `checkFainted()`, so the DECIDING faint
never gets it. The port models the state effect as `status = None` (that clear is what stops
paralysis ×0.25 applying to the replacement sort — `gen3_fnt_clears_status_v1`), so the token has
to be re-derived at render time: `search.rs::slot_status_token` shows `fnt` only for a fainted mon
in the ACTIVE slot of a battle that has NOT ended. The search golden never reaches this (its arms
all end at a live move boundary and it never reports a finished battle), which is why it took
`replay` + re-rolls of LATE turns to surface. All three branches are node-verified.

**GATE 1 — `tests/replay_driver_test.rs`** (12 tests, no node): the one-shot dispatch (bare object,
no trailing newline, exit 0 / exit 1 with `{"error"}`, unknown mode, invalid turn); that the
PERSISTENT protocol is untouched and the process stays alive after a `cmd` request (the regression
the dispatch could plausibly cause); the `recorded_queues` REFUSAL-PULL together with its contrast
case (a single-shot source must NOT re-send); and `recorded_queues`' own per-side split /
`forcelose` stop / cap. Its record is built in-test by playing the fixture board to game-end
through `BridgeSession::new_construct_turn0` — the same constructor `session_from_record` uses, so
the recorded stream replays bit-for-bit.

**Four of the twelve are `gen3_search_turn1_open_v1`** — the FIRST decision of a battle. They pin
the predicate (`at_turn_start` true for `t == 1` at a pre-commit boundary, false for every other
`t`, and unchanged at `t >= 2`), that `build_to_turn(…, 1)` applies NO commands, and that turn 1
opens through the real binary on BOTH verb families (one-shot `reroll` and persistent `open_root`).
Before the fix, `at_turn_start` compared `BattleState::turn` — still `0` at that boundary, because
the driver increments at `commitChoices` for turn 1 and EAGERLY at every later turn end — so
`build_to_turn` walked the whole command log and reported "battle never reached the start of turn
1". Node opened it fine, making this a silent impl-specific hole over ~3.35% of move decisions.

**GATE 2 — `src/rust_sim/harness/replay_impl_parity.py`** (scratch, needs node + `tmp/search_golden_node.json`'s
records): drives `node replay_driver.js` and the rust binary LIVE on the identical request and
diffs every field. Live rather than golden-captured on purpose — the requests are cheap, so a
stored golden would only add staleness. **RESULT: PASS — 76 cases (54 success + 22 error-path),
136 arms, 30689 leaf fields.** It normalizes ONLY `|t:|<epoch>` and prints its allowlist, its hit
counts and a COVERAGE table on every run.
(**Re-measured 2026-08-23 on SEVEN freshly generated record sets: PASS on every one**, 111 cases /
~204 arms / ~45-47k leaf fields each. The only allowlist arm — error message TEXT — takes 30 hits
per run and the verdict, exit code and error CLASS stay strict. Coverage varies by record set and
is REPORTED, never assumed: `pre_state:nonempty-volatiles` was 0 on three of the seven and 26 on
another, and `arm:stuck` / the Intimidate `|-hint|` still miss on most.)

**COVERAGE the search golden was missing, now exercised** (reported by the harness itself):
2 arms that END the battle, 1 `stuck` arm, 133 arms whose `turn_log` carries `|split|` triples,
17 with the owner-only empty-shared form, and 9 distinct error classes across BOTH families
(`invalid turn` ×8, `battle never reached the start of turn N` ×3, `replay`'s not-ended ×4,
`unknown mode` ×2, a seedless arm ×2, `unknown cmd`, `unknown node`, `recorded_exact on a non-root
node`). The `stuck` arm needs a deliberately TRUNCATED record — a well-formed record's turns are
always complete, so `resolveTurnExact` can never run out mid-turn on one; the harness truncates a
copy, which also drives `replay`'s not-ended path. STILL NOT EXERCISED and printed as such: a
NON-EMPTY `pre_state` volatile list, and an Intimidate `|-hint|` inside a re-rolled turn (the one
`split_log_lines` case whose owner attribution is a heuristic).

**THE CHOICE-REJECT DIVERGENCE IS CLOSED here too — the only live allowlist entry is `.error`
TEXT** (the VERDICT, the process EXIT CODE that `reconstruction.py` actually branches on, and the
error CLASS are all compared STRICTLY). An arm that feeds a choice the sim refuses — here
`switch N` into a FAINTED slot — used to diverge in the framing (`|error|[Invalid choice]` to the
offending side only vs no `|error|` and a re-opened boundary to BOTH sides) and, downstream of
that, in `choices_used`. `bridge.rs` now models every reject class, not just the trapped-SWITCH one
(`gen3_choice_reject_framing_v1`), so neither entry remains in
`src/rust_sim/harness/replay_impl_parity.py::ALLOWLIST`.

⚠️ **Retracted, same as the search half's note — do not re-derive a plan from the old text.** It
described the port as "modelling only the trapped-SWITCH reject" on a path poke-env never takes;
that path killed two production launches at ~8 minutes
(`gen3_locked_choice_never_rejected_v1`). Read the search half's note above for the root cause and
the lesson, and verify against the harness + the code rather than this prose.



---

## Why the clone-branch snapshot needs no byte format

stream, boundary progress) is **plain owned data** — no `Rc`/`RefCell`/`Box<dyn>`/closures/raw
pointers/lifetimes — so `#[derive(Clone)]` already gives a DEEP, independent battle. The
snapshot never crosses a process boundary (neither does Node's), so there is nothing to encode,
nothing to keep in sync, and no re-parse cost. The old `Battle::serialize`/`deserialize` +
`BattleSnapshot` stubs are DELETED, not implemented.

PRNG continuity is automatic (the whole dice state is `Prng`'s backend seed), and the `&Dex` is
threaded per method rather than owned, so a snapshot copies a couple of teams' worth of state
rather than ~16 MB. The one thing a clone shares with its parent is the process-global
`POKESIM_PRNG_TRACE` draw counter — diagnostics, not battle state, so branches interleave their
trace INDICES and nothing else.

The public search API on `BridgeSession` (all reads or copies — nothing here advances the
battle or draws a number, so it cannot perturb the production bridge):

---

## Moved from the leaf (2026-10-10)

> Moved VERBATIM from `src/rust_sim/CLAUDE.md` in the 2026-10-10 leaf cleanup (links re-based;
> statements found FALSE against the code were corrected in place, each saying so). The leaf keeps a
> one-line pointer here. Frozen original: `designs/research_state/claude_md_archive/src_rust_sim_CLAUDE_2026-10-10.md`.

### The callable surface (battle.rs) maps to the existing bridge

`battle.rs` deliberately mirrors the SIX ways `src/utils/bridge/` already drives
Showdown, so a finished core is a drop-in:

| Bridge today (Node)                     | Rust surface |
|---|---|
| streaming battle (`local_sim_bridge.js`) | `BattleStream::write_line` |
| websocket server for an OUTSIDE client (`ws_frontend.py`) | the SAME `src/bin/sim_bridge.rs` — one child per battle. The front end adds only the `>battle-…` room framing and the `rqid` the SERVER injects (`server/room-battle.ts:796`), never an engine line; byte-gated against `local_sim_bridge.js` by `ws_frontend_replay.py` (40 battles / 44,034 lines, 2026-09-14). Surface + deferrals: [`designs/rust_sim/ws_frontend.md`](ws_frontend.md) |
| mid-battle RNG swap (counterfactual/search) | `Battle::reseed` |
| clone-and-branch (`State.serialize…`)    | `BridgeSession::snapshot` (a derived `Clone`; PRNG continuity is automatic) |
| search server (`search_driver.js`)       | `src/bin/search_driver.rs` over `src/search.rs` (`gen3_rust_search_driver_v1`) |
| offline replay / re-roll (`replay_driver.js`) | the SAME `src/bin/search_driver.rs`, one-shot `mode` verbs (`gen3_rust_replay_driver_v1`) — node splits the two families across two scripts, the port serves both from one binary, which is what `sim_bridge_bin.py::search_driver_spawn_argv` assumes |
| damage oracle (`damage_probe.js`)        | `Battle::new` + state accessors (TODO) |
| team pack / validate                     | out of core scope — keep as a thin shim |

When you implement the engine, keep these signatures stable; the bridge contract
is the spec.

### Clone-and-branch: the SNAPSHOT primitive (`gen3_bridge_clone_branch_v1`)

The last surface the Rust bridge was missing for the prober's SEARCH path
(`better-line`'s CRN-anchored beam, which branches a paused mid-battle state by CLONING it).
The Node search server does this with `State.serializeBattle`/`deserializeBattle`; the port's
answer is **`BridgeSession::snapshot()` — a derived `Clone`**, and the reason a byte format is
NOT needed is structural: Showdown needs one because its battle graph is full of cyclic object
references, whereas everything under `BridgeSession` (battle state, PRNG, driver phase, chunk

**PRNG continuity is automatic** (the whole dice state is `Prng`'s backend seed), and the `&Dex` is
threaded per method rather than owned, so a snapshot copies a couple of teams' worth of state rather
than ~16 MB. Why a byte format is not needed — and why `Battle::serialize`/`deserialize` are DELETED
rather than implemented — is in
[`designs/rust_sim/search_and_replay_drivers.md`](search_and_replay_drivers.md).

| method | contract |
|---|---|
| `snapshot() -> BridgeSession` | the deep clone (engine AND transport); shares NOTHING with the parent |
| `engine() -> &Engine` / `resume(engine)` | the ENGINE half alone (`gen3_core_engine_split_v1`) / a FRESH transport around an engine clone — no chunks, no `script`, no seed anchors, the outstanding requests' issued bytes shared with the engine (rendered once, at issue). What a `BattleVersion` fork uses, so a fork never copies the wire's history |
| `clear_chunks()` | reset the chunk stream so a branch's `chunks()` is only ITS suffix. Deliberately does NOT touch `prev_log_len` — that is a cursor into the BATTLE LOG, and resetting it would re-emit the whole battle into the next chunk |
| `request_kind(side) -> Option<RequestState>` | Showdown's `side.requestState`. `Move` / `Switch` (a forced replacement) / `Wait`; `None` when no boundary is open |
| `is_choice_done(side) -> bool` | `side.isChoiceDone()`. True also for a `Wait` side and when no boundary is open; FALSE across a REJECT (a reject re-issues the request and holds the boundary open) |
| `active_request_json(side) -> Option<&str>` | `side.activeRequest` — the EXACT bytes that went on the wire, stored (not rebuilt) at every emit site, INCLUDING the `trapped:true`+`update:true` re-request a hidden-trap reject re-issues. Cleared when the boundary closes / the battle ends |
| `battle_state() -> Option<&BattleState>` | the omniscient referee readout (HP/status/boosts/field/seed) |
| `winner() -> Option<usize>` | the winner once ended, from BOTH end paths (the driver's win check and `forfeit`, which ends via the protocol so the driver's phase never becomes `Ended`). `None` = still playing OR a gen-3 TIE — pair with `is_ended()` |

**Gate: `tests/bridge_clone_branch_test.rs`** (7 tests on a real seeded gen3 battle, every one
carrying a non-vacuity guard — without those, a snapshot of a FINISHED battle would pass while
proving nothing). Fault-injection proven.

**Not built here:** `Battle::{new,choose,ended,winner,seed,reseed}`, which stay `todo!()` because
every production path drives the engine through `FullBattleDriver` instead. The SECOND half — the
driver that consumes this API — is below.

### The offline drivers: the SEARCH server + the REPLAY family

`src/bin/search_driver.rs` over `src/search.rs` (`gen3_rust_search_driver_v1` +
`gen3_rust_replay_driver_v1`) is the byte-compatible drop-in for **both** node offline drivers, so
`utils/bridge/search_session.py` (the prober's `better_line` beam and the search-dividend probe) and
`utils/bridge/reconstruction.py` (`better_line` / `lookahead` / `falsify`) swap `node` for the binary
with ZERO protocol change. **Dispatch is on the KEY**: a request carrying `mode` is the one-shot
REPLAY family (`replay` / `reroll` / `reroll_many`) — a BARE JSON object on stdout, **no trailing
newline**, exit **0**, or `{"error"}` + exit **1**, which is what `_run_driver` branches on before it
parses stdout; anything else runs the persistent `{id, cmd}` search loop (`open_root` /
`expand_many` / `close`).

**It can replace node in `better_line`** — `search_session.py` has the `impl` switch (verified 2026-10-10: `impl="node"` is still the DEFAULT in `search_session.py`, `reconstruction.py` and `main/prober/better_line.py`; the binary is the opt-in `impl="rust"`) and
`search_clone_parity_fuzz_test` took `--impl rust` (deleted in P6 slice 6d-1 with the poke-env materializer it compared against). (The trainer-side composition that once ran the
search teacher on it was deleted with the search teacher, deletion pass L3.) The node driver (`impl="node"`) remains the DEFAULT.

| gate | needs node? | what it proves |
|---|---|---|
| `tests/search_driver_test.rs` (5 tests) | no | the aux-RNG stream vs node draw tables (EXACT f64 — a tolerance would only hide a divergence), clone independence, the `stuck` guard's exact 41 iterations |
| `tests/replay_driver_test.rs` (12 tests) | no | the one-shot dispatch + exit codes, the persistent protocol untouched, the `recorded_queues` refusal-pull, turn-1 opening on both verb families |
| `src/rust_sim/harness/search_impl_parity.py` | yes + a captured golden | the node `search_driver.js` wire output diffed field-by-field. ⚠️ Its golden's generators (`search_golden.py`, `gen_search_golden.py`) were poke-env drivers and were DELETED in P6 slice 6c; the golden (`tmp/search_golden_node.json`) is gitignored scratch and none is committed, so this harness has no way to get one |
| `src/rust_sim/harness/replay_impl_parity.py` | yes | `node replay_driver.js` vs the binary LIVE on identical requests |

🚨 **RUN EACH PARITY GATE ON AT LEAST TWO FRESH SEEDS BEFORE CALLING IT GREEN.** A golden is three
random battles; across seven freshly generated ones the per-golden divergence count ran
1 / 0 / 0 / 6 / 8 / 0 / 6 — three of the seven would have read as a green gate, and one bug class
turned up only on the SEVENTH.

🚨 **AN ALLOWLIST ENTRY CAN OUTLIVE ITS OWN FIX AND THEN MISLEAD EVERY READER AFTER.** Two entries
here described the port as emitting no `|error|` "on a path poke-env never takes"; both halves were
false, and that path killed **two production launches at ~8 minutes**
(`gen3_locked_choice_never_rejected_v1`). Verify an allowlist claim against
`src/rust_sim/harness/search_impl_parity.py`'s `ALLOWLIST` and the code, never against prose. The
only live entry in either harness is `.error` message TEXT; the verdict, the exit code and the error
CLASS stay strict.

### The ONE-SIDED VIEW (`view.rs`) — DELETED

The port's one-sided PROJECTION of the omniscient board (`view.rs::one_sided_view`, its per-side
reveal fold in `BridgeChunks`, `enable_view_fold`, the `view_pN` / `view_pN_at` payloads of
`search_driver` and `core_events --views`' `views` / `truth`) and its Python consumers are
DELETED (Rust Core deletion pass, program §4 M2): every successor the search scores is a Rust-core
version read from its own stream (`present()`), and the engine board audit holds that reading to the omniscient
board at every decision (the comparison with the training `LiveView` — slice V — was deleted in P6 slice 6c). 🚨 **Do NOT feed `search::volatile_names` to the obs layer** — that
set is the port's TYPED fields and includes conditions the sim never announces (gen-3 Choice lock
raised `UnknownVolatileError` on the first real board); the reading folds the announcing lines.
History, the reading-rule table (V1–V13) and the findings the projection surfaced:
[`designs/rust_sim/one_sided_view.md`](one_sided_view.md).

🚨 **`pre_state` VOLATILE NAMES ARE UNVERIFIED.** `pre_state` mirrors Node's `preState` and has no
consumer today; its `volatiles` list is a RECONSTRUCTION from the port's typed fields, and exactly ONE
name is positively verified (`substitutebroken`). Do not read a green parity run as evidence that
`choicelock` / `perishsong` / `twoturnmove` / … is spelled or timed right — `replay_impl_parity`
prints a `pre_state:nonempty-volatiles` count precisely so an all-empty record set cannot be mistaken
for coverage.

🚨 **A gen3ou repro MUST be replayed with `{format:'gen3ou', allowHiddenPower:true}`** — the probe
default is customgame, and the sim then diverges from the golden.

🚨 **`expand_many` SHIPS TWICE WHAT A SEARCH READS, UNLESS THE REQUEST SAYS OTHERWISE.** The
driver renders both sides' one-sided payload (`p1_chunks`+`p2_chunks`, and `core_p1`+`core_p2` on a
core arm); a search reads ONE side. Measured on 864 banked arms the discarded half was **43.0% of the reply bytes**, and
`expand_many`'s optional top-level **`side`** (`"p1"`/`"p2"`, `gen3_expand_many_side_elision_v1`)
omits it — **30,326 → 17,330 B/arm**, with the surviving side byte-identical and an omitted `side`
rendering the historical body byte-for-byte. Python's elided slot is a sentinel that is falsy but
RAISES on read, never an empty dict (an empty dict ENCODES). Gates:
`tests/search_side_elision_test.rs` (the byte diff, both ways, in one process) and
`src/utils/bridge/search_session_test.py` (the sentinel's contract; it moved there from `main/search_dividend/side_elision_test.py`, P6 slice 6d-1). Measured (on the since-deleted view / protocol roads), interleaved, load-matched, one road per
process: **1.10x / 1.11x at wide B**, not resolved at B = 1. 🚨 **Two candidates that LOOKED certain were rejected on measurement**: a
compact fixed-order payload (0.024 ms/arm of parse, all of it given back re-keying for the
existing adapter) and holding Python's cyclic collector off for the reply parse (**2.3x on a
captured 454 KB reply, NOTHING end to end** — built, gated and reverted). **A micro-benchmark share
is not a wall share.** The split, both instrument findings, and the four candidates the measurement
killed with their numbers:
[`designs/research_state/measurements/expand_many_2026-09-22/README.md`](../research_state/measurements/expand_many_2026-09-22/README.md).

Full detail — the protocol, the structural simplification over Node's serialize/restart/baseline
dance, the kernel reuse, the `turn_log` split-line reconstruction, and the honest scope of
`pre_state`:
[`designs/rust_sim/search_and_replay_drivers.md`](search_and_replay_drivers.md).
