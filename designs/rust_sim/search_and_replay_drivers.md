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
`better_line` beam, and the search teacher) can swap `node search_driver.js` for this binary
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

**IT ALREADY REPLACES node in `better_line`.** This paragraph used to list three things as STILL
NEEDED and every one of them had already been done — *a note that outlived its own fix*, the exact
class this file polices elsewhere (corrected 2026-09-07, verified against the code named here):

- `utils/bridge/search_session.py` **has** the impl switch — `SearchSession(..., impl="node")`
  selects the child through `search_driver_spawn_argv(impl)`, and `"rust"` spawns this binary.
- `search_clone_parity_fuzz_test` **takes** `--impl rust`, documented in its own header, plus an
  independent `--record-impl` so a record made by one engine is replayed on the other.
- The search TEACHER's `input_log` blocker is **gone, and its stated reason was FALSE.** Nothing
  reads the record's committed-choice lines; the only readers (`replay_kernels.js::writeStart`,
  `ReconstructionRecord.start_options()` / `.players()`) touch the `>start` / `>player` lines,
  which the rust record renders exactly. `main/train/config.py` records that finding where the
  guard used to be and threads `SearchTeacherCallback(impl=args.bridge_impl)` instead, so a rust
  run's teacher no longer silently falls back to node.

**The COMPOSITION was the last ungated rung, and it is gated now** (2026-09-07,
`gen3_search_teacher_composition_rust_v1`). Every leg already ran on rust — better_line node≡rust
candidate values bit-identical · `search_clone_parity` · the counterfactual confirm leg — and
`src/main/train/search_teacher_composition_test.py` (`sim` + `slow`, ~11 min) now adds the rung above
them: the real `train_rl_agent.py` at smoke scale on `--use-bridge rust`, asserted to run **>= 2
search-teacher cycles** whose workers carry `"impl": "rust"`, whose shards come back without a
`worker_no_shard` or `error:*` status, and whose corrections reach the AWR fold (`teacher/loss` +
`grad/searchteacher_share` in the TB events). **It found no seam on the first run.**
`--use-bridge=node` remains the fallback if a cycle misbehaves.

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

