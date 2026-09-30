# M5 Lane I — search on `successors()` in process: PROGRESS (resume point)

Lane I of `designs/endstate/program_rust_core.md` §2 M5. Owns `src/rust_env/src/search.rs` (+ its
submodules `src/rust_env/src/search/{tree,game,playout,ffi_imp}.rs`), `src/utils/rust_env/successors.py`,
and their tests / tools: `src/rust_env/tests/search_game_test.rs`,
`src/utils/rust_env/successors_parity.py`, `successors_integration_test.py`,
`successors_benchmark.py`. Hand-offs (one line or a few, each marked `M5 Lane I` in place):
`src/rust_env/src/lib.rs` (`pub mod search;`); `src/rust_env/src/ffi.rs` (four helpers made
`pub(crate)`, an `FfiRet` impl for the search handle, `pub use crate::search::ffi_imp::*` in `imp`,
the regenerated region); `src/utils/rust_env/ffi.py` (five ABI types + eight `FUNCTIONS` rows);
`src/utils/rust_env/core_cargo_test.py` (two test names in the non-vacuity list);
`src/main/search_dividend/search.py` (`search_impl="inproc"`: the session factory and an ndarray row
in `_materialize_core`); `src/main/search_dividend/__main__.py` (`--search-impl inproc`).

## How to build / test (worktree-local target only)

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m utils.rust_env.ffi --write                    # after any change to FUNCTIONS (ffi.py)
(cd src/rust_env && CARGO_TARGET_DIR=$PWD/target cargo test --profile selfcheck --features emission-selfcheck --test search_game_test)
python3 -m pytest src/utils/rust_env/successors_integration_test.py -q            # COMMIT (routine), ~15 s warm
python3 -m pytest src/utils/rust_env/successors_integration_test.py -q -m slow    # MILESTONE, ~40 s warm
python -m utils.rust_env.successors_parity --battles 12 --source ladder_milestone # a fresh-seed sweep
python3 src/utils/rust_env/successors_benchmark.py                                # unit 4, a descriptor (release)
```

## Units

| unit | what | status |
|---|---|---|
| 1 | the successor API in the core, through the FFI: the search TREE (`open_root` / `expand_many` with rows, `search_driver`'s core road, in process) and the branchable training-path `Game` | **BUILT** |
| 2 | gates: the depth-3 successor slice byte-equal to `search_driver` + the three search gates, COMMIT routine + MILESTONE slow | **BUILT, all PASS** |
| 3 | play out to the end: branch every legal action × seeds (CRN), a caller-supplied batched policy | **BUILT** |
| 4 | throughput descriptor vs the JSON road | **MEASURED** |

## The API

**The search tree** — `Successors` is a drop-in for `SearchSession` on the core road (same `RootView` /
`ExpandedNode`, same node ids `n0, n1, …`); `core_pN["row"]` is a read-only float32 `(OBS_DIM,)` array
instead of a base64 frame. `SearchEngine(cfg=SearchConfig(search_impl="inproc"))` uses it.

```python
from utils.rust_env.successors import Successors
with Successors() as ss:                         # loads THIS checkout's selfcheck cdylib, stamp-checked
    root = ss.open_root(turn, record=rec, side="p1")                 # core="text", trackers=True
    kids = ss.expand_many([{"node_id": root.node_id, "p1_action": "move surf",
                            "p2_action": "switch 3", "seed": "sodium,<hex>", "label": 0}], side="p1")
    kids[0].core_p1["row"], kids[0].core_p1["mask"], kids[0].core_p1["tokens"], kids[0].node_id
```

Refused by name (never degraded): a non-core root, `rows=False`, `recorded_exact`, `integrity`, a node
of a previous root, more nodes than the spec's `max_nodes`.

**Play out to the end (Lane S)**:

```python
from utils.rust_env.successors import play_out, greedy, record_to_log
res = play_out(log, at, "p1",                    # a core INPUT LOG + the command index of the decision
               policy=greedy(scorer),            # scorer(rows (k, OBS_DIM), masks (k, 11)) -> (k, 11) logits
               seeds=["sodium,<hex>", ...],      # SHARED by every action (CRN); None = the battle's own dice
               actions=None,                     # None = every legal action of the side at `at`
               stall="production",               # training's stall forfeit for that side (StallConfig().threshold)
               max_turns=999, keep_cmds=False, core=None)
res.values()          # {action index: mean value over the seeds, from the side's view (+1 / -1 / tie 0)}
res.branches          # per branch: action, seed, reseed, end {winner, forfeit, truncated, turn}, decisions, cmds
res.tokens            # {action index: choice token} at the root
```

The policy is called with EVERY pending decision of every live branch in one batch (`who = 2*branch +
side`); the searched side's root decision is the branch's forced first action and is never pending.
`record_to_log(rec)` converts a reconstruction record; `log_to_record(log)` the other way.

## Gates

| gate | test | result |
|---|---|---|
| unit test: the playout's battle IS the env core's row path; its branch is exact | `tests/search_game_test.rs` (routine via `core_cargo_test.py`) — an inline core plays the bridge corpus (seeded random policy); every finished episode's input log replayed through a `Game`: every decision's row BYTES, mask and ordinal equal the core's; a `Game::branch(None)` at every 9th command fed the rest of the log equals the linear replay | **PASS** — 12 episodes, 2,133 decisions byte-equal, 243 branches exact (the heavier run: 18 episodes, 3,242 decisions, 655 branches); teeth: one flipped row byte fails |
| unit test: playouts are CRN-deterministic, and the recorded continuation replays the banked battle | same file | **PASS** — a rerun byte-identical; two branches sharing (action, seed) share their end and commands; another seed changes a continuation; seed null + the recorded first action + a policy answering the recorded indices reproduces the log command for command |
| **① the depth-3 successor slice** | `successors_integration_test.py::test_gate_1_*` over `successors_parity.py` — the in-process tree and the `search_driver` binary in LOCKSTEP: root fields, then 3 plies (every legal action of ours × ≤ 3 opponent choices × 2 shared seeds, plus the battle's own dice on one arm; frontier ≤ 3 branchable children), every arm field + ROW BYTES; no allowlist | **COMMIT PASS** (4 battles: ladder COMMIT + pool; 6,384 arms, 7,872 rows, 608 D10 leaves). **MILESTONE PASS** (ladder milestone 12 + pool 6 + procedural 6: 47,856 arms, 53,736 rows byte-equal, 4,422 D10 leaves, 650 terminal arms). Teeth: a flipped row byte, a changed token, changed requests each caught. Both roads REFUSED 25 batches with the same error — F-LI-1 |
| **② clone independence / determinism** (the deleted `fork_sharing_parity`'s claim) | `test_gate_2_*` | **PASS** — an arm re-expanded after its siblings is byte-identical (row, outcome, requests, tokens) under a new node id; a stale node id is refused after a new root and the handle stays usable; playouts as above |
| **③ decision equality** (the deleted `materializer_parity`'s claim) | `test_gate_3_search_decides_identically_in_process` — the real `SearchEngine` (oracle arm, pinned caps m_opp 2 / r_dice 2, `max_depth` 2, a sum-of-row scorer) on `search_impl="rust"` vs `"inproc"` | **PASS** — per-action scores bit-equal, same action, same fallback, same widths (incl. `deep_arms_expanded`, `depth_realized`), ≥ 3 decisions, deepening non-vacuous |
| the handle's contract | `test_errors_are_typed_and_do_not_poison` | **PASS** — every refusal is `SuccessorsError` (a `SearchError`) and leaves the handle usable; `max_nodes` / `max_branches` refused by name; a bad spec key is `CallerError` |
| row-level oracle | transitive: `agents/battle/core_row_parity_fuzz_test.py` holds the binary's rows to the poke-env replay; ① holds ours byte-equal to the binary's | not re-run on the in-process road (no hook in that file; byte equality is exact, so transitivity is exact) |

## Unit 4 — throughput (DESCRIPTOR)

Measured 2026-09-29 ~17:45 PT, RELEASE builds of both roads, one thread each, the ladder milestone
tier, load1 0.9–1.1 on 16 cores (`utils.contention`: "box looks idle"). Warn, never stretch.

| shape | JSON road (`search_driver` child) | in process | ratio |
|---|---|---|---|
| search's depth-1 ply (`open_root` + one `expand_many`, every action × 3 opp × 2 seeds, rows, side-elided; 24 roots, 1,080 successors / rep, 4 reps interleaved A B / B A) | 279 µs / successor (3,586 / s) | 232 µs / successor (4,314 / s) | **0.803 [0.793, 0.825]** in-process / JSON (bootstrap 95 % over roots) |
| playouts (every legal action × 4 seeds from a mid-battle decision, a stand-in scorer: one projection) | — (no playout verb; a JSON playout is one round trip per turn per branch) | **165–182 playouts / s, ~16.7 k policy decisions / s (60 µs / decision)** on one thread | — |
| playouts, one handle per Python thread (GIL released in the core) | — | 1 / 2 / 4 / 8 threads: 182 / 334 / 566 / 792 playouts / s (16.9 / 30.9 / 52.5 / 73.4 k decisions / s) | 4.4× at 8 threads |

- The successor's cost is the SIM + the fold + the encode (~230 µs); the JSON road adds the pipe,
  the reply parse and the base64 decode (~20 %). The ply is not the search's whole decision (the
  forward and the Python tree are not in either number).
- Playouts are the training path's rows (the parse fold), so ~60 µs / decision is the env core's
  per-row cost (Lane A: 75–82 µs / row at T = 1) plus a trivial scorer. A real forward dominates
  once a network scores the rows (T2).

## Findings

- **F-LI-1 (search, P1 — a GIGO class, in BOTH roads; not Lane I's file, NOT fixed):**
  `pokesim::search::resolve_turn_sourced_with` (the port's `search.rs`, a faithful port of Node's
  `resolveTurn`) feeds both sides inside one loop iteration without re-checking the boundary between
  them. When side 1's REPLACEMENT switch (after an end-of-turn faint) completes the turn, the sim opens
  the NEXT turn's move request for side 2 at once, and the same iteration answers it with the
  FOLLOW-UP policy — a next-turn choice committed inside this arm. Repro (ladder COMMIT, seed 1,
  `successors_parity.make_logs(..., 3, 1)[0]`, turn 46, side p2, arm `p2 move surf / p1 move
  rockslide / seed original`): `choices_used` = `{p1: [move rockslide, switch 4], p2: [move surf,
  switch 6]}` while p2's stream shows `|turn|47` with Starmie active and no switch. The child node's
  engine then disagrees with its own leaf (row, tokens, request): expanding it with the leaf's own token
  `switch Swampert` fails `bridge fatal: unresolvable choice … active swampert` and FAILS THE WHOLE
  `expand_many` BATCH. In the depth-3 slice: 25 batches over the 24 MILESTONE battles (the ladder
  part's 12: 10 `SwitchSpecies`, 2 `MoveName`; every example read was at the first deeper ply),
  always identical on both roads. Reach: every search that deepens past depth 1
  (`SearchEngine` iterative deepening) — a deepening ply raises, and (**UNVERIFIED**) an arm the
  engine does accept there may score a board the row does not describe. Fix (proposed): in the inner loop, stop feeding once `sess.is_ended() ||
  open_boundary_turn(sess) != start_turn`; it moves `search_driver`'s bytes, so the node driver
  (`search_driver.js`, still diffed by `harness/search_impl_parity.py`) must change with it or that
  parity retire. Owner of the fix: the port / search (not M5 Lane I).
- **F-LI-2 (for the program doc):** the "three search gates" named in the program (§2 M2 / M5 rows)
  were DELETED with the roads they compared (`43712881`); Lane I's gates ②–③ are their in-process
  successors (above), and the row-level oracle (`core_row_parity_fuzz_test`) reaches the in-process
  rows transitively through ①.
- **F-LI-3:** the search handle has no panic probe, so the POISON path (a panic inside a tree / playout
  op) is covered only by construction (the same `catch_unwind` + flag as `crate::ffi::Handle`), not by
  a test — as F-LA-4 for the env core. A test-only probe row in `FUNCTIONS` would close it.
- **F-LI-4:** a search error is status `CALLER` with kind `search` whatever its cause (a bad request
  or a battle the port refuses mid-arm); Python raises `SuccessorsError` for both. The Rust side
  returns `String` errors today; a typed split (request vs battle refusal) is a follow-up if a consumer
  must tell them apart (Lane S banking refused branches, say).
- **F-LI-5 (Lane S):** a playout's rows are the TRAINING path's (the parse fold); the search tree's rows
  are step-built versions (the `search_driver` road). Both are gated equal to the Python encoder
  elsewhere (`step == parse`, slice O), so a policy sees the same bytes on either — but only the
  playout's are gated here against the env core directly (`search_game_test`).
- **F-LI-6 (Lane S / T2):** the policy callback runs in Python between steps; with a real network the
  forward dominates (60 µs / decision is the core's share). T2's `score()` batches across branches of
  many roots only if the caller interleaves several handles; one handle per thread scales 4.4× at 8.
  A persistent worker pool INSIDE a handle (declared `threads`) is not built (a per-call thread spawn
  would violate the declared lifecycle).
- **F-LI-7:** `max_turns` must stay below 1,000 (the port panics at 1,000 committed turns, program §0);
  the playout refuses more. A branch that reaches it ends `truncated`, winner null.
- **F-LI-8 (the JSON road's deletion):** `search.py` still defaults to `search_impl="rust"` (the child).
  Switching the default to `inproc` and then deleting `search_session.py`'s JSON protocol,
  `search_driver`'s search verbs and `driver_timing.rs` (program §4's SKIPPED row) is the next step;
  the prober's `better_line` / counterfactual and the replay family still use the child and are not
  moved here. Gate ① compares against the child, so it retires WITH the JSON road (the in-process road
  then has `search_game_test` + ② + ③ + the transitive row oracle).
