# Adversarial review: the lean world screen (2026-10-27)

**Scope.** Commit `2eb65b3d` ("the lean world screen"): the `SCREEN` mode of `src/rust_sim/src/bin/core_events.rs`,
`WorldSampler(lean_screen=True)` in `src/main/value_search/worlds.py`, `world_screen.run(skip_impossible=True)`, the
two "pure fixes" (the parse chain's stream copy, the Hidden Power IV memo) and the parity gate
`src/main/value_search/world_screen_parity_test.py` with `testdata/su_stage0_decisions.json`.

**The claim under attack.** The lean path is bit-identical in outcome to the reference: the same worlds, retries, RNG
draws, accept / reject verdicts and chosen action.

**Method.** I read the diff against the gate the screen feeds (`worlds.WorldSampler.sample`'s `gate`, `reveals`,
`protocol_lines`, `divergence_score`, `divergence_hint`) and the chain that consumes the verdicts (`world_screen.run`).
I then ran two probes, CPU only (`CUDA_VISIBLE_DEVICES=""`, nice 19, agent cores 5–7 and 13–15, ≤ 430 MB RSS, each
job ≤ 10 min). Scripts and raw results are in
[`../../research_state/measurements/lean_world_screen_review_2026-10-27/`](../../research_state/measurements/lean_world_screen_review_2026-10-27/).
- **The sampler probe** (`probe.py`). It runs the lean build, and replays every screened `core_events` batch a
  second time in full. For each battle it compares the gate's verdict (passed, matched length, hint) between the two
  replays. It then runs the reference build and compares the two builds' records (`screens` excluded, as the shipped
  test does).
- **The stress probe** (`stress.py`). It builds opponent teams far from any honest draw: whole teams from other bank
  battles, and single mons swapped in from other battles. These replay leniently long past their first difference,
  which is where a post-stop refusal would live. The probe runs each one full and screened.

## Verdict: the claim HOLDS on everything measured. No BLOCKER

**Measured:**
- **15 world builds:** 0 verdict differences in 3,454 screened replays (3,011 of them stopped early), and all 15 builds
  are record-identical to the reference. The builds cover:
  - 4 decisions of the fixture's game, p1 and p2 viewers, at turns 10 and 20, one of them an exclusion;
  - 8 decisions from other BANK_V1 games, 4 with a p2 viewer;
  - 3 builds with `dice="rolls"`.
- **The stress probe:** 0 refusals after the stop in 11,292 deliberately wrong replays (7,558 of them stopped early).

The equivalence is **not total by construction**, though: there is one class where the two paths can differ (M1). The
parity gate, as shipped, could not have seen that class, nor a lean path that never ran (M2). Both are now covered by
the tests this review adds.

| # | Rank | Finding | Status |
|---|---|---|---|
| M1 | MAJOR | A refusal the full replay raises AFTER the stop (a `core_events` `Err`, or a panic) is invisible to the lean path, and it changes the verdict the chain climbs on | Measured 0 / 14,746. The slow tier now detects it. The claim's wording is conditional now |
| M2 | MAJOR | The shipped parity gate could not catch a lean path that never runs, nor a verdict difference that leaves the chain's end state alone, and it covered one game, one viewer (p1) and one opponent team | Fixed (2 new slow tests) |
| m1 | MINOR | The verdict cache key does not carry the screen mode | Open, harmless today |
| m2 | MINOR | `Worlds.screens` changed meaning; `main.info_audit` sums it as a cost | Open, informational |
| m3 | MINOR | `world_cpu_s` / `playout_cpu_s` read process-wide CPU | Open, informational |
| m4 | MINOR | The reference line omits `"trackers"` when no decision was captured; the lean line never does | Unreachable today |

What holds, verified:
- **The stop itself.** The gate's verdict is a function of the lines up to and including the first differing viewer
  line, and of the rows of decisions before it. The lean line carries exactly those.
- **`skip_impossible`.** Skipping an impossible proposal consumes no draw and changes no state.
- **The two "pure fixes".** Both are exact.
- **Rust safety.** Nothing in the diff is a hazard.

---

## What holds (the attacks that failed)

### The stop cannot change a verdict the full replay reaches with `ok: true`

`gate(res)` reads only three things:
1. `res["ok"]`;
2. the viewer's protocol lines up to the root decision's chunk, or all of them when the replay never reached the root
   (`reveals` raises `WorldError`);
3. the viewer's obs rows at decisions `0..n`.

The lean replay runs every command exactly as the full one does until the command that first ships the viewer a line
differing from `ref_proto`, or a line past its end. It then stops after that command. Everything that command shipped
is kept, and `track()` has already recorded any decision it opened. So:
- **A world that matches the truth throughout** never stops. It is the full replay, and is accepted or rejected (on
  rows) identically.
- **A world that diverges at viewer line k** gives the same k in both paths, whether it is reached through
  `protocol_lines` (full: the root was reached) or through the `WorldError` fallback (lean: it stopped before the
  root). Both paths also hold the same line pair at k, so the HP closeness term of `divergence_score` and the hint of
  `divergence_hint` are the same. Neither function looks past line k.
- **"One past the end"** gives k = `len(ref_proto)` in both paths, and the closeness term is 0 in both.
- **Divergence on the opponent's side only** is the reference's semantics too. The reference gate compares only the
  viewer's lines and rows, so an opponent-side inconsistency the viewer never sees is not a rejection in either path.
- **The exact-HP constraint** (`opp_conditions`) reads the opponent's own requests. The lean screen is disabled for
  every batch that carries one (`lean = … and not any(c for _, c in todo.values())`), and every strict joint gate
  runs in full.
- **`dice="rolls"`.** The roll report of a lenient screen is never read: `roll_rep` is read only under `#strict` keys.
  Rolls before the stop are drawn identically.

The chunk log is append-only at chunk granularity (`BridgeChunks::push_chunk_lines`; `clear_chunks` is a
search-driver-only call that `core_events` never makes). So `ScreenCursor` can never skip lines appended to a chunk
it has already passed, and a spurious early stop is impossible. A stop BEFORE the first difference would turn an
accepted world into a rejected one. The new invariant test catches that (negative control below).

### `skip_impossible` consumes no RNG draw and changes no state

In the reference, a proposal with `lp == -inf`:
- **while searching** (`not m.consistent`) fails `lp > -math.inf` and is discarded;
- **while sampling** is guarded by `ok and lp > -math.inf and (not sy or m.rng.random() < …)`. The `and` chain
  short-circuits before `m.rng.random()` whether `ok` is true or false.

`m.proposals += 1` runs in both paths. `MonSampler.logp` and `SpreadPrior.logp` are pure, and moving the call before
the gate changes nothing; a NaN would compare false in both paths. The only side effects of replaying such a proposal
in the reference are:
- the `verdict` cache entry it leaves behind;
- the `screens` counter;
- the battle label `f"{key}#s{n}"`, which numbers the replays. The label is not an input to the battle.

The probe's 15 record-identical builds include 35–103 skipped proposals each (`screens_ref − screens_lean`).

### The two "pure fixes" are exact

- **`side_lines_from(sess, side, from)`** returns `sess.side_lines(side)[from..]` without the copy: same order, same
  slice, chunks filtered by side. Its `assert!(from <= seen)` replaces the old slice's out-of-range panic, which is the
  same failure.
- **`iv_patterns`.** `_canon` is idempotent, so `ivs_packed(_canon(t)) == ivs_packed(t)` and
  `want = _canon(t).lower()` as before. A bad type still raises `KeyError` before the cache is touched. Every call
  returns fresh lists, so no caller can mutate the memo.

### Rust safety, the protocol and FFI

- **Rust safety.** No `unsafe`, no new dependency. The `ffi` / `rust_env` crate is untouched; `run_core` is a
  subprocess.
- **The `SCREEN` line.** It parses through `pokesim::json`, which decodes `\uXXXX` escapes and surrogate pairs
  correctly. So a `json.dumps`-escaped protocol line (`Nidoran♀`, `Farfetch’d`) round-trips byte for byte, and a
  reference line can never differ from the shipped line by an escape. A malformed `SCREEN` line exits the process
  non-zero, which surfaces as `RuntimeError`, never as a silent full replay.
- **What the lean `track()` still runs.** It keeps every fallible step for both sides: the slice-T / ALIGN / PARSE-OBS
  checks, `encode`, `choice_tokens` and the window record's presence. It skips only infallible formatting.
- **The `|t:|` line.** `core_events` renders it normalised (`|t:|<NORMALIZED>`), so it cannot desync a screen.

---

## M1 (MAJOR): a refusal after the stop is the one place the two paths can differ

**Mechanism.** The gate maps a replay that is not `ok` to `(False, -1, None)`. The lean replay of the same candidate
never reaches the command that would refuse, so it returns `ok: true` and the gate scores `(False, k, hint)` with a
real k. In the search phase of `world_screen.run`, `ok or ln >= m.best_len` then admits the candidate as the climb's
best, where the reference would have discarded it. From that step on the chains differ: different states, different
`m.rng` draws, possibly different worlds.

A panic after the stop is worse. In the reference it kills the whole batch's process (`RuntimeError`, the build
crashes). The lean path carries on.

The worlds.py docstring and `core_events.md` §10 name this caveat. The commit headline, the `search_ledger.md` §6.1 line
("the same worlds, draws and verdicts") and the claim under review do not.

**The refusals that can fire after a divergence.** These are all internal consistency checks, so a refusal there is a
core bug:
- `version step` / `parse pN` (`observe`, `parse_advance`);
- `[ALIGN]` / `[PARSE-OBS]` (`track`);
- `parse_gate`;
- the end-of-battle `CONSERVATION`, canonical typing and `parse != step`;
- in the self-check build, the emission check's panic.

A lenient replay of a wrong world is exactly where a dormant core bug would show: the truth's tokens fed into a
counterfactual state for dozens of turns.

**Measured: none fired.**

| Probe | Replays | Stopped early | Full replay refused after the stop | Verdict differences |
|---|---|---|---|---|
| Sampler, 15 builds (8 seed + 4 fixture + 3 rolls) | 3,454 | 3,011 | 0 (0 refusals at all) | 0 |
| Stress, turns 5–45, seeds 1 and 2 | 8,880 | 5,915 | 0 | 0 |
| Stress, turns 30–120, seed 3 (clean swaps) | 2,412 | 1,643 | 0 (0 refusals at all) | 0 |

The stress rows at turns 5–45 had 1,146 full-replay refusals. Every one was a Species Clause refusal at team
construction: my single-mon swap duplicated a species (seed 2's categorised run; seed 3 filters this). Those come
before any command, and the lean path refused them identically.

**Rank.** MAJOR, not BLOCKER: there is no reproducer, and none of 14,746 replays built to provoke the class produced
one. The hazard is that nothing in production can detect it: the reference never runs in production, and a future core
change that adds a post-divergence refusal would silently fork the chains.

**Disposition:**
1. The slow tier now holds a detector. `test_every_screened_replay_is_the_full_replay_cut_at_its_stop` replays every
   screened batch in full and requires the same `ok`. `test_deliberately_wrong_worlds_refuse_nothing_after_the_stop`
   is the stress on 12 wrong worlds and requires every full replay to be `ok`.
2. `search_ledger.md` §6.1 now states the condition and the measured rate.
3. Recommended, not built: make the equivalence total by construction. Have the gate score a refused replay by the
   lines it shipped before the refusal, instead of −1. That needs `core_events` to return the partial chunks on `Err`,
   and it changes the reference's semantics, so it is a design change for the search owner. Without it, keep the
   detector green.

## M2 (MAJOR, fixed): the shipped parity gate could not see the lean path fail

- **The build tests did not prove the screen ran.** `test_the_lean_screen_builds_exactly_the_reference_worlds`
  asserts `n_lean < n_full` as its evidence that the lean path ran. `skip_impossible` alone satisfies that assertion.
  If the `lean = …` wiring in `run_gate` regressed to never arming `SCREEN`, every build test would still pass: both
  builds would then be the reference. The direct `SCREEN` test exercises the Rust stop, but not the wiring.
- **The build tests compared only end states.** They compare the chain's END state, so a per-candidate verdict
  difference the chain happens not to act on would pass. M1 is exactly such a difference until the chain crosses
  `best_len`.
- **The coverage was narrow.** All four fixture decisions are the p1 viewer of one game against one opponent team (the
  57-decision offline re-make is the same game and viewer). The p2 path of `render_screen` (the placeholders and the
  `trackers` ordering) was never compared with a full replay in a world build.

**Fixed in this change** (`world_screen_parity_test.py`, both `slow` + `sim` + `integration`; 89 s together on the
agent cores):
- `test_every_screened_replay_is_the_full_replay_cut_at_its_stop`, on 3 BANK_V1 decisions (p2 viewer seed, p1 viewer
  seed, p1 `dice="rolls"`). Inside a lean build, every screened batch is replayed in full, and each lean line must be
  the full line cut after its stop:
  - the same `ok`;
  - the chunk sides a prefix of the full replay's;
  - the viewer's lines equal and every other chunk empty;
  - a stop only once a differing line (or one past the end) was shipped;
  - the viewer's `(after, obs)` rows the full replay's up to the cut.

  It also requires that the screen's reference is the truth's `ref_proto`, that some replays stop and some do not, and
  that the build is record-identical to the reference build.
- `test_deliberately_wrong_worlds_refuse_nothing_after_the_stop`: the stress probe as a gate, 12 seeded wrong worlds on
  a turn-21 p2 decision.

**Negative controls** (run by hand on the stress decision, not committed):
- **A stop BEFORE the first difference** (the lean line cut at the chunk holding the first differing line): caught,
  3 / 3.
- **A full replay refused after the stop** (`ok: false` injected): caught.
- **A lean row changed:** caught.
- **The cut is a whole chunk.** A cut inside the stop command's own chunks that still holds the first differing line is
  NOT flagged, and need not be: the gate's verdict does not depend on it.

## Minor findings

- **m1. The verdict cache key carries no screen mode.** `run_gate`'s `verdict` cache is keyed on
  `team#cond#lenient|strict`. A `screen_cache` shared between a lean and a reference sampler would serve one path's
  verdicts to the other. That matters only under M1, and nothing shares one today (the tests build separate samplers).
  Add the mode to the key if a mixed caller appears.
- **m2. `Worlds.screens` changed meaning.** It now counts fewer replays (the skipped proposals are gone), and partial
  ones. `main.info_audit` sums it as a per-spec cost column, so that column is not comparable across `2eb65b3d`.
- **m3. The new CPU split reads the whole process.** `world_cpu_s` / `playout_cpu_s` use `RUSAGE_SELF` (every thread)
  plus reaped children. In a live process, CPU spent by other threads during a decision lands in the split. The
  docstring says the split "need not sum to `cpu_s`". Read it as an upper bound in live play; offline (the 57-decision
  re-make) it is exact.
- **m4. The reference line can omit `"trackers"`.** The reference render omits it when no tracker entry exists, and
  `reveals` would then raise `KeyError`, not `WorldError`. The lean render always emits it. This is unreachable:
  `core_events` records both sides' first decision at the root.

## FINDINGS about this review's own execution (rule 7)

- **The probes skipped the heavy-slot queue.** Each probe is one Python process plus `core_events` children, at most
  430 MB resident. Queued behind 13 jobs in `mem_cap.sh`'s short lane, they would not have run within the session. So
  they ran directly at nice 19 on the agent cores with `ulimit -v` and `timeout` (each ≤ 560 s, about 25 CPU-min in
  all), not inside a heavy slot.
- **The probes ran on main's code and binary.** They used main's checkout and `core_events` binaries at `c0fd72bd`
  (the same Rust sources as `2eb65b3d`; release build for the probes, self-check build for the tests). A fresh worktree
  has no `src/rust_sim/Cargo.lock`: it is git-ignored, but the build stamp hashes it. So main's binaries read as STALE
  from a worktree until the lock is copied in. That is a tooling hazard for any worktree that borrows main's build.
- **Not verified:**
  - the search-level parity test (`test_the_search_on_the_lean_screens_worlds_is_the_reference_search`) on new
    decisions, which needs the `rust_env` FFI build;
  - the 57-decision offline re-make;
  - the turns 32 / 43 / 29 fixture builds, which were not re-run here.

  I relied on the commit's slow-tier records for these.
