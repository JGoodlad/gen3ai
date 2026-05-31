# CLAUDE.md — Observation Encoder (`src/agents/observation/`)

This directory builds the **3299-dim per-decision observation vector** (`Gen3ObservationEncoder.encode`).
It runs once per agent decision across every training env, so it sits directly on the
training-throughput (FPS) critical path. Two independent things can regress here, and they
have **different** gates:

1. **Observation *values*** — if a change alters what the vector contains, it is
   **retrain-class**: bump `ARCH_SIGNATURE` in `src/agents/model/model_version.py` (see the
   root `CLAUDE.md` → Model Versioning). Value-neutral refactors do **not** bump it.
2. **Observation *build performance*** — if a change makes `encode` slower, training FPS
   drops for the entire run. **This file governs that gate.**

---

## MANDATORY: run the obs-build benchmark on every change to this directory

**Any change to a file under `src/agents/observation/` — even a "pure refactor" or a
value-neutral one — MUST run the performance benchmark before and after the change and
confirm no meaningful regression.** This is not optional and applies to every edit, however
small. A one-line change to a hot loop (e.g. `reactive.py`, `pokemon.py`, `moves.py`,
`state_encoder.py`) can silently halve FPS.

The benchmark is `src/agents/training/obs_build_benchmark.py` (it lives in `training/` rather
than here only because a directly-run script puts its own dir on `sys.path[0]`, and
`observation/types.py` would shadow the stdlib `types` module — see the root `CLAUDE.md`
Benchmarks section).

### The workflow — do this for every change

```bash
export PYTHONPATH=$PYTHONPATH:src

# 1. BEFORE you edit: capture a baseline on the CURRENT code (stash/commit your change away,
#    or run on a clean checkout), saving the full output.
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \
    src/agents/training/obs_build_benchmark.py --turn 25 --reps 400 --top 22 | tee /tmp/obs_before.txt

# 2. Apply your change.

# 3. AFTER: re-run with the SAME flags and the SAME machine load, and diff.
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \
    src/agents/training/obs_build_benchmark.py --turn 25 --reps 400 --top 22 | tee /tmp/obs_after.txt

diff /tmp/obs_before.txt /tmp/obs_after.txt
```

If you cannot easily get a "before" (the change is already applied), compare against the
**canonical baseline pasted below** — but prefer a same-session before/after, because
absolute timings are machine- and load-dependent.

### What counts as a "meaningful regression" — use the LOAD-STABLE signals

Absolute milliseconds scale with whatever else the box is doing (training alone pushes load
past the core count and can inflate the numbers 2–3×). **Do NOT judge by the `ms` line.**
Judge by these load-independent metrics, in priority order:

1. **Total function calls per encode** = `<N function calls>` line ÷ `--reps`. This is the
   single best regression detector — it does not move with machine load. Baseline ≈
   **12.8k calls/encode** (5,131,601 / 400). A jump of **>10%** is a regression — investigate.
2. **cProfile `tottime` top-of-list structure.** A *new* function climbing into the top ~10,
   or a known hot function's **call count** ballooning, means you added work to a hot loop.
3. **Component ratios** (`state_encoder.encode` vs cached turn-history vs `live_view`) and the
   **deque-cache multiplier** (`Nx saved`). If the turn-history "cached" line stops being a
   single encode (`~12x saved` collapses toward `1x`), the deque memoization broke.

A value-neutral refactor that adds <10% calls/encode and doesn't reshuffle the tottime top is
fine. Anything larger needs justification (or a revert).

---

## Canonical baseline (paste — the reference point for regressions)

Captured with `--turn 25 --reps 400`. Paths shown repo-relative. Absolute ms omitted from the
headline on purpose (load-dependent); the **call counts and ordering are the contract**.

```
PER-DECISION OBS BUILD BENCHMARK  (obs dim 3299, turn 25, history slots 10, opp mons w/ revealed moves 5/6)

  full per-decision obs build  :  ~0.8–2.2 ms   (LOAD-DEPENDENT — not a regression signal)
    state_encoder.encode       :  ~92% of build
    turn-history (cached, 1 enc):  ~3% of build   (recompute-all-10 is ~11–13x slower → deque cache working)
    live_view() alone          :  ~8% of build

  Total: ~5.13M function calls / 400 reps  ==>  ~12.8k calls per encode   <-- PRIMARY REGRESSION METRIC

  Top functions by tottime (the hot loop is the reactive matchup matrices):
   ncalls  tottime  cumtime  function
    80800    0.154    0.791   agents/observation/reactive.py:_expected_multiplier   <-- #1 hot path
   158400    0.132    0.195   poke_env/battle/move.py:entry                          (poke-env Move property)
    80800    0.072    0.137   agents/gen3_mechanics.py:effective_multiplier_by_types (memoized; chart lookup)
   171200    0.069    0.097   poke_env/battle/pokemon.py:ability
      400    0.069    0.917   agents/observation/reactive.py:encode                  (cumtime ≈ whole matchup block)
     4800    0.064    0.244   agents/observation/moves.py:encode
    80800    0.055    0.139   agents/observation/reactive.py:_resolve_ability_distribution
   257200    0.053    0.151   {builtins.getattr}
     4800    0.049    0.406   agents/observation/pokemon.py:encode
    93600    0.049    0.230   poke_env/battle/move.py:type
   271200    0.047    0.067   enum.__hash__                                          (lru_cache key hashing)
     4800    0.044    0.153   agents/battle/live_view.py:from_pokemon
```

**Reading it:** the reactive matchup encoder (`reactive.encode` → `_expected_multiplier`,
288 cells = 2 × 6 mons × 4 moves × 6 mons) dominates. Type effectiveness is a memoized
precomputed-chart lookup (`effective_multiplier_by_types` + `_eff_cached` in
`gen3_mechanics.py`) — `PokemonType.damage_multiplier` must **not** reappear in this list
(if it does, something bypassed the chart). The remaining cost is poke-env `Move`/`Pokemon`
property reads (`move.entry`, `move.type`, `pokemon.ability`); the open follow-up is hoisting
those out of the inner loop to team level.

---

## Pitfalls that have caused regressions here

- **Calling `PokemonType.damage_multiplier` / `effective_multiplier(move_type, mon)` per cell.**
  Use the value-based `effective_multiplier_by_types(move_type, t1, t2, ability, status)` and
  read the mon's attributes once outside the loop. The object wrapper re-reads poke-env
  properties every call.
- **Re-reading poke-env properties inside the inner loop.** `move.type`, `mon.type_1/2`,
  `mon.ability` are properties that do real work (`move.entry`, `GenData.from_gen`); hoist
  them above the loop.
- **Breaking the turn-history deque cache** (`EpisodeTracker.prev_N_delta_vecs`): if the
  benchmark's "recompute all 10" multiplier collapses toward 1×, you've reintroduced the
  per-step O(N) re-encode.
- **Wrapping live mons in proxy objects** with `__getattr__` (the deleted
  `_AbilityOverrideMon`): `__getattr__` is slow and gets hit once per attribute per cell.

## Value-correctness (separate from perf, but also gated)

Changes to *what the vector contains* are validated by the bridge-backed fuzz tests
(`*_fuzz_test.py`, real battles, protocol-truth checks) and the unit tests in this directory.
If your change is meant to be **value-neutral**, prove it: the effectiveness fast-path, for
example, is pinned byte-for-byte by the exhaustive parity test in
`src/agents/gen3_mechanics_test.py`. Obs-value changes are retrain-class → bump
`ARCH_SIGNATURE`.
