"""ms per SEARCHED DECISION, and WHERE the per-arm wall goes — the real ``SearchEngine`` over a
BANKED eval trace, on either materializer road.

    export PYTHONPATH=$PYTHONPATH:src
    python3 src/main/search_dividend/search_decision_benchmark.py \
        --traces models/<run>/eval_traces --decisions 6 --b 1 33

**What this is, and what `view_materialize_benchmark.py` is.** That one times the two
materializer roads on a hand-built arm set: it is the road A/B. This one drives
``SearchEngine.choose`` — the thing production calls — on a decision reconstructed from a banked
eval trace, and breaks the wall into the phases a change can actually attack. The two answer
different questions and both are kept.

**The scorer is the pure ``obs.sum``, deliberately** — the same one
``materializer_parity_integration_test`` uses. A trained net's forward is a real cost but it is
not a cost this file's consumers can change, and threading a checkpoint in would make the table a
statement about which checkpoint was on disk.

**The phase table is EXCLUSIVE time, stack-accounted.** Each wrapper subtracts whatever nested
wrapper ran inside it, so the rows sum to the instrumented total and the remainder is reported as
`python glue` rather than hidden. A row is attributed to the road that pays it: the VIEW road has
no snapshot restore and the PROTOCOL road has no event fold, and both print as `-`.

🚨 **WARN, NEVER STRETCH.** The load is printed with every table and the ratios are the claim.
Run the two roads back to back — this script already interleaves them per decision, which is the
only honest A/B on a shared box.
"""

from __future__ import annotations

import argparse
import contextlib
import cProfile
import json
import os
import pstats
import random
import statistics
import sys
import time
from typing import Dict, List

import numpy as np

from main.search_dividend.search import SearchConfig, SearchEngine
from main.search_dividend.budget import WidthCaps
from utils.contention import describe_contention


# ---------------------------------------------------------------------------
# the stack-accounted phase timer
# ---------------------------------------------------------------------------


class Phases:
    """Exclusive wall per named phase, with nesting handled.

    A plain per-function accumulator double-counts the moment one instrumented call sits inside
    another (``open_view_fork`` contains the prefix's own ``encode``), and a table whose rows sum
    to 180% of the wall is worse than no table."""

    def __init__(self) -> None:
        self.excl: Dict[str, float] = {}
        self.incl: Dict[str, float] = {}
        self.calls: Dict[str, int] = {}
        self._stack: List[float] = []

    @contextlib.contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        self._stack.append(0.0)
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            child = self._stack.pop()
            self.excl[name] = self.excl.get(name, 0.0) + (dt - child)
            self.incl[name] = self.incl.get(name, 0.0) + dt
            self.calls[name] = self.calls.get(name, 0) + 1
            if self._stack:
                self._stack[-1] += dt

    def reset(self) -> None:
        self.incl.clear()
        self.excl.clear()
        self.calls.clear()
        self._stack.clear()


PH = Phases()


def _wrap(obj, attr: str, name: str, patches: list) -> None:
    """Wrap ``obj.attr`` to bill its wall to ``name``.

    A MISSING attribute is skipped rather than raising, so this script also runs against an older
    checkout — which is the only way to get an interleaved before/after on a shared box. It is
    reported, because a silently skipped wrap is a phase that lands in `python glue` and reads as
    an improvement."""
    if not hasattr(obj, attr):
        print(f"  (no {getattr(obj, '__name__', obj)}.{attr} in this checkout — "
              f"phase {name!r} not instrumented)")
        return
    orig = getattr(obj, attr)
    patches.append((obj, attr, orig))

    def wrapper(*a, **kw):
        with PH(name):
            return orig(*a, **kw)
    setattr(obj, attr, wrapper)


@contextlib.contextmanager
def instrumented():
    """Wrap every phase BOTH roads can pay. The set was read out of the code, not guessed:
    ``search._materialize`` is the one seam and each name below is a call it makes."""
    import agents.battle.event_fold as EF
    import agents.battle.view_adapter as VA
    import agents.training.obs_materializer as OM
    import agents.training.view_successor as VS
    import agents.action.mask_generator as MG
    import utils.bridge.search_session as SS
    import agents.observation.state_encoder as SE

    patches: list = []
    try:
        # --- the port (rust) ------------------------------------------------
        _wrap(SS.SearchSession, "expand_many", "port expand_many (rust)", patches)
        _wrap(SS.SearchSession, "open_root", "port open_root (rust)", patches)
        # --- shared prefix, and the PROTOCOL road's own internals -----------
        _wrap(OM, "open_view_fork", "prefix replay (open_view_fork)", patches)
        _wrap(OM, "materialize_branches", "protocol road (materialize_branches)", patches)
        # `gen3_one_fork_per_decision_v1` split `materialize_branches` in two and the search calls
        # the halves directly, so BOTH names are wrapped: an un-wrapped half would silently land
        # in `python glue` and read as an improvement.
        _wrap(OM, "open_branch_fork", "protocol prefix replay (open_branch_fork)", patches)
        _wrap(OM, "materialize_branches_from", "protocol arms (materialize_branches_from)",
              patches)
        _wrap(OM, "_feed", "poke-env protocol feed", patches)
        _wrap(OM, "_build_replay_player", "replay player build", patches)
        _wrap(OM._PlayerSnapshot, "restore", "snapshot restore (protocol)", patches)
        _wrap(OM._PlayerSnapshot, "_freeze", "snapshot freeze (protocol)", patches)
        _wrap(OM._ReplayObsPlayer, "_encode_or_track", "decision encode/track", patches)
        _wrap(OM._ReplayObsPlayer, "_choice_map", "action_choices / map_actions_at", patches)
        # --- per arm, VIEW road ---------------------------------------------
        _wrap(VS.ViewSuccessorFactory, "_clone_tracker", "tracker fork (thaw)", patches)
        _wrap(EF.ViewEventFolder, "fold", "event fold (ply protocol)", patches)
        _wrap(EF.ViewEventFolder, "branch", "event-fold branch", patches)
        _wrap(VA, "read_models_from_payload", "view_adapter read-models", patches)
        _wrap(MG.Gen3ActionMasker, "get_mask", "action mask", patches)
        _wrap(VS, "view_context", "successor context", patches)
        _wrap(VS, "_choice_map", "action_choices / map_actions_at", patches)
        _wrap(SE.Gen3ObservationEncoder, "encode", "encode (obs)", patches)
        # --- per arm, CORE road (`gen3_core_search_v1`) ------------------------
        # `core_successor` imports these by name, so its OWN bindings are the ones to wrap.
        import agents.training.core_successor as CS
        _wrap(CS, "live_view_from_core", "core read-models (transport)", patches)
        _wrap(CS, "legal_actions_from_core", "core read-models (transport)", patches)
        _wrap(CS, "events_from_readings", "core ply events (readings)", patches)
        _wrap(CS, "view_context", "successor context", patches)
        _wrap(CS, "_choice_map", "action_choices / map_actions_at", patches)
        # `view_successor` imports the two names at module import time.
        VS.read_models_from_payload = VA.read_models_from_payload
        yield PH
    finally:
        for obj, attr, orig in reversed(patches):
            setattr(obj, attr, orig)
        VS.read_models_from_payload = VA.read_models_from_payload


# ---------------------------------------------------------------------------
# the decision fixtures, from BANKED eval traces
# ---------------------------------------------------------------------------


def discover(traces_dir: str) -> List[str]:
    out: List[str] = []
    for dirpath, _dirs, files in os.walk(traces_dir):
        for f in sorted(files):
            if f.endswith("_reconstruction.json"):
                stem = os.path.join(dirpath, f[: -len("_reconstruction.json")])
                if (os.path.exists(stem + "_summary.json")
                        and os.path.exists(stem + "_states.npz")):
                    out.append(stem)
    out.sort()
    return out


def load_decision(stem: str, impl: str, frac: float = 0.55):
    """``(record, side, turn, our_history, tokens, observed, opp_true)`` or ``None``.

    The surface is read off a ``SearchSession`` root, which is where the live player gets it —
    so the decision this benchmark searches is the decision production would have searched."""
    from main.search_dividend import determinize as dz
    from utils.bridge.reconstruction import ReconstructionRecord
    from utils.bridge.search_session import SearchSession
    import agents.battle.one_sided_view_parity_fuzz_test as G

    record = ReconstructionRecord.load(stem + "_reconstruction.json")
    with open(stem + "_summary.json") as fh:
        summary = json.load(fh)
    npz = np.load(stem + "_states.npz", allow_pickle=True)
    actions = np.asarray(npz["actions"], dtype=int)
    invs = summary.get("invocations", [])
    cand = [i for i, iv in enumerate(invs)
            if iv.get("phase") == "move_selection" and i < len(actions)
            and int(iv.get("turn", 0)) > 1]
    if not cand:
        return None
    anchor = cand[min(len(cand) - 1, int(len(cand) * frac))]
    turn = int(invs[anchor]["turn"])
    side = record.side_of(record.trainee_username)
    our_history = [int(x) for x in actions[:anchor]]
    with SearchSession(record, impl=impl) as ss:
        root = ss.open_root(turn)
        pfx = root.prefix_p1_chunks if side == "p1" else root.prefix_p2_chunks
    tokens = G._choice_map(record, side, our_history, pfx, anchor)
    if not tokens:
        return None
    observed = dz.chunks_to_lines(pfx)
    opp_true = record.packed_team("p2" if side == "p1" else "p1")
    return record, side, turn, our_history, tokens, observed, opp_true, anchor


# ---------------------------------------------------------------------------
# one searched decision
# ---------------------------------------------------------------------------


#: road name -> (materializer, core_path). ``core`` is the typed shortcut, ``core-text`` the
#: side's protocol text through ``parse`` — the pair the Rust Core Program's §6 decision reads.
ROADS = {"protocol": ("protocol", "typed"), "view": ("view", "typed"),
         "core": ("core", "typed"), "core-text": ("core", "text")}


def _cfg(materializer: str, *, m_opp: int, k_worlds: int, arm: str, impl: str) -> SearchConfig:
    mat, path = ROADS[materializer]
    return SearchConfig(
        arm=arm, budget_s=1e9, seed=7, max_depth=1, search_impl=impl,
        materializer=mat, core_path=path, integrity=0,
        caps=WidthCaps(m_opp=m_opp, k_worlds=k_worlds, r_dice=1))


def decide(materializer: str, fx, *, m_opp: int, k_worlds: int, arm: str, impl: str,
           pool_packed):
    record, side, turn, our_history, tokens, observed, opp_true, _anchor = fx
    engine = SearchEngine(model=None, mappings=None,
                          cfg=_cfg(materializer, m_opp=m_opp, k_worlds=k_worlds,
                                   arm=arm, impl=impl),
                          pool_packed=list(pool_packed))
    engine._score_batch = lambda obs, masks: (                      # type: ignore[assignment]
        np.asarray(obs, dtype=np.float64).sum(axis=1), "fake")
    t0 = time.perf_counter()
    try:
        res = engine.choose(
            record=record, side=side, turn=turn, our_history=list(our_history),
            our_tokens=dict(tokens), observed_our_lines=list(observed),
            pub=None, policy_action=next(iter(tokens)),
            opp_true_packed=(opp_true if arm in ("oracle", "playoff") else None))
    finally:
        engine.close()
    return res, (time.perf_counter() - t0) * 1000.0


def _pool(limit: int = 40) -> List[str]:
    """Packed teams for the HONEST arm's determinization. The registered pool, read through the
    data facade — an honest arm with no pool builds no world and the row is then vacuous."""
    from main.search_dividend.__main__ import _pool as _real_pool
    return _real_pool(limit)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


RUST_T: Dict[str, float] = {}


@contextlib.contextmanager
def rust_timing(on: bool):
    """Sum the driver's own `timing_us` (``POKESIM_SEARCH_TIMING=1``) over every expand_many."""
    if not on:
        yield
        return
    import utils.bridge.search_session as SS

    os.environ["POKESIM_SEARCH_TIMING"] = "1"
    orig = SS.SearchSession._call

    def call(self, payload):
        out = orig(self, payload)
        for k, v in (out.get("timing_us") or {}).items():
            RUST_T[k] = RUST_T.get(k, 0.0) + float(v)
        return out
    SS.SearchSession._call = call
    try:
        yield
    finally:
        SS.SearchSession._call = orig


def run(args) -> int:
    with rust_timing(args.rust_timing):
        return _run(args)


def _run(args) -> int:
    impl = args.impl
    stems = discover(args.traces)
    if not stems:
        print(f"no banked reconstructions under {args.traces}", file=sys.stderr)
        return 2
    rng = random.Random(args.seed)
    rng.shuffle(stems)
    pool = _pool() if args.arm == "honest" else []

    print(f"\nSEARCHED-DECISION BENCHMARK  ({args.arm} arm, m_opp={args.m_opp}, "
          f"k_worlds={args.k_worlds}, impl={impl})")
    print(f"  {describe_contention()}")
    print(f"  traces: {args.traces}  ({len(stems)} banked battles)")
    print("  🚨 ratios are the claim; absolute ms scale with whatever else the box is doing.\n")

    roads = [r for r in args.roads]
    wall: Dict[str, List[float]] = {r: [] for r in roads}
    per_arm: Dict[str, List[float]] = {r: [] for r in roads}
    phases: Dict[str, Dict[str, float]] = {r: {} for r in roads}
    spans: Dict[str, Dict[str, float]] = {r: {} for r in roads}
    arms: Dict[str, int] = {r: 0 for r in roads}
    view_arms = 0
    fb_mid = 0
    fb_nopay = 0
    used = 0

    with instrumented():
        for stem in stems:
            if used >= args.decisions:
                break
            try:
                fx = load_decision(stem, impl, args.frac)
            except Exception as e:                               # noqa: BLE001
                print(f"  skip {os.path.basename(stem)}: {type(e).__name__}: {e}")
                continue
            if fx is None:
                continue
            if args.n_actions:
                keep = sorted(fx[4])[:args.n_actions]
                fx = fx[:4] + ({k: fx[4][k] for k in keep},) + fx[5:]
                if not fx[4]:
                    continue
            row: Dict[str, tuple] = {}
            ok = True
            for road in roads:
                PH.reset()
                try:
                    res, ms = decide(road, fx, m_opp=args.m_opp, k_worlds=args.k_worlds,
                                     arm=args.arm, impl=impl, pool_packed=pool)
                except Exception as e:                           # noqa: BLE001
                    print(f"  skip {os.path.basename(stem)} [{road}]: "
                          f"{type(e).__name__}: {e}")
                    ok = False
                    break
                n = int(res.widths.arms_scored)
                if n <= 0:
                    ok = False
                    break
                row[road] = (ms, n, dict(PH.excl), res, dict(PH.incl))
            if not ok or len(row) != len(roads):
                continue
            used += 1
            for road, (ms, n, ex, res, inc) in row.items():
                wall[road].append(ms)
                per_arm[road].append(ms / n)
                arms[road] += n
                for k, v in ex.items():
                    phases[road][k] = phases[road].get(k, 0.0) + v
                for k, v in inc.items():
                    spans[road][k] = spans[road].get(k, 0.0) + v
                if road == "view":
                    view_arms += int(getattr(res.widths, "view_arms", 0))
                    fb_mid += int(getattr(res.widths, "view_fallback_intermediate", 0))
                    fb_nopay += int(getattr(res.widths, "view_fallback_no_payload", 0))
            t = row[roads[0]]
            w0 = row[roads[0]][3].widths
            print(f"  turn {fx[2]:>3}  arms={t[1]:>3}  worlds={w0.worlds_gated_ok}"
                  f"/{w0.worlds_requested}  " +
                  "  ".join(f"{r}={row[r][0]:8.1f} ms" for r in roads))

    if not used:
        print("no decision produced a scored arm — nothing to report", file=sys.stderr)
        return 3

    print(f"\n  DECISIONS: {used}")
    print(f"  {'road':>10}  {'arms':>5}  {'ms/decision':>12}  {'ms/arm':>9}")
    for road in roads:
        print(f"  {road:>10}  {arms[road]:>5}  {statistics.median(wall[road]):>12.1f}  "
              f"{statistics.median(per_arm[road]):>9.3f}")
    if len(roads) == 2:
        a, b = roads
        print(f"  speedup {a} -> {b}: "
              f"{statistics.median(wall[a]) / statistics.median(wall[b]):.2f}x per decision")
    if "view" in roads:
        print(f"  view_arms={view_arms}  fallback_intermediate={fb_mid}  "
              f"fallback_no_payload={fb_nopay}")

    for road in roads:
        tot = sum(phases[road].values())
        dec_tot = sum(wall[road])
        print(f"\n  PHASE BREAKDOWN — {road} road   "
              f"(exclusive wall, summed over {used} decisions / {arms[road]} arms)")
        print(f"  {'phase':<38} {'ms total':>10} {'ms/arm':>9} {'% dec':>7}")
        for k, v in sorted(phases[road].items(), key=lambda kv: -kv[1]):
            print(f"  {k:<38} {v * 1000:>10.1f} {v * 1000 / max(arms[road], 1):>9.3f} "
                  f"{100 * v * 1000 / max(dec_tot, 1e-9):>6.1f}%")
        glue = dec_tot - tot * 1000
        print(f"  {'python glue (uninstrumented)':<38} {glue:>10.1f} "
              f"{glue / max(arms[road], 1):>9.3f} {100 * glue / max(dec_tot, 1e-9):>6.1f}%")
        print(f"  SPANS (INCLUSIVE, so they overlap the rows above) — {road}")
        for k in ("prefix replay (open_view_fork)",
                  "protocol prefix replay (open_branch_fork)",
                  "protocol arms (materialize_branches_from)",
                  "protocol road (materialize_branches)",
                  "port open_root (rust)", "port expand_many (rust)"):
            v = spans[road].get(k)
            if v:
                print(f"    {k:<36} {v * 1000:>10.1f} ms  "
                      f"{100 * v * 1000 / max(dec_tot, 1e-9):>5.1f}% of decision wall")
    if args.rust_timing:
        print("\n  RUST DRIVER PHASES (timing_us, summed over the run; `core` = the version fold, "
              "`core_render` = the leaf's view/legal/events JSON)")
        for k, v in sorted(RUST_T.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<14} {v / 1000:>10.1f} ms  {v / 1000 / max(sum(arms.values()), 1):>8.4f} ms/arm")
    print(f"\n  {describe_contention()}")
    if args.json_out:
        with open(args.json_out, "a") as fh:
            for road in roads:
                fh.write(json.dumps({
                    "road": road, "decisions": used, "arms": arms[road],
                    "ms_per_decision_median": statistics.median(wall[road]),
                    "ms_per_arm_median": statistics.median(per_arm[road]),
                    "ms_total": sum(wall[road]),
                    "phases_ms": {k: v * 1000 for k, v in phases[road].items()},
                    "spans_ms": {k: v * 1000 for k, v in spans[road].items()},
                    "rust_timing_ms": {k: v / 1000 for k, v in RUST_T.items()},
                    "m_opp": args.m_opp, "k_worlds": args.k_worlds, "n_actions": args.n_actions,
                    "arm": args.arm, "load": os.getloadavg(), "contention": describe_contention(),
                }) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="an eval_traces directory (READ-ONLY)")
    ap.add_argument("--decisions", type=int, default=6)
    ap.add_argument("--n-actions", type=int, default=0,
                    help="truncate OUR legal action set to N (B = n_actions * m_opp); 0 = all")
    ap.add_argument("--m-opp", type=int, default=1,
                    help="opponent candidates; arms per world ~= n_actions * m_opp")
    ap.add_argument("--k-worlds", type=int, default=1)
    ap.add_argument("--arm", default="oracle", choices=("base", "honest", "oracle"))
    ap.add_argument("--impl", default="rust")
    ap.add_argument("--frac", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--roads", nargs="+", default=["protocol", "view"],
                    choices=tuple(ROADS),
                    help="ONE road per process for an A/B claim (the two roads in one interpreter "
                         "are not independent — expand_many_2026-09-22/README.md); `core` is the "
                         "typed shortcut, `core-text` the text path")
    ap.add_argument("--rust-timing", action="store_true",
                    help="set POKESIM_SEARCH_TIMING=1 for the driver child and sum its per-phase "
                         "`timing_us` (sim / chunks / core fold / core render / …) per road")
    ap.add_argument("--json-out", default=None,
                    help="append one JSON row per road (medians, arms, phases, rust timing, load)")
    ap.add_argument("--cprofile", default=None,
                    help="also write a cProfile .prof of the whole run here")
    a = ap.parse_args()
    if a.cprofile:
        pr = cProfile.Profile()
        pr.enable()
        rc = run(a)
        pr.disable()
        pr.dump_stats(a.cprofile)
        pstats.Stats(pr).sort_stats("cumulative").print_stats(35)
        return rc
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
