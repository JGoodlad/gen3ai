"""WHERE `expand_many`'s 20.2% GOES — the rust/transport/python split, and the payload census.

    export PYTHONPATH=$PYTHONPATH:src
    POKESIM_SEARCH_DRIVER_BIN=<worktree>/src/rust_sim/target/release/search_driver \
    POKESIM_SEARCH_TIMING=1 \
    python3 designs/research_state/measurements/expand_many_2026-09-22/profile_expand_many.py \
        --traces models/ai_v12_02_winprob_critic/eval_traces --decisions 12 --m-opp 3 --k-worlds 4

`search_decision_benchmark.py` bills ONE span for `expand_many`, which is the request write, the
child's whole handler and `json.loads` of the reply together. Those three want OPPOSITE fixes, so
this script splits them:

* the CHILD's own phases come from `POKESIM_SEARCH_TIMING=1` (`pokesim::driver_timing`), which the
  driver adds as a `timing_us` object and which is ABSENT — not zero — without the env var;
* PYTHON's phases are timed around the same call here (request `json.dumps` + write, the wait for
  the reply line, `json.loads`, and the dict walk that builds the `ExpandedNode` list);
* TRANSPORT is what is left: `wait - child_total`, i.e. the pipe and the scheduler.

It also censuses the reply BYTES by field, because "make the payload smaller" is only a candidate
if the payload is where the bytes are. The census is taken on the RAW reply line (not a
re-serialization), so it is the bytes that actually crossed the pipe.

🚨 The decision fixtures are `search_decision_benchmark.load_decision` — the same banked eval
traces, so this table and that one share a denominator.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import statistics
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))

from utils.contention import describe_contention  # noqa: E402


# ---------------------------------------------------------------------------
# the instrumented `_call`
# ---------------------------------------------------------------------------

ACC: Dict[str, float] = {}
CNT: Dict[str, int] = {}
BYTES: Dict[str, int] = {}
RUST: Dict[str, int] = {}
ARMS = [0]


def _add(k: str, v: float) -> None:
    ACC[k] = ACC.get(k, 0.0) + v
    CNT[k] = CNT.get(k, 0) + 1


def _census(line: str, obj: dict) -> None:
    """Reply bytes by field, measured on the RAW line.

    A field's share is taken as its re-serialized compact length over the sum of those lengths,
    scaled to the real line length — so the columns add to the bytes that crossed the pipe even
    though the re-serialization is not byte-identical to the driver's own rendering."""
    arms = obj.get("arms") or []
    per: Dict[str, int] = {}
    for a in arms:
        for k, v in a.items():
            per[k] = per.get(k, 0) + len(json.dumps(v, separators=(",", ":")))
    tot = sum(per.values()) or 1
    n = len(line)
    for k, v in per.items():
        BYTES[k] = BYTES.get(k, 0) + int(v / tot * n)
    BYTES["__line__"] = BYTES.get("__line__", 0) + n
    ARMS[0] += len(arms)


def install(SS) -> None:
    """Replace `SearchSession._call` with a timed twin (the body is the original's)."""

    def _call(self, payload: dict) -> dict:
        is_expand = payload.get("cmd") == "expand_many"
        if self._closed:
            raise RuntimeError("SearchSession is closed")
        self._seq += 1
        payload = {**payload, "id": self._seq}
        t0 = time.perf_counter()
        blob = json.dumps(payload) + "\n"
        self._proc.stdin.write(blob)
        self._proc.stdin.flush()
        t1 = time.perf_counter()
        try:
            line = self._q.get(timeout=self._timeout)
        except queue.Empty:
            self.close(kill=True)
            raise RuntimeError("search driver timed out")
        t2 = time.perf_counter()
        out = json.loads(line)
        t3 = time.perf_counter()
        if is_expand:
            _add("py: request dumps+write", (t1 - t0) * 1e3)
            _add("wait for reply (child + pipe)", (t2 - t1) * 1e3)
            _add("py: json.loads(reply)", (t3 - t2) * 1e3)
            _add("py: request bytes", 0.0)
            BYTES["__request__"] = BYTES.get("__request__", 0) + len(blob)
            _census(line, out)
            tm = out.get("timing_us") or {}
            for k, v in tm.items():
                RUST[k] = RUST.get(k, 0) + int(v)
            if not tm:
                RUST["__missing__"] = RUST.get("__missing__", 0) + 1
        if not out.get("ok"):
            raise RuntimeError(f"driver: {out.get('error')}")
        return out

    SS.SearchSession._call = _call

    orig_expand = SS.SearchSession.expand_many

    def expand_many(self, arms, **kw):
        t0 = time.perf_counter()
        res = orig_expand(self, arms, **kw)
        # The wrapper's OWN wall minus the three phases `_call` already billed IS the dict walk
        # that builds the ExpandedNode list.
        _add("expand_many wrapper (total)", (time.perf_counter() - t0) * 1e3)
        return res

    SS.SearchSession.expand_many = expand_many


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--decisions", type=int, default=12)
    ap.add_argument("--m-opp", type=int, default=3)
    ap.add_argument("--k-worlds", type=int, default=4)
    ap.add_argument("--n-actions", type=int, default=0)
    ap.add_argument("--arm", default="honest")
    ap.add_argument("--impl", default="rust")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--road", default="view")
    args = ap.parse_args()

    import random

    import utils.bridge.search_session as SS
    from main.search_dividend import search_decision_benchmark as B

    if os.environ.get("POKESIM_SEARCH_TIMING") != "1":
        print("⚠️  POKESIM_SEARCH_TIMING is not 1 — the child's own split will be ABSENT and the "
              "transport row unattributable. Set it.", file=sys.stderr)

    install(SS)

    stems = B.discover(args.traces)
    rng = random.Random(args.seed)
    rng.shuffle(stems)
    pool = B._pool() if args.arm == "honest" else []

    print(f"\nEXPAND_MANY PROFILE  (road={args.road}, {args.arm} arm, m_opp={args.m_opp}, "
          f"k_worlds={args.k_worlds})")
    print(f"  {describe_contention()}")
    print("  🚨 ratios are the claim.\n")

    walls: List[float] = []
    used = 0
    with B.instrumented() as PH:
        for stem in stems:
            if used >= args.decisions:
                break
            try:
                fx = B.load_decision(stem, args.impl, 0.55)
            except Exception as e:                                   # noqa: BLE001
                print(f"  skip {os.path.basename(stem)}: {type(e).__name__}: {e}")
                continue
            if fx is None:
                continue
            if args.n_actions:
                keep = sorted(fx[4])[:args.n_actions]
                fx = fx[:4] + ({k: fx[4][k] for k in keep},) + fx[5:]
                if not fx[4]:
                    continue
            PH.reset()
            try:
                res, ms = B.decide(args.road, fx, m_opp=args.m_opp, k_worlds=args.k_worlds,
                                   arm=args.arm, impl=args.impl, pool_packed=pool)
            except Exception as e:                                   # noqa: BLE001
                print(f"  skip {os.path.basename(stem)}: {type(e).__name__}: {e}")
                continue
            if int(res.widths.arms_scored) <= 0:
                continue
            used += 1
            walls.append(ms)

    if not used:
        print("no decisions ran", file=sys.stderr)
        return 2

    n_arms = ARMS[0] or 1
    dec_ms = statistics.mean(walls)
    print(f"\n{used} decisions, {ARMS[0]} expand_many arms, "
          f"{CNT.get('wait for reply (child + pipe)', 0)} batches")
    print(f"decision wall: {dec_ms:.1f} ms\n")

    wrapper = ACC.get("expand_many wrapper (total)", 0.0)
    dumps = ACC.get("py: request dumps+write", 0.0)
    wait = ACC.get("wait for reply (child + pipe)", 0.0)
    loads = ACC.get("py: json.loads(reply)", 0.0)
    walk = wrapper - dumps - wait - loads
    child = RUST.get("total", 0) / 1e3          # us -> ms
    transport = wait - child

    rows = [
        ("rust: sim (snapshot + resolve_turn)", RUST.get("sim", 0) / 1e3),
        ("rust: one_sided_view x2 (emit + JSON)", RUST.get("view", 0) / 1e3),
        ("rust: p1/p2 chunk arrays", RUST.get("chunks", 0) / 1e3),
        ("rust: outcome/requests/child/tail", RUST.get("render", 0) / 1e3),
        ("transport (pipe + scheduler)", transport),
        ("py: request dumps + write", dumps),
        ("py: json.loads(reply)", loads),
        ("py: ExpandedNode dict walk", walk),
    ]
    total = wrapper
    print(f"{'phase':<42} {'ms/arm':>9} {'% of expand_many':>18} {'% of decision':>15}")
    print("-" * 88)
    for name, v in rows:
        print(f"{name:<42} {v / n_arms:>9.4f} {100 * v / (total or 1):>17.1f}% "
              f"{100 * v / (dec_ms * used or 1):>14.1f}%")
    print("-" * 88)
    print(f"{'expand_many TOTAL':<42} {total / n_arms:>9.4f} {100.0:>17.1f}% "
          f"{100 * total / (dec_ms * used or 1):>14.1f}%")
    if RUST.get("__missing__"):
        print(f"\n⚠️  {RUST['__missing__']} batches carried NO timing_us — the rust rows above "
              f"are incomplete.")

    print("\nREPLY PAYLOAD CENSUS (bytes actually on the pipe)")
    line_b = BYTES.pop("__line__", 0)
    req_b = BYTES.pop("__request__", 0)
    print(f"  request lines: {req_b / 1e6:.2f} MB total, "
          f"{req_b / max(CNT.get('wait for reply (child + pipe)', 1), 1):,.0f} B/batch")
    print(f"  reply lines:   {line_b / 1e6:.2f} MB total, {line_b / n_arms:,.0f} B/arm")
    print(f"\n  {'field':<20} {'B/arm':>10} {'% of reply':>12}")
    for k, v in sorted(BYTES.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<20} {v / n_arms:>10,.0f} {100 * v / (line_b or 1):>11.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
