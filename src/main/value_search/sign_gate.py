"""THE S-U GATE TEST (``designs/research_state/measurements/su_gate_2026-10-27/``; the S-U review's "gate
test", ``su_headroom_2026-10-27/REVIEW.md``, 7aafa527): do S's DEVIATIONS from π beat chance on unfamiliar teams, under
S's own model of the opponent?

S is S-U's registered search (``r0s4@10``: 8 honest worlds, TRUNC-4-K8, piKL λ 0.015), run OFFLINE through the live
searched player's own ``LiveSearcher._search`` (its keying, its root guards, the lean world screen) at decisions of RAW
P* vs Kakuna games whose honest input logs the raw cells kept (``main.anchors --keep-input-logs``,
:mod:`main.anchors.input_log`). Four steps, each a verb:

* ``select``: per game (a deal), the :data:`PER_GAME` kept decisions (turns 10-30, >= 2 legal) with the smallest
  ``sha256("sugate_v1|<deal>|<turn>|<decision index>")`` (V-RANK's draw), from COUNTED units only; written once, frozen.
* ``score``: per selected state, the root replay is checked against the kept decision (tokens, and the live row's
  sha256), then S runs; where S's pick differs from π's top-1 (a DEVIATION) the SELF-REFEREE plays the TWO actions to the
  end in the TRUE world (``playout.referee``: 24 truth seeds, P* greedy on both sides, the dice common across the pair),
  T(a) = the mean outcome (win 1, tie ½, loss 0). Also per state: the exclusion, the CPU split, the leak block and the
  Speed ±1 counterfactual-flip count of the seen mons (``main.info_audit.checks.seed_reuse``, ``search_ledger.md`` §6.1).
  Resumable: a state already scored is skipped.
* ``read``: SIGN ACCURACY = P(T(pick) > T(π top) | S deviates, the two T differ), per panel and pooled over the U panels
  (equal weight), with a game-clustered percentile bootstrap (one ``default_rng(seed)``, B draws, panels resampled in a
  fixed order). GO iff LB(pooled U) > 0.50 (a bound within ``rule8_eps`` of the bar reads STOP, flagged); else STOP.
  Reported, never decided: home72's accuracy, U - home72, IMP, deviation and exclusion rates, CPU, the leak exposure.

The estimator is :func:`cluster_ratio_boot`; with one panel it is exactly the review's ``boot`` (the same draw order),
which is how the gate's GIGO tooth re-scores the review's bank rows to its printed digits
(``su_gate_2026-10-27/scripts/bank_check.py``).

    python -m main.value_search.sign_gate select --state DIR --group G --label L --out selected.jsonl
    python -m main.value_search.sign_gate score  --selected selected.jsonl --model ZIP --out scored.jsonl [--limit N]
        [--budget-s S]
    python -m main.value_search.sign_gate read   --scored scored.jsonl --params params.json [--out verdict.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SELECT_SALT = "sugate_v1"
PER_GAME = 2
WINDOW = (10, 30)
REFEREE_SEEDS = 24
#: the referee's seeds: the maturity battery's layout (``truth.turn_seeds(id, 48)``: leaf 0..15, referee 16..39), keyed
#: by the decision's live search key, so they never coincide with the search's own dice (``search.seeds_for``)
REFEREE_SLICE = (16, 40)
STATE_SCHEMA = "gen3_sign_gate_state_v1"
SELECT_SCHEMA = "gen3_sign_gate_selected_v1"
VERDICT_SCHEMA = "gen3_sign_gate_verdict_v1"
OUTCOME = {"+": 1.0, "0": 0.5, "-": 0.0}


class GateError(RuntimeError):
    """A state that cannot be scored honestly (a kept log that does not replay to the decision we played, a forward
    that disagrees with the greedy choice we played): named, never skipped."""


# --------------------------------------------------------------------------------------------- the estimator
def cluster_ratio_boot(groups: Mapping[str, Mapping[str, Sequence[float]]], panels: Sequence[str], *, B: int,
                       seed: int) -> Dict[str, Any]:
    """The pooled ratio estimator over ``panels`` (equal weight) and each panel's, with a cluster bootstrap.

    ``groups[panel][cluster]`` = the cluster's per-state values (NaN = not countable, dropped; a cluster with none still
    takes part in the resampling, as the review's ``boot`` did). A panel's estimate is Σ values / n over its clusters;
    the pooled one is the mean of the panels'. Replicate b: for each panel in ``panels`` order, ``rng.integers(0, K, K)``
    clusters with replacement. Returns ``{"est", "lo", "hi", "n", "per_panel": {p: [est, n, K]}}`` (percentiles 2.5 /
    97.5). With ONE panel this is the review's ``boot`` draw for draw."""
    rng = np.random.default_rng(seed)
    sums, ns = {}, {}
    for p in panels:
        keys = sorted(groups.get(p, {}))
        arrs = [np.asarray([v for v in groups[p][k] if not np.isnan(v)], dtype=float) for k in keys]
        sums[p] = np.array([a.sum() for a in arrs])
        ns[p] = np.array([len(a) for a in arrs])

    def est(p: str, i: Optional[np.ndarray] = None) -> float:
        s, n = (sums[p], ns[p]) if i is None else (sums[p][i], ns[p][i])
        return float(s.sum() / max(n.sum(), 1))

    point = float(np.mean([est(p) for p in panels]))
    bs = []
    for _ in range(int(B)):
        vals = []
        for p in panels:
            k = len(sums[p])
            vals.append(est(p, rng.integers(0, k, k)) if k else 0.0)
        bs.append(float(np.mean(vals)))
    return {"est": point, "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "n": int(sum(int(ns[p].sum()) for p in panels)),
            "per_panel": {p: [est(p), int(ns[p].sum()), int(len(sums[p]))] for p in panels}}


def pikl_pick(q: Sequence[float], logits: Sequence[float], lam: float) -> int:
    """The registered acting rule on POLICY-ORDERED arrays (index 0 = π's top): ``main.value_search.rules``' piKL pick,
    the one the live search plays (``SearchConfig(rule="pikl")``)."""
    from main.value_search import rules

    lg = np.asarray(logits, dtype=np.float64)
    legal = list(range(len(lg)))
    order = rules.policy_order(lg, legal)
    return int(rules.pick(rules.pikl_scores(lg, {a: float(q[a]) for a in legal}, lam, order), order))


def sign_value(t_pick: float, t_top: float) -> float:
    """1 if the deviation is right under T, 0 if wrong, NaN if the two T are equal (not countable)."""
    return float("nan") if t_pick == t_top else float(t_pick > t_top)


# ---------------------------------------------------------------------------------------------- the selection
def select_hash(deal: str, turn: int, n: int) -> str:
    return hashlib.sha256(f"{SELECT_SALT}|{deal}|{int(turn)}|{int(n)}".encode()).hexdigest()


def select(records: Iterable[Mapping[str, Any]], per_game: int = PER_GAME,
           window: Tuple[int, int] = WINDOW) -> List[Dict[str, Any]]:
    """Per game (deal), the ``per_game`` eligible decisions with the smallest :func:`select_hash`, in (deal, hash)
    order. A record carries ``panel`` and ``cell``; a duplicate (deal, turn, n) is a defect."""
    by: Dict[str, Dict[Tuple[int, int], Dict[str, Any]]] = {}
    for r in records:
        if not (window[0] <= int(r["turn"]) <= window[1]) or int(r["n_legal"]) < 2:
            continue
        k = (int(r["turn"]), int(r["n"]))
        g = by.setdefault(str(r["deal"]), {})
        if k in g:
            raise GateError(f"deal {r['deal']}: decision {k} kept twice")
        g[k] = dict(r)
    out = []
    for deal in sorted(by):
        rows = sorted(by[deal].values(), key=lambda r: select_hash(deal, r["turn"], r["n"]))
        for r in rows[:per_game]:
            r["select_hash"] = select_hash(deal, r["turn"], r["n"])
            out.append(r)
    return out


def counted_cells(state: Path, group: str, label: str) -> List[Dict[str, Any]]:
    """``[{cell, panel, unit_seed, dir}]`` for every COUNTED unit of ``label`` in anchor group ``group`` (the last
    ``end`` event of each unit in ``units.jsonl``; the plan names its team set)."""
    base = Path(state) / "anchor" / group / label
    plan = {u["name"]: u for u in json.loads((base / "plan.json").read_text())["units"]}
    last: Dict[str, Dict[str, Any]] = {}
    for ln in (base / "units.jsonl").read_text().splitlines():
        e = json.loads(ln)
        if e.get("event") == "end":
            last[e["unit"]] = e
    out = []
    for name in sorted(plan):
        e = last.get(name)
        if e is not None and e.get("counted") is True:
            out.append({"cell": name, "panel": plan[name]["teamset"], "unit_seed": int(plan[name]["unit_seed"]),
                        "dir": str(base / "cells" / name)})
    return out


def kept_records(cells: Sequence[Mapping[str, Any]]) -> Iterable[Dict[str, Any]]:
    from main.anchors.input_log import SCHEMA
    from main.anchors.runner import INPUT_LOGS_NAME

    for c in cells:
        p = Path(c["dir"]) / INPUT_LOGS_NAME
        if not p.exists():
            raise GateError(f"{c['cell']}: counted, but it kept no {INPUT_LOGS_NAME}")
        for ln in p.read_text().splitlines():
            r = json.loads(ln)
            if r.get("schema") != SCHEMA:
                raise GateError(f"{c['cell']}: a kept record of schema {r.get('schema')!r}")
            yield {**r, "panel": c["panel"], "cell": c["cell"], "unit_seed": c["unit_seed"]}


# ---------------------------------------------------------------------------------------------- the scorer
def referee_seeds(key: str) -> List[str]:
    from main.policy_spectrum.truth import turn_seeds

    return turn_seeds(key, REFEREE_SLICE[1])[REFEREE_SLICE[0]:REFEREE_SLICE[1]]


def _children_cpu() -> float:
    import resource

    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


class GateScorer:
    """S (the live searcher's ``_search`` on a P* checkpoint, greedy) and the self-referee on the same forward."""

    def __init__(self, model_zip: str, search: str = "r0s4@10", *, searcher: Any = None) -> None:
        if searcher is None:
            from main.anchors.core_side import CorePolicy, load_policy
            from main.anchors.search_side import LiveSearcher
            from main.anchors.session import OurSideState

            model = load_policy(str(model_zip), "cpu", "bare", OurSideState())
            policy = CorePolicy(model, stochastic=False, temperature=1.0)
            searcher = LiveSearcher(model, policy, search, Path("/dev/null"))
        self.s = searcher

    def close(self) -> None:
        self.s.close()

    def root(self, rec: Mapping[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """The TRUE root replayed from the kept log, and the decision's frame dict; refuses (named) unless the replay
        reproduces the kept decision's tokens, turn, index and live row (its sha256)."""
        from agents.battle.core_obs import DTYPE, obs_dim
        from main.anchors.input_log import frame_sha256
        from main.value_search import worlds as W

        log, at, side, key = rec["log"], int(rec["at"]), rec["side"], rec["key"]
        root = self.s.engine.root(log, at, side)
        got = {str(k): str(v) for k, v in (root.get("tokens") or {}).items()}
        if got != {str(k): str(v) for k, v in rec["tokens"].items()}:
            raise GateError(f"{key}: the kept log's replay offers {got}, the kept decision {rec['tokens']}")
        truth = W.prepare_root(log, at, side, key, root_choice_in_log=False)
        if not truth.ref or frame_sha256(truth.ref[-1]) != rec["frame_sha256"]:
            raise GateError(f"{key}: the kept log's replayed row is not the live row we played on")
        d = {"turn": int(rec["turn"]), "n": int(rec["n"]), "tokens": dict(rec["tokens"]),
             "frame": {"dtype": DTYPE, "shape": [obs_dim()], "b64": truth.ref[-1]}}
        return truth, d

    def score(self, rec: Mapping[str, Any], *, flips: bool = True) -> Dict[str, Any]:
        from agents.battle.core_obs import wrap_row
        from main.value_search.playout import referee

        t0 = time.monotonic()
        truth, d = self.root(rec)
        row = np.array(wrap_row(d["frame"]), dtype=np.float32)
        mask = np.asarray(rec["mask"], dtype=np.int8)
        argmax = int(rec["pi_argmax"])
        s = self.s._search(rec["log"], int(rec["at"]), rec["side"], row, mask, d, rec["key"], argmax)
        out: Dict[str, Any] = {k: rec[k] for k in ("panel", "cell", "unit_seed", "deal", "battle", "turn", "n",
                                                   "n_legal", "key", "select_hash")}
        out.update(schema=STATE_SCHEMA, pi_argmax=argmax, search=s)
        if s.get("searched") and s.get("deviation"):
            c0, k0 = time.thread_time(), _children_cpu()
            pick = int(s["played"])
            res = referee(self.s.forward, rec["log"], int(rec["at"]), rec["side"], referee_seeds(rec["key"]),
                          actions=sorted({argmax, pick}))
            tv = {a: float(np.mean([OUTCOME[ch] for ch in res.out[a]])) for a in (argmax, pick)}
            out["referee"] = {"out": {str(a): res.out[a] for a in sorted(res.out)}, "T_pick": tv[pick],
                              "T_top": tv[argmax], "sign": sign_value(tv[pick], tv[argmax]),
                              "cpu_s": round(time.thread_time() - c0 + _children_cpu() - k0, 3)}
        if flips:
            from main.info_audit.checks import seed_reuse

            c0, k0 = time.thread_time(), _children_cpu()
            fl = seed_reuse(truth)
            out["speed_flip"] = {"ok": bool(fl["pass"]), "speed_mons": int(fl["speed_mons"]),
                                 "speed_pinned": int(fl["speed_pinned"]),
                                 "cpu_s": round(time.thread_time() - c0 + _children_cpu() - k0, 3)}
        out["wall_s"] = round(time.monotonic() - t0, 3)
        return out


# ---------------------------------------------------------------------------------------------- the read
def state_values(r: Mapping[str, Any]) -> Dict[str, float]:
    """One scored state's per-state values: ``searched`` (0 / 1), ``excluded``, ``dev`` (searched and deviating),
    ``sign`` (NaN unless a countable deviation), ``imp`` (T(pick) - T(top); 0 where S agrees with π; NaN where not
    searched)."""
    s = r["search"]
    searched = bool(s.get("searched"))
    dev = searched and bool(s.get("deviation"))
    ref = r.get("referee")
    if dev and ref is None:
        raise GateError(f"{r['key']}: a deviation with no referee row")
    return {"searched": float(searched), "excluded": float(s.get("why") == "excluded"),
            "dev": float(dev) if searched else float("nan"),
            "sign": float(ref["sign"]) if dev else float("nan"),
            "imp": (float(ref["T_pick"] - ref["T_top"]) if dev else 0.0) if searched else float("nan")}


def groups_of(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Dict[str, List[float]]]:
    g: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        g.setdefault(r["panel"], {}).setdefault(r["deal"], []).append(state_values(r)[field])
    return g


def read(rows: Sequence[Mapping[str, Any]], params: Mapping[str, Any]) -> Dict[str, Any]:
    """The registered decision (module docstring) and the reported rows. ``params``: ``home_panel``, ``u_panels``,
    ``bar``, ``rule8_eps``, ``bootstrap`` {B, seed}, ``expected_states`` (the selection's count, so a short score file
    is refused)."""
    u, home = list(params["u_panels"]), str(params["home_panel"])
    B, seed = int(params["bootstrap"]["B"]), int(params["bootstrap"]["seed"])
    keys = [r["key"] for r in rows]
    if len(set(keys)) != len(keys):
        raise GateError("a state is scored twice")
    if params.get("expected_states") is not None and len(rows) != int(params["expected_states"]):
        raise GateError(f"{len(rows)} scored states, the selection holds {params['expected_states']}")
    sg = groups_of(rows, "sign")
    primary = cluster_ratio_boot(sg, u, B=B, seed=seed)
    bar, eps = float(params["bar"]), float(params["rule8_eps"])
    boundary = abs(primary["lo"] - bar) <= eps
    verdict = "GO" if primary["lo"] > bar + eps else "STOP"
    rep: Dict[str, Any] = {"home72_sign": cluster_ratio_boot(sg, [home], B=B, seed=seed)}
    # U - home72, sign only: one replicate draws the U panels then home72 from ONE stream
    rng = np.random.default_rng(seed)
    diff = []
    for _ in range(B):
        vals = {}
        for p in u + [home]:
            ks = sorted(sg.get(p, {}))
            arrs = [np.asarray([v for v in sg[p][k] if not np.isnan(v)]) for k in ks]
            i = rng.integers(0, len(ks), len(ks))
            n = sum(len(arrs[j]) for j in i)
            vals[p] = sum(float(arrs[j].sum()) for j in i) / max(n, 1)
        diff.append(float(np.mean([vals[p] for p in u])) - vals[home])
    pe = primary["est"] - rep["home72_sign"]["est"]
    rep["u_minus_home72_sign"] = {"est": pe, "lo": float(np.percentile(diff, 2.5)),
                                  "hi": float(np.percentile(diff, 97.5)), "sign": "+" if pe > 0 else "-" if pe < 0 else "0"}
    per_set: Dict[str, Any] = {}
    for p in u + [home]:
        pr = [r for r in rows if r["panel"] == p]
        vs = [state_values(r) for r in pr]
        srch = [r for r, v in zip(pr, vs) if v["searched"]]
        cpu = [float(r["search"].get("cpu_s", 0.0)) for r in pr]
        ref_cpu = [float(r["referee"]["cpu_s"]) for r in pr if r.get("referee")]
        fl = [r["speed_flip"] for r in pr if r.get("speed_flip")]
        leak = [r["search"].get("leak") or {} for r in srch]
        per_set[p] = {
            "states": len(pr), "games": len({r["deal"] for r in pr}),
            "excluded": int(sum(v["excluded"] for v in vs)),
            "exclusion_rate": float(np.mean([v["excluded"] for v in vs])) if vs else float("nan"),
            "searched": len(srch), "deviations": int(sum(v["dev"] for v in vs if v["searched"])),
            "deviation_rate": cluster_ratio_boot(groups_of(pr, "dev"), [p], B=B, seed=seed),
            "countable_deviations": int(sum(1 for v in vs if not np.isnan(v["sign"]))),
            "sign": cluster_ratio_boot(groups_of(pr, "sign"), [p], B=B, seed=seed),
            "imp": cluster_ratio_boot(groups_of(pr, "imp"), [p], B=B, seed=seed),
            "search_cpu_s": {"median": float(np.median(cpu)) if cpu else None, "mean": float(np.mean(cpu)) if cpu else None,
                             "total": float(np.sum(cpu))},
            "referee_cpu_s": {"mean": float(np.mean(ref_cpu)) if ref_cpu else None, "total": float(np.sum(ref_cpu))},
            "leak": {"rewritten_moves": int(sum(int(x.get("rewritten_moves", 0)) for x in leak)),
                     "rewritten_switches": int(sum(int(x.get("rewritten_switches", 0)) for x in leak)),
                     "attribution_miss": int(sum(int(x.get("attribution_miss", 0)) for x in leak)),
                     "states_with_rewrites": int(sum(1 for x in leak if x.get("rewritten_moves") or x.get("rewritten_switches")))},
            "speed_flip": {"states": len(fl), "states_pinned": int(sum(1 for x in fl if x["speed_pinned"] > 0)),
                           "rate": float(np.mean([x["speed_pinned"] > 0 for x in fl])) if fl else None,
                           "mons": int(sum(x["speed_mons"] for x in fl)), "mons_pinned": int(sum(x["speed_pinned"] for x in fl)),
                           "control_failed": int(sum(1 for x in fl if not x["ok"]))},
        }
    rep["per_set"] = per_set
    return {"schema": VERDICT_SCHEMA, "verdict": verdict, "boundary": boundary,
            "rule": f"GO iff LB(sign accuracy, pooled U {u}, panels equal-weighted) > {bar} (rule 8: within {eps} = STOP)",
            "primary": primary, "reported": rep, "n_states": len(rows), "params": dict(params)}


# ---------------------------------------------------------------------------------------------- the CLI
def _jsonl(p: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.value_search.sign_gate", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="verb", required=True)
    a = sub.add_parser("select")
    a.add_argument("--state", required=True)
    a.add_argument("--group", required=True)
    a.add_argument("--label", required=True)
    a.add_argument("--out", required=True)
    b = sub.add_parser("score")
    b.add_argument("--selected", required=True)
    b.add_argument("--model", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--search", default="r0s4@10")
    b.add_argument("--budget-s", type=float, default=None,
                   help="start no new state after this many seconds (a bounded chunk ends cleanly; resumable)")
    c = sub.add_parser("read")
    c.add_argument("--scored", required=True)
    c.add_argument("--params", required=True)
    c.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.verb == "select":
        out = Path(args.out)
        if out.exists():
            raise SystemExit(f"{out}: a selection is written once (frozen)")
        cells = counted_cells(Path(args.state), args.group, args.label)
        sel = select(kept_records(cells))
        with open(out, "w") as f:
            for r in sel:
                f.write(json.dumps({**r, "schema": SELECT_SCHEMA}, sort_keys=True) + "\n")
        by = {}
        for r in sel:
            by[r["panel"]] = by.get(r["panel"], 0) + 1
        print(json.dumps({"cells": len(cells), "states": len(sel), "per_panel": by}))
        return 0
    if args.verb == "score":
        sel = _jsonl(Path(args.selected))
        done = {r["key"] for r in _jsonl(Path(args.out))}
        todo = [r for r in sel if r["key"] not in done][: args.limit]
        if not todo:
            print(json.dumps({"scored": len(done), "left": 0}))
            return 0
        t0 = time.monotonic()
        sc = GateScorer(args.model, args.search)
        try:
            for i, r in enumerate(todo):
                if args.budget_s is not None and time.monotonic() - t0 > args.budget_s:
                    print(f"[sign_gate] budget {args.budget_s:g} s spent: {i} scored this chunk", flush=True)
                    break
                rec = sc.score(r)
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
                    f.flush()
                # progress only: no outcome is printed while scoring
                print(f"[sign_gate] {len(done) + i + 1}/{len(sel)} {r['panel']} cpu {rec['search'].get('cpu_s')}",
                      flush=True)
        finally:
            sc.close()
        return 0
    rows = _jsonl(Path(args.scored))
    res = read(rows, json.loads(Path(args.params).read_text()))
    txt = json.dumps(res, indent=1, sort_keys=True, default=str)
    if args.out:
        Path(args.out).write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
