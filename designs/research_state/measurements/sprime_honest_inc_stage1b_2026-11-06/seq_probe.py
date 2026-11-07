"""S'-HONEST stage 1b (2026-11-06, OUTCOME-BLIND): the INCREMENTAL builder `r0s4_honest_inc@10` on WHOLE-GAME decision
sequences, against `r0s4_honest@10` at the same states (`designs/endstate/design_sprime_honest.md` §9).

THE STATES (the targets), never a referee field, an outcome, a game result or a summary:
* `stage1`: stage 1's 300 dlive U states (`sprime_honest_stage1_2026-11-05/timing_v2_rows.jsonl.gz` keys), their kept
  logs from dlive's `selected.jsonl`;
* `late`: the replication designer's late probe (144 S'-LIVE states, every turn bucket; `late_probe_rows.jsonl.gz`
  keys), their records from S'-LIVE's re-score sample (`rescore_sample: true` lines of complete attempt dirs'
  `search_decisions.jsonl`, which keep the honest log, mask and tokens).

A GROUP is one battle's side (the record's `deal` and `side`). Its decision SEQUENCE is every decision of that side the
live searcher searches, from turn 10 up to the group's latest target (`honest_inv.sequence` on that target's own log:
each decision cut from it, its root checked against the playout tool's). It runs in order on ONE live searcher (P*
greedy on CPU, one thread; the preset's carry cache): at a TARGET the live searcher's own `_search` on the target's own
record behind `sign_gate.GateScorer`'s root guards (the late probe's path), every field kept; at every other decision
the BUILDER STEP alone (`info_set.extract_points` and the sampler's `sample_info`: what the live decision costs but its
forward and playouts), its CPU (`search.cpu_all`: every thread and the reaped children) plus the validator child's.

`--preset r0s4_honest@10 --targets-only` runs the cold builder at the targets alone (no sequence: it carries nothing).
`--keys F` restricts the targets (a replicate's subset: `floor_keys.txt`, stage 1's first 100 states in its own order,
the noise floor's, and the late-probe states up to turn 63).
`--salt S` replaces the builder's key tag (`honest_worlds.KEY_TAG`) by another, a REPLICATE whose every world draw and
fresh die differ (the playouts' dice are keyed by I and stay the same): the noise floor. Resumable: a group whose rows
are complete is skipped, and a group cut by a chunk's bound resumes at its first unwritten decision from the carry the
last written one returned (saved per group: the same bytes a rebuild gives); `--shard i/k` runs the i-th of k
interleaved slices of the groups.

    PYTHONPATH=src python3 seq_probe.py run --model ZIP --model-sha256 H --out DIR [--preset P] [--salt S]
        [--targets-only] [--shard i/k] [--sets stage1,late]
    PYTHONPATH=src python3 seq_probe.py summarize --out DIR ...
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE1 = HERE.parent / "sprime_honest_stage1_2026-11-05"
LATE_ROWS = Path("/home/goodlad/gen3ai_archive/sprime_honest_play_inputs_2026-11-06/late_probe/rows.jsonl")
SLIVE_CELLS = Path("/home/goodlad/gen3ai_archive/sprime_live_play_2026-11-01/anchor/u/pstar_s/cells")
DLIVE_SELECTED = Path("/home/goodlad/gen3ai_archive/dlive_read_2026-10-28/live/selected.jsonl")
PANELS = {"heldout72_hpfix": "H", "forum72": "F", "p72": "P72"}
BUCKETS = (("10-19", 10, 19), ("20-29", 20, 29), ("30-63", 30, 63), ("64+", 64, 10 ** 6))
FIELDS = ("searched", "why", "error", "played", "deviation", "cpu_s", "world_cpu_s", "playout_cpu_s", "wall_s", "q",
          "log_pi", "root_v", "spent", "world_tries", "search_key")


def bucket(turn):
    return next(b for b, lo, hi in BUCKETS if lo <= turn <= hi)


def validator_cpu():
    """The CPU-s of the process's shared Showdown validator child so far (stage 1's FINDING S'H-9)."""
    from main.value_search import world_legality as WL

    v = WL._SHARED.get(WL.FORMAT)
    p = getattr(v, "_proc", None)
    if p is None or p.poll() is not None:
        return 0.0
    try:
        f = open(f"/proc/{p.pid}/stat").read().rsplit(")", 1)[1].split()
    except OSError:
        return 0.0
    return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")


def targets(sets):
    """Every target record: (set, panel letter, record) with the record's own log, at, side, turn, n, mask, tokens,
    frame digest and key."""
    out = []
    if "stage1" in sets:
        keys = {json.loads(x)["key"]: json.loads(x)["panel"] for x in gzip.open(STAGE1 / "timing_v2_rows.jsonl.gz", "rt")}
        for x in open(DLIVE_SELECTED):
            r = json.loads(x)
            if r["key"] in keys:
                out.append(("stage1", keys[r["key"]], r))
    if "late" in sets:
        keys = {json.loads(x)["key"]: json.loads(x)["panel"] for x in open(LATE_ROWS)}
        for d in sorted(os.listdir(SLIVE_CELLS)):
            if ".partial." in d or d.split("__")[1] not in PANELS:
                continue
            p = SLIVE_CELLS / d / "search_decisions.jsonl"
            if not p.is_file():
                continue
            for x in open(p):
                r = json.loads(x)
                if r.get("rescore_sample") is True and "log" in r and r["key"] in keys:
                    out.append(("late", keys[r["key"]], r))
    seen, uniq = set(), []
    for s, p, r in out:
        if (s, r["key"]) not in seen:
            seen.add((s, r["key"]))
            uniq.append((s, p, r))
    return uniq


def groups(tg):
    by = {}
    for s, p, r in tg:
        by.setdefault((s, r["deal"], r["side"]), []).append((p, r))
    return [(k, sorted(v, key=lambda x: int(x[1]["at"]))) for k, v in sorted(by.items())]


def run(a):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch

    torch.set_num_threads(1)
    from agents.battle.core_obs import wrap_row
    from main.anchors.core_side import CorePolicy, load_policy
    from main.anchors.search_side import LiveSearcher
    from main.anchors.server import engine_stamp
    from main.anchors.session import OurSideState
    from main.value_search import honest_inv as INV
    from main.value_search import honest_worlds as HW
    from main.value_search import info_set as IS
    from main.value_search.search import cpu_all
    from main.value_search.sign_gate import GateScorer
    from main.value_search.world_draws import WorldError

    if a.salt:
        HW.KEY_TAG = f"sprime_honest_v1:replicate:{a.salt}"   # a REPLICATE: every world draw and fresh die moved
    if hashlib.sha256(Path(a.model).read_bytes()).hexdigest() != a.model_sha256:
        sys.exit("the model's bytes are not the registered ones")
    model = load_policy(str(a.model), "cpu", "bare", OurSideState())
    policy = CorePolicy(model, stochastic=False, temperature=1.0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_f = out / "rows.jsonl"
    done, done_n = set(), {}
    if rows_f.exists():
        for x in rows_f.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                if r.get("group_done"):
                    done.add(tuple(r["group"]))
                elif "n" in r:
                    done_n.setdefault(tuple(r["group"]), set()).add(int(r["n"]))
    carries = out / "carries"
    carries.mkdir(exist_ok=True)
    i, k = (int(x) for x in a.shard.split("/"))
    tg = targets(a.sets.split(","))
    if a.keys:
        want = {x.strip() for x in open(a.keys) if x.strip()}
        tg = [t for t in tg if t[2]["key"] in want]
    gs = [g for j, g in enumerate(groups(tg)) if j % k == i]
    eng = engine_stamp()
    for gkey, tg in gs:
        if tuple(gkey) in done:
            continue
        s = LiveSearcher(model, policy, a.preset, Path("/dev/null"))
        guard = GateScorer(a.model, searcher=s)
        by_n = {int(r["n"]): (p, r) for p, r in tg}
        last = tg[-1][1]
        g0 = time.monotonic()
        try:
            if a.targets_only:
                seq = [INV.decision_of(r) for _p, r in tg]
            else:
                seq = INV.sequence(INV.decision_of(last), 10, s.engine)
            missing = sorted(set(by_n) - {x.n for x in seq})
            if missing:
                raise ValueError(f"targets {missing} are not in the sequence")
        except ValueError as exc:                    # a battle whose sequence cannot be cut: a named FINDING row
            s.close()
            with open(rows_f, "a") as f:
                f.write(json.dumps({"group": list(gkey), "group_done": True, "group_error": str(exc)[:400]},
                                   sort_keys=True) + "\n")
            continue
        # RESUME mid-battle: the decisions already written are skipped, the carry the last of them returned is seeded
        # (the same bytes a rebuild would give: `honest_inc`'s cache semantics)
        cfile = carries / (hashlib.sha256(json.dumps(list(gkey)).encode()).hexdigest()[:24] + ".json.gz")
        have = done_n.get(tuple(gkey), set())
        if have and cfile.exists() and hasattr(s.sampler, "seed"):
            s.sampler.seed(gzip.decompress(cfile.read_bytes()).decode())
        lines = []
        for x in seq:
            if x.n in have:
                continue
            v0, c0, w0 = validator_cpu(), cpu_all(), time.monotonic()
            row = {"group": list(gkey), "turn": x.turn, "n": x.n, "bucket": bucket(x.turn), "engine": eng,
                   "preset": a.preset, "salt": a.salt}
            if x.n in by_n:
                p, rec = by_n[x.n]
                if int(rec["turn"]) != x.turn:
                    raise SystemExit(f"{rec['key']}: the sequence's decision {x.n} is at turn {x.turn}")
                _truth, d = guard.root(rec)
                r = s._search(rec["log"], int(rec["at"]), rec["side"], np.array(wrap_row(d["frame"]), dtype=np.float32),
                              np.asarray(rec["mask"], dtype=np.int8), d, rec["key"], int(rec["pi_argmax"]))
                row.update(target=True, key=rec["key"], panel=p, n_legal=int(rec["n_legal"]),
                           pi_argmax=int(rec["pi_argmax"]), res={kk: r.get(kk) for kk in FIELDS})
                h = r.get("honest") or {}
                row["inc"] = h.get("inc")
                row["chain"] = h.get("chain")
                row["fails"] = h.get("fails")
                row["kinds"] = [w.get("kind") for w in h.get("worlds") or []]
            else:
                info, frame, pts = IS.extract_points(x.log, x.at, x.side)
                try:
                    w = s.sampler.sample_info(info, s.cfg.n_worlds, frame, points=pts)
                    res = {"searched": True, "why": None}
                    inc, chain, fails, kinds = w.inc, w.chain, w.fails, [z.get("kind") for z in w.stats]
                except WorldError as e:
                    res = {"searched": False, "why": "excluded", "error": str(e)[:300]}
                    inc, chain, fails, kinds = getattr(e, "inc", None), None, None, None
                row.update(target=False, panel=tg[0][0], res=res, inc=inc, chain=chain, fails=fails, kinds=kinds)
                row["res"]["world_cpu_s"] = round(cpu_all() - c0, 3)
            row["cpu_all_s"] = round(cpu_all() - c0, 3)
            row["validator_cpu_s"] = round(validator_cpu() - v0, 3)
            row["wall_s"] = round(time.monotonic() - w0, 3)
            row["rebuilt"] = getattr(s.sampler, "last_rebuilt", None)
            row["phases"] = dict(getattr(s.sampler, "last_phases", None) or {})
            with open(rows_f, "a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            store = getattr(s.sampler, "store", None)
            if store:
                cfile.write_bytes(gzip.compress(store[next(reversed(store))].encode(), mtime=0))
            lines.append(row)
            print(json.dumps({"g": gkey[1][:8], "t": x.turn, "n": x.n, "tgt": row["target"],
                              "s": row["res"].get("searched"), "cpu": row["cpu_all_s"],
                              "inc": (row.get("inc") or {}).get("extended")}), flush=True)
        s.close()
        with open(rows_f, "a") as f:
            f.write(json.dumps({"group": list(gkey), "group_done": True, "decisions": len(seq),
                                "wall_s": round(time.monotonic() - g0, 3)}, sort_keys=True) + "\n")


COST_MODEL = Path("/home/goodlad/dev/gen3ai-worktrees/sprime_honest_prereg/designs/research_state/measurements/"
                  "sprime_honest_play_2026-11-06/inputs/cost_model.json")


def _rows(d):
    p = Path(d) / "rows.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def honest_reference():
    """r0s4_honest@10's record at every target (OUTCOME-BLIND): stage 1's at budget B3 (its post-fix run, the 44
    limited states' B3 re-runs over it) and the late probe's; per key, (searched, why, played, CPU incl. validator)."""
    ref = {}
    for f in ("timing_v2_rows.jsonl.gz", "curve_b3_rows.jsonl.gz"):
        for x in gzip.open(STAGE1 / f, "rt"):
            r = json.loads(x)
            h = r["res"]["r0s4_honest@10"]
            ref[r["key"]] = {"searched": h["searched"], "why": h["why"], "played": h["played"],
                             "cpu": h["cpu_s"] + (h.get("validator_cpu_s") or 0.0)}
    for x in open(LATE_ROWS):
        r = json.loads(x)
        h = r["res"]["r0s4_honest@10"]
        ref[r["key"]] = {"searched": h["searched"], "why": h["why"], "played": h["played"],
                         "cpu": h["cpu_s"] + (h.get("validator_cpu_s") or 0.0)}
    return ref


def summarize(a):
    import statistics as S

    rows = [r for r in _rows(a.out) if "group_done" not in r]
    errs = [r for r in _rows(a.out) if r.get("group_error")]
    cm = json.load(open(COST_MODEL))
    ko, slive = cm["inputs"]["kappa_overhead"], cm["inputs"]["slive"]
    ref = honest_reference()
    reps = {name: {r["key"]: r for r in _rows(d) if r.get("target")} for name, d in
            (("inc_rep", a.inc_rep), ("honest_rep", a.honest_rep)) if d}

    def q(xs, f):
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(f * len(xs)))], 2) if xs else None

    # the forward and playouts of a searched decision (preset-independent): per bucket over every panel's searched
    # targets, the fallback for a panel's bucket with none
    other_b = {b: [r["res"]["cpu_s"] - r["res"]["world_cpu_s"] for r in rows
                   if r["target"] and r["bucket"] == b and r["res"].get("searched")] for b, _lo, _hi in BUCKETS}
    out = {"engine": sorted({r["engine"] for r in rows}), "decisions": len(rows),
           "targets": sum(1 for r in rows if r["target"]), "groups": len({tuple(r["group"]) for r in rows}),
           "group_errors": [e["group_error"] for e in errs], "by": {}, "panels": {}}
    for panel in ("H", "F", "P72"):
        cost_b = {}
        for b, _lo, _hi in BUCKETS:
            rs = [r for r in rows if r["panel"] == panel and r["bucket"] == b]
            if not rs:
                continue
            world = [r["res"]["world_cpu_s"] + r["validator_cpu_s"] for r in rs if not r["target"]]
            tg = [r for r in rs if r["target"]]
            other = [r["res"]["cpu_s"] - r["res"]["world_cpu_s"] for r in tg if r["res"].get("searched")]
            excl = [r for r in rs if r["res"].get("why") == "excluded"]
            tcpu = [r["res"]["cpu_s"] + r["validator_cpu_s"] for r in tg]
            hcpu = [ref[r["key"]]["cpu"] for r in tg if r["key"] in ref]
            e = {"decisions": len(rs), "world_cpu_s": {"mean": round(S.mean(world), 2) if world else None,
                                                        "median": q(world, 0.5), "p90": q(world, 0.9),
                                                        "max": q(world, 1.0)},
                 "other_cpu_s_mean": round(S.mean(other), 2) if other else None,
                 "excluded": len(excl), "excl_rate": round(len(excl) / len(rs), 4),
                 "targets": len(tg), "targets_excluded": sum(1 for r in tg if r["res"].get("why") == "excluded"),
                 "honest_targets_excluded": sum(1 for r in tg if ref.get(r["key"], {}).get("why") == "excluded"),
                 "target_cpu_s_mean": round(S.mean(tcpu), 2) if tcpu else None,
                 "honest_target_cpu_s_mean": round(S.mean(hcpu), 2) if hcpu else None}
            both = [r for r in tg if r["res"].get("searched") and ref.get(r["key"], {}).get("searched")]
            e["same_pick_vs_honest"] = [sum(1 for r in both if r["res"]["played"] == ref[r["key"]]["played"]), len(both)]
            out["by"][f"{panel}/{b}"] = e
            # a searched game's decision in this bucket: the builder over every decision (exclusions included) plus
            # the forward and playouts of the searched targets
            oth = e["other_cpu_s_mean"] if e["other_cpu_s_mean"] is not None else (
                S.mean(other_b[b]) if other_b[b] else None)
            e["other_cpu_s_used"] = None if oth is None else round(oth, 2)
            cost_b[b] = (e["world_cpu_s"]["mean"] or 0.0) + (oth or 0.0)
        n = slive["per_game"][panel]
        k2 = sum(n[b] * cost_b.get(b, 0.0) for b in n) + ko["overhead_s"]["k2"]
        k6 = sum(n[b] * cost_b.get(b, 0.0) * ko["kappa"][b] for b in n) + ko["overhead_s"]["k6"]
        ex = sum(n[b] * out["by"].get(f"{panel}/{b}", {}).get("excl_rate", 0.0) for b in n) / sum(n.values())
        out["panels"][panel] = {"game_cpu_min_k2": round(k2 / 60, 2), "game_cpu_min_k6": round(k6 / 60, 2),
                                "projected_excl": round(ex, 4),
                                "honest_game_cpu_min_k2": cm["by_panel"][panel]["searched_game_cpu_min_k2"],
                                "honest_game_cpu_min_k6": cm["by_panel"][panel]["searched_game_cpu_min_k6"],
                                "honest_projected_excl": cm["by_panel"][panel]["projected_excl"]}
    P = out["panels"]
    out["U"] = {k: round(S.mean(P[p][k] for p in P), 4) for k in next(iter(P.values()))}
    # THE READ's wall clock (§5: 3,840 searched games and their raw twins): on the agent cores at K = 6 (S'-LIVE life
    # 4's whole-unit rate, CPU-h a wall hour), and on the whole box at K = 14 workers (cores 0-15: seven physical cores
    # with both SMT siblings busy, so a decision costs its K = 6 CPU; the same utilisation a worker)
    rate6 = cm["read"]["k6_cpu_h_per_wall_h"]
    raw6 = slive["raw_unit_cpu_min_k6"]
    for name, game in (("inc", out["U"]["game_cpu_min_k6"]), ("honest", out["U"]["honest_game_cpu_min_k6"])):
        cpu_h = 3840 * (game + raw6) / 60
        out.setdefault("read", {})[name] = {"cpu_h_k6": round(cpu_h, 1), "wall_days_agent_k6": round(cpu_h / rate6 / 24, 2),
                                            "wall_days_box_k14": round(cpu_h / (14 * rate6 / 6) / 24, 2)}
    # agreement at the targets: inc vs honest, and the floors (each preset against its own replicate)
    tg = {r["key"]: r for r in rows if r["target"]}
    ag = {}
    for panel in ("H", "F", "P72", "U"):
        ks = [k for k, r in tg.items() if panel == "U" or r["panel"] == panel]

        def same(x, y):
            pairs = [(x(k), y(k)) for k in ks]
            pairs = [(u, v) for u, v in pairs if u and v and u.get("searched") and v.get("searched")]
            return [sum(1 for u, v in pairs if u["played"] == v["played"]), len(pairs)]

        inc = lambda k: tg[k]["res"]                                            # noqa: E731
        hon = lambda k: ref.get(k)                                              # noqa: E731
        e = {"inc_vs_honest": same(inc, hon)}
        if "inc_rep" in reps:
            e["inc_vs_inc_rep"] = same(inc, lambda k: (reps["inc_rep"].get(k) or {}).get("res"))
        if "honest_rep" in reps:
            e["honest_vs_honest_rep"] = same(hon, lambda k: (reps["honest_rep"].get(k) or {}).get("res"))
            e["inc_vs_honest_on_floor"] = same(inc, lambda k: hon(k) if k in reps["honest_rep"] else None)
        e["deviation_inc"] = round(S.mean([int(tg[k]["res"]["played"] != tg[k]["pi_argmax"]) for k in ks
                                           if tg[k]["res"].get("searched")] or [0]), 4)
        e["deviation_honest"] = round(S.mean([int(ref[k]["played"] != tg[k]["pi_argmax"]) for k in ks
                                              if ref.get(k, {}).get("searched")] or [0]), 4)
        ag[panel] = e
    out["agreement"] = ag
    seqs = {}
    for r in rows:
        seqs.setdefault(tuple(r["group"]), []).append(r)
    whole = sorted((sum(x["res"]["world_cpu_s"] + x["validator_cpu_s"] for x in v if not x["target"]), len(v))
                   for v in seqs.values())
    out["sequence_world_cpu_s"] = {"median": whole[len(whole) // 2] if whole else None, "max": whole[-1] if whole else None}
    out["inc_modes"] = {}
    for r in rows:
        m = (r.get("inc") or {}).get("mode", "excluded_or_none")
        out["inc_modes"][m] = out["inc_modes"].get(m, 0) + 1
    print(json.dumps(out, indent=1, sort_keys=True))
    (Path(a.out) / "summary.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--model-sha256", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--preset", default="r0s4_honest_inc@10")
    r.add_argument("--salt", default="")
    r.add_argument("--targets-only", action="store_true")
    r.add_argument("--shard", default="0/1")
    r.add_argument("--sets", default="stage1,late")
    r.add_argument("--keys", default="", help="only these targets (one key a line): a replicate's subset")
    m = sub.add_parser("summarize")
    m.add_argument("--out", required=True)
    m.add_argument("--inc-rep", default="", help="the r0s4_honest_inc replicate's out dir (another --salt)")
    m.add_argument("--honest-rep", default="", help="the r0s4_honest replicate's out dir (--targets-only, another --salt)")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else summarize(a)
