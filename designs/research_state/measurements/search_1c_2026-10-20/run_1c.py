"""STAGE 1c, PHASE A — THE DRIVER (``PREREGISTRATION.md`` §2–§4, §7): per registered state, build the two PROPOSAL SETS
under ``dice="rolls"`` with F2's legality rule (``N_F`` honest F worlds; ``N_C`` C_ladder worlds, stage 1's corpus
sampler, on the SAME screen cache), score the truth and every proposal by the OPPONENT-CHOICE LIKELIHOOD
(``main.value_search.opp_likelihood``, X26 225M seated as the opponent, L3's definition), record every proposal's
agreement with the truth on each hidden class (``c1c_common.world_facts``), and draw each resampled arm's 8 worlds by
the module's own importance resample (``opp_likelihood.resample``). Forward passes only: no playout, no outcome.

    run_1c.py --one-chunk [--budget-min 40]   # the next states in the registered order, until the budget (resumable)
    run_1c.py --recheck                       # G3: the fixed recheck states rebuilt in a fresh process
    run_1c.py --smoke N                       # ENGINEERING SMOKE on N states OUTSIDE the 287 (a *_smoke archive)
    run_1c.py --status                        # progress counts only (no meter, no outcome)

Exit codes of ``--one-chunk``: 0 = a chunk done, states remain; 3 = every state holds a row (``A_DONE``); 4 = the
registered stop passed (``A_STOPPED``). A state started twice without leaving a row is PARKED (a row ``parked``: a
FINDING), never retried a third time.

Rows: ``<archive>/rows/rows.jsonl`` (weights, likelihoods, facts, resamples, costs) and ``<archive>/worlds/
worlds.jsonl`` (every proposal's team, root prefix and rolls: all a playout needs), one fsync'd line per state each.
A ``WorldError`` building a set is an EXCLUSION of that set (counted); any other exception is a DEFECT (printed).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import resource
import shutil
import sys
import time
import traceback
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c1c_common as K  # noqa: E402

# stage 1's helper modules (its arms' samplers, the corpus, its constants) write their caches under $WS1_ARCHIVE:
# point it HERE, never at stage 1's own archive (which this unit only reads)
os.environ["WS1_ARCHIVE"] = str(K.ARCH / "ws1")
if str(K.REPO / "src") not in sys.path:
    sys.path.insert(0, str(K.REPO / "src"))
sys.path.insert(1, str(K.STAGE1))
import s1_common as E  # noqa: E402  (the bank, its order, the anchor's checkpoint copy; read-only)

STAGE1_ARCH = Path.home() / "gen3ai_archive" / "world_sampler_stage1"


def cpu_seconds() -> float:
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def check_checkout() -> str:
    import main as _m

    if not str(Path(_m.__file__).resolve()).startswith(str(K.REPO / "src")):
        raise SystemExit(f"main imported from {_m.__file__}, not this checkout: run with -I from this checkout")
    return os.popen(f"git -C {K.REPO} rev-parse HEAD").read().strip()


def load_model():
    from main.policy_spectrum.qhat import check_winprob
    from main.policy_spectrum.reader import load_checkpoint

    e = E.anchor_ckpt()
    p = Path(e["file"])
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != e["sha256"]:
        raise SystemExit(f"{p}: sha256 {h} differs from the registered {e['sha256']}")
    m = load_checkpoint(p, device="cpu")
    check_winprob(m)
    return m, e


def ensure_corpus_cache() -> None:
    """Stage 1's normalised, validator-checked ladder corpus: a READ-ONLY copy of its cache file (``corpus_s1.load``
    re-checks the cache's code and source stamps and rebuilds here when they differ; it never writes there)."""
    mine = K.ARCH / "ws1" / "corpus" / "ladder.json"
    theirs = STAGE1_ARCH / "corpus" / "ladder.json"
    if not mine.exists() and theirs.exists():
        mine.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(theirs, mine)


class Builder:
    """One process's samplers: F's honest builder under the dice-marginalised rolls, and stage 1's C_ladder departures
    swapped into the same builder for one ``sample`` call (``samplers_s1.hooks``)."""

    def __init__(self):
        import json

        from main.value_search import world_draws as WD
        from main.value_search import world_legality as WL
        from main.value_search import worlds as W
        from utils.rust_env import ffi as F
        from utils.rust_env.build import ensure_built

        import corpus_s1 as CS
        import samplers_s1 as SS

        self.W, self.SS = W, SS
        self.P = WD.Prior()
        lib = F.load(ensure_built("release")[1])
        self.sampler = W.WorldSampler(prior=self.P, reveal="off", max_tries=K.MAX_TRIES, dice="rolls",
                                      playout_lib=lib)
        ensure_corpus_cache()
        self.joint = SS.JointSmogon(self.P)
        self.ladder = CS.load("ladder", WL.Legality(pad=W._gender_pad()), self.P)
        self.alpha = float(json.loads((K.STAGE1 / "alpha_fit.json").read_text())["ALPHA"])

    def proposals(self, log: dict, at: int, side: str, did: str):
        """{set: (Worlds or None, report)} for the F and C sets, on ONE screen cache (the C set's seen mons take the F
        set's screened item / ability / spread: common random numbers)."""
        cache: dict = {}
        out: Dict[str, dict] = {}
        for name, n in (("F", K.N_F), ("C", K.N_C)):
            c0, w0 = cpu_seconds(), time.monotonic()
            rep: Optional[dict] = None
            try:
                if name == "F":
                    w = self.sampler.sample(log, at, side, n, key=did, screen_cache=cache)
                else:
                    prop = self.SS.CorpusProposal(self.joint, self.ladder, self.alpha, strict=False, name="ladder")
                    with self.SS.hooks(prop):
                        w = self.sampler.sample(log, at, side, n, key=did, screen_cache=cache)
                    rep = prop.report()
                if w.dice != "rolls" or len(w.log_w) != n or len(w.rolls) != n or len(w.logs) != n:
                    raise RuntimeError(f"{name}: {len(w.logs)} worlds with {len(w.log_w)} weights / {len(w.rolls)} "
                                       f"roll tables for {n}")
                for lg, r in zip(w.logs, w.rolls):
                    if list(lg.get("rolls") or []) != list(r):
                        raise RuntimeError(f"{name}: a world's log does not carry the rolls it took")
                out[name] = {"ok": True, "W": w, "prop": rep}
            except self.W.WorldError as exc:
                out[name] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
            except Exception as exc:                     # a DEFECT (a hook or guard included)
                out[name] = {"ok": False, "defect": True, "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                             "trace": traceback.format_exc()[-1500:]}
            out[name]["cost"] = {"cpu_s": round(cpu_seconds() - c0, 2), "wall_s": round(time.monotonic() - w0, 2)}
        return out


def union(a, b):
    """The two proposal sets as one (F first): every per-world record concatenated, the rolls with their worlds."""
    return dc_replace(a, logs=a.logs + b.logs, teams=a.teams + b.teams, tries=a.tries + b.tries,
                      stats=a.stats + b.stats, facts=a.facts + b.facts, log_w=a.log_w + b.log_w,
                      rolls=a.rolls + b.rolls)


def one(did: str, d: dict, bank, st: dict, B: Builder, fwd, head: str) -> tuple:
    """(the state's row, its worlds row)."""
    from build_worlds import split_prefix                # Tier 1's helper (imported, never copied)
    from main.policy_spectrum.truth import cmd_index, to_log
    from main.value_search import opp_likelihood as OL
    from main.value_search.worlds import item_id, parse_set, prepare_root

    b = bank.battles[bank.battle_index[d["battle"]]]
    side = d["side"]
    at = cmd_index(b, side, d["n"])
    log = to_log(b)
    row = {"id": did, "battle": d["battle"], "stratum": st.get("stratum"), "k": st.get("k"), "turn": st.get("turn"),
           "side": side, "at": at, "head": head, "N_F": K.N_F, "N_C": K.N_C, "J": K.J}
    wrow = {"id": did, "at": at, "side": side, "head": head}
    t0 = time.monotonic()
    P = B.proposals(log, at, side, did)
    for name in ("F", "C"):
        p = P[name]
        row[name] = {k: p.get(k) for k in ("ok", "error", "defect", "trace", "prop", "cost") if p.get(k) is not None}
        if p["ok"]:
            w = p["W"]
            row[name].update(log_w=[float(x) for x in w.log_w], tries=w.tries, fails=w.fails, screens=w.screens)
            wrow[name] = {"teams": list(w.teams), "cmds": [split_prefix(lg, at) for lg in w.logs],
                          "rolls": [list(r) for r in w.rolls]}
    if any(P[n].get("defect") for n in ("F", "C")):
        row["defect"] = True
        print(f"[1c] DEFECT {did}: " + "; ".join(str(P[n].get("error"))[:200] for n in ("F", "C")
                                                     if P[n].get("defect")), flush=True)
    sets = [n for n in ("F", "C") if P[n]["ok"]]
    row["ok"] = bool(sets)
    if not sets:
        row["wall_s"] = round(time.monotonic() - t0, 2)
        return row, wrow
    c0 = cpu_seconds()
    try:
        root = prepare_root(log, at, side, did)
        steps = OL.opponent_steps(root)
        logs = [OL.truth_log(root)] + [lg for n in sets for lg in P[n]["W"].logs]
        res = OL.world_logliks(root, logs, fwd, steps, label=did)
        row.update(ll_truth=res[0].ll, conf_truth=res[0].conf, n_scored=res[0].n_scored,
                   n_informative_truth=res[0].n_informative, n_unobserved=res[0].n_unobserved,
                   n_pending=res[0].n_pending, steps=[[s.k, s.cmd, s.status] for s in steps])
        o = 1
        lls: Dict[str, List[float]] = {}
        for n in sets:
            m = len(P[n]["W"].logs)
            lls[n] = [r.ll for r in res[o:o + m]]
            row[n].update(ll=lls[n], conf=[r.conf for r in res[o:o + m]],
                          n_informative=[r.n_informative for r in res[o:o + m]])
            o += m
        true_team = root.team
        item_rev = {}
        for i, sid in enumerate(root.sids):
            if sid in root.kept:
                mr = root.rv.mons.get(sid)
                item_rev[sid] = bool(mr is not None and mr.item is not None and mr.item == item_id(root.sets[i]))
        kept = {s: sorted(v) for s, v in root.kept.items()}
        row["item_revealed"] = item_rev
        for n in sets:
            w = P[n]["W"]
            row[n]["facts"] = [K.world_facts(true_team, t, kept, item_rev) for t in w.teams]
            row[n]["near4"] = [int(K.near_copy(true_team, t)) for t in w.teams]
        # THE RESAMPLES (the module's own function: OL.resample), each arm's 8 kept worlds as proposal indices
        res_ = {}
        for arm in K.RESAMPLED:
            parts, wt = K.ARMS[arm]
            if any(s not in sets for s, _ in parts):
                continue
            Wa = None
            for s, n in parts:                            # the first n of each set, F before C
                Ws = P[s]["W"] if n == len(P[s]["W"].logs) else P[s]["W"].select(range(n))
                Wa = Ws if Wa is None else union(Wa, Ws)
            ll = K.arm_slice(lls, arm)
            kept_w = OL.resample(Wa, K.J, K.RESAMPLE_KEY.format(arm=arm, id=did),
                                 ll=ll if wt == "dice*opp" else None, dice=True)
            idx = list(kept_w.resample["src"])
            for j, i in enumerate(idx):                   # G5: each kept world IS its proposal, rolls included
                if kept_w.logs[j] != Wa.logs[i] or kept_w.rolls[j] != Wa.rolls[i] or kept_w.teams[j] != Wa.teams[i]:
                    raise RuntimeError(f"{arm}: kept world {j} differs from its proposal {i}")
            res_[arm] = {"idx": idx, "ess": round(float(kept_w.resample["ess"]), 6)}
        row["resample"] = res_
    except Exception as exc:                             # a guard raised: a DEFECT, recorded and printed
        row.update(ok=False, defect=True, error=f"{type(exc).__name__}: {str(exc)[:500]}",
                   trace=traceback.format_exc()[-1500:])
        print(f"[1c] DEFECT {did}: {row['error'][:300]}", flush=True)
    row["cost_ll_cpu_s"] = round(cpu_seconds() - c0, 2)
    row["wall_s"] = round(time.monotonic() - t0, 2)
    return row, wrow


def smoke_ids(n: int) -> List[str]:
    """ENGINEERING SMOKE states OUTSIDE the 287: the interim bank's NON-LEAN states (L3's choice), sha256 order."""
    st = E.mat_states()
    out = sorted((k for k, v in st.items() if not v["lean"]),
                 key=lambda i: hashlib.sha256((K.SMOKE_SALT + i).encode()).hexdigest())
    return out[:n]


def recheck_ids() -> List[str]:
    return sorted(E.common_ids(), key=lambda x: hashlib.sha256((K.RECHECK_SALT + x).encode()).hexdigest()
                  )[:K.N_RECHECK]


def attempts_path() -> Path:
    return K.d("logs") / "attempts.log"


def started_counts() -> Dict[str, int]:
    p = attempts_path()
    cnt: Dict[str, int] = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                cnt[line.split()[0]] = cnt.get(line.split()[0], 0) + 1
    return cnt


def status() -> None:
    rows = K.read_by_id(K.d("rows") / "rows.jsonl")
    order = E.ordered_ids()
    done = [i for i in order if i in rows]
    print({"registered": len(order), "rows": len(rows), "prefix": done == order[:len(done)],
           "ok": sum(1 for r in rows.values() if r.get("ok")), "defects": sum(1 for r in rows.values() if r.get("defect")),
           "parked": sum(1 for r in rows.values() if r.get("parked")), "stop_at": K.stop_at(),
           "done": (K.ARCH / "A_DONE").exists(), "stopped": (K.ARCH / "A_STOPPED").exists()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one-chunk", action="store_true")
    ap.add_argument("--budget-min", type=float, default=40.0)
    ap.add_argument("--recheck", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status:
        status()
        return
    head = check_checkout()
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.qhat import logits_and_values
    from main.policy_spectrum.reader import inference_globals

    for p_ in (E.T1, E.PILOT):
        if str(p_) not in sys.path:
            sys.path.insert(2, str(p_))
    st = E.mat_states()
    order = E.ordered_ids()
    if a.smoke:
        if not str(K.ARCH).endswith("_smoke"):
            raise SystemExit("the engineering smoke writes only to a *_smoke archive (S1C_ARCHIVE)")
        ids = smoke_ids(a.smoke)
        rpath, wpath = K.d("rows") / "rows.jsonl", K.d("worlds") / "worlds.jsonl"
    elif a.recheck:
        ids = recheck_ids()
        rpath, wpath = K.d("recheck") / "rows.jsonl", K.d("recheck") / "worlds.jsonl"
    elif a.one_chunk:
        if str(K.ARCH).endswith("_smoke"):
            raise SystemExit("the registered run never writes to a *_smoke archive")
        ids = order
        rpath, wpath = K.d("rows") / "rows.jsonl", K.d("worlds") / "worlds.jsonl"
    else:
        raise SystemExit("one of --one-chunk / --recheck / --smoke N / --status")
    done = set(K.read_by_id(rpath))
    todo = [i for i in ids if i not in done]
    if a.one_chunk:
        if not todo:
            (K.ARCH / "A_DONE").write_text(time.strftime("%F %T") + "\n")
            print("[1c] every state holds a row: A_DONE", flush=True)
            sys.exit(3)
        sa = K.stop_at()
        if sa is not None and time.time() >= sa:
            (K.ARCH / "A_STOPPED").write_text(time.strftime("%F %T") + f" {len(done)} rows\n")
            print("[1c] the registered stop passed: A_STOPPED", flush=True)
            sys.exit(4)
    bank = load_bank(E.MAT_ARCH / "bank")
    dec = {x["id"]: x for x in bank.decisions}
    t_load = time.monotonic()
    B = Builder()
    print(f"[1c] {len(todo)} states to {rpath.parent.name}/ at {head[:8]} (ladder {len(B.ladder.teams)}, alpha "
          f"{B.alpha}, loaded in {time.monotonic() - t_load:.0f} s)", flush=True)
    t0 = time.monotonic()
    with inference_globals(1):
        model, ck = load_model()

        def fwd(rows, masks):
            return logits_and_values(model, rows, masks, batch=1024)

        with open(rpath, "a") as fr, open(wpath, "a") as fw:
            for n_i, did in enumerate(todo):
                if a.one_chunk:
                    if (time.monotonic() - t0) / 60.0 >= a.budget_min:
                        break
                    sa = K.stop_at()
                    if sa is not None and time.time() >= sa:
                        break
                    if started_counts().get(did, 0) >= 2:     # started twice, never finished: PARKED (a FINDING)
                        K.write_row(fr, {"id": did, "ok": False, "parked": True, "head": head,
                                         "error": "started twice without a row"})
                        print(f"[1c] PARKED {did}", flush=True)
                        continue
                    with open(attempts_path(), "a") as fa:
                        fa.write(f"{did} {time.strftime('%F %T')}\n")
                row, wrow = one(did, dec[did], bank, st.get(did, {}), B, fwd, head)
                row["ckpt_step"] = ck["step"]
                K.write_row(fw, wrow)
                K.write_row(fr, row)
                ess = {k: v["ess"] for k, v in (row.get("resample") or {}).items()}
                print(f"[1c] {n_i + 1}/{len(todo)} ok={row.get('ok')} F={row['F'].get('ok')} C={row['C'].get('ok')} "
                      f"build F {row['F']['cost']['cpu_s']:.0f} C {row['C']['cost']['cpu_s']:.0f} "
                      f"ll {row.get('cost_ll_cpu_s', 0):.0f} cpu-s; wall {row['wall_s']:.0f} s"
                      + (f"; ESS {ess}" if a.smoke else "")
                      + f"; {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    if a.one_chunk and all(i in K.read_by_id(rpath) for i in order):
        (K.ARCH / "A_DONE").write_text(time.strftime("%F %T") + "\n")
        print("[1c] every state holds a row: A_DONE", flush=True)
        sys.exit(3)
    print("[1c] chunk done", flush=True)


if __name__ == "__main__":
    main()
