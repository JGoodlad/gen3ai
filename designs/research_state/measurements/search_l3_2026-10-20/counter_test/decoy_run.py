"""L3's FAMILIARITY COUNTER-TEST, THE DRIVER (``PREREGISTRATION.md`` §2–§3; the L3 review's F2). On each of L3's 287
registered states: rebuild L3's 8 honest worlds exactly as L3 did, draw up to D = 4 POOL DECOYS (training-pool teams,
NOT the truth and not a near copy of it, consistent with everything the searched side saw), and score the truth, the 8
honest worlds and the decoys by the opponent-choice likelihood under X26's 225M checkpoint in ONE deduplicated
forward. Forward passes only: no playout, no outcome is read or written.

Runs from a detached checkout of L3's code commit (``--code``, ``8fe4fb89``) so the truth and the honest worlds are
L3's own (gate G5 compares them with L3's archived rows)::

    CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh --slot --name l3decoy 12 \\
        timeout 45m python -I decoy_run.py --code <8fe4fb89 checkout> --budget-min 40

One fsync'd JSONL row per state in ``$L3D_ARCHIVE/rows.jsonl`` (resumable; default ``~/gen3ai_archive/search_l3_decoy``);
``--recheck N`` re-scores N fixed states in a fresh process into ``recheck.jsonl`` (G3).

A DECOY (``PREREGISTRATION.md`` §2): a pool team (X26's pin packing, the C_pool pool, sha-checked) whose species include
every species the searched side saw and whose sets hold every move it saw (``pool_worlds.candidates``); NOT the truth
(its ``team_key``) and sharing FEWER than 4 identical mon sets with it (``set_key``); built as Tier 1's POOL builds it
(``pool_worlds.build_world``: its sets at the seen mons' true slots, its other mons at the never-seen slots); gated by
the SAME damage-roll MATCH gate the honest ``dice="rolls"`` worlds pass (the searched side's rows and protocol
byte-equal to the truth's, some feasible rolls), the playout tool's open check, and the world consistency guard.
Candidates are tried in a fixed sha256 order (``search_l3_decoy_v1:<id>``); the first D distinct teams that pass are the
decoys (a uniform draw without replacement among the passers); at most ``MAX_GATED`` candidates are gated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
ARCH = Path(os.environ.get("L3D_ARCHIVE") or (Path.home() / "gen3ai_archive" / "search_l3_decoy"))
L3_ARCH = Path.home() / "gen3ai_archive" / "search_l3"
POOL_PIN = Path.home() / "gen3ai_archive" / "world_sampler_stage1" / "pool" / "pool_pin.json"
J = 8
D = 4
MAX_TRIES = 60
MAX_GATED = 1000
NEAR = 4
SALT = "search_l3_decoy_v1:"
RECHECK_SALT = "search_l3_decoy_recheck_v1:"


def _rng(tag: str):
    import numpy as np

    return np.random.default_rng(int(hashlib.sha256(tag.encode()).hexdigest()[:16], 16))


def shared_sets(PT, a: str, b: str) -> int:
    """How many identical mon sets (``set_key``: species, item, ability, moves, nature, EVs, IVs) two teams share."""
    ca, cb = Counter(PT.set_key(e) for e in a.split("]")), Counter(PT.set_key(e) for e in b.split("]"))
    return sum((ca & cb).values())


def build_decoys(root, pool, PT, PW, W, sampler, did: str) -> dict:
    """Up to D decoys of this root (module docstring), with the draw's record. Never raises for a missing decoy."""
    from agents.battle.core_replay import run_core

    def gate(res, _c=None):
        return PW.strict_verdict(root, res), 0.0, None

    match = sampler._roll_match(root, did + "#decoy", lambda t: W.render(root.tmpl, t), gate)
    kept = {s: set(m) for s, m in root.kept.items()}
    cands = PW.candidates(pool, kept)
    tk = PT.team_key(root.team)
    meta: dict = {"candidates": len(cands), "is_truth": 0, "near_copy": 0, "dup": 0, "fails": {}}
    seen_keys = set()
    ok_c: List[int] = []
    for c in [cands[i] for i in _rng(SALT + did).permutation(len(cands))]:
        t = pool.teams[c]
        k = PT.team_key(t)
        if k == tk:
            meta["is_truth"] += 1
            continue
        if shared_sets(PT, t, root.team) >= NEAR:
            meta["near_copy"] += 1
            continue
        if k in seen_keys:
            meta["dup"] += 1
            continue
        seen_keys.add(k)
        ok_c.append(c)
    meta["eligible"] = len(ok_c)
    out_t: List[str] = []
    out_c: List[int] = []
    out_r: List[list] = []
    out_lw: List[float] = []
    gated = 0
    for s0 in range(0, min(len(ok_c), MAX_GATED), MAX_TRIES):
        if len(out_t) >= D:
            break
        built = []
        for c in ok_c[s0:min(s0 + MAX_TRIES, MAX_GATED)]:
            tw, why = PW.build_world(root, pool.teams[c], _rng(f"{SALT}{did}:cand:{c}"), sampler.pad)
            if tw is None:
                meta["fails"][why] = meta["fails"].get(why, 0) + 1
            else:
                built.append((c, tw))
        gated += min(s0 + MAX_TRIES, MAX_GATED, len(ok_c)) - s0
        if not built:
            continue
        res = run_core([root.rec(tw, f"{did}#decoy{s0 + i}", W.render(root.tmpl, tw), match)
                        for i, (_, tw) in enumerate(built)], trackers=True, obs=True)
        for (c, tw), r in zip(built, res):
            if len(out_t) >= D:
                break
            rr = r.get("rolls") or {}
            if not (PW.strict_verdict(root, r) and rr.get("feasible")):
                meta["fails"]["gate"] = meta["fails"].get("gate", 0) + 1
                continue
            lg = PW.world_log(root, tw)
            if rr.get("force"):
                lg["rolls"] = [list(x) for x in rr["force"]]
            why = sampler.playout_refusal(lg, root.at, root.side)
            if why is not None:
                meta["fails"]["playout_refused"] = meta["fails"].get("playout_refused", 0) + 1
                continue
            out_t.append(tw)
            out_c.append(c)
            out_r.append(lg)
            out_lw.append(float(rr.get("log_w", 0.0)))
    meta["gated"] = gated
    if out_t:
        from main.value_search import world_guard as WG

        WG.check_teams(root, out_t, did + "#decoy")        # THROWING: a defect, never an exclusion
    meta.update(n=len(out_t), pool_index=out_c, log_w=out_lw,
                shared_sets=[shared_sets(PT, t, root.team) for t in out_t])
    return {"teams": out_t, "logs": out_r, "meta": meta}


def one(did: str, d: dict, bank, st: dict, sampler, fwd, pool, mods, head: str, smoke: bool = False) -> dict:
    from main.policy_spectrum.truth import cmd_index, to_log
    from main.value_search import opp_likelihood as OL
    from main.value_search.worlds import WorldError, prepare_root

    PT, PW, W, L3R = mods
    b = bank.battles[bank.battle_index[d["battle"]]]
    side = d["side"]
    at = cmd_index(b, side, d["n"])
    log = to_log(b)
    row = {"id": did, "battle": d["battle"], "stratum": st["stratum"], "turn": st.get("turn"), "side": side, "at": at,
           "head": head, "J": J, "D_max": D}
    w0 = time.monotonic()
    try:
        Wd = sampler.sample(log, at, side, J, key=did, screen_cache={})
    except WorldError as exc:
        return dict(row, ok=False, error=f"world build: {type(exc).__name__}: {str(exc)[:400]}",
                    wall_s=round(time.monotonic() - w0, 2))
    try:
        root = prepare_root(log, at, side, did)
        dec = build_decoys(root, pool, PT, PW, W, sampler, did)
        if smoke:
            return dict(row, ok=True, smoke=True, n_decoys=len(dec["teams"]), decoy=dec["meta"],
                        wall_s=round(time.monotonic() - w0, 2))
        steps = OL.opponent_steps(root)
        tlog = OL.truth_log(root)
        res = OL.world_logliks(root, [tlog] + list(Wd.logs) + dec["logs"], fwd, steps, label=did)
        mt = [L3R.matches(root.team, t, Wd.revealed) for t in list(Wd.teams) + dec["teams"]]
        row.update(
            ok=True, ll=[r.ll for r in res], conf=[r.conf for r in res], n_decoys=len(dec["teams"]),
            decoy=dec["meta"], n_scored=res[0].n_scored, n_informative=[r.n_informative for r in res],
            n_unobserved=res[0].n_unobserved, n_pending=res[0].n_pending, log_w=list(Wd.log_w),
            match_m=[m["match_m"] for m in mt], exact_m=[m["exact_m"] for m in mt], hit_u=[m["hit_u"] for m in mt],
            n_unrev=mt[0]["n_unrev"], n_never=mt[0]["n_never"])
    except Exception as exc:                             # a guard raised: a DEFECT, recorded and printed
        row.update(ok=False, defect=True, error=f"{type(exc).__name__}: {str(exc)[:400]}",
                   trace=traceback.format_exc()[-1500:])
        print(f"[l3d] DEFECT {did}: {row['error'][:300]}", flush=True)
    row["wall_s"] = round(time.monotonic() - w0, 2)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="the checkout of L3's code commit (8fe4fb89)")
    ap.add_argument("--recheck", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget-min", type=float, default=0.0, help="stop after the state in progress past this")
    ap.add_argument("--smoke", type=int, default=0,
                    help="ENGINEERING SMOKE: build (no forward, no likelihood) on N bank states NOT among the 287")
    a = ap.parse_args()
    code = Path(a.code).resolve()
    l3dir = code / "designs/research_state/measurements/search_l3_2026-10-20"
    stage1 = code / "designs/research_state/measurements/world_sampler_stage1_2026-10-19"
    t1 = code / "designs/research_state/measurements/hidden_info_tier1_2026-10-13"
    for p in (t1, stage1, l3dir, code / "src"):
        sys.path.insert(0, str(p))
    import main as _m

    if not str(Path(_m.__file__).resolve()).startswith(str(code / "src")):
        raise SystemExit(f"main imported from {_m.__file__}, not {code}: run with -I")
    import l3_run as L3R  # noqa: E402  (L3's own driver: its checkpoint loader and its match meter)
    import pool_teams as PT  # noqa: E402
    import pool_worlds as PW  # noqa: E402
    import s1_common as E  # noqa: E402
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.qhat import logits_and_values
    from main.policy_spectrum.reader import inference_globals
    from main.value_search import worlds as W
    from main.value_search.worlds import WorldSampler

    blob = json.loads(POOL_PIN.read_text())
    if hashlib.sha256("\n".join(blob["teams"]).encode()).hexdigest() != blob["sha256"]:
        raise SystemExit("pool_pin.json does not match its sha256")
    idx: Dict[str, List[int]] = {}
    for i, t in enumerate(blob["teams"]):
        idx.setdefault(t, []).append(i)
    pool = PT.Pool(teams=list(blob["teams"]), sha256=blob["sha256"], index_of=idx)

    ARCH.mkdir(parents=True, exist_ok=True)
    ids = E.ordered_ids()
    if a.smoke:
        reg = set(ids)
        st0 = E.mat_states()
        ids = sorted((i for i in st0 if i not in reg), key=lambda x: hashlib.sha256((SALT + "smoke:" + x).encode())
                     .hexdigest())[:a.smoke]
        path = ARCH / "smoke.jsonl"
    elif a.recheck:
        ids = sorted(ids, key=lambda x: hashlib.sha256((RECHECK_SALT + x).encode()).hexdigest())[:a.recheck]
        path = ARCH / "recheck.jsonl"
    else:
        path = ARCH / "rows.jsonl"
    if a.limit:
        ids = ids[:a.limit]
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    ids = [i for i in ids if i not in done]
    head = os.popen(f"git -C {code} rev-parse HEAD").read().strip()
    st = E.mat_states()
    bank = load_bank(E.MAT_ARCH / "bank")
    dec = {d["id"]: d for d in bank.decisions}
    ids = [i for i in ids if i in dec and i in st]
    sampler = WorldSampler(reveal="off", max_tries=MAX_TRIES, dice="rolls")
    print(f"[l3d] {len(ids)} states to {path.name} at {head[:8]} (pool {pool.sha256[:8]})", flush=True)
    t0 = time.monotonic()
    with inference_globals(1):
        model, ck = L3R.load_model()

        def fwd(rows, masks):
            if a.smoke:
                raise SystemExit("the smoke computes no likelihood")
            return logits_and_values(model, rows, masks, batch=1024)

        with open(path, "a") as fo:
            for k, did in enumerate(ids):
                row = one(did, dec[did], bank, st[did], sampler, fwd, pool, (PT, PW, W, L3R), head, smoke=bool(a.smoke))
                row["ckpt_step"] = ck["step"]
                row["pool_sha256"] = pool.sha256
                fo.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                fo.flush()
                os.fsync(fo.fileno())
                el = (time.monotonic() - t0) / 60
                print(f"[l3d] {k + 1}/{len(ids)} ok={row.get('ok')} decoys={row.get('n_decoys')} {el:.1f} min",
                      flush=True)
                if a.budget_min and el > a.budget_min:
                    print("[l3d] BUDGET: stopping; resume to continue", flush=True)
                    return
    print("[l3d] done", flush=True)


if __name__ == "__main__":
    main()
