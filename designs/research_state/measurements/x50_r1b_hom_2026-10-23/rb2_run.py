"""R-b2's SCORING DRIVER (``design_x50_rung1b_human_opponent_model.md`` §8.4): the frozen off-pool bank re-scored with
TWO judges in ONE process, P* and the HOM. Only the ``Forward`` changes.

Per state it is the bank's own scoring (``search_offpool_bank_2026-10-20/offpool_run.py`` ``one()``, imported by path):
the same world draw (``WorldSampler.sample`` with the state's key), the same root, the same off-pool decoys, the same
opponent steps and ONE replay of every candidate (the truth, the 8 honest worlds, the decoys). The candidates' rows
then go through ``opp_likelihood.world_logliks``'s own post-replay code once per judge, P*'s exactly as the bank ran it
(the same deduplicated batch, under ``inference_globals(1)``).

G-REPRO (a gate): P*'s re-score must equal ``rows.jsonl``'s per-candidate log-likelihoods and familiarity terms
(|Δ| <= 1e-9) on every state, and a state the bank excluded must be excluded again. Otherwise the worlds differ: a
defect, and no read. The HOM's zip must hash to its committed sidecar (frozen before any likelihood).

    CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh --slot --name x50r1b_rb2 12 \\
        timeout 45m python -I rb2_run.py --code <8fe4fb89 checkout> --hom <hom zip> --sidecar <sidecar> \\
        --out <archive>/rb2_rows.jsonl --budget-min 38

One fsync'd JSONL row per state (resumable).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
BANK = HERE.parent / "search_offpool_bank_2026-10-20"
sys.path.insert(0, str(BANK))
import offpool_common as E  # noqa: E402

TOL = 1e-9


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def judged(OL, root, logs, fwds, steps, label):
    """``world_logliks`` for several judges over ONE replay: its body after ``_replay``, once per judge."""
    import numpy as np

    steps = list(steps)
    results = OL._replay(root, logs, label)
    per = [OL._opp_inputs(root, res, lg, steps, j) for j, (res, lg) in enumerate(zip(results, logs))]
    rows = [x for xs, _, _ in per for x in xs]
    n_un = sum(1 for s in steps if s.status == "unobserved")
    n_pe = sum(1 for s in steps if s.status == "pending")
    out = {}
    for name, fwd in fwds.items():
        if not rows:
            out[name] = [OL.OppLikelihood(ll=0.0, conf=0.0, n_unobserved=n_un, n_pending=n_pe) for _ in logs]
            continue
        masks = np.stack([m for _, ms, _ in per for m in ms])
        o_ = OL.dedup_forward(fwd)(np.stack(rows), masks)
        lp, cf = OL.log_probs(np.asarray(o_), masks, [a for _, _, acts in per for a in acts])
        res_ = []
        o = 0
        for xs, ms, _ in per:
            k = len(xs)
            n_leg = [int(m.sum()) for m in ms]
            res_.append(OL.OppLikelihood(ll=float(lp[o:o + k].sum()), conf=float(cf[o:o + k].sum()),
                                         logp=[float(x) for x in lp[o:o + k]], n_legal=n_leg, n_scored=k,
                                         n_informative=sum(1 for n in n_leg if n >= 2), n_unobserved=n_un,
                                         n_pending=n_pe))
            o += k
        out[name] = res_
    return out


def one(did, d, bank, st, sampler, fwds, corpus, mods, old):
    from main.policy_spectrum.truth import cmd_index, to_log
    from main.value_search import opp_likelihood as OL
    from main.value_search.worlds import WorldError, prepare_root

    import offpool_run as OR

    PT, PW, W = mods
    b = bank.battles[bank.battle_index[d["battle"]]]
    side = d["side"]
    at = cmd_index(b, side, d["n"])
    log = to_log(b)
    row = {"id": did, "battle": d["battle"], "stratum": st["stratum"], "opp_team": st["opp_team"],
           "outcome": st["outcome"], "k_bin": st["k_bin"]}
    w0 = time.monotonic()
    try:
        Wd = sampler.sample(log, at, side, E.J, key=did, screen_cache={})
    except WorldError as exc:
        row.update(ok=False, error=f"world build: {type(exc).__name__}: {str(exc)[:400]}",
                   repro_excluded_again=not old.get("ok") and not old.get("defect"))
        return row
    try:
        root = prepare_root(log, at, side, did)
        dec = OR.build_decoys(root, corpus, PT, PW, W, sampler, did)
        steps = OL.opponent_steps(root)
        tlog = OL.truth_log(root)
        res = judged(OL, root, [tlog] + list(Wd.logs) + dec["logs"], fwds, steps, did)
        for name, rs in res.items():
            row[f"ll_{name}"] = [r.ll for r in rs]
            row[f"conf_{name}"] = [r.conf for r in rs]
            row[f"n_informative_{name}"] = [r.n_informative for r in rs]
        row.update(ok=True, n_decoys=len(dec["teams"]), n_scored=res["pstar"][0].n_scored,
                   steps=[[s.k, s.cmd, s.status] for s in steps])
        if old.get("ok"):
            dl = [abs(a - b) for a, b in zip(row["ll_pstar"], old["ll"])]
            dc = [abs(a - b) for a, b in zip(row["conf_pstar"], old["conf"])]
            row["repro"] = {"n": len(old["ll"]), "n_now": len(row["ll_pstar"]),
                            "max_abs_ll": max(dl) if dl else 0.0, "max_abs_conf": max(dc) if dc else 0.0,
                            "steps_equal": row["steps"] == old["steps"]}
            row["repro"]["ok"] = (row["repro"]["n"] == row["repro"]["n_now"] and row["repro"]["steps_equal"]
                                  and row["repro"]["max_abs_ll"] <= TOL and row["repro"]["max_abs_conf"] <= TOL)
        else:
            row["repro"] = {"ok": False, "why": "the bank excluded this state; now it scores"}
    except Exception as exc:                             # a guard raised: a DEFECT
        row.update(ok=False, defect=True, error=f"{type(exc).__name__}: {str(exc)[:400]}",
                   trace=traceback.format_exc()[-1500:])
        print(f"[rb2] DEFECT {did}: {row['error'][:300]}", flush=True)
    row["wall_s"] = round(time.monotonic() - w0, 2)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--hom", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget-min", type=float, default=0.0)
    a = ap.parse_args()
    side = json.loads(Path(a.sidecar).read_text())
    if sha256_file(Path(a.hom)) != side["hom_sha256"]:
        raise SystemExit("the HOM's sha256 differs from its sidecar: not the frozen model")
    if side["init_sha256"] != E.PSTAR_SHA:
        raise SystemExit("the HOM's init is not P*")
    E.use_code(Path(a.code))
    import offpool_run as OR  # noqa: E402  (the bank's driver: its decoys, corpus and P* loader)
    import pool_teams as PT  # noqa: E402
    import pool_worlds as PW  # noqa: E402
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.qhat import logits_and_values
    from main.policy_spectrum.reader import inference_globals, load_checkpoint
    from main.value_search import worlds as W
    from main.value_search.worlds import WorldSampler

    corpus = OR.Corpus(PT, W)
    doc = json.loads(E.states_path(False).read_text())
    st = doc["states"]
    ids = sorted(st, key=lambda x: hashlib.sha256((E.STATES_SALT + "order:" + x).encode()).hexdigest())
    olds = {}
    for line in (E.ARCH / "rows.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            olds[r["id"]] = r
    out = Path(a.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [i for i in ids if i not in done]
    if a.limit:
        todo = todo[:a.limit]
    bank = load_bank(E.ARCH / "bank")
    if bank.manifest.get("content_sha256") != doc["bank_content_sha256"]:
        raise SystemExit("the bank's content sha256 differs from states_offpool.json")
    dec = {x["id"]: x for x in bank.decisions}
    sampler = WorldSampler(reveal="off", max_tries=E.MAX_TRIES, dice="rolls")
    head = os.popen(f"git -C {a.code} rev-parse HEAD").read().strip()
    print(f"[rb2] {len(todo)} states to {out.name} at {head[:8]}", flush=True)
    t0 = time.monotonic()
    with inference_globals(1):
        models = {"pstar": OR.load_model(), "hom": load_checkpoint(Path(a.hom), device="cpu")}
        fwds = {k: (lambda rows, masks, _m=m: logits_and_values(_m, rows, masks, batch=1024)) for k, m in models.items()}
        with open(out, "a") as fo:
            for k, did in enumerate(todo):
                row = one(did, dec[did], bank, st[did], sampler, fwds, corpus, (PT, PW, W), olds[did])
                row["code"] = head
                row["hom_sha256"] = side["hom_sha256"]
                fo.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                fo.flush()
                os.fsync(fo.fileno())
                el = (time.monotonic() - t0) / 60
                print(f"[rb2] {k + 1}/{len(todo)} ok={row.get('ok')} repro={(row.get('repro') or {}).get('ok')} "
                      f"wall={row.get('wall_s')}s {el:.1f} min", flush=True)
                if a.budget_min and el > a.budget_min:
                    print("[rb2] BUDGET: stopping; resume to continue", flush=True)
                    return
    print("[rb2] done", flush=True)


if __name__ == "__main__":
    main()
