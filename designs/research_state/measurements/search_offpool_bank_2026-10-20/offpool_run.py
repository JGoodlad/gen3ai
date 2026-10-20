"""THE SCORING DRIVER (``PREREGISTRATION.md`` §5–§6): on every state of the off-pool bank, L3's candidates (the truth and
8 honest worlds, built by L3's own call) plus up to D = 4 OFF-POOL DECOYS (``hl_05_26`` ladder teams that are neither
pool teams nor near copies of the truth, consistent with everything the searched side saw), all scored by L3's
opponent-choice likelihood under P* in ONE deduplicated forward. Forward passes only: no playout, no outcome.

It is L3's counter-test driver (``search_l3_2026-10-20/counter_test/decoy_run.py``) with four changes: the bank (this
unit's), the opponent model (P*, not 225M), the decoy source (the off-pool corpus, ``offpool_refs.corpus``, not the
pool) and one more near-copy rule (>= 4 mon sets sharing species and moves, Hidden Power untyped, as well as >= 4
identical ``set_key``s). Everything else (``_roll_match`` gate, ``pool_worlds.build_world``, the playout-open check, the
throwing consistency guard, the sha256 candidate order, batches of 60, at most 1,000 gated) is the counter-test's.

    CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh --slot --name offpool 12 \\
        timeout 45m python -I offpool_run.py --code <8fe4fb89 checkout> --budget-min 38

One fsync'd JSONL row per state in ``<archive>/rows.jsonl`` (resumable); ``--recheck N`` re-scores N fixed states in a
fresh process into ``recheck.jsonl`` (G3). ``$OFFPOOL_PIN_CORE_EVENTS`` names X26's pin's ``core_events`` (G4).
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
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offpool_common as E  # noqa: E402


def _rng(tag: str):
    import numpy as np

    return np.random.default_rng(int(hashlib.sha256(tag.encode()).hexdigest()[:16], 16))


def loose_key(entry: str) -> tuple:
    """A packed set's species and move set, Hidden Power UNTYPED (the stricter near-copy key)."""
    from main.value_search.worlds import parse_set, toid, typed_moves

    f = parse_set(entry)
    return (toid(f[1] or f[0]), tuple(sorted("hiddenpower" if m.startswith("hiddenpower") else m
                                             for m in typed_moves(f))))


def shared(keyf, a: str, b: str) -> int:
    ca, cb = Counter(keyf(e) for e in a.split("]")), Counter(keyf(e) for e in b.split("]"))
    return sum((ca & cb).values())


class Corpus:
    """The off-pool decoy corpus as a ``pool_teams.Pool`` plus a species index for the candidate pre-filter."""

    def __init__(self, PT, W):
        blob = json.loads((E.ARCH / "decoy_corpus.json").read_text())
        if hashlib.sha256("\n".join(blob["teams"]).encode()).hexdigest() != blob["sha256"]:
            raise SystemExit("decoy_corpus.json does not match its sha256")
        self.sha256 = blob["sha256"]
        self.ids = blob["ids"]
        idx: Dict[str, List[int]] = {}
        for i, t in enumerate(blob["teams"]):
            idx.setdefault(t, []).append(i)
        self.pool = PT.Pool(teams=list(blob["teams"]), sha256=blob["sha256"], index_of=idx)
        self.species = [frozenset(W.toid(f[1] or f[0]) for f in (W.parse_set(e) for e in t.split("]")))
                        for t in blob["teams"]]

    def candidates(self, PT, PW, kept) -> List[int]:
        """``pool_worlds.candidates`` over the corpus, after a species-superset pre-filter (the same predicate)."""
        need = set(kept)
        sub = [i for i, s in enumerate(self.species) if need <= s]
        sp = PT.Pool(teams=[self.pool.teams[i] for i in sub], sha256="", index_of={})
        return [sub[j] for j in PW.candidates(sp, kept)]


def build_decoys(root, corpus: Corpus, PT, PW, W, sampler, did: str) -> dict:
    """Up to D off-pool decoys of this root (module docstring), with the draw's record. Never raises for a missing one."""
    from agents.battle.core_replay import run_core

    def gate(res, _c=None):
        return PW.strict_verdict(root, res), 0.0, None

    match = sampler._roll_match(root, did + "#decoy", lambda t: W.render(root.tmpl, t), gate)
    kept = {s: set(m) for s, m in root.kept.items()}
    cands = corpus.candidates(PT, PW, kept)
    tk = PT.team_key(root.team)
    meta: dict = {"candidates": len(cands), "is_truth": 0, "near_copy": 0, "near_copy_loose": 0, "dup": 0, "fails": {}}
    seen_keys = set()
    ok_c: List[int] = []
    for c in [cands[i] for i in _rng(E.DECOY_SALT + did).permutation(len(cands))]:
        t = corpus.pool.teams[c]
        k = PT.team_key(t)
        if k == tk:
            meta["is_truth"] += 1
            continue
        if shared(PT.set_key, t, root.team) >= E.NEAR:
            meta["near_copy"] += 1
            continue
        if shared(loose_key, t, root.team) >= E.NEAR:
            meta["near_copy_loose"] += 1
            continue
        if k in seen_keys:
            meta["dup"] += 1
            continue
        seen_keys.add(k)
        ok_c.append(c)
    meta["eligible"] = len(ok_c)
    out_t: List[str] = []
    out_c: List[int] = []
    out_r: List[dict] = []
    out_lw: List[float] = []
    gated = 0
    for s0 in range(0, min(len(ok_c), E.MAX_GATED), E.MAX_TRIES):
        if len(out_t) >= E.D:
            break
        built = []
        for c in ok_c[s0:min(s0 + E.MAX_TRIES, E.MAX_GATED)]:
            tw, why = PW.build_world(root, corpus.pool.teams[c], _rng(f"{E.DECOY_SALT}{did}:cand:{c}"), sampler.pad)
            if tw is None:
                meta["fails"][why] = meta["fails"].get(why, 0) + 1
            else:
                built.append((c, tw))
        gated += min(s0 + E.MAX_TRIES, E.MAX_GATED, len(ok_c)) - s0
        if not built:
            continue
        res = run_core([root.rec(tw, f"{did}#decoy{s0 + i}", W.render(root.tmpl, tw), match)
                        for i, (_, tw) in enumerate(built)], trackers=True, obs=True)
        for (c, tw), r in zip(built, res):
            if len(out_t) >= E.D:
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
    meta.update(n=len(out_t), corpus_index=out_c, corpus_id=[corpus.ids[c] for c in out_c], log_w=out_lw,
                shared_sets=[shared(PT.set_key, t, root.team) for t in out_t],
                shared_loose=[shared(loose_key, t, root.team) for t in out_t])
    return {"teams": out_t, "logs": out_r, "meta": meta}


def one(did: str, d: dict, bank, st: dict, sampler, fwd, corpus, mods, head: str, g4: bool) -> dict:
    from main.policy_spectrum.truth import cmd_index, to_log
    from main.value_search import opp_likelihood as OL
    from main.value_search.worlds import WorldError, prepare_root

    PT, PW, W, L3R = mods
    b = bank.battles[bank.battle_index[d["battle"]]]
    side = d["side"]
    at = cmd_index(b, side, d["n"])
    log = to_log(b)
    row = {"id": did, "battle": d["battle"], "stratum": st["stratum"], "turn": st.get("turn"), "side": side, "at": at,
           "head": head, "J": E.J, "D_max": E.D, "opp_team": st["opp_team"], "outcome": st["outcome"],
           "k_bin": st["k_bin"]}
    w0 = time.monotonic()
    try:
        Wd = sampler.sample(log, at, side, E.J, key=did, screen_cache={})
    except WorldError as exc:
        return dict(row, ok=False, error=f"world build: {type(exc).__name__}: {str(exc)[:400]}",
                    wall_s=round(time.monotonic() - w0, 2))
    try:
        root = prepare_root(log, at, side, did)
        dec = build_decoys(root, corpus, PT, PW, W, sampler, did)
        steps = OL.opponent_steps(root)
        tlog = OL.truth_log(root)
        res = OL.world_logliks(root, [tlog] + list(Wd.logs) + dec["logs"], fwd, steps, label=did)
        mt = [L3R.matches(root.team, t, Wd.revealed) for t in list(Wd.teams) + dec["teams"]]
        row.update(
            ok=True, ll=[r.ll for r in res], conf=[r.conf for r in res], n_decoys=len(dec["teams"]),
            decoy=dec["meta"], logp_truth=res[0].logp, n_scored=res[0].n_scored,
            n_informative=[r.n_informative for r in res], n_unobserved=res[0].n_unobserved,
            n_pending=res[0].n_pending, steps=[[s.k, s.cmd, s.status] for s in steps], log_w=list(Wd.log_w),
            match_m=[m["match_m"] for m in mt], exact_m=[m["exact_m"] for m in mt], hit_u=[m["hit_u"] for m in mt],
            n_unrev=mt[0]["n_unrev"], n_never=mt[0]["n_never"], tries=Wd.tries,
            enc_pin_equal=L3R.pin_equal(root, steps, tlog) if g4 else None)
    except Exception as exc:                             # a guard raised: a DEFECT, recorded and printed
        row.update(ok=False, defect=True, error=f"{type(exc).__name__}: {str(exc)[:400]}",
                   trace=traceback.format_exc()[-1500:])
        print(f"[offpool] DEFECT {did}: {row['error'][:300]}", flush=True)
    row["wall_s"] = round(time.monotonic() - w0, 2)
    return row


def load_model():
    from main.policy_spectrum.qhat import check_winprob
    from main.policy_spectrum.reader import load_checkpoint

    p = E.ARCH / "ckpt" / f"checkpoint_{E.PSTAR_STEP}_steps.zip"
    h = E.sha256_file(p)
    if h != E.PSTAR_SHA:
        raise SystemExit(f"{p}: sha256 {h} differs from P*'s {E.PSTAR_SHA}")
    m = load_checkpoint(p, device="cpu")
    check_winprob(m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--recheck", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget-min", type=float, default=0.0, help="stop after the state in progress past this")
    ap.add_argument("--smoke", action="store_true", help="ENGINEERING SMOKE on the smoke bank (never the registered one)")
    a = ap.parse_args()
    code = E.use_code(Path(a.code))
    os.environ.setdefault("L3_PIN_CORE_EVENTS", os.environ.get("OFFPOOL_PIN_CORE_EVENTS", ""))
    if not os.environ["L3_PIN_CORE_EVENTS"]:
        os.environ.pop("L3_PIN_CORE_EVENTS")
    import l3_run as L3R  # noqa: E402  (L3's own driver: its match meter and G4's encoder comparison)
    import pool_teams as PT  # noqa: E402
    import pool_worlds as PW  # noqa: E402
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.qhat import logits_and_values
    from main.policy_spectrum.reader import inference_globals
    from main.value_search import worlds as W
    from main.value_search.worlds import WorldSampler

    corpus = Corpus(PT, W)
    doc = json.loads(E.states_path(a.smoke).read_text())
    st = doc["states"]
    ids = sorted(st, key=lambda x: hashlib.sha256((E.STATES_SALT + "order:" + x).encode()).hexdigest())
    sub = E.ARCH / "smoke" if a.smoke else E.ARCH
    if a.recheck:
        ids = sorted(ids, key=lambda x: hashlib.sha256((E.RECHECK_SALT + x).encode()).hexdigest())[:a.recheck]
        path = sub / "recheck.jsonl"
    else:
        path = sub / "rows.jsonl"
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
    bank = load_bank(sub / "bank")
    if bank.manifest.get("content_sha256") != doc["bank_content_sha256"]:
        raise SystemExit("the bank's content sha256 differs from states_offpool.json")
    dec = {x["id"]: x for x in bank.decisions}
    sampler = WorldSampler(reveal="off", max_tries=E.MAX_TRIES, dice="rolls")
    print(f"[offpool] {len(ids)} states to {path.name} at {head[:8]} (corpus {corpus.sha256[:8]}, "
          f"{len(corpus.pool.teams)} teams)", flush=True)
    t0 = time.monotonic()
    with inference_globals(1):
        model = load_model()

        def fwd(rows, masks):
            return logits_and_values(model, rows, masks, batch=1024)

        with open(path, "a") as fo:
            for k, did in enumerate(ids):
                row = one(did, dec[did], bank, st[did], sampler, fwd, corpus, (PT, PW, W, L3R), head,
                          g4=not a.recheck and "L3_PIN_CORE_EVENTS" in os.environ)
                row["ckpt_step"] = E.PSTAR_STEP
                row["corpus_sha256"] = corpus.sha256
                fo.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                fo.flush()
                os.fsync(fo.fileno())
                el = (time.monotonic() - t0) / 60
                print(f"[offpool] {k + 1}/{len(ids)} ok={row.get('ok')} decoys={row.get('n_decoys')} "
                      f"cands={(row.get('decoy') or {}).get('candidates')} wall={row.get('wall_s')}s {el:.1f} min",
                      flush=True)
                if a.budget_min and el > a.budget_min:
                    print("[offpool] BUDGET: stopping; resume to continue", flush=True)
                    return
    print("[offpool] done", flush=True)


if __name__ == "__main__":
    main()
