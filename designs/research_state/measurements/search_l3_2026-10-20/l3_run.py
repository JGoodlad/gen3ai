"""L3, THE DRIVER (``PREREGISTRATION.md`` §2–§3): on each of the 287 registered states (the world sampler's stage-1
bank: the maturity battery's interim LEAN common states at X26 225M), build today's honest worlds (``WorldSampler(
reveal="off", dice="rolls")``, F2's legality rule on, J = 8, the studies' own call) and score the truth and every world
by the OPPONENT-CHOICE LIKELIHOOD (``main.value_search.opp_likelihood``) under X26's 225M checkpoint. Forward passes
only: no playout, no outcome is read or written.

    PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
    CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh --slot --name l3 12 \\
        designs/research_state/measurements/search_l3_2026-10-20/run.sh

One fsync'd JSONL row per state in ``$L3_ARCHIVE/rows_<shard>_of_<W>.jsonl`` (resumable; default archive
``~/gen3ai_archive/search_l3``); ``--recheck N`` re-scores N fixed states in a fresh process into ``recheck.jsonl``
(the determinism gate G3). Each row carries the 9 log-likelihoods (the truth first), the familiarity control, the step
counts, the worlds' damage-roll weights, each world's match with the truth's hidden moves and never-seen species, and
whether the truth's opponent rows are byte-identical at X26's pin (G4, ``$L3_PIN_CORE_EVENTS``).
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
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE1 = HERE.parent / "world_sampler_stage1_2026-10-19"
ARCH = Path(os.environ.get("L3_ARCHIVE") or (Path.home() / "gen3ai_archive" / "search_l3"))
J = 8
MAX_TRIES = 60
RECHECK_SALT = "search_l3_recheck_v1:"
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def hp_norm(m: str) -> str:
    """A Hidden Power's TYPE is a spread fact: the moves class compares it untyped."""
    return "hiddenpower" if m.startswith("hiddenpower") else m


def matches(true_team: str, world_team: str, revealed: Dict[str, List[str]]) -> dict:
    """The world's match with the truth on the two binding classes (``design_world_sampler.md`` §2): the share of the
    seen mons' UNREVEALED true moves the world holds (``match_m``; ``exact_m`` 1 when every seen mon's unrevealed set is
    exact) and the share of the true never-seen species the world holds among its never-seen mons (``hit_u``)."""
    from main.value_search.worlds import parse_set, species_id, typed_moves

    def mons(team: str):
        return {species_id(f): f for f in (parse_set(e) for e in team.split("]"))}

    tm, wm = mons(true_team), mons(world_team)
    kept = {s: {hp_norm(m) for m in v} for s, v in revealed.items()}
    hit = tot = 0
    exact = True
    for sid, ks in kept.items():
        if sid not in tm:
            continue
        tu = {hp_norm(m) for m in typed_moves(tm[sid])} - ks
        if not tu:
            continue
        wu = {hp_norm(m) for m in typed_moves(wm[sid])} - ks if sid in wm else set()
        hit += len(tu & wu)
        tot += len(tu)
        exact = exact and tu == wu
    never_t = {s for s in tm if s not in kept}
    never_w = {s for s in wm if s not in kept}
    return {"match_m": (hit / tot) if tot else None, "exact_m": (float(exact) if tot else None),
            "hit_u": (len(never_t & never_w) / len(never_t)) if never_t else None,
            "n_unrev": tot, "n_never": len(never_t)}


def load_model():
    sys.path.insert(0, str(STAGE1))
    import s1_common as E  # noqa: E402  (read-only: the anchor's registered checkpoint copy)
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


def pin_equal(root, steps, tlog) -> Optional[bool]:
    """G4: the truth's opponent rows at the scored steps, HEAD's core_events vs X26's pin's (byte-equal)."""
    from main.value_search import opp_likelihood as OL

    pin = os.environ.get("L3_PIN_CORE_EVENTS")
    if not pin:
        return None
    head = OL._opp_inputs(root, OL._replay(root, [tlog], "g4_head")[0], tlog, steps, 0)
    old = os.environ.get("POKESIM_CORE_EVENTS_BIN")
    os.environ["POKESIM_CORE_EVENTS_BIN"] = pin
    try:
        res = OL._replay(root, [tlog], "g4_pin")[0]
        got = OL._opp_inputs(root, res, tlog, steps, 0)
    except Exception:                                   # the pin's encoder disagrees with HEAD's rows or tokens
        return False
    finally:
        if old is None:
            os.environ.pop("POKESIM_CORE_EVENTS_BIN", None)
        else:
            os.environ["POKESIM_CORE_EVENTS_BIN"] = old
    (hx, hm, ha), (gx, gm, ga) = head, got
    return (len(hx) == len(gx) and ha == ga and all(a.tobytes() == b.tobytes() for a, b in zip(hx, gx))
            and all(a.tobytes() == b.tobytes() for a, b in zip(hm, gm)))


def one(did: str, d: dict, bank, st: dict, sampler, fwd, head: str, g4: bool) -> dict:
    from main.policy_spectrum.truth import cmd_index, to_log
    from main.value_search import opp_likelihood as OL
    from main.value_search.worlds import WorldError, prepare_root

    b = bank.battles[bank.battle_index[d["battle"]]]
    side = d["side"]
    at = cmd_index(b, side, d["n"])
    log = to_log(b)
    row = {"id": did, "battle": d["battle"], "stratum": st["stratum"], "turn": st.get("turn"), "side": side, "at": at,
           "head": head, "J": J}
    w0 = time.monotonic()
    try:
        W = sampler.sample(log, at, side, J, key=did, screen_cache={})
    except WorldError as exc:
        return dict(row, ok=False, error=f"world build: {type(exc).__name__}: {str(exc)[:400]}",
                    wall_s=round(time.monotonic() - w0, 2))
    try:
        root = prepare_root(log, at, side, did)
        steps = OL.opponent_steps(root)
        tlog = OL.truth_log(root)
        res = OL.world_logliks(root, [tlog] + list(W.logs), fwd, steps, label=did)
        mt = [matches(root.team, t, W.revealed) for t in W.teams]
        row.update(
            ok=True, ll=[r.ll for r in res], conf=[r.conf for r in res], logp_truth=res[0].logp,
            n_scored=res[0].n_scored, n_informative=[r.n_informative for r in res], n_unobserved=res[0].n_unobserved,
            n_pending=res[0].n_pending, steps=[[s.k, s.cmd, s.status] for s in steps], log_w=list(W.log_w),
            match_m=[m["match_m"] for m in mt], exact_m=[m["exact_m"] for m in mt], hit_u=[m["hit_u"] for m in mt],
            n_unrev=mt[0]["n_unrev"], n_never=mt[0]["n_never"], tries=W.tries,
            enc_pin_equal=pin_equal(root, steps, tlog) if g4 else None)
    except Exception as exc:                             # a guard raised: a DEFECT, recorded and printed
        row.update(ok=False, defect=True, error=f"{type(exc).__name__}: {str(exc)[:400]}",
                   trace=traceback.format_exc()[-1500:])
        print(f"[l3] DEFECT {did}: {row['error'][:300]}", flush=True)
    row["wall_s"] = round(time.monotonic() - w0, 2)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--recheck", type=int, default=0, help="re-score N fixed states into recheck.jsonl (G3)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import main as _m

    if not str(Path(_m.__file__).resolve()).startswith(str(REPO / "src")):
        raise SystemExit(f"main imported from {_m.__file__}, not this checkout: run with -I from this checkout")
    sys.path.insert(0, str(STAGE1))
    import s1_common as E  # noqa: E402
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.qhat import logits_and_values
    from main.policy_spectrum.reader import inference_globals
    from main.value_search.worlds import WorldSampler

    ARCH.mkdir(parents=True, exist_ok=True)
    ids = E.ordered_ids()                                  # the 287, in stage 1's registered order
    if a.recheck:
        ids = sorted(ids, key=lambda x: hashlib.sha256((RECHECK_SALT + x).encode()).hexdigest())[:a.recheck]
        path = ARCH / "recheck.jsonl"
    else:
        ids = ids[a.shard::a.of]
        path = ARCH / f"rows_{a.shard}_of_{a.of}.jsonl"
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
    head = os.popen(f"git -C {REPO} rev-parse HEAD").read().strip()
    st = E.mat_states()
    bank = load_bank(E.MAT_ARCH / "bank")
    dec = {d["id"]: d for d in bank.decisions}
    sampler = WorldSampler(reveal="off", max_tries=MAX_TRIES, dice="rolls")
    print(f"[l3] {len(ids)} states to {path.name} at {head[:8]}", flush=True)
    t0 = time.monotonic()
    with inference_globals(1):
        model, ck = load_model()

        def fwd(rows, masks):
            return logits_and_values(model, rows, masks, batch=1024)

        with open(path, "a") as fo:
            for k, did in enumerate(ids):
                row = one(did, dec[did], bank, st[did], sampler, fwd, head, g4=not a.recheck)
                row["ckpt_step"] = ck["step"]
                fo.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                fo.flush()
                os.fsync(fo.fileno())
                print(f"[l3] {k + 1}/{len(ids)} ok={row.get('ok')} {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    print("[l3] done", flush=True)


if __name__ == "__main__":
    main()
