"""ONE-PLY COUNTERFACTUAL Q FROM THE CRITIC (`gen3_x4_preread_qhat_v1`) — the X4 pre-read.

The question (owner, 2026-09-30): Lane S found the ai_v14 policy STARVES near-best moves (< 1 % mass
on an action whose ground-truth value is within ε of the best, on about half of the decisive turns).
Does the CRITIC see those moves? One-ply counterfactual Q is

    Q̂_V(s, a) = E_seeds[ terminal reward if the game ends before our next decision,
                         else V(s′_a) ]          on the ±1 scale (V ∈ [0, 1] → 2V − 1)

with V the checkpoint's own win-prob critic evaluated on the TRAINEE'S OWN observation of the
successor — the row the training path would hand it at its next decision. If Q̂_V ranks a starved
near-best action near the top, one-ply labels (X4, ``designs/endstate/design_q_head.md``) carry a
gradient the policy does not get today.

How a turn is branched (Lane I's playout core, ``utils.rust_env.successors`` — driven here one
decision at a time, never to the end):

* every legal action of the banked side × S dice seeds shared by every action (COMMON RANDOM
  NUMBERS). The seeds ARE Lane S truth's (``truth.turn_seeds(id, 64)``): seed k's first turn rolls
  the same dice as the ground-truth playout k, so a branch's successor is exactly the state that
  truth playout continued from whenever the opponent's root action agrees.
* THE OPPONENT'S ROOT ACTION, two variants (``variant``):
  - ``R`` (recorded): held at the action the opponent PLAYED in the recorded turn. When the log has
    the opponent answering after us (its request still open at the branch point), its recorded
    command is moved in front of ours — ``prefix + [opp command]``, branch at ``at + 1`` — and the
    root is checked to have NO open opponent decision and our banked tokens.
  - ``M`` (matched to truth's construction): the opponent's open root decision is answered by the
    evaluated checkpoint's GREEDY choice — exactly what the ground truth did (its playout answered
    it with the continuation's greedy). Written only where the root has an open opponent decision;
    elsewhere M ≡ R.
* between our root action and our NEXT decision, any opponent-only decision (its replacement after
  a faint) is answered by the evaluated checkpoint's greedy choice; counted per branch.
* a branch that ENDS first scores the terminal reward (+1 win, −1 loss — a stall forfeit included —,
  0 tie / truncated: truth's convention); otherwise the captured row is scored by V.
* our next decision may be switch-only (our mon fainted, or a mid-turn switch such as Baton Pass):
  flagged per branch, scored by V like any other.

Every turn is one durable, fsync'd JSONL row (resumable). ⚠️ The policy is never run to the end
here; a branch that is captured is abandoned (the next ``open`` resets the handle's table).
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from main.policy_spectrum.bank import Bank
from main.policy_spectrum.truth import cmd_index, to_log, turn_seeds

QHAT_SCHEMA = "gen3_x4_preread_qhat_v1"
ACT = 11
MAX_STEPS = 64

#: ``scorer(rows [k, OBS], masks [k, 11] bool) -> [k, 12]``: the 11 policy logits, then V ∈ [0, 1].
Scorer = Callable[[np.ndarray, np.ndarray], np.ndarray]


class QhatError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------
# the forward: policy logits AND the critic, one extractor pass
# ---------------------------------------------------------------------------------------------

def logits_and_values(model, rows: np.ndarray, masks: np.ndarray, batch: int = 1024) -> np.ndarray:
    """``[N, 12]`` float32: the raw policy logits (masking is the caller's) and the critic's V — for
    a win-prob-critic checkpoint ``sigmoid(win_head)`` = P(win) ∈ [0, 1] — from ONE extractor
    forward (the policy's own ``forward`` wiring: ``_critic_value`` + ``_get_action_dist_from_latent``
    read the same stash)."""
    import torch as th

    from agents.model.extra_obs_keys import zero_extra_obs

    pol = model.policy
    dev = next(pol.parameters()).device
    out = []
    for i in range(0, len(rows), batch):
        mb = th.tensor(masks[i:i + batch].astype(np.float32), device=dev)
        ob = {"observation": th.tensor(rows[i:i + batch], device=dev), "action_mask": mb}
        ob.update(zero_extra_obs(pol.features_extractor, batch=len(mb), device=dev))
        with th.no_grad():
            pi_f, vf_f = pol.extract_features(ob)
            lat_pi = pol.mlp_extractor.forward_actor(pi_f)
            lat_vf = pol.mlp_extractor.forward_critic(vf_f)
            v = pol._critic_value(lat_vf).reshape(-1, 1)
            logits = pol._get_action_dist_from_latent(lat_pi).distribution.logits
        out.append(th.cat([logits, v], dim=1).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float32)


def check_winprob(model) -> str:
    """The critic mode — REFUSES anything but ``winprob`` (the 2V − 1 mapping assumes P(win)), and a
    checkpoint whose critic reads a privileged extra obs key (a V re-forwarded from the row alone
    would be V stripped of it)."""
    from agents.model.critic_mode import is_winprob
    from agents.model.extra_obs_keys import required_extra_obs_keys

    mode = getattr(model.policy, "_critic_mode", None)
    if not is_winprob(mode):
        raise QhatError(f"critic mode {mode!r}: this pre-read maps V ∈ [0, 1] to 2V − 1 and needs 'winprob'")
    extra = required_extra_obs_keys(model.policy.features_extractor)
    if extra:
        raise QhatError(f"the extractor reads extra obs keys {[k.key for k in extra]}: V from the row "
                        "alone is not this checkpoint's critic")
    return str(mode)


def greedy_of(out: np.ndarray, masks: np.ndarray) -> np.ndarray:
    lg = np.where(masks.astype(bool), out[:, :ACT].astype(np.float64), -np.inf)
    return lg.argmax(axis=1).astype(np.int32)


# ---------------------------------------------------------------------------------------------
# the one-ply driver
# ---------------------------------------------------------------------------------------------

def _open(core, req: dict) -> dict:
    return core._cstr(core.lib.rust_env_playout_open, json.dumps(req).encode())


def one_ply(core, log: dict, at: int, side: str, seeds: Sequence[Optional[str]],
            opp_answer: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
            actions: Optional[Sequence[int]] = None, stall: Optional[dict] = "production",
            timing: Optional[dict] = None) -> dict:
    """Branch ``side``'s decision at command ``at`` — every legal action (or ``actions``) × ``seeds``
    — and advance each branch only to ``side``'s NEXT decision (or the end). ``opp_answer(rows,
    masks, j)`` answers the opponent's pending decisions of still-uncaptured branches (``j`` = the
    count of opponent decisions that branch has already answered since the branch point).

    Returns the root (``turn``, ``other_open``, ``tokens``) and, per branch (action-major, seed-minor),
    ``action``, ``seed``, the captured ``row`` (float32, or None when the branch ended first),
    ``mask``, ``end`` (winner / forfeit / truncated / turn, or None), ``opp_dec`` (opponent decisions
    answered before capture) and ``switch_only`` (the captured decision allows switches only)."""
    if stall == "production":
        from agents.training.stall import StallConfig

        stall = {"turn_limit": int(StallConfig().threshold), "side": side}
    me = 0 if side == "p1" else 1
    t0 = time.perf_counter()
    req = {"log": log, "at": int(at), "side": side,
           "actions": None if actions is None else [int(a) for a in actions],
           "seeds": list(seeds), "stall": stall, "max_turns": 999, "keep_cmds": False}
    root = _open(core, req)
    nb = int(root["branches"])
    cap = 2 * nb
    rows = np.empty((cap, core.obs_dim), dtype=np.float32)
    masks = np.empty((cap, ACT), dtype=np.uint8)
    who = np.empty(cap, dtype=np.uint32)
    acts = np.zeros(cap, dtype=np.int32)
    got_row: List[Optional[np.ndarray]] = [None] * nb
    got_mask: List[Optional[np.ndarray]] = [None] * nb
    opp_dec = np.zeros(nb, dtype=np.int64)
    n = 0
    steps = 0
    sim_s = 0.0
    opp_s = 0.0
    while True:
        ts = time.perf_counter()
        with core._lock:
            k = core.lib.rust_env_playout_step(core._h, acts.ctypes.data, n, rows.ctypes.data,
                                               masks.ctypes.data, who.ctypes.data, cap)
            if k == ctypes.c_size_t(-1).value:
                from utils.rust_env.successors import _raise

                _raise(core.lib)
        sim_s += time.perf_counter() - ts
        if k == 0:
            break
        steps += 1
        if steps > MAX_STEPS:
            raise QhatError(f"no next decision after {MAX_STEPS} steps")
        b = (who[:k] // 2).astype(np.int64)
        s = (who[:k] % 2).astype(np.int64)
        ans = np.empty(k, dtype=np.int32)
        need = []
        for i in range(k):
            bi = int(b[i])
            if s[i] == me:
                if got_row[bi] is None:
                    got_row[bi] = rows[i].copy()
                    got_mask[bi] = masks[i].copy()
                ans[i] = int(np.flatnonzero(masks[i])[0])
            elif got_row[bi] is not None:
                ans[i] = int(np.flatnonzero(masks[i])[0])
            else:
                need.append(i)
        if need:
            ix = np.array(need)
            to = time.perf_counter()
            ans[ix] = opp_answer(rows[ix], masks[ix].astype(bool), opp_dec[b[ix]])
            opp_s += time.perf_counter() - to
            opp_dec[b[ix]] += 1
        # stop as soon as no uncaptured branch is still live (captured ones are abandoned)
        live_uncaptured = {int(x) for x in b if got_row[int(x)] is None}
        if not live_uncaptured:
            break
        acts[:k] = ans
        n = k
    res = core._cstr(core.lib.rust_env_playout_results)
    br_out = []
    for bi, br in enumerate(res["branches"]):
        m = got_mask[bi]
        br_out.append({"action": int(br["action"]), "seed": int(br["seed"]), "row": got_row[bi], "mask": m,
                       "end": br["end"] if got_row[bi] is None else None, "opp_dec": int(opp_dec[bi]),
                       "switch_only": bool(m is not None and not m[6:].any())})
        if got_row[bi] is None and br["end"] is None:
            raise QhatError(f"branch {bi}: neither captured nor ended")
    if timing is not None:
        timing["sim_s"] = timing.get("sim_s", 0.0) + sim_s
        timing["opp_fwd_s"] = timing.get("opp_fwd_s", 0.0) + opp_s
        timing["one_ply_s"] = timing.get("one_ply_s", 0.0) + time.perf_counter() - t0
        timing["steps"] = timing.get("steps", 0) + steps
    return {"turn": root["turn"], "other_open": bool(root["other_open"]),
            "tokens": {str(k): v for k, v in sorted(((int(a), t) for a, t in root["tokens"].items()))},
            "branches": br_out}


def end_value(end: dict, side: str) -> float:
    """Truth's convention: +1 win, −1 loss (a stall forfeit included), 0 tie / truncated."""
    if end["truncated"] or end["winner"] is None:
        return 0.0
    return 1.0 if int(end["winner"]) == (0 if side == "p1" else 1) else -1.0


def opp_cmds_after(battle, side: str, at: int) -> List[Tuple[int, str]]:
    """The opponent's CHOOSE commands after command ``at`` and before our next one, as (index,
    token). The first answers the request open at the branch point; a SECOND one follows when the
    sim REJECTED the first (a hidden trap: Showdown's ``maybeTrapped`` request offers a switch the
    sim then refuses, and re-requests)."""
    opp = "p2" if side == "p1" else "p1"
    out: List[Tuple[int, str]] = []
    for j in range(at + 1, len(battle.commands)):
        c = battle.commands[j]
        if c[0] == opp:
            out.append((j, c[1]))
        else:
            break
    return out


def recorded_root_log(core, log: dict, at: int, side: str, battle) -> Optional[Tuple[dict, int, List[str]]]:
    """Variant R's root: the log with the opponent's recorded answer(s) to its open root request
    moved in front of our command — one command at a time until the root has no open opponent
    decision. None when the log holds no such answer."""
    opp = "p2" if side == "p1" else "p1"
    cmds = opp_cmds_after(battle, side, at)
    for m in range(1, len(cmds) + 1):
        lg = dict(log, cmds=log["cmds"][:at] + [f"CHOOSE {opp} {t}" for _, t in cmds[:m]])
        req = {"log": lg, "at": at + m, "side": side, "actions": None, "seeds": [None], "stall": None,
               "max_turns": 999, "keep_cmds": False}
        if not _open(core, req)["other_open"]:
            return lg, at + m, [t for _, t in cmds[:m]]
    return None


# ---------------------------------------------------------------------------------------------
# one durable row per (turn, variant)
# ---------------------------------------------------------------------------------------------

def _score_branches(branches: List[dict], side: str, scorer: Scorer, timing: Optional[dict]) -> List[dict]:
    idx = [i for i, br in enumerate(branches) if br["row"] is not None]
    v = np.full(len(branches), np.nan)
    if idx:
        tf = time.perf_counter()
        out = scorer(np.stack([branches[i]["row"] for i in idx]),
                     np.stack([branches[i]["mask"] for i in idx]).astype(bool))
        if timing is not None:
            timing["v_fwd_s"] = timing.get("v_fwd_s", 0.0) + time.perf_counter() - tf
            timing["v_rows"] = timing.get("v_rows", 0) + len(idx)
        v[idx] = out[:, ACT]
    for i, br in enumerate(branches):
        br["v"] = None if br["row"] is None else float(v[i])
        br["q"] = end_value(br["end"], side) if br["row"] is None else 2.0 * float(v[i]) - 1.0
    return branches


def _pack(branches: List[dict], n_seeds: int) -> dict:
    """Per action: ``q`` (one per seed, ±1 scale), ``src`` (``V`` scored by the critic, else the
    terminal ``+ / - / 0``), ``opp`` (opponent decisions answered before capture, one digit per
    seed, capped 9), ``sw`` (``1`` = the captured decision is switch-only)."""
    out: Dict[str, dict] = {}
    for br in sorted(branches, key=lambda x: (x["action"], x["seed"])):
        a = out.setdefault(str(br["action"]), {"q": [], "src": "", "opp": "", "sw": ""})
        a["q"].append(round(br["q"], 6))
        a["src"] += "V" if br["row"] is not None else {1.0: "+", -1.0: "-", 0.0: "0"}[br["q"]]
        a["opp"] += str(min(9, br["opp_dec"]))
        a["sw"] += "1" if br["switch_only"] else "0"
    for a in out.values():
        if len(a["q"]) != n_seeds:
            raise QhatError(f"an action has {len(a['q'])} branches, not {n_seeds}")
    return out


def qhat_turn(bank: Bank, did: str, scorer: Scorer, core, n_seeds: int, label: str,
              timing: Optional[dict] = None) -> List[dict]:
    """The rows for one banked decision: variant R always, variant M when the root has an open
    opponent decision. A refusal is a row with ``ok`` false (counted, never dropped)."""
    from utils.rust_env.successors import SuccessorsError

    d = next(x for x in bank.decisions if x["id"] == did)
    b = bank.battles[bank.battle_index[d["battle"]]]
    side = d["side"]
    at = cmd_index(b, side, d["n"])
    if b.commands[at][1] != d["played"]:
        raise QhatError(f"{did}: command {at} is {b.commands[at]!r}, the bank played {d['played']!r}")
    seeds = turn_seeds(did, n_seeds)
    base = {"schema": QHAT_SCHEMA, "id": did, "ckpt": label, "side": side, "at": at, "n_seeds": n_seeds}

    def greedy_answer(rows, masks, j):
        return greedy_of(scorer(rows, masks), masks)

    log = to_log(b)
    rows_out = []
    try:
        first = one_ply(core, log, at, side, seeds, greedy_answer, timing=timing)
    except (SuccessorsError, QhatError) as exc:
        return [dict(base, variant="R", ok=False, error=str(exc)[:500])]
    if first["tokens"] != {str(k): v for k, v in sorted((int(a), t) for a, t in d["tokens"].items())}:
        return [dict(base, variant="R", ok=False, error=f"root tokens {first['tokens']} != banked")]
    other_open = first["other_open"]
    if other_open:
        m_row = dict(base, variant="M", ok=True, other_open=True, at_eff=at,
                     actions=_pack(_score_branches(first["branches"], side, scorer, timing), n_seeds))
        try:
            rl = recorded_root_log(core, log, at, side, b)
        except SuccessorsError as exc:
            rl = None
            r_row = dict(base, variant="R", ok=False, other_open=True, error=str(exc)[:500])
        else:
            if rl is None:
                r_row = dict(base, variant="R", ok=False, other_open=True,
                             error="the log has no opponent answer that closes the open root request")
        if rl is not None:
            log_r, at_r, opp_cmds = rl
            try:
                rr = one_ply(core, log_r, at_r, side, seeds, greedy_answer, timing=timing)
            except (SuccessorsError, QhatError) as exc:
                rr = None
                r_row = dict(base, variant="R", ok=False, other_open=True, error=str(exc)[:500])
            if rr is not None:
                if rr["tokens"] != first["tokens"]:
                    r_row = dict(base, variant="R", ok=False, other_open=True,
                                 error=f"the reordered root's tokens {rr['tokens']} are not the banked ones")
                else:
                    r_row = dict(base, variant="R", ok=True, other_open=True, at_eff=at_r, opp_cmds=opp_cmds,
                                 actions=_pack(_score_branches(rr["branches"], side, scorer, timing), n_seeds))
        rows_out += [r_row, m_row]
    else:
        rows_out.append(dict(base, variant="R", ok=True, other_open=False, at_eff=at,
                             actions=_pack(_score_branches(first["branches"], side, scorer, timing), n_seeds)))
    return rows_out


# ---------------------------------------------------------------------------------------------
# the sanity check: the PLAYED action under the battle's OWN dice reproduces the recorded next row
# ---------------------------------------------------------------------------------------------

def sanity_turn(bank: Bank, did: str, scorer: Scorer, core, replayed) -> dict:
    """Seed ``None`` (the recorded dice), our PLAYED action, the opponent answering its RECORDED
    choices (its root one in the prefix; later ones looked up in the replay's tokens): the captured
    row must be BYTE-EQUAL to the replay's row of our next decision, and V on it equal V there."""
    d = next(x for x in bank.decisions if x["id"] == did)
    b = bank.battles[bank.battle_index[d["battle"]]]
    side = d["side"]
    me, opp = (0, 1) if side == "p1" else (1, 0)
    at = cmd_index(b, side, d["n"])
    log = to_log(b)
    ours = replayed.decisions[me]
    theirs = replayed.decisions[opp]
    out = {"id": did}
    if d["n"] + 1 >= len(ours):
        return dict(out, status="no_next_decision")
    # try the log as-is first; if the opponent is open at the root, put its recorded answer in front
    # the PLAYED action's index (``rec_action`` is None on the untraced side of an exploiter's games)
    played = next(int(a) for a, t in d["tokens"].items() if t == d["played"])
    n_opp0 = sum(1 for c in b.commands[:at] if c[0] == ("p2" if side == "p1" else "p1"))

    def recorded_answer(rows, masks, j):
        res = []
        for m, jj in zip(masks, j):
            k = n_opp0 + int(jj)
            if k >= len(theirs):
                raise QhatError("the opponent has no recorded decision there")
            rd = theirs[k]
            if not np.array_equal(rd.mask.astype(bool), m.astype(bool)):
                raise QhatError(f"opponent decision {k}: pending mask != the replay's")
            inv = {t: i for i, t in rd.tokens.items()}
            res.append(inv[rd.choice])
        return np.array(res, dtype=np.int32)

    # an open opponent root decision is answered by its RECORDED choice (``recorded_answer``, j = 0)
    r = one_ply(core, log, at, side, [None], recorded_answer, actions=[played])
    br = r["branches"][0]
    if br["row"] is None:
        return dict(out, status="ended", end=br["end"])
    rec = ours[d["n"] + 1].row
    eq = bool(np.array_equal(br["row"], rec))
    o = scorer(np.stack([br["row"], np.asarray(rec, dtype=np.float32)]),
               np.stack([br["mask"].astype(bool), ours[d["n"] + 1].mask.astype(bool)]))
    return dict(out, status="captured", byte_equal=eq, v_branch=float(o[0, ACT]), v_recorded=float(o[1, ACT]),
                opp_dec=br["opp_dec"])


# ---------------------------------------------------------------------------------------------
# the run: resumable, one fsync'd row per (turn, variant)
# ---------------------------------------------------------------------------------------------

def done_ids(out: Path, label: str, n_seeds: int) -> set:
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("ckpt") == label and r.get("n_seeds") == n_seeds and r.get("variant") == "R":
                    done.add(r["id"])
    return done


def run(bank: Bank, ids: Sequence[str], scorer: Scorer, label: str, out: Path, n_seeds: int,
        core_factory: Callable, workers: int = 1, log: Callable[[str], None] = print,
        chunk: int = 64) -> int:
    done = done_ids(out, label, n_seeds)
    todo = [i for i in ids if i not in done]
    log(f"[qhat] {label}: {len(done)} turns done, {len(todo)} to go (S = {n_seeds}, {workers} worker(s))")
    local = threading.local()
    cores: List = []
    lock = threading.Lock()
    wrote = 0
    t0 = time.monotonic()

    def one(did: str) -> List[dict]:
        core = getattr(local, "core", None)
        if core is None:
            core = local.core = core_factory()
            with lock:
                cores.append(core)
        tm: dict = {}
        tw = time.perf_counter()
        rows = qhat_turn(bank, did, scorer, core, n_seeds, label, timing=tm)
        tm["wall_s"] = time.perf_counter() - tw
        for r in rows:
            r["timing"] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in tm.items()}
        return rows

    try:
        with open(out, "a") as f, ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for c0 in range(0, len(todo), chunk):
                part = todo[c0:c0 + chunk]
                for rows in ex.map(one, part):
                    with lock:
                        # R last: a turn counts as done only once every variant row is on disk
                        for r in sorted(rows, key=lambda x: x["variant"] == "R"):
                            f.write(json.dumps(r, sort_keys=True) + "\n")
                            if not r["ok"]:
                                log(f"[qhat] REFUSED {r['id']} ({r['variant']}): {r['error'][:200]}")
                        f.flush()
                        os.fsync(f.fileno())
                        wrote += 1
                el = time.monotonic() - t0
                log(f"[qhat] {label}: {c0 + len(part)}/{len(todo)} turns, {el / 60:.1f} min, "
                    f"eta {el / (c0 + len(part)) * (len(todo) - c0 - len(part)) / 60:.1f} min")
    finally:
        for c in cores:
            c.close()
    return wrote


def load_qhat(path: Path) -> List[dict]:
    path = Path(path)
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt") as f:
            text = f.read()
    else:
        text = path.read_text()
    return [json.loads(x) for x in text.splitlines() if x.strip()]


def sha(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


# ---------------------------------------------------------------------------------------------
# CLI:  python -m main.policy_spectrum.qhat run|sanity
# ---------------------------------------------------------------------------------------------

def _cli(argv=None) -> int:
    import argparse
    import sys

    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals, load_checkpoint, refuse_under_models
    from main.policy_spectrum.truth import MicroBatcher
    from utils.rust_env import ffi as F
    from utils.rust_env.successors import SearchCore

    ap = argparse.ArgumentParser(prog="python -m main.policy_spectrum.qhat")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "sanity"):
        p = sub.add_parser(name)
        p.add_argument("--bank", required=True)
        p.add_argument("--subset", required=True)
        p.add_argument("--ckpt", required=True, help="<checkpoint .zip>=<label>")
        p.add_argument("--out", required=True, help="JSONL (appended; resumable)")
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--lib-profile", default="release")
        p.add_argument("--limit", type=int, default=None, help="only the first N subset turns")
    sub.choices["run"].add_argument("--seeds", type=int, default=64)
    sub.choices["run"].add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    refuse_under_models(Path(a.out))
    bank = load_bank(Path(a.bank))
    sub_ = json.loads(Path(a.subset).read_text())
    if sub_["bank"] != bank.manifest["content_sha256"]:
        sys.exit("[qhat] the subset was drawn from a different bank")
    ids = sub_["ids"][: a.limit] if a.limit else sub_["ids"]
    path, _, label = a.ckpt.partition("=")
    label = label or Path(path).stem
    lib = F.load(F.default_path(a.lib_profile))
    with inference_globals(a.threads):
        model = load_checkpoint(Path(path))
        check_winprob(model)
        fwd = MicroBatcher(lambda rows, masks: logits_and_values(model, rows, masks))
        if a.cmd == "run":
            run(bank, ids, fwd, label, Path(a.out), a.seeds, core_factory=lambda: SearchCore(lib=lib),
                workers=a.workers, log=lambda m: print(m, flush=True))
            return 0
        from main.policy_spectrum.replay import replay

        need = sorted({next(d for d in bank.decisions if d["id"] == i)["battle"] for i in ids})
        reps = replay([bank.battles[bank.battle_index[b]].recorded() for b in need], workers=2)
        by_b = dict(zip(need, reps))
        done = set()
        outp = Path(a.out)
        if outp.exists():
            done = {r["id"] for r in (json.loads(x) for x in outp.read_text().splitlines() if x.strip())
                    if r.get("status") != "error"}
        with SearchCore(lib=lib) as core, open(outp, "a") as f:
            for i in ids:
                if i in done:
                    continue
                battle = next(d for d in bank.decisions if d["id"] == i)["battle"]
                try:
                    r = sanity_turn(bank, i, fwd, core, by_b[battle])
                except Exception as exc:                      # noqa: BLE001 - counted, never dropped
                    r = {"id": i, "status": "error", "error": str(exc)[:300]}
                f.write(json.dumps(dict(r, ckpt=label), sort_keys=True) + "\n")
                f.flush()
        print(f"[qhat] sanity -> {outp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
