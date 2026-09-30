"""GROUND TRUTH on a bank subset (`gen3_policy_spectrum_truth_v1`, M5 Lane S gate ④).

For each turn of a fixed subset, every legal action is branched and played to the END with COMMON
RANDOM NUMBERS (Lane I's ``utils.rust_env.successors.play_out``: S dice seeds shared by every action,
reseeded at the branch point), both sides continuing under ONE declared greedy policy. An action's
value is its mean outcome over the seeds from the banked side's view (+1 win, −1 loss — a stall
forfeit included —, 0 tie or truncated).

⚠️ **The value is CONDITIONAL on the continuation**: "this action, then the continuation policy's
greedy play on both sides". It is not a Nash value and not a value against the opponent the turn
was recorded against. The continuation is stamped on every row.

Readouts (:func:`readout`), per policy, on the turns' values:

- ``V*`` = the best action's mean value; the GAP of action ``a`` = the paired (per-seed) mean of
  ``o[a*] − o[a]`` with its standard error.
- NEAR-BEST: gap ≤ ε. DOMINATED: gap − 1.96·SE > ε (separably worse than the best by more than ε).
  UNCERTAIN: the rest (neither near-best nor separably worse).
- mass on near-best / uncertain / dominated; STARVATION = a turn where some near-best action gets
  < 1 % of the policy's mass (counted per category of that action, and a strict variant that needs
  the action to be near-best by its upper bound, gap + 1.96·SE ≤ ε); regret = V* − Σ π(a) v(a);
  GUESS turns (≥ 2 near-best actions of different categories): the policy's mass split among them.

``V*`` is the max of noisy means, so it is biased UP (the winner's curse); CRN shrinks the gaps'
noise, not that bias. Everything is reported with S and ε beside it.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from main.policy_spectrum.bank import Bank

TRUTH_SCHEMA = "gen3_policy_spectrum_truth_v1"
SUBSET_SEED = "m5-laneS-gt-v1"
EPS = 0.1
STARVE = 0.01
Z = 1.96


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ---------------------------------------------------------------------------------------------
# the subset
# ---------------------------------------------------------------------------------------------

def select_subset(bank: Bank, n_free: int = 200, n_forced: int = 20, per_battle: int = 2,
                  min_cat: int = 30, seed: str = SUBSET_SEED) -> List[str]:
    """A fixed, stratified draw of decision ids: free turns proportional over phase × opponent class
    (hash order within each), at most ``per_battle`` turns per battle, topped up until hazard /
    setup / recovery / status are each legal on ≥ ``min_cat`` of the free turns; plus ``n_forced``
    forced switches."""
    order = sorted(range(len(bank.decisions)), key=lambda i: _h(f"{seed}:{bank.decisions[i]['id']}"))
    free = [i for i in order if bank.decisions[i]["kind"] == "free"]
    strata: Dict[str, int] = {}
    for i in free:
        d = bank.decisions[i]
        strata[f"{d['phase']}|{d['opp_class']}"] = strata.get(f"{d['phase']}|{d['opp_class']}", 0) + 1
    quota = {k: max(3, round(n_free * v / len(free))) for k, v in strata.items()}
    per_b: Dict[str, int] = {}
    chosen: List[int] = []
    taken: Dict[str, int] = {}

    def take(i: int) -> bool:
        d = bank.decisions[i]
        if per_b.get(d["battle"], 0) >= per_battle or i in chosen_set:
            return False
        chosen.append(i)
        chosen_set.add(i)
        per_b[d["battle"]] = per_b.get(d["battle"], 0) + 1
        return True

    chosen_set: set = set()
    for i in free:
        d = bank.decisions[i]
        k = f"{d['phase']}|{d['opp_class']}"
        if taken.get(k, 0) < quota[k] and take(i):
            taken[k] = taken.get(k, 0) + 1
        if len(chosen) >= n_free:
            break
    for cat in ("hazard", "setup", "recovery", "status"):
        have = sum(1 for i in chosen if cat in bank.decisions[i]["cats"].values())
        for i in free:
            if have >= min_cat:
                break
            if cat in bank.decisions[i]["cats"].values() and take(i):
                have += 1
    forced = [i for i in order if bank.decisions[i]["kind"] == "forced_switch"]
    n = 0
    for i in forced:
        if n >= n_forced:
            break
        if take(i):
            n += 1
    return [bank.decisions[i]["id"] for i in sorted(chosen)]


# ---------------------------------------------------------------------------------------------
# branching
# ---------------------------------------------------------------------------------------------

def to_log(battle) -> dict:
    """A bank battle as Lane I's core INPUT LOG (``successors.record_to_log``'s shape)."""
    cmds = [f"FORCELOSE {c[1]}" if c[0] == "forcelose" else f"CHOOSE {c[0]} {c[1]}"
            for c in battle.commands]
    return {"format_id": battle.format_id, "seed": battle.seed,
            "names": [battle.p1["name"], battle.p2["name"]],
            "teams": [battle.p1["team"], battle.p2["team"]], "cmds": cmds}


def cmd_index(battle, side: str, n: int) -> int:
    """The command index of ``side``'s ``n``-th answered decision (its ``n``-th CHOOSE)."""
    k = -1
    for j, c in enumerate(battle.commands):
        if c[0] == side:
            k += 1
            if k == n:
                return j
    raise IndexError(f"{battle.battle_id}: {side} has no decision {n}")


def turn_seeds(decision_id: str, s: int) -> List[str]:
    return [f"sodium,{_h(f'{SUBSET_SEED}:{decision_id}:{k}')[:32]}" for k in range(s)]


def branch_turn(bank: Bank, did: str, policy, seeds: Sequence[str], continuation: str,
                core=None, max_turns: int = 999) -> dict:
    """One durable row: every legal action × ``seeds`` played to the end."""
    from utils.rust_env.successors import SuccessorsError, play_out

    d = next(x for x in bank.decisions if x["id"] == did)
    b = bank.battles[bank.battle_index[d["battle"]]]
    at = cmd_index(b, d["side"], d["n"])
    if b.commands[at][1] != d["played"]:
        raise RuntimeError(f"{did}: command {at} is {b.commands[at]!r}, the bank played {d['played']!r}")
    row = {"schema": TRUTH_SCHEMA, "id": did, "at": at, "side": d["side"], "seeds": list(seeds),
           "continuation": continuation, "max_turns": max_turns, "stall": "production"}
    t0 = time.monotonic()
    try:
        res = play_out(to_log(b), at, d["side"], policy=policy, seeds=list(seeds), stall="production",
                       max_turns=max_turns, core=core)
    except SuccessorsError as exc:                 # F-LI-4: request vs port refusal not typed yet
        row.update({"ok": False, "error": str(exc)[:500], "wall_s": time.monotonic() - t0})
        return row
    tokens = {str(k): v for k, v in sorted(res.tokens.items())}
    if tokens != d["tokens"]:
        row.update({"ok": False, "error": f"root tokens {tokens} != banked {d['tokens']}",
                    "wall_s": time.monotonic() - t0})
        return row
    outcomes: Dict[str, List[float]] = {}
    ends: Dict[str, List[dict]] = {}
    for br in sorted(res.branches, key=lambda x: (int(x["action"]), int(x["seed"]))):
        outcomes.setdefault(str(int(br["action"])), []).append(res.value(br))
        ends.setdefault(str(int(br["action"])), []).append(
            {k: br["end"].get(k) for k in ("winner", "forfeit", "truncated", "turn")})
    row.update({"ok": True, "outcomes": outcomes, "ends": ends, "answered": res.answered,
                "batches": res.batches, "wall_s": round(time.monotonic() - t0, 3)})
    return row


def run(bank: Bank, ids: Sequence[str], policy, continuation: str, out: Path, s: int = 16,
        log: Callable[[str], None] = print, core_factory: Optional[Callable] = None) -> int:
    """Branch every id not already in ``out`` (JSONL, one durable row per turn, fsync'd):
    resumable, incremental. Returns the number of rows written."""
    from utils.rust_env.successors import SearchCore

    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("continuation") == continuation and len(r.get("seeds", [])) == s:
                    done.add(r["id"])
    todo = [i for i in ids if i not in done]
    log(f"[truth] {len(done)} turns already done, {len(todo)} to go (S = {s}, {continuation})")
    wrote = 0
    with (core_factory or SearchCore)() as core, open(out, "a") as f:
        for k, did in enumerate(todo):
            row = branch_turn(bank, did, policy, turn_seeds(did, s), continuation, core=core)
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
            wrote += 1
            if k % 10 == 0 or not row["ok"]:
                log(f"[truth] {k + 1}/{len(todo)} {did} ok={row['ok']} "
                    f"{row.get('wall_s')} s{'' if row['ok'] else ' ' + row['error'][:160]}")
    return wrote


# ---------------------------------------------------------------------------------------------
# the readout
# ---------------------------------------------------------------------------------------------

def turn_truth(row: dict, eps: float = EPS) -> dict:
    """Per-action gap to the best (paired over seeds) and its class."""
    acts = sorted(row["outcomes"], key=int)
    o = np.array([row["outcomes"][a] for a in acts], dtype=np.float64)       # [A, S]
    means = o.mean(axis=1)
    best = int(np.argmax(means))
    diff = o[best][None, :] - o                                               # [A, S]
    gap = diff.mean(axis=1)
    s = o.shape[1]
    se = diff.std(axis=1, ddof=1) / np.sqrt(s) if s > 1 else np.zeros(len(acts))
    near = gap <= eps
    strict_near = gap + Z * se <= eps
    dominated = gap - Z * se > eps
    return {"actions": [int(a) for a in acts], "value": means, "vstar": float(means[best]),
            "gap": gap, "se": se, "near": near, "strict_near": strict_near, "dominated": dominated}


def readout(bank: Bank, rows: Sequence[dict], probs: np.ndarray, eps: float = EPS,
            decisive_only: bool = False) -> dict:
    """One policy's ground-truth readout on the branched turns (``probs`` in bank order).
    ``decisive_only``: only turns where some action is DOMINATED (separably worse than the best) —
    on a turn where every action is near-best (a won position, say), concentrating the mass on one
    of them starves nothing that matters."""
    from main.policy_spectrum.spectrum import _boot_ci

    idx = {d["id"]: i for i, d in enumerate(bank.decisions)}
    per = {"near": [], "unc": [], "dom": [], "starved": [], "starved_strict": [], "regret": [],
           "battle": [], "argmax_near": []}
    starved_by_cat: Dict[str, List[int]] = {}
    near_by_cat: Dict[str, int] = {}
    guess_split: List[float] = []
    n_guess = 0
    refused = 0
    for r in rows:
        if not r.get("ok"):
            refused += 1
            continue
        i = idx[r["id"]]
        d = bank.decisions[i]
        t = turn_truth(r, eps)
        if decisive_only and not t["dominated"].any():
            continue
        p = probs[i, t["actions"]]
        cats = [d["cats"][str(a)] for a in t["actions"]]
        per["near"].append(float(p[t["near"]].sum()))
        per["dom"].append(float(p[t["dominated"]].sum()))
        per["unc"].append(float(p[~t["near"] & ~t["dominated"]].sum()))
        starved = t["near"] & (p < STARVE)
        per["starved"].append(float(starved.any()))
        per["starved_strict"].append(float((t["strict_near"] & (p < STARVE)).any()))
        per["regret"].append(float(t["vstar"] - (p * t["value"]).sum()))
        per["battle"].append(d["battle"])
        per["argmax_near"].append(float(t["near"][int(np.argmax(p))]))
        for c in set(cats[j] for j in np.flatnonzero(t["near"])):
            near_by_cat[c] = near_by_cat.get(c, 0) + 1
            hit = any(cats[j] == c and starved[j] for j in range(len(cats)))
            starved_by_cat.setdefault(c, []).append(int(hit))
        near_cats = {cats[j] for j in np.flatnonzero(t["near"])}
        if len(near_cats) >= 2:
            n_guess += 1
            pn = p[t["near"]]
            guess_split.append(float(1.0 - pn.max() / max(pn.sum(), 1e-12)))
    cl = np.array(per["battle"])
    out = {"turns": len(per["near"]), "refused": refused, "eps": eps, "starve_mass": STARVE}
    for k in ("near", "unc", "dom", "starved", "starved_strict", "regret", "argmax_near"):
        v = np.array(per[k])
        out[k] = {"mean": round(float(v.mean()), 6) if len(v) else None, "ci": _boot_ci(v, cl)}
    out["starved_by_near_category"] = {
        c: {"turns_near": near_by_cat[c], "starved_share": round(float(np.mean(v)), 6)}
        for c, v in sorted(starved_by_cat.items())}
    out["guess_turns"] = {"n": n_guess,
                          "mass_off_top_near": round(float(np.mean(guess_split)), 6) if guess_split else None}
    return out


def per_turn(bank: Bank, rows: Sequence[dict], probs: np.ndarray, eps: float = EPS,
             decisive_only: bool = False) -> Dict[str, np.ndarray]:
    """Per-turn readout arrays (turn order = ``rows``' ok turns): near / dom mass, starved flag,
    regret, and the battle id (for clustering)."""
    idx = {d["id"]: i for i, d in enumerate(bank.decisions)}
    out: Dict[str, list] = {"near": [], "dom": [], "starved": [], "regret": [], "battle": []}
    for r in rows:
        if not r.get("ok"):
            continue
        t = turn_truth(r, eps)
        if decisive_only and not t["dominated"].any():
            continue
        i = idx[r["id"]]
        p = probs[i, t["actions"]]
        out["near"].append(float(p[t["near"]].sum()))
        out["dom"].append(float(p[t["dominated"]].sum()))
        out["starved"].append(float((t["near"] & (p < STARVE)).any()))
        out["regret"].append(float(t["vstar"] - (p * t["value"]).sum()))
        out["battle"].append(bank.decisions[i]["battle"])
    return {k: np.array(v) for k, v in out.items()}


def paired(bank: Bank, rows: Sequence[dict], pa: np.ndarray, pb: np.ndarray, eps: float = EPS,
           decisive_only: bool = False) -> dict:
    """``b − a`` on the same branched turns, battle-clustered bootstrap."""
    from main.policy_spectrum.spectrum import _boot_ci

    A = per_turn(bank, rows, pa, eps, decisive_only)
    Bq = per_turn(bank, rows, pb, eps, decisive_only)
    out = {"turns": int(len(A["near"]))}
    for k in ("near", "dom", "starved", "regret"):
        d = Bq[k] - A[k]
        out[k] = {"delta": round(float(d.mean()), 6), "ci": _boot_ci(d, A["battle"])}
    return out


def load_rows(path: Path, continuation: Optional[str] = None) -> List[dict]:
    path = Path(path)
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt") as f:
            text = f.read()
    else:
        text = path.read_text()
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [r for r in rows if continuation is None or r["continuation"] == continuation]


def value_summary(rows: Iterable[dict], eps: float = EPS) -> dict:
    """Policy-free facts about the truth itself: how many actions are near-best, how often V* is a
    certain win / loss, how many turns are genuine guesses."""
    n_near, n_legal, vstar, flat = [], [], [], 0
    for r in rows:
        if not r.get("ok"):
            continue
        t = turn_truth(r, eps)
        n_near.append(int(t["near"].sum()))
        n_legal.append(len(t["actions"]))
        vstar.append(t["vstar"])
        flat += int(np.all(t["gap"] <= eps))
    v = np.array(vstar)
    return {"turns": len(v), "mean_near_best": round(float(np.mean(n_near)), 3),
            "mean_legal": round(float(np.mean(n_legal)), 3),
            "all_actions_near_best": flat,
            "vstar_eq_+1": int((v >= 1.0).sum()), "vstar_eq_-1": int((v <= -1.0).sum()),
            "mean_vstar": round(float(v.mean()), 4)}


# ---------------------------------------------------------------------------------------------
# CLI:  python -m main.policy_spectrum.truth select|run|read
# ---------------------------------------------------------------------------------------------

def _cli(argv=None) -> int:
    import argparse
    import sys

    from main.policy_spectrum.bank import load_bank

    ap = argparse.ArgumentParser(prog="python -m main.policy_spectrum.truth")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--bank", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n-free", type=int, default=200)
    s.add_argument("--n-forced", type=int, default=20)
    r = sub.add_parser("run")
    r.add_argument("--bank", required=True)
    r.add_argument("--subset", required=True)
    r.add_argument("--continuation", required=True, help="<checkpoint .zip>=<label>")
    r.add_argument("--out", required=True, help="rows JSONL (appended; resumable)")
    r.add_argument("--seeds", type=int, default=16)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--lib-profile", default="release")
    q = sub.add_parser("read")
    q.add_argument("--bank", required=True)
    q.add_argument("--rows", required=True)
    q.add_argument("--reads", required=True, help="the reader's output dir (<label>.probs.npz)")
    q.add_argument("--labels", required=True)
    q.add_argument("--eps", type=float, default=EPS)
    q.add_argument("--out", required=True)
    q.add_argument("--pair", action="append", help="<label a>:<label b>")
    a = ap.parse_args(argv)
    bank = load_bank(Path(a.bank))
    if a.cmd == "select":
        ids = select_subset(bank, n_free=a.n_free, n_forced=a.n_forced)
        Path(a.out).write_text(json.dumps({"schema": TRUTH_SCHEMA, "seed": SUBSET_SEED,
                                           "bank": bank.manifest["content_sha256"],
                                           "n_free": a.n_free, "n_forced": a.n_forced,
                                           "ids": ids}, indent=1) + "\n")
        print(f"[truth] {len(ids)} turns -> {a.out}")
        return 0
    if a.cmd == "run":
        from main.policy_spectrum.reader import load_checkpoint, policy_logits
        from utils.rust_env import ffi as F
        from utils.rust_env.successors import greedy

        sub_ = json.loads(Path(a.subset).read_text())
        if sub_["bank"] != bank.manifest["content_sha256"]:
            sys.exit("[truth] the subset was drawn from a different bank")
        path, _, label = a.continuation.partition("=")
        model = load_checkpoint(Path(path), a.threads)
        pol = greedy(lambda rows, masks: policy_logits(model, rows, masks.astype(bool)))
        lib = F.load(F.default_path(a.lib_profile))

        from utils.rust_env.successors import SearchCore

        run(bank, sub_["ids"], pol, f"{label or Path(path).stem} greedy (both sides)", Path(a.out),
            s=a.seeds, core_factory=lambda: SearchCore(lib=lib))
        return 0
    from main.policy_spectrum.reader import load_probs

    rows = load_rows(Path(a.rows))
    out = {"schema": TRUTH_SCHEMA, "eps": a.eps, "rows": len(rows),
           "continuations": sorted({r["continuation"] for r in rows}),
           "seeds": sorted({len(r["seeds"]) for r in rows}),
           "truth": value_summary(rows, a.eps), "policies": {}}
    for lab in [x for x in a.labels.split(",") if x]:
        pr = load_probs(Path(a.reads), lab, bank)
        out["policies"][lab] = readout(bank, rows, pr, a.eps)
        out.setdefault("policies_decisive", {})[lab] = readout(bank, rows, pr, a.eps, decisive_only=True)
    for spec in a.pair or []:
        la, lb = spec.split(":", 1)
        pa, pb = load_probs(Path(a.reads), la, bank), load_probs(Path(a.reads), lb, bank)
        out.setdefault("paired", {})[spec] = {
            "all": paired(bank, rows, pa, pb, a.eps),
            "decisive": paired(bank, rows, pa, pb, a.eps, decisive_only=True)}
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"[truth] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
