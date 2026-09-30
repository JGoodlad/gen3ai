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


#: v2's over-sampling targets: FREE turns on which the category is LEGAL (the delayed-payoff
#: categories need ≥ ~100 turns where one of their actions is NEAR-BEST; the v1 read found a legal
#: setup move near-best on ~35 % of its turns, recovery ~46 %, hazard ~57 %).
OVERSAMPLE_V2 = {"setup": 420, "recovery": 330, "hazard": 250, "status": 500}


def select_subset_v2(bank: Bank, include: Sequence[str] = (), n_free: int = 1500, n_forced: int = 100,
                     per_battle: int = 4, oversample: Dict[str, int] = OVERSAMPLE_V2,
                     seed: str = SUBSET_SEED + "-v2") -> Dict[str, object]:
    """The at-scale subset: ``include`` (v1's turns) first, then free turns where each
    over-sampled category is legal until its target, then a stratified random remainder
    (phase × opponent class, proportional) up to ``n_free`` free turns, plus ``n_forced`` forced
    switches. Hash order everywhere; ≤ ``per_battle`` turns per battle. Returns the ids and, per
    id, WHY it was drawn (``v1`` / ``over:<category>`` / ``random`` / ``forced``) so a readout can
    re-weight the over-sampling away."""
    order = sorted(range(len(bank.decisions)), key=lambda i: _h(f"{seed}:{bank.decisions[i]['id']}"))
    pos = {d["id"]: i for i, d in enumerate(bank.decisions)}
    why: Dict[int, str] = {}
    per_b: Dict[str, int] = {}

    def take(i: int, reason: str, cap: bool = True) -> bool:
        b = bank.decisions[i]["battle"]
        if i in why or (cap and per_b.get(b, 0) >= per_battle):
            return False
        why[i] = reason
        per_b[b] = per_b.get(b, 0) + 1
        return True

    for did in include:
        take(pos[did], "v1", cap=False)
    free = [i for i in order if bank.decisions[i]["kind"] == "free"]
    for cat, target in oversample.items():
        have = sum(1 for i in why if bank.decisions[i]["kind"] == "free"
                   and cat in bank.decisions[i]["cats"].values())
        for i in free:
            if have >= target:
                break
            if cat in bank.decisions[i]["cats"].values() and take(i, f"over:{cat}"):
                have += 1
    strata: Dict[str, int] = {}
    for i in free:
        d = bank.decisions[i]
        strata[f"{d['phase']}|{d['opp_class']}"] = strata.get(f"{d['phase']}|{d['opp_class']}", 0) + 1
    n_free_now = sum(1 for i in why if bank.decisions[i]["kind"] == "free")
    room = max(0, n_free - n_free_now)
    quota = {k: round(room * v / len(free)) for k, v in strata.items()}
    got: Dict[str, int] = {}
    for i in free:
        if n_free_now >= n_free:
            break
        d = bank.decisions[i]
        k = f"{d['phase']}|{d['opp_class']}"
        if got.get(k, 0) < quota[k] and take(i, "random"):
            got[k] = got.get(k, 0) + 1
            n_free_now += 1
    for i in free:                                   # rounding / cap shortfall: fill in hash order
        if n_free_now >= n_free:
            break
        if take(i, "random"):
            n_free_now += 1
    n_forced_now = sum(1 for i in why if bank.decisions[i]["kind"] == "forced_switch")
    for i in order:
        if n_forced_now >= n_forced:
            break
        if bank.decisions[i]["kind"] == "forced_switch" and take(i, "forced"):
            n_forced_now += 1
    ids = sorted(why)
    return {"ids": [bank.decisions[i]["id"] for i in ids],
            "why": {bank.decisions[i]["id"]: why[i] for i in ids}}


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


class MicroBatcher:
    """A thread-safe scorer that merges concurrent calls into one forward. A caller enqueues its
    rows, then takes the forward lock; whoever holds it drains the WHOLE queue into one batch and
    hands each caller its slice. Results are the same function of each row, but a row's float
    rounding can depend on the batch it rode in (GPU kernels choose by size) — so a greedy argmax at
    an exact near-tie may differ from a serial run (measured: see PROGRESS)."""

    def __init__(self, fn: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        import threading

        self.fn = fn
        self.fwd = threading.Lock()
        self.qlock = threading.Lock()
        self.queue: List[dict] = []
        self.calls = 0
        self.forwards = 0

    def __call__(self, rows: np.ndarray, masks: np.ndarray) -> np.ndarray:
        import threading

        item = {"rows": rows, "masks": masks, "out": None, "done": threading.Event()}
        with self.qlock:
            self.queue.append(item)
            self.calls += 1
        with self.fwd:
            if not item["done"].is_set():
                with self.qlock:
                    batch, self.queue = self.queue, []
                try:
                    out = self.fn(np.concatenate([b["rows"] for b in batch]),
                                  np.concatenate([b["masks"] for b in batch]))
                except BaseException as exc:
                    for b in batch:
                        b["out"] = exc
                        b["done"].set()
                    raise
                self.forwards += 1
                k = 0
                for b in batch:
                    n = len(b["rows"])
                    b["out"] = out[k:k + n]
                    k += n
                    b["done"].set()
        item["done"].wait()
        if isinstance(item["out"], BaseException):
            raise RuntimeError("the batched forward failed") from item["out"]
        return item["out"]


def run(bank: Bank, ids: Sequence[str], policy, continuation: str, out: Path, s: int = 16,
        log: Callable[[str], None] = print, core_factory: Optional[Callable] = None,
        workers: int = 1, chunk: int = 32, lock_path: Optional[Path] = None) -> int:
    """Branch every id not already in ``out`` (JSONL, one durable row per turn, fsync'd):
    resumable, incremental. ``workers`` turns run concurrently (one core handle per thread; the core
    releases the GIL). With ``lock_path`` the lock is taken through ``utils.gpu_lock`` per CHUNK of
    ``chunk`` turns and released between chunks, so a shared GPU is never held for the whole run —
    and under a holder that already has it (``scripts/ops/gpu_lock.sh … truth --lock <same file>``) the
    per-chunk take is a verified no-op instead of a self-deadlock. Returns the
    number of rows written."""
    import contextlib
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from utils.gpu_lock import gpu_lock

    from utils.rust_env.successors import SearchCore

    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("continuation") == continuation and len(r.get("seeds", [])) == s:
                    done.add(r["id"])
    todo = [i for i in ids if i not in done]
    log(f"[truth] {len(done)} turns already done, {len(todo)} to go (S = {s}, {continuation}, "
        f"{workers} worker(s))")
    factory = core_factory or SearchCore
    local = threading.local()
    cores: List = []
    write_lock = threading.Lock()
    wrote = 0
    t0 = time.monotonic()

    def one(did: str) -> dict:
        core = getattr(local, "core", None)
        if core is None:
            core = local.core = factory()
            with write_lock:
                cores.append(core)
        return branch_turn(bank, did, policy, turn_seeds(did, s), continuation, core=core)

    try:
        with open(out, "a") as f, ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for c0 in range(0, len(todo), chunk):
                part = todo[c0:c0 + chunk]
                held = gpu_lock(lock_path, what="policy_spectrum truth chunk") if lock_path is not None \
                    else contextlib.nullcontext()
                with held:
                    for row in ex.map(one, part):
                        with write_lock:
                            f.write(json.dumps(row, sort_keys=True) + "\n")
                            f.flush()
                            os.fsync(f.fileno())
                            wrote += 1
                        if not row["ok"]:
                            log(f"[truth] REFUSED {row['id']}: {row['error'][:200]}")
                el = time.monotonic() - t0
                log(f"[truth] {c0 + len(part)}/{len(todo)} turns, {el / 60:.1f} min, "
                    f"eta {el / (c0 + len(part)) * (len(todo) - c0 - len(part)) / 60:.1f} min")
    finally:
        for c in cores:
            c.close()
    return wrote


# ---------------------------------------------------------------------------------------------
# the readout
# ---------------------------------------------------------------------------------------------

def first_seeds(row: dict, k: int) -> dict:
    """The same row as if only its first ``k`` seeds had been played (seeds are nested: the first
    ``k`` of ``turn_seeds(id, S)`` ARE ``turn_seeds(id, k)``)."""
    if not row.get("ok"):
        return row
    if len(row["seeds"]) < k:
        raise ValueError(f"{row['id']}: {len(row['seeds'])} seeds < {k}")
    return dict(row, seeds=row["seeds"][:k],
                outcomes={a: v[:k] for a, v in row["outcomes"].items()},
                ends={a: v[:k] for a, v in row.get("ends", {}).items()})


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


_ENC = {1.0: "+", -1.0: "-", 0.0: "0"}
_DEC = {"+": 1.0, "-": -1.0, "0": 0.0}


def compact(row: dict) -> dict:
    """The committed form of a row: outcomes as one ``+ / - / 0`` character per seed, the seeds as
    their COUNT (``turn_seeds(id, n)`` regenerates them exactly), no per-branch end records (kept in
    the archive's full rows)."""
    c = {k: row[k] for k in ("schema", "id", "ok", "continuation", "at", "side", "stall", "max_turns")
         if k in row}
    c["n_seeds"] = len(row["seeds"])
    if row.get("ok"):
        c["outcomes"] = {a: "".join(_ENC[float(x)] for x in v) for a, v in row["outcomes"].items()}
    else:
        c["error"] = row.get("error")
    return c


def expand(row: dict) -> dict:
    """A compact row back to the full shape (a full row is returned unchanged)."""
    if "n_seeds" not in row:
        return row
    r = dict(row)
    r["seeds"] = turn_seeds(r["id"], r.pop("n_seeds"))
    if r.get("ok"):
        r["outcomes"] = {a: [_DEC[ch] for ch in v] for a, v in r["outcomes"].items()}
    return r


def write_compact(src: Path, dst: Path) -> int:
    import gzip

    rows = load_rows(src)
    with gzip.GzipFile(dst, "wb", mtime=0) as g:
        for r in sorted(rows, key=lambda x: x["id"]):
            g.write((json.dumps(compact(r), sort_keys=True) + "\n").encode())
    return len(rows)


def load_rows(path: Path, continuation: Optional[str] = None) -> List[dict]:
    path = Path(path)
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt") as f:
            text = f.read()
    else:
        text = path.read_text()
    rows = [expand(json.loads(line)) for line in text.splitlines() if line.strip()]
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
    s.add_argument("--v2-include", default=None,
                   help="v2 (at scale): the subset file whose turns come first; over-samples setup / "
                        "recovery / hazard / status (OVERSAMPLE_V2)")
    r = sub.add_parser("run")
    r.add_argument("--bank", required=True)
    r.add_argument("--subset", required=True)
    r.add_argument("--continuation", required=True, help="<checkpoint .zip>=<label>")
    r.add_argument("--out", required=True, help="rows JSONL (appended; resumable)")
    r.add_argument("--seeds", type=int, default=16)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--lib-profile", default="release")
    r.add_argument("--device", default="cpu")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--chunk", type=int, default=32)
    r.add_argument("--lock", default=None, help="take this lock per chunk via utils.gpu_lock (the shared GPU lock; re-entrant under gpu_lock.sh)")
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
        meta = {"schema": TRUTH_SCHEMA, "bank": bank.manifest["content_sha256"],
                "n_free": a.n_free, "n_forced": a.n_forced}
        if a.v2_include:
            inc = json.loads(Path(a.v2_include).read_text())["ids"]
            sel = select_subset_v2(bank, include=inc, n_free=a.n_free, n_forced=a.n_forced)
            meta.update({"seed": SUBSET_SEED + "-v2", "include": a.v2_include,
                         "oversample": OVERSAMPLE_V2, "ids": sel["ids"], "why": sel["why"]})
            ids = sel["ids"]
        else:
            ids = select_subset(bank, n_free=a.n_free, n_forced=a.n_forced)
            meta.update({"seed": SUBSET_SEED, "ids": ids})
        Path(a.out).write_text(json.dumps(meta, indent=1) + "\n")
        print(f"[truth] {len(ids)} turns -> {a.out}")
        return 0
    if a.cmd == "run":
        from main.policy_spectrum.reader import inference_globals, load_checkpoint, policy_logits
        from utils.rust_env import ffi as F
        from utils.rust_env.successors import greedy

        sub_ = json.loads(Path(a.subset).read_text())
        if sub_["bank"] != bank.manifest["content_sha256"]:
            sys.exit("[truth] the subset was drawn from a different bank")
        path, _, label = a.continuation.partition("=")
        # the thread count, and TF32 OFF on CUDA, for the load AND every forward of the run — scoped,
        # so nothing leaks past it (reader.inference_globals)
        with inference_globals(a.threads, a.device):
            model = load_checkpoint(Path(path), device=a.device)
            # The extractor keeps per-forward state on the module (`self.stash`), so ONE model shared by
            # the worker threads forwards one batch at a time — and the batcher MERGES every thread's
            # pending rows into that one forward (the GPU cost of 3 small batches ≈ 1 big one).
            scorer = MicroBatcher(lambda rows, masks: policy_logits(model, rows, masks.astype(bool), batch=8192))
            pol = greedy(scorer)
            lib = F.load(F.default_path(a.lib_profile))

            from utils.rust_env.successors import SearchCore

            # the continuation's name is the POLICY (device is stamped per row, not in the name: a CUDA
            # and a CPU continuation of the same checkpoint gave identical outcomes, 28 / 28 actions
            # at 64 seeds, 67k scored rows — measured 2026-09-29)
            run(bank, sub_["ids"], pol, f"{label or Path(path).stem} greedy (both sides)", Path(a.out),
                s=a.seeds, core_factory=lambda: SearchCore(lib=lib), workers=a.workers, chunk=a.chunk,
                lock_path=Path(a.lock) if a.lock else None,
                log=lambda m: print(m, flush=True))
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
