"""X37b -- the in-house TRANSFER read (PREREG.md beside this directory). Subcommands:

* ``register --out REG``      write the frozen registration (checkpoints + sha256, panels + digests, the plan seeds)
* ``plan --reg REG``          print the unit queue (deterministic from the registration)
* ``run --reg REG --state DIR [--stop-at ISO] [--max-units N] [--only-smoke]``
                              work the queue in order, skipping recorded units (atomic files; resumable)
* ``read --reg REG --state DIR --out JSON``   the registered analysis (PREREG.md §4-§5)
* ``selfcheck``               the plan's invariants and the estimator on synthetic rows (no games)

Cells (player always seat p1; the OPPONENT is always P* = X26 @ 280,000,170; mirrored pairs on one battle seed):

* ``HH_<R>``  R vs P*, both teams from home72 (a != b)
* ``UU_<R>``  R vs P*, both teams from ONE unseen panel (half heldout72_hpfix pairs, half forum72; a != b)
* ``X_<R>``   R vs P*, CROSS pairs (u from an unseen panel, h from home72): game 2k R pilots u vs P* on h (game A),
              game 2k+1 R pilots h vs P* on u (game B)
* ``X_P``     P* vs P* on the SAME cross pairs and seeds as every ``X_<R>`` (common random numbers)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "gen3_x37b_registration_v1"
UNIT_SCHEMA = "gen3_x37b_unit_v1"
PANELS = ("home72", "heldout72_hpfix", "forum72")
UNSEEN = ("heldout72_hpfix", "forum72")
SHORT = {"home72": "P", "heldout72_hpfix": "H", "forum72": "F"}
PTS = {"WIN": 1.0, "DRAW": 0.5, "LOSS": 0.0}
BAR = 0.02           # the +2.0 pp transfer bar (design_diversity_teams_opponents.md §3.2 / §5.4)
EPS = 1e-9           # rule 8: a bound within EPS of its boundary is UNDECIDED


def _h(*parts: Any) -> int:
    return int.from_bytes(hashlib.blake2b(":".join(str(p) for p in parts).encode(), digest_size=8).digest(),
                          "big") & ((1 << 62) - 1)


def digest_of(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ the plan
def round_pairs(reg: Mapping[str, Any], block: str, r: int) -> List[Tuple[str, str]]:
    """Round ``r``'s pairs of ``block`` (HH / UU / X): ``(team the PLAYER pilots in game 2k, the opponent's)``.
    Team ids are ``<panel>/<file label>``; draws are uniform, seeded by (registration seed, block, round)."""
    ids = reg["team_ids"]
    n = int(reg["plan"]["unit_pairs"])
    rng = random.Random(_h("x37b", reg["plan"]["seed"], block, r))
    out: List[Tuple[str, str]] = []
    if block == "HH":
        for _ in range(n):
            a, b = rng.sample(ids["home72"], 2)
            out.append((a, b))
    elif block == "UU":
        for i in range(n):
            panel = UNSEEN[0] if i < n // 2 else UNSEEN[1]
            a, b = rng.sample(ids[panel], 2)
            out.append((a, b))
    elif block == "X":
        for i in range(n):
            panel = UNSEEN[0] if i < n // 2 else UNSEEN[1]
            out.append((rng.choice(ids[panel]), rng.choice(ids["home72"])))
    else:
        raise ValueError(block)
    return out


def round_seed(reg: Mapping[str, Any], block: str, r: int) -> int:
    """The per-game seed base of round ``r``'s ``block`` -- SHARED by every player of that block (CRN)."""
    return _h("x37b-cycle", reg["plan"]["seed"], block, r)


def units(reg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The queue, in play order: the PRIMARY rounds (R = 25M) first, interleaved cell by cell so that any stopped
    prefix holds whole rounds; then the TRAJECTORY rounds (R = 100M, 175M), on the primary rounds' keys."""
    p = reg["plan"]
    q: List[Dict[str, Any]] = []
    for r in range(int(p["rounds_primary"])):
        for block, player in (("HH", "25M"), ("UU", "25M"), ("X", "25M"), ("X", "P")):
            q.append({"name": f"r{r:03d}_{block}_{player}", "round": r, "block": block, "player": player,
                      "stage": "primary"})
    for r in range(int(p["rounds_trajectory"])):
        for player in reg["trajectory_refs"]:
            for block in ("HH", "UU", "X"):
                q.append({"name": f"r{r:03d}_{block}_{player}", "round": r, "block": block, "player": player,
                          "stage": "trajectory"})
    return q


# ------------------------------------------------------------------------------------------------ register
def register(out: Path, smoke: bool = False) -> Dict[str, Any]:
    from main import heldout_panels as HP
    from main.h2h.play import file_sha256

    model_dir = Path("/home/goodlad/dev/gen3ai-reads/x37b_2026-10-19/model")
    ck = {"25M": 25000027, "100M": 100000037, "175M": 175000114, "P": 280000170}
    checkpoints = {}
    for lab, step in ck.items():
        z = model_dir / f"checkpoint_{step}_steps.zip"
        checkpoints[lab] = {"path": str(z), "step": step, "sha256": file_sha256(str(z)),
                            "source": f"models/rb_x26_s2601/checkpoints/checkpoint_{step}_steps.zip"}
    if not checkpoints["P"]["sha256"].startswith("814823c5"):
        raise SystemExit("P* sha256 is not the registered 814823c5...")
    panels, team_ids = {}, {}
    for name in (("p72",) if smoke else PANELS):
        ent = HP.panel_entry(name)
        items = HP.panel_team_texts(name)        # count + content digest checked
        panels[name] = {"n": int(ent["n"]), "content_digest": ent["content_digest"], "team_dir": ent["team_dir"]}
        team_ids[name] = sorted(f"{name}/{lab}" for lab, _ in items)
    if smoke:          # the engineering smoke plays p72 only (not a cell of this read; outcomes never read)
        team_ids = {"home72": team_ids["p72"], UNSEEN[0]: team_ids["p72"], UNSEEN[1]: team_ids["p72"]}
    reg: Dict[str, Any] = {
        "schema": SCHEMA, "smoke": bool(smoke),
        "checkpoints": checkpoints, "opponent": "P", "primary_ref": "25M", "trajectory_refs": ["100M", "175M"],
        "panels": panels, "registry": "designs/research_state/heldout_panels.json",
        "team_ids": team_ids,
        "compute": {"device": "cpu", "backend": "eager", "n_envs": 48, "threads": 2, "torch_threads": 1,
                    "front": "ffi", "profile": "release"},
        "plan": {"seed": 3737 if not smoke else 9999, "unit_pairs": 96 if not smoke else 24,
                 "rounds_primary": 14 if not smoke else 1, "rounds_trajectory": 2 if not smoke else 0},
        "analysis": {"bootstrap_B": 2000, "bootstrap_seed": 3738, "bar": BAR, "eps": EPS},
    }
    reg["digest"] = digest_of({k: v for k, v in reg.items() if k != "digest"})
    out.write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")
    return reg


# ------------------------------------------------------------------------------------------------ run
def _atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _texts(reg: Mapping[str, Any]) -> Dict[str, str]:
    from main import heldout_panels as HP

    out: Dict[str, str] = {}
    for name, ent in reg["panels"].items():
        if HP.panel_entry(name)["content_digest"] != ent["content_digest"]:
            raise SystemExit(f"panel {name}: the registry's digest moved since the registration")
        for lab, text in HP.panel_team_texts(name):
            out[f"{name}/{lab}"] = text
    return out


def run(reg: Mapping[str, Any], state: Path, stop_at: Optional[float], max_units: Optional[int]) -> Dict[str, Any]:
    from main.h2h.play import Compute, H2HEngine, file_sha256, resolve_player
    from main.live.halt import read_halt
    from main.team_strength.scheduled import ScheduledTeams
    from main.team_strength.x26_lane import unit_rows

    out_dir = state / "units"
    q = units(reg)
    todo = [u for u in q if not (out_dir / f"{u['name']}.json").exists()]
    print(f"[x37b] {len(q) - len(todo)} / {len(q)} units done; {len(todo)} to play", flush=True)
    if not todo:
        return {"played": 0, "remaining": 0}
    ck = reg["checkpoints"]

    def check() -> None:
        if read_halt() is not None:
            raise SystemExit("a T28 live-parse HALT marker is set: no play")
        for lab, c in ck.items():
            if file_sha256(c["path"]) != c["sha256"]:
                raise SystemExit(f"checkpoint {lab}: sha256 is not the registered {c['sha256'][:12]}")

    check()
    refs = {lab: resolve_player(c["path"]) for lab, c in ck.items()}
    builder = ScheduledTeams(_texts(reg), team_source="gen3_x37b: named pairs on registered panels, mirrored")
    c = reg["compute"]
    comp = Compute(device=c["device"], backend=c["backend"], n_envs=int(c["n_envs"]), threads=int(c["threads"]),
                   torch_threads=int(c["torch_threads"]), front=c["front"], profile=c["profile"])
    opp = refs[reg["opponent"]]
    eng = H2HEngine(refs["25M"], opp, comp, team_builder=builder)
    played = 0
    try:
        for u in todo:
            if stop_at is not None and time.time() > stop_at:
                print(f"[x37b] stop-at reached; {len(todo) - played} units left", flush=True)
                break
            if max_units is not None and played >= max_units:
                break
            check()
            t0 = time.time()
            eng.set_cell(refs[u["player"]], opp)
            pairs = round_pairs(reg, u["block"], u["round"])
            seed = round_seed(reg, u["block"], u["round"])
            builder.schedule(seed, pairs)
            games, stats = eng.play_batch(len(pairs), seed)
            plan_unit = SimpleNamespace(name=u["name"], pairs=[SimpleNamespace(subject=a, ref=b, index=k)
                                                               for k, (a, b) in enumerate(pairs)])
            rows = unit_rows(games, plan_unit, eng.team_packed, builder.id_by_packed)
            rec = {"schema": UNIT_SCHEMA, **u, "seed": seed, "pairs": rows, "n_pairs": len(rows),
                   "player_sha256": ck[u["player"]]["sha256"], "opponent_sha256": ck[reg["opponent"]]["sha256"],
                   "registration_digest": reg["digest"], "regime_id": eng.regime["regime_id"],
                   "team_set_id": builder.team_set_id, "compute": comp.block(), "core_stamp": eng.core_stamp,
                   "torch": eng.torch_version, "commit": _commit(), "near_ties": stats.get("near_ties"),
                   "seconds": round(time.time() - t0, 1),
                   "finished_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
            _atomic(out_dir / f"{u['name']}.json", rec)
            played += 1
            print(f"[x37b] {u['name']}: {len(rows)} pairs in {rec['seconds']} s "
                  f"({2 * len(rows) / max(rec['seconds'], 1e-9):.2f} games/s)", flush=True)
    finally:
        eng.close()
    left = sum(1 for u in q if not (out_dir / f"{u['name']}.json").exists())
    return {"played": played, "remaining": left}


def _commit() -> str:
    from utils.git import get_git_hash

    try:
        return get_git_hash() or ""
    except Exception:
        return ""


# ------------------------------------------------------------------------------------------------ read
def load(reg: Mapping[str, Any], state: Path) -> Dict[str, Dict[str, Any]]:
    """Every recorded unit, checked against the registration and the plan (teams per pair)."""
    out = {}
    for u in units(reg):
        f = state / "units" / f"{u['name']}.json"
        if not f.exists():
            continue
        rec = json.loads(f.read_text())
        if rec["registration_digest"] != reg["digest"]:
            raise SystemExit(f"{f}: recorded under another registration")
        if rec["player_sha256"] != reg["checkpoints"][u["player"]]["sha256"]:
            raise SystemExit(f"{f}: another player checkpoint")
        want = round_pairs(reg, u["block"], u["round"])
        got = [(p["subject"], p["ref"]) for p in rec["pairs"]]
        if got != want:
            raise SystemExit(f"{f}: its pairs are not the plan's")
        out[u["name"]] = rec
    return out


def complete_rounds(reg: Mapping[str, Any], recs: Mapping[str, Any], stage: str, player: Optional[str] = None
                    ) -> List[int]:
    """Rounds whose every unit of ``stage`` (and ``player``, for the trajectory) is recorded -- the read uses WHOLE
    rounds only (PREREG §3), so a stopped queue is a balanced prefix."""
    by: Dict[int, List[bool]] = {}
    for u in units(reg):
        if u["stage"] != stage or (player is not None and u["player"] != player):
            continue
        by.setdefault(u["round"], []).append(u["name"] in recs)
    rounds = sorted(r for r, ok in by.items() if all(ok))
    # a whole round is required from the START: the prefix rule (round r counts only if every r' < r counts)
    pref = []
    for i, r in enumerate(rounds):
        if r != i:
            break
        pref.append(r)
    if stage == "trajectory" and player is not None:
        # a trajectory round also needs the primary round's X_P (its CRN partner)
        pref = [r for r in pref if f"r{r:03d}_X_P" in recs]
    return pref


def pair_obs(reg: Mapping[str, Any], recs: Mapping[str, Any], player: str, rounds: Sequence[int],
             drop_near_ties: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """Per-pair observations for one ref R (scores from R's side, the player's):

    * HH / UU: ``{panel, a, b, s}``, s = the pair's mean score (two games);
    * X: ``{panel, u, h, A_R, B_R, A_P, B_P}`` -- game A (R / P* pilots u), game B (R / P* pilots h); the P* arm
      from ``X_P`` on the same key.
    ``drop_near_ties``: drop a pair (X: a KEY, both arms) any of whose games carried a CPU near-tie decision."""
    out: Dict[str, List[Dict[str, Any]]] = {"HH": [], "UU": [], "X": []}

    def nt(row: Mapping[str, Any]) -> bool:
        return any(int(g.get("near_ties") or 0) for g in row["games"])

    def panel_of(block: str, k: int, n: int) -> str:
        # the plan's position decides the panel (round_pairs: the first half of UU / X pairs is UNSEEN[0])
        return "home72" if block == "HH" else (UNSEEN[0] if k < n // 2 else UNSEEN[1])

    for r in rounds:
        for block in ("HH", "UU"):
            rec = recs[f"r{r:03d}_{block}_{player}"]
            n = len(rec["pairs"])
            for row in rec["pairs"]:
                if drop_near_ties and nt(row):
                    continue
                g0, g1 = row["games"]
                out[block].append({"panel": panel_of(block, int(row["k"]), n), "a": row["subject"], "b": row["ref"],
                                   "s": (PTS[g0["result"]] + PTS[g1["result"]]) / 2.0, "round": r})
        xr, xp = recs[f"r{r:03d}_X_{player}"], recs[f"r{r:03d}_X_P"]
        for rr, rp in zip(xr["pairs"], xp["pairs"]):
            assert (rr["subject"], rr["ref"]) == (rp["subject"], rp["ref"])
            if drop_near_ties and (nt(rr) or nt(rp)):
                continue
            out["X"].append({"panel": panel_of("X", int(rr["k"]), len(xr["pairs"])), "u": rr["subject"], "h": rr["ref"], "round": r,
                             "A_R": PTS[rr["games"][0]["result"]], "B_R": PTS[rr["games"][1]["result"]],
                             "A_P": PTS[rp["games"][0]["result"]], "B_P": PTS[rp["games"][1]["result"]]})
    return out


def _wmean(vals: Sequence[float], w: Sequence[float]) -> float:
    sw = sum(w)
    return sum(v * x for v, x in zip(vals, w)) / sw if sw > 0 else float("nan")


def estimates(obs: Mapping[str, List[Dict[str, Any]]], mult: Optional[Mapping[str, float]] = None
              ) -> Dict[str, float]:
    """The registered estimands (PREREG §4) at team multiplicities ``mult`` (None = every team once). Pair weight =
    the product of its two teams' multiplicities (the two-way team bootstrap)."""
    m = (lambda t: 1.0) if mult is None else (lambda t: float(mult.get(t, 0.0)))
    e: Dict[str, float] = {}
    hh = obs["HH"]
    e["r_HH"] = _wmean([o["s"] for o in hh], [m(o["a"]) * m(o["b"]) for o in hh])
    e["g_home"] = 0.5 - e["r_HH"]
    for panel in UNSEEN:
        k = SHORT[panel]
        uu = [o for o in obs["UU"] if o["panel"] == panel]
        e[f"r_UU_{k}"] = _wmean([o["s"] for o in uu], [m(o["a"]) * m(o["b"]) for o in uu])
        e[f"g_{k}"] = 0.5 - e[f"r_UU_{k}"]
        x = [o for o in obs["X"] if o["panel"] == panel]
        w = [m(o["u"]) * m(o["h"]) for o in x]
        e[f"dA_{k}"] = _wmean([o["A_P"] - o["A_R"] for o in x], w)
        e[f"dB_{k}"] = _wmean([o["B_P"] - o["B_R"] for o in x], w)
        e[f"A_P_{k}"] = _wmean([o["A_P"] for o in x], w)
        e[f"seat_{k}"] = (_wmean([o["A_P"] + o["B_P"] for o in x], w) - 1.0) / 2.0
        e[f"T_tot_{k}"] = e["g_home"] - e[f"g_{k}"]
        e[f"T_pilot_{k}"] = e["g_home"] - e[f"dA_{k}"]
        e[f"T_face_{k}"] = e["g_home"] - e[f"dB_{k}"]
        # logit-scale versions (descriptive): P*'s edge as a log-odds difference
        e[f"A_R_{k}"] = _wmean([o["A_R"] for o in x], w)
        e[f"B_R_{k}"] = _wmean([o["B_R"] for o in x], w)
        e[f"B_P_{k}"] = _wmean([o["B_P"] for o in x], w)
        lg_home = _logit(1.0 - e["r_HH"])
        e[f"T_tot_logit_{k}"] = lg_home - _logit(1.0 - e[f"r_UU_{k}"])
        e[f"T_pilot_logit_{k}"] = lg_home - (_logit(e[f"A_P_{k}"]) - _logit(e[f"A_R_{k}"]))
        e[f"T_face_logit_{k}"] = lg_home - (_logit(e[f"B_P_{k}"]) - _logit(e[f"B_R_{k}"]))
    for q in ("g", "dA", "dB", "T_tot", "T_pilot", "T_face", "A_P", "A_R", "B_P", "B_R", "seat", "T_tot_logit",
              "T_pilot_logit", "T_face_logit"):
        e[f"{q}_U"] = (e[f"{q}_H"] + e[f"{q}_F"]) / 2.0
    return e


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def bootstrap(reg: Mapping[str, Any], obs: Mapping[str, List[Dict[str, Any]]], B: int, seed: int
              ) -> Dict[str, List[float]]:
    """Two-way TEAM bootstrap: each replicate resamples every panel's teams with replacement (one shared index set
    for every estimand); percentile 95 % intervals."""
    rng = random.Random(seed)
    ids = reg["team_ids"]
    reps: Dict[str, List[float]] = {}
    for _ in range(B):
        mult: Dict[str, float] = {}
        for panel in PANELS:
            for t in (rng.choice(ids[panel]) for _ in range(len(ids[panel]))):
                mult[t] = mult.get(t, 0.0) + 1.0
        for k, v in estimates(obs, mult).items():
            reps.setdefault(k, []).append(v)
    out = {}
    for k, v in reps.items():
        s = sorted(x for x in v if not math.isnan(x))
        if len(s) < len(v):          # an estimand undefined in some replicate (an empty cell): no interval
            out[k] = [float("nan"), float("nan")]
            continue
        lo = s[int(math.floor(0.025 * (len(s) - 1)))]
        hi = s[int(math.ceil(0.975 * (len(s) - 1)))]
        out[k] = [lo, hi]
    return out


def pair_normal(obs: Mapping[str, List[Dict[str, Any]]]) -> Dict[str, List[float]]:
    """Co-read: pair-level normal (pentanomial-style) 95 % intervals for T_tot_U and T_pilot_U (teams fixed)."""
    def mv(xs: Sequence[float]) -> Tuple[float, float]:
        n = len(xs)
        mu = sum(xs) / n
        return mu, sum((x - mu) ** 2 for x in xs) / (n - 1) / n

    hh, vh = mv([o["s"] for o in obs["HH"]])
    out = {}
    uu = {p: mv([o["s"] for o in obs["UU"] if o["panel"] == p]) for p in UNSEEN}
    xa = {p: mv([o["A_P"] - o["A_R"] for o in obs["X"] if o["panel"] == p]) for p in UNSEEN}
    # T_tot_U = r_UU_U - r_HH; T_pilot_U = 0.5 - r_HH - dA_U
    t_tot = (uu[UNSEEN[0]][0] + uu[UNSEEN[1]][0]) / 2 - hh
    v_tot = (uu[UNSEEN[0]][1] + uu[UNSEEN[1]][1]) / 4 + vh
    t_pil = 0.5 - hh - (xa[UNSEEN[0]][0] + xa[UNSEEN[1]][0]) / 2
    v_pil = vh + (xa[UNSEEN[0]][1] + xa[UNSEEN[1]][1]) / 4
    for k, (t, v) in {"T_tot_U": (t_tot, v_tot), "T_pilot_U": (t_pil, v_pil)}.items():
        out[k] = [t - 1.959964 * math.sqrt(v), t + 1.959964 * math.sqrt(v)]
    return out


def near(x: float, b: float) -> bool:
    return abs(x - b) <= EPS


def genm_class(t: Sequence[float], g: Sequence[float]) -> str:
    """§5.4 GEN-M classes on (g_P - g_U) = T_tot and g_U; checked in the doc's order (TRANSFERS first)."""
    if any(near(x, BAR) or near(x, 0.0) for x in t) or any(near(x, 0.0) for x in g):
        return "UNDECIDED"
    if t[1] < BAR and g[0] > 0:
        return "TRANSFERS"
    if t[0] > 0 and g[0] > 0:
        return "PARTIAL"
    if t[0] > 0 and g[0] <= 0:
        return "POOL-ONLY"
    return "UNDECIDED"


def d0e_class(t: Sequence[float], g: Sequence[float]) -> str:
    """§3.2 D0e classes on T_S = T_pilot and g_S = dA; checked in the doc's order (TRANSFERS first)."""
    if any(near(x, BAR) or near(x, 0.0) for x in t) or any(near(x, 0.0) for x in g):
        return "UNDECIDED"
    if t[1] < BAR and g[0] > 0:
        return "TRANSFERS"
    if t[0] > 0 and g[0] > 0:
        return "PARTIAL"
    if t[0] > 0 and g[0] <= 0:
        return "MEMORISATION-DOMINATED"
    return "UNDECIDED"


def h1_verdict(t_tot: Sequence[float], t_pilot: Sequence[float]) -> str:
    """PREREG §5: KILLED iff T_tot's upper bound < +2.0 pp AND T_pilot's lower bound <= 0 (the F-DV-19 veto);
    SUPPORTED iff not KILLED and (T_tot's or T_pilot's lower bound > 0); else UNDECIDED. Rule 8: any of the four
    bounds within 1e-9 of the boundary it is compared with makes the verdict UNDECIDED."""
    if near(t_tot[1], BAR) or near(t_tot[0], 0.0) or near(t_pilot[0], 0.0):
        return "UNDECIDED"
    if t_tot[1] < BAR and t_pilot[0] <= 0:
        return "KILLED"
    if t_tot[0] > 0 or t_pilot[0] > 0:
        return "SUPPORTED"
    return "UNDECIDED"


def read(reg: Mapping[str, Any], state: Path) -> Dict[str, Any]:
    recs = load(reg, state)
    a = reg["analysis"]
    res: Dict[str, Any] = {"registration_digest": reg["digest"], "units_recorded": len(recs)}
    rounds = complete_rounds(reg, recs, "primary")
    res["primary_rounds"] = rounds
    for variant, drop in (("primary", False), ("no_near_ties", True)):
        obs = pair_obs(reg, recs, "25M", rounds, drop_near_ties=drop)
        est = estimates(obs)
        ci = bootstrap(reg, obs, int(a["bootstrap_B"]), int(a["bootstrap_seed"]))
        blk: Dict[str, Any] = {"n": {k: len(v) for k, v in obs.items()}, "est": est, "ci": ci}
        blk["pair_normal"] = pair_normal(obs)
        blk["h1"] = h1_verdict(ci["T_tot_U"], ci["T_pilot_U"])
        blk["genm_class"] = {k: genm_class(ci[f"T_tot_{k}"], ci[f"g_{k}"]) for k in ("U", "H", "F")}
        blk["d0e_pilot_class"] = {k: d0e_class(ci[f"T_pilot_{k}"], ci[f"dA_{k}"]) for k in ("U", "H", "F")}
        blk["near_tie_games"] = sum(int(g.get("near_ties") or 0) > 0 for rec in recs.values()
                                    for row in rec["pairs"] for g in row["games"])
        res[variant] = blk
    v1, v2 = res["primary"]["h1"], res["no_near_ties"]["h1"]
    res["H1"] = v1 if v1 == v2 else "UNDECIDED"
    res["trajectory"] = {}
    for ref in reg["trajectory_refs"]:
        tr = complete_rounds(reg, recs, "trajectory", ref)
        if not tr:
            continue
        obs = pair_obs(reg, recs, ref, tr)
        res["trajectory"][ref] = {"rounds": tr, "n": {k: len(v) for k, v in obs.items()}, "est": estimates(obs),
                                  "ci": bootstrap(reg, obs, int(a["bootstrap_B"]), int(a["bootstrap_seed"]))}
        # the 25M ref on the SAME rounds, for a like-for-like trajectory
        obs25 = pair_obs(reg, recs, "25M", tr)
        res["trajectory"][ref]["est_25M_same_rounds"] = estimates(obs25)
    return res


# ------------------------------------------------------------------------------------------------ self-check
def selfcheck() -> None:
    reg = {"plan": {"seed": 1, "unit_pairs": 8, "rounds_primary": 3, "rounds_trajectory": 2},
           "trajectory_refs": ["100M", "175M"], "digest": "d",
           "team_ids": {p: [f"{p}/t{i}" for i in range(6)] for p in PANELS}}
    q = units(reg)
    assert len(q) == 3 * 4 + 2 * 2 * 3, len(q)
    assert [u["name"] for u in q[:4]] == ["r000_HH_25M", "r000_UU_25M", "r000_X_25M", "r000_X_P"]
    assert round_pairs(reg, "X", 1) == round_pairs(reg, "X", 1)                 # deterministic
    assert round_pairs(reg, "X", 1) != round_pairs(reg, "X", 2)
    assert round_seed(reg, "X", 0) == round_seed(reg, "X", 0) != round_seed(reg, "HH", 0)
    for r in range(3):
        assert all(a != b and a.startswith("home72/") and b.startswith("home72/") for a, b in round_pairs(reg, "HH", r))
        uu = round_pairs(reg, "UU", r)
        assert all(a != b and a.split("/")[0] == b.split("/")[0] for a, b in uu)
        assert {a.split("/")[0] for a, _ in uu[:4]} == {UNSEEN[0]} and {a.split("/")[0] for a, _ in uu[4:]} == {UNSEEN[1]}
        x = round_pairs(reg, "X", r)
        assert all(u.split("/")[0] in UNSEEN and h.startswith("home72/") for u, h in x)
    # complete_rounds: prefix rule
    names = {u["name"] for u in q if u["stage"] == "primary" and u["round"] in (0, 2)}
    assert complete_rounds(reg, {n: 1 for n in names}, "primary") == [0]
    # the estimator on synthetic rows with a KNOWN answer: R scores 0.30 on HH, 0.40 on UU; A_R 0.25, A_P 0.45
    rng = random.Random(5)
    obs: Dict[str, List[Dict[str, Any]]] = {"HH": [], "UU": [], "X": []}
    for i in range(4000):
        a, b = rng.sample(reg["team_ids"]["home72"], 2)
        obs["HH"].append({"panel": "home72", "a": a, "b": b, "s": 0.30})
        p = UNSEEN[i % 2]
        a, b = rng.sample(reg["team_ids"][p], 2)
        obs["UU"].append({"panel": p, "a": a, "b": b, "s": 0.40})
        obs["X"].append({"panel": p, "u": a, "h": rng.choice(reg["team_ids"]["home72"]),
                         "A_R": 0.25, "B_R": 0.5, "A_P": 0.45, "B_P": 0.55})
    e = estimates(obs)
    assert abs(e["g_home"] - 0.20) < 1e-12 and abs(e["g_U"] - 0.10) < 1e-12
    assert abs(e["T_tot_U"] - 0.10) < 1e-12              # g_home - g_U
    assert abs(e["T_pilot_U"] - 0.00) < 1e-12            # 0.20 - (0.45 - 0.25)
    assert abs(e["T_face_U"] - 0.15) < 1e-12             # 0.20 - (0.55 - 0.50)
    assert abs(e["seat_U"] - 0.0) < 1e-12
    # T_tot = r_UU - r_HH identically
    assert abs(e["T_tot_U"] - ((e["r_UU_H"] + e["r_UU_F"]) / 2 - e["r_HH"])) < 1e-12
    # verdicts
    assert h1_verdict([-0.01, 0.015], [-0.03, 0.04]) == "KILLED"
    assert h1_verdict([0.001, 0.015], [-0.03, 0.04]) == "KILLED"         # small detected shortfall, surely < 2 pp
    assert h1_verdict([-0.01, 0.015], [0.001, 0.06]) == "SUPPORTED"      # the F-DV-19 veto
    assert h1_verdict([0.005, 0.05], [-0.02, 0.05]) == "SUPPORTED"
    assert h1_verdict([-0.01, 0.03], [-0.02, 0.05]) == "UNDECIDED"
    assert h1_verdict([-0.01, BAR], [-0.02, 0.05]) == "UNDECIDED"        # rule 8
    assert h1_verdict([-0.01, 0.015], [0.0, 0.05]) == "UNDECIDED"         # rule 8
    assert genm_class([-0.01, 0.015], [0.05, 0.2]) == "TRANSFERS"
    assert genm_class([0.03, 0.08], [0.05, 0.2]) == "PARTIAL"
    assert genm_class([0.03, 0.08], [-0.01, 0.2]) == "POOL-ONLY"
    assert d0e_class([0.03, 0.08], [-0.01, 0.2]) == "MEMORISATION-DOMINATED"
    # bootstrap on a team-heterogeneous synthetic set is deterministic and covers the point
    obs2 = {k: v[:600] for k, v in obs.items()}
    for o in obs2["HH"]:
        o["s"] = 0.3 + 0.05 * (int(o["a"][-1]) - 2.5) / 2.5
    ci1 = bootstrap(reg, obs2, 200, 7)
    ci2 = bootstrap(reg, obs2, 200, 7)
    assert ci1 == ci2
    e2 = estimates(obs2)
    assert ci1["g_home"][0] <= e2["g_home"] <= ci1["g_home"][1]
    print("[x37b] selfcheck OK")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="x37b")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--out", required=True)
    r.add_argument("--smoke", action="store_true")
    p = sub.add_parser("plan")
    p.add_argument("--reg", required=True)
    u = sub.add_parser("run")
    u.add_argument("--reg", required=True)
    u.add_argument("--state", required=True)
    u.add_argument("--stop-at", default=None)
    u.add_argument("--max-units", type=int, default=None)
    d = sub.add_parser("read")
    d.add_argument("--reg", required=True)
    d.add_argument("--state", required=True)
    d.add_argument("--out", required=True)
    sub.add_parser("selfcheck")
    a = ap.parse_args(argv)
    if a.cmd == "selfcheck":
        selfcheck()
        return 0
    if a.cmd == "register":
        reg = register(Path(a.out), smoke=a.smoke)
        print(json.dumps({"digest": reg["digest"], "units": len(units(reg))}))
        return 0
    reg = json.loads(Path(a.reg).read_text())
    if reg["digest"] != digest_of({k: v for k, v in reg.items() if k != "digest"}):
        raise SystemExit("the registration's digest does not match its content")
    if a.cmd == "plan":
        for x in units(reg):
            print(x["name"], x["stage"])
        return 0
    if a.cmd == "run":
        stop = _dt.datetime.fromisoformat(a.stop_at).timestamp() if a.stop_at else None
        print(json.dumps(run(reg, Path(a.state), stop, a.max_units)))
        return 0
    if a.cmd == "read":
        res = read(reg, Path(a.state))
        Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"H1": res["H1"], "rounds": len(res["primary_rounds"])}))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
