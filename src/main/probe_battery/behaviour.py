"""BEHAVIOURAL probes for phazing (`gen3_probe_battery_v1`): decodable is not USED, so measure the policy's own
response on bank states, and its response to CONSTRUCTED obs edits.

(a) **Phazing value.** States where our active has a legal Roar / Whirlwind: the policy's mass on the phazing action
    when their active is boosted vs not, and the paired change under three edits — their boosts ZEROED (``zero_their
    _boosts``), their side's Spikes REMOVED (``no_their_spikes``, where Spikes are up), and +2 Atk / +2 SpA INJECTED
    into an unboosted foe (``inject_their_boosts``).
(b) **Phazing threat.** States where our active is boosted and has a legal setup move: mass on setup / attack / switch,
    split by what the viewer knows of the opponent's phazers (revealed / held but unrevealed / none), and the paired
    change when OUR boosts are zeroed (``zero_our_boosts``).
(c) **The critic.** The win-prob's paired change under every edit.

An edit writes ONLY the encoder's columns for that fact (the active-context boost pairs, the global env's Spikes
scalar); the 32-event window still remembers the boost / Spikes EVENTS, and the trackers' derived columns are not
re-derived — so an edit is a LOWER bound on the model's sensitivity to the fact (``designs/prober/probe_battery.md``
§5). Before any edit, the layout is SELF-CHECKED against the views: every row's Spikes and boost columns must equal
the truth they encode, or the run is refused.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

BOOST_ORDER = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
SPIKES_MAX = 3


class Layout:
    def __init__(self, lay: Dict[str, Any]):
        parts = lay["parts"]
        self.ctx = int(parts["context"]["start"])
        self.ctx_dim = int(lay["active_context_dim"])
        self.glob = int(parts["global"]["start"])
        hz = lay["global_layout"]["hazards"]
        self.spikes_ours = self.glob + int(hz["offset"])
        self.spikes_theirs = self.spikes_ours + 1

    def boosts(self, side: int) -> slice:
        s = self.ctx + side * self.ctx_dim
        return slice(s, s + 2 * len(BOOST_ORDER))


def self_check(rows: np.ndarray, decs: Sequence[Dict[str, Any]], L: Layout) -> Dict[str, int]:
    """Refuse unless the edited columns encode exactly the views' truth on every row."""
    bad: Counter = Counter()
    for i, d in enumerate(decs):
        for side, key, col in (("ours", "spikes", L.spikes_ours), ("opp", "spikes", L.spikes_theirs)):
            want = int((d["V"][side].get("side_conditions") or {}).get(key, 0)) / SPIKES_MAX
            if abs(float(rows[i, col]) - want) > 1e-6:
                bad["spikes"] += 1
        for s, side in ((0, "ours"), (1, "opp")):
            act = d["V"][side].get("active")
            m = next((x for x in d["V"][side]["mons"] if x["species"] == act), None)
            b = (m or {}).get("boosts") or {}
            enc = rows[i, L.boosts(s)]
            for k, st in enumerate(BOOST_ORDER):
                v = int(b.get(st, 0) or 0) if m is not None and not m.get("fainted") else None
                if v is None:
                    continue
                if abs(enc[2 * k] - max(v, 0) / 6) > 1e-6 or abs(enc[2 * k + 1] - max(-v, 0) / 6) > 1e-6:
                    bad["boosts"] += 1
                    break
    if bad:
        raise RuntimeError(f"layout self-check FAILED on {dict(bad)} rows — the edit columns do not encode what "
                           "the views say; refusing to edit")
    return {"rows_checked": len(decs)}


def _tok_kind(tok: str) -> Tuple[str, str]:
    verb, _, arg = tok.partition(" ")
    return verb, arg.strip().lower().replace(" ", "").replace("-", "")


def action_sets(d: Dict[str, Any]) -> Dict[str, List[int]]:
    from agents.gen3_data import moves

    out: Dict[str, List[int]] = {"phaze": [], "setup": [], "attack": [], "switch": [], "other": []}
    for k, tok in d["tokens"].items():
        verb, mid = _tok_kind(tok)
        i = int(k)
        if verb == "switch":
            out["switch"].append(i)
            continue
        md = moves.get(mid)
        if mid in ("roar", "whirlwind"):
            out["phaze"].append(i)
        elif md is not None and (md.is_boost or mid == "curse"):
            out["setup"].append(i)
        elif md is not None and (int(md.base_power) > 0 or mid in ("seismictoss", "nightshade", "return",
                                                                   "frustration", "superfang", "dragonrage")):
            out["attack"].append(i)
        else:
            out["other"].append(i)
    return out


def build_cases(rows: np.ndarray, decs: Sequence[Dict[str, Any]], L: Layout, facts_idx: Dict[str, int],
                vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Every row the forwards need: originals once, then each edited copy; a case list names them."""
    def f(name: str, i: int) -> float:
        return float(vals[i, facts_idx[name]])

    keep: List[int] = []
    edits: List[Tuple[str, int, np.ndarray]] = []
    cases: List[Dict[str, Any]] = []
    for i, d in enumerate(decs):
        acts = action_sets(d)
        ta_boosted = f("boosted_TA", i) == 1.0
        sp_th = int((d["V"]["opp"].get("side_conditions") or {}).get("spikes", 0))
        if acts["phaze"] and not math.isnan(f("boosted_TA", i)):
            c = {"probe": "a", "i": i, "acts": acts, "ta_boosted": ta_boosted, "spikes_theirs": sp_th,
                 "edits": {}}
            keep.append(i)
            if ta_boosted:
                r = rows[i].copy()
                r[L.boosts(1)] = 0.0
                c["edits"]["zero_their_boosts"] = len(edits)
                edits.append(("zero_their_boosts", i, r))
            else:
                r = rows[i].copy()
                b = L.boosts(1).start
                r[b + 2 * BOOST_ORDER.index("atk")] = 2 / 6
                r[b + 2 * BOOST_ORDER.index("spa")] = 2 / 6
                c["edits"]["inject_their_boosts"] = len(edits)
                edits.append(("inject_their_boosts", i, r))
            if sp_th > 0:
                r = rows[i].copy()
                r[L.spikes_theirs] = 0.0
                c["edits"]["no_their_spikes"] = len(edits)
                edits.append(("no_their_spikes", i, r))
            cases.append(c)
        if acts["setup"] and f("boosted_OA", i) == 1.0:
            known = f("opp_team_phazer_revealed", i) == 1.0
            held = f("opp_team_phazer_true", i) == 1.0
            c = {"probe": "b", "i": i, "acts": acts, "phazer": "revealed" if known else ("held" if held else "none"),
                 "ta_phazer": f("TA_phazer_true", i) == 1.0, "edits": {}}
            keep.append(i)
            r = rows[i].copy()
            r[L.boosts(0)] = 0.0
            c["edits"]["zero_our_boosts"] = len(edits)
            edits.append(("zero_our_boosts", i, r))
            cases.append(c)
    keep = sorted(set(keep))
    pos = {i: k for k, i in enumerate(keep)}
    base_rows = rows[keep]
    edit_rows = np.stack([e[2] for e in edits]) if edits else np.zeros((0, rows.shape[1]), np.float32)
    all_rows = np.concatenate([base_rows, edit_rows]).astype(np.float32)
    masks = np.array([decs[i]["mask"] for i in keep] + [decs[e[1]]["mask"] for e in edits], dtype=np.int8)
    for c in cases:
        c["orig"] = pos[c["i"]]
        c["edits"] = {k: len(keep) + v for k, v in c["edits"].items()}
    return all_rows, masks, cases


def mass(p: np.ndarray, idx: Sequence[int]) -> float:
    return float(p[list(idx)].sum()) if idx else 0.0


def per_checkpoint(cases: Sequence[Dict[str, Any]], probs: np.ndarray, winp: np.ndarray) -> Dict[str, Any]:
    """One checkpoint's means: mass on each action class per stratum, and the paired change per edit."""
    acc: Dict[str, List[float]] = {}

    def add(k: str, v: float) -> None:
        acc.setdefault(k, []).append(v)

    for c in cases:
        p0, w0 = probs[c["orig"]], float(winp[c["orig"]])
        if c["probe"] == "a":
            st = f"a/{'boosted' if c['ta_boosted'] else 'unboosted'}/{'spikes' if c['spikes_theirs'] else 'nospikes'}"
            add(f"{st}/p_phaze", mass(p0, c["acts"]["phaze"]))
            add(f"{st}/win", w0)
            for e, j in c["edits"].items():
                add(f"a/{e}/d_p_phaze", mass(probs[j], c["acts"]["phaze"]) - mass(p0, c["acts"]["phaze"]))
                add(f"a/{e}/d_win", float(winp[j]) - w0)
                add(f"a/{e}/{'boosted' if c['ta_boosted'] else 'unboosted'}/d_p_phaze",
                    mass(probs[j], c["acts"]["phaze"]) - mass(p0, c["acts"]["phaze"]))
        else:
            st = f"b/{c['phazer']}"
            for cls in ("setup", "attack", "switch"):
                add(f"{st}/p_{cls}", mass(p0, c["acts"][cls]))
            add(f"{st}/win", w0)
            for e, j in c["edits"].items():
                for cls in ("setup", "attack", "switch"):
                    add(f"b/{e}/d_p_{cls}", mass(probs[j], c["acts"][cls]) - mass(p0, c["acts"][cls]))
                    add(f"b/{c['phazer']}/{e}/d_p_{cls}", mass(probs[j], c["acts"][cls]) - mass(p0, c["acts"][cls]))
                add(f"b/{e}/d_win", float(winp[j]) - w0)
    return {k: {"n": len(v), "mean": float(np.mean(v))} for k, v in sorted(acc.items())}


def run(*, checkout: Path, expect_commit: str, bank: Path, specs: Sequence[str], arms: Sequence[str],
        out: Path) -> Dict[str, Any]:
    from main.probe_battery import facts as F
    from main.probe_battery.bank import load_decisions, refuse_models_output, run_worker
    from main.probe_battery.capture import parse_spec
    from main.probe_battery.report import mean_ci, parse_arms, welch

    refuse_models_output(out)
    out.mkdir(parents=True, exist_ok=True)
    decs = load_decisions(bank)
    rows = np.load(Path(bank) / "rows.npy")
    lay = json.loads((Path(bank) / "obs_layout.json").read_text())
    L = Layout(lay)
    chk = self_check(rows, decs, L)
    facts, vals, valid, _ = F.extract(decs)
    fidx = {f.name: j for j, f in enumerate(facts)}
    all_rows, masks, cases = build_cases(rows, decs, L, fidx, vals)
    rows_path = out / "behaviour_rows.npz"
    np.savez(rows_path, rows=all_rows, masks=masks)
    per: Dict[str, Dict[str, Any]] = {}
    for spec in specs:
        label, zp, _ = parse_spec(spec)
        dst = out / f"fwd_{label}.npz"
        if not dst.exists():
            run_worker(checkout, expect_commit, ["forward", "--rows", str(rows_path), "--ckpt", zp, "--label", label,
                                                 "--out", str(dst)], name="probebat-fwd", mem_gb=16)
        d = np.load(dst)
        per[label] = per_checkpoint(cases, d["probs"], d["win_prob"])
    A = parse_arms(arms)
    keys = sorted({k for v in per.values() for k in v})
    summary: Dict[str, Any] = {"self_check": chk, "cases": {"a": sum(1 for c in cases if c["probe"] == "a"),
                                                            "b": sum(1 for c in cases if c["probe"] == "b")},
                               "metrics": {}}
    names = list(A)
    for k in keys:
        ent: Dict[str, Any] = {}
        for arm, labs in A.items():
            xs = [per[lab][k]["mean"] for lab in labs if k in per[lab]]
            ent[arm] = mean_ci(xs)
            ent[arm]["n_rows"] = per[labs[0]][k]["n"] if k in per[labs[0]] else 0
            ent[arm]["per_seed"] = [round(x, 4) for x in xs]
        if len(names) == 2:
            ent["gap"] = welch(ent[names[0]]["per_seed"], ent[names[1]]["per_seed"])
        summary["metrics"][k] = ent
    (out / "behaviour.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    return summary
