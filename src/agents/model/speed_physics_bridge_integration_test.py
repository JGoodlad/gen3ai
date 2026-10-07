"""REAL-BATTLE check of the SPEED PHYSICS (`--speed-physics on`, gen3_speed_physics_v1, audit F7b) on the RUST core —
no poke-env.

Replays banked real battles (the M5 Lane S turn bank: trained-policy games, inputs only) through the Rust core
(`core_events --trackers --obs`). A TURN qualifies when BOTH players chose a MOVE at its start (a switch resolves
before every move — not a speed question). The truth is who ACTED first: the first ``|move|`` (a caller's /
bounce's ``[from]`` line excluded) or ``|cant|`` line of either active after the decision (a full paralysis or
sleep still reveals the order at the mon's own action time). For each qualifying turn, BOTH viewers' rows are run
through a production-config extractor; the op's PRE-gain P(we act first at equal priority) for our active vs
theirs is read with the physics OFF (the logistic) and ON, and the order rule (`move_order.p_seat_first`)
completes it with the two moves' priorities.

* **Priority bracket** (unequal priorities): the rule says P = 0 / 1 — every one is checked against the protocol.
* **Within a bracket** (equal priorities): the reliability table, the Brier score and the log loss of each arm.
* **Quick Claw** is BANNED in gen3ou (the op's term is off); a turn where either active truly holds one (the
  vendored simulator predates the ban, so the bank can contain them) is counted separately and left out of the
  calibration — an illegal-on-the-ladder state.

The beliefs are COLD-START (a freshly built production extractor: the spread / item beliefs equal their Smogon
priors) unless ``--ckpt`` names a checkpoint, whose learned beliefs are read instead.

The committed test runs the first ``N_BATTLES`` banked battles (the same battles every run: a committed bank, a
deterministic replay). The full bank as a script:
    python src/agents/model/speed_physics_bridge_integration_test.py [n_battles] [--ckpt <zip>] [--json <out>]
"""
from __future__ import annotations

import inspect
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.sim, pytest.mark.integration]

N_BATTLES = 40
BANK = Path(__file__).resolve().parents[3] / "designs/research_state/measurements/m5_laneS/bank_v1"
_TURN_RE = re.compile(r"^\|turn\|(\d+)")
BINS = (0.0, 0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0)


def _norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    return "hiddenpower" if s.startswith("hiddenpower") else s


def _extractor() -> Any:
    import gymnasium as gym

    from agents.model.damage_tables import sanitize_historical_move_floor
    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    from utils.paths import repo_path
    cfg = json.load(open(repo_path("designs", "production_config.json")))
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    kw = {k: v for k, v in cfg.items() if k in sig}
    sanitize_historical_move_floor(kw)
    kw["speed_physics"] = "on"
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    torch.manual_seed(0)
    return Gen3FeaturesExtractor(space, layout=layout, mappings=mappings, **kw).eval()


def _ckpt_extractor(zip_path: str) -> Any:
    """A checkpoint's extractor with the physics switched ON in place (no parameter differs between the arms; the
    Quick Claw term is off in gen3ou, so the op needs no buffer the checkpoint's build lacks)."""
    from main.policy_spectrum.reader import load_checkpoint
    fe = load_checkpoint(Path(zip_path)).policy.features_extractor
    fe.eval()
    return fe


def _p_first(fe: Any, rows: np.ndarray, on: bool, batch: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Per row: (the op's PRE-gain P(we act first at equal priority) for our active, our active's TRUE Quick Claw)."""
    from agents.model.damage_tables import QUICK_CLAW_ITEM_NUM
    op = fe.damage_op
    prev = op.speed_physics
    op.speed_physics = on
    ps: List[np.ndarray] = []
    qc: List[np.ndarray] = []
    try:
        with torch.no_grad():
            for i in range(0, len(rows), batch):
                x = torch.as_tensor(rows[i:i + batch], dtype=torch.float32)
                fe({"observation": x})
                ps.append(op.tensors_from_block(op.last_raw_block).out_p_outspeed[:, 0].double().numpy())
                ctx = fe.unpack({"observation": x})
                ar = torch.arange(ctx.batch_size)
                qc.append((ctx.item_ids[ar, ctx.our_active_idx] == QUICK_CLAW_ITEM_NUM).numpy())
    finally:
        op.speed_physics = prev
    return np.concatenate(ps), np.concatenate(qc)


def _turn_of(chunks: List[Any], side: int, after: int) -> int:
    turn = 0
    for s, lines in chunks[: after + 1]:
        if s != side:
            continue
        for line in lines:
            m = _TURN_RE.match(line)
            if m:
                turn = int(m.group(1))
    return turn


def first_actor(chunks: List[Any], side: int, after: int) -> Optional[str]:
    """``"p1"`` / ``"p2"``: the first active to ACT in the viewer's protocol after its decision (to the next turn): a
    ``move`` (no ``[from]``), a ``cant``, or the confusion / Attract check that precedes its action."""
    for s, lines in chunks[after + 1:]:
        if s != side:
            continue
        for line in lines:
            if line.startswith(("|turn|", "|win|", "|tie")):
                return None
            ev = line.split("|")
            # a confused / infatuated mon announces itself AT ITS ACTION TIME (`-activate … confusion` /
            # `move: Attract`), and a confusion self-hit prints no `move` line at all
            act_time = ev[1] == "-activate" and len(ev) > 3 and (ev[3] == "confusion" or ev[3] == "move: Attract")
            if len(ev) < 3 or not (ev[1] in ("move", "cant") or act_time):
                continue
            if ev[1] == "move" and any(x.startswith("[from]") for x in ev[4:]):
                continue
            if ev[2][:3] in ("p1a", "p2a"):
                return ev[2][:2]
    return None


def collect(n_battles: int) -> Dict[str, Any]:
    """Every qualifying turn's two viewer rows, their move priorities and the truth (who acted first)."""
    from agents import gen3_data
    from agents.battle.core_obs import wrap_row
    from agents.battle.rust_core_parity import run_core
    from main.policy_spectrum.bank import load_bank

    battles = load_bank(BANK, verify=False).battles[:n_battles]
    rows: List[np.ndarray] = []
    meta: List[Dict[str, Any]] = []
    skipped = {"unknown_move": 0, "no_actor": 0}
    for res in run_core([b.recorded() for b in battles], trackers=True, obs=True):
        assert res["ok"], res.get("error")
        chunks = res["chunks"]
        by_turn: List[Dict[int, Dict[str, Any]]] = [{}, {}]
        for v in (0, 1):
            for entry in res["trackers"][v]:
                choice = entry.get("choice")
                if "obs" not in entry or not choice:
                    continue
                t = _turn_of(chunks, v, int(entry["after"]))
                if t in by_turn[v]:
                    continue                          # a later (forced-switch) request in the same turn
                by_turn[v][t] = entry
        for t in sorted(set(by_turn[0]) & set(by_turn[1])):
            e = (by_turn[0][t], by_turn[1][t])
            if not all(x["choice"].startswith("move ") for x in e):
                continue
            mids = [_norm(x["choice"][len("move "):]) for x in e]
            mds = [gen3_data.moves.get(m) for m in mids]
            if any(md is None for md in mds) or "struggle" in mids or "recharge" in mids:
                skipped["unknown_move"] += 1
                continue
            actor = first_actor(chunks, 0, int(e[0]["after"]))
            if actor is None:
                skipped["no_actor"] += 1
                continue
            prio = [float(md.priority) for md in mds]                  # type: ignore[union-attr]
            for v in (0, 1):
                rows.append(np.array(wrap_row(e[v]["obs"]), dtype=np.float32))
                meta.append({"battle": res["label"], "turn": t, "viewer": v, "prio_us": prio[v],
                             "prio_them": prio[1 - v], "we_first": actor == f"p{v + 1}", "pair": len(meta) // 2})
    return {"rows": np.stack(rows) if rows else np.zeros((0, 1), np.float32), "meta": meta, "skipped": skipped,
            "battles": len(battles)}


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))) if len(p) else float("nan")


def reliability(p: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
    out = []
    for lo, hi in zip(BINS, BINS[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))
        out.append({"bin": f"[{lo:.2f},{hi:.2f}{']' if hi >= 1.0 else ')'}", "n": int(m.sum()),
                    "mean_p": float(p[m].mean()) if m.any() else None,
                    "observed": float(y[m].mean()) if m.any() else None})
    return out


def _ece(table: List[Dict[str, Any]], n: int) -> float:
    return sum(r["n"] / n * abs(r["mean_p"] - r["observed"]) for r in table if r["n"]) if n else float("nan")


def check(n_battles: int, ckpt: Optional[str] = None) -> Dict[str, Any]:
    from agents.model.move_order import p_seat_first
    data = collect(n_battles)
    meta = data["meta"]
    fe = _ckpt_extractor(ckpt) if ckpt else _extractor()
    p_off, qc = _p_first(fe, data["rows"], on=False)
    p_on, _ = _p_first(fe, data["rows"], on=True)
    prio_us = np.array([m["prio_us"] for m in meta])
    prio_them = np.array([m["prio_them"] for m in meta])
    y = np.array([float(m["we_first"]) for m in meta])
    pair = np.array([m["pair"] for m in meta])
    qc_turn = np.zeros(len(meta), bool)                   # either active truly holds Quick Claw (both rows)
    for k in np.unique(pair):
        idx = np.nonzero(pair == k)[0]
        qc_turn[idx] = qc[idx].any()
    out: Dict[str, Any] = {"battles": data["battles"], "rows": len(meta), "skipped": data["skipped"],
                           "quick_claw_rows": int(qc_turn.sum()), "beliefs": ckpt or "cold-start (Smogon priors)"}
    # the priority bracket: P(we first) = 1 − P(their seat first), deterministic when the priorities differ
    diff = (prio_us != prio_them) & ~qc_turn
    rule = 1.0 - np.asarray(p_seat_first(torch.as_tensor(prio_us), torch.as_tensor(prio_them),
                                         torch.as_tensor(p_on)))
    wrong = diff & (rule != y)
    out["bracket"] = {"rows": int(diff.sum()), "contradicted": int(wrong.sum()),
                      "examples": [meta[i] for i in np.nonzero(wrong)[0][:10]]}
    eq = (prio_us == prio_them) & ~qc_turn
    out["within_bracket_rows"] = int(eq.sum())
    for name, p in (("off", p_off), ("on", p_on)):
        tab = reliability(p[eq], y[eq])
        out[name] = {"brier": _brier(p[eq], y[eq]), "logloss": _logloss(p[eq], y[eq]),
                     "ece": _ece(tab, int(eq.sum())), "table": tab,
                     "certain_wrong": int((((p[eq] == 1.0) & (y[eq] == 0)) | ((p[eq] == 0.0) & (y[eq] == 1))).sum()),
                     "certain_wrong_examples": [dict(meta[i], p=float(p[i])) for i in np.nonzero(
                         eq & (((p == 1.0) & (y == 0)) | ((p == 0.0) & (y == 1))))[0][:10]]}
    # the two viewers of one turn must agree: P(p1 first) + P(p2 first) = 1 within a bracket (a consistency read
    # of each arm, not a truth): the mean |sum − 1| over the turns both rows are in
    for name, p in (("off", p_off), ("on", p_on)):
        s = [abs(p[i] + p[i + 1] - 1.0) for i in range(0, len(meta) - 1, 2) if eq[i] and eq[i + 1]]
        out[name]["viewer_asymmetry_mean"] = float(np.mean(s)) if s else float("nan")
    return out


def test_the_priority_bracket_is_never_contradicted_and_the_physics_is_calibrated() -> None:
    r = check(N_BATTLES)
    assert r["bracket"]["rows"] > 0, f"NOT EXERCISED: no unequal-priority turn ({r['skipped']})"
    assert r["bracket"]["contradicted"] == 0, r["bracket"]["examples"]
    assert r["within_bracket_rows"] >= 200, "too few equal-priority turns to read a calibration"
    # the saturated bins hold: where the physics says "almost surely first / second", it almost surely is (the full
    # bank has ONE float32-certain miss — a Timid 252-Speed Blissey 15 prior spreads above the Smogon mean: the
    # Gaussian belief's tail, a named residual, ledger 2026-10-07)
    tab = {row["bin"]: row for row in r["on"]["table"]}
    assert tab["[0.95,1.00]"]["n"] > 0 and tab["[0.95,1.00]"]["observed"] >= 0.95, tab
    assert tab["[0.00,0.05)"]["n"] > 0 and tab["[0.00,0.05)"]["observed"] <= 0.05, tab
    # the two viewers of one turn price the same order: a physics that read the two sides asymmetrically would
    # break this (the logistic is symmetric by construction too; both are reported by the script)
    assert r["on"]["viewer_asymmetry_mean"] < 0.25, r["on"]


def _fmt(r: Dict[str, Any]) -> str:
    lines = [f"battles {r['battles']} · rows {r['rows']} · within-bracket rows {r['within_bracket_rows']} · "
             f"Quick Claw rows (excluded) {r['quick_claw_rows']} · skipped {r['skipped']} · beliefs: {r['beliefs']}",
             f"priority bracket: {r['bracket']['rows']} rows, {r['bracket']['contradicted']} contradicted", "",
             "| bin | n | off mean P | off observed | on n | on mean P | on observed |", "|---|---|---|---|---|---|---|"]
    for a, b in zip(r["off"]["table"], r["on"]["table"]):
        f = (lambda x: "—" if x is None else f"{x:.3f}")
        lines.append(f"| {a['bin']} | {a['n']} | {f(a['mean_p'])} | {f(a['observed'])} | {b['n']} | "
                     f"{f(b['mean_p'])} | {f(b['observed'])} |")
    for k in ("off", "on"):
        lines.append(f"{k}: Brier {r[k]['brier']:.4f} · log loss {r[k]['logloss']:.4f} · ECE {r[k]['ece']:.4f} · "
                     f"certain-and-wrong {r[k]['certain_wrong']} · viewer asymmetry {r[k]['viewer_asymmetry_mean']:.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    args = sys.argv[1:]
    ck = args[args.index("--ckpt") + 1] if "--ckpt" in args else None
    js = args[args.index("--json") + 1] if "--json" in args else None
    pos = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))]
    r = check(int(pos[0]) if pos else 10**9, ck)
    print(_fmt(r), flush=True)
    if js:
        Path(js).write_text(json.dumps(r, indent=1, default=lambda o: o if not isinstance(o, float) or
                                       not math.isnan(o) else None))
    sys.exit(1 if r["bracket"]["contradicted"] else 0)
