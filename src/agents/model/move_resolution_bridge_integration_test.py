"""REAL-BATTLE check of the MOVE-RESOLUTION family (`--move-resolution on`, gen3_move_resolution_v1) on the RUST core —
no poke-env.

Replays banked real battles (the M5 Lane S turn bank: trained-policy games, inputs only) through the Rust core
(`core_events --trackers --obs`), runs a production-config extractor with the family ON over every answered
decision's observation, and resolves each played MOVE against the viewer's protocol for that turn:

  * EXECUTED   — a ``|move|`` line from our active naming the chosen move (a caller's ``[from]`` line excluded);
  * FAILED     — after it, before the next action line: ``-fail`` / ``-immune`` / ``-miss`` / ``-notarget``, their
                 Protect / Detect / Magic Coat ``-activate``, a clause message, a ``[miss]`` flag; a damaging
                 move that dealt no damage (a silent no-op, an absorbing ability); a stat move that moved no
                 stage; a Destiny Bond that never triggered; a Magic Coat that bounced nothing;
  * BRANCH     — SWITCH when their active switched (or was dragged) before our move, else STAY.

**The gate.** A fact the family states as EXACTLY 0 is never contradicted: ``p_lands_stay == 0`` (or ``p_act == 0``)
and they stayed ⇒ an executed move FAILED; ``p_lands_switch == 0`` and they switched ⇒ it FAILED. The gate must be
EXERCISED. Two classes are excluded, named: Sleep Talk / Snore (a called move failing its own ``onTry`` prints no
line) and Heal Bell / Aromatherapy (``-cureteam`` does not say whether anything was cured).

Run in BOTH belief modes (`gen3_move_resolution_x5_v1`): blob, and X5's ``fixed_mass`` (the same rules on the flat
pointer, OTHER_move / OTHER_species priced) — an exact zero that only the X5 read states would be contradicted here.

The same battles every run (a committed bank, a deterministic replay). The full bank as a script:
    python src/agents/model/move_resolution_bridge_integration_test.py [n_battles] [blob|fixed_mass]
"""
from __future__ import annotations

import collections
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.sim, pytest.mark.integration]

N_BATTLES = 24
BANK = Path(__file__).resolve().parents[3] / "designs/research_state/measurements/m5_laneS/bank_v1"
_FAIL_TAGS = ("-fail", "-immune", "-miss", "-notarget")
_ACTION_TAGS = ("move", "switch", "drag", "cant", "turn", "faint", "upkeep", "win", "tie")
_CALLERS = {"sleeptalk", "snore", "metronome", "mirrormove", "naturepower", "assist"}
_GATE_EXCLUDED = {"sleeptalk", "snore", "healbell", "aromatherapy"}


def _norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    return "hiddenpower" if s.startswith("hiddenpower") else s


def _extractor(belief_tokens: str = "blob") -> Any:
    import gymnasium as gym

    from agents.model.damage_tables import sanitize_historical_move_floor
    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    from utils.paths import repo_path
    cfg = json.load(open(repo_path("designs", "production_config.json")))
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    kw = {k: v for k, v in cfg.items() if k in sig}
    sanitize_historical_move_floor(kw)
    kw["move_resolution"] = "on"
    kw["belief_tokens"] = belief_tokens
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    torch.manual_seed(0)
    return Gen3FeaturesExtractor(space, layout=layout, mappings=mappings, **kw).eval()


def _facts(fe: Any, rows: np.ndarray, batch: int = 128) -> torch.Tensor:
    """The family's RAW move facts ``[N,4,RAW]`` for every row."""
    cell = fe.move_resolution_cell
    outs: List[torch.Tensor] = []
    orig = cell.forward

    def hook(ops):
        outs.append(cell.raw(ops)[0])
        return orig(ops)
    cell.forward = hook
    try:
        with torch.no_grad():
            for i in range(0, len(rows), batch):
                fe({"observation": torch.as_tensor(rows[i:i + batch], dtype=torch.float32)})
    finally:
        cell.forward = orig
    return torch.cat(outs, dim=0)


def outcome(lines: List[Tuple[str, ...]], me: str, want: str, damaging: bool,
            self_boosts: bool) -> Optional[Dict[str, Any]]:
    """Resolve one played move against the viewer's protocol lines for that turn (split on ``|``)."""
    them = "p2" if me == "p1" else "p1"
    switched = False
    for i, ev in enumerate(lines):
        if len(ev) < 3:
            continue
        tag, who = ev[1], ev[2]
        if tag in ("switch", "drag") and who.startswith(them):
            switched = True
        if not (tag == "move" and who.startswith(me) and len(ev) > 3 and _norm(ev[3]) == want
                and not any(x.startswith("[from]") for x in ev[4:])):
            continue
        failed = "[miss]" in ev or "[notarget]" in ev
        dealt = False
        after: List[str] = []
        for nx in lines[i + 1:]:
            if len(nx) < 2:
                continue
            t = nx[1]
            if t in _ACTION_TAGS:
                break
            line = "|".join(nx)
            after.append(line)
            if t in _FAIL_TAGS or (t == "-activate" and len(nx) > 2 and nx[2].startswith(them)
                                   and ("Protect" in line or "Detect" in line or "Magic Coat" in line)) \
                    or (t == "-message" and "Clause" in line):
                failed = True
            if len(nx) > 2 and nx[2].startswith(them) and (
                    (t == "-damage" and "[from]" not in line) or t == "-ohko"
                    or (t == "-activate" and "Substitute" in line and "[damage]" in line)
                    or (t == "-end" and "Substitute" in line)):
                dealt = True
        if damaging and not dealt:
            failed = True
        if want == "destinybond":
            failed = not any("-activate" in x and "Destiny Bond" in x for x in after)
        if want == "magiccoat":
            failed = not any(x.startswith("|-activate|" + me) and "Magic Coat" in x for x in after)
        if want == "curse" or (self_boosts and not damaging):
            moved = [x.split("|") for x in after if x.startswith(("|-boost|" + me, "|-unboost|" + me))]
            failed = failed or not (any(len(x) > 4 and x[4] not in ("0", "") for x in moved)
                                    or (want == "curse" and any(x.startswith("|-start|") and "Curse" in x
                                                                for x in after)))
        return {"executed": True, "failed": failed, "switched": switched}
    return {"executed": False, "failed": None, "switched": switched}


def check(n_battles: int, belief_tokens: str = "blob") -> Dict[str, Any]:
    from agents import gen3_data
    from agents.battle.core_obs import wrap_row
    from agents.battle.rust_core_parity import run_core
    from agents.model.move_resolution_rules import MOVE_RESOLUTION_MOVE_IDX as MI
    from main.policy_spectrum.bank import load_bank

    battles = load_bank(BANK, verify=False).battles[:n_battles]
    rows: List[np.ndarray] = []
    meta: List[Tuple[str, str, int, List[Tuple[str, ...]]]] = []
    for res in run_core([b.recorded() for b in battles], trackers=True, obs=True):
        assert res["ok"], res.get("error")
        chunks = res["chunks"]
        for v in (0, 1):
            for entry in res["trackers"][v]:
                choice = entry.get("choice")
                if "obs" not in entry or not choice or not choice.startswith("move ") or choice == "move struggle":
                    continue
                tokens = {int(k): t for k, t in entry["tokens"].items()}
                slot = next((k for k in range(4) if tokens.get(6 + k) == choice), None)
                if slot is None:
                    continue
                chosen = _norm(choice[len("move "):])
                if chosen in _CALLERS:
                    continue
                turn_lines: List[Tuple[str, ...]] = []
                for side, lines in chunks[int(entry["after"]) + 1:]:
                    if side != v:
                        continue
                    stop = False
                    for line in lines:
                        if line.startswith(("|turn|", "|win|", "|tie")):
                            stop = True
                            break
                        turn_lines.append(tuple(line.split("|")))
                    if stop:
                        break
                rows.append(np.array(wrap_row(entry["obs"]), dtype=np.float32))
                meta.append((res["label"], f"p{v + 1}", slot, turn_lines))
                meta[-1] = meta[-1] + (chosen,)  # type: ignore[assignment]
    fe = _extractor(belief_tokens)
    facts = _facts(fe, np.stack(rows)) if rows else torch.zeros(0, 4, len(MI))
    gates: collections.Counter = collections.Counter()
    certain = [0, 0]
    violations: List[str] = []
    for i, (label, me, slot, lines, chosen) in enumerate(meta):  # type: ignore[misc]
        raw = gen3_data.moves.raw().get(chosen, {})
        md = gen3_data.moves.get(chosen)
        out = outcome(lines, me, chosen, str(raw.get("category")) in ("Physical", "Special"),
                      bool(md is not None and md.self_boosts))
        if out is None or not out["executed"] or chosen in _GATE_EXCLUDED:
            continue
        f = facts[i, slot]
        resolved = not out["failed"]
        p_act = float(f[MI["p_act"]])
        if not out["switched"] and (float(f[MI["p_lands_stay"]]) == 0.0 or p_act == 0.0):
            gates["stay_zero"] += 1
            if resolved:
                violations.append(f"{label} {me} {chosen}: p_lands_stay=0 but it RESOLVED")
        if out["switched"] and (float(f[MI["p_lands_switch"]]) == 0.0 or p_act == 0.0):
            gates["switch_zero"] += 1
            if resolved:
                violations.append(f"{label} {me} {chosen}: p_lands_switch=0 but it RESOLVED")
        if not out["switched"] and float(f[MI["p_lands_stay"]]) == 1.0 and p_act == 1.0:
            certain[0] += 1
            certain[1] += int(not resolved)
    return {"decisions": len(meta), "gates": dict(gates), "certain": certain, "violations": violations}


@pytest.mark.parametrize("belief_tokens", ["blob", "fixed_mass"])
def test_a_stated_zero_is_never_contradicted_on_real_battles(belief_tokens):
    r = check(N_BATTLES, belief_tokens)
    assert r["decisions"] > 0, "no played move reached the check"
    assert sum(r["gates"].values()) > 0, f"NOT EXERCISED: no exact-zero claim met an executed move ({r})"
    assert not r["violations"], "\n".join(r["violations"][:20])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
    r = check(n, sys.argv[2] if len(sys.argv) > 2 else "blob")
    print(json.dumps({k: v for k, v in r.items() if k != "violations"}), flush=True)
    print("\n".join(r["violations"][:50]) or "no violations", flush=True)
    sys.exit(1 if r["violations"] or not sum(r["gates"].values()) else 0)
