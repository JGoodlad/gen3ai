"""REAL-BATTLE check of the damage operator's OUTGOING STATUS-LANDING channel (`DamageOperator._status_landing`,
gen3_unified_status_landing_v1 under gen3_op_ability_known_v1 + gen3_op_status_rules_v1) on the RUST core — no
poke-env.

Replays banked real battles (the M5 Lane S turn bank: trained-policy games, inputs only) through the Rust core
(`core_events --trackers --obs`, the same replay `move_resolution_bridge_integration_test` runs), runs a
production-config extractor (move-resolution OFF, the production default) over every answered decision's
observation, captures the op's ``[B, 8]`` status channel (4 ``p_land`` in REQUEST order == action 6+k, then 4
``known`` bits) and resolves each played STATUS move (op ``MOVE_INFLICTS_STATUS`` = 1, Yawn and Leech Seed
included) against the viewer's protocol for that turn:

  * EXECUTED — a ``|move|`` line from our active naming the chosen move (a caller's ``[from]`` line excluded);
  * LANDED   — after it, before the next action line, a ``-status`` line on THEIR side (any status: only a status
               move's own effect can print one there in that window), or for Yawn a ``-start … move: Yawn`` /
               for Leech Seed a ``-start … Leech Seed`` line on their side.

**The gate.** A CERTAIN zero (``p_land == 0`` with ``known == 1``) is never contradicted: when their active did not
switch before our move, an executed move did NOT land. The gate must be EXERCISED (≥1 certain-zero claim met an
executed move). Exclusion classes, each named and counted:

  * ``not_executed``   — our move never printed (we fainted / flinched / were fully paralysed / slept …);
  * ``switched``       — their active switched (or was dragged) before our move: the claim was about another mon;
  * ``status_changed`` — their side's status changed before our move (a ``-status`` / ``-curestatus`` /
                         ``-cureteam`` on their side): Rest, a wake-up lifting Sleep Clause, Heal Bell … — the
                         claim was true of the board it was made on;
  * callers (Sleep Talk / Snore / Metronome / Mirror Move / Nature Power / Assist) are not the move they call;
  * Struggle and switches carry no status move.

**The fail-on-revert half.** The same rows with the OLD ability read — the per-mon ability ``known`` column of
THEIR six set to ``ability1_ids > 0``, exactly the pre-fix ``id > 0`` rule (only the ability part: the Safeguard /
clause / Yawn rules stay new) — must make ≥1 certain-zero claim the protocol CONTRADICTS on the default N (an
unrevealed Snorlax whose top-1 prior is Immunity, Toxicked through Thick Fat). That assertion pins that the
default N HOLDS the discriminating rows, so the forward gate above fails if gen3_op_ability_known_v1 is reverted:
measured 2026-10-07 at N = 48, the old read makes 9 contradicted certain-zero claims (Toxic, two battles), and an
actual revert (the op's `ability_known` / `revealed_ability1_ids` bound to the ``id > 0`` rule) fails the forward
gate with the same 9. Whole-bank numbers: `designs/research_state/measurements/op_gigo_2026-10-07/`.

The same battles every run (a committed bank, a deterministic replay). The full bank as a script:
    python src/agents/model/op_status_landing_bridge_integration_test.py [n_battles]
"""
from __future__ import annotations

import collections
import dataclasses
import json
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

from agents.model.move_resolution_bridge_integration_test import BANK, _ACTION_TAGS, _CALLERS, _norm

pytestmark = [pytest.mark.sim, pytest.mark.integration]

N_BATTLES = 48
_STATUS_CHANGE_TAGS = ("-status", "-curestatus", "-cureteam")


@dataclasses.dataclass
class Decision:
    """One of our answered decisions, with the protocol that followed it (the viewer's lines up to the turn end)."""
    label: str
    me: str                    # "p1" / "p2"
    turn: int
    choice: str                # the raw choice token ("move toxic", "switch 3", …)
    slot: Optional[int]        # request slot k (action 6+k) for a move choice, else None
    chosen: Optional[str]      # normalized move id for a move choice, else None
    lines: List[Tuple[str, ...]]


def _extractor() -> Any:
    """The PRODUCTION-config extractor (`designs/production_config.json`; move-resolution at its production value)."""
    import gymnasium as gym

    from agents.model.damage_tables import sanitize_historical_move_floor
    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    from utils.paths import repo_path
    import inspect
    cfg = json.load(open(repo_path("designs", "production_config.json")))
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    kw = {k: v for k, v in cfg.items() if k in sig}
    sanitize_historical_move_floor(kw)
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    torch.manual_seed(0)
    fe = Gen3FeaturesExtractor(space, layout=layout, mappings=mappings, **kw).eval()
    assert fe.damage_op is not None and fe.damage_op.outgoing, "production config has no outgoing damage op"
    return fe


def collect(n_battles: int) -> Tuple[np.ndarray, List[Decision]]:
    """Replay the first ``n_battles`` of the bank; every answered decision of BOTH viewers with its obs row."""
    from agents.battle.core_obs import wrap_row
    from agents.battle.rust_core_parity import run_core
    from main.policy_spectrum.bank import load_bank

    battles = load_bank(BANK, verify=False).battles[:n_battles]
    rows: List[np.ndarray] = []
    meta: List[Decision] = []
    for res in run_core([b.recorded() for b in battles], trackers=True, obs=True):
        assert res["ok"], res.get("error")
        chunks = res["chunks"]
        for v in (0, 1):
            for entry in res["trackers"][v]:
                choice = entry.get("choice")
                if "obs" not in entry or not choice:
                    continue
                after = int(entry["after"])
                turn = 0
                for side, lines in chunks[:after + 1]:
                    if side == v:
                        for line in lines:
                            if line.startswith("|turn|"):
                                turn = int(line.split("|")[2])
                slot: Optional[int] = None
                chosen: Optional[str] = None
                if choice.startswith("move ") and choice != "move struggle":
                    tokens = {int(k): t for k, t in entry["tokens"].items()}
                    slot = next((k for k in range(4) if tokens.get(6 + k) == choice), None)
                    chosen = _norm(choice[len("move "):]) if slot is not None else None
                turn_lines: List[Tuple[str, ...]] = []
                for side, lines in chunks[after + 1:]:
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
                meta.append(Decision(res["label"], f"p{v + 1}", turn, choice, slot, chosen, turn_lines))
    return (np.stack(rows) if rows else np.zeros((0, 0), np.float32)), meta


def old_ability_ctx(ctx: Any) -> Any:
    """``ctx`` with THEIR six's ability ``known`` column set to ``ability1_ids > 0`` — the pre-fix "a non-zero id
    is a reveal" rule, reproduced on the same observation (ours are untouched: the op reads our abilities raw)."""
    from agents.model.extractor_ctx import POKEMON_ABILITY_KNOWN_OFFSET
    from agents.observation.constants import TEAM_SIZE
    pp = ctx.pokemon_part.clone()
    opp = slice(TEAM_SIZE, 2 * TEAM_SIZE)
    pp[:, opp, POKEMON_ABILITY_KNOWN_OFFSET] = (ctx.ability1_ids[:, opp] > 0).to(pp.dtype)
    return dataclasses.replace(ctx, pokemon_part=pp)


def op_reads(fe: Any, rows: np.ndarray, fn: Callable[..., Dict[str, torch.Tensor]],
             batch: int = 128) -> Dict[str, torch.Tensor]:
    """Run the extractor over ``rows``; at the op's forward call ``fn(op, ctx, args, kwargs)`` BEFORE the op's own
    forward (so the stashes the rest of the extractor reads are the real forward's) and concatenate the per-row
    tensors it returns. ``fn`` reaches the unhooked forward as ``type(op).forward(op, …)``."""
    op = fe.damage_op
    orig = op.forward
    outs: Dict[str, List[torch.Tensor]] = collections.defaultdict(list)

    def hook(ctx, *args, **kwargs):
        got = fn(op, ctx, args, kwargs)
        out = orig(ctx, *args, **kwargs)
        for k, t in got.items():
            outs[k].append(t.detach())
        return out
    op.forward = hook
    try:
        with torch.no_grad():
            for i in range(0, len(rows), batch):
                fe({"observation": torch.as_tensor(rows[i:i + batch], dtype=torch.float32)})
    finally:
        op.forward = orig
    return {k: torch.cat(v, dim=0) for k, v in outs.items()}


def status_reads(op: Any, ctx: Any, args: Any, kwargs: Any) -> Dict[str, torch.Tensor]:
    """The op's status channel, NEW and OLD-ability, plus the request-order inflictor bits."""
    return {"new": op._status_landing(ctx), "old": op._status_landing(old_ability_ctx(ctx)),
            "inflicts": op.MOVE_INFLICTS_STATUS[ctx.our_active_req_move_ids]}


def status_outcome(lines: List[Tuple[str, ...]], me: str, want: str) -> Dict[str, Any]:
    """Resolve one played status move against the viewer's protocol lines for that turn (split on ``|``)."""
    them = "p2" if me == "p1" else "p1"
    switched = changed = False
    for i, ev in enumerate(lines):
        if len(ev) < 3:
            continue
        tag, who = ev[1], ev[2]
        if tag in ("switch", "drag") and who.startswith(them):
            switched = True
        if tag in _STATUS_CHANGE_TAGS and who.startswith(them):
            changed = True
        if not (tag == "move" and who.startswith(me) and len(ev) > 3 and _norm(ev[3]) == want
                and not any(x.startswith("[from]") for x in ev[4:])):
            continue
        landed = False
        status: Optional[str] = None
        for nx in lines[i + 1:]:
            if len(nx) < 2:
                continue
            t = nx[1]
            if t in _ACTION_TAGS:
                break
            if len(nx) > 2 and nx[2].startswith(them):
                line = "|".join(nx)
                if t == "-status":
                    landed, status = True, (nx[3] if len(nx) > 3 else None)
                elif t == "-start" and ((want == "yawn" and "Yawn" in line)
                                        or (want == "leechseed" and "Leech Seed" in line)):
                    landed, status = True, want
        return {"executed": True, "landed": landed, "status": status, "switched": switched, "changed": changed}
    return {"executed": False, "landed": False, "status": None, "switched": switched, "changed": changed}


def check(n_battles: int) -> Dict[str, Any]:
    rows, meta = collect(n_battles)
    reads = op_reads(_extractor(), rows, status_reads) if len(meta) else {}
    excl: collections.Counter = collections.Counter()
    gate = {"new_certain_zero": 0, "old_certain_zero": 0, "played": 0, "landed": 0}
    violations: List[str] = []
    old_contradicted: List[str] = []
    for i, d in enumerate(meta):
        if d.slot is None or d.chosen is None:
            continue
        if d.chosen in _CALLERS:
            excl["caller"] += 1
            continue
        if float(reads["inflicts"][i, d.slot]) < 0.5:
            continue
        o = status_outcome(d.lines, d.me, d.chosen)
        if not o["executed"]:
            excl["not_executed"] += 1
            continue
        if o["switched"]:
            excl["switched"] += 1
            continue
        if o["changed"]:
            excl["status_changed"] += 1
            continue
        gate["played"] += 1
        gate["landed"] += int(o["landed"])
        where = f"{d.label} {d.me} t{d.turn} {d.chosen}"
        for key, r in (("new", reads["new"]), ("old", reads["old"])):
            if float(r[i, d.slot]) == 0.0 and float(r[i, 4 + d.slot]) == 1.0:
                gate[f"{key}_certain_zero"] += 1
                if o["landed"]:
                    (violations if key == "new" else old_contradicted).append(
                        f"{where}: {key} op p_land=0 known=1 but it LANDED ({o['status']})")
    return {"decisions": len(meta), "gate": gate, "excluded": dict(excl), "violations": violations,
            "old_contradicted": old_contradicted}


def test_a_certain_zero_status_landing_is_never_contradicted_on_real_battles():
    r = check(N_BATTLES)
    assert r["decisions"] > 0, "no decision reached the check"
    assert r["gate"]["new_certain_zero"] > 0, f"NOT EXERCISED: no certain-zero claim met an executed move ({r})"
    assert not r["violations"], "\n".join(r["violations"][:20])
    # fail-on-revert of gen3_op_ability_known_v1: the OLD `id > 0` read makes a certain-zero claim that LANDED
    assert r["old_contradicted"], f"the OLD ability read made no contradicted certain-zero claim ({r['gate']})"


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
    r = check(n)
    print(json.dumps({k: v for k, v in r.items() if k not in ("violations", "old_contradicted")}), flush=True)
    print("NEW violations:\n" + ("\n".join(r["violations"][:50]) or "none"), flush=True)
    print(f"OLD contradicted ({len(r['old_contradicted'])}):\n" + "\n".join(r["old_contradicted"][:50]), flush=True)
    sys.exit(1 if r["violations"] or not r["gate"]["new_certain_zero"] else 0)
