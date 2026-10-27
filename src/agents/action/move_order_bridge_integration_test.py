"""REAL-BRIDGE move-order fuzz (gen3_move_legality_by_id_v1): the scored move slot ↔ the chosen action
index ↔ the move the SIM executed, on real battles.

Replays banked eval battles (the M5 Lane S turn bank: real trained-policy games, inputs only) through the
Rust core (`core_events --trackers --obs`) and checks, at every answered decision of BOTH viewers:

* every obs row passes the row guard `check_obs_move_order` with the mask it was served with;
* action ``6+k``'s choice token names request move ``k`` (``move <id>``, Hidden Power by prefix);
* the sorted per-mon slot the pointer head scores logit ``6+k`` from (`active_request_sorted_match`)
  holds request move ``k`` — so the token the head scores IS the move the action sends;
* for a played move, the viewer's next ``|move|`` line names the chosen move (a caller move such as
  Sleep Talk, or a turn on which the mon did not move, is skipped);
* our active's legality lands on each move's OWN sorted slot (`active_move_legality_sorted`).

The full-bank run of the same audit (580 battles, 42,465 decisions, 2026-10-06): 0 violations of any
check above after the fix; before it, 2,542 of 37,358 move-bearing rows had a legality bit on the wrong
move — ledger FINDING gen3_move_legality_by_id_v1.
"""
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.sim, pytest.mark.integration]

N_BATTLES = 24
BANK = Path(__file__).resolve().parents[3] / "designs/research_state/measurements/m5_laneS/bank_v1"
CALLERS = {"sleeptalk", "metronome", "mirrormove", "naturepower", "assist"}


def _norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    return "hiddenpower" if s.startswith("hiddenpower") else s


def test_scored_slot_action_and_executed_move_agree_on_real_battles():
    from agents import gen3_data
    from agents.action.ordering_integrity import check_obs_move_order, row_offsets
    from agents.battle.core_obs import wrap_row
    from agents.battle.core_replay import run_core
    from agents.model.extractor_ctx import active_move_legality_sorted, active_request_sorted_match
    from main.policy_spectrum.bank import load_bank

    battles = load_bank(BANK, verify=False).battles[:N_BATTLES]
    o = row_offsets()
    n = {"rows": 0, "tokens": 0, "executed": 0, "misaligned_rows": 0}
    rows, masks = [], []
    for res in run_core([b.recorded() for b in battles], trackers=True, obs=True):
        assert res["ok"], res.get("error")
        chunks = res["chunks"]
        for v in (0, 1):
            for entry in res["trackers"][v]:
                if "obs" not in entry or entry.get("choice") is None:
                    continue
                row = np.array(wrap_row(entry["obs"]))
                mask = np.asarray(entry["mask"], dtype=bool)
                check_obs_move_order(row, mask[None], where=f"{res['label']} p{v + 1}")
                rows.append(row)
                masks.append(mask)
                n["rows"] += 1
                team = row[o.team0:o.team0 + 6 * o.mon_dim].reshape(6, o.mon_dim)
                act = np.flatnonzero(team[:, o.active_col] > 0.5)
                req = row[o.req_ids]
                tokens = {int(k): t for k, t in entry["tokens"].items()}
                for k in range(4):
                    tok = tokens.get(6 + k)
                    if tok is None:
                        continue
                    n["tokens"] += 1
                    # the head scores logit 6+k from the sorted slot whose id == request id k
                    slot_ids = team[act[0], o.slot_id_cols]
                    assert (slot_ids == req[k]).sum() == 1, (res["label"], v, slot_ids, req)
                    md = gen3_data.moves.get(tok[len("move "):]) if tok != "move 1" else None
                    if md is not None:
                        assert float(md.num) == float(req[k]), (res["label"], v, k, tok, req)
                choice = entry["choice"]
                if not choice.startswith("move ") or choice in ("move struggle", "move 1"):
                    continue
                chosen = _norm(choice[len("move "):])
                if chosen in CALLERS:
                    continue
                tag = f"|move|p{v + 1}a: "
                executed = None
                for side, lines in chunks[int(entry["after"]) + 1:]:
                    if side != v:
                        continue
                    stop = False
                    for line in lines:
                        if line.startswith(tag) and "[from]" not in line:
                            executed = _norm(line.split("|")[3])
                            stop = True
                            break
                        if line.startswith(("|turn|", "|win|", "|tie")):
                            stop = True
                            break
                    if stop:
                        break
                if executed is not None:
                    n["executed"] += 1
                    assert executed == chosen, (res["label"], v, entry["after"], choice, executed)

    x = torch.as_tensor(np.stack(rows))
    team = x[:, o.team0:o.team0 + 6 * o.mon_dim].reshape(len(x), 6, o.mon_dim)
    active = team[:, :, o.active_col] > 0.5
    ctx = SimpleNamespace(batch_size=len(x), device=torch.device("cpu"),
                          our_active_idx=active.float().argmax(1),
                          all_move_ids=team[:, :, torch.as_tensor(o.slot_id_cols)].long(),
                          our_active_req_move_ids=x[:, o.req_ids], our_active_req_move_legal=x[:, o.req_legal])
    legal_sorted = active_move_legality_sorted(ctx)
    match = active_request_sorted_match(ctx)
    for b in range(len(x)):
        req = x[b, o.req_ids].tolist()
        if not any(r > 0 for r in req):
            continue
        slot_ids = ctx.all_move_ids[b, ctx.our_active_idx[b]].tolist()
        want = [float(masks[b][6 + req.index(s)]) if s in req and s > 0 else 0.0 for s in slot_ids]
        assert legal_sorted[b].tolist() == want, (b, slot_ids, req, legal_sorted[b].tolist())
        if want != x[b, o.req_legal].tolist():
            n["misaligned_rows"] += 1          # where the pre-fix POSITIONAL write was wrong
        assert match[b].sum().item() == sum(1 for r in req if r > 0)
    # non-vacuous: the sample exercises every check, including rows the old positional write got wrong
    assert n["rows"] > 1000 and n["tokens"] > 3000 and n["executed"] > 500, n
    assert n["misaligned_rows"] > 0, n


def test_mimic_transform_and_the_recharge_turn_pass_the_guard_through_the_core():
    """Three constructed battles the bank does not hold: Mimic (the copied move takes Mimic's request slot),
    Transform (the moveset becomes the target's) and Hyper Beam's RECHARGE turn (request `[recharge]`, an
    all-zero request block, one legal action — the guard's single-forced-action exemption; found by the
    `--debug` smoke on this fix's first run). Every row passes the guard with its mask."""
    from agents.action.ordering_integrity import check_obs_move_order, row_offsets
    from agents.battle.core_obs import wrap_row
    from agents.battle.core_replay import RecordedBattle, run_core

    claydol = "Claydol||leftovers|levitate|rapidspin,mimic,earthquake,toxic|Sassy|252,100,112,,16,28|||||"
    ditto = "Ditto||leftovers|limber|transform|Relaxed|252,,252,,4,|||||"
    lax = "Snorlax||leftovers|thickfat|bodyslam,curse,rest,shadowball|Careful|252,,4,,252,|||||"
    beamer = "Snorlax||leftovers|thickfat|hyperbeam,curse,rest,shadowball|Careful|252,,4,,252,|||||"
    skarm = "Skarmory||leftovers|keeneye|spikes,toxic,drillpeck,roar|Impish|252,,252,,4,|||||"

    def battle(label, seed, p1, cmds):
        return RecordedBattle(label=label, format_id="gen3ou", seed=f"sodium,{seed:032d}",
                              p1={"name": "A", "team": p1 + "]" + skarm}, p2={"name": "B", "team": lax + "]" + skarm},
                              commands=[[side, c] for pair in cmds for side, c in zip(("p1", "p2"), pair)])

    battles = [battle("mimic", 1, claydol, [("move earthquake", "move bodyslam"), ("move mimic", "move bodyslam"),
                                            ("move bodyslam", "move curse")]),
               battle("transform", 2, ditto, [("move transform", "move curse"), ("move curse", "move curse")]),
               battle("recharge", 3, beamer, [("move hyperbeam", "move curse"), ("move 1", "move curse"),
                                              ("move curse", "move curse")])]
    o = row_offsets()
    seen = {"mimicked": False, "transformed": False, "recharge": False}
    for res in run_core(battles, trackers=True, obs=True):
        assert res["ok"], res.get("error")
        for entry in res["trackers"][0]:
            if "obs" not in entry:
                continue
            row = np.array(wrap_row(entry["obs"]))
            mask = np.asarray(entry["mask"], dtype=bool)
            check_obs_move_order(row, mask[None], where=res["label"])
            toks = set(entry["tokens"].values())
            seen["mimicked"] |= res["label"] == "mimic" and "move bodyslam" in toks
            seen["transformed"] |= res["label"] == "transform" and "move shadowball" in toks
            if res["label"] == "recharge" and toks == {"move 1"}:
                seen["recharge"] = True
                assert not (row[o.req_ids] > 0).any() and mask.sum() == 1
    assert all(seen.values()), seen
