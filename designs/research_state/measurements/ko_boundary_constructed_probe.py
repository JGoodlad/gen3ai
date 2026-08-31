"""KO-BOUNDARY DECODABILITY, constructed arm — the same question as
`ko_boundary_decodability_probe.py` (the population arm) asked where the truth is EXACT.

`starmie_ttar_risk_probe_v2_2026-08-31.md` §3 measured the win-prob head responding to the
opponent's displayed HP bar at a smooth ~0.7pp per HP point with **zero excess response at the
KO-roll boundary** — the 295→296 step, where the TRUE Surf value drops 5.9pp, is
indistinguishable from its neighbours. That is a statement about the HEAD. This script asks the
same sweep what the LAYER BELOW says: it re-runs v1's construction and, at each engineered
opponent HP, reads the DamageOperator's own outgoing `pko` channel for the Surf slot alongside
the head.

If the op's pko tracks the exact 16-roll KO fraction while the head does not, the boundary
information is present one layer below the blind head, and the defect is delivery/supervision
rather than representation coverage.

Imports v1's construction verbatim (teams, the parity-rule choreography, the decision-state
asserts, the measured roll table) — v1's script and JSON are read-only here.

Run (repo root; needs deps/pokemon-showdown built — node bridge):
    python designs/research_state/measurements/ko_boundary_constructed_probe.py
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

CPU-only, <=2 cores. Resumable: each H is appended to the JSON as it lands.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # the v1 sibling module lives here

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch

torch.set_num_threads(2)

from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.model.snapshot import current_model_version, load_foreign_opponent
from agents.observation.state_encoder import load_mappings
from utils.bridge.local_battle_runner import run_local_battles
from utils.paths import main_models_dir

import starmie_ttar_risk_probe as v1   # noqa: E402  (sibling module — added to sys.path below)

OUT_DIR = Path(__file__).resolve().parent
OUT = OUT_DIR / "ko_boundary_constructed_2026-08-31.json"
V1_JSON = OUT_DIR / "starmie_ttar_risk_probe_2026-08-30.json"
CKPT = str(main_models_dir() / "ai_v9_59_R2ACTION_0827" / "final_model.zip")


class OpTapCapture(v1.OurSideModel):
    """v1's model-backed scripted arm, with the DamageOperator + trunk taps added to the capture."""

    def _capture(self, battle, obs_dict, board):
        idx, cap = super()._capture(battle, obs_dict, board)
        fe = self.model.policy.features_extractor
        op = fe.damage_op
        t = op.last_tensors
        legal = self._get_tracker(battle).last_ctx.legal
        slots = [m.id for m in legal.move_slots]
        opm = t.out_per_move[0].numpy()                 # [4,4] = [low, high, crit, pko] per req slot
        cap["op_out_per_move"] = {slots[k]: [round(float(x), 6) for x in opm[k]]
                                  for k in range(min(len(slots), opm.shape[0]))}
        cap["op_p_outspeed"] = round(float(t.out_p_outspeed[0, 0]), 6)
        # PRE-GAIN cells: `out_per_move` above is the flat block, which is multiplied by a LEARNED
        # per-feature gain, so its pko is `pko x gain`. `last_out_cells` is the same physics before
        # that scaling — the honest number to compare against a ground-truth probability.
        oc = op.last_out_cells                          # [B,4,6,5] = [low,high,crit,pko,type_mult]
        if oc is not None:
            j = int(fe.last_opp_active_local[0])        # which of their 6 slots is active
            cell = oc[0, :, j, :].numpy()
            cap["op_out_cells_raw_vs_active"] = {
                slots[k]: [round(float(x), 6) for x in cell[k]]
                for k in range(min(len(slots), cell.shape[0]))}
            cap["opp_active_local"] = j
        cap["op_incoming_max_pko"] = round(float(t.incoming_rows[0, :, [3, 8]].max()), 6)
        cap["value_pooled_norm"] = round(float(np.linalg.norm(fe.last_value_pooled[0].numpy())), 4)
        self._capture_sink[-1] = cap
        return idx, cap


def run_point(model, mappings, h, tag, seed=(7, 11, 13, 17)) -> dict:
    sink = []
    a1 = AccountConfiguration(f"KBr{tag}"[:17], None)
    a2 = AccountConfiguration(f"KBo{tag}"[:17], None)
    ours = OpTapCapture(
        model=model, team=v1.our_team(v1.STARMIE_FRAIL, v1.marsh_hp_ev_for(h)),
        battle_format=v1.FORMAT, server_configuration=LocalhostServerConfiguration,
        mappings=mappings, account_configuration=a1, start_listening=False,
        stochastic=False, expected_h=h, expected_our_hp=262, capture_sink=sink)
    theirs = v1.TtarSide(battle_format=v1.FORMAT, team=v1.THEIR_TEAM,
                         server_configuration=LocalhostServerConfiguration,
                         account_configuration=a2, start_listening=False)
    asyncio.run(run_local_battles(ours, theirs, 1, battle_format=v1.FORMAT,
                                  seed=list(seed), impl="node"))
    assert len(sink) == 1, f"decision reached {len(sink)} times (wanted exactly 1)"
    return sink[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    args = ap.parse_args()

    v1_res = json.loads(V1_JSON.read_text())
    surf_rolls = v1_res["tables"]["surf_vs_cbtar"]["rolls"]
    assert len(surf_rolls) == 16
    # v1's 17-point sweep ∪ v2's crossover micro-steps -> 7 CONSECUTIVE H around the k=13→12 boundary
    hs = sorted(set([284] + [r + 1 for r in surf_rolls] + [294, 295, 297, 298, 299]))

    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res.setdefault("meta", {
        "date": "2026-08-31", "checkpoint": args.ckpt, "format": v1.FORMAT, "impl": "node",
        "surf_rolls": surf_rolls, "ttar_maxhp": v1.TTAR_MAXHP, "capture_seed": [7, 11, 13, 17],
        "source": "imports starmie_ttar_risk_probe (v1) construction verbatim",
    })
    res.setdefault("sweep", [])
    done = {r["H"] for r in res["sweep"]}

    mappings = load_mappings()
    model, _ = load_foreign_opponent(args.ckpt, current_version=current_model_version(mappings),
                                     device="cpu")

    for i, h in enumerate(hs):
        if h in done:
            continue
        cap = run_point(model, mappings, h, f"k{i}")
        k = sum(1 for r in surf_rolls if r >= h)
        slots = cap["move_slots"]
        row = {
            "H": h, "hp_frac": round(h / v1.TTAR_MAXHP, 4),
            "surf_ko_rolls": k,
            "e_ko_surf": round(v1.CRIT_P + (1 - v1.CRIT_P) * k / 16, 4),
            "p_surf": cap["probs"][6 + slots.index("surf")],
            "p_pump": cap["probs"][6 + slots.index("hydropump")],
            "win_prob": cap["win_prob"], "value": cap["value"],
            "op_pko_surf": cap["op_out_per_move"]["surf"][3],
            "op_pko_pump": cap["op_out_per_move"]["hydropump"][3],
            "op_high_surf": cap["op_out_per_move"]["surf"][1],
            "op_low_surf": cap["op_out_per_move"]["surf"][0],
            "op_pko_surf_raw": cap.get("op_out_cells_raw_vs_active", {}).get("surf", [None] * 4)[3],
            "op_pko_pump_raw": cap.get("op_out_cells_raw_vs_active", {})
                                  .get("hydropump", [None] * 4)[3],
            "op_high_surf_raw": cap.get("op_out_cells_raw_vs_active", {}).get("surf", [None] * 4)[1],
            "op_incoming_max_pko": cap["op_incoming_max_pko"],
            "argmax": slots[cap["argmax_action"] - 6] if cap["argmax_action"] >= 6 else None,
        }
        res["sweep"].append(row)
        res["sweep"].sort(key=lambda r: r["H"])
        OUT.write_text(json.dumps(res, indent=1))
        print(row)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
