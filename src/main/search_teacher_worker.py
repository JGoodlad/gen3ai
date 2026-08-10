"""Search-teacher worker subprocess (mirrors ``main.eval_worker``): run the search + confirm on a
slice of candidates against the FROZEN trainee snapshot, publish the verified-better corrections.

Isolated in a subprocess on purpose (like eval): the search needs a FROZEN model (the live trunk
mutates during training) + its own POKE_LOOP + spare cores. Reads ``config_<wid>.json``, writes a
shard ``<result_path>.json`` (scalars + per-status counts) + ``<result_path>.npz`` (the obs/mask
arrays). The parent ``SearchTeacherCallback`` collects shards into the model's ``CorrectionBuffer``.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np


def run(cfg_path: str) -> None:
    from main.prober.session import ProbeSession
    from utils.bridge.search_session import SearchSession
    from agents.training.teacher.selection import Candidate
    from agents.training.teacher.opponent_resolver import resolve_opponent
    from agents.training.teacher.produce import produce_correction

    with open(cfg_path) as f:
        cfg = json.load(f)
    run_dir = cfg["run_dir"]
    result_path = cfg["result_path"]
    cands = [Candidate(**c) for c in cfg["candidates"]]
    # Which sim engine the search/replay children run — the parent threads the run's bridge impl
    # ("node" default). Session-wide on the ProbeSession AND on the warm SearchSession below, so
    # both halves of a correction (search + confirm) are produced by the same engine.
    impl = cfg.get("impl", "node")

    # The FROZEN trainee snapshot drives both the search (ProbeModel) and the confirm (raw model) —
    # ckpt_override pins every battle's model to it (instead of the exact→nearest→recent ladder).
    sess = ProbeSession(run_dir, ckpt_override=cfg["snapshot_path"], impl=impl)

    # OPD: build the improved distribution π' KL target when requested (config-threaded, mirrors
    # confirm_rollouts/depth); off = the AWR-only behaviour (pi_target stays None → the .npz omits it).
    opd_build_pi_target = bool(cfg.get("opd_build_pi_target", False))
    opd_beta = float(cfg.get("opd_beta", 1.0))

    corrections, status = [], {}
    # ONE warm SearchSession reused across all candidates → the Node spawn is amortized (perf L4).
    with SearchSession(timeout=float(cfg.get("timeout", 300.0)), impl=impl) as ss:
        for c in cands:
            opp_ckpt, opp_src = resolve_opponent(run_dir, c.opponent, c.step)
            try:
                corr, st = produce_correction(
                    sess, c, opponent_ckpt=opp_ckpt, opponent_source=opp_src,
                    confirm_rollouts=int(cfg["confirm_rollouts"]), depth=int(cfg["depth"]),
                    beam=int(cfg["beam"]), top_k=int(cfg["top_k"]),
                    margin_min=float(cfg["margin_min"]), search_session=ss,
                    build_pi_target=opd_build_pi_target, opd_beta=opd_beta)
            except Exception as e:  # noqa: BLE001 — one bad candidate never kills the worker's slice
                corr, st = None, f"error:{type(e).__name__}"
            status[st] = status.get(st, 0) + 1
            if corr is not None:
                corrections.append(corr)

    if corrections:
        arrays = dict(
            obs=np.stack([c.obs for c in corrections]).astype(np.float32),
            mask=np.stack([c.action_mask for c in corrections]).astype(np.int8))
        # OPD: pack π' as a [n, 11] array (a NaN row = None, so the parent None-guards per-row). Omitted
        # entirely when no correction carries a target (an AWR-only run) → the parent reads no key.
        if any(c.pi_target is not None for c in corrections):
            arrays["pi_target"] = np.stack([
                c.pi_target if c.pi_target is not None else np.full(11, np.nan, np.float32)
                for c in corrections]).astype(np.float32)
        np.savez(result_path + ".npz", **arrays)
    with open(result_path + ".json", "w") as f:
        json.dump({"scalars": [c.as_record() for c in corrections],
                   "status": status, "n_candidates": len(cands)}, f)


if __name__ == "__main__":
    run(sys.argv[1])
