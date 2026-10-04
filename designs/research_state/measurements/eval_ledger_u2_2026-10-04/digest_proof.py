"""THE U2 DIGEST PROOF — the in-loop eval cycle plays byte-identical games before and after eval unit U2.

U2 moves the trainer's in-loop eval cycle (``eval_launch.launch_rust_eval_cycle``) onto the eval COUNT ledger. It
must be STORAGE ONLY (design_evaluation.md §0c rule 6, the X5 constraint M3): the same cycle on the same seed plays
the same games. This script plays a fixed, seeded in-loop cycle on the REAL Rust eval core and prints what a
before/after comparison needs. It is self-contained on purpose — it runs unchanged at the commit BEFORE U2
(``e5f393cb``) and at U2's, and ``compare`` diffs the two outputs.

THE CYCLE. Seeded PERTURBED fresh production-arch policies (``rust_eval.parity.build_models``: a trainee and two pool
sentinels; ``build_fixed_models``: one fixed opponent on a pinned team), CPU eager, 4 envs; opponents = 3 roster
bots + the 2 sentinels + the fixed opponent, 4 games each; one UNMIRRORED cycle with greedy sentinels and one
MIRRORED cycle with sentinels sampled at T = 1.0 (both opponent regimes, both pairing rules). The cycle seed is the
in-loop rule (``rust_eval.launch.cycle_seed``) on run seed 7.

WHAT IT RECORDS, per cycle:
* ``inloop`` — the cycle through ``launch_rust_eval_cycle`` exactly as the callbacks call it (at U2: with a
  ``CycleLedger`` on a scratch root and the trainee zip as the snapshot), then the merged shard results the collect
  reads (W, finished, draws, pentanomial per opponent); at U2 also every ledger row's counts and outcome digest;
* ``full_log`` — the same plan replayed through ``run_rust_eval_cycle`` with the executor's FULL ``game_log``: the
  per-game outcome vector ``(game, W/L/D, end turn)`` per opponent and its §0b.2 outcome digest at the near-tie
  margin 2e-3 (both sides' decisions; the rule ``cycle_ledger`` applies) and over every game.

THE PROOF (``compare before.json after.json``): (1) the merged shard results of the in-loop path are identical
before and after; (2) the full per-game outcome vectors are identical before and after; (3) after U2, every ledger
row's outcome digest equals the full-log digest of its opponent, and its W / L / D equal the outcome vector's.

    PYTHONPATH=src python designs/research_state/measurements/eval_ledger_u2_2026-10-04/digest_proof.py play OUT.json
    python designs/research_state/measurements/eval_ledger_u2_2026-10-04/digest_proof.py compare BEFORE.json AFTER.json
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

RUN_SEED = 7
STEP = 4000
GAMES = 4
N_ENVS = 4
DIGEST_MARGIN = 2e-3
BOTS = 3


def _digest_of(games: List[Dict[str, Any]]) -> Dict[str, Any]:
    from agents.training import eval_ledger as L

    letter = {"WIN": "W", "LOSS": "L", "DRAW": "D"}
    vec = [(int(g["game"]), letter[g["result"]], int(g["end_turn"])) for g in sorted(games, key=lambda g: g["game"])]
    near = sorted(int(g["game"]) for g in games
                  if any(float(m) < DIGEST_MARGIN for m in list(g.get("margins", ())) + [o[3] for o in g.get("opp", ())]))
    return {"outcomes": vec, "near_tie_games": near, "digest": L.outcome_digest(vec, near),
            "digest_all": L.outcome_digest(vec)}


def build_host(workdir: Path) -> Dict[str, Any]:
    """The models, the T2 service and the eval core — the trainer's startup, reduced (``parity._run_rust``)."""
    import torch

    from agents.inference.service import InferenceService, ServiceSpec, SlotGroupSpec
    from agents.model.snapshot import (arch_toggles_from_model, current_model_version, historical_load_kwargs,
                                       load_checkpoint_strict, load_foreign_opponent)
    from agents.observation.state_encoder import load_mappings
    from agents.training.fixed_opponent_pool import FixedOpponentEntry
    from agents.training.reward_config import RewardConfig
    from agents.training.rust_eval import parity as PAR
    from agents.training.rust_eval.build import EvalDecl, build_eval_core, eval_builders, eval_extra_slots
    from agents.training.rust_rollout.build import RustEnvDecl
    from utils.rust_env import episode as EP

    trainee, sentinels, mcfg = PAR.build_models(workdir / "models", n_sentinels=2)
    fixed = PAR.build_fixed_models(workdir / "fixed", [{"pins": 1}], mcfg)
    torch.set_num_threads(PAR.RUST_THREADS)
    model = load_checkpoint_strict(trainee, device="cpu", **historical_load_kwargs(trainee))
    model.policy.eval()
    version = current_model_version(load_mappings(), **arch_toggles_from_model(model))
    fixed_pol = {f["label"]: load_foreign_opponent(f["path"], current_version=version, device="cpu",
                                                   config_path=f["config_path"])[0].policy.eval() for f in fixed}
    decl = EvalDecl(n_envs=N_ENVS, n_sentinels=len(sentinels), fixed_labels=tuple(f["label"] for f in fixed))
    extra = eval_extra_slots(decl, model.policy, fixed_pol)
    pols = [pol for _fam, pol in extra]
    svc = InferenceService(ServiceSpec(groups=(SlotGroupSpec("eval", len(pols), pols[0]),), device="cpu",
                                       backend="eager", buckets=(8, 48), lanes=1,
                                       max_rows_per_flush=max(1024, len(pols) * 48, 4 * N_ENVS))).startup()
    cdecl = RustEnvDecl(n_envs=N_ENVS, threads=4, front="ffi", profile="selfcheck")
    entries = [FixedOpponentEntry(label=f["label"], zip_path=f["path"], config_path=f["config_path"], arch_signature="",
                                  team_str=f["team_strs"][0] if f["team_strs"] else None,
                                  team_strs=tuple(f["team_strs"])) for f in fixed]
    tb, flat, fixed_b = eval_builders(None, entries)
    terminal = EP.terminal_from_reward_config(RewardConfig.from_dict(json.loads(Path(mcfg).read_text())))
    ev = build_eval_core(decl, collector_decl=cdecl, svc=svc, extra_ids=list(range(len(pols))), trainee_builder=tb,
                         opp_builder=flat, fixed_builders=fixed_b, turn_limit=EP.stall_threshold(), terminal=terminal,
                         fixed_policies=fixed_pol, emit=lambda _m: None)
    model._rust_collector = SimpleNamespace(evaluator=ev, cfg=SimpleNamespace(run_seed=RUN_SEED))
    return {"model": model, "ev": ev, "trainee": trainee, "sentinels": sentinels, "fixed": fixed}


def plan_items(host: Dict[str, Any], *, games: int = GAMES, bots: int = BOTS, sentinels: int = 2,
               fixed: bool = True) -> List[Any]:
    from agents.training.eval_callback import eval_opponent_names
    from agents.training.eval_sharding import BOT, SENTINEL, EvalItem

    items = [EvalItem(b, BOT, games) for b in list(eval_opponent_names())[:bots]]
    items += [EvalItem(f"sentinel_{i}", SENTINEL, games, path=p, step=(i + 1) * 1000)
              for i, p in enumerate(host["sentinels"][:sentinels])]
    items += [EvalItem.fixed_from_cfg({"label": f["label"], "path": f["path"], "config_path": f["config_path"],
                                       "team_str": f["team_strs"][0] if f["team_strs"] else None,
                                       "team_strs": list(f["team_strs"])}, games) for f in (host["fixed"] if fixed else [])]
    return items


def cb_for(host: Dict[str, Any], *, sentinel_greedy: bool, ledger: Any = None) -> Any:
    from agents.training.eval_player import ForensicQuota

    return SimpleNamespace(model=host["model"], _model_dir=None, _forensic_quota=ForensicQuota(0, 0, 0),
                           _eval_sentinel_greedy=sentinel_greedy, _self_play_temp=1.0, safe_point_fn=None,
                           logger=None, _cycle_ledger=ledger)


def play_cycle(host: Dict[str, Any], workdir: Path, *, mirrored: bool, sentinel_greedy: bool,
               ledger_root: Any = None, full_log: bool = True, plan: Any = None) -> Dict[str, Any]:
    """One cycle (module docs). ``plan`` = ``plan_items`` keyword overrides (a smaller plan for a test);
    ``full_log=False`` skips the full-log replay (the in-loop half only)."""
    import inspect

    from agents.training import eval_launch
    from agents.training.eval_collect import merge_eval_results
    from agents.training.eval_sharding import ShardedEvalPool
    from agents.training.rust_eval.launch import run_rust_eval_cycle

    items = plan_items(host, **dict(plan or {}))
    keys = [it.key for it in items]
    out: Dict[str, Any] = {"mirrored": mirrored, "sentinel_greedy": sentinel_greedy}

    # (1) the in-loop path, as the callbacks call it
    led = None
    has_ledger = "snapshot" in inspect.signature(eval_launch.launch_rust_eval_cycle).parameters
    if has_ledger and ledger_root is not None:
        from agents.training import eval_ledger as L
        from agents.training.cycle_ledger import CycleLedger

        led = CycleLedger(None, run_label="u2_digest_proof", commit="proof",
                          writer=L.LedgerWriter(ledger_root, producer="inloop"))
    run_dir = workdir / f"inloop_{int(mirrored)}"
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    pool = ShardedEvalPool(items, 2, step=STEP, mirrored=mirrored)
    pool.write_plan(str(run_dir))
    cb = cb_for(host, sentinel_greedy=sentinel_greedy, ledger=led)
    if has_ledger:
        eval_launch.launch_rust_eval_cycle(cb, pool, str(run_dir), STEP, snapshot=host["trainee"])
    else:
        eval_launch.launch_rust_eval_cycle(cb, pool, str(run_dir), STEP)
    merged, missing = merge_eval_results(str(run_dir), keys)
    if missing:
        raise RuntimeError(f"the in-loop cycle published nothing for {missing}")
    out["merged"] = {k: {"counts": list(merged["counts"][k]), "draws": merged["draws"].get(k),
                         "pairs": merged["pairs"].get(k)} for k in keys}
    if led is not None:
        from agents.training import eval_ledger as L
        from agents.training.cycle_ledger import PROTOCOL

        decl = L.ReaderDecl(name="u2_digest_proof", purposes=frozenset({"cycle"}),
                            regime=L.RegimeFilter(protocol=PROTOCOL), requests="any", selection="include",
                            flags_ok=frozenset(), inference="conditional")
        rows = [r for rd in L.read_by_regime(decl, root=ledger_root).values() for r in rd.rows
                if r["seed"]["cycle_seed"] == merged_seed(host)]
        out["ledger"] = {r["seed"]["item_key"]: {"counts": r["counts"], "digest": r["compute"]["outcome_digest"],
                                                 "digest_all": r["compute"].get("outcome_digest_all"),
                                                 "near_tie_games": r["compute"]["near_tie_games"],
                                                 "pairs": (r["pairs"] or {}).get("counts"),
                                                 "mirrored": r["regime"]["mirrored"]}
                         for r in rows if r["regime"]["mirrored"] == mirrored}
        led.close()

    if not full_log:
        return out
    # (2) the full game log of the same plan, same seed
    run_dir2 = workdir / f"full_{int(mirrored)}"
    shutil.rmtree(run_dir2, ignore_errors=True)
    run_dir2.mkdir(parents=True)
    pool2 = ShardedEvalPool(items, 2, step=STEP, mirrored=mirrored)
    pool2.write_plan(str(run_dir2))
    full: List[Dict[str, Any]] = []
    run_rust_eval_cycle(cb_for(host, sentinel_greedy=sentinel_greedy), pool=pool2, run_dir=str(run_dir2), step=STEP,
                        game_log=full, forensic=False, record=False)
    out["full_log"] = {k: _digest_of([g for g in full if g["item"] == k]) for k in keys}
    return out


def merged_seed(host: Dict[str, Any]) -> int:
    from agents.training.rust_eval.launch import cycle_seed

    return int(cycle_seed(RUN_SEED, STEP))


def play(out_path: str) -> None:
    from agents.training.rust_rollout.testkit import build_selfcheck
    from utils.git import get_git_hash

    build_selfcheck()
    work = Path(tempfile.mkdtemp(prefix="u2_digest_"))
    try:
        host = build_host(work)
        res = {"commit": get_git_hash(), "cycle_seed": merged_seed(host),
               "cycles": [play_cycle(host, work, mirrored=False, sentinel_greedy=True, ledger_root=work / "ledger"),
                          play_cycle(host, work, mirrored=True, sentinel_greedy=False, ledger_root=work / "ledger")]}
        host["ev"].close()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    Path(out_path).write_text(json.dumps(res, indent=1, sort_keys=True))
    print(f"wrote {out_path} at {res['commit']}")


def compare(before_path: str, after_path: str) -> int:
    a, b = json.loads(Path(before_path).read_text()), json.loads(Path(after_path).read_text())
    bad: List[str] = []
    if a["cycle_seed"] != b["cycle_seed"]:
        bad.append("the cycle seeds differ")
    n_games = n_rows = 0
    for ca, cb_ in zip(a["cycles"], b["cycles"]):
        tag = f"mirrored={ca['mirrored']}"
        if ca["merged"] != cb_["merged"]:
            bad.append(f"{tag}: the in-loop merged shard results differ")
        for k, fa in ca["full_log"].items():
            fb = cb_["full_log"][k]
            n_games += len(fa["outcomes"])
            if fa["outcomes"] != fb["outcomes"]:
                bad.append(f"{tag} {k}: the per-game outcome vectors differ")
            if fa["digest"] != fb["digest"]:
                bad.append(f"{tag} {k}: the full-log digests differ")
        led = cb_.get("ledger")
        if led is None:
            bad.append(f"{tag}: the AFTER run wrote no ledger rows")
            continue
        if sorted(led) != sorted(cb_["full_log"]):
            bad.append(f"{tag}: ledger rows for {sorted(led)} vs opponents {sorted(cb_['full_log'])}")
        for k, row in led.items():
            n_rows += 1
            fl = cb_["full_log"][k]
            w = sum(1 for _g, r, _t in fl["outcomes"] if r == "W")
            l_ = sum(1 for _g, r, _t in fl["outcomes"] if r == "L")
            d = sum(1 for _g, r, _t in fl["outcomes"] if r == "D")
            if row["digest"] != fl["digest"] or row["near_tie_games"] != fl["near_tie_games"]:
                bad.append(f"{tag} {k}: the ledger row's digest is not the full log's")
            if row.get("digest_all") is not None and row["digest_all"] != fl.get("digest_all"):
                bad.append(f"{tag} {k}: the ledger row's all-games digest is not the full log's")
            if (row["counts"]["w"], row["counts"]["l"], row["counts"]["d"]) != (w, l_, d):
                bad.append(f"{tag} {k}: the ledger row's W/L/D {row['counts']} vs the log's {(w, l_, d)}")
            if [row["counts"]["w"], row["counts"]["w"] + row["counts"]["l"] + row["counts"]["d"]] != \
                    cb_["merged"][k]["counts"]:
                bad.append(f"{tag} {k}: the ledger row disagrees with the merged shard results")
    print(f"before {a['commit'][:10]} vs after {b['commit'][:10]}: {n_games} games per side compared, "
          f"{n_rows} ledger rows checked")
    for x in bad:
        print("  MISMATCH:", x)
    print("IDENTICAL — storage only" if not bad else f"{len(bad)} mismatch(es)")
    return 0 if not bad else 1


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "play":
        play(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    else:
        print(__doc__)
        sys.exit(2)
