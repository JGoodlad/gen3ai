"""THE OFFLINE RUST EVAL HARNESS — a cycle plan played on the Rust eval core OUTSIDE a trainer.

The trainer plays its eval cycle on a declared eval core and the T2 slots of its own inference service
(``rust_eval.launch``). An offline caller has no trainer, so this module DECLARES the same pieces itself —
one T2 slot group (the trainee's architecture: the trainee's eval slot, one slot per sentinel, the fixed
opponents' slots), one eval core (``build_eval_core``), the eval builders — and plays one cycle through the
production executor (``RustEvalCore.run_cycle``). Every resource is acquired before the cycle and nothing
after it (the declared lifecycle).

Callers: ``main.ops.eval_trace_gen`` (an eval-traces cycle generated for a saved checkpoint) and the tests
that need a real cycle or a seeded checkpoint (the safe-point gate, the head-to-head fixtures, the
poke-env-free entry points). Two helpers build those checkpoints:

* :func:`build_models` — seeded PERTURBED fresh production-architecture policies, built at one thread
  (``build_fresh_model`` + ``parity_probe.perturb_``), so a greedy decision is not the zero-init pointer
  head's uniform tie;
* :func:`build_fixed_models` — the same for FIXED (``ext_``) opponents, seeds disjoint from the above.

(Re-homed from ``rust_eval.parity`` when the Python EVAL oracle it compared against — ``main.eval_worker`` —
was deleted, poke-env retirement P6 slice 6c. The comparison half went with it.)
"""
from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

#: A flip below ``NEAR_TIE_FACTOR x`` a tier's |Δ legal log-prob| bar is a TIE: 2 is the exact bound (each
#: of the two compared log-probs moves by at most the bar).
NEAR_TIE_FACTOR = 2.0
#: The process-global torch state a harness region declares (the values a fresh process has).
DECLARED_MATMUL_PRECISION = "highest"
#: The |Δ legal log-prob| bars measured by the Lane H gate (Lane E's): eager CPU vs eager CPU, and T2 ``graph``
#: on CUDA vs an eager CPU forward. The near-tie constants of ``main.h2h.play`` / ``cycle_ledger`` are 2x these.
BAR_CPU = 1e-5
BAR_GPU = 1e-3
#: Torch threads of the harness's forward (declared, F-LJ-6).
RUST_THREADS = 4


class GateStateError(RuntimeError):
    """A process-global torch setting the harness depends on is not what it declares."""


@contextlib.contextmanager
def declared_torch_state(threads: int) -> Iterator[None]:
    """Pin the thread count for the block, refuse an undeclared global, restore what was found."""
    import torch

    def check(where: str) -> None:
        bad = {}
        if torch.get_float32_matmul_precision() != DECLARED_MATMUL_PRECISION:
            bad["float32_matmul_precision"] = torch.get_float32_matmul_precision()
        if torch.get_default_dtype() != torch.float32:
            bad["default_dtype"] = str(torch.get_default_dtype())
        if torch.get_num_threads() != int(threads):
            bad["num_threads"] = torch.get_num_threads()
        if bad:
            raise GateStateError(f"{where}: torch state {bad} is not the declared one (precision "
                                 f"{DECLARED_MATMUL_PRECISION!r}, float32, {threads} threads) — a test or "
                                 "import left it changed; fix it at the source")

    found = torch.get_num_threads()
    torch.set_num_threads(int(threads))
    try:
        check("entering the harness region")
        yield
        check("leaving the harness region")
    finally:
        torch.set_num_threads(found)


def build_models(dst: Path, *, n_sentinels: int, trainee: Optional[str] = None,
                 sentinels: Optional[List[str]] = None) -> Tuple[str, List[str], str]:
    """``(trainee zip, [sentinel zips], model_config.json)``. Default: seeded PERTURBED fresh
    production-arch policies (``build_fresh_model``, built at 1 thread — F-LJ-6); else the given files."""
    dst.mkdir(parents=True, exist_ok=True)
    cfg = dst / "model_config.json"
    if trainee is not None:
        if not cfg.exists():
            src = Path(trainee).parent
            for cand in (src / "model_config.json", src.parent / "model_config.json"):
                if cand.exists():
                    cfg.write_text(cand.read_text())
                    break
        return trainee, list(sentinels or []), str(cfg)
    from agents.model.parity_probe import PERTURB_SCALE, perturb_
    from agents.model.snapshot import arch_toggles_from_model, current_model_version
    from agents.observation.state_encoder import load_mappings
    from main.fresh_checkpoint import build_fresh_model

    with declared_torch_state(1):
        out = []
        for i in range(n_sentinels + 1):
            model, _, _ = build_fresh_model(i)
            perturb_(model.policy, seed=2000 + i, scale=PERTURB_SCALE)
            p = dst / f"snapshot_{(i + 1) * 1000:012d}.zip"
            model.save(str(p))
            out.append(str(p))
    cfg.write_text(current_model_version(load_mappings(), **arch_toggles_from_model(model)).to_json())
    return out[-1], out[:-1], str(cfg)


def build_fixed_models(dst: Path, specs: List[Dict[str, Any]], config_path: str) -> List[Dict[str, Any]]:
    """The FIXED opponents: one seeded PERTURBED fresh policy each (seeds disjoint from :func:`build_models`'),
    labelled ``ext_fixed{i}``, pinned to ``pins`` sample teams (0 = the pool); ``reused`` = its slot is the
    TRAINING plan's stable slot rather than an eval slot."""
    if not specs:
        return []
    from agents.model.parity_probe import PERTURB_SCALE, perturb_
    from main.fresh_checkpoint import build_fresh_model
    from utils.team_loader import TeamLoader

    sample = TeamLoader().get_sample_teams()
    dst.mkdir(parents=True, exist_ok=True)
    out = []
    with declared_torch_state(1):
        for i, sp in enumerate(specs):
            model, _, _ = build_fresh_model(100 + i)
            perturb_(model.policy, seed=3000 + i, scale=PERTURB_SCALE)
            path = dst / f"fixed_{i}.zip"
            model.save(str(path))
            pins = [sample[(7 + 11 * i + 5 * k) % len(sample)] for k in range(int(sp.get("pins", 0)))]
            out.append({"label": f"ext_fixed{i}", "path": str(path), "config_path": config_path,
                        "team_strs": pins, "reused": bool(sp.get("reused", False))})
    return out


def plan_items(bots: List[str], sentinels: List[str], games: int, fixed: Optional[List[Dict[str, Any]]] = None,
               sentinel_steps: Optional[List[int]] = None) -> List[Any]:
    """The cycle's ``EvalItem`` list: the bots, ``sentinel_<i>`` per sentinel zip (its step from
    ``sentinel_steps``, else ``(i + 1) * 1000`` — :func:`build_models`' naming), then the fixed opponents (the
    callbacks' own construction, from the entry's ``to_cfg()`` shape)."""
    from agents.training.eval_sharding import BOT, SENTINEL, EvalItem

    steps = list(sentinel_steps) if sentinel_steps is not None else [(i + 1) * 1000 for i in range(len(sentinels))]
    items = [EvalItem(b, BOT, games) for b in bots]
    items += [EvalItem(f"sentinel_{i}", SENTINEL, games, path=p, step=int(steps[i])) for i, p in enumerate(sentinels)]
    items += [EvalItem.fixed_from_cfg({"label": f["label"], "path": f["path"], "config_path": f["config_path"],
                                       "team_str": f["team_strs"][0] if f["team_strs"] else None,
                                       "team_strs": list(f["team_strs"])}, games) for f in (fixed or [])]
    return items


def run_rust(*, run_dir: Path, model_dir: Path, trainee: str, sentinels: List[str], items: List[Any],
             shard_games: int, step: int, cycle_seed: int, quota: Dict[str, int], device: str, backend: str,
             n_envs: int, buckets: Tuple[int, ...] = (8, 48), front: str = "ffi", profile: str = "selfcheck",
             lanes: int = 0, fixed: Optional[List[Dict[str, Any]]] = None, sentinel_greedy: bool = True,
             self_play_temp: float = 1.0, safe_point: Optional[Callable[[str], None]] = None,
             trainee_team_str: Any = None, mirrored: bool = False, core_threads: int = 4,
             torch_threads: int = RUST_THREADS, forensic_root: Optional[Path] = None, keep_games: bool = True,
             oracle_reveal: str = "off", emit: Callable[[str], None] = lambda _m: None) -> Dict[str, Any]:
    """Play the plan ``items`` once on the Rust eval core (module docs) and return the cycle.

    The plan is written to ``run_dir`` and each unit's ``ShardResult`` published there (``ShardedEvalPool.collect``
    reads them back); the kept traces land under ``forensic_root`` (default ``<model_dir>/eval_traces/step_<step>``).
    The core's TERMINAL is the one ``<model_dir>/model_config.json`` declares. ``trainee_team_str`` = the run's
    trainee team pin (the eval trainee builder, ``eval_teams.build_trainee_tb``); ``mirrored`` = the mirrored-pair
    regime (``gen3_mirrored_pairs_v1``); ``oracle_reveal`` = the run's recorded observation mode (both seats, as the
in-loop eval plays it). ``safe_point`` is passed to ``run_cycle`` (called at the top of every host
    step; it may raise to abandon the cycle — the core is closed either way). ``keep_games=False`` keeps no per-game
log (a row carries the game's input log, tens of KB: a large cycle's would hold every one in memory). Returns
    ``{stats, games, svc_startup_s, arch_toggles, gamma, slot_model_ids, fixed_slots}``."""
    import torch

    # `MaskablePPO.load` re-seeds the loaded model (sb3's `set_random_seed`), which on CUDA sets
    # `torch.backends.cudnn.deterministic` — a process global the harness must hand back (the root
    # conftest's torch global-state guard, `1afd2590`).
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        return _run_rust(run_dir=run_dir, model_dir=model_dir, trainee=trainee, sentinels=sentinels, items=items,
                         shard_games=shard_games, step=step, cycle_seed=cycle_seed, quota=quota, device=device,
                         backend=backend, n_envs=n_envs, buckets=buckets, front=front, profile=profile, lanes=lanes,
                         fixed=list(fixed or []), sentinel_greedy=sentinel_greedy, self_play_temp=self_play_temp,
                         safe_point=safe_point, trainee_team_str=trainee_team_str, mirrored=mirrored,
                         core_threads=core_threads, torch_threads=torch_threads, forensic_root=forensic_root,
                         keep_games=keep_games, oracle_reveal=oracle_reveal, emit=emit)
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn


def _run_rust(*, run_dir: Path, model_dir: Path, trainee: str, sentinels: List[str], items: List[Any],
              shard_games: int, step: int, cycle_seed: int, quota: Dict[str, int], device: str, backend: str,
              n_envs: int, buckets: Tuple[int, ...], front: str, profile: str, lanes: int,
              fixed: List[Dict[str, Any]], sentinel_greedy: bool, self_play_temp: float,
              safe_point: Optional[Callable[[str], None]], trainee_team_str: Any, mirrored: bool,
              core_threads: int, torch_threads: int, forensic_root: Optional[Path], keep_games: bool,
              oracle_reveal: str, emit: Callable[[str], None]) -> Dict[str, Any]:
    from agents.inference.service import InferenceService, ServiceSpec, SlotGroupSpec
    from agents.model.snapshot import (arch_toggles_from_model, current_model_version, historical_load_kwargs,
                                       load_checkpoint_strict, load_foreign_opponent)
    from agents.observation.state_encoder import load_mappings
    from agents.training.eval_quota import ForensicQuota
    from agents.training.eval_sharding import ShardedEvalPool
    from agents.training.fixed_opponent_pool import FixedOpponentEntry
    from agents.training.reward_config import RewardConfig
    from agents.training.rust_eval.build import EvalDecl, build_eval_core, eval_builders, eval_extra_slots
    from agents.training.rust_eval.launch import load_sentinels
    from agents.training.rust_rollout.build import RustEnvDecl, _arch_key
    from utils.rust_env import episode as EP

    with declared_torch_state(torch_threads):
        t0 = time.perf_counter()
        # an ARCHIVED trainee pickles the policy kwargs deletion pass L1 removed (`use_popart` ...)
        model = load_checkpoint_strict(trainee, device=device, **historical_load_kwargs(trainee))
        model.policy.eval()
        # FIXED opponents as `rust_env_setup` loads and declares them
        version = current_model_version(load_mappings(), **arch_toggles_from_model(model)) if fixed else None
        # on the CPU, as `rust_env_setup` loads them: a fixed opponent is only a weight source for its slots
        fixed_pol = {f["label"]: load_foreign_opponent(f["path"], current_version=version, device="cpu",
                                                       config_path=f["config_path"])[0].policy.eval() for f in fixed}
        reused_labels = [f["label"] for f in fixed if f["reused"]]
        decl = EvalDecl(n_envs=n_envs, n_sentinels=len(sentinels), fixed_labels=tuple(f["label"] for f in fixed),
                        reused_fixed=tuple((lab, i) for i, lab in enumerate(reused_labels)))
        # Slots: the TRAINING plan's stable slots first (a reused label's), then the eval extras in
        # `eval_extra_slots`' order — one group, its template the FIRST slot's policy (T2 starts every
        # slot with the template's weights; `slot_groups_with_extra` joins same-architecture runs).
        extra = eval_extra_slots(decl, model.policy, fixed_pol)
        pols = [fixed_pol[lab] for lab in reused_labels] + [pol for _fam, pol in extra]
        keys = {_arch_key(p) for p in pols}
        if len(keys) != 1:
            raise RuntimeError("the offline Rust eval harness serves ONE slot group: every fixed opponent must share "
                               "the trainee's architecture")
        n_slots = len(pols)
        ln = lanes or (min(n_slots, 8) if device.startswith("cuda") else 1)
        svc = InferenceService(ServiceSpec(groups=(SlotGroupSpec("eval", n_slots, pols[0]),), device=device,
                                           backend=backend, buckets=tuple(buckets), lanes=ln,
                                           max_rows_per_flush=max(1024, n_slots * max(buckets), 4 * n_envs))).startup()
        for i, lab in enumerate(reused_labels):        # the training host's own load of its stable slot
            svc.load(i, fixed_pol[lab], f"stable:{lab}")
        t_svc = time.perf_counter() - t0
        cdecl = RustEnvDecl(n_envs=n_envs, threads=int(core_threads), front=front, profile=profile,
                            oracle_reveal=oracle_reveal)
        entries = [FixedOpponentEntry(label=f["label"], zip_path=f["path"], config_path=f["config_path"],
                                      arch_signature="", team_str=(f["team_strs"][0] if f["team_strs"] else None),
                                      team_strs=tuple(f["team_strs"])) for f in fixed]
        tb, flat, fixed_b = eval_builders(trainee_team_str, entries)
        # The core's terminal from the run's model_config.json (`RewardConfig.from_dict`).
        terminal = EP.terminal_from_reward_config(
            RewardConfig.from_dict(json.loads((model_dir / "model_config.json").read_text())))
        ev = build_eval_core(decl, collector_decl=cdecl, svc=svc, extra_ids=list(range(len(reused_labels), n_slots)),
                             trainee_builder=tb, opp_builder=flat, fixed_builders=fixed_b,
                             turn_limit=EP.stall_threshold(), terminal=terminal, fixed_policies=fixed_pol, emit=emit)
        try:
            slot_ids = [mid for g in svc.groups for mid in g.model_ids]
            pool = ShardedEvalPool(items, shard_games, step=step, mirrored=bool(mirrored))
            run_dir.mkdir(parents=True, exist_ok=True)
            pool.write_plan(str(run_dir))

            sent = load_sentinels(pool, model)
            glog: Optional[List[Dict[str, Any]]] = [] if keep_games else None
            root = forensic_root if forensic_root is not None else model_dir / "eval_traces" / f"step_{step}"
            st = ev.run_cycle(pool, str(run_dir), step=step, trainee_policy=model.policy, sentinel_policies=sent,
                              forensic_root=str(root), quota=ForensicQuota(**quota), gamma=float(model.gamma),
                              sentinel_greedy=bool(sentinel_greedy), self_play_temp=float(self_play_temp),
                              cycle_seed=cycle_seed, game_log=glog, safe_point=safe_point)
        finally:
            ev.close()
        toggles = arch_toggles_from_model(model)
        return {"stats": st.as_dict(), "games": glog if glog is not None else [], "svc_startup_s": t_svc,
                "arch_toggles": toggles,
                "gamma": float(model.gamma), "slot_model_ids": slot_ids,
                "fixed_slots": [list(x) for x in ev.table.fixed_slots]}
