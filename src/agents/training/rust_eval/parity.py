"""THE LANE H GATE — the same seed set played on BOTH eval paths (M5 Lane H).

One cycle PLAN (``ShardedEvalPool``: roster bots + pool sentinels, split into shard units) and ONE
cycle seed are played twice:

* the RUST path — ``RustEvalCore.run_cycle`` on the eval core with the trainee and the sentinels in T2
  slots (the production executor, in this process);
* the PYTHON path — today's eval worker (``python -m main.eval_worker``, a CUDA-less child) on the
  in-process rust bridge, with the per-GAME seed rule (``seed_rule = "per_game"``): each game's teams,
  battle seed and bot streams from ``rust_eval.seeds`` — the table the Rust path reads.

Neither path reads the other's actions. Compared:

1. PER GAME: winner, end turn and EVERY trainee action. A first differing action is judged by the
   DECLARED tie rule (Lane E's ``judge_flips``): a TIE iff both sides' top-2 legal log-prob margins at
   that decision are below ``NEAR_TIE_FACTOR x`` the tier's |Δ log-prob| bar — counted, reported, and
   the game (which then legitimately diverges) left out of the equality; any other difference FATAL.
2. THE METRICS: both cycles' shard results pooled by the unchanged collect (``ShardedEvalPool.collect``)
   — win rates, reward means, episode lengths, exact W/L counts, draws and the trace-selection tuples
   EQUAL; the critic-residual tails within ``TD_BAR`` (the two forwards differ by float rounding).
3. THE TRACES: the same trace files kept on both sides (same games, same names); each Rust core trace
   EXPANDS in the prober (``main.prober.core_trace``) to the SAME decisions as the Python
   trace of that game (turn, phase, chosen action, both actives, per decision).

Bars (the tier's |Δ legal log-prob|, Lane E's measured ones): eager CPU vs eager CPU 1e-5; T2 graph
on CUDA vs the Python path on CPU 1e-3.

Two optional plan features (F-LH-10's rows):

* FIXED opponents (``cfg["fixed"]``: one dict per opponent, ``pins`` = how many sample teams it pilots,
  ``reused`` = its slot is the TRAINING plan's stable slot rather than an eval slot). Built as
  perturbed-fresh policies of their own; the Rust path declares them exactly as ``rust_env_setup`` does
  (``EvalDecl.fixed_labels`` / ``reused_fixed``, ``eval_extra_slots``, one slot group whose template is
  the first slot's policy, so a non-reused fixed slot STARTS with another policy's weights and only
  ``build_eval_core``'s load makes it the opponent), loads them with ``load_foreign_opponent`` and
  pins their teams through ``eval_builders``; the Python worker plays them as ``FIXED`` plan items.
* The SAMPLED sentinel regime (``cfg["sentinel_greedy"] = False``, i.e. ``--no-eval-sentinel-greedy``):
  the Rust sentinel draws the KEYED draw keyed by the game; the Python worker's sentinel draws the same
  keyed draw (``eval_worker.install_keyed_sampler``, per_game mode only), so the regime is compared
  GAME FOR GAME: every sentinel decision's ``[decision index, action]`` equal on both paths as well as
  the trainee's. A first differing sentinel draw is a TIE iff its CDF margin is below
  ``NEAR_TIE_FACTOR x`` the bar on both sides (Lane E's keyed rule).

The Python worker (per_game mode) logs EVERY policy opponent's decisions (sentinel, fixed; greedy or
keyed), so an opponent's own near-tie flip is judged where it happens rather than surfacing, turns later,
as an unexplained trainee difference (found by the fixed row's milestone: a greedy fixed opponent whose
top-2 log-probs were exactly equal on the Rust path, 2.4e-7 apart on the Python one).
"""
from __future__ import annotations

import json
import contextlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple


#: A flip below ``NEAR_TIE_FACTOR x`` the tier's |Δ legal log-prob| bar is a TIE: 2 is the exact bound (each
#: of the two compared log-probs moves by at most the bar). (Re-homed with `declared_torch_state` from the
#: deleted `rust_env_opponents_parity` — deletion pass U3 / R10.)
NEAR_TIE_FACTOR = 2.0
#: The process-global torch state the gate declares (the values a fresh process has).
DECLARED_MATMUL_PRECISION = "highest"


class GateStateError(RuntimeError):
    """A process-global torch setting the gate depends on is not what it declares."""


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
        check("entering the gate region")
        yield
        check("leaving the gate region")
    finally:
        torch.set_num_threads(found)

#: The critic-residual tail tolerance (V through T2 vs the Python forward).
TD_BAR = 1e-4
BAR_CPU = 1e-5
BAR_GPU = 1e-3
#: Torch threads of the Rust path's forward (declared, F-LJ-6).
RUST_THREADS = 4


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
    from agents.model.snapshot import current_model_version
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
    from agents.model.snapshot import arch_toggles_from_model

    cfg.write_text(current_model_version(load_mappings(), **arch_toggles_from_model(model)).to_json())
    return out[-1], out[:-1], str(cfg)


def build_fixed_models(dst: Path, specs: List[Dict[str, Any]], config_path: str) -> List[Dict[str, Any]]:
    """The FIXED opponents (module docs): one seeded PERTURBED fresh policy each (seeds disjoint from
    ``build_models``'), labelled ``ext_fixed{i}``, pinned to ``pins`` sample teams (0 = the pool)."""
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


def _items(bots: List[str], sentinels: List[str], games: int, fixed: Optional[List[Dict[str, Any]]] = None
           ) -> List[Any]:
    from agents.training.eval_sharding import BOT, SENTINEL, EvalItem

    items = [EvalItem(b, BOT, games) for b in bots]
    items += [EvalItem(f"sentinel_{i}", SENTINEL, games, path=p, step=(i + 1) * 1000) for i, p in enumerate(sentinels)]
    # the callbacks' own construction, from the entry's `to_cfg()` shape
    items += [EvalItem.fixed_from_cfg({"label": f["label"], "path": f["path"], "config_path": f["config_path"],
                                       "team_str": f["team_strs"][0] if f["team_strs"] else None,
                                       "team_strs": list(f["team_strs"])}, games) for f in (fixed or [])]
    return items


def run_rust(*, run_dir: Path, model_dir: Path, trainee: str, sentinels: List[str], items: List[Any],
             shard_games: int, step: int, cycle_seed: int, quota: Dict[str, int], device: str, backend: str,
             n_envs: int, buckets: Tuple[int, ...] = (8, 48), front: str = "ffi", profile: str = "selfcheck",
             lanes: int = 0, fixed: Optional[List[Dict[str, Any]]] = None, sentinel_greedy: bool = True,
             self_play_temp: float = 1.0, safe_point: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """The Rust path (module docs) — the production executor over a T2 service declared here. ``safe_point``
    is passed to ``run_cycle`` (the run's deferred-abort safe point; `safe_point_integration_test`)."""
    import torch

    # `MaskablePPO.load` re-seeds the loaded model (sb3's `set_random_seed`), which on CUDA sets
    # `torch.backends.cudnn.deterministic` — a process global the gate must hand back (the root
    # conftest's torch global-state guard, `1afd2590`).
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        return _run_rust(run_dir=run_dir, model_dir=model_dir, trainee=trainee, sentinels=sentinels, items=items,
                         shard_games=shard_games, step=step, cycle_seed=cycle_seed, quota=quota, device=device,
                         backend=backend, n_envs=n_envs, buckets=buckets, front=front, profile=profile, lanes=lanes,
                         fixed=list(fixed or []), sentinel_greedy=sentinel_greedy, self_play_temp=self_play_temp,
                         safe_point=safe_point)
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn


def _run_rust(*, run_dir: Path, model_dir: Path, trainee: str, sentinels: List[str], items: List[Any],
              shard_games: int, step: int, cycle_seed: int, quota: Dict[str, int], device: str, backend: str,
              n_envs: int, buckets: Tuple[int, ...], front: str, profile: str, lanes: int,
              fixed: List[Dict[str, Any]], sentinel_greedy: bool, self_play_temp: float,
              safe_point: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    from agents.inference.service import InferenceService, ServiceSpec, SlotGroupSpec
    from agents.model.snapshot import (arch_toggles_from_model, current_model_version, historical_load_kwargs,
                                       load_checkpoint_strict, load_foreign_opponent)
    from agents.observation.state_encoder import load_mappings
    from agents.training.eval_sharding import ShardedEvalPool
    from agents.training.fixed_opponent_pool import FixedOpponentEntry
    from agents.training.rust_eval.build import EvalDecl, build_eval_core, eval_builders, eval_extra_slots
    from agents.training.rust_eval.launch import load_sentinels
    from agents.training.rust_rollout.build import RustEnvDecl, _arch_key
    from utils.rust_env import episode as EP

    with declared_torch_state(RUST_THREADS):
        t0 = time.perf_counter()
        # an ARCHIVED trainee pickles the policy kwargs deletion pass L1 removed (`use_popart` ...)
        model = load_checkpoint_strict(trainee, device=device, **historical_load_kwargs(trainee))
        model.policy.eval()
        # FIXED opponents as `rust_env_setup` loads and declares them (module docs).
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
            raise RuntimeError("the Lane H harness serves ONE slot group: every fixed opponent must share the "
                               "trainee's architecture")
        n_slots = len(pols)
        ln = lanes or (min(n_slots, 8) if device.startswith("cuda") else 1)
        svc = InferenceService(ServiceSpec(groups=(SlotGroupSpec("eval", n_slots, pols[0]),), device=device,
                                           backend=backend, buckets=tuple(buckets), lanes=ln,
                                           max_rows_per_flush=max(1024, n_slots * max(buckets), 4 * n_envs))).startup()
        for i, lab in enumerate(reused_labels):        # the training host's own load of its stable slot
            svc.load(i, fixed_pol[lab], f"stable:{lab}")
        t_svc = time.perf_counter() - t0
        cdecl = RustEnvDecl(n_envs=n_envs, threads=4, front=front, profile=profile)
        entries = [FixedOpponentEntry(label=f["label"], zip_path=f["path"], config_path=f["config_path"],
                                      arch_signature="", team_str=(f["team_strs"][0] if f["team_strs"] else None),
                                      team_strs=tuple(f["team_strs"])) for f in fixed]
        tb, flat, fixed_b = eval_builders(None, entries)
        # The terminal from the run's model_config.json — exactly what the Python worker builds its
        # reward from (`RewardConfig.from_dict`).
        from agents.training.reward_config import RewardConfig

        terminal = EP.terminal_from_reward_config(
            RewardConfig.from_dict(json.loads((model_dir / "model_config.json").read_text())))
        ev = build_eval_core(decl, collector_decl=cdecl, svc=svc, extra_ids=list(range(len(reused_labels), n_slots)),
                             trainee_builder=tb, opp_builder=flat, fixed_builders=fixed_b,
                             turn_limit=EP.stall_threshold(), terminal=terminal, fixed_policies=fixed_pol,
                             emit=lambda _m: None)
        slot_ids = [mid for g in svc.groups for mid in g.model_ids]
        pool = ShardedEvalPool(items, shard_games, step=step)
        run_dir.mkdir(parents=True, exist_ok=True)
        pool.write_plan(str(run_dir))

        sent = load_sentinels(pool, model)
        glog: List[Dict[str, Any]] = []
        from agents.training.eval_callback import ForensicQuota

        st = ev.run_cycle(pool, str(run_dir), step=step, trainee_policy=model.policy, sentinel_policies=sent,
                          forensic_root=str(model_dir / "eval_traces" / f"step_{step}"),
                          quota=ForensicQuota(**quota), gamma=float(model.gamma), sentinel_greedy=bool(sentinel_greedy),
                          self_play_temp=float(self_play_temp), cycle_seed=cycle_seed, game_log=glog,
                          safe_point=safe_point)
        ev.close()
        toggles = arch_toggles_from_model(model)
        return {"stats": st.as_dict(), "games": glog, "svc_startup_s": t_svc, "arch_toggles": toggles,
                "gamma": float(model.gamma), "slot_model_ids": slot_ids,
                "fixed_slots": [list(x) for x in ev.table.fixed_slots]}


def run_python(*, run_dir: Path, model_dir: Path, trainee: str, items: List[Any], shard_games: int, step: int,
               cycle_seed: int, quota: Dict[str, int], arch_toggles: Dict[str, Any], gamma: float,
               compile_extractor: bool = False, timeout: float = 7200.0, sentinel_greedy: bool = True,
               self_play_temp: float = 1.0, skip_units: Optional[List[str]] = None) -> Dict[str, Any]:
    """The Python path (module docs): today's eval worker in a CUDA-less child, per-game seeded."""
    from agents.training.eval_sharding import ShardedEvalPool

    pool = ShardedEvalPool(items, shard_games, step=step)
    run_dir.mkdir(parents=True, exist_ok=True)
    pool.write_plan(str(run_dir))
    claim = run_dir / "claims"
    claim.mkdir(exist_ok=True)
    for uid in skip_units or []:            # pre-claimed = never played (the pool's O_EXCL claim lock)
        (claim / f"{uid}.lock").touch()
    glog = run_dir / "games.jsonl"
    cfg = {"snapshot": trainee, "port": None, "use_showdown_bridge": True, "compile_extractor": bool(compile_extractor),
           "bridge_impl": "rust", "model_dir": str(model_dir), "step": step, "claim_dir": str(claim),
           "result_dir": str(run_dir), "concurrency": 1, "device": "cpu", "cycle_tag": "lh", "gamma": gamma,
           "forensic_quota": dict(quota), "arch_toggles": arch_toggles, "trainee_team_str": None,
           "eval_sentinel_greedy": bool(sentinel_greedy), "self_play_temp": float(self_play_temp),
           "seed_base": int(cycle_seed),
           "seed_rule": "per_game", "game_log_path": str(glog), "worker_id": 0}
    cfg_path = run_dir / "config_0.json"
    cfg_path.write_text(json.dumps(cfg))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", "main.eval_worker", str(cfg_path)], env=env, capture_output=True,
                       text=True, timeout=timeout)
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"the Python eval worker failed (rc {r.returncode}):\n{(r.stdout + r.stderr)[-4000:]}")
    games = [json.loads(line) for line in glog.read_text().splitlines() if line.strip()] if glog.exists() else []
    return {"games": games, "wall_s": wall}


def _order_key(turns: Optional[List[int]], i: int) -> Optional[Tuple[int, int]]:
    """A decision's place on the game's clock: (battle turn, its index among this side's decisions in that
    turn) — a turn's move is 0, a forced switch after the turn resolved is 1. None without a clock."""
    if not turns or i >= len(turns):
        return None
    t = int(turns[i])
    return t, sum(1 for x in turns[:i] if int(x) == t)


def compare_games(rust: List[Dict[str, Any]], py: List[Dict[str, Any]], bar: float) -> Dict[str, Any]:
    """Per-game equality under the declared tie rule (module docs). Where BOTH paths logged the policy
    opponent's decisions (``opp`` rows ``[dec, action, argmax, margin, …]``), its ``[dec, action]`` stream
    is compared too. A game with a first difference on either stream is judged at the EARLIEST one on the
    Rust path's clock (:func:`_order_key`): it is a TIE iff every first difference in that earliest group
    is under ``NEAR_TIE_FACTOR x bar`` on both paths (the opponent's margin is its top-2 log-prob margin, or
    for a keyed draw its CDF margin); a later difference on the other stream is then a consequence of the
    tie. Two first differences at the same place (simultaneous, or no clock) must BOTH be ties."""
    rk = {(g["item"], int(g["game"])): g for g in rust}
    pk = {(g["item"], int(g["game"])): g for g in py}
    out: Dict[str, Any] = {"games": len(rk), "python_games": len(pk), "equal": 0, "ties": [], "fatal": [],
                           "decisions": 0, "missing": sorted(set(rk) ^ set(pk)), "max_dlogp": 0.0,
                           "opp_games": 0, "opp_decisions": 0}
    eps = NEAR_TIE_FACTOR * float(bar)
    for k in sorted(set(rk) & set(pk)):
        a, b = rk[k], pk[k]
        ra, pa = list(a["actions"]), list(b["actions"])
        j = next((i for i, (x, y) in enumerate(zip(ra, pa)) if int(x) != int(y)), None)
        oa, ob = a.get("opp"), b.get("opp")
        both_opp = oa is not None and ob is not None
        oj = None
        opp_same = True
        if both_opp:
            sa = [(int(r[0]), int(r[1])) for r in oa]
            sb = [(int(r[0]), int(r[1])) for r in ob]
            oj = next((i for i, (x, y) in enumerate(zip(sa, sb)) if x != y), None)
            opp_same = oj is None and len(sa) == len(sb)
        tclock = a.get("turns")
        oclock = [int(r[4]) for r in oa] if oa and len(oa[0]) > 4 else None
        # |Δ log p(chosen)| over the trainee decisions BEFORE the game's first divergence on either stream
        # (after an opponent's tied flip the trainee sees another state: its log-probs are not comparable)
        o_first = _order_key(oclock, oj) if oj is not None else None
        for i, (x, y) in enumerate(zip(a.get("logp", []), b.get("logp", []))):
            if (j is not None and i >= j) or (oj is not None and (o_first is None or tclock is None
                                                                  or _order_key(tclock, i) > o_first)):
                break
            out["max_dlogp"] = max(out["max_dlogp"], abs(float(x) - float(y)))
        if j is None and len(ra) == len(pa) and opp_same and int(a["winner"]) == int(b["winner"]) \
                and int(a["end_turn"]) == int(b["end_turn"]):
            out["equal"] += 1
            out["decisions"] += len(ra)
            if both_opp:
                out["opp_games"] += 1
                out["opp_decisions"] += len(oa)
            continue
        row: Dict[str, Any] = {"item": k[0], "game": k[1], "shard": a.get("shard"), "first_diff": j,
                               "rust": [a["winner"], a["end_turn"], len(ra)], "python": [b["winner"], b["end_turn"], len(pa)]}
        events = []                                    # (clock key, margin)
        if j is not None:
            mr, mp = float(a["margins"][j]), float(b["margins"][j])
            row.update(margin_rust=mr, margin_python=mp)
            events.append((_order_key(tclock, j), max(mr, mp)))
        if oj is not None:
            omr, omp = float(oa[oj][3]), float(ob[oj][3])
            row.update(opp_first_diff=oj, opp_rust=list(oa[oj]), opp_python=list(ob[oj]), opp_margin_rust=omr,
                       opp_margin_python=omp)
            events.append((o_first, max(omr, omp)))
        if events:
            keys = [e[0] for e in events]
            first = min(keys) if all(x is not None for x in keys) else None
            judged = [m for key, m in events if first is None or key == first]
            if all(m < eps for m in judged):
                out["ties"].append(row)
                continue
        out["fatal"].append(row)
    return out


def sampled_stats(games: List[Dict[str, Any]], sentinel_keys: List[str]) -> Dict[str, Any]:
    """The Rust path's sentinel decisions in a cycle: how many, and how many DREW an action other than
    the argmax of the served log-probs (a greedy regime reads 0; a sampling one must not)."""
    rows = [r for g in games if g["item"] in set(sentinel_keys) for r in (g.get("opp") or [])]
    return {"sentinel_decisions": len(rows), "drawn_not_argmax": sum(int(r[1]) != int(r[2]) for r in rows)}


def compare_metrics(rust_dir: Path, py_dir: Path) -> Dict[str, Any]:
    """Both cycles pooled by the unchanged collect; every metric equal, the TD tails within ``TD_BAR``."""
    from agents.training.eval_sharding import ShardedEvalPool

    mr, miss_r = ShardedEvalPool.from_plan(str(rust_dir)).collect(str(rust_dir))
    mp, miss_p = ShardedEvalPool.from_plan(str(py_dir)).collect(str(py_dir))
    diffs = []
    for block in sorted(set(mr) | set(mp)):
        a, b = mr.get(block, {}), mp.get(block, {})
        for key in sorted(set(a) | set(b)):
            x, y = a.get(key), b.get(key)
            if block == "durations_sec":
                continue
            if block == "td_resid_tails":
                ok = (x is None and y is None) or (x is not None and y is not None and abs(float(x) - float(y)) <= TD_BAR)
            else:
                ok = json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True)
            if not ok:
                diffs.append({"block": block, "key": key, "rust": x, "python": y})
    return {"missing": [miss_r, miss_p], "diffs": diffs, "rust": {k: v for k, v in mr.items() if k != "durations_sec"},
            "python": {k: v for k, v in mp.items() if k != "durations_sec"}}


def compare_traces(rust_model_dir: Path, py_model_dir: Path, step: int, *, limit: int = 0,
                   exclude_units: Optional[List[Tuple[str, int]]] = None) -> Dict[str, Any]:
    """The same trace files on both sides; every (or the first ``limit``) Rust core trace EXPANDS in the
    prober to the Python trace's decisions (turn, phase, chosen, both actives, per decision).
    ``exclude_units``: the (item, shard) units holding a declared TIE — a tied game legitimately
    diverges, and the unit's later quota decisions follow its outcome, so its traces are not compared."""
    skip = {(str(i), int(sh)) for i, sh in (exclude_units or []) if sh is not None}

    def unit_of(rel: str) -> Tuple[str, int]:
        item, _, name = rel.partition("/")
        for part in name.split("_"):
            if part[:1] == "s" and part[1:].isdigit():
                return item, int(part[1:])
        return item, -1

    def names(root: Path) -> Dict[str, Path]:
        d = root / "eval_traces" / f"step_{step}"
        found = {str(p.relative_to(d)): p for p in d.rglob("*_summary.json")} if d.exists() else {}
        return {rel: p for rel, p in found.items() if unit_of(rel) not in skip}

    rn, pn = names(rust_model_dir), names(py_model_dir)
    out: Dict[str, Any] = {"rust": len(rn), "python": len(pn), "only_rust": sorted(set(rn) - set(pn)),
                           "only_python": sorted(set(pn) - set(rn)), "checked": 0, "decision_diffs": []}
    try:
        from main.prober import core_trace
    except ImportError as e:                 # the expander is part of this lane; its absence is a failure
        out["error"] = f"main.prober.core_trace unavailable: {e}"
        return out
    keys = ("turn", "phase", "chosen", "our", "opp")
    for rel in sorted(set(rn) & set(pn))[: limit or None]:
        rs = core_trace.load_summary(str(rn[rel]))
        ps = core_trace.load_summary(str(pn[rel]))      # the Python side: passed through as stored
        ri, pi = rs.get("invocations", []), ps.get("invocations", [])
        out["checked"] += 1
        if len(ri) != len(pi):
            out["decision_diffs"].append({"trace": rel, "n": [len(ri), len(pi)]})
            continue
        for i, (x, y) in enumerate(zip(ri, pi)):
            bad = [k for k in keys if x.get(k) != y.get(k)]
            if bad:
                out["decision_diffs"].append({"trace": rel, "decision": i, "keys": bad,
                                              "rust": {k: x.get(k) for k in bad}, "python": {k: y.get(k) for k in bad}})
                break
    return out


def run(cfg: Dict[str, Any], workdir: Optional[str] = None) -> Dict[str, Any]:
    """The whole gate (module docs). ``cfg``: bots (None = the roster, [] = none), sentinels (count), games,
    shard_games, n_envs, device, backend, seed, quota, trainee/sentinel_paths (optional real files), bar,
    trace_limit, fixed, sentinel_greedy, self_play_temp; teeth only: python_seed. The run's inputs and the
    Rust path's games are kept in ``workdir`` for :func:`rerun_python`."""
    from agents.training.eval_callback import eval_opponent_names

    wd = Path(workdir or tempfile.mkdtemp(prefix="laneH_gate_"))
    step = int(cfg.get("step", 1000))
    bots = list(eval_opponent_names() if cfg.get("bots") is None else cfg["bots"])   # [] = no bot items
    trainee, sentinels, mcfg = build_models(wd / "models", n_sentinels=int(cfg.get("sentinels", 2)),
                                            trainee=cfg.get("trainee"), sentinels=cfg.get("sentinel_paths"))
    fixed = build_fixed_models(wd / "models" / "fixed", list(cfg.get("fixed") or []), mcfg)
    greedy = bool(cfg.get("sentinel_greedy", True))
    temp = float(cfg.get("self_play_temp", 1.0))
    items = _items(bots, sentinels, int(cfg.get("games", 4)), fixed)
    quota = dict(cfg.get("quota") or {"win": 5, "loss": 10, "draw": 5})
    seed = int(cfg.get("seed", 20260930))
    dirs = {}
    for side in ("rust", "python"):
        md = wd / f"run_{side}"
        md.mkdir(parents=True, exist_ok=True)
        (md / "model_config.json").write_text(Path(mcfg).read_text())
        dirs[side] = md
    t0 = time.perf_counter()
    rust = run_rust(run_dir=dirs["rust"] / ".eval_runs" / f"step_{step}", model_dir=dirs["rust"], trainee=trainee,
                    sentinels=sentinels, items=items, shard_games=int(cfg.get("shard_games", 25)), step=step,
                    cycle_seed=seed, quota=quota, device=str(cfg.get("device", "cpu")),
                    backend=str(cfg.get("backend", "eager")), n_envs=int(cfg.get("n_envs", 16)),
                    front=str(cfg.get("front", "ffi")), profile=str(cfg.get("profile", "selfcheck")),
                    fixed=fixed, sentinel_greedy=greedy, self_play_temp=temp)
    t_rust = time.perf_counter() - t0
    inputs = {"trainee": trainee, "sentinels": sentinels, "bots": bots, "fixed": fixed, "games": int(cfg.get("games", 4)),
              "shard_games": int(cfg.get("shard_games", 25)), "step": step, "seed": seed, "quota": quota,
              "arch_toggles": rust["arch_toggles"], "gamma": rust["gamma"], "sentinel_greedy": greedy,
              "self_play_temp": temp, "bar": float(cfg.get("bar", BAR_CPU))}
    (wd / "gate_inputs.json").write_text(json.dumps(inputs, default=str))
    (wd / "rust_games.json").write_text(json.dumps([{k: v for k, v in g.items() if k != "script"} for g in rust["games"]]))
    py = run_python(run_dir=dirs["python"] / ".eval_runs" / f"step_{step}", model_dir=dirs["python"],
                    trainee=trainee, items=items, shard_games=int(cfg.get("shard_games", 25)), step=step,
                    cycle_seed=int(cfg.get("python_seed", seed)),   # a different one only in the teeth test
                    quota=quota, arch_toggles=rust["arch_toggles"], gamma=rust["gamma"],
                    compile_extractor=bool(cfg.get("compile_python", False)), sentinel_greedy=greedy,
                    self_play_temp=temp)
    bar = float(cfg.get("bar", BAR_CPU))
    games = compare_games(rust["games"], py["games"], bar)
    metrics = compare_metrics(dirs["rust"] / ".eval_runs" / f"step_{step}", dirs["python"] / ".eval_runs" / f"step_{step}")
    traces = compare_traces(dirs["rust"], dirs["python"], step, limit=int(cfg.get("trace_limit", 0)),
                            exclude_units=[(t["item"], t["shard"]) for t in games["ties"]])
    ok = (not games["fatal"] and not games["missing"] and games["max_dlogp"] <= bar
          and (bool(games["ties"]) or not metrics["diffs"])
          and not traces["only_rust"] and not traces["only_python"] and not traces["decision_diffs"]
          and "error" not in traces and not rust["stats"]["lifecycle"].get("bad"))
    return {"pass": bool(ok), "workdir": str(wd), "cfg": {**cfg, "bots": bots, "bar": bar, "seed": seed},
            "games": {k: v for k, v in games.items()}, "metrics_diffs": metrics["diffs"],
            "sampled": sampled_stats(rust["games"], [f"sentinel_{i}" for i in range(len(sentinels))]),
            "fixed": [{k: f[k] for k in ("label", "reused")} | {"pins": len(f["team_strs"])} for f in fixed],
            "rust_fixed_slots": rust["fixed_slots"], "rust_slot_model_ids": rust["slot_model_ids"],
            "rust_opp_teams": {k: len({int(g["teams"][1]) for g in rust["games"] if g["item"] == k})
                               for k in sorted({g["item"] for g in rust["games"]})},
            "metrics_equal": not metrics["diffs"], "metrics": metrics["rust"], "traces": traces,
            "rust_stats": rust["stats"], "rust_wall_s": t_rust, "python_wall_s": py["wall_s"],
            "svc_startup_s": rust["svc_startup_s"]}


def rerun_python(workdir: str, *, keep: List[str], python_self_play_temp: Optional[float] = None,
                 python_fixed_paths: Optional[Dict[str, str]] = None, shards: Optional[List[int]] = None,
                 python_seed: Optional[int] = None, tag: str = "rerun") -> Dict[str, Any]:
    """TEETH: replay the Python path of a finished :func:`run` on a SUBSET of its items (``keep``) with one
    input MUTATED — a sentinel sampling at another temperature, another cycle seed (``python_seed``), a fixed opponent from another file
    (``"trainee"`` = the trainee's own zip: what an unloaded Rust fixed slot plays) — and judge its games
    against the Rust games that run recorded. A subset keeps every game's key: an item's units, game
    indices and per-unit quota depend on that item alone; ``shards`` keeps only those shard units of it
    (the others are pre-claimed, so the worker never plays them)."""
    wd = Path(workdir)
    inp = json.loads((wd / "gate_inputs.json").read_text())
    rust_games = json.loads((wd / "rust_games.json").read_text())
    swap = {lab: (inp["trainee"] if path == "trainee" else path) for lab, path in (python_fixed_paths or {}).items()}
    fixed = [dict(f, path=swap.get(f["label"], f["path"])) for f in inp["fixed"]]
    items = [it for it in _items(inp["bots"], inp["sentinels"], inp["games"], fixed) if it.key in set(keep)]
    if {it.key for it in items} != set(keep):
        raise ValueError(f"rerun_python: unknown items {sorted(set(keep) - {it.key for it in items})}")
    md = wd / f"run_python_{tag}"
    md.mkdir(parents=True, exist_ok=True)
    (md / "model_config.json").write_text((wd / "run_python" / "model_config.json").read_text())
    temp = inp["self_play_temp"] if python_self_play_temp is None else float(python_self_play_temp)
    from agents.training.eval_sharding import ShardedEvalPool

    units = ShardedEvalPool(items, inp["shard_games"], step=inp["step"]).units
    skip = [u.unit_id for u in units if shards is not None and u.shard_index not in set(shards)]
    kept = {(u.item.key, u.shard_index) for u in units if u.unit_id not in set(skip)}
    py = run_python(run_dir=md / ".eval_runs" / f"step_{inp['step']}", model_dir=md, trainee=inp["trainee"],
                    items=items, shard_games=inp["shard_games"], step=inp["step"], cycle_seed=inp["seed"] if python_seed is None else int(python_seed),
                    quota=inp["quota"], arch_toggles=inp["arch_toggles"], gamma=inp["gamma"],
                    sentinel_greedy=inp["sentinel_greedy"], self_play_temp=temp, skip_units=skip)
    return compare_games([g for g in rust_games if (g["item"], int(g["shard"])) in kept], py["games"], inp["bar"])


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--shard-games", type=int, default=25)
    ap.add_argument("--sentinels", type=int, default=2)
    ap.add_argument("--bots", default="")
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--backend", default="eager")
    ap.add_argument("--front", default="ffi")
    ap.add_argument("--profile", default="selfcheck")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--trainee", default=None)
    ap.add_argument("--sentinel-paths", default="")
    ap.add_argument("--bar", type=float, default=None)
    ap.add_argument("--compile-python", action="store_true")
    ap.add_argument("--trace-limit", type=int, default=0)
    ap.add_argument("--fixed", default="", help="fixed opponents, comma-separated '<pins>[r]' (r = the training "
                    "plan's reused stable slot), e.g. '2r,0'")
    ap.add_argument("--sampled", action="store_true", help="the sampled sentinel regime (--no-eval-sentinel-greedy)")
    ap.add_argument("--self-play-temp", type=float, default=1.0)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = {"games": a.games, "shard_games": a.shard_games, "sentinels": a.sentinels, "n_envs": a.n_envs,
           "device": a.device, "backend": a.backend, "front": a.front, "profile": a.profile, "seed": a.seed,
           "bots": [b for b in a.bots.split(",") if b] or None, "trainee": a.trainee,
           "sentinel_paths": [p for p in a.sentinel_paths.split(",") if p] or None,
           "bar": a.bar if a.bar is not None else (BAR_GPU if a.device.startswith("cuda") else BAR_CPU),
           "compile_python": a.compile_python, "trace_limit": a.trace_limit,
           "fixed": [{"pins": int(t.rstrip("r")), "reused": t.endswith("r")} for t in a.fixed.split(",") if t],
           "sentinel_greedy": not a.sampled, "self_play_temp": a.self_play_temp}
    rep = run(cfg, a.workdir)
    text = json.dumps(rep, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(text)
    g = rep["games"]
    print(f"LANE H GATE {'PASS' if rep['pass'] else 'FAIL'}: {g['equal']}/{g['games']} games equal "
          f"({g['decisions']} trainee decisions, max |dlogp(chosen)| {g['max_dlogp']:.2e}), {len(g['ties'])} ties, {len(g['fatal'])} fatal; metrics "
          f"{'EQUAL' if rep['metrics_equal'] else 'DIFFER'}; traces {rep['traces'].get('checked')} checked, "
          f"{len(rep['traces'].get('decision_diffs', []))} diffs; sentinel draws off-argmax "
          f"{rep['sampled']['drawn_not_argmax']}/{rep['sampled']['sentinel_decisions']}; rust {rep['rust_wall_s']:.1f}s, python "
          f"{rep['python_wall_s']:.1f}s")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
