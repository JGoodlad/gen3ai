"""SLICE N WITH A POLICY OPPONENT — the M5 Lane E gate (a harness + CLI; the tests drive it).

RECORD through the new path, REPLAY through today's, compare every opponent decision:

1. RECORD (this process; the GPU when ``--device cuda``): the Rust env core (FFI) plays N envs whose
   opponent routes come from :class:`RustEnvOpponents` — the production sampler at self-play
   fraction 1.0 over a real ``SnapshotPool`` directory — and whose POLICY rows are answered by
   :class:`PolicyOpponentServer` over a started T2 ``InferenceService`` (greedy, or sampled from the
   per-env generators). p1 plays a seeded random policy. Optionally a mid-run POOL REFRESH: a
   snapshot is added to the pool directory and the generation pushed, i.e. a declared LOAD into a
   free slot; the service's ``*_after_freeze`` counters must stay 0 across it. Every COMPLETED
   episode is written out: its teams, seed, route, the snapshot its slot held, the p2 generator
   state at its start, both sides' actions, and per p2 decision the core row, mask, T2 log-probs,
   greedy and action; and its outcome (reward / terminated / truncated).
2. REPLAY (a CHILD process with CUDA hidden — exactly an env worker): per episode, a production-
   surface ``Gen3Env`` on the rust bridge (Lane C's replay) plays p1's recorded indices against the
   PER-ENV path: an ``RLPlayer`` holding the SAME snapshot loaded by ``SnapshotPool.load_model`` —
   ``--compile-opponents``' ``maybe_compile_extractor(hide_cuda=True, strict=True)`` when
   ``compile`` — at the same temperature, its private generator restored to the recorded state,
   polled by the real ``SingleAgentWrapper`` on EVERY step as training polls it. At every p2
   decision the env actually sends: its embedded row == the core's row (bytes), its greedy / its
   sample == the recorded one, max |Δ legal log-prob|; per episode the outcome and the decision
   counts. A mismatch is counted and the RECORDED action is played on, so one disagreement cannot
   hide the rest.

THE BARS (declared in the ASSERTION, :func:`judge_flips`). Every legal log-prob agrees within the
tier's bar ``B`` (``max_dlogp < B``). An ARGMAX FLIP (the per-env path's greedy or sample differs from
the recorded one) is data, recorded with its MARGIN — the recorded top-2 legal log-prob gap (greedy),
or the top-2 gap of ``logp / T - log q`` times ``T`` (sampled; ``q`` the draw's Exp(1) variates,
recomputed from the recorded generator state) — and judged: a flip needs ``margin <= |Δ_a| + |Δ_b| <=
2 x max|Δ|``, so with every |Δ| under ``B`` a flip at ``margin >= NEAR_TIE_FACTOR x B`` cannot come from
rounding and is FATAL, while one below it is a TIE (an agreement, counted and reported). Exact ties
(margin 0.0) exist in real and fresh policies (T2, measured) — two switch targets the pointer head
scores identically — and there either argmax is correct.

DECLARED TORCH STATE (F-LJ-6, found by Lane J). The recorded log-probs and the fresh snapshots depend
on process-global torch state: the intra-op THREAD COUNT changes CPU rounding — measured: a fresh
production policy's orthogonal init (a QR) differs in 94 of 721 tensors by up to 7.5e-6 between 8 and 4
threads, and one policy's forward bytes differ between 1, 4 and 8 threads. A test that set the count
and left it (this harness's own ``record``, before the fix) changed the NEXT test's snapshots, and so
which way an exact tie fell. :func:`declared_torch_state` pins every such global for the region it
wraps (the snapshot build at ``BUILD_THREADS``, the recording at ``RECORD_THREADS``, the replay child at
1), refuses to start when a global it does not own is off its declared value
(``GateStateError``: fp32 matmul precision "highest", default dtype float32), re-checks on exit, and
restores the thread count it found.

PHANTOM POLLS (F-LF-2's policy twin). The wrapper calls ``choose_move`` on steps whose p2 order is
never sent; today that DRAWS from the opponent's generator. The core asks only at real decisions,
so the replay restores the generator after a phantom poll (the stream the new path defines) and
counts the phantoms — the stream today's path consumes is not reproducible by construction.

    export PYTHONPATH=$PYTHONPATH:src
    python -m agents.training.rust_env_opponents_parity --pool fresh:3 --n-envs 4 --episodes 8
    scripts/ops/gpu_lock.sh python -m agents.training.rust_env_opponents_parity \\
        --pool run:/home/goodlad/dev/gen3ai/models/ai_v14_06_lbat_ctrl_fix/snapshots:6 --device cuda \\
        --backend graph --n-envs 16 --episodes 64 --mode sampled --compile --refresh-at 150
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

LOGP_BAR = 1e-3
#: A flip below ``NEAR_TIE_FACTOR x`` the tier's |Δ legal log-prob| bar is a TIE (see THE BARS): 2 is the
#: exact bound (each of the two compared log-probs moves by at most the bar).
NEAR_TIE_FACTOR = 2.0
#: Intra-op threads while BUILDING fresh snapshots (their init is thread-sensitive) and while RECORDING.
BUILD_THREADS = 1
RECORD_THREADS = 4
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


def judge_flips(rep: Dict[str, Any], dlogp_bar: float) -> Tuple[int, List[Dict[str, Any]]]:
    """The declared tie rule: ``(ties, fatal)`` — flips whose margin is below ``NEAR_TIE_FACTOR x
    dlogp_bar`` are TIES; every other flip is FATAL (see THE BARS)."""
    eps = NEAR_TIE_FACTOR * float(dlogp_bar)
    flips = rep.get("flips", [])
    fatal = [f for f in flips if f["margin"] is None or f["margin"] >= eps]
    return len(flips) - len(fatal), fatal


def near_ties(rep: Dict[str, Any], dlogp_bar: float) -> Dict[str, int]:
    """How many DECISIONS (flipped or not) sat inside the tie band at the TIER's bar — the same
    ``NEAR_TIE_FACTOR x dlogp_bar`` threshold :func:`judge_flips` uses (one threshold, not a fixed one)."""
    eps = NEAR_TIE_FACTOR * float(dlogp_bar)
    return {k: sum(1 for m in rep.get(f"margins_{k}", []) if m is not None and m < eps) for k in ("greedy", "sample")}
NAMES = ("lepone", "leptwo")


@dataclass
class P2Dec:
    dec_n: int
    row: bytes
    mask: np.ndarray
    logp: np.ndarray
    greedy: int
    action: int


@dataclass
class Episode:
    env: int
    episode: int
    teams: Tuple[int, int]
    seed: List[int]
    route: int
    model_id: str
    temperature: float
    gen_state: Optional[bytes]
    p1: List[int] = field(default_factory=list)
    p2: List[P2Dec] = field(default_factory=list)
    reward: float = 0.0
    terminated: int = 0
    truncated: int = 0


# ----------------------------------------------------------------------------------- the pool dir


def build_pool_dir(dst: Path, spec: str) -> Tuple[List[str], List[str]]:
    """``fresh:K`` — K+1 perturbed fresh production-arch snapshots saved as a pool; ``run:<dir>:K`` —
    symlinks to K+1 of a real pool's snapshots (evenly spaced; nothing is written under the source).
    The first K form the initial pool; the last is HELD BACK for the refresh. Returns (initial,
    held-back) file names."""
    dst.mkdir(parents=True, exist_ok=True)
    kind, _, rest = spec.partition(":")
    if kind == "fresh":
        k = int(rest)
        from agents.model.parity_probe import PERTURB_SCALE, perturb_
        from agents.model.snapshot import current_model_version
        from agents.observation.state_encoder import load_mappings
        from main.fresh_checkpoint import build_fresh_model

        ver = current_model_version(load_mappings())
        names = []
        for i in range(k + 1):
            model, _, _ = build_fresh_model(i)
            perturb_(model.policy, seed=1000 + i, scale=PERTURB_SCALE)
            name = f"snapshot_{(i + 1) * 1000:012d}.zip"
            model.save(str(dst / name))
            names.append(name)
        (dst / "model_config.json").write_text(ver.to_json())
    elif kind == "run":
        src_s, _, k_s = rest.rpartition(":")
        src, k = Path(src_s), int(k_s)
        allz = sorted(src.glob("snapshot_*.zip"))
        if len(allz) < k + 1:
            raise SystemExit(f"{src} holds {len(allz)} snapshots; {k + 1} needed")
        pick = [allz[round(i * (len(allz) - 1) / k)] for i in range(k + 1)]
        names = []
        for z in pick:
            (dst / z.name).symlink_to(z)
            names.append(z.name)
        (dst / "model_config.json").symlink_to(src / "model_config.json")
    else:
        raise SystemExit(f"--pool {spec!r}: fresh:K or run:<dir>:K")
    held = names[-1]
    hold = dst.parent / (dst.name + "_held")
    hold.mkdir(exist_ok=True)
    os.replace(dst / held, hold / held)
    (hold / "model_config.json").write_text((dst / "model_config.json").read_text())
    return names[:-1], [held]


def build_gate_pool(dst: Path, spec: str) -> Tuple[List[str], List[str]]:
    """:func:`build_pool_dir` under the declared state: a fresh policy's init is thread-sensitive
    (F-LJ-6), so the gate's snapshots are built at ``BUILD_THREADS`` whatever the process holds."""
    with declared_torch_state(BUILD_THREADS):
        return build_pool_dir(dst, spec)


def _pool_version(pool_dir: Path) -> Any:
    """The version every snapshot is checked against: the pool's own ``model_config.json`` — in
    training the TRAINEE's version (``SnapshotPool._write`` puts it there), never the code defaults."""
    from agents.model.model_version import ModelVersion

    return ModelVersion.from_json_file(str(Path(pool_dir) / "model_config.json"))


def _seed(key: int) -> List[int]:
    return [(11 + key) % 65536, (22 + key) % 65536, (33 + key) % 65536, (44 + key) % 65536]


# ----------------------------------------------------------------------------------- 1. record


def record(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Under :func:`declared_torch_state` (``RECORD_THREADS``): the thread count is pinned for the
    recording and RESTORED after it — the leak F-LJ-6 traced."""
    with declared_torch_state(RECORD_THREADS):
        return _record(cfg)


def _record(cfg: Dict[str, Any]) -> Dict[str, Any]:

    from agents.inference.service import InferenceService, ServiceSpec, SlotGroupSpec
    from agents.training import rust_env_opponents as E
    from agents.training.snapshot_pool import SnapshotPool
    from main.rust_core_cutover.envs import packed_teams
    from utils.rust_env import episode as EP
    from utils.rust_env import ffi
    from utils.rust_env import protocol as P

    pool_dir = Path(cfg["pool_dir"])
    ver = _pool_version(pool_dir)
    pool = SnapshotPool(pool_dir, current_version=ver, device="cpu", lru_cache_size=64)
    n_init = len(pool)
    plan = E.OpponentPlan(pool_slots=n_init + 1 + int(cfg.get("spare", 2)), bots=("random",),
                          self_play_temp=float(cfg.get("temperature", 1.0)))
    template = pool.load_model(pool._entries[0]).policy.eval()
    lanes = int(cfg.get("lanes") or (min(plan.n_policy_slots, 8) if cfg["device"].startswith("cuda") else 1))
    buckets = tuple(int(b) for b in cfg["buckets"])
    t0 = time.perf_counter()
    svc = InferenceService(ServiceSpec(groups=(SlotGroupSpec("pool", plan.n_policy_slots, template),),
                                       device=cfg["device"], backend=cfg["backend"], buckets=buckets,
                                       lanes=lanes, max_rows_per_flush=max(1024, buckets[-1] * 4))).startup()
    startup_s = time.perf_counter() - t0
    routes = plan.routes()
    load_log: List[Tuple[int, str, float]] = []

    def load(route: int, mid: str) -> None:
        step = int(mid.split(":")[1])
        entry = next(e for e in pool._entries if e.step == step)
        t = time.perf_counter()
        svc.load(int(routes[route].slot), pool.load_model(entry).policy.eval(), mid)
        load_log.append((route, mid, time.perf_counter() - t))

    n, kb = int(cfg["n_envs"]), int(cfg["key_base"])
    host = E.RustEnvOpponents(plan, n, load=load, pool=pool, self_play_fraction=1.0,
                              pool_rng_seeds=[kb + i for i in range(n)], rng_seeds=list(range(n)))
    host.set_self_play_target(1.0, 0)
    greedy = cfg["mode"] == "greedy"
    keyed = cfg["mode"] == "keyed"
    # M5 Lane G (F-LE-8): mode `keyed` draws p2's sample with the counter-based KEYED DRAW
    # (`agents.training.keyed_draw`, key = (key_base, the opponent stream, env, episode, dec_n)); the replay
    # recomputes it from ITS OWN log-probs — no generator state to carry.
    server = E.PolicyOpponentServer(plan, svc, n, policy_seed=kb, seed_stride=1, force_greedy=greedy,
                                    sampling="keyed" if keyed else "generator", run_seed=kb)
    teams = packed_teams(cfg.get("teams", "pool"))
    turn_limit = int(cfg["turn_limit"]) if cfg.get("turn_limit") else EP.stall_threshold()
    spec = P.spec_json(n=n, threads=int(cfg.get("threads", 4)), teams=list(teams), names=NAMES,
                       decision_tense=False, switch_freeze=False, turn_limit=turn_limit, refusal_budget=0,
                       bank_dir=None, opponents=plan.spec_rows(bots="external"))
    lib = ffi.load(ffi.default_path(cfg.get("profile", "selfcheck")), nan_poison=cfg.get("profile", "selfcheck") == "selfcheck")
    rng = np.random.default_rng(kb)
    staged: Dict[int, Tuple[Tuple[int, int], List[int]]] = {}
    count = [0] * n
    cur: Dict[int, Episode] = {}
    done_eps: List[Episode] = []
    svc_before = {k: svc.counters[k] for k in ("compiles_after_freeze", "captures_after_freeze", "cuda_segments_after_freeze")}
    refresh_at = int(cfg.get("refresh_at") or 0)
    refreshed = None

    def stage_team(c: Any, i: int) -> None:
        a = int(rng.integers(len(teams)))
        b = int(rng.integers(len(teams) - 1))
        b += b >= a
        sd = _seed(kb * 1000 + i * 97 + count[i])
        count[i] += 1
        c["ep_team"][i] = [a, b]
        c["ep_seed"][i] = sd
        staged[i] = ((a, b), sd)

    t_steps = 0.0
    steps = 0
    with ffi.FfiCore(spec, lib=lib) as core:
        c = core.cols
        host.stage_all(c)
        for i in range(n):
            stage_team(c, i)
        core.reset()

        def moved_envs(moved: Any) -> None:
            for i in moved:
                i = int(i)
                route = int(c["opp_route"][i])
                fam = host.families["pool"]
                mid = fam.resident[fam.routes.index(route)]
                gen = None if (greedy or keyed) else server.generator(i, "pool")
                cur[i] = Episode(env=i, episode=int(c["episode"][i]), teams=staged[i][0], seed=staged[i][1],
                                 route=route, model_id=mid, temperature=server.temperature["pool"],
                                 gen_state=None if gen is None else bytes(gen.get_state().numpy().tobytes()))
                stage_team(c, i)

        moved_envs(host.after_op(c))
        while len(done_eps) < int(cfg["episodes"]):
            if steps >= int(cfg.get("max_steps", 200_000)):
                raise RuntimeError(f"{len(done_eps)} episodes in {steps} steps")
            ts = time.perf_counter()
            need = c["need"]
            rows = {int(i): (int(c["dec_n"][i, 1]), c["obs"][i, 1].tobytes(), c["mask"][i, 1].copy())
                    for i in np.flatnonzero(need[:, 1] == 1)}
            bad = [i for i in rows if c["opp_slot"][i] < 0]
            if bad:
                raise RuntimeError(f"envs {bad} need a p2 action on a non-policy route")
            rec: List[Any] = []
            server.serve(c, record=rec)
            for (i, _slot, logp, gr, act) in rec:
                d = rows[i]
                cur[i].p2.append(P2Dec(dec_n=d[0], row=d[1], mask=d[2], logp=logp, greedy=gr, action=act))
            for i in np.flatnonzero(need[:, 0] == 1):
                a = int(rng.choice(np.flatnonzero(c["mask"][i, 0])))
                c["action"][i, 0] = a
                cur[int(i)].p1.append(a)
            core.step()
            steps += 1
            for i in np.flatnonzero(c["done"] == 1):
                i = int(i)
                if c["refused"][i]:
                    raise RuntimeError(f"env {i} quarantined: {core.bank()[-1:]}")
                e = cur.pop(i)
                e.reward, e.terminated, e.truncated = float(c["reward"][i]), int(c["terminated"][i]), int(c["truncated"][i])
                done_eps.append(e)
            moved_envs(host.after_op(c))
            t_steps += time.perf_counter() - ts
            if refresh_at and steps == refresh_at:
                held_dir = pool_dir.parent / (pool_dir.name + "_held")
                for z in sorted(held_dir.glob("snapshot_*.zip")):
                    os.replace(z, pool_dir / z.name)
                loaded = host.set_self_play_target(1.0, 1)
                refreshed = {"step": steps, "routes_loaded": loaded, "roster": host.families["pool"].roster}
        after_freeze = core.after_freeze()
    svc_after = {k: svc.counters[k] for k in svc_before}
    return {
        "cfg": cfg, "episodes": done_eps, "startup_s": startup_s, "steps": steps, "t_steps": t_steps,
        "loads": load_log, "refresh": refreshed, "svc_counters_delta": {k: svc_after[k] - svc_before[k] for k in svc_before},
        "svc_loads": svc.counters["loads"], "core_after_freeze": after_freeze, "route_counts": host.route_counts.tolist(),
        "serve": vars(server.stats), "startup_parity_worst": max(r.legal_logprob_max for r in svc.startup_reports),
    }


# ----------------------------------------------------------------------------------- 2. replay


def _replay_player_cls() -> Any:
    import torch

    from agents.inference.player import RLPlayer

    class ReplayRLPlayer(RLPlayer):
        """``RLPlayer`` exactly as training polls it, instrumented: at a SENT decision its own choice
        is compared with the recording and the RECORDED action is played on; a phantom poll's draw is
        undone (the stream the core defines) and counted."""

        def bind(self, env: Any, ep: Any, out: Dict[str, Any], sampled: bool) -> None:
            self._env, self._ep, self._out, self._sampled, self._j = env, ep, out, sampled, 0
            self._cap: Optional[Dict[str, Any]] = None
            if sampled:
                g = torch.Generator(device="cpu")
                g.set_state(torch.frombuffer(bytearray(ep.gen_state), dtype=torch.uint8))
                self._policy_seed = 0
                self.__dict__["_policy_gens"] = {"cpu": g}

        def embed_battle(self, battle: Any) -> Any:
            d = super().embed_battle(battle)
            self._cap = d
            return d

        def _keyed_sample(self, masked_logits: Any, temperature: float) -> int:
            """RLPlayer's ``_action_sampler`` under mode ``keyed``: the keyed draw of THIS path's masked
            logits at the pending decision's key (a phantom poll, which has none, takes the argmax — its
            result is never sent)."""
            from agents.training import keyed_draw as KD

            ml = masked_logits[0].detach().float()
            if self._keyed_key is None:
                self._last_keyed_margin = float("inf")
                return int(torch.argmax(ml).item())
            legal = (ml > -1e8).numpy()
            lp = torch.log_softmax(ml, -1).numpy()
            lpm = np.where(legal, lp, -np.inf).astype(np.float32)[None]
            env, episode, dec = self._keyed_key
            u = KD.keyed_uniforms(self._keyed_seed, KD.STREAM_OPPONENT, env, episode, dec)
            a, m = KD.keyed_actions(lpm, np.atleast_1d(u), float(temperature))
            self._last_keyed_margin = float(m[0])
            return int(a[0])

        def _predict_best_action(self, battle: Any, stochastic: bool = False, need_aux: bool = True,
                                 temperature: float = 1.0) -> Any:
            o = self._out
            sent = bool(self._env.agent2_to_move)
            tracker = self._get_tracker(battle)
            snap = tracker.snapshot()
            g = self.__dict__.get("_policy_gens", {}).get("cpu")
            gstate = g.get_state() if g is not None else None
            if getattr(self, "_keyed_seed", None) is not None:
                nxt = self._ep.p2[self._j] if (sent and self._j < len(self._ep.p2)) else None
                self._keyed_key = None if nxt is None else (self._ep.env, self._ep.episode, nxt.dec_n)
            idx, probs, mask = super()._predict_best_action(battle, stochastic=stochastic, need_aux=need_aux,
                                                            temperature=temperature)
            if not sent:
                o["phantom_polls"] += 1
                o.setdefault("phantom_at", []).append((self._ep.env, self._ep.episode, self._j,
                                                       int(getattr(battle, "turn", -1)), idx))
                if g is not None and idx is not None:
                    o["phantom_draws"] += 1
                    g.set_state(gstate)
                return idx, probs, mask
            if idx is None:
                o["no_decision_sent"] += 1
                return idx, probs, mask
            ep = self._ep
            if self._j >= len(ep.p2):
                o["div"]["p2_extra"] = o["div"].get("p2_extra", 0) + 1
                return idx, probs, mask
            d = ep.p2[self._j]
            self._j += 1
            o["decisions"] += 1
            row = np.asarray(self._cap["observation"], dtype=np.float32).tobytes()
            if row != d.row:
                o["div"]["row"] = o["div"].get("row", 0) + 1
                a = np.frombuffer(row, dtype=np.float32)
                b = np.frombuffer(d.row, dtype=np.float32)
                cells = np.flatnonzero(a != b)
                o.setdefault("row_examples", []).append({
                    "env": ep.env, "episode": ep.episode, "dec": self._j - 1, "phantoms_before": o["phantom_polls"],
                    "n_cells": int(cells.size), "cells": [(int(x), float(a[x]), float(b[x])) for x in cells[:12]]})
            if not np.array_equal(np.asarray(self._cap["action_mask"]).astype(np.uint8), d.mask.astype(np.uint8)):
                o["div"]["mask"] = o["div"].get("mask", 0) + 1
            ml = self._last_masked_logits[0].detach().float()
            legal = d.mask.astype(bool)
            lp = torch.log_softmax(ml, -1).numpy()
            o["max_dlogp"] = max(o["max_dlogp"], float(np.abs(lp[legal] - d.logp[legal]).max()))
            srt = np.sort(d.logp[legal])[::-1]
            margin_g = float(srt[0] - srt[1]) if len(srt) > 1 else None
            if margin_g is not None:              # judged at the TIER's bar by `near_ties`
                o["margins_greedy"].append(margin_g)
            dl_row = float(np.abs(lp[legal] - d.logp[legal]).max())
            where = {"env": ep.env, "episode": ep.episode, "dec": self._j - 1, "dlogp_row": dl_row}
            g_cpu = int(torch.argmax(ml).item())
            if g_cpu != d.greedy:        # an ARGMAX FLIP is data; `judge_flips` decides (THE BARS)
                o["flips"].append({"kind": "greedy", "margin": margin_g, "path": g_cpu, "recorded": d.greedy, **where})
            if getattr(self, "_keyed_seed", None) is not None:
                # the KEYED draw (M5 Lane G): RLPlayer drew it from THIS path's log-probs through its
                # `_action_sampler` hook (below), keyed by the decision. Its margin is a CDF fraction; a
                # |Δ log p| <= B moves each cumulative share by <~ 2B, so the NEAR_TIE_FACTOR x B rule judges it.
                margin_k = float(self._last_keyed_margin)
                o["margins_sample"].append(margin_k)
                if idx != d.action:
                    o["flips"].append({"kind": "keyed", "margin": margin_k, "path": idx, "recorded": d.action, **where})
            if self._sampled:
                g2 = torch.Generator(device="cpu")
                g2.set_state(gstate)
                q = torch.empty(1, ml.shape[0]).exponential_(1, generator=g2)[0].numpy()
                r = d.logp[legal] / temperature - np.log(q[legal])
                rs = np.sort(r)[::-1]
                margin_s = float(rs[0] - rs[1]) * temperature if len(rs) > 1 else None
                if margin_s is not None:
                    o["margins_sample"].append(margin_s)
                if idx != d.action:
                    o["flips"].append({"kind": "sample", "margin": margin_s, "path": idx, "recorded": d.action, **where})
            if idx != d.action:                     # play the RECORDED action on (alignment)
                tracker.restore(snap)
                self.embed_battle(battle)
                tracker.advance(d.action)
                return d.action, probs, mask
            return idx, probs, mask

    return ReplayRLPlayer


def replay(rec_path: str, out_path: str) -> Dict[str, Any]:

    with declared_torch_state(1):                             # an env worker's pin, declared
        return _replay(rec_path, out_path)


def _replay(rec_path: str, out_path: str) -> Dict[str, Any]:
    with open(rec_path, "rb") as f:
        rec = pickle.load(f)
    cfg, eps = rec["cfg"], rec["episodes"]
    from poke_env import AccountConfiguration
    from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

    from agents.observation.state_encoder import load_mappings
    from agents.training.gen3_env import Gen3Env
    from agents.training.reward_config import RewardConfig
    from agents.training.reward_manager import Gen3RewardManager
    from agents.training.snapshot_pool import SnapshotPool
    from agents.training.stall import StallConfig
    from main.rust_core_cutover.envs import SequenceTeambuilder, packed_teams, production_args
    from main.train.env_factory import trainee_env_kwargs
    from utils.bridge.bridge_session import attach_bridge_transport
    from utils.rust_env import episode as EP

    teams = packed_teams(cfg.get("teams", "pool"))
    threshold = int(cfg["turn_limit"]) if cfg.get("turn_limit") else EP.stall_threshold()
    args = production_args()
    kw = trainee_env_kwargs(args)
    kw["obs_source"] = "core"
    kw["stall_config"] = StallConfig(threshold=threshold)
    mappings = load_mappings()
    env = Gen3Env(mappings, battle_format="gen3ou",
                  team=SequenceTeambuilder([teams[e.teams[0]] for e in eps]),
                  opponent_team=SequenceTeambuilder([teams[e.teams[1]] for e in eps]),
                  reward_fn=Gen3RewardManager(config=RewardConfig.from_args(args)),
                  account_configuration1=AccountConfiguration("LEre", None), start_listening=False, **kw)
    session = attach_bridge_transport(env, battle_format="gen3ou", persistent=True, impl="rust", core_obs=True)
    pool = SnapshotPool(Path(cfg["pool_dir"]), current_version=_pool_version(Path(cfg["pool_dir"])), device="cpu",
                        lru_cache_size=64, compile_extractor=bool(cfg.get("compile")), compile_hide_cuda=True,
                        compile_strict=True)
    held = Path(cfg["pool_dir"]).parent / (Path(cfg["pool_dir"]).name + "_held")
    by_step = {e.step: e for e in pool._entries}
    Player = _replay_player_cls()
    sampled = cfg["mode"] == "sampled"
    keyed_seed = int(cfg["key_base"]) if cfg["mode"] == "keyed" else None
    out: Dict[str, Any] = {"decisions": 0, "phantom_polls": 0, "phantom_draws": 0, "no_decision_sent": 0,
                           "margins_greedy": [], "margins_sample": [], "max_dlogp": 0.0, "div": {}, "flips": [],
                           "episodes": 0, "p2_forfeits": 0, "p1_forfeits": 0, "models": sorted({e.model_id for e in eps})}
    w = None
    t0 = time.perf_counter()
    try:
        for k, ep in enumerate(eps):
            step = int(ep.model_id.split(":")[1])
            if step not in by_step:                          # a refresh-loaded snapshot (moved in later)
                pool._scan()
                by_step = {e.step: e for e in pool._entries}
            model = pool.load_model(by_step[step])
            rl = Player(model=model, team=None, battle_format="gen3ou", server_configuration=None, mappings=mappings,
                        account_configuration=AccountConfiguration(f"LEo{k % 1000}", None), start_listening=False,
                        stochastic=sampled or keyed_seed is not None, temperature=ep.temperature,
                        stall_config=StallConfig(threshold=threshold))
            rl.bind(env, ep, out, sampled)
            rl._keyed_seed = keyed_seed
            rl._keyed_key = None
            if keyed_seed is not None:
                rl._action_sampler = rl._keyed_sample
            if w is None:
                w = SingleAgentWrapper(env, rl)
                w.action_space, w.observation_space = env.action_space, env.observation_space
            else:
                w.opponent = rl
            session.seed = ep.seed
            _obs, _ = w.reset()
            j, total = 0, 0.0
            for _ in range(8000):
                if env.agent1_to_move:
                    if j >= len(ep.p1):
                        out["div"]["p1_extra"] = out["div"].get("p1_extra", 0) + 1
                        break
                    act = ep.p1[j]
                    j += 1
                else:
                    act = 0
                _obs, r, term, trunc, _i = w.step(act)
                total += float(r)
                if term or trunc:
                    break
            else:
                raise AssertionError(f"episode {k} did not end")
            b1 = env.battle1
            p2_ff = bool(b1.won) and rl._j == len(ep.p2) and int(b1.turn) >= threshold
            out["p2_forfeits"] += int(p2_ff)
            out["p1_forfeits"] += int(b1.won is False and int(b1.turn) >= threshold)
            if (round(total, 6), int(term), int(trunc)) != (round(ep.reward, 6), ep.terminated, ep.truncated):
                out["div"]["outcome"] = out["div"].get("outcome", 0) + 1
                out.setdefault("outcome_examples", []).append((ep.env, ep.episode, (total, term, trunc), (ep.reward, ep.terminated, ep.truncated)))
            if j != len(ep.p1):
                out["div"]["p1_count"] = out["div"].get("p1_count", 0) + 1
            if rl._j != len(ep.p2):
                out["div"]["p2_count"] = out["div"].get("p2_count", 0) + 1
                out.setdefault("p2_count_examples", []).append((ep.env, ep.episode, rl._j, len(ep.p2)))
            out["episodes"] += 1
    finally:
        if w is not None:
            w.close()
    out["replay_s"] = time.perf_counter() - t0
    out["compiled"] = bool(cfg.get("compile"))
    del held
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    return out


def run(cfg: Dict[str, Any], workdir: Optional[str] = None, mutate: Any = None
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Record here, replay in a CUDA-less child; returns (record summary, replay result). ``mutate``
    (teeth tests) edits the recording before the replay reads it."""
    wd = Path(workdir or tempfile.mkdtemp(prefix="laneE_gate_"))
    from agents.model.compile_cache import ensure_hermetic_cache
    ensure_hermetic_cache("rust_env_opponents_parity")      # K3: the declared cache, or a private one
    if "pool_dir" not in cfg:
        build_gate_pool(wd / "pool", cfg["pool"])
        cfg = dict(cfg, pool_dir=str(wd / "pool"))
    rec = record(cfg)
    if mutate is not None:
        mutate(rec)
    rp = wd / "record.pkl"
    with open(rp, "wb") as f:
        pickle.dump(rec, f)
    # K3: the replay child's compile cache is FRESH and DELETED when the child is done (it used to be
    # two `mkdtemp`s nobody removed — ~2 GB of them on tmpfs by 2026-09-30).
    from agents.model.compile_cache import scratch_root
    with tempfile.TemporaryDirectory(prefix="laneE_replay_cache_", dir=scratch_root()) as cache:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", GEN3AI_COMPILE_CACHE_DIR=cache,
                   TORCHINDUCTOR_CACHE_DIR=os.path.join(cache, "inductor"),
                   TRITON_CACHE_DIR=os.path.join(cache, "triton"))
        r = subprocess.run([sys.executable, "-m", "agents.training.rust_env_opponents_parity", "--replay",
                            str(rp), "--out", str(wd / "replay.json")], env=env, capture_output=True,
                           text=True, timeout=7200)
    if r.returncode != 0:
        raise RuntimeError(f"replay child failed ({r.returncode}):\n{r.stderr[-6000:]}")
    rep = json.loads((wd / "replay.json").read_text())
    summary = {k: v for k, v in rec.items() if k != "episodes"}
    summary["n_episodes"] = len(rec["episodes"])
    summary["p2_decisions"] = sum(len(e.p2) for e in rec["episodes"])
    import hashlib

    h = hashlib.sha256()
    for e in rec["episodes"]:
        h.update(repr((e.env, e.episode, e.teams, e.seed, e.route, e.model_id, e.p1)).encode())
        for d in e.p2:
            h.update(d.row)
            h.update(d.logp.tobytes())
            h.update(bytes([d.action & 0xFF]))
    summary["battles_sha"] = h.hexdigest()[:16]
    return summary, rep


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replay", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    ap.add_argument("--pool", default="fresh:2", help="fresh:K | run:<snapshots dir>:K")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--mode", choices=("greedy", "sampled", "keyed"), default="greedy")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--backend", default="eager", choices=("eager", "graph"))
    ap.add_argument("--buckets", default="2,8")
    ap.add_argument("--lanes", type=int, default=0)
    ap.add_argument("--turn-limit", type=int, default=0, help="0 = the production stall threshold")
    ap.add_argument("--compile", action="store_true", help="replay through --compile-opponents' compiled path")
    ap.add_argument("--refresh-at", type=int, default=0, help="the step of a mid-run pool refresh (a LOAD)")
    ap.add_argument("--key-base", type=int, default=71_000)
    ap.add_argument("--teams", default="pool")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if a.replay:
        replay(a.replay, a.out)
        return 0
    cfg = {"pool": a.pool, "n_envs": a.n_envs, "threads": a.threads, "episodes": a.episodes, "mode": a.mode,
           "device": a.device, "backend": a.backend, "buckets": [int(x) for x in a.buckets.split(",")],
           "lanes": a.lanes, "turn_limit": a.turn_limit, "compile": a.compile, "refresh_at": a.refresh_at,
           "key_base": a.key_base, "teams": a.teams}
    summary, rep = run(cfg, a.workdir)
    res = {"record": summary, "replay": rep, "near_ties_at_bar": {"bar": LOGP_BAR, **near_ties(rep, LOGP_BAR)}}
    print(json.dumps(res, indent=1, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))
    ties, fatal = judge_flips(rep, LOGP_BAR)
    return 0 if not rep["div"] and not fatal else 1


if __name__ == "__main__":
    raise SystemExit(main())
