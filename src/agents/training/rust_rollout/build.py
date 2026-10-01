"""The collector's STARTUP DECLARATION and its build (M5 Lane G; package docstring).

``RustEnvDecl`` is everything the steady state will ever use, named before anything starts (the M5
DECLARED LIFECYCLE): the env count and worker threads, the front end (``proc`` — the default for
training: a core fault must not take the learner's optimizer and GPU context with it — or ``ffi``),
the build profile, the trigger and its band (which sizes the row arena), the terminal / stall rules,
the refusal and respawn budgets, the inference backend, buckets and lanes, and version pinning.
``build_collector`` acquires it all and returns a ``RustCollector`` ready for ``start()``.

T2 SLOT GROUPS (F-LE-7): one group per ARCHITECTURE, in the route table's slot order, so a global T2
slot id IS the plan's slot id: the policy routes (pool, each stable, the exploiter — two slots under
``--exploiter-ladder``, F-LE-6) and then the trainee's slot(s); consecutive routes whose weights share
a state-dict signature and forward fingerprint share a group (and its compiled buckets), so the
production pool + trainee is ONE group. Every template holds REAL weights (the trainee's own policy;
a stable / exploiter checkpoint), as T2's parity gate asks.

BUCKETS (F-LE-9, decided 2026-09-29): ``(8, n_envs)`` by default — 8 serves an opponent slot's rows
(every per-slot count in Lane E's 600-step read at 48 envs / pool 20 was 1–6, 94 % were 1–4; padding is nearly free at
small batches, 1.06 → 1.23 ms from 2 → 8 rows), ``n_envs`` the trainee's batch (≤ N rows every step)
and any opponent slot with more than 8 rows (a small pool early in a run). Lane E's 2 / 4 / 16 are
dropped: 16 served nothing, and 2 / 4 each cost a compile (~75 s on 2.5.1) and a capture per slot for
a sub-0.2 ms saving. The SIZING study (order constraint 5) re-reads it at its N.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from agents.training.rust_rollout import store as S
from agents.training.rust_rollout.collector import CollectorConfig, RustCollector
from agents.training.rust_rollout.teams import TeamStager, TeamTable
from agents.training.rust_rollout.trigger import trigger_for

FRONTS = ("proc", "ffi")
PROFILES = ("release", "selfcheck")
NAMES = ("rgone", "rgtwo")


@dataclass(frozen=True)
class RustEnvDecl:
    n_envs: int
    threads: int = 8
    front: str = "proc"
    profile: str = "release"
    trigger: str = "complete_game"
    n_steps: int = 2048
    micro_batch: int = 2048
    target: int = 0
    band_lo: int = 0
    band_hi: int = 0
    gamma: float = 1.0
    gae_lambda: float = 0.8
    run_seed: int = 0
    turn_limit: int = 250
    terminal: Optional[Dict[str, Any]] = None
    decision_tense: bool = False
    switch_freeze: bool = False
    refusal_budget: int = 64
    respawn_budget: int = 2
    bank_dir: Optional[str] = None
    device: str = "cuda"
    backend: str = "graph"
    buckets: Tuple[int, ...] = ()
    lanes: int = 0
    version_pinning: bool = False
    trainee_slots: int = 1
    max_game_rows: int = 0
    op_timeout: Optional[float] = None
    opponent_sampling: str = "keyed"
    policy_seed: int = 0
    #: the fork arm's declaration (``fork.ForkDecl``; ``designs/training/forks.md`` §14). None = OFF —
    #: nothing of the arm is built, and the arena, the obs keys and the FIFO are exactly as without it.
    fork: Any = None

    def __post_init__(self) -> None:
        if self.front not in FRONTS:
            raise ValueError(f"RustEnvDecl.front {self.front!r} not in {FRONTS}")
        if self.profile not in PROFILES:
            raise ValueError(f"RustEnvDecl.profile {self.profile!r} not in {PROFILES}")
        if self.version_pinning and self.trainee_slots < 2:
            raise ValueError("version pinning needs >= 2 trainee slots")
        if self.opponent_sampling not in ("keyed", "generator"):
            raise ValueError(f"opponent_sampling {self.opponent_sampling!r}")
        if self.fork is not None:
            # §14.3 / §14.7: the branches replay the parent's draws by KEY, and join the complete-game FIFO
            if self.opponent_sampling != "keyed":
                raise ValueError("the fork arm needs --opponent-sampling keyed (a per-env generator stream "
                                 "cannot be replayed per branch)")
            if self.trigger != "complete_game":
                raise ValueError("the fork arm needs --rollout-trigger complete_game (branch games join the "
                                 "completed-game FIFO; the window fill is the parity schedule)")

    @property
    def resolved_buckets(self) -> Tuple[int, ...]:
        if self.buckets:
            return tuple(sorted(set(int(b) for b in self.buckets)))
        return tuple(sorted({8, max(8, int(self.n_envs))}))

    @property
    def resolved_max_game_rows(self) -> int:
        return int(self.max_game_rows) or (2 * int(self.turn_limit) + 16)

    def capacity(self, trigger: Any) -> int:
        n = int(self.n_envs)
        per_env = 2 * self.resolved_max_game_rows
        if trigger.mode == "window":
            return n * (int(self.n_steps) + 1) + n * per_env
        fork_rows = self.fork.row_budget(int(trigger.hi)) if self.fork is not None else 0
        return int(trigger.hi) + n * per_env + fork_rows


def core_label_families(obs_space: Any) -> Tuple[str, ...]:
    """The Lane-C label families the learner's observation space needs (a family is declared iff one
    of its keys is in the space; every core key in the space must be covered)."""
    from utils.rust_env import columns as C

    keys = set(obs_space.spaces)
    fams = tuple(f for f, ks in C.LABEL_FAMILIES.items() if keys & set(ks))
    covered = {k for f in fams for k in C.LABEL_FAMILIES[f]}
    missing = sorted(k for f, ks in C.LABEL_FAMILIES.items() for k in ks if k in keys and k not in covered)
    if missing:
        raise ValueError(f"label keys {missing} are core keys no declared family writes")
    return fams


def _arch_key(policy: Any) -> Tuple[Any, str]:
    from agents.inference.service.slots import forward_fingerprint, state_signature

    return state_signature(policy.state_dict()), forward_fingerprint(policy)


def slot_groups(plan: Any, templates: Mapping[int, Any], trainee_policy: Any, n_trainee: int
                ) -> Tuple[List[Tuple[str, int, Any]], List[int]]:
    """``([(group name, n_slots, template)], trainee global slots)`` — module docs. ``templates`` maps
    a policy route's slot to the policy whose architecture (and weights) that slot serves."""
    groups, trainee, _extra = slot_groups_with_extra(plan, templates, trainee_policy, n_trainee, ())
    return groups, trainee


def slot_groups_with_extra(plan: Any, templates: Mapping[int, Any], trainee_policy: Any, n_trainee: int,
                           extra: Sequence[Tuple[str, Any]]
                           ) -> Tuple[List[Tuple[str, int, Any]], List[int], List[int]]:
    """:func:`slot_groups` plus EXTRA slots declared after the trainee's (M5 Lane H: eval's trainee,
    sentinel and fixed-opponent slots), each ``(family, template)``; a run of the trainee's architecture
    joins the trainee's group (the same compiled buckets). Returns the extra slots' global ids too."""
    runs: List[List[Any]] = []            # [name, n, template, key]
    for r in plan.routes():
        if r.kind != "policy":
            continue
        pol = templates[int(r.slot)]
        key = _arch_key(pol)
        fam = r.family.split(":")[0]
        if runs and runs[-1][3] == key:
            runs[-1][1] += 1
        else:
            runs.append([f"{fam}{len(runs)}", 1, pol, key])
    tkey = _arch_key(trainee_policy)
    first_trainee = sum(r[1] for r in runs)
    if runs and runs[-1][3] == tkey:
        runs[-1][1] += n_trainee
    else:
        runs.append([f"trainee{len(runs)}", n_trainee, trainee_policy, tkey])
    first_extra = first_trainee + n_trainee
    for fam, pol in extra:
        key = tkey if pol is trainee_policy else _arch_key(pol)
        if runs[-1][3] == key:
            runs[-1][1] += 1
        else:
            runs.append([f"{fam}{len(runs)}", 1, pol, key])
    return ([(name, n, tpl) for name, n, tpl, _ in runs], list(range(first_trainee, first_trainee + n_trainee)),
            list(range(first_extra, first_extra + len(extra))))


def open_core(decl: RustEnvDecl, spec_json: str) -> Any:
    """The env core behind the declared front end (the process child by default)."""
    nan = decl.profile == "selfcheck"
    if decl.front == "ffi":
        from utils.rust_env import ffi

        return ffi.FfiCore(spec_json, lib=ffi.load(ffi.default_path(decl.profile), nan_poison=nan))
    from utils.rust_env import proc

    return proc.ProcCore(spec_json, binary=proc.default_path(decl.profile), nan_poison=nan, auto_respawn=True,
                         op_timeout=decl.op_timeout)


@dataclass
class OpponentSources:
    """Where every policy route's weights come from (the ``load`` callable resolves a model id)."""

    pool: Any = None                                      # SnapshotPool (self-play) or None
    stable: Dict[str, Any] = field(default_factory=dict)  # label -> loaded policy
    exploiter: Any = None                                 # loaded policy or None
    self_play_fraction: float = 0.0

    def policy_for(self, model_id: str) -> Any:
        kind, _, rest = model_id.partition(":")
        if kind == "pool":
            step = int(rest)
            entry = next(e for e in self.pool._entries if e.step == step)
            pol = self.pool.load_model(entry).policy.eval()
            on_card = sorted({str(p.device) for p in pol.parameters() if p.device.type != "cpu"})
            if on_card:
                from agents.training.learner_lifecycle import FATAL_TAG, LazyAcquisitionError
                raise LazyAcquisitionError(
                    f"{FATAL_TAG} — pool snapshot {model_id} was loaded onto {on_card}: a pool refresh is "
                    f"a DECLARED LOAD into its T2 slot, so the snapshot itself must stay on the CPU — a "
                    f"device copy is a new allocation at every promotion (gen3_declared_slot_load_v1)")
            return pol
        if kind == "stable":
            return self.stable[rest]
        if kind == "exploiter":
            return self.exploiter
        raise KeyError(model_id)


#: A route's weight refresh may leave at most this much NEW memory allocated on the card (the
#: allocator's granularity): it is a DECLARED LOAD into an existing T2 slot (gen3_declared_slot_load_v1).
SLOT_LOAD_ALLOC_TOLERANCE = 1 << 20


def _allocated(device: Any) -> int:
    import torch
    dev = torch.device(device)
    return int(torch.cuda.memory_allocated(dev)) if dev.type == "cuda" and torch.cuda.is_available() else 0


def checked_slot_load(svc: Any, slot: int, policy_fn: Callable[[], Any], model_id: str, device: Any,
                      allocated: Callable[[Any], int] = _allocated) -> int:
    """``svc.load(slot, policy_fn(), model_id)``, refusing a load that leaves memory allocated on the
    card (a typed `LazyAcquisitionError`, FATAL_CONFIG): a pool refresh is a declared load into an
    existing slot, never a new allocation. Returns the bytes it left allocated (<= the tolerance)."""
    before = allocated(device)
    svc.load(int(slot), policy_fn(), model_id)
    grew = allocated(device) - before
    if grew > SLOT_LOAD_ALLOC_TOLERANCE:
        from agents.training.learner_lifecycle import FATAL_TAG, LazyAcquisitionError
        raise LazyAcquisitionError(
            f"{FATAL_TAG} — loading {model_id} into T2 slot {slot} left {grew / (1 << 20):.1f} MiB NEWLY "
            f"allocated on {device}: a slot refresh is a DECLARED LOAD, never an acquisition "
            f"(gen3_declared_slot_load_v1)")
    return grew


def build_collector(decl: RustEnvDecl, *, obs_space: Any, trainee_policy: Any, plan: Any, sources: OpponentSources,
                    trainee_builder: Any, opponent_builder: Any, route_builders: Optional[Mapping[int, Any]] = None,
                    external_p2: Optional[Callable[..., np.ndarray]] = None, team_wr_tracking: bool = True,
                    emit: Callable[[str], None] = print, svc: Any = None,
                    extra_slots: Sequence[Tuple[str, Any]] = ()) -> RustCollector:
    """Acquire everything ``decl`` names and return the collector (``start()`` is the caller's). ``svc``: an
    already STARTED service with this plan's slot layout (a benchmark's arms share one; training builds
    its own)."""
    import time

    from agents.inference.service import InferenceService, ServiceSpec, SlotGroupSpec
    from agents.training import rust_env_opponents as E
    from utils.rust_env import episode as EP
    from utils.rust_env import protocol as P
    from utils.rust_env import columns as C

    n = int(decl.n_envs)
    fe = getattr(trainee_policy, "features_extractor", None)
    if fe is not None and "forward" in vars(fe):
        raise S.CollectorError(
            "the trainee policy's extractor carries an instance-level `forward` (--compile-trainer's patch): T2 "
            "deep-copies the policy as its slot templates, and a copied bound forward runs the ORIGINAL extractor "
            "— start the Rust env BEFORE the trainer compile (model_build._start_rust_env), or pass an uncompiled "
            "load of the checkpoint")
    trig = trigger_for(decl.trigger, n_envs=n, n_steps=decl.n_steps, micro_batch=decl.micro_batch,
                       target=decl.target, band_lo=decl.band_lo, band_hi=decl.band_hi)
    fams = core_label_families(obs_space)
    label_keys = tuple(k for f in fams for k in C.LABEL_FAMILIES[f])

    # ---- T2: one group per architecture, in slot order; the trainee last
    templates: Dict[int, Any] = {}
    for r in plan.routes():
        if r.kind != "policy":
            continue
        if r.family == "pool":
            templates[int(r.slot)] = trainee_policy
        elif r.family.startswith("stable:"):
            templates[int(r.slot)] = sources.stable[r.family.split(":", 1)[1]]
        else:
            templates[int(r.slot)] = sources.exploiter
    groups, trainee_slots, extra_ids = slot_groups_with_extra(plan, templates, trainee_policy,
                                                              int(decl.trainee_slots), extra_slots)
    n_slots = sum(g[1] for g in groups)
    lanes = int(decl.lanes) or (min(n_slots, 8) if str(decl.device).startswith("cuda") else 1)
    buckets = decl.resolved_buckets
    t0 = time.perf_counter()
    shared = svc is not None
    if shared and len(getattr(svc, "_slots", ())) != n_slots:
        raise S.CollectorError(f"the shared service declares {len(svc._slots)} slots; this plan needs {n_slots}")
    svc = svc if shared else InferenceService(ServiceSpec(
        groups=tuple(SlotGroupSpec(name, k, tpl) for name, k, tpl in groups), device=decl.device,
        backend=decl.backend, buckets=buckets, lanes=lanes,
        # T2's startup CONCURRENT gate puts one full chunk of the largest bucket per slot into ONE flush,
        # so the declared arena must hold n_slots x the largest bucket (F-LG-4: T2 does not check it).
        max_rows_per_flush=max(1024, 4 * n, buckets[-1] * 4, n_slots * buckets[-1]))).startup()
    emit(f"🦀 [RUST ENV] T2 {'SHARED' if shared else 'up'} in {time.perf_counter() - t0:.1f}s: {len(groups)} slot group(s) "
         f"{[(g[0], g[1]) for g in groups]}, trainee slot(s) {trainee_slots}, buckets {buckets}, lanes {lanes}, "
         f"backend {decl.backend} on {decl.device}")

    # ---- the opponents (Lane E)
    routes = plan.routes()

    def load(route: int, model_id: str) -> None:
        checked_slot_load(svc, int(routes[route].slot), lambda: sources.policy_for(model_id), model_id,
                          decl.device)

    stable_ids = [f"stable:{s.label}" for s in plan.stable]
    ex_id = "exploiter:target" if plan.exploiter is not None else None
    host = E.RustEnvOpponents(plan, n, load=load, pool=sources.pool, stable_ids=stable_ids, exploiter_id=ex_id,
                              self_play_fraction=float(sources.self_play_fraction),
                              pool_rng_seeds=[int(decl.run_seed) * 1000 + 5000 + i for i in range(n)],
                              rng_seeds=[int(decl.run_seed) * 1000 + i for i in range(n)])
    server = None
    if plan.n_policy_slots:
        server = E.PolicyOpponentServer(plan, svc, n, policy_seed=int(decl.policy_seed), seed_stride=1,
                                        run_seed=int(decl.run_seed), sampling=decl.opponent_sampling)

    # ---- teams + the core
    table = TeamTable()
    stager = TeamStager(table, n, trainee_builder=trainee_builder, opponent_builder=opponent_builder,
                        route_builders=route_builders, run_seed=int(decl.run_seed),
                        team_wr_tracking=team_wr_tracking)
    terminal = decl.terminal if decl.terminal is not None else EP.PRODUCTION_TERMINAL
    spec = P.spec_json(n=n, threads=int(decl.threads), teams=list(table.teams), names=NAMES,
                       decision_tense=bool(decl.decision_tense), switch_freeze=bool(decl.switch_freeze),
                       turn_limit=int(decl.turn_limit), refusal_budget=int(decl.refusal_budget),
                       bank_dir=decl.bank_dir, labels=fams, terminal=terminal,
                       opponents=plan.spec_rows(bots="external" if external_p2 is not None else "core"))
    t1 = time.perf_counter()
    core = open_core(decl, spec)
    emit(f"🦀 [RUST ENV] core up in {time.perf_counter() - t1:.1f}s: {n} envs x {decl.threads} threads "
         f"({decl.front} front end, {decl.profile} build), {len(table)} teams validated by use, label "
         f"families {list(fams)}, {len(routes)} routes")
    cfg = CollectorConfig(n_envs=n, trigger=trig, gamma=float(decl.gamma), gae_lambda=float(decl.gae_lambda),
                          run_seed=int(decl.run_seed), turn_limit=int(decl.turn_limit),
                          capacity=decl.capacity(trig), max_game_rows=decl.resolved_max_game_rows,
                          label_keys=label_keys, version_pinning=bool(decl.version_pinning),
                          respawn_budget=int(decl.respawn_budget),
                          victory_value=float(terminal["victory_value"]), fork=decl.fork is not None)
    col = RustCollector(cfg, core=core, obs_space=obs_space, svc=svc, trainee_slots=trainee_slots,
                        opponents=host, server=server, stager=stager, external_p2=external_p2)
    if decl.fork is not None:
        from agents.training.rust_rollout import fork as FK
        from utils.rust_env import ffi as F

        FK.check_terminal(terminal)
        FK.check_obs_keys(list(obs_space.spaces))
        col.fork = FK.RustForkPass(decl.fork, lib_path=F.default_path(decl.profile), nan_poison=decl.profile == "selfcheck",
                                   turn_limit=int(decl.turn_limit), victory_value=float(terminal["victory_value"]),
                                   target_hi=int(trig.hi), max_rows_per_flush=int(svc.spec.max_rows_per_flush),
                                   obs_dim=int(obs_space.spaces[S.KEY_OBSERVATION].shape[0]), emit=emit)
    col.decl = decl
    col.extra_slots = extra_ids      # M5 Lane H: the declared eval slots (``extra_slots``' order)
    emit(f"🦀 [RUST ENV] collector: {trig.describe()}; arena {cfg.capacity:,} rows; version pinning "
         f"{'ON' if decl.version_pinning else 'off'}; opponent sampling {decl.opponent_sampling}")
    return col


def trainee_spaces(args: Any, mappings: Any = None) -> Tuple[Any, Any]:
    """``(observation_space, action_space)`` of the trainee ``Gen3Env`` this run's args build — the
    SAME surface (``env_factory.trainee_env_kwargs``) the Python path's workers declare, so a checkpoint
    loads identically on either env core. Built from a ``Gen3Env`` that never connects."""
    from poke_env import AccountConfiguration

    from agents.observation.state_encoder import load_mappings
    from agents.training.gen3_env import Gen3Env
    from agents.training.reward_config import RewardConfig
    from agents.training.reward_manager import Gen3RewardManager
    from main.train.env_factory import trainee_env_kwargs

    env = Gen3Env(mappings if mappings is not None else load_mappings(), battle_format="gen3ou",
                  reward_fn=Gen3RewardManager(config=RewardConfig.from_args(args)),
                  account_configuration1=AccountConfiguration("rgspaces", None), start_listening=False,
                  **trainee_env_kwargs(args))
    try:
        return env.observation_space, env.action_space
    finally:
        env.close()
