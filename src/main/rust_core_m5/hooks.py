"""The THROUGHPUT A/B's typed HOOKS (M5 Lane J, ``designs/endstate/program_rust_core.md`` §2 M5).

The A/B (``throughput.py``) steps today's Python rollout path and the Rust env core at matched N.
Everything that is NOT the env — who picks the trainee's action, who the opponent is, how a rollout
is cut — is a hook, so the A/B isolates the env and a lane that lands later plugs its piece in
without touching the harness:

* :class:`TraineeInference` — ``(obs [n, OBS_DIM] f32, masks [n, 11]) -> actions [n] int``.
  BUILT: :class:`RandomLegal` (CPU, the default) and :class:`T2Inference` (greedy argmax over the
  legal log-probs of a T2 ``InferenceService``). Both arms call the SAME hook instance type.
* :class:`OpponentMix` — BUILT: :class:`UniformRandom` (python: poke-env's ``RandomPlayer``; rust:
  the in-core bot ``random``). DECLARED, NOT BUILT: :class:`ProductionMix` (Lane E + Lane G).
* :class:`Collector` — BUILT: :class:`StepCollector` (today's fixed window). DECLARED, NOT BUILT:
  :class:`CompleteGameCollector` (Lane G, order constraint 6).

A NOT-BUILT hook RAISES :class:`HookNotBuilt` at construction, naming exactly what must be plugged
in. It never falls back to a built one: a throughput number measured against a silently different
opponent or collector is a wrong number with a right-looking label.

Every hook has a ``name`` the A/B records in its JSON regime.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

if TYPE_CHECKING:  # numpy is imported lazily at call time: the env workers import this module
    import numpy as np


class HookNotBuilt(NotImplementedError):
    """A DECLARED hook whose implementation has not landed. ``owner`` is the lane that builds it,
    ``needs`` the inputs it must be constructed from — the message is the plug-in instruction."""

    def __init__(self, hook: str, owner: str, needs: Sequence[str], why: str):
        self.hook, self.owner, self.needs, self.why = hook, owner, tuple(needs), why
        super().__init__(f"{hook} is DECLARED, NOT BUILT (owner: {owner}). {why} It must be constructed "
                         f"from: {'; '.join(self.needs)}. The throughput A/B never substitutes a built hook "
                         "for it.")


# ---------------------------------------------------------------------------------------------
# 1. the trainee's inference
# ---------------------------------------------------------------------------------------------

@runtime_checkable
class TraineeInference(Protocol):
    """The trainee's action for a batch of decision rows. Rows are only ever ones with at least one
    legal action; the result is an int array of legal action indices, one per row."""

    name: str

    def __call__(self, obs: "np.ndarray", masks: "np.ndarray") -> "np.ndarray": ...

    def describe(self) -> Dict[str, Any]: ...


class RandomLegal:
    """A uniformly random LEGAL action per row, from one seeded generator (CPU, near-zero cost —
    the default, so the A/B reads the ENV, not a network)."""

    name = "random_legal"

    def __init__(self, seed: int = 0):
        import numpy as np

        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, obs: "np.ndarray", masks: "np.ndarray") -> "np.ndarray":
        import numpy as np

        m = np.asarray(masks).astype(bool)
        if m.ndim != 2 or not m.any(axis=1).all():
            raise ValueError("RandomLegal: every row needs at least one legal action")
        # uniform over each row's legal set: argmax of iid uniforms restricted to the legal cells
        u = self._rng.random(m.shape)
        u[~m] = -1.0
        return u.argmax(axis=1).astype(np.int64)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "seed": self.seed, "device": "cpu"}


class T2Inference:
    """Greedy argmax over the LEGAL log-probs of one T2 ``InferenceService`` slot (M5 Lane T2).

    ``ckpt``: a checkpoint ``.zip`` path, a ``designs/baselines.json`` registry name (e.g.
    ``production``), or ``None`` for T2's own seeded perturbed fresh policy (its parity fixture's
    policy — a non-vacuous forward at production shape, no checkpoint needed). ``backend``: T2's
    (``eager`` on CPU; ``graph`` / ``aot`` are CUDA only). The service is STARTED here (the declared
    lifecycle: every slot x bucket compiled and parity-gated, then frozen) — construction is the
    startup cost, reported apart from the A/B's timed blocks. ``seconds`` accumulates the time spent
    inside calls, so the A/B can report inference separately from the env.
    """

    name = "t2_greedy"

    def __init__(self, ckpt: Optional[str] = None, backend: str = "eager", device: str = "cpu",
                 buckets: Tuple[int, ...] = (8, 48, 128), max_rows_per_flush: int = 1024):
        from agents.inference.service import InferenceService, ServiceSpec, SlotGroupSpec

        self.ckpt, self.backend, self.device = ckpt, backend, device
        self.buckets = tuple(int(b) for b in buckets)
        t0 = time.perf_counter()
        policy = _load_policy(ckpt)
        self.load_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        self.svc = InferenceService(ServiceSpec(
            groups=(SlotGroupSpec("trainee", 1, policy),), device=device, backend=backend,
            buckets=self.buckets, max_rows_per_flush=max(int(max_rows_per_flush), self.buckets[-1]))).startup()
        self.startup_s = time.perf_counter() - t1
        self.slot = self.svc.slot("trainee", 0)
        self.calls = 0
        self.rows = 0

    def __call__(self, obs: "np.ndarray", masks: "np.ndarray") -> "np.ndarray":
        import numpy as np

        t = self.svc.submit(self.slot, obs, np.asarray(masks).astype(bool))
        self.svc.flush()
        if not t.done:
            self.svc.drain()
        _logp, _value, greedy = t.host()
        self.calls += 1
        self.rows += int(t.n)
        return np.array(greedy, dtype=np.int64, copy=True)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "ckpt": self.ckpt or "perturbed_fresh_policy(0)", "backend": self.backend,
                "device": self.device, "buckets": list(self.buckets), "load_s": round(self.load_s, 2),
                "startup_s": round(self.startup_s, 2),
                "startup_phases_s": {k: round(v, 2) for k, v in self.svc.startup_seconds.items()},
                "after_freeze": {k: self.svc.counters[k] for k in
                                 ("compiles_after_freeze", "captures_after_freeze", "cuda_segments_after_freeze")}}


def _load_policy(ckpt: Optional[str]) -> Any:
    """The policy a :class:`T2Inference` serves: a registry name, a ``.zip`` path, or ``None``."""
    if ckpt is None:
        from agents.inference.service.fixtures import perturbed_fresh_policy

        return perturbed_fresh_policy(0)
    if not ckpt.endswith(".zip"):
        from agents.training import baselines

        return baselines.load(ckpt, device="cpu").policy.eval()
    from agents.model.snapshot import current_model_version, load_foreign_opponent
    from agents.observation.state_encoder import load_mappings

    model, _ = load_foreign_opponent(ckpt, current_version=current_model_version(load_mappings()), device="cpu")
    return model.policy.eval()


# ---------------------------------------------------------------------------------------------
# 2. the opponent
# ---------------------------------------------------------------------------------------------

@runtime_checkable
class OpponentMix(Protocol):
    """Who p2 is, on BOTH arms. ``python_opponent`` runs INSIDE a SubprocVecEnv worker (the hook
    object is pickled to it), ``rust_routes`` is the core spec's ``opponents`` route table and
    ``ep_opp`` the route index staged for an env's next episode."""

    name: str

    def python_opponent(self, idx: int, tag: str, teambuilder: Any) -> Any: ...

    def rust_routes(self) -> List[Dict[str, Any]]: ...

    def ep_opp(self, env: int) -> int: ...

    def describe(self) -> Dict[str, Any]: ...


@dataclass
class UniformRandom:
    """A uniformly random legal opponent. Python: poke-env's ``RandomPlayer`` (what the soak and
    slice-N envs play); rust: the in-core Lane-F bot ``random`` on ONE route, seeded (the core
    derives each env's streams from ``seed``, ``opponents.rs::stream_seed``)."""

    seed: int = 20260929
    name: str = field(default="uniform_random", init=False)

    def python_opponent(self, idx: int, tag: str, teambuilder: Any) -> Any:
        from poke_env import AccountConfiguration
        from poke_env.player import RandomPlayer

        return RandomPlayer(battle_format="gen3ou", team=teambuilder,
                            account_configuration=AccountConfiguration(f"{tag}o{idx}"[:18], None),
                            start_listening=False)

    def rust_routes(self) -> List[Dict[str, Any]]:
        return [{"kind": "bot", "bot": "random", "seed": int(self.seed)}]

    def ep_opp(self, env: int) -> int:
        return 0

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "python": "poke_env.player.RandomPlayer",
                "rust": self.rust_routes()}


class ProductionMix:
    """DECLARED, NOT BUILT — the PRODUCTION opponent mixture on both arms.

    The Rust side exists (Lane E: ``agents.training.rust_env_opponents.RustEnvOpponents`` +
    ``PolicyOpponentServer`` over T2 slots, bots in the core); the Python side is today's
    ``main.train.env_factory.create_training_env_random`` (``MaskableAgentWrapper`` with the bot
    roster, the snapshot pool, stable / exploiter opponents). What is missing is the harness-side
    wiring that builds BOTH from ONE resolved args namespace and ONE pool, which is Lane G's
    rollout host (it owns the per-step ``after_op`` call inside the collector)."""

    name = "production_mix"
    NEEDS = (
        "the resolved training args namespace (main.rust_core_cutover.envs.production_args() or the run's own)",
        "a snapshot pool directory (SnapshotPool; the model_config.json of the pool's version)",
        "a started T2 InferenceService with the pool's policy slots (OpponentPlan.n_policy_slots)",
        "Lane G unit 1's build_collector (agents.training.rust_rollout.build) with an OpponentSources over the "
        "pool — it already does the ONE-flush serve of p2 + trainee rows; on the Rust arm it replaces this hook",
        "Lane E's RustEnvOpponents host, whose after_op(cols) must run after EVERY core op (F-LE-4 re-staging) "
        "and PolicyOpponentServer.serve(cols) before every step",
        "on the Python arm, create_training_env_random's kwargs (OPPONENT_CLASSES, stable_opponents, "
        "exploiter_entry, heuristic_weights, the self-play fraction) from the SAME args",
    )

    def __init__(self, *args: Any, **kwargs: Any):
        raise HookNotBuilt("OpponentMix 'production_mix'", "Lane G (rollout host) over Lane E's routing",
                           self.NEEDS, "Both arms must face the SAME production mixture drawn from one "
                           "declared plan, and nothing builds that pair yet.")


# ---------------------------------------------------------------------------------------------
# 3. the collector
# ---------------------------------------------------------------------------------------------

@runtime_checkable
class Collector(Protocol):
    """How a rollout is CUT. ``observe`` is called once per vec step with the per-env ``done``
    flags; ``ready`` says the rollout is complete (the learner would train now)."""

    name: str

    def observe(self, dones: "np.ndarray") -> None: ...

    def ready(self) -> bool: ...

    def describe(self) -> Dict[str, Any]: ...


class StepCollector:
    """Today's collector: a FIXED WINDOW of ``n_steps`` vec steps per rollout, games continuing
    across windows. The A/B only steps — it counts windows, it trains nothing."""

    name = "fixed_window_steps"

    def __init__(self, n_steps: int = 2048):
        if int(n_steps) < 1:
            raise ValueError("StepCollector: n_steps must be >= 1")
        self.n_steps = int(n_steps)
        self.steps = 0
        self.windows = 0
        self.episodes_ended = 0

    def observe(self, dones: "np.ndarray") -> None:
        import numpy as np

        self.steps += 1
        self.episodes_ended += int(np.count_nonzero(dones))
        if self.steps % self.n_steps == 0:
            self.windows += 1

    def ready(self) -> bool:
        return self.steps > 0 and self.steps % self.n_steps == 0

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "n_steps": self.n_steps, "steps": self.steps, "windows": self.windows,
                "episodes_ended": self.episodes_ended}


class CompleteGameCollector:
    """DECLARED, NOT BUILT — Lane G's COMPLETE-GAME BUFFER + SAMPLE-COUNT TRIGGER (program doc §2 M5,
    order constraint 6, owner 2026-09-29): a rollout is complete GAMES, cut when the buffer holds a
    declared sample count, so the win-prob critic never drops the rows of a game still running at
    a window edge."""

    name = "complete_game"
    NEEDS = (
        "Lane G unit 1's collector (BUILT 2026-09-29, not yet wired): agents.training.rust_rollout.build."
        "build_collector(RustEnvDecl, obs_space=, trainee_policy=, plan=, sources=OpponentSources) -> RustCollector "
        "— the A/B steps IT instead of stepping the core itself",
        "the declared SAMPLE-COUNT trigger (rows per rollout; a startup input per the SIZING study, constraint 5)",
        "a complete-game buffer keyed by (env, episode) that back-fills each row's outcome when its game ends",
        "the rule for games still running at the trigger (carried into the next rollout, never dropped)",
        "its gate: slice N at the ROLLOUT level with the trigger set to reproduce today's fixed window",
    )

    def __init__(self, *args: Any, **kwargs: Any):
        raise HookNotBuilt("Collector 'complete_game'", "Lane G (order constraint 6)", self.NEEDS,
                           "The fixed-window StepCollector is today's shape, not this one.")


#: The CLI's names -> constructors (a NOT-BUILT name raises on construction, never falls back).
INFERENCE = {"random": RandomLegal, "t2": T2Inference}
OPPONENTS = {"uniform_random": UniformRandom, "production": ProductionMix}
COLLECTORS = {"step": StepCollector, "complete_game": CompleteGameCollector}
