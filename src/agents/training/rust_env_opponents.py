"""OPPONENT ROUTING for the Rust env core (M5 Lane E): per-EPISODE opponents in, POLICY opponents'
rows out to the inference service (T2), actions back in.

Design of record: ``designs/endstate/program_rust_core.md`` §2 M5 (the Lane E row and paragraph);
progress: ``designs/research_state/measurements/m5_laneE/PROGRESS.md``. The core half is
``src/rust_env/src/opponents.rs``: a ROUTE TABLE declared at startup (the spec's ``opponents``), a
caller-staged ``ep_opp`` column read WITH the teams at every episode start, and two outputs,
``opp_route`` / ``opp_slot``, describing the episode the other columns describe.

FOUR PIECES, each testable alone:

* :class:`OpponentPlan` — the DECLARATION: which opponents a run can ever face and the route table
  they become. Route order: the POOL's slots (``pool_slots`` = the pool's ``max_snapshots`` + a
  declared spare, see :class:`SlotFamily`), then one slot per STABLE opponent, then the EXPLOITER
  target's slots, then one route per scripted BOT (Lane F's bots, played INSIDE the core with a
  declared seed rule — :func:`bot_stream_seed`; their decisions are never exposed).
* :class:`EpisodeOpponentSampler` — one env's per-episode draw, **rule for rule
  the draw of the deleted Python wrapper (``MaskableAgentWrapper._select_episode_opponent``, U3)** (exploiter / keep-bots / the self-play coin /
  the challenge bucket with its capped stable share and stable PFSP / the floor bucket with
  ``--bot-weights`` and mastered stables / one pool snapshot per env per GENERATION through
  ``SnapshotPool.sample``'s recency x PFSP weights), from the SAME per-env ``random.Random(idx)``
  stream, returning a ROUTE instead of a player. Pinned against the real wrapper draw for draw
  (``rust_env_opponents_test.py``).
* :class:`SlotFamily` — the slots one opponent kind owns and the model each holds. A pool refresh
  between rollouts is a DECLARED LOAD into a FREE slot (T2 ``load``: an in-place copy into
  captured storage, parity-verified — never a compile); a slot is free only when no env's current
  OR staged episode names it, so no episode is ever played against two opponents (today's
  "swap only at reset" rule). Running out of free slots is a typed :class:`SlotCapacityExceeded`,
  never a silent reuse.
* :class:`PolicyOpponentServer` — the T2 adapter: every env whose p2 needs an action on a POLICY
  route has its row submitted to that route's slot (grouped by ``opp_slot``), ONE flush, then the
  action is the argmax (greedy) or a SAMPLE at the route's temperature, written into
  ``action[:, 1]``. :func:`sample_actions` is today's ``RLPlayer`` sampler
  (``torch.multinomial(Categorical(logits=x / T).probs, 1, True, generator=g)``) written out: its
  CPU one-sample path is ``argmax(p / q)`` with ``q ~ Exp(1)`` drawn from ``g`` (Gumbel), so the
  same generator state gives the SAME action bit for bit on the same probabilities
  (``rust_env_opponents_test.py`` pins it against ``torch.multinomial``), one generator per env per
  opponent PLAYER (pool / each stable / exploiter — the ``RLPlayer`` objects a worker holds today).

:class:`RustEnvOpponents` puts them together for a host loop (Lane G wires it into training): it
stages ``ep_opp`` one episode ahead, re-stages every env whose ``episode`` moved (an auto-reset, a
quarantine, a refused start — the core consumed the staged row), and takes the live pushes the
wrapper takes today (``set_self_play_target`` … ``set_stable_win_rates``).

WHAT DIFFERS FROM TODAY, by construction (each stated in the program doc):

1. A draw happens ONE EPISODE EARLIER (the core auto-resets inside STEP, so the next episode's
   opponent is staged while the current one plays — exactly as ``ep_team`` is). The ORDER of draws
   per env is unchanged, so with no live push the route sequence equals the wrapper's; a live push
   (a new self-play fraction, a generation) reaches an env's draw one episode later than today.
2. The pool is SCANNED ONCE per generation by the host (``refresh``), not by each worker at its own
   reset — every env draws from the same roster, and a snapshot written mid-generation waits for
   the next push.
3. Every stream is SEEDED: the per-env pool draw stream (today the process-global ``random`` unless
   ``$GEN3AI_POOL_SEED``) and the sampling generators (today torch's shared default generator unless
   ``$GEN3AI_POLICY_SEED``). Today's default is unreproducible; this path is reproducible by
   declaration.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from agents.training.snapshot_pool import DEFAULT_MAX_SNAPSHOTS
from agents.training.opponent_classes import (  # noqa: F401 — re-exported (collector, tests)
    OPP_CLASS_BOT, OPP_CLASS_EXPLOITER, OPP_CLASS_POOL, OPP_CLASS_STABLE, STABLE_CHALLENGE_SHARE)

ROUTE_KINDS = ("external", "policy", "bot")

#: The opponent classes (``gen3_opp_class_v1``, imported above from ``opponent_classes``) — the
#: ``opp_class`` label is the class of the episode's route (``Route.klass``).

#: Spare pool slots beyond ``max_snapshots``: a snapshot EVICTED at a refresh stays resident until no
#: current or staged episode names it (at most ~2 episodes); one promotion evicts one snapshot, so a
#: small spare absorbs it. Exhausting it is a typed failure, never a silent reuse.
DEFAULT_POOL_SPARE = 4


class OpponentRoutingError(RuntimeError):
    """Base of every typed Lane-E failure."""


class SlotCapacityExceeded(OpponentRoutingError):
    """A LOAD found no free slot: every slot of the family holds a roster model or a model a current or
    staged episode still plays. The declared capacity is too small for the refresh rate."""


# ------------------------------------------------------------------------------------ the plan


@dataclass(frozen=True)
class StableSpec:
    """One stable cross-run opponent (``fixed_opponent_pool`` entry): its label, sampling
    temperature and pinned team strings (empty = pilots the shared pool builder)."""

    label: str
    temperature: float = 1.0
    team_strs: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ExploiterSpec:
    """The ``--exploiter`` target: temperature (live-mutable), keep-bots mix and team pin. It owns ONE
    T2 slot."""

    temperature: float = 1.0
    keep_bots: bool = False
    bot_fraction: float = 0.5
    team_strs: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Route:
    """One row of the route table, with what the HOST needs to know about it."""

    index: int
    kind: str                      # ROUTE_KINDS
    family: str                    # "pool" | "stable:<label>" | "exploiter" | "bot"
    player: str                    # the RLPlayer a worker holds for it today (its sampling stream)
    klass: int                     # OPP_CLASS_*
    slot: Optional[int] = None     # the T2 global slot (policy routes)
    temperature: Optional[float] = None
    stochastic: bool = True
    bot: Optional[str] = None
    team_strs: Tuple[str, ...] = ()


@dataclass(frozen=True)
class OpponentPlan:
    """What a run can ever face (see the module docs). ``pool_slots`` 0 = no self-play pool."""

    pool_slots: int = 0
    stable: Tuple[StableSpec, ...] = ()
    exploiter: Optional[ExploiterSpec] = None
    bots: Tuple[str, ...] = ()
    bot_weights: Optional[Tuple[float, ...]] = None
    self_play_temp: float = 1.0
    stable_challenge_share: float = STABLE_CHALLENGE_SHARE
    stable_pfsp: bool = False
    #: The bots' declared seed (F-LF-3: today's production bots are unseeded) and BaitBot's dial
    #: (``--bait-bot-p``).
    bot_seed: int = 0
    bait_bot_p: float = 0.6

    def __post_init__(self) -> None:
        if self.pool_slots < 0:
            raise ValueError("pool_slots must be >= 0")
        if not self.bots and self.exploiter is None:
            # The wrapper refuses an empty roster; the floor bucket is the fallthrough of every draw.
            raise ValueError("OpponentPlan: the floor bucket needs at least one bot (the wrapper's rule)")
        if self.bot_weights is not None:
            w = [float(x) for x in self.bot_weights]
            if len(w) != len(self.bots) or any(x < 0 for x in w) or sum(w) <= 0:
                raise ValueError("bot_weights must align with bots, be non-negative and sum > 0")
        labels = [s.label for s in self.stable]
        if len(set(labels)) != len(labels):
            raise ValueError(f"duplicate stable labels {labels}")

    @property
    def exploiter_slots(self) -> int:
        return 0 if self.exploiter is None else 1

    @property
    def n_policy_slots(self) -> int:
        return self.pool_slots + len(self.stable) + self.exploiter_slots

    def routes(self) -> Tuple[Route, ...]:
        out: List[Route] = []
        s = 0

        def add(**kw: Any) -> None:
            out.append(Route(index=len(out), **kw))

        for _ in range(self.pool_slots):
            add(kind="policy", family="pool", player="pool", klass=OPP_CLASS_POOL, slot=s,
                temperature=self.self_play_temp)
            s += 1
        for st in self.stable:
            add(kind="policy", family=f"stable:{st.label}", player=f"stable:{st.label}",
                klass=OPP_CLASS_STABLE, slot=s, temperature=st.temperature, team_strs=st.team_strs)
            s += 1
        if self.exploiter is not None:
            add(kind="policy", family="exploiter", player="exploiter", klass=OPP_CLASS_EXPLOITER,
                slot=s, temperature=self.exploiter.temperature, team_strs=self.exploiter.team_strs)
            s += 1
        for b in self.bots:
            add(kind="bot", family="bot", player=f"bot:{b}", klass=OPP_CLASS_BOT, bot=b)
        return tuple(out)

    def spec_rows(self, *, bots: str = "core") -> List[Dict[str, Any]]:
        """The spec's ``opponents`` array. ``bots="core"`` declares each bot route as a Lane-F bot the
        core plays (seed ``bot_seed + route index``, BaitBot's ``p_bait``); ``"external"`` (harnesses
        only) declares them external, i.e. the CALLER answers p2."""
        if bots not in ("core", "external"):
            raise ValueError(bots)
        rows: List[Dict[str, Any]] = []
        for r in self.routes():
            if r.kind == "policy":
                rows.append({"kind": "policy", "slot": int(r.slot)})
            elif bots == "external":
                rows.append({"kind": "external"})
            else:
                row: Dict[str, Any] = {"kind": "bot", "bot": r.bot, "seed": int(self.bot_seed) + r.index}
                if r.bot == "baitbot":
                    row["p_bait"] = float(self.bait_bot_p)
                rows.append(row)
        return rows

    @classmethod
    def from_args(cls, args: Any, *, bot_names: Sequence[str], stable_entries: Sequence[Any] = (),
                  exploiter_entry: Any = None, max_snapshots: int = DEFAULT_MAX_SNAPSHOTS,
                  pool_spare: int = DEFAULT_POOL_SPARE, heuristic_weights: Optional[Sequence[float]] = None,
                  bot_seed: int = 0) -> "OpponentPlan":
        """The plan ``env_factory.create_training_env_random`` builds per worker today, from the SAME
        inputs: ``args`` (``self_play``, ``self_play_temp``, ``stable_opponent_selfplay_share``,
        ``stable_opponent_pfsp``, ``exploiter_keep_bots``, ``exploiter_bot_fraction``), the resolved stable entries and exploiter entry (``label`` /
        ``temperature`` / ``team_strs``), the floor roster's names and ``--bot-weights``.
        ``max_snapshots`` is the pool's window (``SnapshotPool``'s ``DEFAULT_MAX_SNAPSHOTS`` in production)."""
        self_play = bool(getattr(args, "self_play", False))
        ex = None
        if exploiter_entry is not None:
            ex = ExploiterSpec(temperature=float(exploiter_entry.temperature),
                               keep_bots=bool(getattr(args, "exploiter_keep_bots", False)),
                               bot_fraction=float(getattr(args, "exploiter_bot_fraction", 0.5)),
                               team_strs=tuple(getattr(exploiter_entry, "team_strs", ()) or ()))
        return cls(
            pool_slots=(int(max_snapshots) + int(pool_spare)) if self_play else 0,
            stable=tuple(StableSpec(e.label, float(e.temperature), tuple(getattr(e, "team_strs", ()) or ()))
                         for e in (stable_entries if self_play else ())),
            exploiter=ex, bots=tuple(bot_names),
            bot_weights=None if heuristic_weights is None else tuple(float(x) for x in heuristic_weights),
            self_play_temp=float(getattr(args, "self_play_temp", 1.0)),
            stable_challenge_share=float(getattr(args, "stable_opponent_selfplay_share", STABLE_CHALLENGE_SHARE)),
            stable_pfsp=bool(getattr(args, "stable_opponent_pfsp", False)),
            bot_seed=int(bot_seed), bait_bot_p=float(getattr(args, "bait_bot_p", 0.6)))


def _splitmix64(z: int) -> int:
    m = 0xFFFFFFFFFFFFFFFF
    z = (z + 0x9E3779B97F4A7C15) & m
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & m
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & m
    return z ^ (z >> 31)


def bot_stream_seed(seed: int, env: int, stream: int) -> int:
    """``opponents::stream_seed`` — env ``env``'s bot stream ``stream`` (choice 0, protect 1, bait 2)
    is ``random.Random(bot_stream_seed(route seed, env, stream))`` (the declared seed rule, F-LF-3)."""
    return _splitmix64(int(seed) ^ _splitmix64(((int(env) << 2) | int(stream)) & 0xFFFFFFFFFFFFFFFF))


def pack_seed_words(words: Sequence[int]) -> int:
    """``opponents::pack_seed_words`` — an episode's four 16-bit battle-seed words, low word first."""
    return sum((int(w) & 0xFFFF) << (16 * i) for i, w in enumerate(list(words)[:4]))


def episode_bot_stream_seed(seed: int, words: Sequence[int], stream: int) -> int:
    """``opponents::episode_stream_seed`` — stream ``stream`` (choice 0, protect 1, bait 2) of a bot route
    declared ``"streams": "episode"`` (M5 Lane H), for the episode staged with battle seed ``words``: the
    bot's streams are ``random.Random(episode_bot_stream_seed(route seed, words, stream))``, re-seeded at
    every episode start, so a game's bot draws are a function of the game alone."""
    return bot_stream_seed(int(seed) ^ pack_seed_words(words), 0, stream)


# ------------------------------------------------------------------------------------ slots


class SlotFamily:
    """The routes (= T2 slots) one opponent family owns, the model id each holds, and the family's
    ROSTER (the model ids a draw may pick). ``resident[k]`` is route ``routes[k]``'s model id.

    ``admit(model_ids, in_use, load)`` makes every roster model resident: a model already resident
    keeps its slot; a new one is LOADED into a free slot (``load(route, model_id)``); a slot is free
    iff its model is not on the new roster AND its route is not in ``in_use`` (every env's current
    and staged route). No free slot → :class:`SlotCapacityExceeded`."""

    def __init__(self, name: str, routes: Sequence[int]):
        self.name = name
        self.routes = list(routes)
        self.resident: List[Optional[str]] = [None] * len(self.routes)
        self.roster: List[str] = []
        self.loads = 0

    def route_of(self, model_id: str) -> int:
        try:
            return self.routes[self.resident.index(model_id)]
        except ValueError:
            raise OpponentRoutingError(f"{self.name}: model {model_id!r} is not resident") from None

    def admit(self, model_ids: Sequence[str], in_use: "set[int]", load: Callable[[int, str], None]) -> List[int]:
        model_ids = list(model_ids)
        loaded: List[int] = []
        for mid in model_ids:
            if mid in self.resident:
                continue
            free = [k for k, r in enumerate(self.routes)
                    if self.resident[k] not in model_ids and r not in in_use]
            if not free:
                raise SlotCapacityExceeded(
                    f"{self.name}: no free slot for {mid!r} — {len(self.routes)} slots hold the roster "
                    f"{len(model_ids)} plus models still played by a current or staged episode "
                    f"({sorted(r for r in self.routes if r in in_use)}); declare a larger spare")
            k = free[0]
            load(self.routes[k], mid)
            self.resident[k] = mid
            self.loads += 1
            loaded.append(self.routes[k])
        self.roster = model_ids
        return loaded


# ------------------------------------------------------------------------------------ the sampler


class PoolRoster:
    """The self-play pool as the host sees it: ONE ``SnapshotPool`` (weights = recency x PFSP),
    rescanned at a refresh, sampled per env with that env's own stream (``SnapshotPool.sample(rng)``).
    A model id is ``f"pool:{step}"``."""

    def __init__(self, pool: Any):
        self.pool = pool

    def is_empty(self) -> bool:
        return bool(self.pool.is_empty())

    def model_ids(self) -> List[str]:
        return [f"pool:{s}" for s in self.pool.steps()]

    def sample(self, rng: random.Random) -> str:
        return f"pool:{self.pool.sample(rng=rng).step}"


class EpisodeOpponentSampler:
    """One env's per-episode opponent draw — the deleted Python wrapper's ``_select_episode_opponent`` rule for
    rule (see the module docs), returning a ROUTE index. ``rng_seed`` is the wrapper's (``idx``);
    ``pool_rng_seed`` the per-env pool stream (``SnapshotPool(rng_seed=…)``'s)."""

    def __init__(self, plan: OpponentPlan, families: Mapping[str, SlotFamily], *, rng_seed: int,
                 pool: Optional[PoolRoster], pool_rng_seed: int, self_play_fraction: float = 0.0):
        self.plan = plan
        self._routes = plan.routes()
        self._families = families
        self._rng = random.Random(rng_seed)
        self._pool = pool if plan.pool_slots else None
        self._pool_rng = random.Random(pool_rng_seed)
        self._self_play_fraction = float(self_play_fraction)
        self._target_generation = 0
        self._scanned_generation = -1
        self._has_pool_model = False
        self._gen_model: Optional[str] = None
        self._stable_labels = [s.label for s in plan.stable]
        self._stable_mastered = {lab: False for lab in self._stable_labels}
        self._stable_win_rates: Dict[str, float] = {}
        self._bot_routes = [r.index for r in self._routes if r.kind == "bot"]
        self._stable_route = {r.family.split(":", 1)[1]: r.index for r in self._routes if r.family.startswith("stable:")}

    # -- live pushes (the wrapper's env_method API)
    def set_self_play_target(self, fraction: float, generation: int) -> None:
        self._self_play_fraction = float(fraction)
        self._target_generation = int(generation)

    def set_stable_mastered(self, labels: Any) -> None:
        s = set(labels or ())
        self._stable_mastered = {lab: (lab in s) for lab in self._stable_labels}

    def set_stable_win_rates(self, rates: Any) -> None:
        self._stable_win_rates = dict(rates or {})

    # -- the draw
    def _stable_in(self, mastered: bool) -> List[str]:
        return [lab for lab in self._stable_labels if self._stable_mastered.get(lab, False) == mastered]

    def _floor(self) -> int:
        mastered = self._stable_in(True)
        candidates = self._bot_routes + [self._stable_route[lab] for lab in mastered]
        if self.plan.bot_weights is None:
            return self._rng.choice(candidates)
        weights = list(self.plan.bot_weights) + [1.0] * len(mastered)
        return self._rng.choices(candidates, weights=weights, k=1)[0]

    def _pick_stable(self, labels: List[str]) -> int:
        if not self.plan.stable_pfsp or not self._stable_win_rates or len(labels) == 1:
            return self._stable_route[self._rng.choice(labels)]
        weights = [max(0.05, 1.0 - float(self._stable_win_rates.get(lab, 0.5))) for lab in labels]
        return self._stable_route[self._rng.choices(labels, weights=weights, k=1)[0]]

    def _ensure_pool_model(self) -> bool:
        if self._pool is None:
            return False
        if self._target_generation != self._scanned_generation or not self._has_pool_model:
            if not self._pool.is_empty():
                self._gen_model = self._pool.sample(self._pool_rng)
                self._has_pool_model = True
                self._scanned_generation = self._target_generation
        return self._has_pool_model

    def _challenge(self) -> Optional[int]:
        stable = self._stable_in(False)
        pool_ready = self._ensure_pool_model()
        pool_route = self._families["pool"].route_of(self._gen_model) if pool_ready else None
        if stable and pool_ready:
            if self._rng.random() < self.plan.stable_challenge_share:
                return self._pick_stable(stable)
            return pool_route
        if pool_ready:
            return pool_route
        if stable:
            return self._pick_stable(stable)
        return None

    def draw(self) -> int:
        ex = self.plan.exploiter
        if ex is not None:
            if ex.keep_bots and self._rng.random() < ex.bot_fraction:
                return self._floor()
            fam = self._families["exploiter"]
            if not fam.roster:
                raise OpponentRoutingError("the exploiter target has no model loaded")
            return fam.route_of(fam.roster[0])
        if self._rng.random() < self._self_play_fraction:
            r = self._challenge()
            if r is not None:
                return r
        return self._floor()


# ------------------------------------------------------------------------------------ sampling


def sample_actions(logp: Any, temperature: Any, gens: Sequence[Any]) -> np.ndarray:
    """Today's ``RLPlayer`` stochastic action, batched: row ``i`` is
    ``torch.multinomial(Categorical(logits=logp[i] / T[i]).probs, 1, True, generator=gens[i])``,
    bit for bit (the division is skipped at T == 1, as ``RLPlayer`` skips it). ``logp`` ``[B, A]``
    float32 with illegal entries ``-inf`` (T2's contract; ``RLPlayer``'s ~-1e9 gives the same exact
    zeros). torch's one-sample multinomial draws ``q ~ Exp(1)`` per entry from the generator and takes
    ``argmax(p / q)``; the draws are per ROW (each env's own generator), the arithmetic batched."""
    import torch

    x = torch.as_tensor(np.asarray(logp, dtype=np.float32))
    t = np.asarray(temperature, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or len(gens) != x.shape[0] or t.shape[0] != x.shape[0]:
        raise ValueError("sample_actions: logp [B, A], one temperature and one generator per row")
    if (t != 1.0).any():
        tt = torch.as_tensor(t.astype(np.float32)).reshape(-1, 1)
        x = torch.where(torch.as_tensor(t != 1.0).reshape(-1, 1), x / tt, x)
    probs = torch.softmax(x - x.logsumexp(dim=-1, keepdim=True), dim=-1)
    q = torch.empty_like(probs)
    for i, g in enumerate(gens):
        q[i].exponential_(1, generator=g)
    return torch.argmax(probs / q, dim=-1).numpy().astype(np.int32)


# ------------------------------------------------------------------------------------ the server


@dataclass
class ServeStats:
    """Where an opponent serve's time goes. ``flush_s`` is the T2 flush CALL (it LAUNCHES the per-slot
    graph replays asynchronously); ``wait_s`` is the host waiting for them (``ticket.host()`` → the output
    event's sync — the GPU forward's completion); ``draw_s`` is choosing the actions (greedy / the sample)
    itself. Until 2026-09-30 the wait was billed to a single ``sample_s`` (F-LE-8's "5.1 ms of sampling"
    was the forward's completion wait — Lane G / the coordinator); ``sample_s`` stays their sum."""

    steps: int = 0
    rows: int = 0
    flush_s: float = 0.0
    wait_s: float = 0.0
    draw_s: float = 0.0
    gather_s: float = 0.0
    per_slot_rows: Dict[int, int] = field(default_factory=dict)
    #: ``submit`` calls that had rows, and the DISTINCT slots they submitted to (their ratio is the mean
    #: fan-out per flush — the number of slot replays one host step's opponent forward costs)
    submits: int = 0
    slot_submits: int = 0

    @property
    def sample_s(self) -> float:
        return self.wait_s + self.draw_s


class PolicyOpponentServer:
    """The T2 adapter (see the module docs). ``svc`` is a STARTED ``InferenceService`` whose global slot
    ``r.slot`` serves each policy route ``r``; ``policy_seed`` seeds one CPU ``torch.Generator`` per
    (env, opponent player) — today's ``RLPlayer`` objects, each with its own stream."""

    def __init__(self, plan: OpponentPlan, svc: Any, n_envs: int, *, policy_seed: int,
                 seed_stride: int = 0, force_greedy: bool = False, sampling: str = "generator",
                 run_seed: int = 0):
        import torch

        self.plan, self.svc, self.n = plan, svc, int(n_envs)
        self.routes = plan.routes()
        self.force_greedy = bool(force_greedy)
        # M5 Lane G (F-LE-8): ``keyed`` draws p2's sample with the COUNTER-BASED keyed draw
        # (``agents.training.keyed_draw``: key = (run_seed, the opponent stream, env, episode, p2's
        # dec_n)) in one vectorised op; ``generator`` (the default here, Lane E's gate) is today's
        # per-player ``torch.multinomial`` stream, bit for bit.
        if sampling not in ("generator", "keyed"):
            raise ValueError(f"PolicyOpponentServer: sampling {sampling!r}")
        self.sampling = sampling
        self.run_seed = int(run_seed)
        self.near_boundary = 0
        slot_route = {r.slot: r for r in self.routes if r.kind == "policy"}
        if svc is not None:
            n_svc = len(getattr(svc, "_slots", ())) or None
            # The policy routes own global slots 0 .. n_policy_slots - 1; a service may declare MORE
            # (M5 Lane G: the trainee's slot(s) follow them in the same service, one flush for both).
            if n_svc is not None and n_svc < plan.n_policy_slots:
                raise OpponentRoutingError(
                    f"the service declares {n_svc} slots; the plan's policy routes need {plan.n_policy_slots}")
        self._slot_route = slot_route
        self.temperature: Dict[str, float] = {}
        for r in self.routes:
            if r.kind == "policy":
                self.temperature[r.player] = float(r.temperature if r.temperature is not None else 1.0)
        players = sorted(self.temperature)       # built in either mode: `set_sampling` may switch (declared)
        self._gens: Dict[Tuple[int, str], Any] = {}
        for i in range(self.n):
            for p in players:
                g = torch.Generator(device="cpu")
                g.manual_seed(int(policy_seed) + int(seed_stride) * i)
                self._gens[(i, p)] = g
        self._logp = np.zeros((self.n, 11), dtype=np.float32)
        self._greedy = np.zeros(self.n, dtype=np.int64)
        self.stats = ServeStats()

    def set_sampling(self, sampling: str) -> None:
        """Switch between ``generator`` and ``keyed`` (a benchmark's A/B; both streams exist from startup)."""
        if sampling not in ("generator", "keyed"):
            raise ValueError(f"PolicyOpponentServer: sampling {sampling!r}")
        self.sampling = sampling

    def generator(self, env: int, player: str) -> Any:
        return self._gens[(env, player)]

    def serve(self, cols: Mapping[str, np.ndarray], *, record: Optional[List[Any]] = None) -> np.ndarray:
        """Answer every p2 decision on a POLICY route; returns the env indices served. ``record``
        (harnesses): appends ``(env, slot, logp row copy, greedy, action)`` per served row.
        ``submit`` + ONE flush + ``complete``."""
        pending = self.submit(cols)
        if pending is None:
            return np.zeros(0, dtype=np.int64)
        t1 = time.perf_counter()
        self.svc.flush()
        self.stats.flush_s += time.perf_counter() - t1
        return self.complete(cols, pending, record=record)

    def submit(self, cols: Mapping[str, np.ndarray]) -> Optional[Tuple[np.ndarray, List[Any]]]:
        """Phase 1 (M5 Lane G): submit p2's POLICY rows grouped by ``opp_slot`` WITHOUT flushing, so a
        caller can put the trainee's rows in the same T2 flush. Returns ``(order, tickets)`` or None."""
        from agents.inference.service.spec import Priority

        t0 = time.perf_counter()
        need = cols["need"][:, 1] == 1
        slot = cols["opp_slot"]
        envs = np.flatnonzero(need & (slot >= 0))
        if envs.size == 0:
            return None
        order = envs[np.argsort(slot[envs], kind="stable")]
        tickets = []
        for s in np.unique(slot[order]):
            rows = order[slot[order] == s]
            tickets.append((rows, self.svc.submit(int(s), cols["obs"][rows, 1], cols["mask"][rows, 1],
                                                   Priority.ROLLOUT)))
        self.stats.gather_s += time.perf_counter() - t0
        self.stats.submits += 1
        self.stats.slot_submits += len(tickets)
        return order, tickets

    def complete(self, cols: Mapping[str, np.ndarray], pending: Tuple[np.ndarray, List[Any]], *,
                 record: Optional[List[Any]] = None) -> np.ndarray:
        """Phase 2: after the flush, read the served rows, choose p2's actions into ``action[:, 1]``
        (greedy, or sampled at the route's temperature) and return the env indices served."""
        t2 = time.perf_counter()
        order, tickets = pending
        slot = cols["opp_slot"]
        for rows, t in tickets:
            lp, _v, gr = t.host()
            self._logp[rows] = lp
            self._greedy[rows] = gr
        t2w = time.perf_counter()
        act = cols["action"]
        stoch_rows, temps, gens = [], [], []
        for i in order:
            r = self._slot_route[int(slot[i])]
            if self.force_greedy or not r.stochastic:
                act[i, 1] = int(self._greedy[i])
            else:
                stoch_rows.append(i)
                temps.append(self.temperature[r.player])
                gens.append(self._gens.get((int(i), r.player)))
        if stoch_rows and self.sampling == "keyed":
            from agents.training import keyed_draw as KD

            sr = np.asarray(stoch_rows, dtype=np.int64)
            u = KD.keyed_uniforms(self.run_seed, KD.STREAM_OPPONENT, sr, cols["episode"][sr], cols["dec_n"][sr, 1])
            a, margin = KD.keyed_actions(self._logp[sr], u, np.asarray(temps, dtype=np.float64))
            act[sr, 1] = a
            self.near_boundary += int((margin < KD.NEAR_MARGIN).sum())
        elif stoch_rows:
            act[stoch_rows, 1] = sample_actions(self._logp[stoch_rows], temps, gens)
        t3 = time.perf_counter()
        st = self.stats
        st.steps += 1
        st.rows += int(order.size)
        st.wait_s += t2w - t2
        st.draw_s += t3 - t2w
        for s in slot[order]:
            st.per_slot_rows[int(s)] = st.per_slot_rows.get(int(s), 0) + 1
        if record is not None:
            for i in order:
                record.append((int(i), int(slot[i]), self._logp[i].copy(), int(self._greedy[i]), int(act[i, 1])))
        return order


# ------------------------------------------------------------------------------------ the host


class RustEnvOpponents:
    """Per-episode routing for one env core (see the module docs). ``load(route, model_id)`` is the
    caller's LOAD (resolve the model id to weights, then T2 ``load`` into the route's slot) — the
    refresh never compiles. ``pool`` a ``SnapshotPool`` (None without self-play); ``stable_ids`` /
    ``exploiter_id`` the model ids of the fixed families."""

    def __init__(self, plan: OpponentPlan, n_envs: int, *, load: Callable[[int, str], None],
                 pool: Any = None, stable_ids: Sequence[str] = (), exploiter_id: Optional[str] = None,
                 self_play_fraction: float = 0.0, pool_rng_seeds: Optional[Sequence[int]] = None,
                 rng_seeds: Optional[Sequence[int]] = None):
        self.plan, self.n = plan, int(n_envs)
        self.routes = plan.routes()
        self._load = load
        fam: Dict[str, SlotFamily] = {}
        by_family: Dict[str, List[int]] = {}
        for r in self.routes:
            if r.kind == "policy":
                by_family.setdefault(r.family, []).append(r.index)
        for name, rs in by_family.items():
            fam[name] = SlotFamily(name, rs)
        self.families = fam
        self.pool = PoolRoster(pool) if (pool is not None and plan.pool_slots) else None
        self.cur = np.full(self.n, -1, dtype=np.int64)       # the route each env's episode plays
        self.staged = np.full(self.n, -1, dtype=np.int64)    # the route staged for its next episode
        self._last_episode = np.full(self.n, -1, dtype=np.int64)
        self.generation = 0
        pool_rng_seeds = list(pool_rng_seeds) if pool_rng_seeds is not None else list(range(self.n))
        rng_seeds = list(rng_seeds) if rng_seeds is not None else list(range(self.n))
        self.samplers = [EpisodeOpponentSampler(plan, fam, rng_seed=rng_seeds[i], pool=self.pool,
                                                pool_rng_seed=pool_rng_seeds[i],
                                                self_play_fraction=self_play_fraction)
                         for i in range(self.n)]
        for i, lab in enumerate(s.label for s in plan.stable):
            fam[f"stable:{lab}"].admit([stable_ids[i]], set(), load)
        if plan.exploiter is not None:
            if exploiter_id is None:
                raise OpponentRoutingError("an exploiter plan needs the target's model id")
            fam["exploiter"].admit([exploiter_id], set(), load)
        if self.pool is not None:
            self._refresh_pool()
        self.route_counts = np.zeros(len(self.routes), dtype=np.int64)

    def _in_use(self) -> "set[int]":
        return {int(r) for r in np.concatenate([self.cur, self.staged]) if r >= 0}

    def _refresh_pool(self) -> List[int]:
        self.pool.pool._scan()
        return self.families["pool"].admit(self.pool.model_ids(), self._in_use(), self._load)

    # -- live pushes
    def set_self_play_target(self, fraction: float, generation: int) -> List[int]:
        """The eval push. A NEW generation rescans the pool and LOADS every new snapshot (between
        rollouts, by contract: the caller calls this between STEPs); returns the routes loaded."""
        loaded: List[int] = []
        if int(generation) != self.generation and self.pool is not None:
            loaded = self._refresh_pool()
        self.generation = int(generation)
        for s in self.samplers:
            s.set_self_play_target(fraction, generation)
        return loaded

    def set_stable_mastered(self, labels: Any) -> None:
        for s in self.samplers:
            s.set_stable_mastered(labels)

    def set_stable_win_rates(self, rates: Any) -> None:
        for s in self.samplers:
            s.set_stable_win_rates(rates)

    def set_opponent_win_rates(self, rates: Any) -> None:
        if self.pool is not None:
            self.pool.pool.set_win_rates(rates)

    # -- staging
    def _stage(self, cols: Mapping[str, np.ndarray], i: int) -> None:
        r = self.samplers[i].draw()
        self.staged[i] = r
        cols["ep_opp"][i] = r

    def stage_all(self, cols: Mapping[str, np.ndarray]) -> None:
        """Before the first RESET: stage every env's first episode."""
        for i in range(self.n):
            self._stage(cols, i)

    def after_op(self, cols: Mapping[str, np.ndarray]) -> np.ndarray:
        """After every RESET / STEP: an env whose ``episode`` moved consumed its staged route (a start,
        an auto-reset, a quarantine's restart, a refused start) — record it and stage the next.
        Returns the envs re-staged."""
        ep = cols["episode"].astype(np.int64)
        moved = np.flatnonzero(ep != self._last_episode)
        for i in moved:
            got = int(cols["opp_route"][i])
            if got != self.staged[i]:
                raise OpponentRoutingError(f"env {i}: the core started route {got}, the host staged {self.staged[i]}")
            self.cur[i] = got
            self.route_counts[got] += 1
            self._stage(cols, int(i))
        self._last_episode = ep
        return moved

    def opp_class(self, cols: Mapping[str, np.ndarray]) -> np.ndarray:
        """The ``opp_class`` label per env (the class of the episode's route) — Lane C's host-filled key."""
        k = np.array([r.klass for r in self.routes], dtype=np.int64)
        return k[cols["opp_route"].astype(np.int64)]
