"""``search_decision``: one-ply value search at a decision of a core input log — a playout BUDGET spent
by sequential halving (or uniformly) over the policy's candidates, each sample a greedy playout in a
DETERMINIZED world (``worlds.WorldSampler``) scored at the end or truncated at a turn depth by the
critic; the action by Gumbel's prior-aware rule; and the policy-improvement TARGET (Gumbel MuZero's
completed-Q softmax) plus a value target, for an offline label factory.

The design, the measured decision improvement behind its defaults and the bootstrap plan:
`designs/endstate/design_value_discrimination.md` (§7 results, §13 the bootstrap). Offline / evaluation
use only — nothing here is wired into training.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from main.value_search import rules
from main.value_search.playout import ACT, LeafEngine, Request, decode_turn


@dataclass
class SearchConfig:
    """``budget``: playouts per decision (an int), or ``per_action`` × the legal count when ``budget`` is
    None. ``allocation``: ``"halving"`` (sequential halving over the policy's top-``m``) or
    ``"uniform"`` (every legal action the same count, then the ``rule`` over all). ``depth``: leaf at
    the critic ``depth`` turns past the root (None = to the end). ``n_worlds``: determinized worlds (a
    sample k of an action uses world k); ``oracle`` plays the TRUE log instead (an upper bound, never an
    honest search)."""

    budget: Optional[int] = None
    per_action: int = 4
    #: UNIFORM is the default: the pilot measured sequential halving over the policy's top-4 no better at
    #: B = n·1…n·2 and WORSE at B ≥ n·4 (design doc §7.11, R-SH)
    allocation: str = "uniform"
    m: Optional[int] = 4
    #: uniform allocation only: "gumbel" (c_scale 1.0) | "argmax" | "pikl" (L1's argmax Q + λ·log π at ``pikl_lambda``)
    rule: str = "gumbel"
    #: the piKL anchor λ (``rule="pikl"`` only): L1's fitted value by default (``rules.PIKL_LAMBDA_L1``)
    pikl_lambda: float = rules.PIKL_LAMBDA_L1
    #: 4 turns then the critic: half the full leaf's value at a seventh of its cost (§7.11); None = to the end
    depth: Optional[int] = 4
    n_worlds: int = 16
    oracle: bool = False
    #: how an action's leaf values average over its worlds: ``"uniform"`` (every world the same weight: today) or
    #: ``"likelihood"`` (each world by its self-normalised likelihood weight, ``rules.world_weights`` over the
    #: sampler's ``Worlds.log_w``: F1 tier A's ``dice="rolls"`` worlds; seed-dice worlds carry none, so uniform). Uniform
    #: allocation only (sequential halving's means are not world-indexed: refused)
    weights: str = "uniform"
    #: STAGE 1c (``design_world_sampler.md`` §4 (f), §6.3; OFF = None, today): draw ``proposals`` worlds, then keep
    #: ``n_worlds`` of them by an importance RESAMPLE on each world's damage-roll weight × the likelihood of the
    #: opponent's own past choices with this engine's network seated as the opponent
    #: (``opp_likelihood.resample_by_opponent``). The kept worlds are equally weighted (the resample spends the weights)
    proposals: Optional[int] = None


@dataclass
class SearchResult:
    action: int
    policy_target: np.ndarray             # [11], 0 on illegal actions
    q: Dict[int, float]                   # leaf means (win scale), visited actions
    visits: Dict[int, int]
    completed_q: Dict[int, float]
    root_value: float                     # the critic at the root
    value_target: float                   # Σ π' · completed q
    prior: np.ndarray                     # the policy's masked softmax [11]
    spent: int
    cpu_s: float
    wall_s: float
    world_tries: List[int] = field(default_factory=list)
    leak: dict = field(default_factory=dict)
    #: the OPPONENT's team in each searched world, in world order (empty for the oracle): what a caller measures the
    #: worlds' quality against the truth with (S-U's REPORTED world-quality covariate)
    world_teams: List[str] = field(default_factory=list)


def seeds_for(key: str, n: int) -> List[str]:
    return [f"sodium,{hashlib.sha256(f'value_search_dice_v1:{key}:{k}'.encode()).hexdigest()[:32]}"
            for k in range(n)]


def _cpu() -> float:
    import resource

    a = resource.getrusage(resource.RUSAGE_SELF)
    return a.ru_utime + a.ru_stime


def search_decision(engine: LeafEngine, log: dict, at: int, side: str, root_row: np.ndarray,
                    root_mask: np.ndarray, cfg: SearchConfig, key: str, sampler=None,
                    root_choice_in_log: Optional[bool] = None) -> SearchResult:
    """Search the decision ``side`` takes at command index ``at`` of ``log`` (``root_row`` /
    ``root_mask``: that decision's obs row and mask). ``key`` seeds the worlds and the dice
    (deterministic). ``sampler``: a ``worlds.WorldSampler`` (required unless ``cfg.oracle``).
    ``root_choice_in_log``: whether the log's last opponent command is the opponent's choice AT THE ROOT (the sampler
    then keeps it); None reads it off the root (a banked log at a decision the opponent answered first holds it), and a
    LIVE log that never holds the opponent's pending choice passes False (``main.anchors.search_side``)."""
    if cfg.weights not in ("uniform", "likelihood"):
        raise ValueError(f"unknown world weighting {cfg.weights!r}")
    if cfg.weights == "likelihood" and cfg.allocation != "uniform":
        raise ValueError("likelihood world weights need uniform allocation (halving's means are not world-indexed)")
    if cfg.proposals is not None and (cfg.oracle or cfg.proposals < cfg.n_worlds):
        raise ValueError(f"proposals={cfg.proposals}: an honest search resamples n_worlds={cfg.n_worlds} of at least "
                         "as many proposals (never the oracle)")
    c0, w0 = _cpu(), time.monotonic()
    teams_w: List[str] = []
    out = engine.forward(root_row[None].astype(np.float32), root_mask[None].astype(np.uint8))[0]
    logits = np.where(root_mask.astype(bool), out[:ACT].astype(np.float64), -np.inf)
    v_root = float(out[ACT])
    legal = [int(a) for a in np.flatnonzero(root_mask)]
    root_turn = int(decode_turn(root_row[None], engine.off)[0])
    tries, leak = [], {}
    log_w: List[float] = []
    if cfg.oracle:
        logs = [log] * cfg.n_worlds
    else:
        if sampler is None:
            raise ValueError("an honest search needs a world sampler (or cfg.oracle=True)")
        root = engine.root(log, at, side)
        in_log = (not bool(root["other_open"])) if root_choice_in_log is None else bool(root_choice_in_log)
        w = sampler.sample(log, at, side, cfg.n_worlds if cfg.proposals is None else cfg.proposals, key,
                           root_choice_in_log=in_log)
        if cfg.proposals is not None:
            from main.value_search import opp_likelihood as OL
            from main.value_search import worlds as W

            w, _ = OL.resample_by_opponent(W.prepare_root(log, at, side, key, in_log), w, engine.forward,
                                           cfg.n_worlds, key)
        logs, tries, leak = w.logs, w.tries, w.leak
        log_w = list(getattr(w, "log_w", None) or [])
        teams_w = list(getattr(w, "teams", None) or [])
    seeds = seeds_for(key, cfg.n_worlds)
    budget = cfg.budget if cfg.budget is not None else cfg.per_action * len(legal)

    def play(req: Dict[int, int], have: Dict[int, int]) -> Dict[int, List[float]]:
        reqs, who = [], []
        for a, n in req.items():
            for k in range(have.get(a, 0), have.get(a, 0) + n):
                reqs.append(Request(world=k, action=a, seed=seeds[k]))
                who.append(a)
        vals = engine.leaves(logs, at, side, root_turn, reqs, depth=cfg.depth)
        got: Dict[int, List[float]] = {}
        for a, v in zip(who, vals):
            got.setdefault(a, []).append(v)
        return got

    if cfg.allocation == "halving":
        sh = rules.SequentialHalving(logits=logits, legal=legal, budget=budget, m=cfg.m, cap=cfg.n_worlds)
        while not sh.done:
            req = sh.next_round()
            if not req:
                continue
            got = play(req, {a: len(sh.samples[a]) for a in req})
            for a in req:
                sh.observe(a, got[a])
        q, visits, action, spent = sh.means(), sh.visits(), sh.pick(), sh.spent
    elif cfg.allocation == "uniform":
        k = min(cfg.n_worlds, max(1, budget // len(legal)))
        got = play({a: k for a in legal}, {})
        if cfg.weights == "likelihood":
            wk = rules.world_weights(log_w, k)        # sample i of every action is world i (play's request order)
            q = {a: float(np.dot(wk, np.asarray(v, dtype=np.float64))) for a, v in got.items()}
        else:
            q = {a: float(np.mean(v)) for a, v in got.items()}
        visits = {a: k for a in legal}
        order = rules.policy_order(logits, legal)
        if cfg.rule == "gumbel":
            score = rules.gumbel_scores(logits, q, k)
        elif cfg.rule == "pikl":
            score = rules.pikl_scores(logits, q, cfg.pikl_lambda, order)
        elif cfg.rule == "argmax":
            score = q
        else:
            raise ValueError(f"unknown acting rule {cfg.rule!r}")
        action, spent = rules.pick(score, order), k * len(legal)
    else:
        raise ValueError(f"unknown allocation {cfg.allocation!r}")
    pi_t = rules.improved_policy(logits, legal, q, visits, v_root)
    cq = rules.completed_q(logits, legal, q, visits, v_root)
    return SearchResult(action=int(action), policy_target=pi_t, q=q, visits=visits, completed_q=cq,
                        root_value=v_root, value_target=float(sum(pi_t[a] * cq[a] for a in legal)),
                        prior=rules.softmax_masked(logits, legal), spent=int(spent),
                        cpu_s=_cpu() - c0, wall_s=time.monotonic() - w0, world_tries=tries, leak=leak,
                        world_teams=teams_w)
