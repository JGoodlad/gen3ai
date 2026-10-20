"""THE OPPONENT-CHOICE LIKELIHOOD (candidate (f) of ``designs/endstate/design_world_sampler.md`` §4; the literature
review's L3): how likely the opponent's OBSERVED past choices were if a candidate world were true, with our policy
network, seated as the opponent, as the opponent model. Offline / evaluation use only, and OFF by default: nothing in
the sampler or the search calls it; a caller (a measurement driver, a later stage-1c reweighting) asks for it.

    log L(w) = Σ_t log π(a_t | o_t^opp(w))

over the opponent's past decisions t before the root, where ``o_t^opp(w)`` is the OPPONENT's own observation row at
its t-th decision in world w (world w's prefix replayed through the core, ``core_events --trackers --obs``, read from
the opponent's seat: its own team as world w has it, our team as the opponent saw it, nothing of ours it did not see),
``a_t`` the action it took, and π the masked softmax of the network's 11 policy logits (float64, temperature 1).

WHICH DECISIONS ARE SCORED (``opponent_steps``; every world of a root shares one list, so the sum is over the same t):
  * ``scored``     — a choice the searched side SAW the result of, kept as the truth's token in every world;
  * ``unobserved`` — a choice the world template rewrote to a SLOT (``worlds.prepare_root``: a move the searched side
                     never saw that mon use, a switch to a mon it never saw): the token is world-specific and its slot is
                     the truth's, so scoring it would be a leak. Excluded (factor 1: the observation is treated as
                     carrying no evidence), counted;
  * ``pending``    — the opponent's choice for the ROOT round when it chose first: the searched side has not seen it
                     (simultaneous moves). Decided by replay: the choice is pending iff the searched side's root
                     request is emitted WITHOUT it (``pending_cmd``). Excluded, counted.
A decision with ONE legal action contributes exactly 0 (log 1) in every world; ``n_informative`` counts the scored
steps with ≥ 2 legal actions (a world can make a step forced, e.g. a Choice lock, so it is per world).

THE FAMILIARITY CONTROL: ``conf`` = Σ_t log max_a π(a | o_t^opp(w)) over the same steps, the log-probability of the
network's OWN top choice; it reads no observed action, so a ranking by ``conf`` measures how SURE the network is in a
world, not what the opponent did (a network trained against a pool is surer on that pool's teams).

THE GUARDS (a defect raises, never an exclusion): every world's replay must reproduce the searched side's rows up to
its root decision byte for byte (the gate the world passed; a world replayed without its damage rolls fails it), the
opponent's tracker choice at every one of its decisions must equal the log's token, and a scored token must be one of
the decision's legal tokens. Measurement and verdict: ``designs/research_state/measurements/search_l3_2026-10-20/``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from main.value_search.playout import ACT, dedup_forward
from main.value_search.worlds import Root, WorldError, log_commands, render, reveals

#: ``forward(rows [k, OBS] f32, masks [k, 11] u8) -> [k, ≥ 11]``: the 11 policy logits first (``playout.Forward``)
Forward = Callable[[np.ndarray, np.ndarray], np.ndarray]

STATUSES = ("scored", "unobserved", "pending")


class OppLikelihoodError(RuntimeError):
    """A replay or a token that breaks the likelihood's contract: a DEFECT, never an exclusion."""


@dataclass(frozen=True)
class OppStep:
    """One past opponent decision of a root: its index among the opponent's answered decisions, its command index
    in the log, and how the likelihood treats it (``STATUSES``)."""

    k: int
    cmd: int
    status: str


@dataclass
class OppLikelihood:
    """One world's opponent-choice log-likelihood over its root's scored steps."""

    ll: float
    conf: float
    logp: List[float] = field(default_factory=list)     # per scored step, in step order
    n_legal: List[int] = field(default_factory=list)    # per scored step
    n_scored: int = 0
    n_informative: int = 0
    n_unobserved: int = 0
    n_pending: int = 0


def pending_cmd(root: Root) -> Optional[int]:
    """The command index of the opponent's choice for the ROOT round when it is already in the log (the opponent
    chose first), else None. Only the opponent's LAST command before the root can be pending, and only when it comes
    after the searched side's last command; it is pending iff the replay WITHOUT it still emits the searched side's
    root request (a choice the root request did not wait for is one whose result the searched side has not seen)."""
    from agents.battle.core_replay import run_core

    opp = "p2" if root.side == "p1" else "p1"
    last_opp = max((i for i, c in enumerate(root.cmds) if c[0] == opp), default=None)
    last_me = max((i for i, c in enumerate(root.cmds) if c[0] == root.side), default=-1)
    if last_opp is None or last_opp < last_me:
        return None
    res = run_core([root.rec(None, "opp_ll#pending", root.cmds[:last_opp])], trackers=True, obs=True)[0]
    if not res["ok"]:
        raise OppLikelihoodError(f"the prefix without the opponent's last choice does not replay: {res.get('error')}")
    ent = [e for e in res["trackers"][root.viewer] if "obs" in e]
    return last_opp if len(ent) > root.n else None


def opponent_steps(root: Root) -> List[OppStep]:
    """The opponent's past decisions before the root, each with its status (module docstring)."""
    opp = "p2" if root.side == "p1" else "p1"
    pend = pending_cmd(root)
    out: List[OppStep] = []
    k = 0
    for i, (s_, t) in enumerate(root.tmpl):
        if s_ != opp:
            continue
        status = "pending" if i == pend else ("unobserved" if isinstance(t, tuple) else "scored")
        out.append(OppStep(k=k, cmd=i, status=status))
        k += 1
    return out


def truth_log(root: Root) -> dict:
    """The TRUE battle through the same template every world is rendered from (the truth as one more candidate)."""
    cmds = [f"FORCELOSE {tok}" if s_ == "forcelose" else f"CHOOSE {s_} {tok}" for s_, tok in render(root.tmpl, root.team)]
    return dict(root.log, cmds=cmds + list(root.log["cmds"][root.at:]))


def _replay(root: Root, logs: Sequence[dict], label: str) -> List[dict]:
    """Each log's prefix (to the root) through ``core_events --trackers --obs``, in ONE process; a log carrying
    ``"rolls"`` (a ``dice="rolls"`` world) replays the damage rolls its gate accepted."""
    from agents.battle.core_replay import RecordedBattle, run_core

    recs = []
    for j, lg in enumerate(logs):
        rolls = lg.get("rolls")
        recs.append(RecordedBattle(
            label=f"{label}#{j}", format_id=lg["format_id"], seed=lg["seed"],
            p1={"name": lg["names"][0], "team": lg["teams"][0]}, p2={"name": lg["names"][1], "team": lg["teams"][1]},
            commands=log_commands(lg, root.at),
            rolls={"viewer": root.viewer, "mode": "force", "force": [list(x) for x in rolls]} if rolls else None))
    return run_core(recs, trackers=True, obs=True)


def _opp_inputs(root: Root, res: dict, lg: dict, steps: Sequence[OppStep], j: int
                ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """The opponent's (row, mask, action) at every SCORED step of one replayed world, after its guards."""
    from agents.battle.core_obs import wrap_row

    if not res["ok"]:
        raise OppLikelihoodError(f"world {j}: the prefix does not replay: {res.get('error')}")
    try:
        _, rows = reveals(res, root.viewer, root.n)
    except WorldError as exc:
        raise OppLikelihoodError(f"world {j}: {exc}") from exc
    if rows != root.ref:
        raise OppLikelihoodError(f"world {j}: the searched side's rows differ from the truth's (a world replayed off "
                                 "the history its gate accepted, e.g. without its damage rolls)")
    ent = [e for e in res["trackers"][root.oi] if "obs" in e]
    cmds = log_commands(lg, root.at)
    if len(ent) < len(steps):
        raise OppLikelihoodError(f"world {j}: the opponent has {len(ent)} decisions, the log {len(steps)} choices")
    xs, ms, acts = [], [], []
    for st in steps:
        e = ent[st.k]
        tok = cmds[st.cmd][1]
        if e.get("choice") != tok:
            raise OppLikelihoodError(f"world {j}: the opponent's decision {st.k} took {e.get('choice')!r}, "
                                     f"the log's command {st.cmd} is {tok!r}")
        if st.status != "scored":
            continue
        tokens = {int(a): t for a, t in e["tokens"].items()}
        mask = np.asarray(e["mask"], dtype=np.uint8)
        if sorted(tokens) != [int(a) for a in np.flatnonzero(mask)]:
            raise OppLikelihoodError(f"world {j}: decision {st.k}'s tokens disagree with its mask")
        hit = [a for a, t in tokens.items() if t == tok]
        if len(hit) != 1:
            raise OppLikelihoodError(f"world {j}: decision {st.k}'s choice {tok!r} is not exactly one legal token "
                                     f"({sorted(tokens.values())})")
        xs.append(np.asarray(wrap_row(e["obs"]), dtype=np.float32))
        ms.append(mask)
        acts.append(hit[0])
    return xs, ms, acts


def log_probs(logits: np.ndarray, masks: np.ndarray, acts: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Per row: log π(act) and log max_a π(a) under the masked softmax of the 11 logits, in float64."""
    lg = np.where(masks.astype(bool), logits[:, :ACT].astype(np.float64), -np.inf)
    top = lg.max(axis=1)
    lse = top + np.log(np.exp(lg - top[:, None]).sum(axis=1))
    a = np.asarray(acts, dtype=np.int64)
    chosen = lg[np.arange(len(a)), a]
    if not np.all(np.isfinite(chosen)):
        raise OppLikelihoodError("a scored action is masked out")
    return chosen - lse, top - lse


def world_logliks(root: Root, logs: Sequence[dict], forward: Forward, steps: Optional[Sequence[OppStep]] = None,
                  label: str = "opp_ll") -> List[OppLikelihood]:
    """One :class:`OppLikelihood` per log (a world's full core input log, ``Worlds.logs[j]``, or :func:`truth_log`),
    every row of every log through ONE deduplicated forward (identical rows score identically in every world)."""
    steps = list(opponent_steps(root) if steps is None else steps)
    results = _replay(root, logs, label)
    per = [_opp_inputs(root, res, lg, steps, j) for j, (res, lg) in enumerate(zip(results, logs))]
    rows = [x for xs, _, _ in per for x in xs]
    n_un = sum(1 for s in steps if s.status == "unobserved")
    n_pe = sum(1 for s in steps if s.status == "pending")
    if not rows:
        return [OppLikelihood(ll=0.0, conf=0.0, n_unobserved=n_un, n_pending=n_pe) for _ in logs]
    masks = np.stack([m for _, ms, _ in per for m in ms])
    out = dedup_forward(forward)(np.stack(rows), masks)
    lp, cf = log_probs(np.asarray(out), masks, [a for _, _, acts in per for a in acts])
    res_: List[OppLikelihood] = []
    o = 0
    for xs, ms, _ in per:
        k = len(xs)
        n_leg = [int(m.sum()) for m in ms]
        res_.append(OppLikelihood(ll=float(lp[o:o + k].sum()), conf=float(cf[o:o + k].sum()),
                                  logp=[float(x) for x in lp[o:o + k]], n_legal=n_leg, n_scored=k,
                                  n_informative=sum(1 for n in n_leg if n >= 2), n_unobserved=n_un, n_pending=n_pe))
        o += k
    return res_
