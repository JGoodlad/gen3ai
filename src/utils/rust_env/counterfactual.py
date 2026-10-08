"""COUNTERFACTUAL replay-to-end on the RUST CORE (``gen3_cf_core_playout_v1``, poke-env retirement P6).

Pick up a recorded battle at turn ``T``, substitute a different action for OUR side, and play the rest
to a win / loss — the prober's ``replay-counterfactual`` ("could it have won if it hadn't choked this
turn?") and the shape any Monte-Carlo continuation label wants. It replaces the poke-env road
(``utils/bridge/counterfactual.py``, deleted in P6 slice 6d-1: two scripted poke-env players over the in-process bridge) with ONE
in-process play-out (:func:`utils.rust_env.successors.play_out`): no player objects, no protocol
re-parse — the rows are the TRAINING observation path's (program §6c), the choices the core mapper's.

THE DIVERGENCE (the old road's rule, kept): replay the record to the START of turn ``T`` (both sides at
the move request, none of turn ``T``'s choices fed); the OPPONENT plays its RECORDED turn-``T`` move (it
could not have reacted to our change on the same turn); OUR side plays the SUBSTITUTE; everything after
— an off-script forced switch on turn ``T`` included — is live. The core's root is the playout request
``at = {"turn": T, "other": "recorded"}`` (``src/rust_env/src/search/playout.rs``).

THE DICE: ``post_t_seeds[k]`` reseeds rollout ``k``'s engine AT the branch point — after the opponent's
recorded turn-``T`` choice was fed, before ours. The old road swapped the PRNG (``resumeReseed``) just
before turn ``T``'s FIRST choice was fed. Feeding one side's choice draws nothing, so both land on the
same stream position (held by the P6 identity read, ``designs/research_state/measurements/
pokeenv_p6_replay_cf_2026-10-07/``). ``None`` keeps the battle's own dice.

THE SIDES: OUR side and a model opponent answer through :class:`ModelPolicy` — ``RLPlayer.
_predict_best_action``'s arithmetic on the core row, one B=1 forward per decision (masked logits
``logits + (mask - 1) * 1e9``; greedy = argmax; stochastic = ``torch.multinomial`` over the
categorical's probs from a PER-ROLLOUT ``torch.Generator``, so a rollout's draws are a function of its
seed alone, whatever the batch order). A BOT opponent is the IN-CORE Lane-F port, asked only at a real
decision; rollout ``k``'s streams are ``random.Random(bot_stream_seed(bot_seed, k, s))``.

THE STALL RULE: ``stall_sides`` — every side that stall-forfeits (a model side does, as an ``RLPlayer``
did; a bot does not, F-LF-5); at a decision whose turn is ``>= StallConfig().threshold`` that side
forfeits, p1 first when both are open (``playout.rs``).

Nothing here imports poke-env (``src/poke_env_free_entry_points_test.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from utils.rust_env import successors as S

CF_SCHEMA = "gen3_cf_core_playout_v1"

_OTHER = {"p1": "p2", "p2": "p1"}


class ModelPolicy:
    """One side's answers from a loaded SB3 policy — ``RLPlayer._predict_best_action``'s decision on a
    core row (the same arithmetic as ``main.anchors.core_side.CorePolicy``), per row with B=1.

    ``stochastic`` samples at ``temperature`` from a PER-ROLLOUT generator seeded ``seeds[rollout]``
    (required then: an unseeded sampler would make the play-out irreproducible)."""

    def __init__(self, model: Any, *, stochastic: bool = False, temperature: float = 1.0,
                 seeds: Optional[Sequence[int]] = None) -> None:
        if stochastic and seeds is None:
            raise ValueError("a stochastic ModelPolicy needs per-rollout seeds (the play-out is reproducible)")
        self.model = model
        self.stochastic = bool(stochastic)
        self.temperature = float(temperature)
        self.seeds = None if seeds is None else [int(s) for s in seeds]
        self._gens: Dict[int, Any] = {}

    def _generator(self, rollout: int, device: Any) -> Any:
        import torch

        gen = self._gens.get(rollout)
        if gen is None:
            assert self.seeds is not None
            gen = self._gens[rollout] = torch.Generator(device=device)
            gen.manual_seed(int(self.seeds[rollout]))
        return gen

    def decide(self, row: np.ndarray, mask: np.ndarray, rollout: int) -> int:
        import torch

        with torch.no_grad():
            obs_t = torch.as_tensor(np.expand_dims(np.array(row, dtype=np.float32), 0)).to(self.model.device)
            mask_t = torch.as_tensor(np.expand_dims(np.asarray(mask, dtype=np.int8), 0)).to(self.model.device)
            dist = self.model.policy.get_distribution({"observation": obs_t, "action_mask": mask_t})
            masked = dist.distribution.logits + (mask_t - 1.0) * 1e9
            if not self.stochastic:
                idx = int(torch.argmax(masked, dim=1).item())
            else:
                logits = masked / self.temperature if self.temperature != 1.0 else masked
                cat = torch.distributions.Categorical(logits=logits)
                idx = int(torch.multinomial(cat.probs, 1, True, generator=self._generator(rollout, logits.device)).item())
        if not mask[idx]:
            raise S.SuccessorsError(f"the policy chose illegal action {idx} (mask {list(map(int, mask))})")
        return idx


@dataclass(frozen=True)
class Rollout:
    """One counterfactual line, from OUR side's view."""

    outcome: str                  # win | loss | tie | unfinished (truncated at max_turns)
    ended: bool
    turns: int
    forfeit: bool                 # it ended by a FORCELOSE (a stall forfeit past the root)
    capped: bool                  # a stall forfeit at the turn cap: decided by SEAT, read as a draw-at-cap
    reseed: Optional[str]
    decisions: Sequence[int]      # [p1, p2] decisions answered after the root
    text: Optional[List[str]]     # OUR side's protocol lines, the WHOLE battle (``text=True`` only)
    cmds: Optional[List[str]]     # the branch's full input log commands (``keep_cmds=True`` only)


def _outcome(end: Mapping[str, Any], our: str) -> str:
    if end is None or end.get("truncated"):
        return "unfinished"
    w = end.get("winner")
    if w is None:
        return "tie"
    return "win" if w == (0 if our == "p1" else 1) else "loss"


def replay_counterfactual(record, *, divergence_turn: int, substitute_action: int, our_policy: ModelPolicy,
                          opp_policy: Optional[ModelPolicy] = None, opp_bot: Optional[Mapping[str, Any]] = None,
                          post_t_seeds: Sequence[Optional[str]] = (None,), stall_sides: Sequence[str] = (),
                          text: bool = False, keep_cmds: bool = False, max_turns: int = 999,
                          core: Optional[S.SearchCore] = None) -> Dict[str, Any]:
    """Play the counterfactual of ``record`` (a ``ReconstructionRecord``) at ``divergence_turn``: OUR side
    (the record's trainee) plays ``substitute_action`` (an action INDEX of the root decision), the
    opponent its recorded turn-T move, then ``our_policy`` vs ``opp_policy`` / the in-core ``opp_bot``
    (``{"name", "seed"}``) to the end, once per ``post_t_seeds`` entry (rollout ``k`` = branch ``k``).

    Returns ``{"schema", "our_side", "root", "rollouts": [Rollout], "answered", "wall_s"}``; ``root`` is
    the core's root (``tokens`` — OUR legal choice tokens —, ``other_recorded``, ``turn``, ``at``)."""
    if (opp_policy is None) == (opp_bot is None):
        raise ValueError("exactly one of opp_policy / opp_bot plays the opponent")
    if not record.trainee_username:
        raise ValueError("reconstruction record lacks trainee_username")
    our = record.side_of(record.trainee_username)
    opp = _OTHER[our]
    our_i = 0 if our == "p1" else 1
    log = S.record_to_log(record)

    def policy(rows: np.ndarray, masks: np.ndarray, who: np.ndarray) -> np.ndarray:
        out = np.empty(len(who), dtype=np.int32)
        for i, w in enumerate(who):
            branch, side = int(w) // 2, int(w) % 2
            pol = our_policy if side == our_i else opp_policy
            if pol is None:
                raise S.SuccessorsError(f"branch {branch}: the bot's side was handed out as pending")
            out[i] = pol.decide(rows[i], masks[i], branch)
        return out

    stall = None
    if stall_sides:
        stall = S.production_stall(*sorted(set(stall_sides)))
    bot = None if opp_bot is None else {"side": opp, "name": str(opp_bot["name"]), "seed": int(opp_bot["seed"])}
    res = S.play_out(log, {"turn": int(divergence_turn), "other": "recorded"}, our, policy=policy,
                     seeds=list(post_t_seeds), actions=[int(substitute_action)], stall=stall, max_turns=max_turns,
                     keep_cmds=keep_cmds, core=core, bot=bot, text=our if text else None)
    turn_limit = stall["turn_limit"] if stall else None
    rollouts: List[Rollout] = []
    for b in res.branches:
        e = b["end"] or {}
        forfeit = bool(e.get("forfeit"))
        turns = int(e.get("turn", 0))
        rollouts.append(Rollout(
            outcome=_outcome(b["end"], our), ended=b["end"] is not None and not e.get("truncated"), turns=turns,
            forfeit=forfeit, capped=bool(forfeit and turn_limit is not None and turns >= turn_limit),
            reseed=b.get("reseed"), decisions=tuple(b["decisions"]),
            text=(res.prefix_text + list(b.get("text") or [])) if text else None,
            cmds=list(b["cmds"]) if keep_cmds else None))
    return {"schema": CF_SCHEMA, "our_side": our, "root": res.root, "rollouts": rollouts,
            "answered": res.answered, "wall_s": res.wall_s}


# ------------------------------------------------------------------ the narrated play-by-play


def summarize_trajectory(side: str, sink: list, *, max_turns: int = 80) -> list:
    """Parse ``side``'s protocol chunks (``[(side, chunk_text), …]``) into a compact, human-readable
    move-by-move log: a list of ``{turn, events:[str]}`` in chronological order, from ``side``'s one-sided
    view — so a 'could it have won' rollout reads as an actual play-by-play (what each side did, the
    damage/faints/crits/status, who won). Best-effort: a malformed line is skipped, never raised.
    Moved verbatim from ``utils/bridge/counterfactual.py`` (P6); ``max_turns`` caps the WHOLE battle's
    turns from turn 0, the prefix included (FINDING F-P6-3 of the P6 read)."""
    def who(slot: str) -> str:
        return "we" if slot[:2] == side else "opp"

    def mon(tok: str) -> str:
        t = tok.split(": ", 1)[1] if ": " in tok else tok
        return t.split(",")[0].strip()

    def frac(tok: str):
        head = tok.strip().split(" ")[0]
        if "fnt" in tok or head.startswith("0/") or head == "0":
            return 0
        try:
            num, den = head.split("/")
            return int(round(100 * int(num) / max(1, int(den))))
        except (ValueError, IndexError):
            return None

    lines = []
    for s, chunk in sink:
        if s == side:
            lines.extend(chunk.split("\n"))

    turns, cur = [], {"turn": 0, "events": []}
    for ln in lines:
        if not ln.startswith("|"):
            continue
        p = ln.split("|")
        tag = p[1] if len(p) > 1 else ""
        try:
            if tag == "turn":
                if cur["events"]:
                    turns.append(cur)
                cur = {"turn": int(p[2]), "events": []}
            elif tag == "move":
                cur["events"].append(f"{who(p[2])} used {p[3]}")
            elif tag in ("switch", "drag"):
                cur["events"].append(f"{who(p[2])} sent in {mon(p[2])}")
            elif tag == "-damage":
                f = frac(p[3])
                if f is not None:
                    cur["events"].append(f"  {mon(p[2])} → {f}% hp")
            elif tag == "-heal":
                f = frac(p[3])
                if f is not None:
                    cur["events"].append(f"  {mon(p[2])} healed → {f}% hp")
            elif tag == "-crit":
                cur["events"].append("  (crit)")
            elif tag == "-supereffective":
                cur["events"].append("  (super-effective)")
            elif tag == "-resisted":
                cur["events"].append("  (resisted)")
            elif tag == "-immune":
                cur["events"].append(f"  {mon(p[2])} immune (no effect)")
            elif tag in ("-miss", "-fail"):
                cur["events"].append(f"  ({'missed' if tag == '-miss' else 'failed'})")
            elif tag == "-status":
                cur["events"].append(f"  {mon(p[2])} is now {p[3]}")
            elif tag == "faint":
                cur["events"].append(f"  {mon(p[2])} FAINTED [{who(p[2])}]")
            elif tag == "win":
                cur["events"].append(f"→ {p[2]} WINS")
        except (IndexError, ValueError):
            continue
    if cur["events"]:
        turns.append(cur)

    def _dedup(evs):                          # collapse consecutive identical lines (sand/leftovers spam)
        out = []
        for e in evs:
            if not out or out[-1] != e:
                out.append(e)
        return out

    return [{"turn": t["turn"], "events": _dedup(t["events"])} for t in turns[:max_turns]]
