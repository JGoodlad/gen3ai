"""Counterfactual replay-to-end — pick up a recorded bridge battle at turn T, substitute a different
move, and play the rest LIVE (the trainee's policy vs the RELOADED real opponent) to a win / loss.

Feature 2 of the prober's counterfactual analysis, and the answer to "could the model have won if it
hadn't choked this one turn?". It reuses the in-process :func:`run_local_battles` driver wholesale:
both players are real poke-env players, but their ``choose_move`` is SCRIPTED to replay the recorded
commands until the divergence turn, then handed back to the live policy.

Faithful prefix (the design):

- The bridge ``START`` uses the recorded RESOLVED seed + both recorded packed teams, so turns 1..T-1
  reproduce the real board exactly.
- Each scripted decision sends the recorded sim choice as a pass-through ``BattleOrder`` AND — for a
  ``Gen3Player`` whose obs the live policy will read after the handoff — runs ``embed_battle`` +
  ``tracker.advance(recorded_idx)`` so the post-divergence turn-history is faithful. The recorded
  action index is recovered by INVERTING the recorded choice string through the real action mapper
  (the ``map_actions_at`` mechanism), so the runner is self-contained on the record.
- At turn T OUR side plays the SUBSTITUTE (a legal alternative / a human's move) and goes LIVE; the
  opponent plays its RECORDED turn-T move (it could not have reacted to our change on the same turn)
  and goes live from T+1.

The opponent past T is the RELOADED real opponent (a bot is reproducible code; a self-play sentinel /
stable model reloads from disk) — a faithful counterfactual against the ACTUAL opponent, not a
self-play approximation. Building the players (trainee policy + reloaded opponent, both with the
recorded teams) is the caller's job (the prober session); this module is policy-agnostic.

``divergence_turn=None`` scripts the WHOLE recorded game with no handoff — the full-replay mode used
as a correctness oracle (it must reproduce the recorded winner + the recorded one-sided obs).

The result is ONE realization. A statistically meaningful "could it have won" resamples the
post-divergence dice over many rollouts (the ``post_t_seed`` reseed hook + the session's Monte-Carlo
loop); a single deterministic line (the realized dice continued) is the default.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import List, Optional

import numpy as np

from poke_env.player.battle_order import SingleBattleOrder
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

from utils.bridge.local_battle_runner import run_local_battles
from utils.bridge.reconstruction import ReconstructionRecord

_OTHER = {"p1": "p2", "p2": "p1"}


def _passthrough(choice_str: str) -> SingleBattleOrder:
    """A ``BattleOrder`` that sends the recorded sim choice verbatim (``SingleBattleOrder`` returns a
    bare string ``order`` as its message). The faithful way to replay a recorded command — incl. a
    refused ``[Unavailable choice]`` maybe-trapped probe — without re-deriving it from the live state."""
    return SingleBattleOrder("/choose " + choice_str)


def _force_switch(battle) -> bool:
    fs = getattr(battle, "force_switch", False)
    return any(fs) if isinstance(fs, (list, tuple)) else bool(fs)


def _invert_choice(player, battle, choice_str: str) -> Optional[int]:
    """Recover the action INDEX a recorded sim choice string corresponds to, by mapping every legal
    action through the REAL action mapper at this decision state and matching the message (zero mapping
    reimplementation — the inverse of ``map_actions_at``). ``None`` when the choice isn't a currently
    legal action (a refused maybe-trapped probe) or the player has no decision context yet."""
    tracker = getattr(player, "_get_tracker", None)
    if tracker is None:
        return None
    ctx = tracker(battle).last_ctx
    if ctx is None:
        return None
    for idx in np.flatnonzero(np.asarray(ctx.mask)):
        try:
            order = player.action_to_order(int(idx), battle)
        except Exception:  # noqa: BLE001 — a stale/illegal candidate is just not a match
            continue
        if order.message[len("/choose "):] == choice_str:
            return int(idx)
    return None


def install_scripted_prefix(
    player,
    *,
    side: str,
    record: ReconstructionRecord,
    divergence_turn: Optional[int],
    substitute_choice: Optional[str],
    is_our_side: bool,
    obs_sink: Optional[list] = None,
) -> None:
    """Override ``player.choose_move`` (instance-level) so it replays ``side``'s recorded commands until
    the divergence, then defers to the live policy. See the module header for the divergence semantics.

    ``obs_sink`` (a list) collects this player's per-decision obs vectors during the SCRIPTED prefix —
    used by the faithfulness oracle to assert the scripted obs == the recorded ``states.npz`` obs.
    """
    script = deque(c for (s, c) in record.commands if s == side)
    is_gen3 = hasattr(player, "embed_battle")
    original_choose = player.choose_move          # bound method = the live policy / bot
    state = {"live": False, "substituted": False}

    def _advance_tracker(battle, choice_str: str) -> None:
        if not is_gen3:
            return
        idx = _invert_choice(player, battle, choice_str)
        if idx is not None:                       # a refused probe inverts to None → no commit, no advance
            player._get_tracker(battle).advance(idx)

    def scripted_choose_move(battle):
        if is_gen3:                               # stall safety, mirroring the live path
            forfeit = player._handle_stall(battle, "COUNTERFACTUAL_STALL")
            if forfeit:
                return forfeit
        t = int(getattr(battle, "turn", 0) or 0)
        # A FORCED SWITCH at the divergence turn is OFF-SCRIPT: our substitute changed turn-T's
        # resolution, so a faint (and its mid-turn switch request) that the recording didn't contain
        # has no valid recorded answer — popping the flat command deque would steal the recorded
        # turn-(T+1) move and feed it to a forceSwitch (rejected, desync). Hand any such forced switch
        # to the LIVE policy instead (the faithful resolution — the reloaded opponent / the trainee
        # actually decides). The opp's turn-T MOVE round (force_switch False) is still replayed: it
        # committed not knowing our change. Genuine recorded force-switches on PREFIX turns (t < T)
        # replay normally.
        off_script_fs = (divergence_turn is not None and t == divergence_turn and _force_switch(battle))
        past_divergence = divergence_turn is not None and t > divergence_turn
        if is_our_side:
            live_now = state["substituted"] or off_script_fs or past_divergence
        else:
            live_now = off_script_fs or past_divergence
        if live_now:
            state["live"] = True
            if is_our_side and off_script_fs:
                state["substituted"] = True   # a forced switch pre-empts our move round → fully live
            return original_choose(battle)

        # --- a SCRIPTED decision (prefix, or our divergence move round) ---
        obs_dict = None
        if is_gen3:
            obs_dict = player.embed_battle(battle)        # rebuild tracker history (record + clock)
            if int(np.asarray(obs_dict["action_mask"]).sum()) == 0:
                return player.choose_default_move()       # no decision this call (mirror live)
            if obs_sink is not None:
                obs_sink.append(np.asarray(obs_dict["observation"], dtype=np.float32))

        # Our side at the divergence move round → substitute and go live.
        if (is_our_side and divergence_turn is not None and t == divergence_turn
                and not state["substituted"] and not _force_switch(battle)
                and substitute_choice is not None):
            state["substituted"] = True
            _advance_tracker(battle, substitute_choice)
            return _passthrough(substitute_choice)

        # Replay this side's next recorded choice.
        if not script:                            # exhausted before the divergence → defer to live
            state["live"] = True
            return original_choose(battle)
        choice_str = script.popleft()
        _advance_tracker(battle, choice_str)
        return _passthrough(choice_str)

    player.choose_move = scripted_choose_move


def summarize_trajectory(side: str, sink: list, *, max_turns: int = 80) -> list:
    """Parse the trainee-side protocol chunks (from ``run_local_battles``'s ``chunk_sink``) into a
    compact, human-readable move-by-move log: a list of ``{turn, events:[str]}`` in chronological order,
    from OUR (``side``) one-sided view — so a 'could it have won' rollout reads as an actual play-by-play
    (what each side did, the damage/faints/crits/status, who won). Best-effort: a malformed line is
    skipped, never raised."""
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


def _battle_outcome(player, trainee_username: str) -> dict:
    """Read the terminal result of the (single) battle ``player`` just finished. ``player`` is the
    trainee, so ``battle.won`` is from our perspective."""
    battles = list(getattr(player, "_battles", {}).values())
    battle = battles[-1] if battles else None
    if battle is None:
        return {"outcome": "unfinished", "ended": False, "turns": 0, "winner": None}
    won = getattr(battle, "won", None)
    finished = getattr(battle, "finished", False)
    return {
        "outcome": ("win" if won is True else "loss" if won is False else
                    "tie" if finished else "unfinished"),
        "ended": bool(finished),
        "turns": int(getattr(battle, "turn", 0) or 0),
        "winner": (trainee_username if won is True else None),
    }


def replay_counterfactual(
    record: ReconstructionRecord,
    *,
    trainee,
    opponent,
    divergence_turn: Optional[int],
    substitute_choice: Optional[str] = None,
    seed=None,
    post_t_seed=None,
    timeout: float = 180.0,
    capture_obs: bool = False,
    capture_trajectory: bool = False,
    impl: str = "node",
) -> dict:
    """Play ONE counterfactual line: replay ``record`` to ``divergence_turn``, substitute
    ``substitute_choice`` for OUR side, then continue live (``trainee`` policy vs ``opponent``) to a
    terminal. Returns ``{outcome, ended, turns, winner, divergence_turn, substitute, [trainee_obs]}``.

    ``trainee`` / ``opponent`` are pre-built poke-env players (the caller sets the policy / reloads the
    opponent); this runner forces their teams to the RECORDED packed teams and scripts the prefix.
    ``seed`` defaults to the record's resolved START seed (so the prefix reproduces the real board);
    pass an explicit ``[s0,s1,s2,s3]`` / ``"sodium,<hex>"`` to vary the WHOLE line's dice.
    ``post_t_seed`` (with a ``divergence_turn``) swaps the sim PRNG at the START of the divergence turn
    — so the prefix keeps the recorded dice but the post-divergence resolution is resampled: vary it
    across rollouts for a Monte-Carlo "could it have won" win-probability.
    ``divergence_turn=None`` is the full-replay oracle (no substitute, no handoff).

    ``impl`` (``"node"`` default | ``"rust"``) picks the in-process sim-bridge child the live
    post-divergence battle runs on — this leg plays a REAL game, so it rides the same
    ``bridge_spawn_argv`` seam as training/eval, not the offline replay-driver one."""
    if not record.trainee_username:
        raise ValueError("reconstruction record lacks trainee_username")
    our_side = record.side_of(record.trainee_username)
    opp_side = _OTHER[our_side]

    # Force the recorded teams (defensive: the players may have been built with a random teambuilder).
    trainee._team = ConstantTeambuilder(record.packed_team(our_side))
    opponent._team = ConstantTeambuilder(record.packed_team(opp_side))

    obs_sink: Optional[list] = [] if capture_obs else None
    install_scripted_prefix(
        trainee, side=our_side, record=record, divergence_turn=divergence_turn,
        substitute_choice=substitute_choice, is_our_side=True, obs_sink=obs_sink)
    install_scripted_prefix(
        opponent, side=opp_side, record=record, divergence_turn=divergence_turn,
        substitute_choice=None, is_our_side=False)

    if seed is None:
        seed = record.start_options().get("seed")

    start_extra = None
    if post_t_seed is not None and divergence_turn is not None:
        start_extra = {"resumeReseed": {"turn": int(divergence_turn), "seed": post_t_seed}}

    traj_sink: Optional[list] = [] if capture_trajectory else None
    p1_player, p2_player = (trainee, opponent) if our_side == "p1" else (opponent, trainee)
    asyncio.run(run_local_battles(
        p1_player, p2_player, 1, battle_format=record.format_id, seed=seed, start_extra=start_extra,
        chunk_sink=traj_sink, impl=impl))

    result = _battle_outcome(trainee, record.username(our_side))
    result.update(divergence_turn=divergence_turn, substitute=substitute_choice)
    if capture_obs:
        result["trainee_obs"] = obs_sink
    if capture_trajectory:
        result["trajectory"] = summarize_trajectory(our_side, traj_sink or [])
    return result
