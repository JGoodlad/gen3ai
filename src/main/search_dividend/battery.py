"""The evaluation battery: play matched games per (arm, budget, opponent) and append the rows.

**Matched games are the whole point.** Three arms are compared, so every cell must play the SAME
battles: the same opponent, the same team draw, the same dice. The driver therefore derives a
per-game SEED from ``(opponent, game_index, --games-seed)`` and pins it on the bridge ``START``,
and it draws both teams from a per-game RNG rather than the players' own teambuilders. Two arms
that disagree on a game then disagree about the DECISION, not about the battle they were handed —
without that, an arm difference at N=10 is mostly team-draw noise (the exploiter work measured
exactly that trap: an apparent edge that vanished under an equal-pilot mirror).

**The MIRROR mode, and why it needs side-swap.** ``--opponents self`` plays the searched side
against the SAME network with search structurally off, so the two sides differ in exactly one
thing. That pins the no-effect point at 50% by construction — unlike the scripted roster, which
saturates near 90% and hides a dividend in its ceiling. But a mirror game's TEAM DRAW is
asymmetric, and at small n that asymmetry is most of the variance: the searched side simply gets a
better or worse team. ``--side-swap`` plays every game index in BOTH orientations off one pinned
seed, so the summary can difference the pair and report a win rate the team draw cannot fake.

**Resumable, append-only.** Each finished game appends one JSON line. A relaunch reads the file,
counts the ``(arm, budget, opponent)`` cells' finished ORIENTATION-GAMES, and plays only what is
missing. The file is never rewritten — the same discipline ``eval_results.jsonl`` follows, and for
the same reason: a summary can always be recomputed from rows, but a row lost to a crashed rewrite
is gone.

**Every row carries an outcome XOR a named error** (:func:`finalize_row`). A row that says nothing
went wrong on a battle that never happened is the one shape this file must not be able to hold.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from agents.model.forward_guard import install_model_forward_guard
from main.search_dividend.defensive import fold_defensive
from main.search_dividend.player import (SearchDividendPlayer, battle_bounds,
                                         play_one_battle)
from main.search_dividend.playoff import (PlayoffConfig, PlayoffRunner, bump_error_class,
                                          error_class, fold_playoff,
                                          playoff_error_refusal, root_failure_refusal,
                                          short_r_refusal)
from main.search_dividend.racing import fold_racing
from main.search_dividend.record import install_choice_tap
from main.search_dividend.search import SearchConfig, SearchEngine

BATTLE_FORMAT = "gen3ou"
# v2 ADDED `orientation` / `tied` / `turns` / `max_depth` and let `result` be ``"tie"``. Every
# reader defaults the new keys, so a v1 file still summarizes and still resumes — an append-only
# file outlives its schema by construction, and a version that could not be read back would make
# the append-only discipline pointless.
#
# v3 adds NO field. It is a MEANING change: before it, dice draw 0 was the sim's ``"original"``
# seed — the realized stream, which reproduced the actual turn byte-for-byte in 11 of 12 live
# decisions — so every score mixed one clairvoyant ply with `R-1` honest ones and a cell's reading
# tracked its realized `r_dice`. A version bump is the only thing that can tell those rows apart
# from these in an append-only file, and `summary.format_report` prints a banner when it sees one.
ROW_VERSION = 3


def game_seed(opponent: str, game: int, salt: int) -> str:
    """The sim seed for one CELL-INDEPENDENT game.

    Derived from ``(opponent, game, salt)`` and NOT from the arm or the budget, which is what
    makes the arms matched: cell (base, 1 s, heuristic) game 7 and cell (oracle, 3 s, heuristic)
    game 7 are the same dice."""
    h = hashlib.sha256(f"{salt}|{opponent}|{game}".encode()).hexdigest()
    return "sodium," + h[:32]


def team_pair(opponent: str, game: int, salt: int, packed_teams: Sequence[str],
              orientation: int = 0) -> tuple:
    """The (our, their) packed teams for one game-orientation — matched across arms as above.

    **Orientation 1 SWAPS them, and that is the whole point of side-swap pairing.** A mirror
    game's team draw is asymmetric: two copies of one network pilot two different teams, so at
    small n the reported win rate is mostly a measurement of which team is stronger. Playing each
    ``game`` in both orientations lets the summary difference them out — the same paired design the
    exploiter work needed when an apparent edge turned out to be pure team draw. The SEED is
    deliberately NOT varied with orientation, so the pair starts from one dice stream.
    """
    rng = random.Random(f"{salt}|{opponent}|{game}|teams")
    a, b = rng.choice(list(packed_teams)), rng.choice(list(packed_teams))
    return (a, b) if not int(orientation) else (b, a)


#: The MIRROR opponent's name. ``--opponents self`` means "the same network with search off" —
#: the sensitive contrast, because the scripted roster saturates near 90% and pins nothing, while
#: a mirror pins the no-effect point at exactly 50% by construction.
MIRROR = "self"


@dataclass
class Cell:
    arm: str
    budget: float
    opponent: str

    def key(self) -> str:
        return f"{self.arm}|{self.budget:g}|{self.opponent}"

    @property
    def is_mirror(self) -> bool:
        return self.opponent == MIRROR


def row_unit(row: dict) -> tuple:
    """A row's ``(game, orientation)`` unit of account, defaulting a pre-side-swap row to 0."""
    return (int(row["game"]), int(row.get("orientation", 0) or 0))


def _cell_tag(cell: Cell) -> str:
    """A short, STABLE account-name suffix for a cell.

    ``hash()`` on a str is salted per process (``PYTHONHASHSEED``), so using it here would give
    the same cell a different account name on every relaunch — which is invisible on the bridge
    (no server, no matchmaking) right up until someone points this driver at ``--impl off`` and
    spends an afternoon on a username collision."""
    return hashlib.md5(cell.key().encode()).hexdigest()[:6]


class ResultsFile:
    """Append-only JSONL of per-game rows, with the resume index built by reading it back."""

    def __init__(self, path: str):
        self.path = path
        self._done: Dict[str, set] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A row truncated by a kill mid-write. Skipping it is right — it will simply
                    # be replayed — but a SILENT skip would make a short file look complete, so
                    # the loader stops at the first bad line rather than reading past it.
                    break
                self._index(row)

    def _index(self, row: dict) -> None:
        """A game counts as DONE only when it FINISHED. An unfinished row (a crashed bridge
        child, a killed process, the 2026-08-23 pruned-worktree incident that scored 8 games in
        0.1 s each) stays in the file as evidence but is REPLAYED on resume — the battery's unit
        of account is finished games per cell, and marking a crash done would quietly shrink a
        cell's n. A deterministic crash then costs one extra error row per relaunch, which is
        bounded and visible, against silently under-powered cells, which is neither.

        The unit is the ORIENTATION-GAME, not the game: under side-swap a ``game`` is two battles
        and finishing one of them finishes half a pair. Rows written before side-swap existed
        carry no ``orientation`` and read as 0, so an old file resumes exactly as it always did."""
        if not int(row.get("finished", 0)):
            return
        k = Cell(row["arm"], float(row["budget"]), row["opponent"]).key()
        self._done.setdefault(k, set()).add(row_unit(row))

    def done_units(self, cell: Cell) -> set:
        """The ``(game, orientation)`` pairs already finished for this cell."""
        return set(self._done.get(cell.key(), ()))

    def done_games(self, cell: Cell) -> set:
        """The game INDEXES with at least one finished orientation (the pre-side-swap view)."""
        return {g for (g, _o) in self._done.get(cell.key(), ())}

    def n_done(self, cell: Cell) -> int:
        """How many orientation-games are finished — the battery's unit of account."""
        return len(self._done.get(cell.key(), ()))

    def append(self, row: dict) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._index(row)

    def rows(self) -> List[dict]:
        out: List[dict] = []
        if not os.path.exists(self.path):
            return out
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        break
        return out


#: A per-decision NOTE that means the search's action was not played, even though no fallback
#: reason was recorded. `policy_default` is the policy declining (no legal action, so poke-env
#: sends `/choose default` and our own action history becomes unreconstructable) and
#: `order_failed` is the chosen index refusing to map to a legal order — the second is already a
#: `FALLBACK_REASONS` member, the first was in no vocabulary at all. Both used to be counted as
#: SEARCHED decisions because `_log_decision` leaves `fallback` unset on them.
NOTE_AS_FALLBACK = ("policy_default", "order_failed")


def decision_reason(d: dict) -> Optional[str]:
    """The reason this decision was NOT a search, or ``None`` when it was one.

    ONE classifier, used by the fold and by :func:`check_decision_accounting`, so
    ``n_searched + Σfallbacks == n_decisions`` holds by construction rather than by coincidence.
    """
    fb = d.get("fallback")
    if fb:
        return str(fb)
    note = d.get("note")
    return str(note) if note in NOTE_AS_FALLBACK else None


def check_decision_accounting(row: dict) -> Optional[str]:
    """``None`` when a row's decision counters add up, else the discrepancy, spelled out.

    The identity is `n_decisions == n_searched + Σ fallbacks`. It is checked rather than assumed
    because the two halves are produced in different places (the player writes the decisions, the
    fold classifies them) and a decision that belongs to NEITHER bucket is invisible in both — the
    exact shape that let `policy_default` rows read as searched.
    """
    n = int(row.get("n_decisions", 0) or 0)
    searched = int(row.get("n_searched", 0) or 0)
    fell_back = sum(int(v or 0) for v in (row.get("fallbacks") or {}).values())
    if n == searched + fell_back:
        return None
    return (f"decision accounting: n_decisions={n} but n_searched={searched} + "
            f"fallbacks={fell_back} = {searched + fell_back}; "
            f"{n - searched - fell_back:+d} decision(s) are in neither bucket")


def summarize_decisions(decisions: Sequence[dict]) -> dict:
    """Fold one game's per-decision rows into the counters the report needs.

    ``fallbacks`` is a HISTOGRAM by reason, never a single total: "the search fell back" is not a
    finding, "the search fell back because every determinized world failed the prefix gate" is.

    🚨 **TWO ACCOUNTING DEFECTS, both closed 2026-09-22 (`gen3_decision_accounting_v1`), and the
    first is why a row could report `prefix_gate_failed: 9` beside `worlds_gate_failed: 0` — two
    statements that cannot both be true.**

    1. **The WIDTH counters were summed over SEARCHED decisions only.** `worlds_gate_failed` and
       `deadline_truncated` sat after a `continue` that every fallback took — and a decision whose
       every world failed the gate is BY DEFINITION a fallback (`prefix_gate_failed`), so the one
       row shape that most needs the counter was the one guaranteed to report zero. They are now
       summed over every decision that carries a width record, and `worlds_open_failed` (the
       counter behind `root_failed`, previously folded nowhere at all) joins them.
    2. **A decision with a NOTE but no fallback was counted as SEARCHED.** `policy_default` (the
       policy itself produced no legal action) and an `order_failed` whose search had SUCCEEDED
       both reach `_log_decision` with `fallback` unset, so `n_searched` counted decisions the
       search's action was never played on, and `n_searched + Σfallbacks` did not equal
       `n_decisions`. The classification is now exhaustive BY CONSTRUCTION — one
       :func:`decision_reason` per decision, `None` meaning searched — and
       :func:`check_decision_accounting` states the identity that follows.
    """
    fallbacks: Dict[str, int] = {}
    # `depth` and `beam` are a schema ADDITION, not a fork: ladder requirement 3 says the ladder's
    # per-decision search trace is THIS format extended, never a second one, so a new signal joins
    # the dict a reader already walks.
    realized = {"m_opp": [], "k_worlds": [], "r_dice": [], "arms": [], "elapsed": [],
                "depth": [], "beam": []}
    errors: List[str] = []
    # The exception text behind a DRIVER failure, CLASSED and counted — the field whose absence
    # made `root_failed` undiagnosable on 2026-09-22 (51/63 on the node driver and 60/63 on rust,
    # with nothing anywhere naming what `open_root` raised). Same treatment the playoff's rollout
    # errors already get, and the same reason: the identifiers vary per decision, the SHAPE is
    # what a reader acts on. Not restricted to `root_failed` — any fallback carrying a message
    # contributes, so `search_error` and `no_world` become readable off the row too.
    fallback_errors: Dict[str, int] = {}
    n_changed = 0
    n_searched = 0
    deepened = 0
    truncated = 0
    gate_failed = 0
    open_failed = 0
    for d in decisions:
        # THE WIDTH COUNTERS RUN OVER EVERY DECISION. A decision that fell back still OPENED
        # worlds and still burned clock, and the two counters that say so are precisely the ones
        # a fallback row is read for — see the docstring's defect 1.
        w = d.get("widths") or {}
        truncated += 1 if w.get("deadline_truncated") else 0
        gate_failed += int(w.get("worlds_gate_failed", 0) or 0)
        open_failed += int(w.get("worlds_open_failed", 0) or 0)
        fb = decision_reason(d)
        if fb:
            fallbacks[fb] = fallbacks.get(fb, 0) + 1
            det = d.get("error_detail")
            if det:
                if det not in errors and len(errors) < 3:
                    errors.append(det)
                for part in str(det).split(" | "):
                    bump_error_class(fallback_errors, error_class(part))
            continue
        n_searched += 1
        n_changed += 1 if d.get("changed") else 0
        realized["m_opp"].append(w.get("opp_candidates", 0))
        realized["k_worlds"].append(w.get("worlds_gated_ok", 0))
        realized["r_dice"].append(w.get("dice", 0))
        realized["arms"].append(w.get("arms_scored", 0))
        realized["elapsed"].append(w.get("elapsed_s", 0.0))
        realized["depth"].append(w.get("depth_realized", 1))
        realized["beam"].append(w.get("beam_m", 0))
        deepened += 1 if int(w.get("depth_realized", 1) or 1) > 1 else 0
    out = {
        "n_decisions": len(decisions),
        "n_searched": n_searched,
        "n_changed": n_changed,
        "n_deepened": deepened,
        "max_depth_realized": max((int(d) for d in realized["depth"]), default=0),
        "fallbacks": fallbacks,
        "fallback_details": errors,
        "fallback_errors": fallback_errors,
        "deadline_truncated": truncated,
        "worlds_gate_failed": gate_failed,
        # The counter behind `root_failed`, folded for the first time. Its absence is half of why
        # the 2026-09-22 reading could not tell a dead DRIVER from a bad WORLD off the row —
        # `search.py` has kept the two apart since the beginning and the fold threw one away.
        "worlds_open_failed": open_failed,
        "realized_mean": {k: (round(sum(v) / len(v), 3) if v else 0.0)
                          for k, v in realized.items()},
        # ADDITIVE (ladder requirement 3, 87a3f91). Zero on every arm but `playoff`, so a row
        # written by any other cell — or by any earlier version of this file — folds identically.
        **fold_playoff(decisions),
        **fold_racing(decisions),
        **fold_defensive(decisions),
    }
    # UNREACHABLE while `decision_reason` is the only classifier — which is the point. The two
    # halves of the identity are produced in different places, so the guard stands against the
    # next edit rather than against today's data, and it RAISES because a row whose decisions do
    # not add up is a defect in this file, not a property of the battle.
    bad = check_decision_accounting(out)
    if bad:
        raise ValueError(bad)
    return out


#: The per-decision wall the FIRST game of a cell sizes its bounds against, before any game has
#: been played. 5 s is the measured cost of one `playoff` decision at `--budget 60
#: --playoff-rollouts 2` (30 decisions in 148 s; 60 in 246 s) rounded up — and the estimate only
#: ever moves UP from here, because under-estimating kills a healthy battle while over-estimating
#: merely delays a livelock verdict, and only the first of those deletes a measurement.
FIRST_GAME_DECISION_S = 5.0


#: The error a game gets when it produced neither an exception nor an outcome. Named, because the
#: alternative is a row that says nothing went wrong on a battle that never happened — measured
#: 2026-08-23, when a pruned worktree took ``local_sim_bridge.js`` out from under a running battery
#: and 8 straight games recorded clean-looking and empty.
NEVER_FINISHED = ("battle_never_finished: no exception, no result — the bridge child likely died "
                  "at spawn (deleted worktree? missing node/dist?)")
NEVER_CREATED = ("battle_never_created: run_local_battles returned without ever creating a battle "
                 "object for this player — the bridge child produced no protocol at all")


def finalize_row(row: dict) -> dict:
    """Enforce the one invariant every result row obeys, and repair it rather than trust it.

    **A row carries a real outcome XOR a named error. Never neither.** A game that neither threw
    nor produced a winner is the single most dangerous shape this file can hold: it reads as a
    played battle to every consumer, it dilutes a cell's win rate toward whatever it happened to
    be, and nothing in it points at a cause. So the invariant is checked HERE, at the one place
    rows are made, instead of being an expectation about the call sites above.
    """
    result = row.get("result")
    if result in ("win", "loss", "tie"):
        return row
    if not row.get("error"):
        row["error"] = NEVER_CREATED if not row.get("battle_created", True) else NEVER_FINISHED
    row["result"] = "unfinished"
    row["finished"] = 0
    return row


def build_players(model, mappings, cfg: SearchConfig, opponent_name: str, *,
                  pool_packed: Optional[Sequence[str]] = None, tag: str = "",
                  playoff_cfg: Optional[PlayoffConfig] = None):
    """The trainee (search-wrapped) + one scripted bot, both bridge-transport (no server).

    Teams are injected per game by :func:`set_teams`, so the teambuilders here are placeholders
    that the driver overrides — the battery must not let each player draw its own team."""
    from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration
    from agents.training.eval_callback import build_eval_opponents
    from utils.teambuilder import Gen3Teambuilder
    from utils.team_loader import TeamLoader

    # 🚨 gen3_extractor_forward_guard_v1. THIS is the function that creates the hazard: every
    # player built below shares ONE `model`, and the searched side runs `SearchEngine.choose` in
    # a `run_in_executor` worker while the other side decides on POKE_LOOP (and, on the `playoff`
    # arm, while whole rollout battles decide there too). The extractor keeps its per-forward
    # state on `self`, so two concurrent forwards corrupt each other — measured at 1,063 failures
    # in 2,400 interleaved forwards, one class of which is the `ValueThreatInject shape mismatch:
    # tokens (1, 6) vs rows (9, 6)` that killed the 2026-09-22 playoff repro outright. Installed
    # HERE rather than inside the extractor so that a single-threaded tree never pays for a lock.
    install_model_forward_guard(model)
    teams = list(pool_packed) if pool_packed else None
    tb = Gen3Teambuilder(TeamLoader().get_all_teams()) if teams is None else _FixedTeam(teams[0])
    if opponent_name == "self":
        # The MIRROR opponent (owner-ordered): the SAME network with search structurally OFF —
        # a base-arm SearchDividendPlayer, so the two sides differ in exactly one thing, the
        # search. This is the sensitive contrast: the scripted roster is saturated (~90% either
        # way, the dividend hides in the ceiling), while a mirror pins the no-effect point at
        # 50% by construction. `dataclasses.replace` keeps every cap/score knob identical to
        # the searched side's config rather than re-deriving a second one that could drift.
        from dataclasses import replace
        opp_engine = SearchEngine(model=model, mappings=mappings,
                                  cfg=replace(cfg, arm="base", budget_s=0.0),
                                  pool_packed=pool_packed)
        opp = SearchDividendPlayer(
            model=model, team=tb, battle_format=BATTLE_FORMAT,
            server_configuration=LocalhostServerConfiguration, mappings=mappings,
            account_configuration=AccountConfiguration(f"SDivSelf{tag}", "password"),
            start_listening=False, engine=opp_engine, opp_player=None)
    else:
        (_name, opp) = build_eval_opponents(
            LocalhostServerConfiguration, tb, [opponent_name], tag=tag, start_listening=False)[0]
    # The second-stage scorer exists ONLY where a second stage runs: the `playoff` ARM, and the
    # `defensive` ROOT STRATEGY when --defensive-confirm asked for one. Building it
    # unconditionally would be harmless (it is never called) but would misdescribe every other
    # cell's engine. The defensive path overrides `rollouts` with its OWN flag, so one runner
    # cannot silently inherit the playoff arm's much larger R.
    runner = None
    dcfg = cfg.defensive_cfg()
    if cfg.arm == "playoff":
        runner = PlayoffRunner(model=model, mappings=mappings, battle_format=BATTLE_FORMAT,
                               cfg=playoff_cfg or PlayoffConfig(), tag=tag)
    elif dcfg is not None and int(dcfg.confirm_rollouts) > 0:
        from dataclasses import replace as _replace
        runner = PlayoffRunner(
            model=model, mappings=mappings, battle_format=BATTLE_FORMAT,
            cfg=_replace(playoff_cfg or PlayoffConfig(), rollouts=int(dcfg.confirm_rollouts)),
            tag=tag)
    engine = SearchEngine(model=model, mappings=mappings, cfg=cfg, pool_packed=pool_packed,
                          playoff=runner)
    me = SearchDividendPlayer(
        model=model, team=tb, battle_format=BATTLE_FORMAT,
        server_configuration=LocalhostServerConfiguration, mappings=mappings,
        account_configuration=AccountConfiguration(f"SDiv{tag}", "password"),
        start_listening=False, engine=engine, opp_player=opp)
    if opponent_name == "self":
        # Each side's in-flight record builder names the OTHER side; wire the back-reference
        # only after both exist (the ctor merely stores it).
        opp._opp_player = me
    return me, opp, engine


class _FixedTeam:
    """A teambuilder whose team the driver sets per game (poke-env calls ``yield_team``)."""

    def __init__(self, packed: str):
        self.packed = packed

    def yield_team(self) -> str:
        return self.packed


def set_teams(me, opp, our_packed: str, their_packed: str) -> None:
    """Pin both sides' teams for the next battle. ``run_local_battles`` calls ``get_next_team()``
    on each player, which reads ``player._team.yield_team()``."""
    me._team = _FixedTeam(our_packed)
    opp._team = _FixedTeam(their_packed)


async def run_cell(cell: Cell, *, model, mappings, cfg: SearchConfig, games: int,
                   results: ResultsFile, salt: int, impl: str,
                   pool_packed: Sequence[str], progress=None,
                   side_swap: bool = False, games_start: int = 0,
                   playoff_cfg: Optional[PlayoffConfig] = None) -> int:
    """Play the missing orientation-games of one cell, appending a row each.

    With ``side_swap`` every ``game`` index is played TWICE — orientation 0 and orientation 1 — so
    the summary can difference out the team draw. ``games`` therefore keeps meaning "distinct team
    draws" and the battle count doubles, which is the honest way round: halving ``games`` to keep
    the battle count would halve the number of independent draws, which is the thing that actually
    bounds the interval.

    ``games_start`` plays the HALF-OPEN index window ``[games_start, games_start + games)`` instead
    of ``[0, games)``. It is a SHARDING knob and nothing more: :func:`game_seed` and
    :func:`team_pair` are functions of the index alone, so game 500 is the same dice and the same
    team draw whichever process plays it, and two shards over disjoint windows concatenate into
    exactly the file one process would have written. That is what lets a cell whose per-game wall
    exceeds the session budget be split across cores without any arm becoming a different
    experiment — the alternative (a different ``--games-seed`` per shard) would re-use index 0 for
    two different battles and silently break the pairing the summary does on it.
    """
    install_choice_tap()
    orientations = (0, 1) if side_swap else (0,)
    done = results.done_units(cell)
    lo = int(games_start)
    todo = [(g, o) for g in range(lo, lo + games) for o in orientations if (g, o) not in done]
    if not todo:
        return 0
    me, opp, engine = build_players(model, mappings, cfg, cell.opponent,
                                    pool_packed=pool_packed, tag=_cell_tag(cell),
                                    playoff_cfg=playoff_cfg)
    played = 0
    # The seed for the FIRST game's bound estimate. Deliberately pessimistic — over-estimating
    # costs a later livelock detection, under-estimating kills a healthy battle, and only the
    # second of those silently deletes a measurement.
    per_decision_s = FIRST_GAME_DECISION_S
    try:
        for g, orient in todo:
            ours, theirs = team_pair(cell.opponent, g, salt, pool_packed, orient)
            set_teams(me, opp, ours, theirs)
            seed = game_seed(cell.opponent, g, salt)
            t0 = time.monotonic()
            # 🚨 THE PER-BATTLE BOUNDS ARE SIZED FROM THIS CELL, not from a constant. On a search
            # arm one decision is the longest gap between two protocol chunks and the whole battle
            # is as long as the decisions make it — both are set by the argv, and the runner's
            # 180 s default threw away the longer orientation of every side-swapped playoff cell
            # (measured: a 246.6 s game reported as `livelock, not a stall` at 180 s). The
            # estimate is REALIZED, carried forward from the games already played in this cell,
            # because game 1 is the only one that has to guess.
            idle_s, total_s = battle_bounds(per_decision_s)
            try:
                out = await play_one_battle(me, opp, battle_format=BATTLE_FORMAT,
                                            seed=seed, impl=impl,
                                            idle_budget_s=idle_s, total_budget_s=total_s)
                err = out.get("error")
            except Exception as e:                    # noqa: BLE001
                # A crashed game is RECORDED as an error row, not dropped. A dropped game biases
                # the win rate by whatever made it crash. (`play_one_battle` already keeps the
                # decisions of a battle that timed out; this stays as the backstop for a failure
                # it cannot reach, e.g. one raised before the battle object exists.)
                out = {"outcome": "unfinished", "won": 0, "tied": 0, "finished": 0,
                       "battle_created": True, "decisions": []}
                err = f"{type(e).__name__}: {e}"
            row = finalize_row({
                "v": ROW_VERSION, "arm": cell.arm, "budget": cell.budget,
                "opponent": cell.opponent, "game": g, "orientation": int(orient),
                "result": out.get("outcome"),
                "finished": int(out.get("finished", 0)), "won": int(out.get("won", 0)),
                "tied": int(out.get("tied", 0)), "turns": int(out.get("turns", 0) or 0),
                "battle_created": bool(out.get("battle_created", False)),
                "wall_s": round(time.monotonic() - t0, 3),
                "seed": seed,
                # The score mode the search was ASKED for — `effective_score`, not the raw flag,
                # because `--root-strategy defensive` names its own head and a row recording
                # "auto" would misdescribe which leaf the cell was measured on.
                "score_mode": cfg.effective_score(), "search_impl": cfg.search_impl,
                "root_strategy": cfg.root_strategy,
                "max_depth": int(getattr(cfg, "max_depth", 1)),
                "playoff_rollouts": (int(playoff_cfg.rollouts) if playoff_cfg
                                     and cfg.arm == "playoff" else 0),
                "error": err,
                **summarize_decisions(out.get("decisions") or []),
            })
            results.append(row)
            played += 1
            # REALIZED, from the game just played — the same "seed it from the previous game's
            # cost" rule the playoff's own rollout-cost model needs. A game with no decisions
            # (a crash before the first request) leaves the estimate alone rather than driving it
            # to an arbitrary number.
            if int(row.get("n_decisions", 0) or 0) > 0:
                per_decision_s = max(per_decision_s,
                                     float(row["wall_s"]) / int(row["n_decisions"]))
            if progress is not None:
                progress(row)
            # 🚨 THE REALIZED-R GUARD, on the FIRST game rather than after the cell. A playoff
            # cell whose per-decision budget could not buy 2xR rollouts realizes a different R
            # than its flags name and, below MIN_PAIRS, cannot conclude at all — measured
            # 2026-09-19 as two cells at R=4 and R=8 producing byte-identical no-op behaviour.
            # The row is APPENDED first so the evidence for the refusal is on disk.
            # 🚨 `root_failure_refusal` is FIRST and applies to EVERY search arm, not just
            # `playoff`: a dead search driver never reaches a screen, so the other two guards see
            # `attempted == 0` and stay silent (2026-09-22 — the hole that let `root_failed` on
            # 60 of 63 decisions report a clean cell).
            msg = root_failure_refusal(row) if cfg.arm != "base" else None
            if not msg and (playoff_cfg is not None and cfg.arm == "playoff"
                            and not playoff_cfg.allow_short_r):
                msg = (playoff_error_refusal(row)
                       or short_r_refusal(row, int(playoff_cfg.rollouts)))
            if msg:
                raise SystemExit(f"[search_dividend] {msg}")
    finally:
        engine.close()
    return played
