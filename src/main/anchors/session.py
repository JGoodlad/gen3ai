"""OUR side, and the watchdog that refuses to report nothing.

Two jobs.

**1. Run our checkpoint through `main.play`'s own code path.** Not a copy of it — `play.main` is
called, so the client, the account, the connect-or-raise deadline and the 250-turn forfeit limit
are exactly what a ladder session uses. Three patches are installed around it, each of which adds
something a MEASUREMENT needs and a session runner has no reason to print:

* a **team source** (our 719-team pool, or a directory of Showdown exports) with a seeded private
  draw RNG, and the team actually yielded recorded per game;
* a **per-battle observer** that appends one row as each battle finishes, so a series that dies
  halfway keeps what completed; and
* a **regime observer** that records the ``stochastic`` keyword each decision REALLY received,
  rather than the flag we believe we passed.

**2. Watch the pair, and FAIL LOUDLY with a named cause.** This is the half that hazard H-B bought:
when our side forfeits at turn 250, Metamon's long-tail handler force-resets and then calls itself
(~985 levels, then a ``RecursionError``), the process dies, and a naive harness sits there with a
live client, no opponent, and nothing to report. Four named causes, never a silent zero:

=====================  ===================================================================
``peer_never_ready``   the peer process never printed its online banner
``peer_exited``        the peer died before the series finished (rc and its log tail named)
``no_first_game``      nothing finished inside the first-game deadline
``no_progress``        games stopped finishing inside the progress deadline
=====================  ===================================================================

Each is returned as a ``failure`` dict that reaches ``summary.json``'s ``status: "FAILED"`` and a
non-zero exit code, **with whatever games did finish attached** — a partial n is honest only when
it is labelled partial.
"""
from __future__ import annotations

import asyncio
import glob
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from main.anchors.peers import TEAM_FILE_GLOBS, PeerPlan


class SeriesFailure(RuntimeError):
    """A named, reportable failure of the pair — never raised for an ordinary loss."""

    def __init__(self, cause: str, detail: str) -> None:
        super().__init__(f"{cause}: {detail}")
        self.cause = cause
        self.detail = detail

    def as_dict(self) -> Dict[str, str]:
        return {"cause": self.cause, "detail": self.detail}


# ------------------------------------------------------------------ the team source we draw from
def load_team_texts(directory: Path) -> "list[tuple[str, str]]":
    """``(basename, text)`` for every team file in ``directory``, each ``.strip()``ed.

    The glob allowlist lives in :data:`main.anchors.peers.TEAM_FILE_GLOBS` so BOTH sides count the
    same thing — Metamon's `competitive` directory ships an `index.csv` beside its 20 teams, and a
    count that included it would report a 20-vs-21 team-source asymmetry that does not exist.

    🚨 **The ``.strip()`` is load-bearing history, not tidiness.** Before 2026-09-14 our vendored
    fork's ``Teambuilder.parse_showdown_team`` skipped a split chunk only when it was exactly
    ``""``, so a file ending ``"\\n\\n\\n"`` — which every Metamon ``competitive`` file does —
    produced an EMPTY 7th Pokemon, Showdown rejected the packed team, and the challenge never
    became a battle: the series HUNG. The fork is fixed and ``Gen3Teambuilder`` now raises on an
    oversize pack, so this is the third line of defence rather than the first; it stays because
    normalising a file we did not write costs nothing and the failure it prevents costs a session.
    """
    seen: Dict[str, str] = {}
    for pattern in TEAM_FILE_GLOBS:
        for path in sorted(glob.glob(os.path.join(str(directory), pattern))):
            name = os.path.basename(path)
            if name.startswith("."):
                continue
            with open(path) as fh:
                seen[name] = fh.read().strip()
    return sorted(seen.items())


def build_team_source(spec: Dict[str, Any], seed: Optional[int], draw_log: List[Optional[str]]):
    """A ``Gen3Teambuilder`` over the configured source, plus a label for every pool index.

    ``spec`` is ``{"kind": "pool"}`` or ``{"kind": "dir", "path": ...}``.
    """
    from utils.teambuilder import Gen3Teambuilder

    kind = spec.get("kind")
    if kind == "pool":
        from utils.team_loader import TeamLoader

        tb = Gen3Teambuilder([t.strip() for t in TeamLoader().get_all_teams()], rng_seed=seed)
        # Our pool's own fingerprints ARE the Metamon export's filenames (`<sha>.gen3ou_team`),
        # so the two sides' team identifiers join with no translation table.
        labels = list(tb.get_pool_team_keys())
    elif kind == "dir":
        directory = Path(spec["path"])
        items = load_team_texts(directory)
        if not items:
            raise SeriesFailure("no_teams",
                                f"{directory} holds no team files matching {TEAM_FILE_GLOBS}")
        tb = Gen3Teambuilder([text for _, text in items], rng_seed=seed)
        # `Gen3Teambuilder` DROPS locally-invalid teams, so labels must follow the SURVIVORS, not
        # the original file order — otherwise every recorded team name is off by the drop count.
        from utils.bridge.team_validator import validate_teams_locally

        verdicts = validate_teams_locally("gen3ou", [text for _, text in items])
        labels = [name for (name, _), v in zip(items, verdicts) if v.get("valid")]
    else:
        raise SeriesFailure("bad_team_source", f"unknown team source kind {kind!r}")

    tb._label_by_idx = labels

    inner_yield = tb.yield_team

    def yield_team():
        packed = inner_yield()
        idx = tb._last_pool_idx
        draw_log.append(labels[idx] if (idx is not None and idx < len(labels)) else None)
        return packed

    tb.yield_team = yield_team
    return tb


# ------------------------------------------------------------------------- the per-battle record
@dataclass
class BattleRecord:
    """One finished battle, as our side saw it. The tool's authoritative per-game view."""

    battle_tag: str
    turns: int
    won: Optional[bool]
    finished: Optional[bool]
    hit_forfeit_limit: bool
    our_team: Optional[str]
    n_decisions: Optional[int]
    n_defaults: Optional[int]
    n_redecides: Optional[int]
    t_finished: float

    @property
    def result(self) -> str:
        """``won is None`` with ``finished`` true is a TIE — poke-env's own three flags.

        Ties count in the denominator and not the numerator, as both 2026-09-14 batteries did.
        """
        if self.won is True:
            return "win"
        if self.won is False:
            return "loss"
        return "tie"


@dataclass
class OurSideState:
    records: List[BattleRecord] = field(default_factory=list)
    draws: List[Optional[str]] = field(default_factory=list)
    stochastic_kwargs: List[bool] = field(default_factory=list)
    decision_times: List[float] = field(default_factory=list)
    player: Any = None
    last_progress: float = field(default_factory=time.time)
    error: Optional[str] = None


def install_our_side(state: OurSideState, team_spec: Dict[str, Any], team_seed: Optional[int],
                     forfeit_limit: int, server_config: Any) -> Callable[[], None]:
    """Patch `main.play` + `RLPlayer` for one half-series. Returns the undo.

    Patching the RLPlayer CLASS (not a module global) is what makes the observers reach the code
    under test: `play.build_model_player` constructs the instance itself, so there is no other
    seam, and a class attribute is resolved at call time by every instance.
    """
    import main.play as play
    from agents.inference.player import RLPlayer

    saved_build_tb = play.build_teambuilder
    saved_resolve = play.resolve_server
    saved_build_player = play.build_model_player
    saved_finish = RLPlayer._battle_finished_callback
    saved_predict = RLPlayer._predict_best_action

    seen: set = set()

    def build_teambuilder(team_file, pool):
        return build_team_source(team_spec, team_seed, state.draws)

    def resolve_server(server, port):
        return server_config

    def build_model_player(args, teambuilder, cfg, account):
        player = saved_build_player(args, teambuilder, cfg, account)
        # Stashed so the driver can wait on a REAL login rather than a sleep — the acceptor must
        # be online before the challenger's first `/challenge`, or Showdown drops it and the
        # challenger waits forever.
        state.player = player
        return player

    def finished(self, battle):
        view = battle.strict_view()
        tag = view.battle_tag
        if tag not in seen:
            seen.add(tag)
            n = len(seen)
            state.records.append(BattleRecord(
                battle_tag=tag,
                turns=view.turn,
                won=getattr(battle, "won", None),
                finished=getattr(battle, "finished", None),
                # A cap forfeit is reported by the server as an ordinary loss; only the turn count
                # separates it from a decisive one (agents.training.trace_result, same rule).
                hit_forfeit_limit=view.turn >= forfeit_limit,
                our_team=(state.draws[n - 1] if n <= len(state.draws) else None),
                n_decisions=getattr(self, "_n_decisions", None),
                n_defaults=getattr(self, "_n_defaults", None),
                n_redecides=getattr(self, "_n_redecides", None),
                t_finished=time.time(),
            ))
            state.last_progress = time.time()
            rec = state.records[-1]
            print(f"[anchors] game {n}: {rec.result} in {rec.turns} turns "
                  f"(cap={rec.hit_forfeit_limit}, team={rec.our_team})", flush=True)
        return saved_finish(self, battle)

    def predict(self, *a, **k):
        # THE REGIME VERIFICATION for our half: the `stochastic` keyword the decision REALLY
        # received, not the flag we think we passed. Signature is
        # `_predict_best_action(self, battle, stochastic=False, ...)`; `choose_move` passes it by
        # keyword, so the positional fallback is belt and braces.
        value = bool(k["stochastic"] if "stochastic" in k else (a[1] if len(a) > 1 else False))
        if value not in state.stochastic_kwargs:
            state.stochastic_kwargs.append(value)
        t0 = time.perf_counter()
        try:
            return saved_predict(self, *a, **k)
        finally:
            state.decision_times.append(time.perf_counter() - t0)

    play.build_teambuilder = build_teambuilder
    play.resolve_server = resolve_server
    play.build_model_player = build_model_player
    RLPlayer._battle_finished_callback = finished
    RLPlayer._predict_best_action = predict

    def undo() -> None:
        play.build_teambuilder = saved_build_tb
        play.resolve_server = saved_resolve
        play.build_model_player = saved_build_player
        RLPlayer._battle_finished_callback = saved_finish
        RLPlayer._predict_best_action = saved_predict

    return undo


# --------------------------------------------------------------------------------- the watchdog
def log_has(path: Path, pattern: str) -> bool:
    if not path.exists():
        return False
    return re.search(pattern, path.read_text(errors="replace")) is not None


def log_tail(path: Path, n: int = 20) -> str:
    if not path.exists():
        return f"(no log at {path})"
    lines = path.read_text(errors="replace").splitlines()[-n:]
    return f"last {len(lines)} lines of {path}:\n" + "\n".join(lines)


def start_peer(plan: PeerPlan, nice: int = 10) -> subprocess.Popen:
    """Start the peer, recording its PID. ``nice`` because a training arm owns this box."""
    plan.log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(plan.log_path, "w", buffering=1)
    argv = (["nice", "-n", str(nice)] if nice else []) + plan.argv
    proc = subprocess.Popen(argv, cwd=str(plan.cwd), env=plan.env,
                            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"[anchors] peer {plan.label} pid={proc.pid} -> {plan.log_path}", flush=True)
    return proc


async def await_peer_exit(proc: Optional[subprocess.Popen], timeout_s: float,
                          poll_s: float = 1.0) -> bool:
    """Let the peer finish on its OWN and write its report before anything terminates it.

    🚨 Measured 2026-09-14: our side finishes its last battle a beat before the peer does, so a
    harness that kills the peer the moment its own games are done kills it MID-REPORT — and the
    cell then has games but no ``argmax_match_rate``, i.e. a win rate whose regime is UNVERIFIED.
    That is precisely the number this tool refuses to emit, so the wait is part of the
    measurement, not tidiness. Both peers exit by themselves once ``--total-battles`` is reached.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is None or proc.poll() is not None:
            return True
        await asyncio.sleep(poll_s)
    return False


def stop_peer(proc: Optional[subprocess.Popen], grace_s: float = 15.0) -> None:
    """Terminate EXACTLY the PID we started. Idempotent, never raises, never `pkill`."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            pass


async def await_peer_ready(proc: subprocess.Popen, plan: PeerPlan, timeout_s: float,
                           poll_s: float = 2.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if log_has(plan.log_path, plan.ready_pattern):
            print(f"[anchors] peer {plan.label} online", flush=True)
            return
        if proc.poll() is not None:
            raise SeriesFailure(
                "peer_exited",
                f"{plan.label} exited rc={proc.returncode} before it came online. "
                + log_tail(plan.log_path))
        await asyncio.sleep(poll_s)
    raise SeriesFailure(
        "peer_never_ready",
        f"{plan.label} never matched /{plan.ready_pattern}/ within {timeout_s:g}s. "
        + log_tail(plan.log_path))


async def await_our_login(state: OurSideState, timeout_s: float) -> None:
    """Wait for a REAL login, not a sleep.

    `logged_in` is an `asyncio.Event` created in poke-env's OWN loop thread, so it must not be
    awaited from this loop — its `is_set()` is polled instead. A challenge aimed at a user who is
    not yet online is dropped by Showdown and the challenger then waits forever, which is why this
    is a gate and not an optimisation.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        player = state.player
        if player is not None:
            try:
                if player.ps_client.logged_in.is_set():
                    print(f"[anchors] our side online as {player.username}", flush=True)
                    return
            except AttributeError:
                pass
        await asyncio.sleep(1.0)
    raise SeriesFailure("our_side_never_ready",
                        f"our client did not log in within {timeout_s:g}s")


async def disconnect_our_side(state: OurSideState, timeout_s: float = 30.0) -> None:
    """Close OUR websocket at the end of a half — measured 2026-09-14, and not optional.

    `play.main` returns when the battles are done but poke-env keeps the socket open, so the
    SECOND half's client logs in while the first is still holding the name. Showdown answers with
    `|nametaken|`, which the fork logs and then **continues as a guest** with a server-assigned
    name — after which the peer's `/challenge` is addressed to a user that no longer exists and the
    half dies 8 minutes later with Metamon's own "Agent is not challenging". The failure was NAMED
    by the watchdog (peer_exited), which is the point of it, but the cause was ours.

    Belt AND braces: each half also plays under its own username suffix, so a socket that somehow
    survives this cannot collide with the next half either.
    """
    player = state.player
    if player is None:
        return
    try:
        await asyncio.wait_for(player.ps_client.stop_listening(), timeout=timeout_s)
    except Exception as exc:                        # noqa: BLE001 - teardown must never mask a result
        print(f"[anchors] warning: could not close our websocket cleanly: {exc}", flush=True)


async def watch(proc: subprocess.Popen, plan: PeerPlan, state: OurSideState, expected: int,
                first_game_timeout_s: float, progress_timeout_s: float,
                poll_s: float = 2.0) -> None:
    """Raise a NAMED :class:`SeriesFailure` the moment the pair stops making progress.

    Runs beside our side's task; whichever finishes first decides the half.
    """
    started = time.time()
    while True:
        await asyncio.sleep(poll_s)
        done = len(state.records)
        if done >= expected:
            return
        if proc.poll() is not None:
            raise SeriesFailure(
                "peer_exited",
                f"{plan.label} exited rc={proc.returncode} after {done}/{expected} games. "
                "Metamon answers a 250-turn forfeit with unbounded recursion (hazard H-B), so a "
                "mid-series death is expected there and must never read as 'no result'. "
                + log_tail(plan.log_path))
        idle = time.time() - (state.last_progress if done else started)
        budget = progress_timeout_s if done else first_game_timeout_s
        if idle > budget:
            raise SeriesFailure(
                "no_progress" if done else "no_first_game",
                f"{done}/{expected} games finished and nothing has completed for {idle:.0f}s "
                f"(budget {budget:.0f}s). A rejected team, a dropped challenge or a wedged login "
                "all present as a HANG rather than an error. " + log_tail(plan.log_path))
