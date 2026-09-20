"""One anchor read, end to end: two role-balanced half-series against one external opponent.

**Role is balanced INSIDE every read, never confounded with one.** Showdown makes the CHALLENGER
p1, and p1/p2 is not a priori neutral; measured over 800 games on 2026-09-14 the role effect is
NOT DETECTED (+0.050 / −0.020 pooled), which is exactly what makes balancing it cheap insurance
rather than a wasted axis — inside a single 50-game half it looks large and inconsistent, because
that is the shape of a 50-game coin.

**Both sides move together or the cell is not matched.** ``--regime greedy`` sets our side to
``--temperature 0`` and the peer to its own greedy operation (for Metamon, ``get_actions(
sample=False)``; Foul Play has no sampling knob at all and is refused for ``t1`` by the CLI). Each
side's regime is then VERIFIED per decision and the verification is written onto every row.

The half-series is driven in ONE process: our side is `main.play` on this event loop, the peer is a
subprocess, and :func:`main.anchors.session.watch` runs beside them so a dead peer or a stalled
series becomes a named failure instead of a report with nothing in it.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from main.anchors import peers as peers_mod
from main.anchors.results import CellSpec, GameRow
from main.anchors.session import (
    OurSideState,
    SeriesFailure,
    await_our_login,
    await_peer_exit,
    await_peer_ready,
    disconnect_our_side,
    install_our_side,
    read_peer_battles,
    start_peer,
    stop_peer,
    watch,
    watch_peer_pair,
)

#: ``(our play.py mode, the peer's role, the username suffix)`` for each half. The names say who
#: sends the challenge; the SUFFIX is a hazard fix, not decoration — see :func:`half_usernames`.
HALVES = {
    "ours_challenge": ("challenge", "acceptor", "1"),
    "peer_challenge": ("accept", "challenger", "2"),
}


def half_usernames(plan: "SeriesPlan", half: str) -> "tuple[str, str]":
    """``(our name, the peer's name)`` for this half — each carrying the half's own suffix.

    🚨 **A name still held by the previous half is not an error, it is a GUEST.** Showdown answers
    a second login under a live name with ``|nametaken|``, and poke-env's fork logs it and carries
    on under a server-assigned guest name; the peer's ``/challenge`` is then addressed to a user
    that does not exist and the half hangs until the peer's own timeout. Measured here on
    2026-09-14. The sockets are closed between halves as well — this is the second lock on the
    same door, because the failure it prevents costs eight minutes and produces no games.

    Both names stay inside Showdown's 18-character ceiling: 19+ gets a ';'-prefixed REFUSAL from
    `action.php` that a client may accept as an assertion and then hang on forever.
    """
    _mode, _role, suffix = HALVES[half]
    return plan.our_username + suffix, plan.peer_username + suffix


@dataclass
class SeriesPlan:
    """Everything one anchor read needs, resolved. Built by the CLI, printed by ``--dry-run``."""

    opponent: str                 # "metamon:SmallRL" / "foulplay"
    opponent_kind: str
    opponent_agent: str
    regime: str                   # "greedy" / "t1"
    regime_matched: bool
    teamset: str
    games: int
    battle_format: str
    model_spec: str
    model_zip: str
    model_step: Optional[int]
    model_rung: str
    device: str
    server_uri: str
    server_port: Optional[int]
    started_server: bool
    out_dir: Path
    our_username: str
    peer_username: str
    team_seed: int
    search_time_ms: Optional[int]
    search_parallelism: Optional[int]
    forfeit_turn_limit: int
    connect_timeout_s: float
    peer_ready_timeout_s: float
    first_game_timeout_s: float
    progress_timeout_s: float
    nice: int
    showdown_pin: str
    our_team_spec: Dict[str, Any] = None  # type: ignore[assignment]
    #: "model" or "bot:<name>" — see :func:`main.anchors.session.install_our_side`.
    our_side: str = "model"
    #: "auto" / "bare" / "foreign" — which loader builds our checkpoint.
    model_loader: str = "auto"
    #: WHICH transport served this read — "rust" (the websocket front end over the Rust bridge,
    #: the DEFAULT: no Node process exists), "node" (deps/pokemon-showdown) or "external" (a
    #: `--server-uri` this tool did not start, and therefore cannot vouch for).
    server_impl: str = "node"
    #: The transport's own identity, stamped on every row — `ws_frontend@<head>+rust:<binary>` or
    #: `showdown:<pin>`. A win rate that did not say which transport produced it could not be
    #: compared with one that did.
    server_version: str = ""
    #: `--server rust` only: the reproducibility pair. There is no Node counterpart.
    seed_base: Optional[int] = None
    capture_dir: Optional[Path] = None

    @property
    def our_side_is_bot(self) -> bool:
        return self.our_side.startswith("bot:")

    @property
    def our_side_is_peer(self) -> bool:
        """Our side is ANOTHER external agent — an anchor-vs-anchor cell, with no player of ours
        in the process at all."""
        return self.our_side.startswith("metamon:") or self.our_side == "foulplay"

    def half_sizes(self) -> Dict[str, int]:
        """``games`` split across the two roles; an odd game goes to the first half and is
        recorded there rather than silently dropped."""
        half = self.games // 2
        return {"ours_challenge": self.games - half, "peer_challenge": half}


def our_argv(plan: SeriesPlan, mode: str, n_games: int, our_name: str = "",
             peer_name: str = "") -> List[str]:
    """The argv `main.play`'s OWN parser receives. Spelled out rather than hand-built into a
    namespace, so every default this tool does not set comes from `play.build_parser()` — the
    forfeit limit included."""
    return [
        "--mode", mode,
        "--server", "local",
        # Unused when `--server-uri` was given (resolve_server is patched), but a real value keeps
        # the parser honest and the reserved-port refusal reachable.
        "--port", str(plan.server_port or 9500),
        "--format", plan.battle_format,
        # A bot our-side has no checkpoint, but `play.main` refuses --mode challenge/accept
        # without a --model, so the SPEC goes here and the patched `build_model_player` ignores
        # it. It is printed by --dry-run and stamped as `model_spec` on every row, so nothing
        # about which policy played is hidden by the placeholder.
        "--model", plan.model_zip or plan.our_side,
        "--device", plan.device,
        "--username", our_name or plan.our_username,
        "--opponent", peer_name or plan.peer_username,
        "--n-battles", str(n_games),
        "--concurrency", "1",
        # THE REGIME, our half. `--temperature 0` is `play.py`'s documented measurement setting and
        # the protocol every other strength number in this project is taken under.
        "--temperature", "0.0" if plan.regime == "greedy" else "1.0",
        "--connect-timeout", str(plan.connect_timeout_s),
        "--forfeit-turn-limit", str(plan.forfeit_turn_limit),
    ]


def peer_plan(plan: SeriesPlan, cfg: Any, role: str, n_games: int, half: str) -> Any:
    our_name, peer_name = half_usernames(plan, half)
    adapter = peers_mod.PEERS[plan.opponent_kind]
    return adapter.plan(
        cfg=cfg.opponent(plan.opponent_kind),
        agent=plan.opponent_agent,
        regime=plan.regime,
        teamset=plan.teamset,
        battle_format=plan.battle_format,
        server_uri=plan.server_uri,
        username=peer_name,
        opponent_username=our_name,
        role=role,
        n_games=n_games,
        team_seed=plan.team_seed + 1,
        search_time_ms=plan.search_time_ms,
        search_parallelism=plan.search_parallelism,
        out_dir=plan.out_dir / half,
    )


def our_peer_plan(plan: SeriesPlan, cfg: Any, role: str, n_games: int, half: str) -> Any:
    """The plan for OUR side when our side is itself an external peer. Mirrors :func:`peer_plan`
    with the two usernames and the two roles swapped, and its own team seed."""
    our_name, peer_name = half_usernames(plan, half)
    kind, _, agent = plan.our_side.partition(":")
    adapter = peers_mod.PEERS[kind]
    return adapter.plan(
        cfg=cfg.opponent(kind),
        agent=agent,
        regime=plan.regime,
        teamset=plan.teamset,
        battle_format=plan.battle_format,
        server_uri=plan.server_uri,
        username=our_name,
        opponent_username=peer_name,
        # The roles are OPPOSITE the opponent's: one side challenges, the other accepts.
        role="challenger" if role == "acceptor" else "acceptor",
        n_games=n_games,
        team_seed=plan.team_seed,
        search_time_ms=plan.search_time_ms,
        search_parallelism=plan.search_parallelism,
        out_dir=plan.out_dir / half / "ours",
    )


async def run_peer_pair_half(plan: SeriesPlan, cfg: Any, half: str, n_games: int,
                             ) -> Tuple[List[Any], Dict[str, Any], Optional[SeriesFailure]]:
    """One half of an ANCHOR-vs-ANCHOR cell — two external peers, nothing of ours in the middle.

    🚨 **Role order is the same load-bearing fact as everywhere else**: Showdown drops a challenge
    aimed at a user who is not online, so the ACCEPTOR must have printed its banner before the
    CHALLENGER starts. For the 200M `SyntheticRLV2` that banner is minutes away — it builds the
    network and loads 804 MB before its client connects — which is why the acceptor is started and
    awaited first rather than both being launched together.

    The per-game record comes from the acting side's own battle CSV (`read_peer_battles`); there
    is no `RLPlayer` here to observe, and the weaker instrument is NAMED on the rows it produces.
    """
    mode, their_role, _suffix = HALVES[half]
    our_name, peer_name = half_usernames(plan, half)
    peers_mod.check_username(our_name, "our")
    peers_mod.check_username(peer_name, "peer")

    ours = our_peer_plan(plan, cfg, their_role, n_games, half)
    theirs = peer_plan(plan, cfg, their_role, n_games, half)
    for pp in (ours, theirs):
        pp.log_path.parent.mkdir(parents=True, exist_ok=True)

    results_dir = Path(str(ours.argv[ours.argv.index("--results-dir") + 1]))
    state = OurSideState()
    procs: List[Any] = []
    failure: Optional[SeriesFailure] = None
    try:
        first, second = ((theirs, ours) if their_role == "acceptor" else (ours, theirs))
        p1 = start_peer(first, nice=plan.nice)
        procs.append(p1)
        await await_peer_ready(p1, first, plan.peer_ready_timeout_s)
        p2 = start_peer(second, nice=plan.nice)
        procs.append(p2)
        await await_peer_ready(p2, second, plan.peer_ready_timeout_s)
        await watch_peer_pair(procs, [first, second], results_dir, state, n_games,
                              plan.first_game_timeout_s, plan.progress_timeout_s)
    except SeriesFailure as exc:
        failure = exc
        print(f"[anchors] 🚨 {half} FAILED — {exc.cause}: {exc.detail}", flush=True)
    except Exception as exc:                            # noqa: BLE001 - any death must be NAMED
        failure = SeriesFailure("our_side_error", f"{type(exc).__name__}: {exc}")
        print(f"[anchors] 🚨 {half} FAILED — {exc}", flush=True)
    finally:
        for proc in procs:
            await await_peer_exit(proc, timeout_s=180.0 if failure is None else 15.0)
        for proc in procs:
            stop_peer(proc)
    state.records = read_peer_battles(results_dir)

    adapter = peers_mod.PEERS[plan.opponent_kind]
    report = adapter.read_report(theirs)
    our_report = adapter.read_report(ours)
    report["peer_rc"] = procs[-1].returncode if procs else None
    report["team_count"] = peers_mod.team_source_count(theirs)
    report["version"] = theirs.version
    report["commit"] = theirs.commit
    report["their_regime"] = theirs.their_regime
    # OUR side is a peer too, so its regime needs the same verification theirs gets — an
    # anchor-vs-anchor cell with only one side verified is half a measurement.
    report["our_argmax_match_rate"] = our_report.get("argmax_match_rate")
    report["our_regime_verified"] = our_report.get("regime_verified")
    report["our_stochastic_kwargs"] = list(our_report.get("sample_kwargs") or [])
    report["model_loader"] = ""
    return state.records, report, failure


async def run_half(plan: SeriesPlan, cfg: Any, half: str, n_games: int,
                   ) -> Tuple[List[Any], Dict[str, Any], Optional[SeriesFailure]]:
    """One half-series. Returns ``(records, peer_report, failure_or_None)``.

    🚨 **ROLE ORDER IS LOAD-BEARING.** The acceptor must be logged in before the challenger sends
    its first ``/challenge``: Showdown drops a challenge aimed at a user who is not online, and the
    challenger then waits forever with nothing in either log to say why.
    """
    if plan.our_side_is_peer:
        return await run_peer_pair_half(plan, cfg, half, n_games)

    import main.play as play

    mode, role, _suffix = HALVES[half]
    our_name, peer_name = half_usernames(plan, half)
    pplan = peer_plan(plan, cfg, role, n_games, half)
    pplan.log_path.parent.mkdir(parents=True, exist_ok=True)

    peers_mod.check_username(our_name, "our")
    peers_mod.check_username(peer_name, "peer")

    server_config = server_mod_config(plan)
    state = OurSideState()
    undo = install_our_side(state, plan.our_team_spec, plan.team_seed,
                            plan.forfeit_turn_limit, server_config,
                            our_side=plan.our_side, model_loader=plan.model_loader)
    args = play.build_parser().parse_args(our_argv(plan, mode, n_games, our_name, peer_name))

    proc = None
    our_task: Optional[asyncio.Task] = None
    failure: Optional[SeriesFailure] = None
    try:
        if role == "acceptor":
            # The PEER accepts, so the peer must be online first. Its banner is the gate — for the
            # 200M Metamon policy this is minutes of model build before the client connects.
            proc = start_peer(pplan, nice=plan.nice)
            await await_peer_ready(proc, pplan, plan.peer_ready_timeout_s)
            our_task = asyncio.create_task(play.main(args))
        else:
            # WE accept, so we must be online first. `build_model_player` stashes the player, and
            # `await_our_login` polls the real `logged_in` event rather than sleeping at it.
            our_task = asyncio.create_task(play.main(args))
            await await_our_login(state, plan.connect_timeout_s + 60.0)
            proc = start_peer(pplan, nice=plan.nice)

        watchdog = asyncio.create_task(
            watch(proc, pplan, state, n_games,
                  plan.first_game_timeout_s, plan.progress_timeout_s))
        done, _pending = await asyncio.wait({our_task, watchdog},
                                            return_when=asyncio.FIRST_COMPLETED)
        if watchdog in done and not watchdog.cancelled():
            exc = watchdog.exception()
            if exc is not None:
                raise exc
        watchdog.cancel()
        if our_task in done:
            our_task.result()          # re-raise anything our side died of
        else:
            # The watchdog saw the expected game count; let our side settle out.
            try:
                await asyncio.wait_for(asyncio.shield(our_task), timeout=120.0)
            except asyncio.TimeoutError:
                our_task.cancel()
    except SeriesFailure as exc:
        failure = exc
        print(f"[anchors] 🚨 {half} FAILED — {exc.cause}: {exc.detail}", flush=True)
    except Exception as exc:                            # noqa: BLE001 - any death must be NAMED
        failure = SeriesFailure("our_side_error", f"{type(exc).__name__}: {exc}")
        print(f"[anchors] 🚨 {half} FAILED — our side raised: {exc}", flush=True)
    finally:
        if our_task is not None and not our_task.done():
            our_task.cancel()
            try:
                await our_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        # Let the peer write its own report BEFORE anything terminates it — a cell with games
        # and no `argmax_match_rate` is a win rate whose regime is unverified, which is exactly
        # the number this tool exists not to emit. Short grace on a failure: there the peer is
        # already dead or wedged and its log, not its report, is the evidence.
        await await_peer_exit(proc, timeout_s=180.0 if failure is None else 15.0)
        # Close OUR socket before the next half logs in — see `disconnect_our_side`.
        await disconnect_our_side(state)
        stop_peer(proc)
        undo()

    adapter = peers_mod.PEERS[plan.opponent_kind]
    report = adapter.read_report(pplan)
    report["peer_rc"] = proc.returncode if proc is not None else None
    report["team_count"] = peers_mod.team_source_count(pplan)
    report["version"] = pplan.version
    report["commit"] = pplan.commit
    report["their_regime"] = pplan.their_regime
    report["our_stochastic_kwargs"] = list(state.stochastic_kwargs)
    report["model_loader"] = state.model_loader
    return state.records, report, failure


def server_mod_config(plan: SeriesPlan) -> Any:
    """The `ServerConfiguration` both halves use — built from the URI, so the tool can point at a
    websocket FRONT END over the Rust bridge later without touching anything else here."""
    from poke_env.ps_client.server_configuration import ServerConfiguration

    return ServerConfiguration(plan.server_uri, "https://play.pokemonshowdown.com/action.php?")


def cell_spec(plan: SeriesPlan, report: Dict[str, Any], our_team_count: int) -> CellSpec:
    return CellSpec(
        opponent=plan.opponent,
        opponent_kind=plan.opponent_kind,
        opponent_agent=plan.opponent_agent,
        opponent_version=str(report.get("version") or ""),
        opponent_commit=str(report.get("commit") or ""),
        # 🚨 A BOT HAS NO SAMPLING KNOB, exactly like Foul Play. Its policy IS the one the
        # bot-vs-bot round robin pinned, so the cell is a real measurement — but the two sides
        # are NOT at one nominal regime and the row says so rather than borrowing the peer's
        # label. `regime_matched` is False here by construction, and that is the honest stamp.
        our_regime=(plan.our_side if plan.our_side_is_bot
                    else (f"{plan.regime} ({plan.our_side})" if plan.our_side_is_peer
                          else plan.regime)),
        their_regime=str(report.get("their_regime") or plan.regime),
        regime_matched=plan.regime_matched and not plan.our_side_is_bot,
        teamset=plan.teamset,
        our_team_source=str(plan.our_team_spec.get("path", plan.our_team_spec.get("kind"))),
        our_team_count=our_team_count,
        their_team_source=str(report.get("team_source") or ""),
        their_team_count=int(report.get("team_count") or 0),
        battle_format=plan.battle_format,
        model_spec=plan.model_spec,
        model_zip=plan.model_zip,
        model_step=plan.model_step,
        model_rung=plan.model_rung,
        search_time_ms=plan.search_time_ms,
        search_parallelism=plan.search_parallelism,
        forfeit_turn_limit=plan.forfeit_turn_limit,
        server_uri=plan.server_uri,
        showdown_pin=plan.showdown_pin,
        server_impl=plan.server_impl,
        server_version=plan.server_version,
        our_side=plan.our_side,
        model_loader=str(report.get("model_loader") or ""),
    )


def rows_from(records: List[Any], half: str, cell: CellSpec, report: Dict[str, Any],
              start_index: int) -> List[GameRow]:
    out: List[GameRow] = []
    for i, rec in enumerate(records, start=start_index):
        out.append(GameRow(
            cell=cell,
            half=half,
            index=i,
            battle_tag=rec.battle_tag,
            turns=rec.turns,
            result=rec.result,
            won=rec.won,
            hit_forfeit_limit=rec.hit_forfeit_limit,
            our_team=rec.our_team,
            n_decisions=rec.n_decisions,
            n_defaults=rec.n_defaults,
            n_redecides=rec.n_redecides,
            t_finished=rec.t_finished,
            our_stochastic_kwargs=list(report.get("our_stochastic_kwargs") or []),
            their_sample_kwargs=list(report.get("sample_kwargs") or []),
            their_argmax_match_rate=report.get("argmax_match_rate"),
            their_visits_mean=report.get("visits_mean"),
            their_visits_n=report.get("visits_n"),
        ))
    return out


async def run_series(plan: SeriesPlan, cfg: Any) -> Tuple[List[GameRow], Dict[str, Any],
                                                          Optional[SeriesFailure]]:
    """Both halves, in order. A failure in the first half STOPS the read — a half-series against a
    dead peer is not a cheaper measurement, it is a different one."""
    from utils.team_loader import TeamLoader

    from main.anchors.session import load_team_texts

    # Counted through the SAME allowlist both sides use, so `team_source_asymmetry` in the summary
    # means a real difference in what the two sides draw from and not a stray bookkeeping file.
    if plan.our_side_is_peer:
        # Our side draws through the PEER's own team set, not our team source — count what it
        # will actually draw from, so `team_source_asymmetry` stays a real check.
        our_team_count = peers_mod.team_source_count(
            our_peer_plan(plan, cfg, "acceptor", 1, "ours_challenge"))
    else:
        our_team_count = (len(TeamLoader().get_all_teams())
                          if plan.our_team_spec.get("kind") == "pool"
                          else len(load_team_texts(Path(plan.our_team_spec["path"]))))

    all_rows: List[GameRow] = []
    last_report: Dict[str, Any] = {}
    failure: Optional[SeriesFailure] = None
    t0 = time.time()
    for half, n_games in plan.half_sizes().items():
        if n_games <= 0:
            continue
        print(f"[anchors] === half {half}: {n_games} games, {plan.opponent}, "
              f"regime={plan.regime}, teamset={plan.teamset}", flush=True)
        records, report, failure = await run_half(plan, cfg, half, n_games)
        last_report = report
        cell = cell_spec(plan, report, our_team_count)
        all_rows.extend(rows_from(records, half, cell, report, len(all_rows) + 1))
        if failure is not None:
            break
    last_report["wall_s"] = time.time() - t0
    return all_rows, last_report, failure
