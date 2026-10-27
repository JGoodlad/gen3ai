"""One anchor read, end to end: two role-balanced half-series against one external opponent.

**Role is balanced INSIDE every read, never confounded with one.** Showdown makes the CHALLENGER
p1, and p1/p2 is not a priori neutral; measured over 800 games on 2026-09-14 the role effect is
NOT DETECTED (+0.050 / −0.020 pooled), which is exactly what makes balancing it cheap insurance
rather than a wasted axis — inside a single 50-game half it looks large and inconsistent, because
that is the shape of a 50-game coin.

**Both sides move together or the cell is not matched.** ``--regime greedy`` sets our side to
greedy (the argmax of the masked logits) and the peer to its own greedy operation (for Metamon, ``get_actions(
sample=False)``; Foul Play has no sampling knob at all and is refused for ``t1`` by the CLI). Each
side's regime is then VERIFIED per decision and the verification is written onto every row.

The half-series is driven in ONE process: our side (the in-process core slot, or the Rust-stack live
client — :mod:`main.anchors.core_side` / :mod:`main.anchors.live_side`) runs on this event loop, the peer is a
subprocess, and :func:`main.anchors.session.watch` runs beside them so a dead peer or a stalled
series becomes a named failure instead of a report with nothing in it.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from main.anchors import mirrored as mirrored_mod
from main.anchors import peers as peers_mod
from main.anchors.results import CellSpec, GameRow
from main.anchors.session import (
    OurSideState,
    SeriesFailure,
    await_peer_exit,
    await_peer_ready,
    read_peer_battles,
    start_peer,
    team_set_digest_or_none,
    stop_peer,
    watch,
    watch_peer_pair,
)

#: ``(what our side does — challenge / accept, the peer's role, the username suffix)`` for each half. The names say who
#: sends the challenge; the SUFFIX is a hazard fix, not decoration — see :func:`half_usernames`.
HALVES = {
    "ours_challenge": ("challenge", "acceptor", "1"),
    "peer_challenge": ("accept", "challenger", "2"),
}


def half_usernames(plan: "SeriesPlan", half: str) -> "tuple[str, str]":
    """``(our name, the peer's name)`` for this half — each carrying the half's own suffix.

    🚨 **A name still held by the previous half is not an error, it is a GUEST.** Showdown answers
    a second login under a live name with ``|nametaken|``, and a client that carries on (the legacy
    poke-env fork did) plays under a server-assigned guest name; the peer's ``/challenge`` is then addressed to a user
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
    #: "model", "bot:<name>" (the Rust port of a roster bot, :mod:`main.anchors.live_side`) or a peer.
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
    #: The Rust engine the battles play on (``server.engine_stamp``; empty when none of ours serves them).
    engine_stamp: str = ""
    #: `--server rust` only: the reproducibility pair. There is no Node counterpart.
    seed_base: Optional[int] = None
    capture_dir: Optional[Path] = None
    #: WHEN our side emits the next `/challenge`: "serial" — only after the previous battle ENDED, the
    #: one mode both transports implement. 🚨 Hazard **H14**: poke-env's own ("pipelined") loop emitted
    #: it 0.4 s into the PREVIOUS battle, and a peer that reads its PMs only between battles (Foul
    #: Play) dropped it; that loop went with the legacy client (P6) and the CLI refuses it.
    challenge_mode: str = "serial"
    #: `--opponent-temperature`: the Metamon peer SAMPLES at this action temperature while our
    #: side stays greedy (`regime` stays "greedy" — it names OUR half). None = the peer follows
    #: `regime`. A cell with it set is stamped regime_matched=false.
    opponent_temperature: Optional[float] = None
    #: `--our-temperature`: OUR checkpoint SAMPLES at this temperature (``core_side.our_stochastic``)
    #: instead of the regime's own setting. None = the regime decides. A cell
    #: with it set is stamped our_regime='sample:T=<T>' (X22 f).
    our_temperature: Optional[float] = None
    #: `--our-belief-prior-mix`: OUR checkpoint's opponent belief is anchored to its Smogon prior at λ
    #: (`agents.model.belief_prior_mix`, INFERENCE ONLY). None = the unchanged model (λ = 0 is None: bit-identical).
    #: A cell with it set is stamped our_regime='<regime>+belief_prior_mix=<λ>', so it never pools with λ = 0.
    our_belief_prior_mix: Optional[float] = None
    #: T17 `--mirrored-pairs`: every half plays MIRRORED TEAM PAIRS (`main.anchors.mirrored`) on a
    #: front end that keys each battle's seed by its unordered team pair and logs it to `pair_log`.
    mirrored_pairs: bool = False
    pair_log: Optional[Path] = None
    #: `--our-transport`: "core" (the DEFAULT, P3) — our side is an IN-PROCESS slot of the Rust
    #: front end reading the core's own observation (`main.anchors.core_side`); "live" (P6) — a
    #: websocket client on the Rust stack (`main.anchors.live_side`: `main.live`'s reader + the
    #: checkpoint, or the Rust port of a bot) for `--server node`, `--server-uri` and a `bot:`
    #: our-side; "peer" for an anchor-vs-anchor cell. Neither imports poke-env. Stamped per row.
    our_transport: str = "core"
    #: How `our_transport` was chosen ("explicit" or the `auto` resolution and its reason) — printed.
    our_transport_note: str = ""
    #: `--bot-seed`: a `bot:` our-side's declared stream seed (the env core's bot-route seed): half h's bot
    #: draws `random.Random(stream_seed(bot_seed, h, k))` (`main.anchors.live_side.bot_streams`).
    bot_seed: Optional[int] = None
    #: B-D9 `--our-teamset`: the team set OUR side draws from in a PAIR cell (a peer our-side). None
    #: = the same as `teamset`; the CLI always resolves it. Read through :attr:`our_set`.
    our_teamset: Optional[str] = None
    #: B-D6 `--declare-asymmetric-teams`: "asymmetric" or "symmetric". Changes no play; stamped on
    #: the summary so a registration reads the unit under `results.unit_counts`' declared branch.
    declared_team_cell: str = "symmetric"
    #: `--our-search`: OUR checkpoint plus a registered one-ply search (`main.anchors.search_presets`, canonical
    #: `<preset>@<from turn>`; `main.anchors.search_side`). None = the checkpoint alone. Stamped
    #: our_regime='<regime>+search=<preset>@<turn>', so a searched cell never pools with the raw one.
    our_search: Optional[str] = None
    #: `--deal-game G`: play ONLY game G of the `games`-game series this plan names (S-U's per-game units): its half
    #: (the role), its team draws (both sides skip the half's earlier draws) and its battle seed (the front end counts
    #: battles from G) are the ones game G of the whole series gets. None = the whole series.
    deal_game: Optional[int] = None

    @property
    def our_set(self) -> str:
        """The team set OUR side draws from (``our_teamset``, else ``teamset``)."""
        return self.our_teamset or self.teamset

    @property
    def our_side_is_bot(self) -> bool:
        return self.our_side.startswith("bot:")

    @property
    def our_side_is_peer(self) -> bool:
        """Our side is ANOTHER external agent — an anchor-vs-anchor cell, with no player of ours
        in the process at all."""
        return self.our_side.startswith("metamon:") or self.our_side == "foulplay"

    def series_half_sizes(self) -> Dict[str, int]:
        """``games`` split across the two roles; an odd game goes to the first half and is
        recorded there rather than silently dropped."""
        half = self.games // 2
        return {"ours_challenge": self.games - half, "peer_challenge": half}

    def half_sizes(self) -> Dict[str, int]:
        """The games each half PLAYS: the series' split, or, with ``deal_game`` G, one game in G's half."""
        sizes = self.series_half_sizes()
        if self.deal_game is None:
            return sizes
        half, _k = self.deal_position()
        return {h: (1 if h == half else 0) for h in sizes}

    def deal_position(self) -> "tuple[str, int]":
        """``(half, k)``: the half game ``deal_game`` of the series falls in, and its index inside that half (the
        team draws each side skips first)."""
        if self.deal_game is None:
            raise ValueError("deal_position: the plan plays the whole series")
        g = int(self.deal_game)
        first = self.series_half_sizes()["ours_challenge"]
        if not 0 <= g < self.games:
            raise ValueError(f"--deal-game {g} is not a game of a {self.games}-game series")
        return ("ours_challenge", g) if g < first else ("peer_challenge", g - first)

    def deal_skip(self, half: str) -> int:
        """The team draws both sides skip in ``half`` (0 unless ``deal_game`` falls in it)."""
        if self.deal_game is None:
            return 0
        h, k = self.deal_position()
        return k if h == half else 0


def peer_plan(plan: SeriesPlan, cfg: Any, role: str, n_games: int, half: str,
              team_sequence: Optional[Path] = None) -> Any:
    our_name, peer_name = half_usernames(plan, half)
    adapter = peers_mod.PEERS[plan.opponent_kind]
    opp_t = plan.opponent_temperature
    extra: Dict[str, Any] = {} if opp_t is None else {"temperature": opp_t}
    if team_sequence is not None:
        extra["team_sequence"] = team_sequence
    return adapter.plan(
        cfg=cfg.opponent(plan.opponent_kind),
        agent=plan.opponent_agent,
        regime=plan.regime if opp_t is None else "t1",
        **extra,
        teamset=plan.teamset,
        battle_format=plan.battle_format,
        server_uri=plan.server_uri,
        username=peer_name,
        opponent_username=our_name,
        role=role,
        n_games=n_games,
        team_seed=plan.team_seed + 1,
        team_skip=plan.deal_skip(half),
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
        # B-D9: our peer draws from ITS OWN declared set (an asymmetric pair cell, e.g. K(P, H)).
        teamset=plan.our_set,
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
    is no side of ours here to observe, and the weaker instrument is NAMED on the rows it produces.
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

    their_adapter, our_adapter = pair_adapters(plan)
    report = their_adapter.read_report(theirs)
    our_report = our_adapter.read_report(ours)
    report["peer_rc"] = procs[-1].returncode if procs else None
    report["team_count"] = peers_mod.team_source_count(theirs)
    report["team_digest"] = team_set_digest_or_none(theirs.team_dir)
    report["version"] = theirs.version
    report["commit"] = theirs.commit
    report["their_regime"] = theirs.their_regime
    # OUR side is a peer too, so its regime needs the same verification theirs gets — an
    # anchor-vs-anchor cell with only one side verified is half a measurement.
    report["our_argmax_match_rate"] = our_report.get("argmax_match_rate")
    report["our_regime_verified"] = our_report.get("regime_verified")
    report["our_regime_verified_decisions"] = our_report.get("regime_verified_decisions")
    report["our_stochastic_kwargs"] = list(our_report.get("sample_kwargs") or [])
    # OUR peer's weights too (FR-1 FINDING 3): what it verified, beside what theirs verified.
    report["our_hf_revision"] = our_report.get("hf_revision")
    report["our_weights_sha256"] = our_report.get("weights_sha256")
    # ...and OUR peer's team pack (gen3_peer_showdown_pack_v1, SOP H22): its own regime-id input.
    report["our_peer_pack_changed"] = our_report.get("peer_pack_changed")
    report["our_frontend_pack_changed"] = our_report.get("frontend_pack_changed")
    report["our_peer_code"] = our_report.get("peer_code")         # gen3_peer_code_v1: OUR peer's code too
    report["model_loader"] = ""
    report["policy_mode"] = ""
    # BOTH peers are peers here, so the regime question is an AND over the two and the process
    # question is an AND over BOTH return codes.
    report["regime_verified_decisions"] = bool(report.get("regime_verified_decisions")) and bool(
        our_report.get("regime_verified_decisions"))
    report["peer_error_free"] = bool(report.get("peer_error_free")) and bool(
        our_report.get("peer_error_free"))
    finalize_regime_fields(report, peer_rcs=[p.returncode for p in procs] or [None])
    pair_faults = 0
    for pp in (theirs, ours):                           # H21: either peer's fault voids the pair cell
        failure = apply_peer_fault(report, pp, len(state.records), n_games, failure)
        pair_faults += int(report["peer_fault_lines"])
    report["peer_fault_lines"] = pair_faults
    if pair_faults:
        report["peer_clean"] = False
        report["regime_verified"] = False
    note = pair_exit_note(plan, half, report, our_report, len(state.records), n_games)
    if note is not None:
        report["peer_exit_note"] = note
        print(f"[anchors] ⚠️  {half}: {note['cause']} — the unit is VOID (H21; apply_peer_fault), re-run it with the same seed.",
              flush=True)
    return state.records, report, failure


def pair_adapters(plan: SeriesPlan) -> Tuple[Any, Any]:
    """``(their adapter, our adapter)`` for an anchor-vs-anchor cell. 🚨 EACH side's report is
    read by ITS OWN adapter: reading a Metamon report through Foul Play's reader finds no argmax
    rate, so a pair cell against Foul Play read ``regime_verified_decisions = false`` on a Metamon
    side whose every decision matched (found by the n0_endofrun_2026-09-27 smoke). Metamon vs
    Metamon hid it, because there the two adapters are the same one."""
    return (peers_mod.PEERS[plan.opponent_kind],
            peers_mod.PEERS[plan.our_side.partition(":")[0]])


#: In a pair cell OUR peer challenges in ``ours_challenge``, so hazard H17's "the half where
#: METAMON challenges" is the MIRROR of the half's name when the Metamon process is ours.
_MIRROR_HALF = {"ours_challenge": "peer_challenge", "peer_challenge": "ours_challenge"}


def pair_exit_note(plan: SeriesPlan, half: str, their_report: Dict[str, Any],
                   our_report: Dict[str, Any], n_records: int,
                   expected: int) -> Optional[Dict[str, str]]:
    """Hazard **H17** for an anchor-vs-anchor half: either side may be the Metamon process that
    CHALLENGES in this half, so each side is classified in its own challenging half. At most one
    side challenges per half, so at most one note is returned."""
    our_kind = plan.our_side.partition(":")[0]
    return (classify_peer_error(their_report, half, plan.opponent_kind, n_records, expected)
            or classify_peer_error(our_report, _MIRROR_HALF[half], our_kind, n_records, expected))


async def run_half(plan: SeriesPlan, cfg: Any, half: str, n_games: int, front: Any = None,
                   ) -> Tuple[List[Any], Dict[str, Any], Optional[SeriesFailure]]:
    """One half-series. Returns ``(records, peer_report, failure_or_None)``.

    🚨 **ROLE ORDER IS LOAD-BEARING.** The acceptor must be logged in before the challenger sends
    its first ``/challenge``: Showdown drops a challenge aimed at a user who is not online, and the
    challenger then waits forever with nothing in either log to say why.

    ``front`` is the IN-PROCESS front end (:class:`main.anchors.server.InProcessFrontEnd`'s
    ``ShowdownFrontEnd``) when our side is the core slot (``plan.our_transport == "core"``).
    """
    if plan.our_side_is_peer:
        return await run_peer_pair_half(plan, cfg, half, n_games)

    mode, role, _suffix = HALVES[half]
    our_name, peer_name = half_usernames(plan, half)
    pplan = peer_plan(plan, cfg, role, n_games, half)
    pplan.log_path.parent.mkdir(parents=True, exist_ok=True)
    team_spec = plan.our_team_spec
    mirror = None
    if plan.mirrored_pairs:
        # T17: draw this half's pairings, write both sides' team files, and hand the peer its
        # battle-ordered list; our side plays the mirror-image sequence.
        mirror = mirrored_mod.plan_half(half, n_games, our_team_items(plan.our_team_spec),
                                        their_team_items(pplan), seed=plan.team_seed,
                                        out_dir=plan.out_dir, battle_format=plan.battle_format)
        team_spec = {"kind": "sequence", "items": mirror.ours}
        pplan = peer_plan(plan, cfg, role, n_games, half, team_sequence=mirror.sequence_path)

    peers_mod.check_username(our_name, "our")
    peers_mod.check_username(peer_name, "peer")

    if plan.our_transport == "core":
        state = OurSideState()
        proc, failure = await _play_core_half(plan, half, n_games, pplan, team_spec, front, state,
                                              our_name, peer_name, role)
        return _half_report(plan, half, n_games, pplan, proc, state, mirror, failure)
    if plan.our_transport == "live":
        from main.anchors import live_side

        state = OurSideState()
        proc, failure = await live_side.play_half(plan, half, n_games, pplan, team_spec, state,
                                                  our_name, peer_name, role)
        return _half_report(plan, half, n_games, pplan, proc, state, mirror, failure)

    raise SeriesFailure("our_side_error", f"unknown --our-transport {plan.our_transport!r} "
                        "(the legacy poke-env client was DELETED in P6)")


async def _play_core_half(plan: SeriesPlan, half: str, n_games: int, pplan: Any,
                          team_spec: Dict[str, Any], front: Any, state: OurSideState,
                          our_name: str, peer_name: str, role: str,
                          ) -> Tuple[Any, Optional[SeriesFailure]]:
    """One half with OUR side as the in-process core slot (:mod:`main.anchors.core_side`).

    The same role order, watchdog and peer teardown as the live client: the acceptor is online
    before the challenger, :func:`watch` turns a dead peer or a stall into a named failure, the
    peer writes its own report before anything terminates it, and our name is released before the
    next half logs in (hazard H10's equivalent — the slot leaves the user table).
    """
    from main.anchors import core_side

    if front is None:
        raise SeriesFailure("core_slot_error",
                            "--our-transport core needs the in-process front end, and none was started")
    stochastic, temperature = core_side.our_stochastic(plan)
    slot = None
    proc = None
    failure: Optional[SeriesFailure] = None
    our_task: Optional[asyncio.Task] = None
    try:
        slot = core_side.build_slot(
            front, username=our_name, model_zip=plan.model_zip, device=plan.device,
            model_loader=plan.model_loader, stochastic=stochastic, temperature=temperature,
            team_spec=team_spec, team_seed=plan.team_seed,
            forfeit_limit=plan.forfeit_turn_limit, state=state, battle_format=plan.battle_format,
            belief_prior_mix=plan.our_belief_prior_mix, team_skip=plan.deal_skip(half),
            search=plan.our_search, search_out=plan.out_dir / "search_decisions.jsonl")
        await slot.login()
        if role == "acceptor":
            proc = start_peer(pplan, nice=plan.nice)
            await await_peer_ready(proc, pplan, plan.peer_ready_timeout_s)
            our_task = asyncio.create_task(
                slot.challenge_series(peer_name, n_games, plan.first_game_timeout_s))
        else:
            our_task = asyncio.create_task(slot.accept_series(peer_name, n_games))
            proc = start_peer(pplan, nice=plan.nice)
        watchdog = asyncio.create_task(
            watch(proc, pplan, state, n_games, plan.first_game_timeout_s, plan.progress_timeout_s))
        done, _pending = await asyncio.wait({our_task, watchdog},
                                            return_when=asyncio.FIRST_COMPLETED)
        if watchdog in done and not watchdog.cancelled():
            exc = watchdog.exception()
            if exc is not None:
                raise exc
        watchdog.cancel()
        if our_task in done:
            our_task.result()
        else:
            try:
                await asyncio.wait_for(asyncio.shield(our_task), timeout=120.0)
            except asyncio.TimeoutError:
                our_task.cancel()
        slot.check()
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
        await await_peer_exit(proc, timeout_s=180.0 if failure is None else 15.0)
        if slot is not None:
            await slot.leave()
            if getattr(slot, "searcher", None) is not None:
                state.search_stats = slot.searcher.summary()
                slot.searcher.close()
        stop_peer(proc)
    return proc, failure


def _half_report(plan: SeriesPlan, half: str, n_games: int, pplan: Any, proc: Any,
                 state: OurSideState, mirror: Any, failure: Optional[SeriesFailure],
                 ) -> Tuple[List[Any], Dict[str, Any], Optional[SeriesFailure]]:
    """The half's peer report with OUR side's verification beside it — shared by both transports."""
    adapter = peers_mod.PEERS[plan.opponent_kind]
    report = adapter.read_report(pplan)
    report["peer_rc"] = proc.returncode if proc is not None else None
    report["team_count"] = peers_mod.team_source_count(pplan)
    report["team_digest"] = team_set_digest_or_none(pplan.team_dir)
    report["version"] = pplan.version
    report["commit"] = pplan.commit
    report["their_regime"] = pplan.their_regime
    report["our_stochastic_kwargs"] = list(state.stochastic_kwargs)
    report["model_loader"] = state.model_loader
    report["policy_mode"] = state.policy_mode          # train / eval, READ BACK from the loaded policy ("" for a bot)
    if plan.our_side_is_bot and plan.our_transport == "live":
        from main.anchors.live_side import bot_streams

        # the bot's declared streams in THIS half — what a reader needs to re-seed the same bot
        report["our_bot_streams"] = {"seed": plan.bot_seed, "half": half,
                                     **bot_streams(int(plan.bot_seed), half)}
    # `--our-temperature`: OUR half's regime is verified too — every decision must have received
    # stochastic=True. A sampled cell whose decisions ran greedy is not the cell it names.
    if plan.our_temperature is not None and state.stochastic_kwargs != [True]:
        report["regime_verified_decisions"] = False
        report["our_regime_mismatch"] = list(state.stochastic_kwargs)
    # `--our-search`: a searched cell must have had its searcher (its counters are the half's search record). A raw
    # cell's report is unchanged.
    if plan.our_search is not None:
        report["our_search"] = plan.our_search
        report["search"] = state.search_stats
        if state.search_stats is None or state.search_stats.get("search") != plan.our_search:
            report["regime_verified_decisions"] = False
            report["our_search_mismatch"] = [plan.our_search, (state.search_stats or {}).get("search")]
    # `--our-belief-prior-mix`: the λ the loaded policy READS BACK must be the one the cell names.
    report["our_belief_prior_mix"] = state.belief_prior_mix
    if not plan.our_side_is_bot and not plan.our_side_is_peer and state.belief_prior_mix != plan.our_belief_prior_mix:
        report["regime_verified_decisions"] = False
        report["our_belief_prior_mix_mismatch"] = [plan.our_belief_prior_mix, state.belief_prior_mix]
    finalize_regime_fields(report, peer_rcs=[proc.returncode if proc is not None else None])
    failure = apply_peer_fault(report, pplan, len(state.records), n_games, failure)
    if mirror is not None:
        report["mirrored_half"] = mirrored_mod.score_half(
            mirror, state.records, list(state.draws), list(report.get("team_draws") or []),
            mirrored_mod.read_pair_log(plan.pair_log))
    note = classify_peer_error(report, half, plan.opponent_kind, len(state.records), n_games)
    if note is not None:
        report["peer_exit_note"] = note
        print(f"[anchors] ⚠️  {half}: {note['cause']} — the unit is VOID (H21; apply_peer_fault), re-run it with the same seed.",
              flush=True)
    return state.records, report, failure


def _row_transport(plan: SeriesPlan) -> str:
    from main.anchors.core_side import row_transport

    return row_transport(plan)


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
                          else (f"sample:T={float(plan.our_temperature):g}"
                                if plan.our_temperature is not None else plan.regime)
                          + (f"+belief_prior_mix={float(plan.our_belief_prior_mix):g}"
                             if plan.our_belief_prior_mix is not None else "")
                          + (f"+search={plan.our_search}" if plan.our_search is not None else ""))),
        their_regime=str(report.get("their_regime") or plan.regime),
        regime_matched=plan.regime_matched and not plan.our_side_is_bot,
        teamset=plan.teamset,
        # A PEER our-side draws through its own team set, so ITS directory is our source (run_series).
        our_team_source=str(report.get("our_team_source")
                            or plan.our_team_spec.get("path", plan.our_team_spec.get("kind"))),
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
        engine_stamp=plan.engine_stamp,
        our_side=plan.our_side,
        model_loader=str(report.get("model_loader") or ""),
        policy_mode=str(report.get("policy_mode") or ""),
        mirrored_pairs=bool(plan.mirrored_pairs),
        our_transport=_row_transport(plan),
        our_bot_seed=(plan.bot_seed if plan.our_side_is_bot and plan.our_transport == "live" else None),
        our_teamset=plan.our_set,
        our_team_digest=report.get("our_team_digest"),
        their_team_digest=report.get("team_digest"),
        hf_revision=report.get("hf_revision"),
        weights_sha256=report.get("weights_sha256"),
        our_hf_revision=report.get("our_hf_revision"),
        our_weights_sha256=report.get("our_weights_sha256"),
        peer_pack_changed=report.get("peer_pack_changed"),
        our_peer_pack_changed=report.get("our_peer_pack_changed"),
        frontend_pack_changed=report.get("frontend_pack_changed"),
        our_frontend_pack_changed=report.get("our_frontend_pack_changed"),
        peer_code=report.get("peer_code"),
        our_peer_code=report.get("our_peer_code"),
    )


#: The cell's team-pack facts: THEIR Metamon peer's and OUR Metamon peer's (gen3_peer_showdown_pack_v1), THEIR Foul Play's
#: and OUR Foul Play's (gen3_frontend_validated_pack_v1).
PEER_PACK_KEYS = ("peer_pack_changed", "our_peer_pack_changed", "frontend_pack_changed", "our_frontend_pack_changed")


#: What each side PLAYED WITH (``main.reads.side_identity``): the Metamon peer's code (theirs, ours) and our policy's
#: load mode. One value per cell: a half that differs from an earlier half voids the regime verification.
SIDE_KEYS = ("peer_code", "our_peer_code", "policy_mode")


def check_side_halves(half_reports: List[Dict[str, Any]]) -> None:
    """The latest half's report must play with what every earlier half played with (:data:`SIDE_KEYS`): else its
    ``regime_verified_decisions`` is False (the series' is their AND) and ``side_mismatch`` names the values."""
    last = half_reports[-1]
    for key in SIDE_KEYS:
        seen = [r.get(key) for r in half_reports]
        if len(set(seen)) > 1:
            last["regime_verified_decisions"] = False
            last.setdefault("side_mismatch", {})[key] = seen


def any_half(values: Any) -> Optional[bool]:
    """True if any half says True, else False if any says False, else None (no half reported it: not a Metamon peer,
    or a report that predates the pack)."""
    seen = [v for v in values if isinstance(v, bool)]
    return any(seen) if seen else None


#: The exact upstream signature of hazard **H-H / H5**, matched on the peer report's ``error``
#: string. Metamon's ``MetamonAMAGOWrapper.step`` answers ANY exception with
#: ``self.reset(); return self.step(action)`` — no depth bound, no re-raise, no distinction
#: between a transient error and a permanent one — so an error ``reset()`` cannot clear becomes
#: ~988 stack frames and then this.
UPSTREAM_RECURSION_SIGNATURE = "RecursionError"

#: The role in which it has fired, all FOUR recorded times. See :func:`classify_peer_error`.
UPSTREAM_RECURSION_ROLE = "peer_challenge"


def classify_peer_error(report: Dict[str, Any], half: str, opponent_kind: str,
                        n_records: int, expected: int) -> Optional[Dict[str, str]]:
    """Name Metamon's post-game recursion for what it is, instead of leaving an rc for a reader.

    🚨 **The defect is INSIDE Metamon and it is about WHO CHALLENGES.** Its evaluator drives two
    different challenge loops depending on the role, and only one of them is serialized:

    * **acceptor** — Metamon's OWN ``_accept_challenge_loop``, which its docstring says "accepts
      one challenge at a time and **fully awaits the battle** before accepting the next … ensuring
      terminated/truncated signals propagate correctly";
    * **challenger** — poke-env's ``start_challenging(n_challenges=num_battles)``, the PIPELINED
      loop, whose battle semaphore is released when a battle *starts*.

    That is the same defect class as our own hazard **H14** (our own
    serialized challenge send), on the other side of the wire:
    Metamon serialized one role and left the other pipelined. The env's ``current_battle`` and the
    agent's then disagree, ``openai_api.step`` raises ``Battle is already finished, call reset``,
    and the unbounded handler above turns it into a ``RecursionError``.

    **It costs no recorded game, but it voids the unit.** All four recorded occurrences are
    post-game, and the games were all present with the regime verified on every decision. This
    function only NAMES the cause; :func:`apply_peer_fault` (with ``results.unit_counts``) voids
    any unit whose peer logged a long-tail line (H21), and the unit is re-run with the same seed.

    Returns a ``{"cause", "detail"}`` note for the summary, or ``None`` when this is not that.
    """
    error = str(report.get("error") or "")
    if opponent_kind != "metamon" or UPSTREAM_RECURSION_SIGNATURE not in error:
        return None
    if half != UPSTREAM_RECURSION_ROLE or n_records < expected:
        return None
    return {
        "cause": "peer_recursion_upstream",
        "detail": (
            f"{error} — Metamon's OWN post-game teardown (hazard H-H/H5), in the half where "
            "METAMON challenges. All "
            f"{n_records}/{expected} games are present and the regime verified on every "
            "decision; the defect is upstream and post-game, but the peer logged a long-tail line, so the unit is VOID and is re-run with the same seed (apply_peer_fault, H21). Cause: "
            "MetamonAMAGOWrapper.step answers any exception with `self.reset(); return "
            "self.step(action)` — unbounded — and Metamon drives the CHALLENGER role through "
            "poke-env's pipelined start_challenging() while driving the ACCEPTOR role through "
            "its own serialized _accept_challenge_loop. The minimal upstream patch is in "
            "designs/research_state/measurements/anchors_p2_batch_2026-09-22/README.md."),
    }


def finalize_regime_fields(report: Dict[str, Any], *, peer_rcs: List[Optional[int]]) -> None:
    """Fill a half's report with the TWO separate verification facts, in place.

    🚨 **THE SPLIT** (``gen3_anchor_regime_split_v1``). ``regime_verified_decisions`` says the
    regime took effect on every decision the peer made; ``peer_clean`` says every peer process
    exited 0. They are different facts with different consequences and they must never be read
    through one flag: on the 2026-09-20 continuation campaign the composite was FALSE on **31 of
    84 sub-cells** whose per-decision argmax rate was 1.0000 throughout, because Metamon raises a
    ``RecursionError`` in its post-game teardown when our side forfeits at turn 250 — AFTER the
    last decision, and after every game was played and recorded.

    A nonzero rc is still recorded, loudly. What it is not is a reason to throw away a win rate
    whose regime verified on every one of its decisions.

    ``regime_verified`` is kept as their AND for ONE release
    (:data:`peers_mod.REGIME_VERIFIED_DEPRECATION`), so nothing silently re-points at a different
    quantity — but it is no longer what the SOP reads.
    """
    rcs = [rc for rc in peer_rcs if rc is not None]
    report["peer_rcs"] = peer_rcs
    report["peer_clean"] = bool(report.get("peer_error_free", True)) and all(rc == 0 for rc in rcs)
    report["regime_verified"] = bool(report.get("regime_verified_decisions")) and report["peer_clean"]


def apply_peer_fault(report: Dict[str, Any], pplan: Any, done: int, expected: int,
                     failure: Optional[SeriesFailure]) -> Optional[SeriesFailure]:
    """🚨 H21: a peer fault ANYWHERE in a half (mid-half, or after its last game) means the unit never counts.

    ``peer_clean`` alone relied on the fault ending in a recorded ``error`` or a nonzero rc. This reads the peer's
    log itself: any :data:`session.PEER_FAULT_LINE` makes ``peer_clean`` False and the half FAILED
    (``peer_long_tail_fault``, unless it already failed), so the verdict does not depend on whether the watchdog's
    poll saw the line before the last record landed. A unit whose peer never printed it is untouched."""
    from main.anchors.session import PeerFaultScan, peer_fault_failure

    log = getattr(pplan, "log_path", None)
    scan = PeerFaultScan(Path(log)) if log is not None else None
    n = scan.final() if scan is not None else 0
    report["peer_fault_lines"] = n
    if not n:
        return failure
    report["peer_clean"] = False
    report["regime_verified"] = False
    print(f"[anchors] 🚨 {pplan.label}: {n} long-tail force-reset line(s) — Metamon's fault (H5/H21); "
          "this unit is VOID and never counts", flush=True)
    return failure if failure is not None else peer_fault_failure(pplan.label, scan, done, expected)


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
            their_router_no_rng_rate=report.get("router_no_rng_rate"),
            their_visits_mean=report.get("visits_mean"),
            their_visits_n=report.get("visits_n"),
            our_argmax_matches=getattr(rec, "our_argmax_matches", None),
            our_argmax_decisions=getattr(rec, "our_argmax_decisions", None),
        ))
    return out


def our_front_end_serves(plan: SeriesPlan) -> bool:
    """Did OUR websocket front end serve this read (in process or as our subprocess)? Only then is a forced end seen."""
    return plan.server_impl == "rust" and bool(plan.started_server)


def forced_end_verdict(plan: SeriesPlan, rows: List[GameRow], records: Optional[List[Dict[str, Any]]],
                       ) -> Dict[str, Any]:
    """🚨 THE FORFEIT-WIN GUARD (S-U, 2026-10-27): which battles ended by a FORCELOSE that is NOT an outcome.

    The sim's win batch after a FORCELOSE is byte-identical to a win by play, so a peer that DROPS (poke-env's
    keepalive timeout when our event loop blocked: the S-U build's negative control booked a forfeit WIN for us at
    turn 3), a ``/forfeit`` from the peer, or our own side disconnecting would otherwise read as an ordinary win or
    loss. ``records`` are the front end's forced ends (``ws_frontend.FORCED_END_SCHEMA``; None = the record is
    missing). The ONE forced end that is an outcome is OUR side's own turn-limit forfeit (the trainer's cap: a
    ``forfeit`` by one of our names at a turn >= ``plan.forfeit_turn_limit`` > 0, on a row stamped
    ``hit_forfeit_limit``). Every other one is VOID: the game is no outcome at all and the series fails by name.

    Returns ``{guard, forced_ends, void}``: ``guard`` is ``armed`` (our front end served and its record was read),
    ``blind`` (our front end served but the record is missing: nothing it served can count) or ``unavailable`` (Node
    or an external server: a forced end there is invisible to this guard).
    """
    from utils.bridge.ws_frontend import FORCED_END_SCHEMA, to_id

    if not our_front_end_serves(plan):
        return {"guard": "unavailable", "forced_ends": [], "void": []}
    if records is None:
        return {"guard": "blind", "forced_ends": [], "void": []}
    ours = {to_id(half_usernames(plan, half)[0]) for half in HALVES}
    limit = int(plan.forfeit_turn_limit or 0)
    by_tag = {r.battle_tag: r for r in rows}
    void: List[Dict[str, Any]] = []
    for rec in records:
        row = by_tag.get(rec.get("tag"))
        cap = (rec.get("schema") == FORCED_END_SCHEMA and rec.get("cause") == "forfeit"
               and rec.get("user") in ours and not plan.our_side_is_peer and limit > 0
               and int(rec.get("turn") or 0) >= limit and (row is None or bool(row.hit_forfeit_limit)))
        if not cap:
            void.append({**rec, "booked": None if row is None else row.result})
    return {"guard": "armed", "forced_ends": list(records), "void": void}


def read_forced_ends(path: Path) -> Optional[List[Dict[str, Any]]]:
    """The front end's forced-end JSONL; None when it does not exist (the guard is then BLIND, never clean)."""
    try:
        text = path.read_text()
    except OSError:
        return None
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def apply_forced_end_guard(plan: SeriesPlan, rows: List[GameRow], report: Dict[str, Any],
                           failure: Optional[SeriesFailure],
                           ) -> Tuple[List[GameRow], Optional[SeriesFailure]]:
    """Run :func:`forced_end_verdict` on the series: a VOID battle's row is DROPPED (it is no outcome), the report
    carries ``forced_end_guard`` / ``forced_ends`` / ``non_battle_endings``, and the series FAILS by name
    (``non_battle_termination``, or ``forced_end_guard_blind``) unless it already failed."""
    from main.anchors.server import FORCED_ENDS_NAME

    verdict = forced_end_verdict(plan, rows, read_forced_ends(plan.out_dir / FORCED_ENDS_NAME)
                                 if our_front_end_serves(plan) else [])
    report["forced_end_guard"] = verdict["guard"]
    report["forced_ends"] = verdict["forced_ends"]
    report["non_battle_endings"] = verdict["void"]
    if verdict["void"]:
        tags = {v.get("tag") for v in verdict["void"]}
        rows = [r for r in rows if r.battle_tag not in tags]
        what = "; ".join(f"{v.get('tag')}: {v.get('cause')} of {v.get('slot')} ({v.get('user')}) at turn "
                         f"{v.get('turn')}, booked {v.get('booked')}" for v in verdict["void"])
        print(f"[anchors] 🚨 NON-BATTLE TERMINATION — VOID, no outcome counted: {what}", flush=True)
        if failure is None:
            failure = SeriesFailure("non_battle_termination",
                                    f"{len(verdict['void'])} battle(s) ended by a forced end that is not our "
                                    f"turn-limit forfeit (VOID, never an outcome): {what}")
    elif verdict["guard"] == "blind":
        print("[anchors] 🚨 the forfeit-win guard is BLIND: our front end served but wrote no forced-end record",
              flush=True)
        if failure is None:
            failure = SeriesFailure("forced_end_guard_blind",
                                    f"our front end served this read but {FORCED_ENDS_NAME} is missing: a peer's "
                                    "drop could not be told from a win, so nothing here counts")
    return rows, failure


async def run_series(plan: SeriesPlan, cfg: Any) -> Tuple[List[GameRow], Dict[str, Any],
                                                          Optional[SeriesFailure]]:
    """Both halves, in order. A failure in the first half STOPS the read — a half-series against a
    dead peer is not a cheaper measurement, it is a different one."""
    # Counted through the SAME allowlist both sides use, so `team_source_asymmetry` in the summary
    # means a real difference in what the two sides draw from and not a stray bookkeeping file.
    our_team_count, our_team_source, our_team_digest = our_team_facts(plan, cfg)

    all_rows: List[GameRow] = []
    half_reports: List[Dict[str, Any]] = []
    last_report: Dict[str, Any] = {}
    failure: Optional[SeriesFailure] = None
    t0 = time.time()
    in_process = None
    if plan.our_transport == "core" and not plan.our_side_is_peer:
        # P3: the front end runs IN THIS PROCESS so our side can be a slot of it; the peer still
        # dials it over a real websocket on the read's 95XX port.
        from main.anchors.server import FORCED_ENDS_NAME, InProcessFrontEnd

        in_process = InProcessFrontEnd(
            plan.server_port, battle_format=plan.battle_format, seed_base=plan.seed_base,
            capture_dir=plan.capture_dir, pair_log=plan.pair_log,
            log_path=plan.out_dir / InProcessFrontEnd.log_name,
            battle_index_base=plan.deal_game or 0, forced_ends_log=plan.out_dir / FORCED_ENDS_NAME)
        await in_process.__aenter__()
        print(f"[anchors] {in_process.label} serving {in_process.uri} inside pid={in_process.pid} "
              f"({plan.server_version})", flush=True)
    try:
        for half, n_games in plan.half_sizes().items():
            if n_games <= 0:
                continue
            print(f"[anchors] === half {half}: {n_games} games, {plan.opponent}, "
                  f"regime={plan.regime}, teamset={plan.teamset}, our transport={plan.our_transport}",
                  flush=True)
            records, report, failure = await run_half(
                plan, cfg, half, n_games, front=in_process.front if in_process else None)
            report.update(our_team_count=our_team_count, our_team_source=our_team_source,
                          our_team_digest=our_team_digest)
            last_report = report
            half_reports.append(report)
            for key in PEER_PACK_KEYS:                  # a cell's pack changed iff ANY half's did (SOP H22)
                report[key] = any_half(r.get(key) for r in half_reports)
            check_side_halves(half_reports)
            cell = cell_spec(plan, report, our_team_count)
            all_rows.extend(rows_from(records, half, cell, report, len(all_rows) + 1))
            if failure is not None:
                break
    finally:
        if in_process is not None:
            await in_process.__aexit__(None, None, None)
            print(f"[anchors] {in_process.label} closed (every battle child reaped)", flush=True)
    # 🚨 the forfeit-win guard: a battle a FORCELOSE decided (other than our own turn-limit cap) is no outcome
    all_rows, failure = apply_forced_end_guard(plan, all_rows, last_report, failure)
    # 🚨 SERIES-LEVEL, STAMPED ON EVERY ROW. The two verification facts are properties of the
    # WHOLE cell — "both halves' regime verified", "every peer exited 0" — but a row has to be
    # self-contained, so they are back-filled onto the rows the first half already produced. An
    # empty series verifies NOTHING rather than vacuously everything.
    decisions = bool(half_reports) and all(bool(r.get("regime_verified_decisions"))
                                           for r in half_reports)
    clean = bool(half_reports) and all(bool(r.get("peer_clean")) for r in half_reports)
    for row in all_rows:
        row.regime_verified_decisions = decisions
        row.peer_clean = clean
        row.regime_verified = decisions and clean
    last_report["series_regime_verified_decisions"] = decisions
    last_report["series_peer_clean"] = clean
    # Carried to the SUMMARY, not just the half that produced it: a reader looking at
    # `peer_clean = false` needs the reason on the same page as the flag.
    last_report["peer_exit_notes"] = [r["peer_exit_note"] for r in half_reports
                                      if r.get("peer_exit_note")]
    last_report["wall_s"] = time.time() - t0
    if plan.mirrored_pairs:
        last_report["mirrored_pairs"] = mirrored_mod.summary(
            [r["mirrored_half"] for r in half_reports if r.get("mirrored_half")])
    return all_rows, last_report, failure


def our_team_facts(plan: SeriesPlan, cfg: Any) -> Tuple[int, Optional[str], Optional[str]]:
    """``(count, source, gen3_team_set_digest_v1)`` of the team set OUR side DRAWS FROM.

    Counted and digested through the SAME allowlist both sides use, so `team_source_asymmetry`
    (counts) and `team_content_asymmetry` (digests) in the summary mean a real difference in what
    the two sides draw from — not a stray bookkeeping file. ``source`` is None for a non-peer side
    (the plan's `our_team_spec` names it).
    """
    from main.anchors.session import load_team_texts, pool_team_texts

    if plan.our_side_is_peer:
        # Our side draws through the PEER's own team set (`--our-teamset` in a pair cell), not our
        # team source — count and digest what it will actually draw from.
        ours = our_peer_plan(plan, cfg, "acceptor", 1, "ours_challenge")
        return (peers_mod.team_source_count(ours), str(ours.team_dir) if ours.team_dir else None,
                team_set_digest_or_none(ours.team_dir))
    if plan.our_team_spec.get("kind") == "pool":
        texts = pool_team_texts()
        return len(texts), None, team_set_digest_or_none(texts)
    directory = Path(plan.our_team_spec["path"])
    return len(load_team_texts(directory)), None, team_set_digest_or_none(directory)


def our_team_items(spec: Dict[str, Any]) -> List[Tuple[str, str]]:
    """``(label, text)`` for every team OUR side can draw — the mirrored planner's ``A`` side."""
    from main.anchors.session import load_team_texts

    if spec.get("kind") == "pool":
        from main.anchors.session import pool_team_texts

        return [(f"pool{i:03d}", t) for i, t in enumerate(pool_team_texts())]
    return load_team_texts(Path(spec["path"]))


def their_team_items(pplan: Any) -> List[Tuple[str, str]]:
    """``(label, text)`` for every team the PEER draws from (its configured team directory)."""
    from main.anchors.session import load_team_texts

    team_dir = getattr(pplan, "team_dir", None)
    if not team_dir:
        raise mirrored_mod.MirrorError("this peer names no team directory — it cannot play a mirrored pair")
    return load_team_texts(Path(team_dir))
