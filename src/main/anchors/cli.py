"""``python -m main.anchors`` — one command for an EXTERNAL-ANCHOR read.

    python -m main.anchors --model models/<run> --opponent metamon:SmallRL \\
        --regime greedy --teamset away --games 100 --out <dir>

What it does, in the order it does it:

1. **starts a server** on a caller-given or auto-picked 9XXX port (**8000 and 8001 are refused
   in code**), records the PID, and stops exactly that PID on exit or failure. 🚨 **That server is
   the in-repo websocket FRONT END over the Rust bridge by default** (``--server rust``): no Node
   process is started at all, and each battle is backed by one ``sim_bridge`` child.
   ``--server node`` is the explicit opt-out that starts ``deps/pokemon-showdown``, kept because a
   transport differential needs a reference that is not ours. ``--server-uri`` starts nothing and
   stamps the rows ``server_impl = external``;
2. **runs OUR checkpoint through `main.play`'s own code path** — not a copy — against the named
   opponent at a MATCHED regime, role-balanced across two half-series;
3. **verifies the regime per decision on both sides** and writes the ``argmax_match_rate``;
4. **writes ``games.jsonl`` + ``summary.json``** with the Wilson interval AND, on every row, the
   regime, the team set, the opponent's version and commit, our checkpoint's step, the search
   budget and the realized visit count;
5. **fails loudly with a named cause** on a dead peer or a stalled series, rather than reporting
   nothing.

🚨 **A number never leaves this tool without its regime.** "Temperature 1.0" is not one setting
across models — at T = 1.0 ``SmallRL`` plays its own argmax 64.7% of the time and
``SyntheticRLV2`` 88.0%, so the same nominal knob perturbs 35% of one policy's decisions and 12%
of the other's, and the mixed-regime de-risk headline of ``0.742`` fell to ``0.520`` on the
like-for-like matched cell. The full procedure, the three tiers and the standing numbers:
``designs/ops/EXTERNAL_ANCHORS_SOP.md``.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from main.anchors import config as config_mod
from main.anchors import peers as peers_mod
from main.anchors import results as results_mod
from main.anchors import runner as runner_mod
from main.anchors import server as server_mod

#: ``--opponent`` values. Metamon policies are named ``metamon:<AgentName>``; the agent must be in
#: ``designs/ops/anchors.json`` WITH its checkpoint, because a policy is only an anchor once its
#: checkpoint is pinned.
OPPONENT_HELP = ("metamon:SmallRL | metamon:SyntheticRLV2 | foulplay "
                 "(metamon agents come from designs/ops/anchors.json)")


def parse_our_side(spec: str) -> str:
    """Validate ``--our-side`` here, where a bad name is a refusal — never at game 1, where it is
    a hang. The roster lookup raises KeyError naming the nine."""
    if spec == "model":
        return spec
    if spec.startswith("metamon:"):
        # An ANCHOR-vs-ANCHOR cell: both sides are external peers and nothing of ours plays.
        # The agent still has to be PINNED in designs/ops/anchors.json, which `peer_plan` checks.
        if not spec.partition(":")[2]:
            raise SystemExit("--our-side metamon needs an agent, e.g. --our-side metamon:SmallRL")
        return spec
    if not spec.startswith("bot:"):
        raise SystemExit(
            f"--our-side {spec!r}: expected 'model', 'bot:<name>' or 'metamon:<Agent>'")
    from agents.training.eval_callback import eval_opponent_class

    try:
        eval_opponent_class(spec.split(":", 1)[1])
    except KeyError as exc:
        raise SystemExit(f"--our-side {spec!r}: {exc}") from exc
    return spec


def parse_opponent(spec: str) -> "tuple[str, str]":
    kind, _, agent = spec.partition(":")
    kind = kind.strip().lower()
    if kind not in ("metamon", "foulplay"):
        raise SystemExit(f"--opponent {spec!r}: unknown anchor {kind!r} (known: metamon, foulplay)")
    if kind == "metamon" and not agent:
        raise SystemExit("--opponent metamon needs an agent, e.g. --opponent metamon:SmallRL")
    return kind, agent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m main.anchors",
        description="Play OUR checkpoint against an EXTERNAL anchor opponent at a matched, "
                    "VERIFIED regime, and write games.jsonl + summary.json with the Wilson CI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Procedure, tiers and standing numbers: designs/ops/EXTERNAL_ANCHORS_SOP.md")
    p.add_argument("--model", required=False, default=None,
                   help="run dir / .zip / '<run>@<step>' — resolved by "
                        "agents.training.fixed_opponent_pool.resolve_model_ref, so a BARE RUN DIR "
                        "means that run's LAST SNAPSHOT. Name the .zip or @step to pin a file.")
    p.add_argument("--opponent", default="metamon:SmallRL", help=OPPONENT_HELP)
    p.add_argument("--our-side", dest="our_side", default="model",
                   help="'model' (a checkpoint, the default) or 'bot:<name>' — one of the NINE "
                        "PINNED eval bots (random, heuristic, heuristic2, staller, staller_v2, "
                        "aggressive, aggressive_v2, setup_sweep, setup_sweep_v2). A bot cell is "
                        "what puts an external anchor on the ABSOLUTE scale without routing "
                        "through one of our own checkpoints: the bots carry fixed ratings from "
                        "the bot-vs-bot round robin. A bot has no sampling knob, so such a cell "
                        "is stamped regime_matched=false and our_regime='bot:<name>'.")
    p.add_argument("--model-load", dest="model_load", default="auto",
                   choices=("auto", "bare", "foreign"),
                   help="how to load --model. 'bare' is MaskablePPO.load, what a ladder session "
                        "uses. 'foreign' is load_foreign_opponent, which reads the zip's OWN "
                        "model_config.json and checks the arch_signature — REQUIRED for a frozen "
                        "snapshot from an older run (a bare load dies on an unexpected "
                        "ExtractorBuild kwarg). 'auto' tries bare, falls back to foreign, and "
                        "PRINTS which ran; the winner is stamped on every row.")
    p.add_argument("--regime", choices=("greedy", "t1"), default="greedy",
                   help="BOTH sides move together. greedy is the recurring protocol: it is what "
                        "ladder.json and every other strength number here is taken under, and "
                        "T=1.0 is not a fixed yardstick across opponents.")
    p.add_argument("--teamset", choices=("home", "away"), default="home",
                   help="home = OUR pool (nickname-free, explicit Hidden Power IVs); "
                        "away = Metamon's own 20-team `competitive` gen3ou set. Report BOTH.")
    p.add_argument("--games", type=int, default=100,
                   help="total games, split evenly across the two challenge roles")
    p.add_argument("--out", default=None,
                   help="directory for games.jsonl + summary.json + logs. 🚨 There is NO relative "
                        "default: with --out omitted the read goes to a run-scoped directory "
                        f"under {OUT_ROOT_ENV_VAR} (default: the system temp dir), and the path "
                        "is PRINTED. A read taken from the MAIN checkout — which is where "
                        "models/ lives, so it is where these reads are taken — must never write "
                        "into the tree it is measuring.")
    p.add_argument("--format", dest="battle_format", default="gen3ou")
    p.add_argument("--device", default="cpu",
                   help="torch device for OUR inference; keep 'cpu' — a training arm owns the GPU")
    p.add_argument("--port", type=int, default=None,
                   help="Showdown port to start on (9500-9599; 8000/8001 are REFUSED). "
                        "Default: the first free port in the configured range.")
    p.add_argument("--server", dest="server_kind", default="rust",
                   choices=server_mod.SERVER_KINDS,
                   help="WHICH server this tool starts. 'rust' (the DEFAULT) is the in-repo "
                        "websocket front end (utils.bridge.ws_frontend --impl rust) in its own "
                        "subprocess — NO Node server exists for the life of the read. 'node' "
                        "starts deps/pokemon-showdown and is the explicit opt-out a transport "
                        "differential is taken against. Either way the PID is recorded and "
                        "exactly that PID is stopped.")
    p.add_argument("--server-uri", default=None,
                   help="use an EXISTING server at this ws:// URI and start nothing. The rows are "
                        "then stamped server_impl=external: this tool cannot vouch for a "
                        "transport it did not start.")
    p.add_argument("--seed-base", type=int, default=None,
                   help="--server rust only: derive each battle's PRNG seed from this base, so "
                        "the series is REPRODUCIBLE. There is no counterpart on the Node path.")
    p.add_argument("--capture-dir", default=None,
                   help="--server rust only: write each battle's repro record (commands + "
                        "per-side chunks) here. Pair with --seed-base or it is NOT replayable.")
    p.add_argument("--challenge-mode", dest="challenge_mode", default="serial",
                   choices=("serial", "pipelined"),
                   help="WHEN our side emits the next /challenge in a half we challenge. "
                        "'serial' (the DEFAULT) waits for the previous battle to END; "
                        "'pipelined' is poke-env's own loop, which emits it ~0.4 s into the "
                        "previous battle — hazard H14, the reason --opponent foulplay could not "
                        "run a multi-game ours_challenge half. At --concurrency 1 the two differ "
                        "ONLY in when the PM is sent: battle k+1 could never start before battle "
                        "k ended either way.")
    p.add_argument("--search-time-ms", type=int, default=1000,
                   help="foulplay only: its ONLY budget, and it is WALL CLOCK — the realized visit "
                        "count is recorded per cell because two runs at the same nominal budget "
                        "are not the same opponent unless the box is (UNDERSTANDING rule 23)")
    p.add_argument("--search-parallelism", type=int, default=1,
                   help="foulplay only: how many SAMPLED opponent-set worlds it searches, each "
                        "for the full --search-time-ms")
    p.add_argument("--username", default="Gen3AIAnchor", help="our client's name (<=18 chars)")
    p.add_argument("--peer-username", default=None,
                   help="the peer's name (<=18 chars; default derived from the opponent)")
    p.add_argument("--team-seed", type=int, default=20260914,
                   help="seeds BOTH sides' team draws (ours, and theirs at seed+1)")
    p.add_argument("--nice", type=int, default=10, help="niceness for the peer process")
    p.add_argument("--connect-timeout", type=float, default=60.0, metavar="SECONDS")
    p.add_argument("--peer-ready-timeout", type=float, default=1800.0, metavar="SECONDS",
                   help="how long the peer may take to come online. Generous by default: "
                        "SyntheticRLV2 spends minutes building a 200M network and loading 804 MB "
                        "of weights before its client connects.")
    p.add_argument("--first-game-timeout", type=float, default=1800.0, metavar="SECONDS",
                   help="no first game inside this ⇒ FAIL 'no_first_game'. A rejected team or a "
                        "dropped challenge presents as a HANG, not an error.")
    p.add_argument("--progress-timeout", type=float, default=900.0, metavar="SECONDS",
                   help="no game finishing inside this ⇒ FAIL 'no_progress'")
    p.add_argument("--allow-unmatched-regime", action="store_true",
                   help="permit a cell where the two sides are NOT at the same regime. Every row "
                        "is then stamped regime_matched=false. Foul Play has no sampling knob at "
                        "all, so --regime t1 against it needs this and is not a matched read.")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan — both commands, both team sources, every deadline — and "
                        "exit without starting anything")
    p.add_argument("--show-config", action="store_true",
                   help="print designs/ops/anchors.json as resolved, with each env override")
    return p


#: Points the DEFAULT output root somewhere else. Only the default moves; an explicit ``--out`` is
#: always taken verbatim, because a measurement directory under ``designs/research_state/`` is a
#: deliberate and committed destination.
OUT_ROOT_ENV_VAR = "GEN3AI_ANCHORS_OUT_ROOT"


def default_out_root() -> Path:
    """The scratch root a ``--out``-less read writes under — **never the calling directory**.

    🚨 The old default was ``Path.cwd() / "anchors_out"``. Anchor reads are taken from the MAIN
    checkout, because that is the only tree with ``models/``, so the relative default filled the
    repo it was measuring: one ``git clean`` from a lost measurement and one ``git status`` from a
    landing that stops (it already happened, 2026-09-16). The root is independent of
    ``$GEN3AI_MODELS_DIR`` and of the repo — it is the system temp dir unless
    ``$GEN3AI_ANCHORS_OUT_ROOT`` names another.
    """
    override = os.environ.get(OUT_ROOT_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return Path(tempfile.gettempdir()) / "gen3ai_anchors"


def _slug(text: str) -> str:
    """Filesystem-safe, and never empty — a cell whose slug collapsed to '' would share a
    directory with every other such cell."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return cleaned or "cell"


def default_out_dir(args: argparse.Namespace, now: Optional[float] = None) -> Path:
    """A RUN-SCOPED directory under :func:`default_out_root`, named for the cell it holds.

    Two reads a second apart do not collide (the stamp carries the seconds) and two different
    cells never share a directory even inside one second (the slug carries opponent, regime and
    team set), so a partially-written ``summary.json`` can only ever belong to the read that
    wrote it.
    """
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now if now is not None else time.time()))
    cell = "_".join((_slug(args.opponent), _slug(args.regime), _slug(args.teamset)))
    return default_out_root() / f"{stamp}_{cell}"


def _default_peer_username(kind: str, agent: str) -> str:
    """<= 18 characters, alphanumeric. Both clients guest-login through Smogon's `action.php`
    even against a `--no-security` LOCAL server, and 19+ characters comes back as a REFUSAL that
    Foul Play accepts as an assertion and then hangs on forever."""
    stem = "Meta" + agent if kind == "metamon" else "FoulPlay"
    return "".join(ch for ch in stem if ch.isalnum())[:18] or "Anchor"


def resolve_model(spec: str) -> "tuple[str, Optional[int], str]":
    """``(zip path, step, rung)`` — through the ONE choke point every run-spec consumer uses.

    🚨 A BARE RUN DIRECTORY MEANS THE RUN'S LAST SNAPSHOT. The rung is recorded on every row so a
    reader never has to guess which file a directory resolved to.
    """
    from agents.training.fixed_opponent_pool import resolve_model_ref

    resolved = resolve_model_ref(spec)
    return resolved.zip_path, resolved.num_timesteps, resolved.rung


#: The pinned submodule commit both clients played on. Recorded on BOTH transports — the Rust port
#: was ported from this tree, and the front end still validates a `/utm` team through its
#: `validate_team.js` — because a comparison against numbers a third party produced on ITS own
#: server is a different measurement.
showdown_pin = server_mod.showdown_pin


def build_plan(args: argparse.Namespace, cfg: config_mod.AnchorsConfig,
               ) -> runner_mod.SeriesPlan:
    kind, agent = parse_opponent(args.opponent)
    our_side = parse_our_side(args.our_side)

    # A bot plays its own fixed policy and there is no temperature to turn down, so `t1` against
    # a bot our-side would name a regime that only one side is at while pretending both are.
    if our_side == args.opponent:
        raise SystemExit(
            f"--our-side {our_side} equals --opponent: a cell of a policy against ITSELF measures "
            "nothing about the scale, and both sides would try to log in under names derived "
            "from the same agent.")
    if our_side.startswith("bot:") and args.regime != "greedy":
        raise SystemExit(
            f"--regime {args.regime} with --our-side {our_side}: a bot has no sampling knob, so "
            "only the PEER would move. Use --regime greedy; the cell is stamped "
            "regime_matched=false either way because the two sides are not at one nominal "
            "regime, and that is the honest label for a bot edge.")

    # Foul Play searches; it has no temperature and no sampling knob, so `t1` cannot be MATCHED
    # against it. Refusing is the point of the tool: an unmatched cell reported as a matched one is
    # the exact mistake the 2026-09-14 battery was run to correct.
    # A bot our-side is never "matched": it plays its own fixed policy and has no knob to set to
    # the peer's regime. Same shape as Foul Play, and stamped the same way.
    # An anchor-vs-anchor cell IS matched — both peers take the same `--regime` and both verify
    # it per decision. A BOT our-side is not, because a bot has no knob to set.
    matched = not our_side.startswith("bot:")
    if kind == "foulplay" and args.regime != "greedy":
        if not args.allow_unmatched_regime:
            raise SystemExit(
                "--regime t1 against foulplay is NOT a matched regime: Foul Play is a search bot "
                "with no sampling knob, so only OUR side would move. Pass "
                "--allow-unmatched-regime to take it anyway (every row is stamped "
                "regime_matched=false), or use --regime greedy.")
        matched = False

    # The reproducibility pair is the FRONT END's, and there is no Node counterpart — a
    # `--seed-base` silently ignored on the Node path would make an unrepeatable series look
    # seeded, which is the one failure a seed exists to prevent.
    if (args.seed_base is not None or args.capture_dir) and (
            args.server_uri or args.server_kind != "rust"):
        raise SystemExit(
            "--seed-base/--capture-dir need --server rust (the websocket front end owns both). "
            "The Node server mints its own seed per battle and has no capture; a seed accepted "
            "and ignored is worse than a refusal.")
    if args.capture_dir and args.seed_base is None:
        print("⚠️  --capture-dir without --seed-base: each battle's child mints its own seed, so "
              "the records will NOT be replayable.", flush=True)

    if args.server_uri:
        uri = args.server_uri
        port = server_mod.port_of(uri)
        server_mod.refuse_reserved(port)
        started = False
        # We did not start it, so we cannot say what it is. "external" is the honest stamp; the
        # alternative — copying --server onto a row about a process this tool never saw — is how
        # a number ends up attributed to a transport it never touched.
        server_impl, server_version = "external", ""
    else:
        port = args.port if args.port is not None else server_mod.pick_port(cfg.port_range)
        server_mod.refuse_reserved(port)
        uri = server_mod.server_uri(port)
        started = True
        server_impl = args.server_kind
        server_version = server_mod.build_server(
            args.server_kind, port, node=cfg.node, battle_format=args.battle_format,
            seed_base=args.seed_base,
            capture_dir=Path(args.capture_dir) if args.capture_dir else None).version()

    if args.model and our_side == "model":  # noqa: SIM108 - a peer/bot our-side has no zip
        zip_path, step, rung = resolve_model(args.model)
    else:
        zip_path, step, rung = "", None, ""

    from main.play import DEFAULT_FORFEIT_TURN_LIMIT

    # 🚨 NEVER the calling directory. See `default_out_root`.
    out_dir = Path(args.out).expanduser() if args.out else default_out_dir(args)
    return runner_mod.SeriesPlan(
        opponent=args.opponent,
        opponent_kind=kind,
        opponent_agent=agent,
        regime=args.regime,
        regime_matched=matched,
        teamset=args.teamset,
        games=args.games,
        battle_format=args.battle_format,
        model_spec=args.model or "",
        model_zip=zip_path,
        model_step=step,
        model_rung=rung,
        device=args.device,
        server_uri=uri,
        server_port=port,
        started_server=started,
        server_impl=server_impl,
        server_version=server_version,
        seed_base=args.seed_base,
        capture_dir=Path(args.capture_dir) if args.capture_dir else None,
        challenge_mode=args.challenge_mode,
        out_dir=out_dir,
        # A peer our-side is named after ITS OWN agent: Metamon keys its per-battle CSV by the
        # player's username, and "Gen3AIAnchor" on an anchor-vs-anchor row would name a client
        # that is not ours at all. 18 characters is Showdown's ceiling and the suffix costs one.
        our_username=(_default_peer_username("metamon", our_side.partition(":")[2])[:17]
                      if our_side.startswith("metamon:") else args.username),
        peer_username=args.peer_username or _default_peer_username(kind, agent),
        team_seed=args.team_seed,
        search_time_ms=args.search_time_ms if kind == "foulplay" else None,
        search_parallelism=args.search_parallelism if kind == "foulplay" else None,
        forfeit_turn_limit=DEFAULT_FORFEIT_TURN_LIMIT,
        connect_timeout_s=args.connect_timeout,
        peer_ready_timeout_s=args.peer_ready_timeout,
        first_game_timeout_s=args.first_game_timeout,
        progress_timeout_s=args.progress_timeout,
        nice=args.nice,
        showdown_pin=showdown_pin(),
        our_team_spec=cfg.our_team_source(args.teamset),
        our_side=our_side,
        model_loader=args.model_load,
    )


def render_plan(plan: runner_mod.SeriesPlan, cfg: config_mod.AnchorsConfig) -> str:
    """``--dry-run``: everything that will happen, including both peer commands verbatim."""
    lines = [
        "ANCHOR READ — PLAN (nothing has been started)",
        f"  opponent          {plan.opponent}",
        f"  regime            {plan.regime}  (both sides; matched={plan.regime_matched})",
        f"  team set          {plan.teamset}  ours={plan.our_team_spec}",
        f"  games             {plan.games}  ->  {plan.half_sizes()}",
        f"  our side          {plan.our_side}"
        + (f"  (loader={plan.model_loader})" if plan.our_side == "model" else ""),
        f"  our model         {plan.model_zip or '(none — our side is not a checkpoint)'}"
        + (f"  @ step {plan.model_step} (via {plan.model_rung})" if plan.model_step else ""),
        f"  device            {plan.device}",
        f"  server            {plan.server_uri}"
        + (f"  [{plan.server_impl}] (this tool starts and stops it by PID)"
           if plan.started_server else "  (EXISTING — nothing started; stamped external)"),
        f"  server version    {plan.server_version or '(unknown — not started by this tool)'}"
        + ("  🚨 NO NODE SERVER IS STARTED" if plan.server_impl == "rust" else ""),
        f"  showdown pin      {plan.showdown_pin}",
        f"  usernames         ours={plan.our_username}<N> peer={plan.peer_username}<N>  "
        "(a per-half suffix; a name still held by the previous half logs in as a GUEST)",
        f"  forfeit limit     {plan.forfeit_turn_limit} turns (the TRAINER's number)",
        f"  challenge mode    {plan.challenge_mode}"
        + ("  (the next /challenge waits for the previous battle to END — hazard H14)"
           if plan.challenge_mode == "serial"
           else "  🚨 poke-env's own loop: the next /challenge lands ~0.4 s INTO the previous "
                "battle (hazard H14)"),
        f"  deadlines         peer_ready={plan.peer_ready_timeout_s:g}s "
        f"first_game={plan.first_game_timeout_s:g}s progress={plan.progress_timeout_s:g}s",
        f"  out               {plan.out_dir}",
    ]
    if plan.search_time_ms:
        lines.append(f"  search budget     {plan.search_time_ms} ms x "
                     f"{plan.search_parallelism} world(s) — WALL CLOCK, so the realized visit "
                     "count is recorded per cell")
    for half, n in plan.half_sizes().items():
        if n <= 0:
            continue
        mode, role, _suffix = runner_mod.HALVES[half]
        our_name, peer_name = runner_mod.half_usernames(plan, half)
        try:
            pplan = runner_mod.peer_plan(plan, cfg, role, n, half)
            peer_cmd = pplan.command_line()
            teams = f"{pplan.team_dir} ({peers_mod.team_source_count(pplan)} files)"
        except Exception as exc:                     # noqa: BLE001 - a plan must still PRINT
            peer_cmd = f"<unavailable: {type(exc).__name__}: {exc}>"
            teams = "<unavailable>"
        if plan.our_side_is_peer:
            # Our side is a SECOND external process, not `main.play`. Printing the play.py argv
            # here would name a command this cell never runs — a plan a human cannot execute is
            # worse than no plan at all.
            try:
                ours_cmd = runner_mod.our_peer_plan(plan, cfg, role, n, half).command_line()
            except Exception as exc:                 # noqa: BLE001 - a plan must still PRINT
                ours_cmd = f"<unavailable: {type(exc).__name__}: {exc}>"
        else:
            ours_cmd = ("python -m main.play "
                        + " ".join(runner_mod.our_argv(plan, mode, n, our_name, peer_name)))
        lines += [
            "",
            f"  --- half {half} ({n} games; we {mode}, peer is {role}) ---",
            f"    ours: {ours_cmd}",
            f"    peer: {peer_cmd}",
            f"    peer teams: {teams}",
        ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_mod.load_config()

    if args.show_config:
        print(f"anchors config: {cfg.source}")
        print(config_mod.describe(cfg))
        return 0

    plan = build_plan(args, cfg)
    if not args.out:
        print(f"⚠️  no --out given — this read goes to {plan.out_dir}\n"
              f"    (run-scoped, under ${OUT_ROOT_ENV_VAR} or the system temp dir; NEVER the "
              "calling directory, which for an anchor read is the repo being measured)",
              flush=True)

    if args.dry_run:
        print(render_plan(plan, cfg))
        return 0

    if not args.model and plan.our_side == "model":
        raise SystemExit("--model is required for a real read (only --dry-run/--show-config "
                         "work without it)")

    plan.out_dir.mkdir(parents=True, exist_ok=True)
    print(render_plan(plan, cfg), flush=True)

    srv: Optional[server_mod.ManagedServer] = None
    rows: List[results_mod.GameRow] = []
    report: dict = {}
    failure = None
    try:
        if plan.started_server:
            srv = server_mod.build_server(
                plan.server_impl, plan.server_port, node=cfg.node,
                battle_format=plan.battle_format, seed_base=plan.seed_base,
                capture_dir=plan.capture_dir, out_dir=plan.out_dir).start()
            print(f"[anchors] {srv.label} pid={srv.pid} on {srv.uri} ({plan.server_version})",
                  flush=True)
        rows, report, failure = asyncio.run(runner_mod.run_series(plan, cfg))
    except KeyboardInterrupt:
        failure = runner_mod.SeriesFailure("interrupted", "KeyboardInterrupt")
    finally:
        if srv is not None:
            srv.stop()
            print(f"[anchors] {srv.label} stopped (pid was {srv.pid})", flush=True)

    cell = runner_mod.cell_spec(plan, report, rows[0].cell.our_team_count if rows else 0)
    if rows:
        cell = rows[0].cell
    status = "FAILED" if failure is not None else "OK"
    if failure is None and len(rows) < plan.games:
        status = "FAILED"
        failure = runner_mod.SeriesFailure(
            "short_series",
            f"{len(rows)}/{plan.games} games completed with no named failure — a partial n is "
            "only honest when it is labelled partial")

    games_path = plan.out_dir / "games.jsonl"
    summary_path = plan.out_dir / "summary.json"
    results_mod.write_games(games_path, rows)
    summary = results_mod.summarize(
        cell, rows, status=status,
        failure=failure.as_dict() if failure is not None else None,
        provenance={
            "anchors_config": str(cfg.source),
            "peer_report": {k: v for k, v in report.items() if k != "team_draws"},
            "games_path": str(games_path),
        })
    results_mod.write_summary(summary_path, summary)

    print("")
    print("ANCHOR READ — RESULT")
    print(results_mod.render(summary))
    print(f"  wrote       {games_path} ({len(rows)} rows)")
    print(f"  wrote       {summary_path}")
    if status != "OK":
        print("")
        print("🚨 THIS READ IS NOT A MEASUREMENT — see summary.json's `failure` block.")
        return 2
    if not summary["their_argmax_match_rates"] and plan.opponent_kind == "metamon":
        print("")
        print("⚠️  the peer reported no argmax_match_rate — the regime is UNVERIFIED for this "
              "cell, so the number above must not be quoted as a matched read.")
    return 0


def run() -> int:
    try:
        return main()
    except (config_mod.AnchorConfigError, server_mod.ServerError) as exc:
        print(f"🚨 {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(run())
