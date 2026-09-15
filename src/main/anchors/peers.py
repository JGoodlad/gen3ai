"""The OTHER client — how each external anchor is launched, watched, and read back.

Both anchors are third-party checkouts with their own interpreters, so each is a SUBPROCESS and
nothing here imports either. A peer adapter owns four things and no more:

1. **the command** — argv, cwd and env, including every hazard workaround the 2026-09-14 de-risks
   paid for (the hardcoded ``ws://localhost:8000`` rebind, ``PYTHONUNBUFFERED``, CPU pinning,
   ``VanillaAttention``);
2. **readiness** — the line in its own log that means "online and waiting", because the ACCEPTOR
   must be logged in before the CHALLENGER sends its first ``/challenge``: Showdown drops a
   challenge aimed at a user who is not online and the challenger then waits forever;
3. **what it reports about ITSELF** — the regime it actually used (``sample`` kwarg,
   ``argmax_match_rate``) and, for Foul Play, the REALIZED search width;
4. **its version** — the checkout's git HEAD, since neither checkout is ours to pin.

**The 19-character username rule is enforced here, for both names.** Both clients guest-login
through Smogon's ``action.php`` *even against a ``--no-security`` LOCAL server*; a name of 19+
characters comes back as ``;;Your username must be less than 19 characters long.`` — a refusal,
not an assertion. Foul Play's guest path accepts that string verbatim, logs "Successfully logged
in", and then **blocks forever**. It cost a session and ~18 minutes on 2026-09-14, and it is
exactly the failure class our own ladder audit fixed as gap #10. A refusal must never present as a
hang, so the length is checked in code before anything is started.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.paths import src_path

#: Showdown's own ceiling, via `action.php`. 19 is REFUSED, so 18 is the last legal length.
MAX_USERNAME_LEN = 18

#: The team-file shapes a ``--teamset`` directory may hold, on EITHER side. An allowlist rather
#: than "every file", because these directories carry bookkeeping beside the teams — Metamon's
#: `competitive` gen3ou set ships an `index.csv`, which a naive count reports as a 21st team.
TEAM_FILE_GLOBS = ("*_team", "*.txt", "*.team")


class PeerError(RuntimeError):
    """The peer could not be configured, started, or read back."""


def check_username(name: str, whose: str) -> None:
    """Refuse a name ``action.php`` will not issue an assertion for — see the module docstring."""
    if len(name) > MAX_USERNAME_LEN:
        raise PeerError(
            f"{whose} username {name!r} is {len(name)} characters; Smogon's action.php refuses "
            f"{MAX_USERNAME_LEN + 1}+ with a ';'-prefixed message that Foul Play's guest path "
            "accepts as an assertion and then HANGS on forever. Shorten it."
        )
    if not re.fullmatch(r"[A-Za-z0-9]+", name):
        raise PeerError(
            f"{whose} username {name!r} must be alphanumeric — a name Showdown normalises away "
            "makes the two sides disagree about who they are challenging."
        )


def git_head(checkout: Path) -> str:
    """The checkout's short HEAD, or ``"unknown"``. Never raises: a missing git is a weaker
    provenance record, not a reason to refuse a measurement."""
    try:
        out = subprocess.run(["git", "-C", str(checkout), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


@dataclass
class PeerPlan:
    """Everything needed to start one peer process, and to read it back afterwards."""

    label: str
    argv: List[str]
    cwd: Path
    env: Dict[str, str]
    ready_pattern: str
    log_path: Path
    report_path: Optional[Path] = None
    #: Human-readable regime for the OTHER side, as it will be stamped on every row.
    their_regime: str = "greedy"
    version: str = ""
    commit: str = ""
    team_dir: Optional[Path] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def command_line(self) -> str:
        return " ".join(self.argv)


class Peer:
    """Base adapter. Subclasses build the plan and read their own log back."""

    kind = ""

    def plan(self, **kwargs: Any) -> PeerPlan:          # pragma: no cover - interface
        raise NotImplementedError

    @staticmethod
    def read_report(plan: PeerPlan) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


def _count_team_files(d: Optional[Path]) -> int:
    """How many team files the peer will draw from. 0 when the directory is unreadable — recorded
    as 0 rather than guessed, so a team-source asymmetry shows up in the summary rather than
    looking like agreement."""
    if d is None or not d.is_dir():
        return 0
    names = {p.name for g in TEAM_FILE_GLOBS for p in d.rglob(g)
             if p.is_file() and not p.name.startswith(".")}
    return len(names)


class MetamonPeer(Peer):
    """A Metamon pretrained policy as one websocket client.

    Four hazard workarounds ride in the plan, each measured rather than assumed:

    * **H4, the hardcoded server.** ``PokeEnvWrapper.server_configuration`` returns the module-level
      ``LocalhostServerConfiguration``, which poke-env hardcodes to ``ws://localhost:8000`` — our
      shared dev server, one port from the live training server. No flag, no env var, no
      constructor argument. The driver rebinds the module global before any env is built.
    * **H2, FlashAttention.** ``amago``'s ``TformerTrajEncoder`` defaults to ``FlashAttention``, a
      CUDA-only wheel, so every Metamon transformer is unrunnable on CPU as shipped.
      ``VanillaAttention`` is the same exact causal softmax attention, computed the slow way.
    * **H5, unflushed prints.** Metamon's ``print()`` calls carry no ``flush=True``, so a redirected
      stdout is block-buffered and the banner a driver waits on never lands.
    * **H-D, the clipped distribution.** ``MetamonDiscrete`` clips probabilities to [0.001, 0.99]
      AFTER dividing by the temperature, so on the 9-way action space the argmax tops out at
      ~0.992 however cold the temperature. **Greedy is ``sample=False`` and nothing else.**
    """

    kind = "metamon"

    def plan(self, *, cfg: Any, agent: str, regime: str, teamset: str, battle_format: str,
             server_uri: str, username: str, opponent_username: str, role: str, n_games: int,
             team_seed: int, out_dir: Path, **_: Any) -> PeerPlan:
        cfg.require(agent)
        checkpoint = cfg.agents[agent].get("checkpoint")
        team_set = cfg.team_set(teamset)
        driver = src_path("main", "anchors", "peer_scripts", "metamon_side.py")
        log_path = out_dir / f"peer_{role}.log"
        report_path = out_dir / f"peer_{role}_report.json"
        argv = [
            str(cfg.python), str(driver),
            "--server-uri", server_uri,
            "--agent", agent,
            "--username", username,
            "--opponent-username", opponent_username,
            "--role", role,
            "--total-battles", str(n_games),
            "--battle-format", battle_format,
            "--team-set", team_set,
            "--team-seed", str(team_seed),
            "--regime", regime,
            "--report-out", str(report_path),
            "--results-dir", str(out_dir / f"peer_{role}_results"),
        ]
        if checkpoint is not None:
            argv += ["--checkpoint", str(checkpoint)]
        env = dict(os.environ)
        env.update({
            "PYTHONUNBUFFERED": "1",                    # H5
            "CUDA_VISIBLE_DEVICES": "",                 # a training arm owns the GPU
            "METAMON_CACHE_DIR": str(cfg.cache_dir),
            # The metamon env runs UPSTREAM poke-env; our src/ must not reach its sys.path or the
            # two packages named `poke_env` fight and the loser is silent.
            "PYTHONPATH": "",
        })
        return PeerPlan(
            label=f"metamon:{agent}",
            argv=argv,
            cwd=cfg.checkout,
            env=env,
            # Metamon's own banner. `run_metamon_side.py` proved this is the line that means the
            # policy is BUILT (minutes, for the 200M model) and the client is connecting.
            ready_pattern=r"Made Challenge Env|\[peer\] READY",
            log_path=log_path,
            report_path=report_path,
            their_regime=regime,
            version=f"{agent}@ckpt{checkpoint}",
            commit=git_head(cfg.checkout),
            team_dir=cfg.team_dir(teamset, battle_format),
        )

    @staticmethod
    def read_report(plan: PeerPlan) -> Dict[str, Any]:
        """The driver's own JSON: the ``sample`` kwarg it really received and the
        ``argmax_match_rate`` that proves the regime took effect.

        An absent report is reported as absent — ``regime_verified: False`` — never as a pass.
        """
        import json
        out: Dict[str, Any] = {"regime_verified": False, "sample_kwargs": [],
                               "argmax_match_rate": None, "visits_mean": None, "visits_n": 0}
        if plan.report_path is None or not plan.report_path.exists():
            out["note"] = f"no peer report at {plan.report_path}"
            return out
        blob = json.loads(plan.report_path.read_text())
        out.update({
            "sample_kwargs": blob.get("sample_kwarg_values", []),
            "argmax_match_rate": blob.get("argmax_match_rate"),
            "regime_verified": bool(blob.get("regime_check_ok")),
            "peer_wins": blob.get("peer_wins"),
            "peer_games": blob.get("peer_games"),
            "n_decisions": blob.get("n_decisions"),
            "median_decision_s": blob.get("median_s"),
        })
        return out


class FoulPlayPeer(Peer):
    """Foul Play + poke-engine (gen3) as one websocket client.

    🚨 **Its only budget is WALL CLOCK.** ``--search-time-ms`` goes straight to
    ``poke_engine.monte_carlo_tree_search(state, search_time_ms, threads=...)``; there is no
    iteration, visit or depth budget anywhere in the CLI or the binding. Per UNDERSTANDING rule 23
    that makes it a **width meter**, not a setting: at a constant 1000 ms the realized search
    measured 1.21–1.53 M visits/decision across sessions on one box. :meth:`read_report` therefore
    scrapes the realized width out of its own ``Iterations {i}: {total_visits}`` lines, and every
    row carries it — a Foul Play win rate without its visit count is not reproducible anywhere.

    There is no sampling knob, so ``--regime t1`` cannot be matched against it; the CLI refuses
    that combination rather than quietly reporting an unmatched cell as a matched one.
    """

    kind = "foulplay"

    def plan(self, *, cfg: Any, regime: str, teamset: str, battle_format: str, server_uri: str,
             username: str, opponent_username: str, role: str, n_games: int,
             search_time_ms: int, search_parallelism: int, out_dir: Path, **_: Any) -> PeerPlan:
        cfg.require()
        log_path = out_dir / f"peer_{role}.log"
        mode = "challenge_user" if role == "challenger" else "accept_challenge"
        argv = [
            str(cfg.python), "-u", str(cfg.checkout / "run.py"),
            # The full URI, scheme included — Foul Play's `get_websocket` passes anything
            # that is not the literal "ps"/"local" through unchanged to `websockets.connect`.
            "--websocket-uri", server_uri,
            "--ps-username", username,
            "--bot-mode", mode,
            "--pokemon-format", battle_format,
            "--team-name", cfg.team_name(teamset),
            "--search-time-ms", str(search_time_ms),
            "--search-parallelism", str(search_parallelism),
            "--search-threads", "1",
            "--run-count", str(n_games),
            "--log-level", "INFO",
        ]
        if mode == "challenge_user":
            argv += ["--user-to-challenge", opponent_username]
        env = dict(os.environ)
        env.update({"PYTHONUNBUFFERED": "1", "PYTHONPATH": ""})
        return PeerPlan(
            label="foulplay",
            argv=argv,
            cwd=cfg.checkout,
            env=env,
            # Foul Play's own formatter prints no timestamps; this is the first line it emits once
            # the websocket is up and it has a userid.
            ready_pattern=r"Successfully logged in|Starting battle|Challenging",
            log_path=log_path,
            their_regime=f"search:{search_time_ms}ms",
            version=f"poke-engine gen3 @ {search_time_ms}ms x{search_parallelism}",
            commit=git_head(cfg.checkout),
            team_dir=cfg.team_dir(teamset),
            extra={"search_time_ms": search_time_ms, "search_parallelism": search_parallelism},
        )

    #: Foul Play's own line. ``total_visits`` is the engine's realized MCTS width for that decision.
    VISITS_RE = re.compile(r"Iterations\s+\d+:\s+([\d.eE+]+)")

    @staticmethod
    def read_report(plan: PeerPlan) -> Dict[str, Any]:
        out: Dict[str, Any] = {"regime_verified": True, "sample_kwargs": [],
                               "argmax_match_rate": None, "visits_mean": None, "visits_n": 0}
        if not plan.log_path.exists():
            out["regime_verified"] = False
            out["note"] = f"no peer log at {plan.log_path}"
            return out
        text = plan.log_path.read_text(errors="replace")
        visits = [float(m) for m in FoulPlayPeer.VISITS_RE.findall(text)]
        if visits:
            out["visits_mean"] = sum(visits) / len(visits)
            out["visits_n"] = len(visits)
            out["visits_min"] = min(visits)
            out["visits_max"] = max(visits)
        else:
            # 🚨 Not an aside. The nominal budget is wall clock, so without the realized width the
            # number cannot be compared to any other box or any other day.
            out["regime_verified"] = False
            out["note"] = ("no `Iterations N: <visits>` lines in the Foul Play log — the REALIZED "
                           "search width is unknown, and a wall-clock budget without it is not a "
                           "reproducible opponent (UNDERSTANDING rule 23)")
        out["errors"] = text.count("ERROR") + text.count("CRITICAL")
        return out


PEERS: Dict[str, Peer] = {"metamon": MetamonPeer(), "foulplay": FoulPlayPeer()}


def team_source_count(plan: PeerPlan) -> int:
    return _count_team_files(plan.team_dir)
