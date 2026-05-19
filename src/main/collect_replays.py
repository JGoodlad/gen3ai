"""
Daemon script that spectates Pokémon Showdown battles and saves raw replay logs.

Runs until Ctrl+C. Each battle is saved immediately when it finishes.
Restarts are safe — already-saved files are not overwritten.

Usage:
    python src/main/collect_replays.py --format gen3ou --save-dir replays/gen3ou
    python src/main/collect_replays.py --format gen3ou --save-dir replays/gen3ou --local
"""

import argparse
import asyncio
import collections
import logging
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import rich.box
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from poke_env.concurrency import POKE_LOOP
from poke_env.ps_client.server_configuration import (
    LocalhostServerConfiguration,
    ShowdownServerConfiguration,
)
from poke_env.spectator import BattleSpectator, SpectatedBattle


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

@dataclass
class _CompletedEntry:
    index: int
    battle_tag: str
    winner: Optional[str]
    turn: int


class CollectorState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_collected: int = 0
        self.skipped: int = 0
        self.recent: List[_CompletedEntry] = []
        self.log_lines: collections.deque = collections.deque(maxlen=100)
        self.start_time: float = time.time()

    def add_completed(self, battle: SpectatedBattle, saved: bool) -> None:
        with self._lock:
            self.total_collected += 1
            if not saved:
                self.skipped += 1
            self.recent.append(_CompletedEntry(
                index=self.total_collected,
                battle_tag=battle.battle_tag,
                winner=battle.winner,
                turn=battle.turn,
            ))
            if len(self.recent) > 5:
                self.recent = self.recent[-5:]  # keep 5; display cap is layout-dependent

    def add_log(self, line: str) -> None:
        with self._lock:
            self.log_lines.append(line)

    def snapshot_recent(self) -> List[_CompletedEntry]:
        with self._lock:
            return list(reversed(self.recent))

    def snapshot_logs(self) -> List[str]:
        with self._lock:
            return list(self.log_lines)


# ---------------------------------------------------------------------------
# Async worker
# ---------------------------------------------------------------------------

async def _run(
    format_id: str,
    save_dir: Path,
    spectator: BattleSpectator,
    state: CollectorState,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    async for battle in spectator.watch(format_id):
        path = save_dir / f"{battle.battle_tag}.log"
        saved = not path.exists()
        if saved:
            path.write_text(battle.log_text, encoding="utf-8")
        state.add_completed(battle, saved)


# ---------------------------------------------------------------------------
# Rich dashboard
# ---------------------------------------------------------------------------

def _elapsed_str(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _room_num(battle_tag: str) -> str:
    return battle_tag.rsplit("-", 1)[-1]


_OVERHEAD = 19  # border + stats + section headers + separators + spacing (no data rows)
_LOG_ROWS = 5


def _layout(terminal_rows: int) -> tuple[int, int]:
    """Return (max_active_rows, recent_cap) for the given terminal height."""
    # Try fitting 5 recent completions first, fall back to 3
    for recent_cap in (5, 3):
        active = terminal_rows - _OVERHEAD - recent_cap - _LOG_ROWS
        if active >= 1:
            return active, recent_cap
    return 1, 3


def render_dashboard(
    spectator: BattleSpectator,
    state: CollectorState,
    format_id: str,
    save_dir: Path,
    max_concurrent: int,
    proxy_url: Optional[str] = None,
) -> object:
    elapsed = time.time() - state.start_time
    active = spectator.active_battles
    term_rows = shutil.get_terminal_size(fallback=(80, 24)).lines
    max_active, recent_cap = _layout(term_rows)

    # ── Proxy badge ──────────────────────────────────────────────────────
    if proxy_url:
        proxy_badge = f"[bold green]PROXIED[/bold green] [dim]{proxy_url}[/dim]"
    else:
        proxy_badge = "[bold yellow]DIRECT[/bold yellow] [dim](no proxy)[/dim]"

    # ── Stats row ────────────────────────────────────────────────────────
    stats = Table.grid(padding=(0, 3), expand=True)
    for _ in range(6):
        stats.add_column(justify="center")
    stats.add_row(
        f"[bold green]{state.total_collected}[/bold green]\n[dim]collected[/dim]",
        f"[bold cyan]{len(active)}/{max_concurrent}[/bold cyan]\n[dim]watching[/dim]",
        f"[bold yellow]{spectator.pending_count}[/bold yellow]\n[dim]queued[/dim]",
        f"[bold white]{spectator.seen_count}[/bold white]\n[dim]seen total[/dim]",
        f"[dim]{_elapsed_str(elapsed)}[/dim]\n[dim]elapsed[/dim]",
        f"{proxy_badge}\n[dim]connection[/dim]",
    )

    # ── Active battles ───────────────────────────────────────────────────
    active_tbl = Table(
        "Room #", "Turn", "Players",
        box=rich.box.SIMPLE_HEAD, header_style="dim",
        show_edge=False, expand=True, padding=(0, 1),
    )
    if active:
        now = time.time()
        visible = sorted(active.items())[:max_active]
        hidden = len(active) - len(visible)
        for tag, battle in visible:
            num = _room_num(tag)
            p = battle.players
            players_str = f"{p.get('p1', '?')} vs {p.get('p2', '?')}" if p else "[dim]loading…[/dim]"
            age_str = f"[dim]{_elapsed_str(now - battle.joined_at)}[/dim]"
            turn_str = f"[bold]{battle.turn}[/bold]" if battle.turn > 0 else "[dim]--[/dim]"
            active_tbl.add_row(f"[cyan]#{num}[/cyan]  {age_str}", turn_str, players_str)
        if hidden:
            active_tbl.add_row(f"[dim]… {hidden} more[/dim]", "", "")
    else:
        active_tbl.add_row("[dim]—[/dim]", "[dim]—[/dim]", "[dim]connecting…[/dim]")

    # ── Recent completions (last 5) ──────────────────────────────────────
    recent_tbl = Table(
        "#", "Room #", "Winner", "Turns",
        box=rich.box.SIMPLE_HEAD, header_style="dim",
        show_edge=False, expand=True, padding=(0, 1),
    )
    for entry in state.snapshot_recent()[:recent_cap]:
        winner_str = entry.winner if entry.winner else "[dim italic]tie[/dim italic]"
        recent_tbl.add_row(
            f"[dim]{entry.index}[/dim]",
            f"[cyan]#{_room_num(entry.battle_tag)}[/cyan]",
            winner_str,
            f"[dim]{entry.turn}[/dim]",
        )

    # ── Logs (last 5 INFO+ lines) ────────────────────────────────────────
    log_text = Text()
    for line in state.snapshot_logs()[-5:]:
        if "ERROR" in line or "CRITICAL" in line:
            log_text.append(line + "\n", style="bold red")
        elif "WARNING" in line:
            log_text.append(line + "\n", style="yellow")
        else:
            log_text.append(line + "\n", style="dim")
    if not log_text._spans and not log_text._text:
        log_text.append("[dim]No log messages yet…[/dim]")

    title = (
        f"[bold blue]{format_id}[/bold blue] Replay Collector"
        f"  ·  [dim]{save_dir}/[/dim]"
    )
    return Panel(
        Group(
            stats,
            Text(""),
            Text(" Active Battles", style="bold"),
            active_tbl,
            Text(""),
            Text(" Recent Completions", style="bold"),
            recent_tbl,
            Text(""),
            Text(" Logs", style="bold"),
            log_text,
        ),
        title=title,
        border_style="blue",
        padding=(0, 1),
    )


# ---------------------------------------------------------------------------
# Logging capture
# ---------------------------------------------------------------------------

class _CaptureHandler(logging.Handler):
    def __init__(self, state: CollectorState) -> None:
        super().__init__()
        self._state = state
        self.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._state.add_log(self.format(record))
        except Exception:
            pass


def _setup_logging(state: CollectorState, level: int) -> None:
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not isinstance(h, logging.StreamHandler)]
    root.addHandler(_CaptureHandler(state))
    root.setLevel(level)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Showdown battle replays as a daemon.")
    parser.add_argument("--format", default="gen3ou", help="Showdown format ID")
    parser.add_argument("--save-dir", default="replays", help="Directory for .log files")
    parser.add_argument("--local", action="store_true", help="Use localhost:8000")
    parser.add_argument("--max-concurrent", type=int, default=20, help="Max simultaneous rooms")
    parser.add_argument("--proxy", type=str, default=None, metavar="SOCKS5_URL",
                        help="SOCKS5 proxy URL, e.g. socks5h://127.0.0.1:1080")
    parser.add_argument("--verbose", action="store_true", help="Show debug-level logs in UI")
    args = parser.parse_args()

    server = LocalhostServerConfiguration if args.local else ShowdownServerConfiguration
    save_dir = Path(args.save_dir)
    state = CollectorState()

    spectator = BattleSpectator(
        server_configuration=server,
        max_concurrent=args.max_concurrent,
        proxy_url=args.proxy,
    )
    _setup_logging(state, logging.DEBUG if args.verbose else logging.INFO)

    future = asyncio.run_coroutine_threadsafe(
        _run(args.format, save_dir, spectator, state), POKE_LOOP
    )

    try:
        with Live(refresh_per_second=2, screen=True) as live:
            while True:
                live.update(render_dashboard(
                    spectator, state, args.format, save_dir, args.max_concurrent,
                    proxy_url=args.proxy,
                ))
                time.sleep(0.5)
    except KeyboardInterrupt:
        future.cancel()
        print(
            f"\nStopped. {state.total_collected} replays saved to {save_dir}/"
            + (f"  ({state.skipped} already existed)" if state.skipped else ""),
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
