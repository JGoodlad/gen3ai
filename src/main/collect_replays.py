"""
Daemon script that spectates Pokémon Showdown battles and saves raw replay logs.

Runs until Ctrl+C. Each battle is saved immediately when it finishes.
Restarts are safe — already-saved files are not overwritten.

Usage:
    python src/main/collect_replays.py --format gen3ou --save-dir replays/gen3ou
    python src/main/collect_replays.py --format gen3ou --save-dir replays/gen3ou --local

The script runs the spectator on poke-env's background POKE_LOOP rather than
asyncio.run(), because PSClient lives on that loop and all callbacks execute there.
"""

import argparse
import asyncio
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
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
# Shared state between the POKE_LOOP async task and the main-thread display
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
            if len(self.recent) > 15:
                self.recent = self.recent[-15:]

    def snapshot_recent(self) -> List[_CompletedEntry]:
        with self._lock:
            return list(reversed(self.recent))


# ---------------------------------------------------------------------------
# Async worker (runs on POKE_LOOP)
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
# Rich dashboard renderer
# ---------------------------------------------------------------------------

def _elapsed_str(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _room_num(battle_tag: str) -> str:
    """Extract the numeric suffix from a battle tag for compact display."""
    return battle_tag.rsplit("-", 1)[-1]


def render_dashboard(
    spectator: BattleSpectator,
    state: CollectorState,
    format_id: str,
    save_dir: Path,
    max_concurrent: int,
) -> object:
    elapsed = time.time() - state.start_time
    active = spectator.active_battles  # benign snapshot — display only

    # ── Stats row ────────────────────────────────────────────────────────
    stats = Table.grid(padding=(0, 3), expand=True)
    for _ in range(5):
        stats.add_column(justify="center")
    stats.add_row(
        f"[bold green]{state.total_collected}[/bold green]\n[dim]collected[/dim]",
        f"[bold cyan]{len(active)}/{max_concurrent}[/bold cyan]\n[dim]watching[/dim]",
        f"[bold yellow]{spectator.pending_count}[/bold yellow]\n[dim]queued[/dim]",
        f"[bold white]{spectator.seen_count}[/bold white]\n[dim]seen total[/dim]",
        f"[dim]{_elapsed_str(elapsed)}[/dim]\n[dim]elapsed[/dim]",
    )

    # ── Active battles ───────────────────────────────────────────────────
    active_tbl = Table(
        "Room #", "Turn", "Players",
        box=rich.box.SIMPLE_HEAD,
        header_style="dim",
        show_edge=False,
        expand=True,
        padding=(0, 1),
    )
    if active:
        now = time.time()
        for tag, battle in sorted(active.items()):
            num = _room_num(tag)
            p = battle.players
            if p:
                players_str = f"{p.get('p1', '?')} vs {p.get('p2', '?')}"
            else:
                players_str = "[dim]loading…[/dim]"
            age = now - battle.joined_at
            turn_str = f"[bold]{battle.turn}[/bold]" if battle.turn > 0 else "[dim]--[/dim]"
            age_str = f"[dim]{_elapsed_str(age)}[/dim]"
            active_tbl.add_row(f"[cyan]#{num}[/cyan]  {age_str}", turn_str, players_str)
    else:
        active_tbl.add_row("[dim]—[/dim]", "[dim]—[/dim]", "[dim]connecting…[/dim]")

    # ── Recent completions ───────────────────────────────────────────────
    recent_tbl = Table(
        "#", "Room #", "Winner", "Turns",
        box=rich.box.SIMPLE_HEAD,
        header_style="dim",
        show_edge=False,
        expand=True,
        padding=(0, 1),
    )
    for entry in state.snapshot_recent()[:12]:
        num = _room_num(entry.battle_tag)
        winner_str = entry.winner if entry.winner else "[dim italic]tie[/dim italic]"
        recent_tbl.add_row(
            f"[dim]{entry.index}[/dim]",
            f"[cyan]#{num}[/cyan]",
            winner_str,
            f"[dim]{entry.turn}[/dim]",
        )

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
        ),
        title=title,
        border_style="blue",
        padding=(0, 1),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Showdown battle replays as a daemon."
    )
    parser.add_argument("--format", default="gen3ou", help="Showdown format ID")
    parser.add_argument("--save-dir", default="replays", help="Directory for .log files")
    parser.add_argument("--local", action="store_true", help="Use localhost:8000")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max simultaneous rooms")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    server = LocalhostServerConfiguration if args.local else ShowdownServerConfiguration
    save_dir = Path(args.save_dir)

    spectator = BattleSpectator(
        server_configuration=server,
        max_concurrent=args.max_concurrent,
    )
    state = CollectorState()

    future = asyncio.run_coroutine_threadsafe(
        _run(args.format, save_dir, spectator, state), POKE_LOOP
    )

    try:
        with Live(refresh_per_second=2, screen=True) as live:
            while True:
                live.update(render_dashboard(spectator, state, args.format, save_dir, args.max_concurrent))
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
