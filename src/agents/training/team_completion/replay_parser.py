"""
Parse Gen 3 OU battle logs into TeamRecord objects.

Each .log file contains raw Showdown protocol lines. Since Gen 3 has no team
preview, we only know a Pokémon once it switches in. Items and moves are
accumulated from battle events as they're revealed.

Output: JSONL where each line is a serialized TeamRecord (one per player per battle).
"""

import json
import re
import glob
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


def _normalize(name: str) -> str:
    """Lowercase + strip all non-alphanumeric characters to match JSON map keys."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _parse_player(identifier: str) -> str:
    """Extract 'p1' or 'p2' from identifiers like 'p1a: Nickname'."""
    return identifier[:2]


def _parse_nickname(identifier: str) -> str:
    """Extract nickname from identifiers like 'p1a: Nickname'."""
    return identifier.split(": ", 1)[1]


def _parse_species(details: str) -> str:
    """
    Extract species key from a details string like 'Zapdos', 'Blissey, F',
    'Metagross, shiny', 'Nidoran-F', etc. Returns the normalized key.
    """
    raw = details.split(",")[0].strip()
    return _normalize(raw)


@dataclass
class PokemonRecord:
    species: str                    # normalized key, e.g. "zapdos"
    moves: list[str] = field(default_factory=list)   # normalized move keys, revealed only
    item: Optional[str] = None      # normalized item key, or None if never revealed


@dataclass
class TeamRecord:
    battle_id: str
    player: str                     # "p1" or "p2"
    winner: bool
    pokemon: list[PokemonRecord]    # 1–6 entries, sorted by species key


def _resolve_mon(
    identifier: str,
    nickname_to_species: dict[str, dict[str, str]],
    teams: dict[str, dict[str, PokemonRecord]],
) -> tuple[Optional[str], Optional[PokemonRecord]]:
    """
    Parse player + nickname from an identifier like 'p1a: Nickname', look up
    the species, and return (player, record) — or (None, None) if unresolvable.
    """
    player = identifier[:2]
    if player not in ("p1", "p2"):
        return None, None
    nick = identifier.split(": ", 1)[1]
    species = nickname_to_species[player].get(nick)
    if not species or species not in teams[player]:
        return None, None
    return player, teams[player][species]


def _set_item(record: PokemonRecord, item_name: str) -> None:
    """Normalize item_name and assign to record if non-empty."""
    key = _normalize(item_name)
    if key:
        record.item = key


def _sort_team(team: dict[str, PokemonRecord]) -> list[PokemonRecord]:
    """Return team as a list sorted by species key for permutation invariance."""
    mons = list(team.values())
    mons.sort(key=lambda m: m.species)
    for mon in mons:
        mon.moves = sorted(set(mon.moves))  # dedup + sort for move invariance
    return mons


def parse_replay(path: str) -> Optional[tuple[TeamRecord, TeamRecord]]:
    """
    Parse a single .log file and return (p1_record, p2_record), or None if the
    battle is malformed (no winner, no Pokémon revealed, etc.).
    """
    # Per-player state
    # nickname_to_species[player] = {nickname: species_key}
    nickname_to_species: dict[str, dict[str, str]] = {"p1": {}, "p2": {}}
    # teams[player] = {species_key: PokemonRecord}
    teams: dict[str, dict[str, PokemonRecord]] = {"p1": {}, "p2": {}}
    # active nickname per player
    active: dict[str, Optional[str]] = {"p1": None, "p2": None}
    winner: Optional[str] = None
    battle_id = Path(path).stem

    with open(path, encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.startswith("|"):
                continue
            parts = line.split("|")
            # parts[0] is always "" (line starts with |)
            if len(parts) < 2:
                continue
            event = parts[1]

            if event in ("switch", "drag"):
                # |switch|p1a: Nickname|Species, details|HP
                if len(parts) < 4:
                    continue
                player = parts[2][:2]
                if player not in ("p1", "p2"):
                    continue
                nick = parts[2].split(": ", 1)[1]
                species = _parse_species(parts[3])
                if not species:
                    continue
                nickname_to_species[player][nick] = species
                active[player] = nick
                if species not in teams[player]:
                    teams[player][species] = PokemonRecord(species=species)

            elif event == "move":
                # |move|p1a: Nickname|MoveName|target|[flags...]
                if len(parts) < 4:
                    continue
                _, record = _resolve_mon(parts[2], nickname_to_species, teams)
                if record is not None:
                    move_key = _normalize(parts[3])
                    if move_key and move_key not in record.moves:
                        record.moves.append(move_key)

            elif event in ("-item", "-enditem"):
                # |-item|p1a: Nickname|ItemName|[flags...]
                # |-enditem|p1a: Nickname|ItemName|[flags...]
                if len(parts) < 4:
                    continue
                _, record = _resolve_mon(parts[2], nickname_to_species, teams)
                if record is not None:
                    _set_item(record, parts[3])

            elif event in ("-heal", "-boost"):
                # |-heal|p1a: Nickname|HP|[from] item: ItemName
                # |-boost|p1a: Nickname|stat|n|[from] item: ItemName
                if len(parts) < 3:
                    continue
                from_item = next(
                    (p[len("[from] item:"):].strip() for p in parts[3:] if p.startswith("[from] item:")),
                    None,
                )
                if not from_item:
                    continue
                _, record = _resolve_mon(parts[2], nickname_to_species, teams)
                if record is not None:
                    _set_item(record, from_item)

            elif event == "win":
                # |win|PlayerName — map player name back to p1/p2 using |player| lines
                # We handle this below after reading the full file; for now just store raw name
                winner_name = parts[2] if len(parts) > 2 else None
                winner = ("_name_", winner_name)

    # We need to resolve winner name → p1/p2, so do a second lightweight pass
    player_names: dict[str, str] = {}  # "p1" -> username
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.startswith("|player|"):
                continue
            parts = line.split("|")
            # |player|p1|Username|avatar|rating
            if len(parts) >= 4:
                player_names[parts[2]] = parts[3]

    winner_player: Optional[str] = None
    if isinstance(winner, tuple) and winner[0] == "_name_" and winner[1]:
        for p, name in player_names.items():
            if name == winner[1]:
                winner_player = p
                break

    if not any(teams.values()):
        return None

    records = []
    for player in ("p1", "p2"):
        if not teams[player]:
            continue
        sorted_team = _sort_team(teams[player])
        records.append(TeamRecord(
            battle_id=battle_id,
            player=player,
            winner=(winner_player == player),
            pokemon=sorted_team,
        ))

    if len(records) == 2:
        return records[0], records[1]
    return None


def parse_all_replays(replay_dir: str, output_path: str, verbose: bool = True) -> dict:
    """
    Parse all .log files in replay_dir and write TeamRecords to output_path as JSONL.
    Returns summary stats.
    """
    paths = sorted(glob.glob(os.path.join(replay_dir, "*.log")))
    if not paths:
        raise FileNotFoundError(f"No .log files found in {replay_dir}")

    total_battles = 0
    total_teams = 0
    skipped = 0
    mon_counts: list[int] = []
    move_counts: list[int] = []
    item_revealed = 0
    total_mons = 0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for path in paths:
            try:
                result = parse_replay(path)
            except Exception:
                skipped += 1
                continue

            if result is None:
                skipped += 1
                continue

            total_battles += 1
            for record in result:
                total_teams += 1
                mon_counts.append(len(record.pokemon))
                for mon in record.pokemon:
                    total_mons += 1
                    move_counts.append(len(mon.moves))
                    if mon.item is not None:
                        item_revealed += 1
                out.write(json.dumps(asdict(record)) + "\n")

    stats = {
        "battles_parsed": total_battles,
        "battles_skipped": skipped,
        "team_records": total_teams,
        "avg_pokemon_per_team": round(sum(mon_counts) / len(mon_counts), 2) if mon_counts else 0,
        "avg_moves_per_pokemon": round(sum(move_counts) / len(move_counts), 2) if move_counts else 0,
        "pct_pokemon_with_item": round(100 * item_revealed / total_mons, 1) if total_mons else 0,
    }

    if verbose:
        print(f"Parsed {total_battles} battles ({skipped} skipped)")
        print(f"Team records written: {total_teams}")
        print(f"Avg Pokémon revealed per team: {stats['avg_pokemon_per_team']}")
        print(f"Avg moves known per Pokémon:   {stats['avg_moves_per_pokemon']}")
        print(f"Pokémon with item revealed:     {stats['pct_pokemon_with_item']}%")
        print(f"Output: {output_path}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse Gen 3 OU replays into team JSONL")
    parser.add_argument("--replay-dir", default="replays/showdown/1", help="Directory of .log files")
    parser.add_argument("--output", default="data/replay_teams.jsonl", help="Output JSONL path")
    parser.add_argument("--sample", type=int, default=0, help="Parse only N replays (0 = all)")
    args = parser.parse_args()

    replay_dir = args.replay_dir
    if args.sample > 0:
        import tempfile, shutil, random
        all_logs = glob.glob(os.path.join(replay_dir, "*.log"))
        sample = random.sample(all_logs, min(args.sample, len(all_logs)))
        tmpdir = tempfile.mkdtemp()
        for p in sample:
            shutil.copy(p, tmpdir)
        replay_dir = tmpdir

    parse_all_replays(replay_dir, args.output)
