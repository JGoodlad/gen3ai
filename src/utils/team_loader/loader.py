import os
import json
from typing import List, Optional, Tuple

from utils.paths import repo_path
from utils.team_corpora import check_label_encodable


class EmptyTeamPoolError(RuntimeError):
    """The team pool loaded NO team: its base directory is missing, or no manifest entry resolved to a file.

    A silent empty pool was a wrong input, not a refusal: every caller drew from it (a team-set id hashed over
    nothing, ``IndexError`` at the eval core's first draw, a 2026-10-17 anchors read failing every unit)."""


def default_base_dir() -> str:
    """``data/teams`` of the checkout the code was IMPORTED from (``utils.paths.repo_path``), never the CWD.

    A launcher-pinned child imports its pin's ``src/`` but runs with cwd = the launching (main) checkout, so a
    CWD-relative default read MAIN's teams in a pinned run (`e74c0610`, 2026-09-23); this default reads the pin's."""
    return str(repo_path("data", "teams"))


class TeamLoader:
    """
    A utility class to load Pokémon teams from the data/teams directory.
    Supports categorized loading for sample teams and community/dump teams.

    ``base_dir`` defaults to :func:`default_base_dir` (this checkout's ``data/teams``, CWD-independent). A pool
    that loads no team raises :class:`EmptyTeamPoolError`.

    🚨 **The LABEL-PATH GUARD** (the corpora's, ``c6497e66e``; TECH_DEBT (f) P2). The env's spread labels recompute
    every truth mon's stats at LEVEL 100 and FAULT otherwise, so a pool team with a ``Level:`` line other than 100
    (the gen3ou validator accepts one) would kill every run that draws it, as ``sfv2_5051`` did D1. Construction
    REFUSES such a pool with :class:`utils.team_corpora.CorpusError` (the SAME function, ``check_label_encodable``,
    the corpus readers run), naming each team's file. This is the pool's single entry: every team in ``sample_teams``
    / ``other_teams`` has passed it.
    """
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir if base_dir is not None else default_base_dir()
        self.sample_teams: List[str] = []
        self.other_teams: List[str] = []
        self._loaded: List[Tuple[str, str]] = []  # (file, text) of every loaded team, for the label-path guard
        self._load_teams()
        check_label_encodable([f for f, _ in self._loaded], [t for _, t in self._loaded],
                              f"the team pool {os.path.abspath(self.base_dir)}")
        if not self.sample_teams and not self.other_teams:
            state = "does not exist" if not os.path.isdir(self.base_dir) else "holds no loadable team"
            raise EmptyTeamPoolError(f"the team pool {os.path.abspath(self.base_dir)} {state}")

    def _load_teams(self):
        """Discovers all teams.json files and loads the corresponding team text."""
        if not os.path.isdir(self.base_dir):
            return

        # Defense-in-depth against a malformed manifest that references the same file more than
        # once (the per-Pokémon Yak Attack bug — fixed at the acquisition layer in
        # tools/others_team_downloader, but a manifest is just data and the next bad one should be
        # LOUD, not silently inflate one team's draw weight). Dedupe by resolved file path, keep
        # first occurrence; the acquisition-layer collapse is the real fix.
        seen = set()
        for root, dirs, files in os.walk(self.base_dir):
            if "teams.json" in files:
                json_path = os.path.join(root, "teams.json")
                try:
                    with open(json_path, "r") as f:
                        meta = json.load(f)
                except Exception as e:
                    print(f"Error loading {json_path}: {e}")
                    continue

                manifest_dups = 0
                for entry in meta:
                    # Skip invalid teams if the metadata flag is present
                    if entry.get("valid") is False:
                        continue

                    rel_file_path = entry.get("file")
                    if not rel_file_path:
                        continue

                    # Resolve path: entry['file'] is relative to the DATA dir, the base dir's
                    # parent ('teams/sample/abc.txt' -> '<base>/../teams/sample/abc.txt'). Never
                    # against the CWD: a CWD-relative fallback is how a pinned run read another
                    # checkout's file.
                    full_path = os.path.join(os.path.dirname(os.path.normpath(self.base_dir)),
                                             rel_file_path)

                    if not os.path.exists(full_path):
                        # Fallback for other potential path structures
                        full_path = os.path.join(self.base_dir, os.path.basename(rel_file_path))
                        if not os.path.exists(full_path) and os.path.isabs(rel_file_path):
                            full_path = rel_file_path

                    try:
                        if os.path.exists(full_path):
                            resolved = os.path.realpath(full_path)
                            if resolved in seen:
                                # Same file already loaded — a manifest referencing it twice
                                # would otherwise over-weight this team. Skip the duplicate.
                                manifest_dups += 1
                                continue
                            seen.add(resolved)

                            with open(full_path, "r") as f:
                                team_text = f.read().strip()

                            self._loaded.append((rel_file_path, team_text))

                            # Categorize by the folder structure UNDER the base (an absolute base's
                            # own path, e.g. a worktree named "resample-x", must not decide it)
                            if "sample" in os.path.relpath(root, self.base_dir).split(os.sep):
                                self.sample_teams.append(team_text)
                            else:
                                self.other_teams.append(team_text)
                        else:
                            print(f"Warning: Team file not found: {rel_file_path} (resolved to {full_path})")
                    except Exception as e:
                        print(f"Error reading {full_path}: {e}")

                if manifest_dups:
                    print(f"Warning: {json_path} references {manifest_dups} duplicate team "
                          f"file(s) (same file listed more than once) — skipped to avoid "
                          f"inflating those teams' draw weight. Re-run the acquisition tool "
                          f"(tools/others_team_downloader) to collapse the manifest.")

    def get_sample_teams(self):
        """Returns only the curated sample teams."""
        return self.sample_teams

    def get_other_teams(self):
        """Returns teams that are not in the sample category."""
        return self.other_teams

    def get_all_teams(self):
        """Returns all discovered teams."""
        return self.sample_teams + self.other_teams

    def __repr__(self):
        return f"<TeamLoader(samples={len(self.sample_teams)}, others={len(self.other_teams)})>"
