import os
import json

#: The ROLE of a ``teams.json`` manifest is named by the FIRST directory under ``data/teams/`` that
#: holds it — never by a substring test (the old rule was ``"sample" in root``, which any future
#: folder with "sample" anywhere in its path would have silently joined).
#:
#: * ``sample``     — EXACTLY the teams linked from the first post of Smogon's ADV OU sample-teams
#:                    thread, written ONLY by ``tools/sample_team_downloader`` (the curated set).
#: * ``promoted``   — pool teams promoted by ``python -m main.promote_teams`` so they can be legal
#:                    ``--exploiter`` trainees (the 2026-08-31 40-team fleet draw). NOT curated.
#: * ``superseded`` — a former Smogon sample paste that the thread has since REPLACED with a newer
#:                    paste of the same team. Kept on disk because archived runs recorded its path
#:                    and fingerprint; never drawn (not in ``get_all_teams()``).
#: * anything else  — ``other``: bulk-downloaded / hand-added pool teams.
ROLE_SAMPLE = "sample"
ROLE_PROMOTED = "promoted"
ROLE_SUPERSEDED = "superseded"
ROLE_OTHER = "other"
_NAMED_ROLES = (ROLE_SAMPLE, ROLE_PROMOTED, ROLE_SUPERSEDED)


def manifest_role(manifest_dir, base_dir="data/teams"):
    """The role of the ``teams.json`` in ``manifest_dir`` — see the table above. Shared with
    ``main.promote_teams.load_pool`` so the promotion tool's mirror of this loader cannot drift."""
    rel = os.path.relpath(os.path.abspath(manifest_dir), os.path.abspath(base_dir))
    top = rel.replace(os.sep, "/").split("/")[0]
    return top if top in _NAMED_ROLES else ROLE_OTHER


class TeamLoader:
    """
    A utility class to load Pokémon teams from the data/teams directory.

    Every team is listed in exactly one ``teams.json`` manifest, and the manifest's folder names its
    ROLE (:func:`manifest_role`). The pool (``get_all_teams()``) is ``sample + promoted + other``,
    in that order — the order the pre-split loader produced, so an index into the pool is stable.
    ``superseded`` teams are loaded but are NOT in the pool.
    """
    def __init__(self, base_dir="data/teams"):
        self.base_dir = base_dir
        self.sample_teams = []
        self.promoted_teams = []
        self.superseded_teams = []
        self.other_teams = []
        self._load_teams()

    def _load_teams(self):
        """Discovers all teams.json files and loads the corresponding team text."""
        if not os.path.exists(self.base_dir):
            print(f"Warning: Base directory {self.base_dir} does not exist.")
            return

        by_role = {ROLE_SAMPLE: self.sample_teams, ROLE_PROMOTED: self.promoted_teams,
                   ROLE_SUPERSEDED: self.superseded_teams, ROLE_OTHER: self.other_teams}

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

                target = by_role[manifest_role(root, self.base_dir)]
                manifest_dups = 0
                for entry in meta:
                    # Skip invalid teams if the metadata flag is present
                    if entry.get("valid") is False:
                        continue

                    rel_file_path = entry.get("file")
                    if not rel_file_path:
                        continue

                    # Resolve path: entry['file'] is relative to 'data/'
                    # e.g., 'teams/sample/abc.txt' -> 'data/teams/sample/abc.txt'
                    full_path = os.path.join("data", rel_file_path)

                    if not os.path.exists(full_path):
                        # Fallback for other potential path structures
                        full_path = os.path.join(self.base_dir, os.path.basename(rel_file_path))
                        if not os.path.exists(full_path):
                            # Try absolute or direct relative
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
                            target.append(team_text)
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
        """The CURATED set: exactly Smogon's ADV OU sample teams (``data/teams/sample/``)."""
        return self.sample_teams

    def get_training_bias_teams(self):
        """The teams the DEFAULT trainee builder over-draws (``bias_prob``, 10%): the curated
        Smogon set. Named separately from :meth:`get_sample_teams` so the role is explicit at the
        call site; since 2026-09-23 (``gen3_curated_sample_split_v1``) it is the 32 curated teams,
        where the pre-split loader handed it all 72 (curated + promoted)."""
        return self.sample_teams

    def get_promoted_teams(self):
        """Pool teams promoted by ``python -m main.promote_teams`` (``data/teams/promoted/``)."""
        return self.promoted_teams

    def get_superseded_teams(self):
        """Former Smogon sample pastes the thread has replaced (``data/teams/superseded/``). NOT in
        the pool — kept because archived runs pinned them."""
        return self.superseded_teams

    def get_exploiter_trainee_teams(self):
        """Every team an ``--exploiter`` run may pilot: curated + promoted + superseded. The promoted
        teams exist precisely to be legal trainees; superseded pastes stay legal so every archived
        exploiter argv that pinned one still validates."""
        return self.sample_teams + self.promoted_teams + self.superseded_teams

    def get_other_teams(self):
        """Pool teams that are neither curated nor promoted (bulk-downloaded / hand-added)."""
        return self.other_teams

    def get_all_teams(self):
        """The POOL: curated + promoted + other, in that (pre-split) order. Excludes superseded."""
        return self.sample_teams + self.promoted_teams + self.other_teams

    def __repr__(self):
        return (f"<TeamLoader(samples={len(self.sample_teams)}, "
                f"promoted={len(self.promoted_teams)}, others={len(self.other_teams)}, "
                f"superseded={len(self.superseded_teams)})>")
