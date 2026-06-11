"""Guards for the team-manifest one-entry-per-team contract.

Two layers:
  * a DATA-CONTRACT test over the committed ``data/teams/**/teams.json`` files — every manifest
    must list each team file at most once (the regression guard for the per-Pokémon Yak Attack
    bug, where 174 real teams became 1056 rows → ~66% of all training/eval draws), and
  * unit tests for the acquisition-layer collapse helper that enforces it
    (``tools/others_team_downloader/sync.py::collapse_duplicate_teams``), loaded by path the same
    way ``agents/gen3_data/extractor_parity_test.py`` loads the other sync tool.
"""

import importlib.util
import json
import os
import glob

import pytest

from src.utils.git import get_repo_root


def _load_collapse_fn():
    repo = get_repo_root()
    path = os.path.join(repo, "tools", "others_team_downloader", "sync.py")
    spec = importlib.util.spec_from_file_location("oth_sync_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------------
# DATA-CONTRACT: the committed manifests must already satisfy one-entry-per-file.
# --------------------------------------------------------------------------------------------

def test_committed_manifests_are_one_entry_per_file():
    """Every data/teams/**/teams.json references each team file at most once.

    This is the regression guard for the per-Pokémon manifest format: a team listed once per
    Pokémon (six rows -> one file) silently multiplies that team's uniform draw weight 6-12x in
    both training and eval. Files are keyed by their resolved ``file`` field.
    """
    repo = get_repo_root()
    manifests = sorted(glob.glob(os.path.join(repo, "data", "teams", "**", "teams.json"),
                                 recursive=True))
    assert manifests, "no data/teams/**/teams.json manifests found"

    offenders = {}
    for manifest in manifests:
        with open(manifest) as f:
            rows = json.load(f)
        files = [r["file"] for r in rows if r.get("file")]
        dup_files = {f for f in files if files.count(f) > 1}
        if dup_files:
            offenders[os.path.relpath(manifest, repo)] = sorted(dup_files)

    assert not offenders, (
        "manifest(s) reference the same team file more than once (per-Pokémon format regression — "
        "collapse via tools/others_team_downloader): "
        + "; ".join(f"{m}: {len(fs)} dup file(s)" for m, fs in offenders.items())
    )


# --------------------------------------------------------------------------------------------
# UNIT: the collapse helper that the acquisition tool runs before writing a manifest.
# --------------------------------------------------------------------------------------------

def _per_mon_rows():
    """A synthetic Yak-Attack-style manifest: one team named once per Pokémon."""
    mons = ["Aerodactyl", "Cloyster", "Magneton", "Tyranitar", "Blissey", "Skarmory"]
    return [
        {
            "id": "deadbeef",
            "name": f"{mon}/Cloy Aero",
            "format": "gen3ou",
            "valid": True,
            "errors": [],
            "file": "teams/others/yak_attack/deadbeef.txt",
            "source": "https://pokepast.es/x",
        }
        for mon in mons
    ]


def test_collapse_per_mon_to_single_entry():
    mod = _load_collapse_fn()
    collapsed = mod.collapse_duplicate_teams(_per_mon_rows())
    assert len(collapsed) == 1
    entry = collapsed[0]
    assert entry["id"] == "deadbeef"
    assert entry["file"] == "teams/others/yak_attack/deadbeef.txt"
    # the "<Mon>/" prefix is stripped to the real team name
    assert entry["name"] == "Cloy Aero"
    assert entry["valid"] is True


def test_collapse_valid_is_conjunction_and_errors_union():
    mod = _load_collapse_fn()
    rows = _per_mon_rows()
    # mark one of the six rows invalid with its own error
    rows[2]["valid"] = False
    rows[2]["errors"] = ["Magneton is banned."]
    rows[4]["valid"] = False
    rows[4]["errors"] = ["Blissey clause.", "Magneton is banned."]  # overlapping error
    collapsed = mod.collapse_duplicate_teams(rows)
    assert len(collapsed) == 1
    entry = collapsed[0]
    # one invalid row -> the whole team is invalid (conservative AND)
    assert entry["valid"] is False
    # errors are the order-preserving union (no duplicates)
    assert entry["errors"] == ["Magneton is banned.", "Blissey clause."]


def test_collapse_is_order_preserving_and_idempotent():
    mod = _load_collapse_fn()
    rows = []
    for i in range(3):
        rows += [
            {"id": f"team{i}", "name": f"Mon{j}/Name {i}", "format": "gen3ou",
             "valid": True, "errors": [], "file": f"teams/others/x/team{i}.txt",
             "source": "u"}
            for j in range(6)
        ]
    collapsed = mod.collapse_duplicate_teams(rows)
    assert [c["id"] for c in collapsed] == ["team0", "team1", "team2"]
    # collapsing an already-collapsed manifest is a no-op
    assert mod.collapse_duplicate_teams(collapsed) == collapsed


def test_collapse_keeps_single_occurrence_name_verbatim():
    """A team that appears once is NOT a per-mon duplicate — its name (even with a '/') is kept."""
    mod = _load_collapse_fn()
    rows = [{
        "id": "solo", "name": "Skarm/Bliss Core", "format": "gen3ou",
        "valid": True, "errors": [], "file": "teams/others/x/solo.txt", "source": "u",
    }]
    collapsed = mod.collapse_duplicate_teams(rows)
    assert len(collapsed) == 1
    assert collapsed[0]["name"] == "Skarm/Bliss Core"  # not stripped — single occurrence
