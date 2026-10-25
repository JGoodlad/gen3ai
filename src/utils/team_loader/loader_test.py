import pytest
import os
import json
import glob
import tempfile
import shutil
from src.utils.team_loader.loader import TeamLoader
from src.utils.git import get_repo_root

@pytest.fixture
def mock_teams_structure():
    """
    Creates a temporary directory structure that mimics the data/teams folder.
    """
    temp_dir = tempfile.mkdtemp()
    
    # We need to mimic the 'data/teams' structure because the loader 
    # uses 'data' as a hardcoded prefix for file resolution.
    data_dir = os.path.join(temp_dir, "data")
    teams_dir = os.path.join(data_dir, "teams")
    os.makedirs(teams_dir)
    
    # Sample teams
    sample_dir = os.path.join(teams_dir, "sample")
    os.makedirs(sample_dir)
    with open(os.path.join(sample_dir, "team_s1.txt"), "w") as f:
        f.write("Sample Team 1 Content")
    with open(os.path.join(sample_dir, "teams.json"), "w") as f:
        json.dump([{"file": "teams/sample/team_s1.txt"}], f)
        
    # Other teams
    others_dir = os.path.join(teams_dir, "others", "johnnyg2")
    os.makedirs(others_dir)
    with open(os.path.join(others_dir, "team_o1.txt"), "w") as f:
        f.write("Other Team 1 Content")
    with open(os.path.join(others_dir, "teams.json"), "w") as f:
        json.dump([{"file": "teams/others/johnnyg2/team_o1.txt"}], f)
        
    # Store the original cwd to restore later
    old_cwd = os.getcwd()
    os.chdir(temp_dir)
    
    yield teams_dir
    
    os.chdir(old_cwd)
    shutil.rmtree(temp_dir)

def test_loader_finds_all_teams(mock_teams_structure):
    loader = TeamLoader(base_dir=mock_teams_structure)
    
    assert len(loader.get_sample_teams()) == 1
    assert len(loader.get_other_teams()) == 1
    assert len(loader.get_all_teams()) == 2
    
    assert "Sample Team 1 Content" in loader.get_sample_teams()
    assert "Other Team 1 Content" in loader.get_other_teams()

def test_a_missing_or_empty_pool_is_REFUSED_never_an_empty_list(tmp_path):
    """An empty pool is a wrong input (GIGO), not a result: every caller drew from it (2026-10-17 anchors read)."""
    from src.utils.team_loader.loader import EmptyTeamPoolError
    with pytest.raises(EmptyTeamPoolError, match="does not exist"):
        TeamLoader(base_dir=str(tmp_path / "no" / "such" / "teams"))
    teams = tmp_path / "data" / "teams" / "sample"
    teams.mkdir(parents=True)
    (teams / "teams.json").write_text(json.dumps([{"file": "teams/sample/ghost.txt"}]))
    with pytest.raises(EmptyTeamPoolError, match="holds no loadable team"):
        TeamLoader(base_dir=str(tmp_path / "data" / "teams"))


def test_the_default_pool_is_the_IMPORTED_checkouts_from_any_cwd(tmp_path, monkeypatch):
    """The default base is ``repo_path("data", "teams")``, never the CWD: a launcher-pinned child stands in MAIN's
    checkout and must read its pin's pool (e74c0610). From a directory holding ANOTHER ``data/teams`` the default
    still loads this checkout's pool, byte for byte."""
    from src.utils.team_loader.loader import default_base_dir
    from utils.paths import repo_path
    want = TeamLoader(base_dir=str(repo_path("data", "teams"))).get_all_teams()
    decoy = tmp_path / "data" / "teams" / "sample"
    decoy.mkdir(parents=True)
    (decoy / "d.txt").write_text("DECOY TEAM")
    (decoy / "teams.json").write_text(json.dumps([{"file": "teams/sample/d.txt"}]))
    monkeypatch.chdir(tmp_path)
    assert default_base_dir() == str(repo_path("data", "teams"))
    got = TeamLoader().get_all_teams()
    assert got == want and "DECOY TEAM" not in got


def test_sample_is_decided_UNDER_the_base_not_by_its_absolute_path(tmp_path):
    """An absolute base whose own path says "sample" (a worktree named ``resample-x``) must not class every team as a
    sample team; only a ``sample`` directory under the base does."""
    base = tmp_path / "resample-x" / "data" / "teams"
    other = base / "others" / "a"
    other.mkdir(parents=True)
    (other / "o.txt").write_text("OTHER TEAM")
    (other / "teams.json").write_text(json.dumps([{"file": "teams/others/a/o.txt"}]))
    loader = TeamLoader(base_dir=str(base))
    assert loader.get_other_teams() == ["OTHER TEAM"] and loader.get_sample_teams() == []

def _pool_with(tmp_path, texts):
    """A tiny pool whose files hold ``texts`` (name -> paste)."""
    base = tmp_path / "data" / "teams"
    d = base / "others" / "a"
    d.mkdir(parents=True)
    for name, text in texts.items():
        (d / name).write_text(text)
    (d / "teams.json").write_text(json.dumps([{"file": f"teams/others/a/{n}"} for n in texts]))
    return str(base)


def test_a_pool_team_the_label_path_cannot_encode_is_REFUSED_naming_its_file(tmp_path):
    """The corpora's guard (``c6497e66e``), one reader over: a non-100 ``Level:`` in the TRAINING pool would fault the
    env's L100 spread labels (D1 s3501, ``sfv2_5051``). Fails on revert of the guard in ``TeamLoader.__init__``."""
    from utils.team_corpora import CorpusError
    ok = "Mew @ Leftovers\nLevel: 100\n- Psychic"
    bad = "Krabby @ Salac Berry\nLevel: 5\n- Endure"
    base = _pool_with(tmp_path, {"ok.txt": ok, "bad.txt": bad})
    with pytest.raises(CorpusError, match=r"bad\.txt.*Level: 5"):
        TeamLoader(base_dir=base)
    # the same pool without the violator loads, and a team with no Level line is Showdown's 100
    clean = _pool_with(tmp_path / "c", {"ok.txt": ok, "nolevel.txt": "Mew\n- Psychic"})
    assert len(TeamLoader(base_dir=clean).get_all_teams()) == 2


def test_the_committed_pool_holds_no_non_100_team():
    """The CURRENT ``data/teams`` passes the guard (checked 2026-10-25: 719 teams, 0 violators)."""
    from utils.team_corpora import label_path_defects
    teams = TeamLoader().get_all_teams()
    assert teams and not [t for t in teams if label_path_defects(t)]


def test_loader_handles_invalid_json(mock_teams_structure):
    # Add an invalid JSON file
    invalid_json_dir = os.path.join(mock_teams_structure, "invalid")
    os.makedirs(invalid_json_dir)
    with open(os.path.join(invalid_json_dir, "teams.json"), "w") as f:
        f.write("not a json")
        
    # Should not crash
    loader = TeamLoader(base_dir=mock_teams_structure)
    # Should still find the other valid teams
    assert len(loader.get_all_teams()) >= 2

def test_loader_handles_missing_team_files(mock_teams_structure):
    # Add a JSON entry pointing to a missing file
    missing_file_dir = os.path.join(mock_teams_structure, "missing")
    os.makedirs(missing_file_dir)
    with open(os.path.join(missing_file_dir, "teams.json"), "w") as f:
        json.dump([{"file": "teams/missing/ghost.txt"}], f)

    # Should not crash, just skip the missing file
    loader = TeamLoader(base_dir=mock_teams_structure)
    # ghost.txt doesn't exist, so it shouldn't be added to any list
    for team in loader.get_all_teams():
        assert "ghost" not in team


def test_loader_dedupes_per_mon_manifest(mock_teams_structure):
    """Defense-in-depth: even a malformed per-Pokémon manifest (one file listed once per mon)
    yields a single loaded team, not one per row — so a future manifest bug can't inflate a
    team's draw weight."""
    dump_dir = os.path.join(mock_teams_structure, "others", "yak_attack")
    os.makedirs(dump_dir)
    with open(os.path.join(dump_dir, "shared.txt"), "w") as f:
        f.write("Shared Team Content")
    # six rows, all pointing at the SAME file (the per-Pokémon format)
    per_mon = [
        {"id": "shared", "name": f"{mon}/Cool Team", "valid": True,
         "file": "teams/others/yak_attack/shared.txt"}
        for mon in ["Aero", "Cloy", "Mag", "Tar", "Bliss", "Skarm"]
    ]
    with open(os.path.join(dump_dir, "teams.json"), "w") as f:
        json.dump(per_mon, f)

    loader = TeamLoader(base_dir=mock_teams_structure)
    # the shared file is loaded exactly once despite six manifest rows
    assert loader.get_other_teams().count("Shared Team Content") == 1
    # and there are no duplicate team strings anywhere in the pool
    all_teams = loader.get_all_teams()
    assert len(all_teams) == len(set(all_teams))


def test_loader_real_counts_pin():
    """Pin the real committed pool against the per-mon-collapse fix.

    Derived (not hand-counted): each manifest contributes its distinct valid+present files.
    After collapsing the Yak Attack manifest to one-entry-per-team this was samples=32,
    others=687, all=719 (was others=1569/all=1601 with the per-mon inflation — Yak Attack alone
    was 1056 of the 1569 'others', i.e. ~66% of all draws). The 2026-08-31 40-team promotion
    (`python -m main.promote_teams`, seed 1383414976) MOVED 40 teams from the others manifests into
    `data/teams/sample/`, so the pins are now samples=72, others=647, all=719 — the total is
    invariant under a promotion, which is what the third assert protects. If the validity policy
    ever shifts these by a few, update the pins below to match the derived number it prints.
    """
    repo = get_repo_root()
    old_cwd = os.getcwd()
    os.chdir(repo)
    try:
        # Independently derive the expected counts straight from the manifests, deduped by file.
        seen, exp_sample, exp_other = set(), 0, 0
        for manifest in sorted(glob.glob("data/teams/**/teams.json", recursive=True)):
            root = os.path.dirname(manifest)
            for entry in json.load(open(manifest)):
                if entry.get("valid") is False:
                    continue
                rel = entry.get("file")
                if not rel:
                    continue
                full = os.path.join("data", rel)
                if not os.path.exists(full):
                    continue
                key = os.path.realpath(full)
                if key in seen:
                    continue
                seen.add(key)
                if "sample" in root:
                    exp_sample += 1
                else:
                    exp_other += 1

        loader = TeamLoader()
        assert len(loader.get_sample_teams()) == exp_sample
        assert len(loader.get_other_teams()) == exp_other
        assert len(loader.get_all_teams()) == exp_sample + exp_other

        # Documented absolute pins (the post-fix distribution). Derived == documented.
        assert exp_sample == 72, f"sample teams = {exp_sample}, expected 72"
        assert exp_other == 647, f"other teams = {exp_other}, expected 647"
        assert exp_sample + exp_other == 719
    finally:
        os.chdir(old_cwd)
