"""The team ROLES (``gen3_curated_sample_split_v1``, 2026-09-23) — curated / promoted / superseded /
other — the relocation map that keeps archived team paths working, and the sha-pinned lists that keep
goldens and measurements byte-identical across the split.

Synthetic trees for the rules; the committed tree for the facts. The archive section reads
``models/`` and SKIPS when there is none.
"""
import glob
import hashlib
import json
import os

import pytest

from utils.paths import repo_path, repo_root
from utils.team_loader import TeamLoader, manifest_role
from utils.team_loader import pins
from utils.team_loader.relocations import relocation_map, resolve_team_file

OLD_CRL, NEW_CRL = "45995e432f", "1808014a9a"   # the Curse RestLax paste the thread replaced


def _sha(t):
    return hashlib.sha1(t.strip().encode()).hexdigest()[:10]


@pytest.fixture
def in_repo():
    prev = os.getcwd()
    os.chdir(repo_root())
    try:
        yield
    finally:
        os.chdir(prev)


# ── the role rule (synthetic) ───────────────────────────────────────────────────────────────────

def _manifest(teams_dir, sub, names):
    d = os.path.join(teams_dir, *sub.split("/"))
    os.makedirs(d, exist_ok=True)
    rows = []
    for n in names:
        with open(os.path.join(d, f"{n}.txt"), "w") as fh:
            fh.write(f"team {n}")
        rows.append({"file": f"teams/{sub}/{n}.txt"})
    with open(os.path.join(d, "teams.json"), "w") as fh:
        json.dump(rows, fh)


def test_roles_come_from_the_top_folder_and_the_pool_order_is_sample_promoted_other(tmp_path):
    teams = tmp_path / "data" / "teams"
    _manifest(str(teams), "sample", ["s1", "s2"])
    _manifest(str(teams), "promoted", ["p1"])
    _manifest(str(teams), "superseded", ["x1"])
    _manifest(str(teams), "others/author", ["o1"])
    # the OLD rule was `"sample" in root`: a dump folder merely NAMED like it joined the curated set
    _manifest(str(teams), "others/sample_dump", ["o2"])
    prev = os.getcwd()
    os.chdir(tmp_path)
    try:
        L = TeamLoader(base_dir=str(teams))
    finally:
        os.chdir(prev)
    assert L.get_sample_teams() == ["team s1", "team s2"]
    assert L.get_training_bias_teams() == ["team s1", "team s2"]
    assert L.get_promoted_teams() == ["team p1"]
    assert L.get_superseded_teams() == ["team x1"]
    assert sorted(L.get_other_teams()) == ["team o1", "team o2"]
    assert L.get_all_teams()[:3] == ["team s1", "team s2", "team p1"]
    assert "team x1" not in L.get_all_teams()                        # superseded is never drawn
    assert L.get_exploiter_trainee_teams() == ["team s1", "team s2", "team p1", "team x1"]
    assert manifest_role(str(teams / "others" / "sample_dump"), str(teams)) == "other"


# ── the committed tree ──────────────────────────────────────────────────────────────────────────

def test_the_committed_roles_and_counts(in_repo):
    L = TeamLoader()
    assert (len(L.get_sample_teams()), len(L.get_promoted_teams()), len(L.get_other_teams()),
            len(L.get_superseded_teams()), len(L.get_all_teams())) == (32, 40, 647, 1, 719)
    assert [_sha(t) for t in L.get_superseded_teams()] == [OLD_CRL]
    assert NEW_CRL in {_sha(t) for t in L.get_sample_teams()}
    assert len(L.get_exploiter_trainee_teams()) == 73
    # no content duplicates across roles (a promoted copy of a curated team would double-draw it)
    shas = [_sha(t) for t in L.get_all_teams() + L.get_superseded_teams()]
    assert len(shas) == len(set(shas))


def test_the_pool_is_the_pre_split_pool_up_to_the_one_declared_swap(in_repo):
    """THE SPLIT'S PROOF, standing. The pre-split pool was ``sample(72) + other``; today's is
    ``sample(32) + promoted(40) + other``. The first 72 entries must equal the pinned pre-split list
    except at index 17, where the thread's new Curse RestLax paste replaced the old one — the ONLY
    pool change the split made. Fails if the move reorders, drops or duplicates anything."""
    allt = [_sha(t) for t in TeamLoader().get_all_teams()]
    assert len(allt) == 719
    head = list(pins.PRE_SPLIT_SAMPLE_72)
    diff = [i for i, (a, b) in enumerate(zip(allt[:72], head)) if a != b]
    assert diff == [17] and head[17] == OLD_CRL and allt[17] == NEW_CRL
    # and the curated / promoted halves are exactly the pinned list's two halves
    L = TeamLoader()
    assert [_sha(t) for t in L.get_promoted_teams()] == head[32:]
    assert [_sha(t) for t in L.get_sample_teams()] == head[:17] + [NEW_CRL] + head[18:32]


# ── the pinned lists ────────────────────────────────────────────────────────────────────────────

def test_the_pre_split_list_resolves_including_the_superseded_paste(in_repo):
    teams = pins.pre_split_sample_teams()
    assert len(teams) == 72 and [_sha(t) for t in teams] == list(pins.PRE_SPLIT_SAMPLE_72)
    assert teams[17] == open("data/teams/superseded/0972146213a667c9.txt").read().strip()
    assert pins.measurement_bias_teams() == teams


def test_a_missing_pinned_sha_raises_naming_it(in_repo):
    with pytest.raises(KeyError, match="deadbeef00"):
        pins.teams_by_sha(["bcd4d09ee9", "deadbeef00"])


def test_pins_team_sha_is_the_repo_convention():
    from agents.training.team_archetypes import team_sha
    for body in ("Tyranitar @ Leftovers\n", "\n  x  \n\n"):
        assert pins.team_sha(body) == team_sha(body)


# ── relocations ─────────────────────────────────────────────────────────────────────────────────

def test_every_relocation_is_a_byte_identical_move(in_repo):
    """Data contract: every OLD path is gone, every NEW path exists, and the map covers exactly the
    41 files that left data/teams/sample/ (40 promoted + 1 superseded)."""
    moved = relocation_map()
    assert len(moved) == 41
    for old, new in moved.items():
        assert old.startswith("teams/sample/")
        assert not os.path.exists(os.path.join("data", old)), old
        assert os.path.isfile(os.path.join("data", new)), new
    promoted = json.loads(repo_path("data", "teams", "promoted", "teams.json").read_text())
    superseded = json.loads(repo_path("data", "teams", "superseded", "teams.json").read_text())
    assert sorted(moved.values()) == sorted(e["file"] for e in promoted + superseded)


def test_resolve_follows_a_relocation_and_leaves_everything_else_alone(in_repo, capsys):
    assert resolve_team_file("data/teams/sample/f6229d2c867e21d6.txt") == \
        "data/teams/sample/f6229d2c867e21d6.txt"                      # exists → unchanged
    got = resolve_team_file("data/teams/sample/8bdb5796b9.txt")
    assert got == "data/teams/promoted/8bdb5796b9.txt"
    assert "relocated" in capsys.readouterr().err                   # it says so when it fires
    absolute = os.path.join(str(repo_root()), "data/teams/sample/0972146213a667c9.txt")
    assert resolve_team_file(absolute) == os.path.join(
        str(repo_root()), "data/teams/superseded/0972146213a667c9.txt")
    assert resolve_team_file("data/teams/sample/nope.txt") == "data/teams/sample/nope.txt"
    assert resolve_team_file("elsewhere/x.txt") == "elsewhere/x.txt"


def test_read_recorded_trainee_teams_follows_a_relocation(tmp_path, in_repo):
    """A run that recorded a now-relocated path — with the fingerprint it recorded — still reads,
    and the fingerprint check still binds (the move is byte-identical)."""
    from agents.training.matchup_spec import read_recorded_trainee_teams
    legacy = "data/teams/sample/0972146213a667c9.txt"
    raw = open(resolve_team_file(legacy, quiet=True)).read()
    meta = {"cli_args": {"trainee_teams": legacy},
            "matchup_history": [{"spec": {"trainee_teams": {
                "pin_shas": [hashlib.sha1(raw.encode()).hexdigest()[:10]]}}}]}
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    assert read_recorded_trainee_teams(str(tmp_path)) == [
        "data/teams/superseded/0972146213a667c9.txt"]
    meta["matchup_history"][0]["spec"]["trainee_teams"]["pin_shas"] = ["0000000000"]
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="no longer match"):
        read_recorded_trainee_teams(str(tmp_path))


# ── the archive: every recorded team path still loads, every exploiter pin still validates ───────

def test_every_archived_trainee_pin_resolves_and_every_exploiter_pin_validates(in_repo):
    from utils.paths import main_models_dir
    md = main_models_dir()
    if md is None:
        pytest.skip("no models/ archive on this box")
    vetted = {_sha(t) for t in TeamLoader().get_exploiter_trainee_teams()}
    n_runs = n_exploiters = 0
    missing, refused = [], []
    for meta_path in sorted(glob.glob(os.path.join(str(md), "*", "metadata.json"))):
        try:
            with open(meta_path) as fh:
                cli = json.load(fh).get("cli_args") or {}
        except (OSError, ValueError):
            continue
        raw = cli.get("trainee_teams") or cli.get("trainee_team")
        if not raw:
            continue
        n_runs += 1
        run = os.path.basename(os.path.dirname(meta_path))
        files = [resolve_team_file(x.strip(), quiet=True) for x in str(raw).split(",") if x.strip()]
        missing += [(run, f) for f in files if not os.path.isfile(f)]
        if cli.get("exploiter") and not cli.get("allow_nonsample_trainee"):
            n_exploiters += 1
            refused += [(run, f) for f in files
                        if os.path.isfile(f) and _sha(open(f).read()) not in vetted]
    assert n_runs, "no archived run pinned a trainee team — this test would assert nothing"
    assert not missing, f"{len(missing)} recorded team path(s) no longer resolve: {missing[:5]}"
    assert not refused, f"{len(refused)} archived exploiter pin(s) now fail the guard: {refused[:5]}"
    assert n_exploiters
