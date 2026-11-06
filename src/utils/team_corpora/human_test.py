"""Human team corpora outside git (``utils.team_corpora.human``) and REVIEW F F-M7's one-entry-point guard.

Every corpus here is SYNTHETIC (two test teams in a temp ``$GEN3AI_ARCHIVE_DIR``); no human team is ever read or
written by a test. Each test names the revert it fails on.
"""
import json
from types import SimpleNamespace

import pytest

import utils.team_corpora as TC
from agents.training.team_mix_testkit import TEST_HUMAN_CORPUS, install_corpus, install_human_corpus
from utils.team_corpora import CorpusError
from utils.team_corpora import human as H
from utils.teambuilder_test import TEAM_A, TEAM_B


@pytest.fixture(autouse=True)
def _clear():
    H.load.cache_clear()
    TC.load_corpus.cache_clear()
    yield
    H.load.cache_clear()
    TC.load_corpus.cache_clear()


def test_the_committed_registry_holds_stubs_only():
    """The committed file is stubs (ids, sha256s, counts), never a team text: human data never leaves the box."""
    doc = json.loads(H.REGISTRY.read_text())
    assert doc["schema"] == H.STUB_SCHEMA
    for cid, s in doc["corpora"].items():
        assert s["human_data"] is True and s["id"] == cid == s["rowset"]
        assert not {"text", "texts", "teams", "rows"} & set(s), f"{cid}: a stub must not carry team data"
        assert "\n" not in json.dumps(s), "a stub value holds a multi-line (team) text"


def test_register_then_load_verifies_every_sha(monkeypatch, tmp_path):
    d = install_human_corpus(monkeypatch, tmp_path, [TEAM_A, TEAM_B])
    s = H.stub(TEST_HUMAN_CORPUS)
    assert s["n"] == 2 and s["tier"] == "ALL" and s["human_data"] is True and s["rowset"] == TEST_HUMAN_CORPUS
    assert str(d).startswith(str(tmp_path / "archive" / "human_teams"))
    c = H.load(TEST_HUMAN_CORPUS)
    assert len(c) == 2 and c.tier == "ALL" and c.data_sha256 == s["data_sha256"]
    ident = H.identity(TEST_HUMAN_CORPUS, c)
    assert ident["human_data"] is True and ident["soak_sha256"] == s["soak_sha256"]


@pytest.mark.parametrize("what", ["data", "manifest", "soak", "missing_dir", "missing_soak", "unregistered"])
def test_a_miss_or_a_mismatch_is_refused(monkeypatch, tmp_path, what):
    """Fails on revert of the stub comparison in ``load`` (a changed or absent corpus would train under a stub that
    names other bytes)."""
    d = install_human_corpus(monkeypatch, tmp_path, [TEAM_A, TEAM_B])
    cid = TEST_HUMAN_CORPUS
    if what == "data":           # a re-built corpus: consistent with its own manifest, not with the stub
        from agents.training.team_mix_testkit import write_corpus
        write_corpus(d, [TEAM_B, TEAM_A], {"ALL": 2}, name=cid)
        man = json.loads((d / "manifest.json").read_text())
        man["human_data"] = True
        (d / "manifest.json").write_text(json.dumps(man, indent=1))
    elif what == "manifest":
        man = json.loads((d / "manifest.json").read_text())
        man["note"] = "edited"
        (d / "manifest.json").write_text(json.dumps(man, indent=1))
    elif what == "soak":
        (d / H.SOAK_NAME).write_text("{}")
    elif what == "missing_dir":
        monkeypatch.setenv("GEN3AI_ARCHIVE_DIR", str(tmp_path / "elsewhere"))
    elif what == "missing_soak":
        (d / H.SOAK_NAME).unlink()
    else:
        cid = "hr_never.r9"
    with pytest.raises(CorpusError):
        H.load(cid)


def test_register_refuses_a_corpus_that_is_not_marked_human_or_not_whole(monkeypatch, tmp_path):
    d = install_human_corpus(monkeypatch, tmp_path, [TEAM_A, TEAM_B], register=False)
    man = json.loads((d / "manifest.json").read_text())
    man.pop("human_data")
    (d / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(CorpusError, match="human_data"):
        H.register(TEST_HUMAN_CORPUS)
    man["human_data"] = True
    man["tiers"] = {"ALL": {"n": 1, "sha256": TC.tier_sha256([TEAM_A.strip()])}}
    (d / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(CorpusError, match="must cover all 2 rows"):
        H.register(TEST_HUMAN_CORPUS)


# ── REVIEW F F-M7: one corpus, one entry point ──────────────────────────────────────────────────────────────────

def test_team_mix_refuses_a_human_data_corpus(monkeypatch, tmp_path):
    """THE F-M7 THROWING GUARD. A corpus whose manifest says ``"human_data": true`` cannot enter through
    ``--team-mix`` even when it is registered as a team-mix corpus: ``load_dir`` / ``load_corpus`` refuse it.
    FAILS ON REVERT of ``refuse_human_data`` in ``load_dir`` (the run would train on human teams with its
    provenance reading ``off``)."""
    from agents.training.matchup_spec import team_mix_from_args

    d = install_corpus(monkeypatch, tmp_path, [TEAM_A, TEAM_B], tiers={"ALL": 2}, name="t_humanish_v1")
    man = json.loads((d / "manifest.json").read_text())
    man["human_data"] = True
    (d / "manifest.json").write_text(json.dumps(man))
    TC.load_corpus.cache_clear()
    with pytest.raises(CorpusError, match="HUMAN DATA"):
        TC.load_dir(d, "ALL")
    with pytest.raises(CorpusError, match="--human-data teams"):
        team_mix_from_args(SimpleNamespace(team_mix="t_humanish_v1@ALL:0.5", human_data=None))
    # the switch's own reader is the one way in
    assert len(TC.load_dir(d, "ALL", human_data_ok=True)) == 2


def test_team_mix_refuses_a_registered_human_stub_id_at_parse_time(monkeypatch, tmp_path):
    """The parse-time twin: a name the human registry holds is refused by ``--team-mix``'s syntax check, so
    ``main.checkargs`` refuses it offline. Fails on revert of the ``is_registered`` test in ``parse_team_mix``."""
    from agents.training.matchup_spec import parse_team_mix

    install_human_corpus(monkeypatch, tmp_path, [TEAM_A, TEAM_B], name="hr_nodots_v1")
    monkeypatch.setitem(TC.CORPORA, "hr_nodots_v1", tmp_path / "archive" / "human_teams" / "hr_nodots_v1")
    with pytest.raises(ValueError, match="HUMAN team corpus"):
        parse_team_mix("hr_nodots_v1@ALL:0.5")
