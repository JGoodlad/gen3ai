"""The LADDER-USAGE corpus is what its manifest says it is (`gen3_ladder_usage_corpus_v1`)."""
import json
import subprocess

import pytest

from utils import ladder_corpus as L


def test_every_tier_matches_its_stamp():
    assert L.verify() == {"commit": L.COMMIT_N, "milestone": L.MILESTONE_N,
                          "full": L.manifest()["tiers"]["full"]["n"]}


def test_tiers_are_prefixes_and_the_rows_are_well_formed():
    full = L.rows("full")
    assert [r["i"] for r in full] == list(range(len(full)))
    assert L.teams("commit") == L.teams("milestone")[:L.COMMIT_N]
    assert L.teams("milestone") == L.teams("full")[:L.MILESTONE_N]
    assert len({r["packed"] for r in full}) == len(full), "exact duplicates are dropped"
    for r in full:
        assert len(r["packed"].split("]")) == 6, r["file"]
        assert r["file"].endswith(".gen3ou_team") and len(r["sha"]) == 10


def test_the_filter_accounts_for_every_source_team():
    f = L.manifest()["filter"]
    c = f["counts"]
    dropped = c["import_fail"] + c["not_six"] + c["validator_fail"] + c["engine_unplayable"] + c["duplicate"]
    assert c["kept"] + dropped == c["total"] == L.manifest()["source"]["files"]
    assert c["kept"] == L.manifest()["tiers"]["full"]["n"]
    # only what the ENGINE cannot play is excluded — every id named, every drop attributed
    assert sum(f["engine_unplayable_by_id"].values()) >= c["engine_unplayable"]
    assert f["heal_bell_teams_kept"] > 0, "the already-fixed Heal Bell teams stay IN"


def test_the_corpus_holds_what_the_pool_never_showed():
    """The point of the corpus: Heal Bell is on real ladder teams (the crash the pool hid)."""
    full = L.teams("full")
    assert sum("healbell" in t.lower().replace(" ", "") for t in full) == \
        L.manifest()["filter"]["heal_bell_teams_kept"]


def test_the_pair_recipe_plays_each_milestone_team_once():
    n = L.MILESTONE_N
    seen = [t for k in range(n // 2) for t in L.pair(k)]
    assert seen == L.teams("milestone")


def test_a_tampered_data_file_is_refused(tmp_path, monkeypatch):
    bad = tmp_path / "teams.jsonl.gz"
    raw = bytearray(L.DATA.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    bad.write_bytes(bytes(raw))
    monkeypatch.setattr(L, "DATA", bad)
    L._all_rows.cache_clear()
    try:
        with pytest.raises(L.LadderCorpusError):
            L.rows("commit")
    finally:
        L._all_rows.cache_clear()


@pytest.mark.integration
def test_the_js_twin_reads_the_same_corpus():
    """`harness/ladder_corpus.js` (the fuzzers' reader) sees the same tiers, in the same order."""
    from utils.paths import repo_path

    js = repo_path("src", "rust_sim", "harness", "ladder_corpus.js")
    code = (f"const L=require({json.dumps(str(js))});"
            "console.log(JSON.stringify(['commit','milestone','full'].map(t=>L.ladderRows(t).length)));"
            "console.log(L.ladderRows('milestone')[799].packed);")
    out = subprocess.run(["node", "-e", code], capture_output=True, text=True, check=True).stdout.split("\n")
    assert json.loads(out[0]) == [L.COMMIT_N, L.MILESTONE_N, L.manifest()["tiers"]["full"]["n"]]
    assert out[1] == L.teams("milestone")[799]
