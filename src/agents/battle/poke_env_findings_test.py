"""The poke-env findings registry is VALUE-AWARE: each predicate explains exactly its finding's
difference and nothing near it (a blanket tolerance would pass every case below)."""

from types import SimpleNamespace as NS

from agents.battle.poke_env_findings import FINDINGS, explain, obs_block_explained


def mon(**kw):
    base = dict(fainted=False, active=True, status=None, status_counter=0, boosts={}, volatiles={})
    base.update(kw)
    return NS(**base)


def test_pe_v10_only_a_fainted_mon_whose_stages_the_truth_cleared():
    assert explain("boosts", mon(fainted=True, boosts={"spa": 1}), mon(fainted=True)) == "PE-V10"
    assert explain("boosts", mon(boosts={"spa": 1}), mon()) is None, "a LIVING mon's stages are a sim fact"
    assert explain("boosts", mon(fainted=True, boosts={"spa": 1}), mon(fainted=True, boosts={"spa": 2})) is None


def test_pe_r1b_exactly_one_apart_while_active_or_frozen_at_the_faint():
    assert explain("status_counter", mon(status="tox", status_counter=1), mon(status="tox")) == "PE-R1b"
    assert explain("status_counter", mon(status="fnt", status_counter=1), mon(status="fnt")) == "PE-R1b"
    assert explain("status_counter", mon(status="tox", status_counter=1), mon(status="tox", status_counter=2)) \
        == "PE-R1b", "one BEHIND between the residual and the next |turn|"
    assert explain("status_counter", mon(status="tox", status_counter=2), mon(status="tox")) is None, "+2"
    assert explain("status_counter", mon(status="slp", status_counter=1), mon(status="slp")) is None, "sleep"
    assert explain("status_counter", mon(status="tox", status_counter=1, active=False),
                   mon(status="tox", active=False)) is None, "a benched mon's count is not this finding"


def test_pe_v16_only_the_missing_flashfire():
    assert explain("volatiles", mon(volatiles={"confusion": 1}),
                   mon(volatiles={"confusion": 1, "flashfire": 0})) == "PE-V16"
    assert explain("volatiles", mon(), mon(volatiles={"flashfire": 0, "taunt": 1})) is None, "a second difference"


def test_every_finding_names_its_obs_blocks_and_reach():
    for f in FINDINGS.values():
        assert f.obs_blocks and f.reaches_obs and f.reproduce and f.source
    assert obs_block_explained("opp_team[3].status_counters", ["PE-R1b"])
    assert not obs_block_explained("opp_team[3].moves", ["PE-R1b"])
    assert not obs_block_explained("context", [])
