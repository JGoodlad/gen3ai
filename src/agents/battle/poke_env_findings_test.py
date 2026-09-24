"""The poke-env findings registry: EMPTY since `gen3_pe_reading_fixes_v1`, and the mechanism that
routes a core-vs-reading difference through it is still VALUE-AWARE (a blanket tolerance would pass
every case below)."""

from types import SimpleNamespace as NS

from agents.battle import poke_env_findings as pef
from agents.battle.poke_env_findings import FINDINGS, Finding, explain, obs_block_explained


def mon(**kw):
    base = dict(fainted=False, active=True, status=None, status_counter=0, boosts={}, volatiles={})
    base.update(kw)
    return NS(**base)


def test_the_registry_is_empty_after_the_fork_fixes():
    """PE-V10 / PE-R1b / PE-V16 were fixed in the fork; a difference in their fields is now a
    DIVERGENCE. Re-registering one of them (instead of fixing the regression) fails here."""
    assert FINDINGS == {}
    assert explain("boosts", mon(fainted=True, boosts={"spa": 1}), mon(fainted=True)) is None
    assert explain("status_counter", mon(status="tox", status_counter=1), mon(status="tox")) is None
    assert explain("volatiles", mon(), mon(volatiles={"flashfire": 0})) is None
    assert not obs_block_explained("context", [])


def test_a_registered_finding_explains_exactly_its_own_difference(monkeypatch):
    """The mechanism, with a planted finding: value-aware on its field, blind to every other."""
    planted = Finding(
        id="PE-TEST", field="boosts", title="t", poke_env_reads="r", truth="t", source="s",
        reproduce="x", reaches_obs="YES", obs_blocks=("context",),
        predicate=lambda r, c: bool(c.fainted and not dict(c.boosts) and dict(r.boosts)))
    monkeypatch.setattr(pef, "FINDINGS", {"PE-TEST": planted})
    assert pef.explain("boosts", mon(fainted=True, boosts={"spa": 1}), mon(fainted=True)) == "PE-TEST"
    assert pef.explain("boosts", mon(boosts={"spa": 1}), mon()) is None, "a LIVING mon: not it"
    assert pef.explain("volatiles", mon(fainted=True, boosts={"spa": 1}), mon(fainted=True)) is None
    assert pef.obs_block_explained("context", ["PE-TEST"])
    assert not pef.obs_block_explained("opp_team[3].moves", ["PE-TEST"])
