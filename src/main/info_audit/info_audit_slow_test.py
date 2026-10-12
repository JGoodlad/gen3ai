"""The information-lever audit suite on the LARGER fixed corpus (the slow tier): 12 committed-bank decisions, the
honest builder, the pilot's, the oracle and the study's move-lever arms. Every check must pass; a decision a spec
cannot build is an EXCLUSION (counted in the report, as the hidden-info study counts them), never a pass."""
import pytest

from main.info_audit import __main__ as A

pytestmark = [pytest.mark.sim, pytest.mark.integration, pytest.mark.slow]

SPECS = ("off", "pilot", "oracle", "F+M", "F+U", "O-M", "D50", "W90", "MAP90", "F+S")


def test_the_suite_passes_on_twelve_decisions_and_every_spec():
    rep = A.run(A.select(12), SPECS, 4, log=lambda m: None)
    bad = {k: v for k, v in rep["checks"].items() if not v["pass"]}
    assert not bad, bad
    for s in ("oracle", "pilot"):
        assert not rep["specs"][s]["excluded"], (s, rep["specs"][s]["excluded"])
    # the honest builder draws every unrevealed seen fact: nothing is TOLD under `off`
    assert not any(k.endswith(":told") for k in rep["leakage"]["off"]["table"])
    # a mass-1 truth lever: its class is told and always the truth
    assert rep["leakage"]["F+M"]["table"]["seen:moves:told"]["truth_match"] == 1.0
    assert rep["leakage"]["O-M"]["table"]["unseen:species:told"]["truth_match"] == 1.0
