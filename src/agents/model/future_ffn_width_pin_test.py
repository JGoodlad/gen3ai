"""X59-S pin branch (x59s-pin-a8, PREREG Amendment 8): a v164 checkpoint's pickled ``trunk_ffn_dim`` loads at this
pin only at the base width. Each test fails on the revert of ``sanitize_future_ffn_width``."""
import pytest

from agents.model.arch_constants import TRANSFORMER_FFN_DIM
from agents.model.model_version import ModelVersionError
from agents.model.snapshot import sanitize_dead_extractor_kwargs, sanitize_future_ffn_width


def test_base_width_pops():
    fek = {"trunk_ffn_dim": TRANSFORMER_FFN_DIM, "trunk_ffn_by_kind": "off"}
    assert sanitize_future_ffn_width(fek) is True
    assert fek == {"trunk_ffn_by_kind": "off"}


def test_absent_is_untouched():
    fek = {"trunk_ffn_by_kind": "off"}
    assert sanitize_future_ffn_width(fek) is False
    assert fek == {"trunk_ffn_by_kind": "off"}


@pytest.mark.parametrize("width", [1024, 512, 255, True, "256", 256.0])
def test_any_other_width_is_refused(width):
    with pytest.raises(ModelVersionError, match="trunk_ffn_dim"):
        sanitize_future_ffn_width({"trunk_ffn_dim": width})


def test_the_zip_sanitizer_runs_it():
    fek = {"trunk_ffn_dim": TRANSFORMER_FFN_DIM}
    assert sanitize_dead_extractor_kwargs(fek) is True
    assert "trunk_ffn_dim" not in fek


def test_the_config_twin_pops_the_base_width_and_refuses_others():
    from agents.model.model_version.migrations import _migrate_config

    assert "trunk_ffn_dim" not in _migrate_config({"config_version": 164, "trunk_ffn_dim": 256})
    with pytest.raises(ModelVersionError, match="trunk_ffn_dim"):
        _migrate_config({"config_version": 164, "trunk_ffn_dim": 1024})


def test_a_v164_student_config_reads_and_matches_itself():
    """The follow-up's own sidecar config (skipped where the archive is absent): it builds a ModelVersion here, and
    the opponent gate h2h's slot groups run passes against itself."""
    import os

    from agents.model.model_version import ModelVersion

    p = os.path.expanduser("~/gen3ai_archive/x59s/followup_1500/read/students/s_h1500_5901/model/model_config.json")
    if not os.path.exists(p):
        pytest.skip("the X59-S follow-up archive is not on this box")
    v = ModelVersion.from_json_file(p)
    v.check_opponent_snapshot_compatible(ModelVersion.from_json_file(p.replace("s_h1500_5901", "s_hmix_5901")))
