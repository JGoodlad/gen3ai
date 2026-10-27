"""The OBS-SEMANTICS marker (`OBS_SEMANTICS_VERSION`, declared 2026-10-09 with config v151).

`ARCH_SIGNATURE` / `MIGRATION_FLOOR` catch a checkpoint whose WEIGHTS no longer fit. Nothing caught one whose
weights still load while the OBSERVATION it trained on changed meaning underneath it: v151 re-scaled the Toxic
counter cell (stage/8 -> stage/15) and re-defined the Wish flag with no shape change at all, so a v150 checkpoint
opened in the prober and answered about inputs it never saw. The marker is the one declared line that says
"records below this version read a different observation"; the prober compares a checkpoint's RECORDED config
version against it (`main.prober.arch_status`, tested in `main/prober/arch_status_test.py`).

This file pins the marker's own invariants and the one coupling that stops it being forgotten: the obs GOLDEN
(`agents/training/golden_obs_fixture.json`, the per-decision hash of the observation rows) is pinned HERE beside the
marker, so re-recording the golden — the signature of an observation-value change — fails until the author has
answered "did a cell change meaning?" by setting the marker (or re-pinning with the reason).
"""
import hashlib
import json
from pathlib import Path

from agents.model.model_version import (
    ARCH_SIGNATURE,
    MIGRATION_FLOOR,
    MODEL_CONFIG_VERSION,
    OBS_SEMANTICS_REASON,
    OBS_SEMANTICS_VERSION,
    SIGNATURE_FIRST_VERSION,
)

#: (the obs golden's content hash, the OBS_SEMANTICS_VERSION in force when it was pinned). Move BOTH, together,
#: with the commit that moves the golden: if a cell's MEANING changed set OBS_SEMANTICS_VERSION to the config
#: version that commit stamps; if the golden moved for a reason that changes no cell's meaning (it should not
#: move at all), say why in the commit and keep the marker.
_GOLDEN_PIN = ("1b29ad8f1f5c0c8f7bd01b4cf114cd1aacb1bafb5c4f241598048e01ff77d2fc", 151)

_GOLDEN = Path(__file__).resolve().parents[1] / "training" / "golden_obs_fixture.json"


def _golden_hash() -> str:
    hashes = json.loads(_GOLDEN.read_text())["hashes"]
    return hashlib.sha256("\n".join(hashes).encode()).hexdigest()


def test_the_marker_sits_between_the_floor_and_the_code():
    """A marker below the migration floor would be dead (no loadable record is that old); one above the code's
    version would flag every checkpoint, including HEAD's own."""
    assert MIGRATION_FLOOR <= OBS_SEMANTICS_VERSION <= MODEL_CONFIG_VERSION
    assert MIGRATION_FLOOR == SIGNATURE_FIRST_VERSION[ARCH_SIGNATURE]
    assert OBS_SEMANTICS_REASON.strip(), "the diagnosis quotes this clause; an empty one reads as a typo"


def test_the_golden_cannot_move_without_the_marker_being_considered():
    have, pinned_at = _golden_hash(), _GOLDEN_PIN[1]
    assert have == _GOLDEN_PIN[0], (
        "the obs GOLDEN moved. That is the signature of an OBSERVATION-VALUE change, and a checkpoint trained "
        "before it reads different inputs from now on, even if its weights load. Decide now: did a cell change "
        "MEANING? If yes, set OBS_SEMANTICS_VERSION (model_version/constants.py) to the config version this "
        "commit stamps, add its clause to OBS_SEMANTICS_REASON, and re-pin _GOLDEN_PIN here to "
        f"({have!r}, <that version>). If the golden moved but no cell's meaning did, re-pin with the same "
        "marker and say why in the commit body.")
    assert pinned_at == OBS_SEMANTICS_VERSION, (
        f"_GOLDEN_PIN was pinned at marker {pinned_at} but OBS_SEMANTICS_VERSION is {OBS_SEMANTICS_VERSION}: the "
        "marker moved without the pin (or the reverse). They move together.")
