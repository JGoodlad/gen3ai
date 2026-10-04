"""The X5 dex-row table's two ENCODER gates (`gen3_x5_dex_rows_v1`, U1;
`designs/endstate/design_x5_belief_tokens.md` §3.4):

(i) the BYTE gate — regenerating the table through THE observation encoder (`core_events
--dex-rows`, the Rust `encoder::hypothesis::hypothesis_slot`) reproduces the committed artifact
byte for byte. A change to the encoder's slot writer, to `data/pokemon/` or to the species list
that moves any row fails here until the artifact is regenerated (`python -m
agents.model.hypothesis_dex_rows --write`) — and the regeneration is then a reviewed diff.

(ii) the REAL-STATE cross-check — `src/rust_sim/tests/hypothesis_dex_rows_test.rs`, run here in the
self-check build so it rides the gate: at the first appearance of every base-form species (and of
every mon of the bridge corpus's real teams) in real Rust battles, the encoder's real row equals the
synthetic row on every cell outside the DECLARED on-field blocks. No tolerance.
"""
import os
import re
import shutil
import subprocess

import pytest

from agents.model import hypothesis_dex_rows as H
from utils.paths import src_path

pytestmark = pytest.mark.sim


def test_the_committed_table_is_what_the_encoder_writes():
    want = H.ARTIFACT.read_text(encoding="utf-8")
    got = H.generate()
    if got != want:
        a, b = want.splitlines(), got.splitlines()
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        pytest.fail(f"{H.ARTIFACT.name} is STALE (first differing line {first + 1}: "
                    f"{(b[first] if first < len(b) else '<end>')[:160]!r}) — regenerate with "
                    "`python -m agents.model.hypothesis_dex_rows --write` and review the diff")


def test_the_real_state_cross_check_passes():
    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo):
        pytest.fail("cargo is not installed — the real-state cross-check cannot run (install rustup)")
    p = subprocess.run(
        [cargo, "test", "--profile", "selfcheck", "--features", "emission-selfcheck",
         "--test", "hypothesis_dex_rows_test", "--", "--nocapture", "--test-threads", "2"],
        cwd=src_path("rust_sim"), capture_output=True, text=True, check=False)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-4000:]
    assert "test result: ok. 2 passed" in out, out[-4000:]
    # non-vacuity: the coverage set met every species the table holds
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    table = H.load_hypothesis_dex_rows(Gen3ObservationEncoder(load_mappings()).get_layout()["max_species"])
    cov = re.search(r"COVERAGE: .* first appearances of (\d+) species", out)
    assert cov and int(cov.group(1)) == int(table.valid.sum()), out[-2000:]
