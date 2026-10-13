"""The committed legal-movepool table EQUALS Showdown's own validator, species by species and move by move.

``data/pokemon/gen3_learnset.json`` is the hard legality gate of the move-belief prior. It was once built from each
species' OWN learnset entry, so every move reachable only through a pre-evolution (Dragon Dance on Tyranitar, via
Larvitar's egg / event sources) or through Sketch read as illegal (``gen3_learnset_prevo_chain_v1``). The authority
is ``TeamValidator.checkCanLearn``; ``src/rust_sim/harness/dump_gen3_learnset.js --check`` asks it for every base
species x every gen-3 move and fails on any difference in either direction. Needs the Showdown submodule's
``dist/`` + ``node_modules`` (a dev/test ORACLE, never a runtime path).
"""
import subprocess

import pytest

from utils.paths import repo_path

pytestmark = pytest.mark.integration

_HARNESS = repo_path("src", "rust_sim", "harness", "dump_gen3_learnset.js")


def test_committed_learnset_equals_the_showdown_validator():
    proc = subprocess.run(["node", str(_HARNESS), "--check"], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"gen3_learnset.json drifts from Showdown's validator:\n{proc.stderr}{proc.stdout}"
    assert "Showdown validator verdicts" in proc.stdout
