"""The MANDATORY encoder benchmark runs end to end (a benchmark nobody can start is indistinguishable from one that
passes): two banked decisions, a few reps, both timing shapes printed and the per-decision JSON written."""
import json

import pytest

pytestmark = [pytest.mark.sim, pytest.mark.integration]


def test_the_encoder_benchmark_runs(tmp_path, capsys):
    from agents.observation import rust_encoder_benchmark as B

    out = tmp_path / "bench.json"
    assert B.main(["--decisions", "2", "--reps", "3", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "encode_us" in text and "present_encode_us" in text
    rows = json.loads(out.read_text())["rows"]
    assert len(rows) == 2 and all(r["encode_us"] > 0 and r["present_encode_us"] > 0 for r in rows)
