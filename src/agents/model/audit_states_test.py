"""Regression gate for `gen3_audit_state_sampler_v1` — the ablation probes' state sampler.

The bug this pins: `sorted(glob)` + break-at-cap sampled every state from the LEXICALLY first
step dir (`step_10000032` < `step_2000016`), so every committed probe measurement was one
mid-run trace dir silently labelled a pool average (design_op_tensors.md §2.5.1). The
concat-deletion acceptance clause requires stratified sampling; these tests fail if the sampler
ever collapses to a single step dir again, and pin determinism (the outputs are committed
measurement artifacts — no wall-clock, no unseeded RNG).
"""
import numpy as np
import pytest

from agents.model.audit_states import collect_states

_A = 11  # action-space width for the fake logits


def _write(tmp_path, step, opp, name, n_rows, fill):
    d = tmp_path / "eval_traces" / step / opp
    d.mkdir(parents=True, exist_ok=True)
    # Every row unique (fill + a row index) so the determinism test can SEE which rows a
    # seed picked — constant-fill rows made different subsets compare equal.
    obs = (np.full((n_rows, 8), fill, dtype=np.float32)
           + 0.001 * np.arange(n_rows, dtype=np.float32)[:, None])
    logits = np.zeros((n_rows, _A), dtype=np.float32)   # all legal (0 > -1e8)
    np.savez(d / name, obs=obs, logits=logits)


@pytest.fixture()
def trap_tree(tmp_path):
    """The literal lexical trap: `step_10000032` sorts BEFORE `step_2000016`, and alone it
    holds more rows than the cap — the old sampler never left it."""
    _write(tmp_path, "step_10000032", "heuristic", "loss_001_states.npz", 40, fill=10.0)
    _write(tmp_path, "step_10000032", "aggressive", "win_001_states.npz", 40, fill=11.0)
    _write(tmp_path, "step_2000016", "heuristic", "loss_001_states.npz", 40, fill=2.0)
    _write(tmp_path, "step_22000032", "sentinel_0", "loss_001_states.npz", 40, fill=22.0)
    return tmp_path


def test_sampler_spans_every_step_dir(trap_tree):
    pattern = str(trap_tree / "eval_traces" / "**" / "*_states.npz")
    obs, masks, cov = collect_states([pattern], max_states=60)
    assert len(obs) == 60 and masks.shape == (60, _A)
    # THE regression: all three step dirs represented, none dominant beyond its share.
    assert set(cov["per_step"]) == {"step_10000032", "step_2000016", "step_22000032"}, (
        f"sampler collapsed to {set(cov['per_step'])} — the sorted-glob break-at-cap bug is back")
    assert max(cov["per_step"].values()) < 60, "one step dir supplied the entire sample"
    # Opponent buckets recorded too.
    assert set(cov["per_opponent"]) == {"heuristic", "aggressive", "sentinel_0"}
    # Coverage totals must describe the ACTUAL sample.
    assert sum(cov["per_step"].values()) == 60 == cov["n_states"]


def test_sampler_is_deterministic(trap_tree):
    pattern = str(trap_tree / "eval_traces" / "**" / "*_states.npz")
    a = collect_states([pattern], max_states=50, seed=3)
    b = collect_states([pattern], max_states=50, seed=3)
    assert np.array_equal(a[0], b[0]) and a[2] == b[2]
    c = collect_states([pattern], max_states=50, seed=4)
    assert not np.array_equal(a[0], c[0]), "different seeds must draw different row subsets"


def test_cap_larger_than_data_takes_everything(trap_tree):
    pattern = str(trap_tree / "eval_traces" / "**" / "*_states.npz")
    obs, _, cov = collect_states([pattern], max_states=10_000)
    assert len(obs) == 160
    assert cov["n_files_read"] == cov["n_files_matched"] == 4


def test_no_match_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect_states([str(tmp_path / "nothing" / "*.npz")], max_states=10)
