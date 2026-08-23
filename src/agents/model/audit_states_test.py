"""Regression gate for `gen3_audit_state_sampler_v1` + `gen3_audit_mask_recovery_v1` — the
ablation probes' state sampler and its LEGAL-MASK recovery.

Bug 1 (sampler): `sorted(glob)` + break-at-cap sampled every state from the LEXICALLY first
step dir (`step_10000032` < `step_2000016`), so every committed probe measurement was one
mid-run trace dir silently labelled a pool average (design_op_tensors.md §2.5.1). The
concat-deletion acceptance clause requires stratified sampling; these tests fail if the sampler
ever collapses to a single step dir again, and pin determinism (the outputs are committed
measurement artifacts — no wall-clock, no unseeded RNG).

Bug 2 (mask): the mask was recovered as `logits > -1e8`, but the recorder stores PRE-mask
logits — measured, 0 of 800+ archived `states.npz` back to ai_v5 carries a logit below -1e8 —
so the recovery returned ALL-LEGAL on every row ever audited and `edge_ablation_audit`'s
legality guard passed vacuously. These tests pin the three real sources (npz `action_mask`,
genuinely post-mask logits, the sibling summary's `valid` flags), the format-detection rule
between them, and that an unrecoverable trace REFUSES instead of defaulting to legal.
"""
import json

import numpy as np
import pytest

from agents.model.audit_states import TraceMaskUnavailable, collect_states, recover_legal_mask

_A = 11  # action-space width for the fake logits


def _write(tmp_path, step, opp, name, n_rows, fill, *, mask=True):
    d = tmp_path / "eval_traces" / step / opp
    d.mkdir(parents=True, exist_ok=True)
    # Every row unique (fill + a row index) so the determinism test can SEE which rows a
    # seed picked — constant-fill rows made different subsets compare equal.
    obs = (np.full((n_rows, 8), fill, dtype=np.float32)
           + 0.001 * np.arange(n_rows, dtype=np.float32)[:, None])
    # PRE-mask logits, exactly as the recorder writes them (inference/player.py stashes the raw
    # row; the -1e9 offset never reaches disk) — so `logits > -1e8` alone recovers nothing here.
    logits = np.zeros((n_rows, _A), dtype=np.float32)
    kw = {}
    if mask:
        am = np.ones((n_rows, _A), dtype=bool)
        am[:, -1] = False                               # struggle illegal — never all-legal
        kw["action_mask"] = am
    np.savez(d / name, obs=obs, logits=logits, **kw)


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


# ── gen3_audit_mask_recovery_v1 ───────────────────────────────────────────────────────────────

def _summary_for(valid_rows):
    """The recorder's summary shape: one ordered 11-entry `actions` dict per invocation, whose
    insertion order IS the action index (battle_recorder._all_action_labels)."""
    labels = [f"switch:mon{i}" for i in range(6)] + [f"move{i}" for i in range(4)] + ["struggle"]
    return {"invocations": [
        {"i": r + 1, "actions": {lbl: {"prob": "0.0%", "valid": bool(v)}
                                 for lbl, v in zip(labels, row)}}
        for r, row in enumerate(valid_rows)]}


def _npz(tmp_path, name, logits, **extra):
    obs = np.zeros((len(logits), 8), dtype=np.float32)
    np.savez(tmp_path / f"{name}_states.npz", obs=obs,
             logits=np.asarray(logits, dtype=np.float32), **extra)
    return str(tmp_path / f"{name}_states.npz")


def test_mask_comes_from_the_npz_action_mask_when_present(tmp_path):
    am = np.array([[True] * 7 + [False] * 4, [False] * 5 + [True] * 6])
    p = _npz(tmp_path, "a", np.zeros((2, _A)), action_mask=am)
    with np.load(p) as z:
        assert np.array_equal(recover_legal_mask(p, z), am)


def test_post_mask_logits_are_detected_and_used(tmp_path):
    """THE FORMAT-DETECTION RULE: a trace is post-mask iff ANY logit is below -1e8. Only the
    `(mask-1)*1e9` offset can put one there, so the test is exact in both directions."""
    lg = np.zeros((3, _A), dtype=np.float32)
    lg[:, 7:] = -1e9                                     # the offset the player applies
    p = _npz(tmp_path, "b", lg)                          # no action_mask, no summary
    with np.load(p) as z:
        m = recover_legal_mask(p, z)
    assert m[:, :7].all() and not m[:, 7:].any()


def test_pre_mask_logits_fall_through_to_the_sibling_summary(tmp_path):
    """The branch EVERY archived trace takes: pre-mask logits + a summary with `valid` flags.
    Fails on the old `logits > -1e8` recovery, which returns all-legal here."""
    rows = [[True] * 6 + [True, False, True, False] + [False],
            [False] * 6 + [True, True, False, False] + [False]]
    p = _npz(tmp_path, "c", np.zeros((2, _A)))
    with open(tmp_path / "c_summary.json", "w") as fh:
        json.dump(_summary_for(rows), fh)
    with np.load(p) as z:
        m = recover_legal_mask(p, z)
    assert np.array_equal(m, np.asarray(rows))
    assert not m.all(), "the summary carries real illegality — all-legal means recovery failed"


def test_pre_mask_logits_with_no_mask_source_REFUSE(tmp_path):
    """A vacuous guard is the bug, so silence is not an acceptable fallback: no action_mask, no
    -1e8 floor and no summary must STOP the audit, never default to all-legal."""
    p = _npz(tmp_path, "d", np.zeros((2, _A)))
    with np.load(p) as z, pytest.raises(TraceMaskUnavailable, match="PRE-mask"):
        recover_legal_mask(p, z)


def test_a_misaligned_summary_refuses_rather_than_shifting_indices(tmp_path):
    p = _npz(tmp_path, "e", np.zeros((3, _A)))
    with open(tmp_path / "e_summary.json", "w") as fh:
        json.dump(_summary_for([[True] * _A]), fh)       # 1 invocation vs 3 npz rows
    with np.load(p) as z, pytest.raises(TraceMaskUnavailable, match="not aligned"):
        recover_legal_mask(p, z)


def test_a_collapsed_actions_dict_refuses(tmp_path):
    """Duplicate action LABELS would collapse the summary's dict and shift every index after
    the collision — a silently wrong mask, so it is refused instead."""
    p = _npz(tmp_path, "f", np.zeros((1, _A)))
    s = _summary_for([[True] * _A])
    s["invocations"][0]["actions"].pop("struggle")       # 10 entries for an 11-wide action space
    with open(tmp_path / "f_summary.json", "w") as fh:
        json.dump(s, fh)
    with np.load(p) as z, pytest.raises(TraceMaskUnavailable, match="action entries"):
        recover_legal_mask(p, z)


def test_collect_states_refuses_a_maskless_trace(tmp_path):
    _write(tmp_path, "step_1", "heuristic", "loss_001_states.npz", 4, fill=1.0, mask=False)
    with pytest.raises(TraceMaskUnavailable):
        collect_states([str(tmp_path / "eval_traces" / "**" / "*_states.npz")], max_states=10)


_REAL_TRACES = ("/home/goodlad/dev/gen3ai/models/ai_v9_21_gen17_pfspoff_0820/"
                "eval_traces/**/*_states.npz")


def test_real_gen17_traces_recover_a_mask_with_illegal_actions():
    """THE test that matters — it FAILS on the `logits > -1e8` behaviour.

    Real gen-3 decisions always mask something: a fainted/absent bench slot cannot be switched
    to, struggle is illegal whenever any move has PP. Measured over 400 archived trace files,
    100% of rows carry at least one illegal action and ~38% of the action space is illegal on
    average. So a sample that comes back ALL-LEGAL is proof the recovery is broken — which is
    exactly what the pre-fix threshold returned on every trace in the archive.
    """
    try:
        _obs, masks, _cov = collect_states([_REAL_TRACES], 256, seed=0)
    except FileNotFoundError:
        pytest.skip("no gen-17 eval traces on this machine (models/ lives in the main checkout)")
    assert not masks.all(), (
        "every sampled real state decoded to ALL actions legal — the mask recovery is back to "
        "reading PRE-mask logits with a -1e8 threshold")
    assert masks.any(axis=1).all(), "no real state has zero legal actions"
    illegal = 1.0 - masks.mean()
    assert 0.05 < illegal < 0.95, f"implausible illegal fraction {illegal:.3f} for real traces"
