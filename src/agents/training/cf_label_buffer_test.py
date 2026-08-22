"""Unit gates for the counterfactual LABEL BUFFER + the reconstruction-record RING.

The buffer is the trainer's only contact with an out-of-process label producer, so the contract it
must not break is *tolerance*: an unknown schema, an unknown kind, a truncated line, a bad digest,
a wrong-width obs — every one of those is a COUNTED skip, never a crash and never a silently
accepted row. Those two failure modes are opposite and both fatal (a crash takes a training run
down over a producer bug; a silent accept feeds the critic garbage), which is why every case below
asserts BOTH that the run survives and that the counter moved.
"""
import base64
import hashlib
import json

import numpy as np
import pytest

from agents.training.cf_label_buffer import (
    DEFAULT_LAG_BOUND,
    SCHEMA_VERSION,
    CfLabelBuffer,
    batch_tensors,
)
from agents.training.cf_records import CfRecordRing


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _obs(dim=8, fill=1.0):
    return np.full(dim, fill, dtype=np.float32)


def _row(obs=None, *, label=0.7, policy_step=1000, schema=SCHEMA_VERSION, kind="mc_winprob",
         sha1=None, inline=True, npz=None, **extra):
    obs = _obs() if obs is None else obs
    raw = np.ascontiguousarray(obs, dtype=np.float32).tobytes()
    row = {
        "schema": schema, "kind": kind, "battle": "battle-gen3ou-1", "decision_idx": 3,
        "obs_sha1": hashlib.sha1(raw).hexdigest() if sha1 is None else sha1,
        "obs_npz": npz,
        "obs_inline": base64.b64encode(raw).decode() if inline else None,
        "label": label, "n_rollouts": 8, "wilson_lo": 0.5, "wilson_hi": 0.9,
        "policy_step": policy_step, "opponent": "pool_3", "created_unix": 1.0,
    }
    row.update(extra)
    return row


def _write(tmp_path, rows, name="labels_probe_0.jsonl", mode="w"):
    p = tmp_path / name
    with open(p, mode) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


# --------------------------------------------------------------------------------------
# ingest + FIFO
# --------------------------------------------------------------------------------------
def test_ingests_valid_rows_and_resolves_inline_obs(tmp_path):
    _write(tmp_path, [_row(_obs(fill=2.0), label=0.25)])
    buf = CfLabelBuffer(tmp_path, obs_dim=8)
    assert buf.poll(1000) == 1
    assert len(buf) == 1
    row = buf.sample(4)[0]
    assert row.label == pytest.approx(0.25)
    assert np.allclose(row.obs, 2.0)
    assert row.policy_step == 1000 and row.opponent == "pool_3"
    assert buf.ingested_total == 1 and buf.skipped_total == 0


def test_poll_is_incremental_and_never_re_ingests(tmp_path):
    """A second poll over an UNCHANGED file must ingest nothing — the byte offset is the memory.

    Without this the buffer would re-teach the same labels every rollout, which reads as a healthy
    `labels_ingested_total` while the effective sample is one file.
    """
    _write(tmp_path, [_row(), _row()])
    buf = CfLabelBuffer(tmp_path, obs_dim=8)
    assert buf.poll(1000) == 2
    assert buf.poll(1000) == 0
    _write(tmp_path, [_row()], mode="a")          # the producer APPENDS to the same file
    assert buf.poll(1000) == 1
    assert buf.ingested_total == 3


def test_a_partial_trailing_line_is_left_for_the_next_poll(tmp_path):
    """A producer caught mid-write must not have its half-line parsed as malformed JSON."""
    p = tmp_path / "labels_probe_0.jsonl"
    good = json.dumps(_row())
    p.write_text(good + "\n" + good[: len(good) // 2])       # second row truncated, no newline
    buf = CfLabelBuffer(tmp_path, obs_dim=8)
    assert buf.poll(1000) == 1
    assert buf.skipped_total == 0                            # NOT counted as malformed
    p.write_text(good + "\n" + good + "\n")                  # the producer finishes the line
    assert buf.poll(1000) == 1
    assert buf.skipped_total == 0


def test_fifo_evicts_the_oldest_at_capacity(tmp_path):
    _write(tmp_path, [_row(_obs(fill=float(i)), policy_step=1000 + i) for i in range(6)])
    buf = CfLabelBuffer(tmp_path, obs_dim=8, capacity=3)
    buf.poll(1000)
    assert len(buf) == 3
    steps = sorted(r.policy_step for r in buf.sample(10))
    assert steps == [1003, 1004, 1005]                       # the three NEWEST survived


def test_sample_is_bounded_and_without_replacement(tmp_path):
    _write(tmp_path, [_row(_obs(fill=float(i)), policy_step=1000 + i) for i in range(10)])
    buf = CfLabelBuffer(tmp_path, obs_dim=8)
    buf.poll(1000)
    got = buf.sample(4)
    assert len(got) == 4
    assert len({r.policy_step for r in got}) == 4
    assert buf.sample(100) and len(buf.sample(100)) == 10
    assert buf.sample(0) == []


# --------------------------------------------------------------------------------------
# STALENESS
# --------------------------------------------------------------------------------------
def test_rows_expire_at_exactly_the_lag_bound(tmp_path):
    """age == lag_bound SURVIVES; age == lag_bound + 1 does not. The boundary is the whole test —
    an off-by-one here silently halves or doubles the staleness the critic is taught on."""
    lag = 100
    _write(tmp_path, [_row(policy_step=1000)])
    buf = CfLabelBuffer(tmp_path, obs_dim=8, lag_bound=lag)
    buf.poll(1000)
    assert len(buf) == 1
    assert buf.expire(1000 + lag) == 0                       # age == bound → kept
    assert len(buf) == 1
    assert buf.expire(1000 + lag + 1) == 1                   # age == bound + 1 → dropped
    assert len(buf) == 0 and buf.expired_total == 1


def test_stale_rows_are_rejected_at_ingest_too(tmp_path):
    """A producer that fell far behind must not be able to flood the buffer with rows that the
    very next expire() would drop — the ingest-side bound is what keeps `buffer_fill` honest."""
    _write(tmp_path, [_row(policy_step=10)])
    buf = CfLabelBuffer(tmp_path, obs_dim=8, lag_bound=100)
    assert buf.poll(1_000_000) == 0
    assert len(buf) == 0 and buf.expired_total == 1 and buf.ingested_total == 0


def test_lag_bound_zero_disables_expiry(tmp_path):
    _write(tmp_path, [_row(policy_step=1)])
    buf = CfLabelBuffer(tmp_path, obs_dim=8, lag_bound=0)
    buf.poll(10_000_000)
    assert len(buf) == 1
    assert buf.expire(10 ** 9) == 0


def test_default_lag_bound_is_about_one_ppo_iteration():
    assert DEFAULT_LAG_BOUND == 150_000


# --------------------------------------------------------------------------------------
# TOLERANCE — every one of these is a counted skip, never a crash
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("row,reason", [
    (_row(schema=0), "schema"),
    (_row(schema=2), "schema"),
    (_row(kind="mc_value"), "kind"),
    (_row(kind="outcome_fact"), "kind"),
    (_row(label=1.5), "label_range"),
    (_row(label=-0.1), "label_range"),
    (_row(sha1="deadbeef"), "obs_sha1"),
    (_row(inline=False), "obs_unresolvable"),
    (_row(inline=False, npz="no_separator"), "obs_npz_spec"),
    (_row(inline=False, npz="/nonexistent.npz::obs"), "obs_npz_unreadable"),
])
def test_bad_rows_are_skipped_with_a_counter(tmp_path, row, reason):
    _write(tmp_path, [row])
    buf = CfLabelBuffer(tmp_path, obs_dim=8)
    assert buf.poll(1000) == 0
    assert len(buf) == 0
    assert buf.skipped_total == 1
    assert buf.skip_reasons.get(reason) == 1, buf.skip_reasons


def test_malformed_json_is_skipped_not_raised(tmp_path):
    p = tmp_path / "labels_probe_0.jsonl"
    p.write_text("{not json at all\n[1,2,3]\n" + json.dumps(_row()) + "\n")
    buf = CfLabelBuffer(tmp_path, obs_dim=8)
    assert buf.poll(1000) == 1                     # the ONE good row still lands
    assert buf.skipped_total == 2
    assert buf.skip_reasons["malformed_json"] == 1 and buf.skip_reasons["not_an_object"] == 1


def test_wrong_obs_width_is_a_gigo_skip(tmp_path):
    """The producer's obs must match THIS run's obs dim. A silent accept here is the architecture
    -drift bug class: a stale producer feeding a resized network."""
    _write(tmp_path, [_row(_obs(dim=5))])
    buf = CfLabelBuffer(tmp_path, obs_dim=8)
    assert buf.poll(1000) == 0
    assert buf.skip_reasons.get("obs_dim") == 1


def test_obs_dim_unset_accepts_any_width(tmp_path):
    _write(tmp_path, [_row(_obs(dim=5))])
    buf = CfLabelBuffer(tmp_path, obs_dim=None)
    assert buf.poll(1000) == 1


def test_npz_obs_resolves_when_inline_is_absent(tmp_path):
    arr = _obs(fill=3.0)
    npz = tmp_path / "states.npz"
    np.savez(npz, obs=np.stack([arr, arr]))
    # A one-row npz stored flat: the loader reshapes to 1-D, so store a single vector.
    np.savez(npz, obs=arr)
    _write(tmp_path, [_row(arr, inline=False, npz=f"{npz}::obs")])
    buf = CfLabelBuffer(tmp_path, obs_dim=8)
    assert buf.poll(1000) == 1
    assert np.allclose(buf.sample(1)[0].obs, 3.0)


def test_inline_wins_over_npz(tmp_path):
    """The declared resolution order is obs_inline > obs_npz. Pointing npz at a DIFFERENT array
    proves which one was actually read."""
    inline_arr, npz_arr = _obs(fill=1.0), _obs(fill=9.0)
    npz = tmp_path / "states.npz"
    np.savez(npz, obs=npz_arr)
    _write(tmp_path, [_row(inline_arr, npz=f"{npz}::obs")])
    buf = CfLabelBuffer(tmp_path, obs_dim=8)
    buf.poll(1000)
    assert np.allclose(buf.sample(1)[0].obs, 1.0)


def test_missing_dir_polls_to_zero_without_raising(tmp_path):
    buf = CfLabelBuffer(tmp_path / "never_created", obs_dim=8)
    assert buf.poll(1000) == 0 and len(buf) == 0


def test_a_truncated_or_replaced_file_is_re_read_from_zero(tmp_path):
    """A producer restart that rewrites its file shorter must not leave the reader seeked past the
    new end, silently ingesting nothing forever."""
    _write(tmp_path, [_row(), _row(), _row()])
    buf = CfLabelBuffer(tmp_path, obs_dim=8)
    assert buf.poll(1000) == 3
    _write(tmp_path, [_row()])                     # truncate + rewrite (mode "w")
    assert buf.poll(1000) == 1


# --------------------------------------------------------------------------------------
# the five liveness scalars
# --------------------------------------------------------------------------------------
def test_stats_publishes_the_five_scalars_even_when_starving(tmp_path):
    """An empty buffer must still REPORT. Silent starvation is this tree's oldest failure mode —
    absence of a scalar reads as absence of a problem."""
    buf = CfLabelBuffer(tmp_path, obs_dim=8)
    s = buf.stats(0)
    assert set(s) == {"cf/buffer_fill", "cf/label_age_steps_p50", "cf/labels_ingested_total",
                      "cf/labels_expired_total", "cf/labels_skipped_total"}
    assert s["cf/buffer_fill"] == 0.0 and s["cf/labels_ingested_total"] == 0.0


def test_stats_age_p50_tracks_staleness(tmp_path):
    _write(tmp_path, [_row(policy_step=1000), _row(policy_step=1200), _row(policy_step=1400)])
    buf = CfLabelBuffer(tmp_path, obs_dim=8, lag_bound=0)
    buf.poll(1400)
    s = buf.stats(1400)
    assert s["cf/buffer_fill"] == 3.0
    assert s["cf/label_age_steps_p50"] == pytest.approx(200.0)   # ages 400/200/0 → median 200


def test_batch_tensors_shapes():
    import torch as th
    rows = [_r for _r in _fake_rows(5)]
    obs, lab = batch_tensors(rows, th.device("cpu"))
    assert obs.shape == (5, 8) and lab.shape == (5,)
    assert obs.dtype == th.float32 and lab.dtype == th.float32


def _fake_rows(n):
    from agents.training.cf_label_buffer import CfLabel
    return [CfLabel(obs=_obs(fill=float(i)), label=0.5, policy_step=1, battle="b",
                    decision_idx=i, opponent="x", n_rollouts=8) for i in range(n)]


# --------------------------------------------------------------------------------------
# the record RING
# --------------------------------------------------------------------------------------
def _b64(obj):
    return base64.b64encode(json.dumps(obj).encode()).decode()


def test_ring_writes_a_loadable_record(tmp_path):
    ring = CfRecordRing(tmp_path, keep=4)
    path = ring.write_b64("battle-gen3ou-7", _b64({"seed": "sodium,ab", "input_log": []}))
    assert path is not None and path.exists()
    assert path.name.endswith("_reconstruction.json")
    rec = json.loads(path.read_text())
    assert rec["battle_tag"] == "battle-gen3ou-7" and rec["seed"] == "sodium,ab"
    assert not list(tmp_path.glob("*.tmp"))         # the crash-safe temp was renamed away


def test_ring_is_count_capped(tmp_path):
    ring = CfRecordRing(tmp_path, keep=3)
    for i in range(10):
        ring.write_b64(f"battle-{i}", _b64({"i": i}))
    files = sorted(p.name for p in tmp_path.iterdir())
    assert len(files) == 3
    # Filenames sort chronologically, so the survivors must be the LAST three written.
    assert all("battle-" in n for n in files)
    kept = {json.loads((tmp_path / n).read_text())["i"] for n in files}
    assert kept == {7, 8, 9}


def test_ring_survives_an_undecodable_payload(tmp_path):
    ring = CfRecordRing(tmp_path, keep=4)
    assert ring.write_b64("battle-1", "not base64 at all!!") is None
    assert list(tmp_path.iterdir()) == []
    assert ring.write_b64("battle-2", _b64({"ok": True})) is not None   # still working


def test_ring_prune_tolerates_a_racing_deleter(tmp_path):
    """Every env worker prunes the SAME shared dir, so a lost delete race is routine, not an error."""
    ring = CfRecordRing(tmp_path, keep=1)
    for i in range(3):
        ring.write_b64(f"b{i}", _b64({"i": i}))
    for p in list(tmp_path.iterdir()):
        p.unlink()                                   # simulate another worker clearing the dir
    assert ring.prune() == 0                         # no exception, nothing to do
    assert ring.write_b64("b9", _b64({"i": 9})) is not None


def test_ring_tags_are_filename_safe(tmp_path):
    ring = CfRecordRing(tmp_path, keep=4)
    path = ring.write_b64("../../evil/tag", _b64({}))
    assert path is not None and path.parent == tmp_path
    assert ".." not in path.name and "/" not in path.name
