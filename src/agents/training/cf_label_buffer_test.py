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

from unittest import mock

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
# the liveness scalars
# --------------------------------------------------------------------------------------
def test_stats_publishes_every_scalar_even_when_starving(tmp_path):
    """An empty buffer must still REPORT. Silent starvation is this tree's oldest failure mode —
    absence of a scalar reads as absence of a problem."""
    buf = CfLabelBuffer(tmp_path, obs_dim=8)
    s = buf.stats(0)
    assert set(s) == {"cf/buffer_fill", "cf/label_age_steps_p50", "cf/labels_ingested_total",
                      "cf/labels_expired_total", "cf/labels_future_total",
                      "cf/labels_replaced_total", "cf/labels_skipped_total",
                      # gen3_cf_twin_heads_v1 — the twin arm's own liveness. A twin-heads run whose
                      # producer ships no outcome_label trains head B on nothing, B becomes A, and
                      # C-B silently becomes C-A while every other counter reads healthy.
                      "cf/outcome_label_coverage", "cf/mc_return_coverage",
                      "cf/labels_mc_return_rejected_total", "cf/labels_field_skipped_total"}
    assert s["cf/buffer_fill"] == 0.0 and s["cf/labels_ingested_total"] == 0.0


def test_stats_age_p50_tracks_staleness(tmp_path):
    # Distinct obs per row — same state, different steps would DEDUP to one resident row.
    _write(tmp_path, [_row(_obs(fill=float(i)), policy_step=st)
                      for i, st in enumerate((1000, 1200, 1400))])
    buf = CfLabelBuffer(tmp_path, obs_dim=8, lag_bound=0)
    buf.poll(1400)
    s = buf.stats(1400)
    assert s["cf/buffer_fill"] == 3.0
    assert s["cf/label_age_steps_p50"] == pytest.approx(200.0)   # ages 400/200/0 → median 200


def test_batch_tensors_shapes():
    import torch as th
    rows = [_r for _r in _fake_rows(5)]
    b = batch_tensors(rows, th.device("cpu"))
    obs, lab, n = b.obs, b.label, b.n_rollouts
    assert obs.shape == (5, 8) and lab.shape == (5,) and n.shape == (5,)
    assert obs.dtype == th.float32 and lab.dtype == th.float32 and n.dtype == th.float32
    # gen3_cf_twin_heads_v1: every optional stream is present as a tensor + a MASK, and the mask is
    # what a consumer supervises through. A zero-filled absent label with no mask would be
    # indistinguishable from a confident "you lose".
    assert b.outcome.shape == (5,) and b.outcome_mask.tolist() == [0.0] * 5
    assert b.mc_return.shape == (5,) and b.mc_return_mask.tolist() == [0.0] * 5


def test_batch_tensors_carries_the_rollout_COUNT_not_just_the_ratio():
    """gen3_cf_binomial_likelihood_v1: `label` alone is not a sufficient statistic — 0.75 from 4
    rollouts and 0.75 from 16 are the same number carrying four times the evidence. The count must
    reach the loss, and the win count must be recoverable from the pair."""
    import torch as th
    from agents.training.cf_label_buffer import CfLabel
    rows = [CfLabel(obs=_obs(fill=0.0), label=0.75, policy_step=1, battle="b", decision_idx=0,
                    opponent="x", n_rollouts=r) for r in (4, 16)]
    _b = batch_tensors(rows, th.device("cpu"))
    lab, n = _b.label, _b.n_rollouts
    assert lab.tolist() == [0.75, 0.75]
    assert n.tolist() == [4.0, 16.0]
    assert th.round(lab * n).tolist() == [3.0, 12.0]


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
    # `prune_every=1` — the CAP is what this test is about, so the throttle is turned off rather
    # than reasoned around. The throttle's own bound has its own test below.
    ring = CfRecordRing(tmp_path, keep=3, prune_every=1)
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


# --------------------------------------------------------------------------------------
# the two UNBOUNDED / SILENT paths (adversarial review, 2026-08-22)
# --------------------------------------------------------------------------------------
def test_a_failed_ring_write_leaves_no_orphaned_tmp(tmp_path):
    """The `.tmp` is invisible to the count cap, so a failed write must clean up after itself.

    `prune` only ever considers names ending in RECON_SUFFIX; `<...>_reconstruction.json.tmp` is
    not one of those. Before this was guarded, a persistently failing write — the FULL DISK the
    module's docstring explicitly promises to survive — leaked one file per episode per env worker
    onto the filesystem that was already full, unbounded, and announced it exactly once (measured:
    5 failed writes left 5 orphans that 6 later successful writes never collected).
    """
    ring = CfRecordRing(tmp_path, keep=3)
    with mock.patch("agents.training.cf_records.os.replace",
                    side_effect=OSError(28, "No space left on device")):
        for i in range(5):
            assert ring.write_record(f"b{i}", {"i": i}) is None
    assert ring.errors == 5
    assert list(tmp_path.iterdir()) == [], "a failed write orphaned its .tmp"
    assert ring.write_record("b9", {"i": 9}) is not None      # still works once the disk is back


def test_ring_prune_sweeps_a_tmp_orphaned_by_a_CRASH(tmp_path):
    """A SIGKILL between `open` and `os.replace` cannot unlink its own tmp — prune must.

    Safe against a live writer by construction: a tmp being filled right now carries a NEWER
    `time_ns` than every record already on disk, and the sweep only deletes tmps older than the
    OLDEST KEPT record.
    """
    ring = CfRecordRing(tmp_path, keep=2, prune_every=1)
    stale = tmp_path / ("0" * 18 + "1_999_crashed_reconstruction.json.tmp")   # ancient ns prefix
    stale.write_text("{partial")
    live = tmp_path / ("9" * 19 + "_999_inflight_reconstruction.json.tmp")    # newer than anything
    live.write_text("{partial")
    for i in range(3):
        ring.write_record(f"b{i}", {"i": i})
    assert not stale.exists(), "a crash-orphaned .tmp was never collected"
    assert live.exists(), "the sweep deleted a tmp a concurrent writer may still be filling"
    kept = [p.name for p in tmp_path.iterdir() if p.name.endswith("_reconstruction.json")]
    assert len(kept) == 2


def test_a_recreated_label_file_is_read_from_the_START(tmp_path):
    """A producer that DELETES and RECREATES `labels_x.jsonl` must not lose its first rows.

    The byte offset is keyed on (name, INODE). Keyed on the name alone the buffer seeks past the
    NEW file's first `offset` bytes: those rows vanish with no skip counter and no warning — the
    one outcome this buffer's design says is impossible ("never a silent accept" has a mirror:
    never a silent DROP). Measured before the inode half existed: a recreated 3-row file ingested 1.
    """
    # DISTINCT obs per row: rows are deduped on the obs digest, so identical fixture states would
    # collapse to one resident row and hide the very loss this test is about.
    p = _write(tmp_path, [_row(_obs(fill=float(i)), policy_step=100) for i in range(2)],
               name="labels_a.jsonl")
    buf = CfLabelBuffer(tmp_path, obs_dim=8, lag_bound=0)
    assert buf.poll(100) == 2
    p.unlink()
    _write(tmp_path, [_row(_obs(fill=float(i)), policy_step=200) for i in range(2, 5)],
           name="labels_a.jsonl")
    assert buf.poll(200) == 3, "rows of a recreated label file were silently skipped"
    assert len(buf) == 5
    assert buf.skipped_total == 0


def test_a_same_size_rewrite_of_a_label_file_is_not_silently_ignored(tmp_path):
    """The nastier shape of the same defect: rewritten IN PLACE at the identical byte length."""
    p = _write(tmp_path, [_row(label=0.1, policy_step=100)], name="labels_a.jsonl")
    buf = CfLabelBuffer(tmp_path, obs_dim=8, lag_bound=0)
    assert buf.poll(100) == 1
    p.unlink()
    _write(tmp_path, [_row(label=0.9, policy_step=100)], name="labels_a.jsonl")   # same length
    assert buf.poll(100) == 1, "a recreated same-size file was skipped as 'nothing new'"


def test_the_offset_map_forgets_files_that_are_gone(tmp_path):
    """One entry per label file FOREVER is a slow leak on a multi-day run; track the DIRECTORY."""
    buf = CfLabelBuffer(tmp_path, obs_dim=8, lag_bound=0)
    for i in range(5):
        p = _write(tmp_path, [_row(policy_step=100)], name=f"labels_{i}.jsonl")
        buf.poll(100)
        p.unlink()
    buf.poll(100)                       # the poll that observes the last file is gone
    assert buf.ingested_total == 5
    assert buf._offsets == {}, "the offset map grew one entry per label file ever seen"


# --------------------------------------------------------------------------------------
# LABEL QUALITY — the pre-coefficient trio (task #28, 2026-08-22)
#
# All three are corrections to what the buffer TEACHES rather than to whether it survives, which
# is why they land before `--cf-winprob-coef` is ever nonzero: at coefficient zero a duplicated or
# immortal row costs nothing, and the moment the coefficient is live it is a silent reweighting of
# the sampler nobody declared.
# --------------------------------------------------------------------------------------
def test_a_repeated_state_REPLACES_its_resident_row_instead_of_stacking(tmp_path):
    """Dedup on the obs digest, keep-NEWEST — the sampler's declared weights are per STATE.

    A producer re-labelling ground it already covered (an overlapping cycle, a re-run over the same
    trace tree) would otherwise give that one state N x the weight of every other, changing the
    distribution the critic is supervised on with no flag, no counter and no log line.
    """
    same = _obs(fill=3.0)
    _write(tmp_path, [_row(same, label=0.2, policy_step=100),
                      _row(same, label=0.9, policy_step=200),
                      _row(_obs(fill=4.0), label=0.5, policy_step=200)])
    buf = CfLabelBuffer(tmp_path, obs_dim=8, lag_bound=0)
    assert buf.poll(200) == 3                       # all three were ACCEPTED …
    assert len(buf) == 2                            # … and two of them are the same state
    assert buf.replaced_total == 1
    got = {round(r.label, 3) for r in buf.sample(10)}
    assert got == {0.9, 0.5}, "the SUPERSEDED label survived — dedup kept the older measurement"


def test_a_rewritten_label_file_converges_to_the_file_s_own_row_count(tmp_path):
    """The truncate-and-rewrite re-ingest: a 5-row file re-read from zero must leave fill 5, not 6.

    Measured before dedup: the same file rewritten in place re-ingested its rows beside the
    originals, so a producer that rotates its output inflates the buffer by exactly the overlap.
    """
    rows = [_row(_obs(fill=float(i)), policy_step=100) for i in range(5)]
    p = _write(tmp_path, rows, name="labels_a.jsonl")
    buf = CfLabelBuffer(tmp_path, obs_dim=8, lag_bound=0)
    assert buf.poll(100) == 5 and len(buf) == 5
    p.unlink()
    _write(tmp_path, rows + [_row(_obs(fill=99.0), policy_step=100)], name="labels_a.jsonl")
    buf.poll(100)
    assert len(buf) == 6, "the one genuinely new row did not land"
    assert buf.replaced_total == 5, "the five re-read rows were not recognised as re-reads"


def test_a_FUTURE_dated_label_expires_exactly_like_a_stale_one(tmp_path):
    """Symmetric staleness: |age| > bound, in EITHER direction.

    A crash-restart resumes from the last checkpoint, so `num_timesteps` moves BACKWARDS while the
    label files still carry pre-crash steps. Under a one-sided `current - policy > bound` those
    rows are IMMORTAL — they never age out and quietly become the whole buffer. The tell was
    measured live: `cf/label_age_steps_p50` reading -4,999,000.
    """
    lag = 100
    _write(tmp_path, [_row(policy_step=5_000_000)])
    buf = CfLabelBuffer(tmp_path, obs_dim=8, lag_bound=lag)
    assert buf.poll(1_000) == 0, "a future-dated label was ingested"
    assert buf.expired_total == 1 and buf.future_total == 1


def test_the_future_boundary_is_the_same_INCLUSIVE_bound_as_the_past_one(tmp_path):
    """|age| == bound survives; |age| == bound + 1 does not. Both signs, one rule."""
    lag = 100
    _write(tmp_path, [_row(_obs(fill=1.0), policy_step=1000)])
    buf = CfLabelBuffer(tmp_path, obs_dim=8, lag_bound=lag)
    buf.poll(1000)
    assert buf.expire(1000 - lag) == 0               # age == -bound  → kept
    assert len(buf) == 1
    assert buf.expire(1000 - lag - 1) == 1           # age == -bound-1 → dropped
    assert len(buf) == 0 and buf.future_total == 1


def test_the_first_future_label_names_its_cause_out_loud(tmp_path, capsys):
    """A negative age is a DIAGNOSIS (someone resumed from an older checkpoint), not noise — so it
    says so once, by name, rather than only moving a counter nobody is watching yet."""
    _write(tmp_path, [_row(_obs(fill=float(i)), policy_step=900_000) for i in range(3)])
    buf = CfLabelBuffer(tmp_path, obs_dim=8, lag_bound=100)
    buf.poll(1_000)
    out = capsys.readouterr().out
    assert "NEWER snapshot" in out and "crash-restart rollback" in out
    assert out.count("NEWER snapshot") == 1, "the warning repeated per row"
    assert buf.future_total == 3


# --------------------------------------------------------------------------------------
# obs_npz — the ROW index and the per-file cache
# --------------------------------------------------------------------------------------
def test_obs_npz_selects_the_ROW_named_by_decision_idx(tmp_path):
    """`obs_npz` points at a battle's whole obs MATRIX and `decision_idx` selects the row — the
    schema's own wording, and what `cf_audit` emits by default (without `--inline-obs`).

    Ignoring the index flattened the matrix into one 1-D vector, which then failed the obs-width
    GIGO guard: the entire non-inline half of the schema was unconsumable, loudly but for a reason
    that pointed at architecture drift rather than at the reader.
    """
    mat = np.stack([np.full(8, float(i), dtype=np.float32) for i in range(5)])
    npz = tmp_path / "states.npz"
    np.savez(npz, obs=mat)
    _write(tmp_path, [_row(mat[3], inline=False, npz=f"{npz}::obs", decision_idx=3)])
    buf = CfLabelBuffer(tmp_path, obs_dim=8)
    assert buf.poll(1000) == 1, f"npz row not resolved: {buf.skip_reasons}"
    assert np.allclose(buf.sample(1)[0].obs, 3.0)


def test_an_out_of_range_npz_row_is_a_counted_skip(tmp_path):
    mat = np.stack([np.full(8, 1.0, dtype=np.float32)] * 2)
    npz = tmp_path / "states.npz"
    np.savez(npz, obs=mat)
    _write(tmp_path, [_row(mat[0], inline=False, npz=f"{npz}::obs", decision_idx=9)])
    buf = CfLabelBuffer(tmp_path, obs_dim=8)
    assert buf.poll(1000) == 0
    assert buf.skip_reasons.get("obs_npz_row") == 1


def test_many_rows_of_one_npz_open_the_file_ONCE(tmp_path):
    """N rows of a battle share one archive; re-opening it per row re-inflated and re-parsed it."""
    mat = np.stack([np.full(8, float(i), dtype=np.float32) for i in range(20)])
    npz = tmp_path / "states.npz"
    np.savez(npz, obs=mat)
    _write(tmp_path, [_row(mat[i], inline=False, npz=f"{npz}::obs", decision_idx=i)
                      for i in range(20)])
    buf = CfLabelBuffer(tmp_path, obs_dim=8)
    real_load = np.load
    with mock.patch("agents.training.cf_label_buffer.np.load",
                    side_effect=real_load) as loader:
        assert buf.poll(1000) == 20
    assert loader.call_count == 1, f"npz reopened {loader.call_count}x for 20 rows"


def test_the_npz_cache_is_bounded(tmp_path):
    """A multi-day run must not accumulate one decoded battle per label file ever seen."""
    from agents.training.cf_label_buffer import _NPZ_CACHE_FILES
    rows = []
    for i in range(_NPZ_CACHE_FILES + 3):
        arr = np.full(8, float(i), dtype=np.float32)
        npz = tmp_path / f"states_{i}.npz"
        np.savez(npz, obs=arr)
        rows.append(_row(arr, inline=False, npz=f"{npz}::obs"))
    _write(tmp_path, rows)
    buf = CfLabelBuffer(tmp_path, obs_dim=8)
    assert buf.poll(1000) == len(rows)
    assert len(buf._npz_cache) == _NPZ_CACHE_FILES


# --------------------------------------------------------------------------------------
# the ring: the prune THROTTLE and the cross-PROCESS cap
# --------------------------------------------------------------------------------------
def test_the_prune_throttle_bounds_the_overshoot_and_then_reclaims_it(tmp_path):
    """The prune is a full readdir on the bridge reader's coroutine, so it runs once in
    `prune_every` writes. The price is a BOUNDED transient overshoot, not an unbounded directory:
    at most `prune_every` unpruned writes, all collected by the next sweep."""
    keep, every = 3, 4
    ring = CfRecordRing(tmp_path, keep=keep, prune_every=every)
    for i in range(7):
        ring.write_b64(f"b{i}", _b64({"i": i}))
    n = len(list(tmp_path.iterdir()))
    assert keep <= n <= keep + every, f"overshoot {n} outside [keep, keep+prune_every]"
    ring.write_b64("b7", _b64({"i": 7}))             # the 8th write crosses the threshold
    assert len(list(tmp_path.iterdir())) == keep


def test_the_cap_is_GLOBAL_across_sequential_PROCESSES(tmp_path):
    """The launcher-restart survival claim, which stood on construction alone until this test.

    A restart brings a FRESH ring object over the SAME directory (a new pid, zeroed counters). The
    cap must still hold — nothing is keyed on the process — and the new writer must not
    double-count the old one's files as its own work.
    """
    keep = 4
    total = 0
    for gen in range(3):                              # three sequential "processes"
        ring = CfRecordRing(tmp_path, keep=keep, prune_every=1)
        assert ring.written == 0 and ring.pruned == 0, "a fresh ring inherited counters"
        for i in range(5):
            ring.write_b64(f"g{gen}-b{i}", _b64({"gen": gen, "i": i}))
        total += ring.written
        files = [p.name for p in tmp_path.iterdir()]
        assert len(files) == keep, f"generation {gen} left {len(files)} files (cap {keep})"
    assert total == 15                                # each process counted only its OWN writes
    # …and the survivors are the newest ones, written by the LAST process.
    kept = {json.loads((tmp_path / p.name).read_text())["gen"]
            for p in tmp_path.iterdir()}
    assert kept == {2}


def test_a_restarted_ring_prunes_the_PREVIOUS_process_leftovers(tmp_path):
    """The overshoot a crash leaves behind is not permanent: the next process's first sweep is over
    the WHOLE directory, so it collects files it never wrote."""
    dead = CfRecordRing(tmp_path, keep=2, prune_every=100)
    for i in range(9):
        dead.write_b64(f"old{i}", _b64({"i": i}))
    assert len(list(tmp_path.iterdir())) == 9         # the crashed process never swept
    fresh = CfRecordRing(tmp_path, keep=2, prune_every=1)
    fresh.write_b64("new", _b64({"i": 99}))
    assert len(list(tmp_path.iterdir())) == 2


# --------------------------------------------------------------------------------------
# THE TWIN STREAMS (gen3_cf_twin_heads_v1) — `outcome_label`, `mc_return`, `reward_sha1`.
#
# Additive-optional at schema v1, deliberately: `schema` is a REFUSAL gate, so bumping it to 2
# would make a new producer's output unreadable by an existing trainer — the opposite of backward
# compatible. The tests below pin BOTH directions of that claim.
# --------------------------------------------------------------------------------------

def test_an_OLD_producers_row_still_ingests_and_carries_no_extra_streams(tmp_path):
    """Backward compatibility, direction one: a row written before the amendment must be accepted
    unchanged and simply supervise nothing extra. If it were skipped, enabling the twin heads would
    silently starve the ENTIRE cf pipeline, not just the new arms."""
    _write(tmp_path, [_row(_obs(fill=1.0))])
    buf = CfLabelBuffer(tmp_path, obs_dim=8)
    assert buf.poll(1000) == 1
    row = buf.sample(1)[0]
    assert row.outcome_label is None and row.mc_return is None and row.mc_return_n == 0
    assert buf.skipped_total == 0


def test_a_NEW_row_ingests_both_streams(tmp_path):
    _write(tmp_path, [_row(_obs(fill=1.0), outcome_label=0.0, mc_return=-3.25,
                           mc_return_n=8, reward_sha1="abc")])
    buf = CfLabelBuffer(tmp_path, obs_dim=8)
    assert buf.poll(1000) == 1
    row = buf.sample(1)[0]
    assert row.outcome_label == 0.0 and row.mc_return == -3.25
    assert row.mc_return_n == 8 and row.reward_sha1 == "abc"


@pytest.mark.parametrize("bad", [1.5, -0.1, float("nan"), "win"])
def test_an_out_of_range_outcome_label_is_a_COUNTED_field_skip_not_a_lost_row(tmp_path, bad):
    """`outcome_label` is a win/loss/tie scalar, exactly like `label`. A value outside [0,1] is a
    producer bug and head B trained on it would be trained on garbage with no tell — but the ROW's
    tight-MC label is still perfectly good, so the row survives and the FIELD is dropped."""
    _write(tmp_path, [_row(_obs(fill=1.0), outcome_label=bad)])
    buf = CfLabelBuffer(tmp_path, obs_dim=8)
    assert buf.poll(1000) == 1, "the row itself must survive a bad optional field"
    assert buf.sample(1)[0].outcome_label is None
    # Its own counter, NOT the row-level GIGO meter: `ingested_total` and `skipped_total` must keep
    # partitioning the input, or "is the producer feeding me garbage rows" climbs at the ingestion
    # rate on a buffer that is refusing nothing.
    assert buf.field_skipped_total == 1 and buf.skipped_total == 0
    assert set(buf.skip_reasons) <= {"outcome_label_range", "outcome_label_malformed"}
    assert buf.stats(1000)["cf/labels_field_skipped_total"] == 1.0


def test_an_mc_return_with_the_WRONG_reward_digest_is_refused_and_counted(tmp_path):
    """THE GIGO GUARD for the shadow critic.

    A shaped return is a fact about a board UNDER A REWARD COMPOSITION. A return measured under a
    different `RewardConfig` is not a noisier sample of this run's value function — it is a
    measurement of a DIFFERENT one, and averaging it into the target is silent corruption with no
    shape error and no range violation to catch it. The FIELD is refused (the row's win-prob labels
    are untouched), the refusal has its own counter, and it warns once by name.
    """
    _write(tmp_path, [_row(_obs(fill=1.0), mc_return=2.0, mc_return_n=8,
                           reward_sha1="theirs")])
    buf = CfLabelBuffer(tmp_path, obs_dim=8, reward_sha1="ours")
    assert buf.poll(1000) == 1
    row = buf.sample(1)[0]
    assert row.mc_return is None and row.mc_return_n == 0
    assert buf.mc_return_rejected_total == 1
    # NOT counted as a skip: the row was accepted. Conflating "this row is garbage" with "this
    # row's shadow label is for a different reward" would hide a whole-arm misconfiguration
    # inside the GIGO meter.
    assert buf.skipped_total == 0
    assert buf.stats(1000)["cf/labels_mc_return_rejected_total"] == 1.0


def test_a_matching_reward_digest_is_accepted(tmp_path):
    _write(tmp_path, [_row(_obs(fill=1.0), mc_return=2.0, mc_return_n=8, reward_sha1="ours")])
    buf = CfLabelBuffer(tmp_path, obs_dim=8, reward_sha1="ours")
    buf.poll(1000)
    assert buf.sample(1)[0].mc_return == 2.0 and buf.mc_return_rejected_total == 0


def test_no_configured_digest_disables_the_check(tmp_path):
    """A run WITHOUT the shadow critic reads no `mc_return`, so it must not reject rows over a
    field it does not use — an unrelated arm's labels would otherwise show a rising rejection
    counter that means nothing."""
    _write(tmp_path, [_row(_obs(fill=1.0), mc_return=2.0, mc_return_n=8, reward_sha1="theirs")])
    buf = CfLabelBuffer(tmp_path, obs_dim=8)
    buf.poll(1000)
    assert buf.sample(1)[0].mc_return == 2.0 and buf.mc_return_rejected_total == 0


def test_coverage_scalars_report_the_arms_own_liveness(tmp_path):
    """The launch-window reading the arm can produce a confident WRONG answer without.

    A twin-heads run whose producer ships no `outcome_label` trains head B on nothing; B then
    equals A, and the pre-registered C−B contrast silently becomes C−A while every other counter
    reads healthy.
    """
    _write(tmp_path, [_row(_obs(fill=float(i)), outcome_label=(0.0 if i < 2 else None))
                      for i in range(4)])
    buf = CfLabelBuffer(tmp_path, obs_dim=8)
    buf.poll(1000)
    s = buf.stats(1000)
    assert s["cf/buffer_fill"] == 4.0
    assert s["cf/outcome_label_coverage"] == pytest.approx(0.5)
    assert s["cf/mc_return_coverage"] == 0.0


def test_batch_tensors_masks_the_absent_streams_rather_than_zero_filling_them(tmp_path):
    """A zero-filled absent label is indistinguishable from a confident "you lose" — the single
    most dangerous silent target this schema could produce. The mask is what a consumer supervises
    through, and it must be 0 exactly where the row carried nothing."""
    import torch as th
    from agents.training.cf_label_buffer import CfLabel
    rows = [
        CfLabel(obs=_obs(fill=0.0), label=0.5, policy_step=1, battle="b", decision_idx=0,
                opponent="x", n_rollouts=8, outcome_label=1.0, mc_return=4.0, mc_return_n=8),
        CfLabel(obs=_obs(fill=1.0), label=0.5, policy_step=1, battle="b", decision_idx=1,
                opponent="x", n_rollouts=8),
    ]
    b = batch_tensors(rows, th.device("cpu"))
    assert b.outcome.tolist() == [1.0, 0.0] and b.outcome_mask.tolist() == [1.0, 0.0]
    assert b.mc_return.tolist() == [4.0, 0.0] and b.mc_return_mask.tolist() == [1.0, 0.0]
