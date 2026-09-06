"""Unit tests for the label row schema + its batch writer (`cf_producer_labels.py`).

The row is a CONTRACT with `cf_label_buffer`, so these pin the v1 key set, the
ecology field every row must carry, the optional streams that ride beside it, and
the writer's never-reuse-a-name rule (the buffer keys offsets on `(name, inode)`).

These moved out of `cf_producer_test.py` with the functions they cover (2026-09-06, the file-size
ratchet's third cut). They still reach every subject through `cf_producer`'s re-exports — as `P.<name>`,
unchanged — which is what proves the extraction changed nothing a caller can see, and the
extraction-parity golden that pins it stays in `cf_producer_test.py` beside the fixtures.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# `P` is the HUB, deliberately: these tests assert the names still resolve there.
from agents.training import cf_producer as P


class TestLabelRows:
    def _row(self, **over):
        from agents.training.obs_materializer import RecordDecision
        obs = np.arange(8, dtype=np.float32)
        d = RecordDecision(index=3, turn=11, action=7, choice="move icebeam",
                           mask=np.ones(11, dtype=np.int8), obs=obs)
        kw = dict(record_path="/r/x_reconstruction.json", decision=d, wins=5, n=8,
                  step=24_000_000, surprise=0.4, entropy=0.6, score=0.61, win_prob=0.8)
        kw.update(over)
        return P.label_row(**kw)

    def test_every_row_names_the_ecology_it_was_measured_in(self):
        """THE ECOLOGY DECISION, enforced: a training record names no opponent, so the row must
        say `self_current` and never a bot name it cannot verify."""
        r = self._row()
        assert r["opponent"] == P.OPPONENT_LABEL == "self_current"
        assert r["label_regime"] == "self_current_stochastic_both_sides"

    def test_the_row_is_the_shared_v1_schema(self):
        r = self._row()
        assert r["schema"] == 1 and r["kind"] == "mc_winprob"
        assert r["label"] == pytest.approx(5 / 8)
        assert r["n_rollouts"] == 8 and r["policy_step"] == 24_000_000
        assert r["wilson_lo"] <= r["label"] <= r["wilson_hi"]
        assert r["obs_npz"] is None and r["obs_inline"]

    def test_the_row_carries_head_Bs_SINGLE_OUTCOME_stream(self):
        """gen3_cf_twin_heads_v1. `outcome_label` is the RECORDED battle's realized outcome — the
        SAME quantity the on-policy BCE eats, on the states the sampler selected. That identity is
        what makes B−A a read of COVERAGE alone; a row that shipped anything else there would make
        the contrast measure two things at once."""
        assert self._row(outcome_label=0.0)["outcome_label"] == 0.0
        assert self._row(outcome_label=1.0)["outcome_label"] == 1.0
        assert self._row(outcome_label=0.5)["outcome_label"] == 0.5      # the turn cap / a tie
        # ABSENT is a first-class value: an older consumer ignores it, a newer one supervises
        # nothing extra rather than being handed a fabricated 0.
        assert self._row()["outcome_label"] is None

    def test_the_mc_return_stream_is_written_ONLY_when_it_was_measured(self):
        """A `null` and an absent key mean the same thing to the buffer — but writing the reward
        provenance unconditionally would imply a measurement that was not taken, and the digest is
        the one field a reader uses to decide whether the number is theirs."""
        bare = self._row()
        assert "mc_return" not in bare and "reward_sha1" not in bare
        got = self._row(mc_return=-2.5, mc_return_n=8, reward_sha1="deadbeef",
                        reward_composition="1 TERMINAL + 7 PBRS + 1 BIAS (x)")
        assert got["mc_return"] == pytest.approx(-2.5) and got["mc_return_n"] == 8
        assert got["reward_sha1"] == "deadbeef"
        # The human line rides beside the digest for the reason the launch banner exists: a hex
        # digest says two rewards DIFFER and nothing about how.
        assert "PBRS" in got["reward_composition"]

    def test_the_new_streams_do_not_disturb_the_v1_schema_version(self):
        """`schema` is a REFUSAL gate: a consumer skips every row whose version it does not know.
        Bumping it for additive-optional fields would make a new producer's output unreadable by an
        existing trainer — the opposite of backward compatible."""
        assert self._row(outcome_label=1.0, mc_return=1.0, mc_return_n=4,
                         reward_sha1="x")["schema"] == 1

    def test_the_obs_digest_is_of_the_bytes_actually_shipped(self):
        import base64
        r = self._row()
        arr = np.frombuffer(base64.b64decode(r["obs_inline"]), dtype=np.float32)
        from agents.training.cf_audit import obs_digest
        assert obs_digest(arr) == r["obs_sha1"]

    def test_the_sampler_and_its_score_ride_on_every_row(self):
        r = self._row()
        assert r["sampler_version"] == P.SAMPLER_VERSION
        assert r["priority"]["critic_surprise"] == pytest.approx(0.4)
        assert r["priority"]["win_prob"] == pytest.approx(0.8)

    def test_a_headless_checkpoints_rows_say_win_prob_none_not_zero(self):
        assert self._row(win_prob=None)["priority"]["win_prob"] is None

    def test_a_batch_is_a_new_file_written_atomically(self, tmp_path):
        rows = [self._row(), self._row()]
        p1 = P.write_label_batch(str(tmp_path), rows, step=100, seq=1)
        p2 = P.write_label_batch(str(tmp_path), rows, step=100, seq=2)
        assert os.path.basename(p1) == "labels_cf_producer_100_1.jsonl"
        assert p1 != p2, "a batch must never reuse a name — the buffer keys offsets on (name,inode)"
        assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
        assert len(open(p1).read().strip().split("\n")) == 2
