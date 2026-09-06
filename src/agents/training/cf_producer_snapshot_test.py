"""Unit tests for the checkpoint + the loaded snapshot (`cf_producer_snapshot.py`).

Two contracts, neither of them arithmetic. WHICH checkpoint is freshest (a forced
save must not rank below every periodic one), and WHAT SIGNATURE reaches a
compiled graph — B=1 under compile, float32 masks on both paths, and the two-key
warm-up that pays the first re-trace up front. Each was a measured regression.

These moved out of `cf_producer_test.py` with the functions they cover (2026-09-06, the file-size
ratchet's third cut). They still reach every subject through `cf_producer`'s re-exports — as `P.<name>`,
unchanged — which is what proves the extraction changed nothing a caller can see, and the
extraction-parity golden that pins it stays in `cf_producer_test.py` beside the fixtures.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

# `P` is the HUB, deliberately: these tests assert the names still resolve there. `S` is the
# owning module, reached ONLY for `_warm_the_compiled_graph` — a private name the hub does not
# re-export, because a leading underscore says it has no callers outside its own file.
from agents.training import cf_producer as P
from agents.training import cf_producer_snapshot as S


class TestCheckpointResolution:
    def test_no_checkpoint_is_none_not_a_crash(self, tmp_path):
        assert P.resolve_latest_checkpoint(str(tmp_path)) is None

    def test_the_highest_step_wins_over_latest_txt(self, tmp_path):
        """`latest.txt` and the newest zip disagree exactly in the window between a checkpoint
        write and the pointer update; the higher step is the one whose weights are on disk."""
        ck = tmp_path / "checkpoints"
        ck.mkdir()
        (ck / "checkpoint_100_steps.zip").write_text("a")
        (ck / "checkpoint_900_steps.zip").write_text("b")
        (tmp_path / "latest.txt").write_text("checkpoints/checkpoint_100_steps.zip")
        path, step = P.resolve_latest_checkpoint(str(tmp_path))
        assert step == 900 and path.endswith("checkpoint_900_steps.zip")

    def test_a_legacy_run_root_checkpoint_is_found(self, tmp_path):
        (tmp_path / "checkpoint_42_steps.zip").write_text("a")
        path, step = P.resolve_latest_checkpoint(str(tmp_path))
        assert step == 42

    def test_a_dangling_latest_txt_does_not_hide_the_glob(self, tmp_path):
        ck = tmp_path / "checkpoints"
        ck.mkdir()
        (ck / "checkpoint_7_steps.zip").write_text("a")
        (tmp_path / "latest.txt").write_text("checkpoints/gone.zip")
        assert P.resolve_latest_checkpoint(str(tmp_path))[1] == 7

    def test_a_forced_checkpoints_step_parses(self):
        """SIGUSR1 writes `checkpoint_forced_<step:010d>_<HHMMSS>.zip` — a resumable checkpoint
        under a second name. Reading only the periodic form makes its step unparseable, and an
        unparseable step ranks BELOW every periodic zip in `_key`."""
        assert P.step_from_checkpoint_name("checkpoint_forced_0000060000_120000.zip") == 60000
        assert P.step_from_checkpoint_name("checkpoint_50000_steps.zip") == 50000
        assert P.step_from_checkpoint_name("final_model.zip") is None

    def test_a_NEWER_forced_checkpoint_beats_an_older_periodic_one(self, tmp_path):
        """The regression: an operator hits the launcher's `c` (force checkpoint) after the last
        periodic save. Before the fix the producer resolved the OLDER periodic zip and went on
        stamping its step — silently labelling against a snapshot it had already moved past."""
        ck = tmp_path / "checkpoints"
        ck.mkdir()
        (ck / "checkpoint_50000_steps.zip").write_text("a")
        (ck / "checkpoint_forced_0000060000_120000.zip").write_text("b")
        (tmp_path / "latest.txt").write_text(
            "checkpoints/checkpoint_forced_0000060000_120000.zip")
        path, step = P.resolve_latest_checkpoint(str(tmp_path))
        assert step == 60000, "a newer FORCED checkpoint must outrank an older periodic one"
        assert path.endswith("checkpoint_forced_0000060000_120000.zip")

    def test_a_forced_checkpoint_is_found_without_latest_txt(self, tmp_path):
        """It must be reachable by the GLOB too, not only through the pointer file — `latest.txt`
        is written after the zip, so there is a window in which it names the previous save."""
        ck = tmp_path / "checkpoints"
        ck.mkdir()
        (ck / "checkpoint_forced_0000012288_091921.zip").write_text("a")
        assert P.resolve_latest_checkpoint(str(tmp_path))[1] == 12288

    def test_an_older_forced_checkpoint_still_loses_to_a_newer_periodic_one(self, tmp_path):
        """The other direction, so the fix is a step comparison and not a name preference."""
        ck = tmp_path / "checkpoints"
        ck.mkdir()
        (ck / "checkpoint_forced_0000040448_092309.zip").write_text("a")
        (ck / "checkpoint_50000_steps.zip").write_text("b")
        path, step = P.resolve_latest_checkpoint(str(tmp_path))
        assert step == 50000 and path.endswith("checkpoint_50000_steps.zip")



class _RecordingPolicy:
    """A policy stand-in that REMEMBERS the shape and dtype of every forward it was handed.

    The whole point of the scoring changes is *which signature* reaches a compiled graph, so the
    test subject is the call record, not the numbers — and the numbers are checked too, because a
    chunked forward that changes the ranking would be a silent sampler change.
    """

    def __init__(self, *, win_head: bool = True) -> None:
        self.calls: list = []
        self.features_extractor = SimpleNamespace(last_win_prob_logits=None)
        self._win_head = win_head

    def get_distribution(self, obs):
        import torch as th
        o, m = obs["observation"], obs["action_mask"]
        self.calls.append({"n": int(o.shape[0]), "obs_dtype": o.dtype, "mask_dtype": m.dtype})
        # Deterministic per-row logits so a chunked pass and a batched one are comparable.
        logits = th.arange(o.shape[0], dtype=th.float32).unsqueeze(1) + th.arange(
            m.shape[1], dtype=th.float32).unsqueeze(0)
        if self._win_head:
            self.features_extractor.last_win_prob_logits = th.zeros(o.shape[0], 1)
        return SimpleNamespace(distribution=SimpleNamespace(logits=logits))


def _snapshot_with(policy, *, compiled: bool):
    return P.Snapshot("/ckpt.zip", 7, SimpleNamespace(policy=policy), None, compiled=compiled)


class TestScoreForwardSignature:
    def _inputs(self, n=5):
        obs = np.arange(n * 4, dtype=np.float32).reshape(n, 4)
        masks = np.ones((n, 3), dtype=np.int8)
        masks[:, 2] = 0
        return obs, masks

    def test_a_compiled_snapshot_scores_ONE_ROW_AT_A_TIME(self):
        """B=1 is the shape every rollout forwards at; a batched score would force a SECOND trace.

        Measured 2026-08-23 on the live checkpoint: with a batched score in front of them, the
        first label's rollouts cost 79 s against 3 s for the second — pure recompilation."""
        pol = _RecordingPolicy()
        obs, masks = self._inputs(5)
        _snapshot_with(pol, compiled=True).score(obs, masks)
        assert [c["n"] for c in pol.calls] == [1, 1, 1, 1, 1]

    def test_an_eager_snapshot_still_takes_the_single_batched_forward(self):
        """There is no graph to keep one signature for, so the cheap path stays the cheap path."""
        pol = _RecordingPolicy()
        obs, masks = self._inputs(5)
        _snapshot_with(pol, compiled=False).score(obs, masks)
        assert [c["n"] for c in pol.calls] == [5]

    def test_the_mask_reaches_the_graph_as_float32_not_int8(self):
        """A materialized mask is int8 and a live one is float32 — and dynamo guards on DTYPE as
        hard as on shape. That mismatch measured a 19.5 s re-trace on the first scored row."""
        import torch as th
        for compiled in (True, False):
            pol = _RecordingPolicy()
            obs, masks = self._inputs(3)
            assert masks.dtype == np.int8
            _snapshot_with(pol, compiled=compiled).score(obs, masks)
            assert {c["mask_dtype"] for c in pol.calls} == {th.float32}
            assert {c["obs_dtype"] for c in pol.calls} == {th.float32}

    def test_chunking_does_not_change_a_single_number(self):
        """The sampler ranks on these values, so the two paths must agree exactly."""
        obs, masks = self._inputs(6)
        wp_c, ent_c = _snapshot_with(_RecordingPolicy(), compiled=True).score(obs, masks)
        wp_e, ent_e = _snapshot_with(_RecordingPolicy(), compiled=False).score(obs, masks)
        assert np.allclose(ent_c, ent_e)
        assert np.allclose(wp_c, wp_e)

    def test_a_headless_checkpoint_reports_no_win_probs_through_either_path(self):
        for compiled in (True, False):
            obs, masks = self._inputs(4)
            wp, ent = _snapshot_with(
                _RecordingPolicy(win_head=False), compiled=compiled).score(obs, masks)
            assert wp is None and len(ent) == 4


class TestCompiledGraphWarmUp:
    def test_a_warm_up_that_raises_is_survivable_and_says_so(self, capsys):
        """It is a perf warm-up, not a gate: a model whose spaces are not what we assumed must
        cost the warm-up and nothing else."""
        broken = SimpleNamespace(observation_space={}, policy=SimpleNamespace())
        assert S._warm_the_compiled_graph(broken) >= 0.0
        assert "warm-up skipped" in capsys.readouterr().out

    def test_it_forwards_the_LIVE_signature_both_keys(self):
        """`maybe_compile_extractor` warms with `observation` ALONE; every real call also carries
        `action_mask`, and a dict's KEY SET is part of the guard — so warming with one key leaves
        the first real decision to re-trace (19.5 s, measured)."""
        seen = {}

        class _Space:
            def __init__(self, n):
                self.shape = (n,)

        def _get_distribution(obs):
            seen["keys"] = sorted(obs)
            seen["shapes"] = {k: tuple(v.shape) for k, v in obs.items()}

        model = SimpleNamespace(
            observation_space={"observation": _Space(9), "action_mask": _Space(11)},
            policy=SimpleNamespace(get_distribution=_get_distribution))
        S._warm_the_compiled_graph(model)
        assert seen["keys"] == ["action_mask", "observation"]
        assert seen["shapes"] == {"observation": (1, 9), "action_mask": (1, 11)}
