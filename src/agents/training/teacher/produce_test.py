"""Unit tests for produce_correction's 3-tier gate (search → confirm → CI-gate) with a fake session —
no torch, no bridge. The real search/confirm pipeline is the prober's, already tested end-to-end."""

import numpy as np
import pytest

from agents.training.teacher.produce import produce_correction
from agents.training.teacher.selection import Candidate

_OBS_DIM = 12


def _cand():
    return Candidate(summary_path="/x/loss_0_summary.json", recon_path="/x/loss_0_reconstruction.json",
                     inv_index=3, turn=7, opponent="sentinel_1", step=100, anchor_delta=20.0,
                     verdict="MISTAKE")


class _FakeModel:
    def __init__(self, argmax=99):
        self._argmax = argmax            # 99 = never equals A* (so "already_known" doesn't trip)

    def action_probs_batch(self, obs, masks):
        p = np.zeros((len(obs), 11)); p[:, self._argmax % 11] = 1.0
        return p


class _FakeSession:
    """Returns a canned better_line dict; mimics _battle/_npz/_model_for for the obs/mask/staleness."""

    def __init__(self, better_line_out, argmax=99):
        self._out = better_line_out
        self._model = _FakeModel(argmax)

    def better_line(self, *a, **k):
        return self._out

    def _battle(self, _id):
        return object()

    def _npz(self, _b):
        return {"obs": np.tile(np.arange(_OBS_DIM, dtype=np.float32), (10, 1))}

    def _model_for(self, _b):
        return self._model, None


def _bl(best_action=6, win_rate=0.7, ci_low=0.55, candidates=(0, 2, 6, 8)):
    return {
        "candidates": [{"action": a} for a in candidates],
        "best_alternative": {"action": best_action, "label": "spore"},
        "confirm": {"win_rate": win_rate, "ci": [ci_low, 0.9]},
    }


def test_gate_pass_returns_correction():
    corr, st = produce_correction(
        _FakeSession(_bl(win_rate=0.7, ci_low=0.55)), _cand(),
        opponent_ckpt="/snap.zip", opponent_source="ckpt", confirm_rollouts=8)
    assert st == "ok" and corr is not None
    assert corr.better_action == 6
    assert abs(corr.advantage - 0.7) < 1e-6          # confirmed win-rate − played 0.0
    assert corr.action_mask.tolist() == [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]   # legal {0,2,6,8}


def test_unresolved_opponent_is_skipped_never_approximated():
    corr, st = produce_correction(
        _FakeSession(_bl()), _cand(), opponent_ckpt=None, opponent_source="unresolved")
    assert corr is None and st == "unresolved"       # NOT a self_model_approx fallback (soundness)


def test_no_better_line():
    out = _bl(); out["best_alternative"] = None
    corr, st = produce_correction(_FakeSession(out), _cand(), opponent_ckpt="/s.zip",
                                  opponent_source="ckpt")
    assert corr is None and st == "no_better_line"


def test_ci_gate_fails_when_wilson_low_not_above_played():
    # win_rate looks good but the Wilson LOWER bound is ≤ 0 (the played loss rate) → NOT verified.
    corr, st = produce_correction(
        _FakeSession(_bl(win_rate=0.6, ci_low=0.0)), _cand(),
        opponent_ckpt="/s.zip", opponent_source="ckpt")
    assert corr is None and st == "gate_failed"


def test_confirm_error_is_skipped():
    out = _bl(); out["confirm"] = {"error": "node died"}
    corr, st = produce_correction(_FakeSession(out), _cand(), opponent_ckpt="/s.zip",
                                  opponent_source="ckpt")
    assert corr is None and st == "confirm_failed"


def test_already_known_when_model_argmaxes_astar():
    # the frozen model already greedily plays A* (action 6) → nothing to teach.
    corr, st = produce_correction(
        _FakeSession(_bl(best_action=6), argmax=6), _cand(),
        opponent_ckpt="/s.zip", opponent_source="ckpt")
    assert corr is None and st == "already_known"
