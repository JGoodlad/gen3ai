"""Library functions that set torch globals for their own work give the caller its own back.

`untaught_meter.play_cells` and `harvest.score_candidates` cap the intra-op
thread count; `policy_spectrum.reader.load_checkpoint` used to set it AND turn TF32 off on CUDA and
return with both still set. A caller running two of them in one process (a test, a notebook, a
multi-meter CLI) then ran under whichever set last — Lane E's F-LJ-6 was exactly that class: a leaked
thread count moved a fresh init by rounding and flipped an exact tie. Each test sets a thread count
nobody else uses (3), makes the function fail right AFTER its old set point, and asserts the count is
still 3 — every one FAILS on a revert."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from utils.torch_state_guard import restores_torch_globals, torch_globals

SENTINEL_THREADS = 3


def _assert_restored(call) -> None:
    with torch_globals(num_threads=SENTINEL_THREADS):
        with pytest.raises(Exception):
            call()
        assert torch.get_num_threads() == SENTINEL_THREADS


def test_the_decorator_restores_on_return_and_on_raise():
    @restores_torch_globals
    def ok():
        torch.set_num_threads(1)
        return "v"

    @restores_torch_globals
    def boom():
        torch.set_num_threads(1)
        raise RuntimeError("x")

    with torch_globals(num_threads=SENTINEL_THREADS):
        assert ok() == "v" and torch.get_num_threads() == SENTINEL_THREADS
        with pytest.raises(RuntimeError):
            boom()
        assert torch.get_num_threads() == SENTINEL_THREADS


def test_play_cells_restores_the_thread_count(tmp_path):
    from agents.training.untaught_meter import play_cells

    opp = SimpleNamespace(zip_path=str(tmp_path / "missing.zip"), config_path=str(tmp_path / "missing.json"))
    _assert_restored(lambda: play_cells([], [], opp, games_per_team=1))


def test_score_candidates_restores_the_thread_count(tmp_path):
    from main.harvest import score_candidates

    _assert_restored(lambda: score_candidates(str(tmp_path / "missing.zip"), [], models_root=str(tmp_path),
                                              verbose=False))


def test_policy_spectrum_load_checkpoint_sets_no_global(tmp_path):
    from main.policy_spectrum.reader import load_checkpoint

    _assert_restored(lambda: load_checkpoint(tmp_path / "missing.zip"))


def test_policy_spectrum_inference_globals_is_scoped():
    from main.policy_spectrum.reader import inference_globals

    with torch_globals(num_threads=SENTINEL_THREADS):
        before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        with inference_globals(2, "cuda"):
            assert torch.get_num_threads() == 2
            assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == (False, False)
        assert torch.get_num_threads() == SENTINEL_THREADS
        assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == before
