"""CPU tests of the ``aot`` backend's build config (gen3_aot_real_autotune_inputs_v1).

The card proof is ``service_cuda_test.py::test_aot_backend_one_package_per_bucket_serves_every_slot``
(GPU, slow); these pin the cause on CPU: torch's compile-time autotune feeds randn to every float buffer,
a scatter over a float-decoded id refuses a negative where indexing wraps it, and the build turns that
autotune off so the kernels tune on the export's real example inputs.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from agents.inference.service import aot


class _ScatterOfAnObsId(torch.nn.Module):
    """The move-resolution op's shape in miniature: a float obs column decoded to an id, then one-hot by
    SCATTER (`move_resolution.py`: ``type_rev.scatter(2, t2[..., None], 1.0)``)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(19, 3)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        ids = obs[:, 0].long()
        hot = obs.new_zeros(obs.shape[0], 19).scatter(1, ids[:, None], 1.0)
        logits = self.lin(hot).masked_fill(~mask, float("-inf"))
        return logits.log_softmax(-1), logits.sum(-1)


def test_aot_build_turns_off_compile_time_autotune_inside_the_compile(monkeypatch, tmp_path: Path):
    """Inside `aoti_compile_and_package` the explicit setting is False (not torch's unset None, which the
    cpp wrapper resolves to True on a Triton box) and the cpp-wrapper config resolves it to False; the
    process's own setting is handed back untouched. FAILS ON REVERT of `aot.AOT_BUILD_CONFIG`."""
    import torch._inductor as inductor
    import torch._inductor.config as inductor_config
    from torch._inductor.compile_fx import get_cpp_wrapper_config

    seen = {}

    def fake_compile(ep, package_path):
        seen["explicit"] = inductor_config.triton.autotune_at_compile_time
        seen["resolved"] = get_cpp_wrapper_config()["triton.autotune_at_compile_time"]
        return package_path

    before = inductor_config.triton.autotune_at_compile_time
    monkeypatch.setattr(aot, "cuda_home", lambda: str(tmp_path))      # no CUDA headers needed on CPU
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))                    # build_package's setdefault: no leak
    monkeypatch.setattr(inductor, "aoti_compile_and_package", fake_compile)
    monkeypatch.setattr(inductor, "aoti_load_package", lambda pkg: pkg)
    obs = torch.tensor([[3.0, 0.5], [18.0, -0.2]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    runner, names = aot.build_package(_ScatterOfAnObsId().eval(), obs, mask, tmp_path / "p.pt2")
    assert runner == str(tmp_path / "p.pt2") and names == ["lin.weight", "lin.bias"]
    assert seen == {"explicit": False, "resolved": False}
    assert inductor_config.triton.autotune_at_compile_time == before


def test_random_example_values_break_a_scatter_of_a_decoded_id_but_not_an_index():
    """The mechanism the config removes, on CPU: torch's autotune example value for a float buffer is
    randn, its ``.long()`` holds negatives, indexing wraps them (in range) and scatter refuses them."""
    from torch._inductor.select_algorithm import AlgorithmSelectorCache

    with torch.random.fork_rng(devices=[]):            # the global generator is restored on exit
        torch.manual_seed(0)
        buf = AlgorithmSelectorCache.generate_example_value((48,), (1,), "cpu", torch.float32, 0)
    ids = buf.long()
    assert bool((ids < 0).any()), "precondition: seed 0's 48 randn draws hold a value <= -1"
    assert bool((ids.abs() < 19).all()), "precondition: every draw decodes inside +-19"
    table = torch.arange(19.0)
    assert table[ids].shape == (48,)                   # indexing wraps a negative id: no error
    with pytest.raises(RuntimeError, match="out of bounds"):
        torch.zeros(48, 19).scatter(1, ids[:, None], 1.0)
