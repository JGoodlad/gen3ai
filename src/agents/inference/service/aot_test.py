"""CPU tests of the ``aot`` backend's build config (gen3_aot_real_autotune_inputs_v1).

The card proof is ``service_cuda_test.py::test_aot_backend_one_package_per_bucket_serves_every_slot``
(GPU, slow); these pin the cause on CPU: torch's compile-time autotune feeds randn to every float buffer,
a scatter over a float-decoded id refuses a negative where indexing wraps it, and the build turns that
autotune off so the kernels tune on the export's real example inputs; that tuning pass RUNS the graph on
those inputs, so the build carries no autograd (gen3_aot_build_no_grad_v1), and a real CPU build serves eager.
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
        raw = self.lin(hot)
        return raw.masked_fill(~mask, float("-inf")).log_softmax(-1), raw.sum(-1)


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


class _Abort(Exception):
    pass


def test_aot_build_hands_the_compile_no_autograd(monkeypatch, tmp_path: Path):
    """THE REAL torch path (export -> `aoti_compile_and_package` -> `aot_compile`) up to `compile_fx_aot`, with
    requires-grad Parameters as the weights: Inductor receives grad mode OFF and example inputs that require no
    grad. The real-input tuning pass runs the Python-wrapper graph on exactly these inputs, and on the card a
    requires-grad weight under grad mode made its `extern_kernels.addmm(..., out=)` raise (c0d0c9714). On CPU the
    cpp wrapper is one-pass, so the raise itself is card-only; this pins its two preconditions. FAILS ON REVERT."""
    import torch._inductor.compile_fx as compile_fx

    seen = {}

    def spy(gm, example_inputs, *a, **k):
        seen["grad_enabled"] = torch.is_grad_enabled()
        seen["requires_grad"] = [i for i, t in enumerate(example_inputs)
                                 if isinstance(t, torch.Tensor) and t.requires_grad]
        seen["n"] = len(example_inputs)
        raise _Abort

    monkeypatch.setattr(aot, "cuda_home", lambda: str(tmp_path))
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setattr(compile_fx, "compile_fx_aot", spy)
    module = _ScatterOfAnObsId().eval()
    assert all(p.requires_grad for p in module.parameters()), "precondition: the weights require grad"
    obs = torch.tensor([[3.0, 0.5], [18.0, -0.2]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    assert torch.is_grad_enabled(), "precondition: the caller runs with grad mode on"
    with pytest.raises(_Abort):
        aot.build_package(module, obs, mask, tmp_path / "p.pt2")
    assert seen == {"grad_enabled": False, "requires_grad": [], "n": 4}
    assert all(p.requires_grad for p in module.parameters())          # the module itself is untouched


@pytest.mark.slow
def test_a_real_cpu_aot_build_serves_eager(monkeypatch, tmp_path: Path):
    """A WHOLE build on CPU (export, Inductor's cpp wrapper, the C++ compile, the package load) of a module with
    requires-grad weights and a scatter over a decoded id, then a call through the weights-as-inputs runner:
    it equals eager, on rows the build never saw."""
    try:
        aot.check_available()
    except Exception as exc:  # noqa: BLE001 - the typed refusal names the missing headers
        pytest.skip(f"the AOT build needs the gen3ai_torch28 CUDA headers: {exc}")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setenv("CUDA_HOME", aot.cuda_home())
    module = _ScatterOfAnObsId().eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        for p in module.parameters():
            p.data.normal_()
    obs = torch.tensor([[3.0, 0.5], [18.0, -0.2], [0.0, 1.0]])
    mask = torch.tensor([[True, True, False], [True, True, True], [False, True, True]])
    runner, names = aot.build_package(module, obs, mask, tmp_path / "p.pt2")
    other = torch.tensor([[7.0, 0.0], [11.0, 2.0], [1.0, -1.0]])
    with torch.no_grad():
        logp, value, greedy = runner(other, mask, *aot.weight_tensors(module, names))
        e_logp, e_value = module(other, mask)
    fin = torch.isfinite(e_logp)
    assert torch.equal(torch.isfinite(logp), fin)
    assert float((logp - e_logp)[fin].abs().max()) < 1e-5
    assert float((value - e_value).abs().max()) < 1e-5
    assert torch.equal(greedy, e_logp.argmax(-1))
