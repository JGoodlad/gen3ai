"""T25 perf levers B / C (`gen3_r1_inductor_presets_v1`): R1's named Inductor option presets. The default passes
NO options (the production compile, byte for byte); a preset reaches R1's `torch.compile` and only it; every key is
a real Inductor config of this torch; an unknown name refuses. CPU, dynamo's `eager` backend."""
from __future__ import annotations

import pytest
import torch

from agents.model import compile_control as cc
from agents.model import compile_regions as cr


def test_every_preset_key_is_a_real_inductor_config():
    import torch._inductor.config as ic
    for name, opts in cr.R1_INDUCTOR_PRESETS.items():
        for key, val in opts.items():
            obj = ic
            for part in key.split("."):
                assert hasattr(obj, part), f"preset {name}: {key} is not an Inductor config of torch {torch.__version__}"
                obj = getattr(obj, part)
            assert type(obj) is type(val), (name, key)


def test_presets_merge_with_plus_and_unknown_names_refuse():
    assert cr.r1_inductor_options("default") == {}
    assert cr.r1_inductor_options("coordesc+combo") == {"coordinate_descent_tuning": True, "combo_kernels": True}
    with pytest.raises(ValueError, match="unknown R1 Inductor preset"):
        cr.r1_inductor_options("coordesc+turbo")


@pytest.fixture
def learner():
    from agents.training import learner_golden as LG
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    model = LG.build_learner()
    try:
        yield model
    finally:
        cr.uninstall(model)
        cc._reset_control_for_tests()
        torch._dynamo.reset()


@pytest.mark.parametrize("preset,expect", [("default", None), ("coordesc+cudagraphs",
                                            {"coordinate_descent_tuning": True, "triton.cudagraphs": True})])
def test_install_passes_the_preset_to_R1s_compile_only(learner, monkeypatch, preset, expect):
    seen = []
    real = torch.compile

    def spy(fn, **kw):
        seen.append(dict(kw))
        return real(fn, **kw)

    monkeypatch.setattr(torch, "compile", spy)
    if preset != "default":
        learner._r1_inductor_preset = preset
    cc.control().install()
    cr.install(learner, backend="eager")
    assert len(seen) == 1
    assert seen[0].get("options") == expect
    assert ("options" in seen[0]) == (expect is not None)
