"""The Metamon peer's CPU attention swap keeps each model's OWN attention function.

Found 2026-09-28 wiring ``metamon:Kakuna`` (X22a): ``superkazam.gin`` binds
``TformerTrajEncoder.attention_type = @FlashAttention`` itself, and ``amago.cli_utils.use_config``
parses gin FILES after the override DICT, so the old swap silently lost and the peer died on the
missing flash-attn wheel. The same file sets ``FlashAttention.window_size = (96, 0)``, so a plain
``VanillaAttention`` would have been a DIFFERENT function (full 128-step context) — see the module
docstring of ``peer_scripts/metamon_side.py``.

Three properties, each run under the METAMON interpreter (the script is never imported by this
repo's code — two ``poke_env`` packages must not meet):

1. ``WindowedVanillaAttention`` equals a naive sliding-window reference, both on the full-sequence
   path and step by step through the KV cache (including past the window);
2. ``Kakuna``'s parsed config resolves to the WINDOWED class at left = 96;
3. ``SyntheticRLV2`` (whose gin binds neither key) still resolves to plain ``VanillaAttention``.

**It SKIPS with a named reason** when the Metamon interpreter or checkout is absent (every box
but this one), exactly like ``anchors_integration_test.py``.
"""
from __future__ import annotations

import json
import subprocess
import textwrap

import pytest

from main.anchors.config import load_config
from utils.paths import src_path

pytestmark = pytest.mark.integration

_SCRIPT = src_path("main", "anchors", "peer_scripts", "metamon_side.py")


def _metamon_or_skip():
    cfg = load_config().opponent("metamon")
    for what, path in (("interpreter", cfg.python), ("checkout", cfg.checkout)):
        if not path.exists():
            pytest.skip(f"no Metamon {what} at {path} (opponents.metamon in "
                        "designs/ops/anchors.json) — this box cannot run the peer")
    return cfg


def _run(cfg, body: str) -> dict:
    code = textwrap.dedent(f"""
        import importlib.util, json, sys
        spec = importlib.util.spec_from_file_location("metamon_side", {str(_SCRIPT)!r})
        ms = importlib.util.module_from_spec(spec); spec.loader.exec_module(ms)
    """) + textwrap.dedent(body)
    env = {"PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
           "PATH": "/usr/bin:/bin", "METAMON_CACHE_DIR": str(cfg.cache_dir)}
    proc = subprocess.run([str(cfg.python), "-c", code], cwd=str(cfg.checkout), env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_windowed_attention_matches_a_naive_sliding_window_reference() -> None:
    cfg = _metamon_or_skip()
    out = _run(cfg, """
        import math, torch
        torch.manual_seed(0)
        LEFT, L, H, E = 3, 9, 2, 4
        W = ms.make_windowed_vanilla_attention(LEFT)(causal=True, dropout=0.0).eval()
        qkv = torch.randn(1, L, 3, H, E)
        q, k, v = qkv.unbind(2)
        ref = torch.empty(1, L, H, E)
        for i in range(L):
            lo = max(0, i - LEFT)
            s = torch.einsum("he,jhe->hj", q[0, i], k[0, lo:i + 1]) / math.sqrt(E)
            ref[0, i] = torch.einsum("hj,jhe->he", s.softmax(-1), v[0, lo:i + 1])
        mask = torch.triu(torch.ones((1, 1, L, L), dtype=torch.bool), diagonal=1)
        full = W._forward_without_cache(qkv, mask)
        kc = torch.zeros(1, 16, H, E); vc = torch.zeros(1, 16, H, E)
        steps = [W._inference_with_cache(qkv[:, i:i + 1], kc, vc,
                                         torch.tensor([i], dtype=torch.int32))[0, 0]
                 for i in range(L)]
        cached = torch.stack(steps)[None]
        print(json.dumps({"full": float((full - ref).abs().max()),
                          "cached": float((cached - ref).abs().max())}))
    """)
    assert out["full"] < 1e-5 and out["cached"] < 1e-5, out


@pytest.mark.parametrize("agent, want_class, want_window", [
    ("Kakuna", "WindowedVanillaAttention", [96, 0]),
    ("SyntheticRLV2", "VanillaAttention", [-1, -1]),
])
def test_each_model_resolves_to_its_own_attention(agent, want_class, want_window) -> None:
    cfg = _metamon_or_skip()
    out = _run(cfg, f"""
        import gin, amago.cli_utils
        from metamon.rl.pretrained import get_pretrained_model
        rec = {{}}
        ms.install_cpu_attention("vanilla", rec)
        m = get_pretrained_model({agent!r})
        amago.cli_utils.use_config(m.base_config, [m.model_gin_config_path,
                                   m.train_gin_config_path], finalize=False)
        bound = gin.query_parameter("traj_encoders.TformerTrajEncoder.attention_type")
        rec["bound"] = getattr(bound, "__name__", str(bound))
        print(json.dumps(rec))
    """)
    assert out["attention"] == want_class and out["bound"] == want_class, out
    assert out["attention_window"] == want_window, out
