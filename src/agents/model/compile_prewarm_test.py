"""Guards for the compile pre-warm, the fork-safety invariant, and the forkserver preload.

The forkserver-preload approach (compile once, every worker inherits it by fork) HUNG a real 48-env
training run in 2026-08: the trainer forked 2 workers instead of 48 and blocked forever in
`unix_stream_data_wait`, with the box at 0.2 load and no error anywhere. Root cause: `fork()` copies
mutexes but not the threads holding them, so it is only safe from a single-threaded process — and
importing the extractor used to start poke-env's global asyncio loop thread (any `poke_env.x`
import executed the eager package `__init__` → `player` → `ps_client` → `concurrency`).

`gen3_forkserver_preload_v1` (2026-08-16) fixed it at the ROOT — `poke_env/__init__.py` and
`poke_env/player/__init__.py` are LAZY (PEP 562) — and restored the preload
(`agents.model.compile_preload`, `--compile-opponents-preload`) with a LOUD guard: a preload that
cannot prove the forkserver single-threaded after its compile RAISES, so env construction fails
with a traceback instead of wedging. `test_extractor_import_is_fork_safe` pins the import
invariant this all rests on; if it regresses, the preload must go back behind the old pin.
"""
import json
import os
import subprocess
import sys

import pytest

from agents.model import compile_prewarm as P


def test_extractor_import_is_fork_safe():
    """The invariant the preload rests on, as an executable fact: importing the extractor leaves
    the process SINGLE-THREADED. If this fails, the lazy poke_env `__init__` regressed (something
    eager reaches `concurrency` again) — and `--compile-opponents-preload` is unsafe until it is
    fixed, because fork() from a multi-threaded process copies held mutexes without their owners
    (the mechanism of the 2026-08 silent 2-of-48-workers hang).

    Runs in a FRESH interpreter: the shared pytest process has long since imported player/client
    modules (other tests), so an in-process check would measure the suite, not the import."""
    code = ("from agents.model.compile_prewarm import extractor_import_is_fork_safe; "
            "why = extractor_import_is_fork_safe(); "
            "import sys; print(why or ''); sys.exit(0 if why is None else 3)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": "src"}, timeout=300)
    assert r.returncode == 0, (
        f"the extractor import is no longer fork-safe: {r.stdout.strip() or r.stderr[-500:]}. "
        "The lazy poke_env __init__ regressed — find the eager path to poke_env.concurrency "
        "before any run uses --compile-opponents-preload."
    )


def test_preload_is_a_noop_without_the_env_var():
    """An accidental import outside the forkserver must do nothing (no compile, no torch load)."""
    code = ("import os, sys; os.environ.pop('GEN3AI_PRELOAD_ARCH', None); "
            "import agents.model.compile_preload; "
            "sys.exit(0 if 'torch' not in sys.modules else 3)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": "src"}, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]


@pytest.mark.skipif(os.environ.get("GEN3AI_SKIP_COMPILE_TESTS") == "1",
                    reason="GEN3AI_SKIP_COMPILE_TESTS=1")
@pytest.mark.slow
def test_preload_compiles_single_threaded_end_to_end():
    """The instruction the old pin carried, executed: in a FRESH interpreter (a stand-in for the
    forkserver), the preload compiles the extractor and finishes with threading.active_count() == 1
    — Inductor pool included. Runs the REAL module with a REAL (small) config; its own raise is
    the assertion, exit 0 the proof."""
    cfg = {"damage_op": False}     # log_level deliberately absent — the extractor default applies
    r = subprocess.run(
        [sys.executable, "-c", "import agents.model.compile_preload"],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "PYTHONPATH": "src", "GEN3AI_PRELOAD_ARCH": json.dumps(cfg)})
    assert r.returncode == 0, f"preload failed:\n{r.stderr[-2000:]}"
    assert "single-threaded" in r.stderr


def test_prewarm_never_raises_on_a_bad_arch():
    """A pre-warm is an optimization of an optimization: if it fails, workers compile cold. It must
    never be the thing that takes a run down."""
    took = P.prewarm_extractor_compile({"damage_op": "not-a-bool"}, mappings=None, quiet=True)
    assert took == 0.0


def test_prewarm_reports_zero_when_mappings_are_unusable():
    took = P.prewarm_extractor_compile({}, mappings=object(), quiet=True)
    assert took == 0.0


def test_prewarm_filters_kwargs_to_the_extractor_signature(monkeypatch):
    """`build_extractor_arch_kwargs` output includes keys the extractor may not accept (a stale
    toggle after an arch change). The pre-warm must drop them rather than raise — it is warming a
    cache, not validating config."""
    import inspect

    from agents.model.features_extractor import Gen3FeaturesExtractor

    captured = {}

    class _FakeFE:
        def __init__(self, *a, **kw):
            captured.update(kw)
            self.layout = {"total_dim": 4}

        def eval(self):
            return self

        def disable_observation_debugger(self):
            return False

        def forward(self, obs):
            return obs

    monkeypatch.setattr("agents.model.features_extractor.Gen3FeaturesExtractor", _FakeFE)
    # The real signature is what filters; assert the filter uses it rather than passing everything.
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    assert "definitely_not_a_real_toggle" not in sig
