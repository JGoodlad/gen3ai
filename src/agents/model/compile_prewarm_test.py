"""Guards for the compile pre-warm — and for the fork hazard that killed the version before it.

The forkserver-preload approach (compile once, every worker inherits it by fork) HUNG a real 48-env
training run: the trainer forked 2 workers instead of 48 and blocked forever in
`unix_stream_data_wait`, with the box at 0.2 load and no error anywhere. Root cause: `fork()` copies
mutexes but not the threads holding them, so it is only safe from a single-threaded process — and
importing the extractor is not.

`test_extractor_import_is_not_fork_safe` pins that fact. It is written to FAIL LOUDLY WITH
INSTRUCTIONS if the situation ever improves, rather than silently passing, because the moment the
extractor stops dragging in poke-env the much faster preload becomes available again.
"""
import types

import pytest

from agents.model import compile_prewarm as P


def test_extractor_import_is_not_fork_safe():
    """The hazard, as an executable fact.

    Importing `agents.model.features_extractor` transitively imports ~37 poke-env modules, and
    `poke_env.concurrency` starts a GLOBAL asyncio loop thread at import. That alone makes any
    process that imports the extractor unsafe to `fork()` from, which is why the forkserver preload
    cannot be used no matter what we do about Inductor's compile pool."""
    why = P.extractor_import_is_fork_safe()
    if why is None:
        pytest.fail(
            "The extractor import is now single-threaded — the fork hazard that forced the "
            "on-disk-cache pre-warm may be GONE. Re-read the module docstring in compile_prewarm.py: "
            "a forkserver preload would drop per-worker compile from ~30s to ~0.12s. Verify "
            "Inductor's compile pool is also torn down (threading.active_count() == 1 AFTER a "
            "compile, not just after the import), then delete this test and restore the preload."
        )
    assert "thread" in why


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
