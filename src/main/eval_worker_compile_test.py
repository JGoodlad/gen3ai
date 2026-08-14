"""The eval worker must compile BOTH the trainee and its neural opponents.

Why this file exists. `--compile-opponents` was wired into `eval_worker` and looked verified by a
bridge-backed fuzz run — but that plan contained only SCRIPTED BOTS, so the opponent half of the
wiring (`_get_opponent_model` -> `maybe_compile_extractor`) never executed. Bots are the default in
every quick check, so that half is exactly the part that stays silently uncovered.

`eval_sharding_fuzz_test.py --compile --neural-opponent` covers it end to end with real battles
(verified: `eval-opp:final_model.zip returned=True forward_patched=True`). These are the FAST
mock-level guards so a regression fails the normal unit suite rather than waiting for someone to run
the fuzz with the right flags.
"""
import types

import main.eval_worker as ew


def _spy():
    calls = []

    def fn(model, enabled, label="opponent", **kw):
        calls.append({"label": label, "enabled": enabled, **kw})
        return bool(enabled)

    return fn, calls


def test_opponent_load_compiles_and_is_labelled(monkeypatch):
    spy, calls = _spy()
    monkeypatch.setattr(ew, "maybe_compile_extractor", spy)
    cache = {}
    sentinel = object()
    got = ew._get_opponent_model(cache, "/pool/snapshot_000000500000.zip", lambda: sentinel,
                                 compile_extractor=True, device="cpu")
    assert got is sentinel
    assert len(calls) == 1
    assert calls[0]["label"] == "eval-opp:snapshot_000000500000.zip"
    assert calls[0]["enabled"] is True
    assert calls[0]["hide_cuda"] is True, "a CPU eval worker must not take a CUDA context"


def test_opponent_compile_is_paid_once_per_path(monkeypatch):
    """The compile rides the model cache, so a finely-sharded opponent must not recompile per shard."""
    spy, calls = _spy()
    monkeypatch.setattr(ew, "maybe_compile_extractor", spy)
    cache = {}
    for _ in range(4):
        ew._get_opponent_model(cache, "/pool/a.zip", lambda: object(),
                               compile_extractor=True, device="cpu")
    assert len(calls) == 1, f"compiled {len(calls)} times for one opponent path"


def test_opponent_is_not_compiled_when_the_flag_is_off(monkeypatch):
    spy, calls = _spy()
    monkeypatch.setattr(ew, "maybe_compile_extractor", spy)
    ew._get_opponent_model({}, "/pool/a.zip", lambda: object(),
                           compile_extractor=False, device="cpu")
    assert calls and calls[0]["enabled"] is False


def test_cuda_is_not_hidden_for_a_gpu_eval_worker(monkeypatch):
    """`--eval-device cuda` runs eval on the GPU; hiding the device there would be wrong."""
    spy, calls = _spy()
    monkeypatch.setattr(ew, "maybe_compile_extractor", spy)
    ew._get_opponent_model({}, "/pool/a.zip", lambda: object(),
                           compile_extractor=True, device="cuda")
    assert calls[0]["hide_cuda"] is False


def test_play_unit_threads_the_flag_through(monkeypatch):
    """`_play_unit` receives `compile_extractor` positionally from `_run`'s claim loop; if that
    argument is ever dropped the opponents silently stop compiling."""
    import inspect
    params = list(inspect.signature(ew._play_unit).parameters)
    assert "compile_extractor" in params
    src = inspect.getsource(ew._run)
    assert "compile_extractor" in src, "_run must read the cfg key and pass it to _play_unit"
    assert 'cfg.get("compile_extractor"' in src


def test_trainee_is_compiled_too(monkeypatch):
    """The trainee plays EVERY eval game, so it is the hottest forward in the worker."""
    import inspect
    src = inspect.getsource(ew._run)
    assert 'label="eval-trainee"' in src
