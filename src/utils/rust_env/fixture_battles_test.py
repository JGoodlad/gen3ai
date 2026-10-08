"""`fixture_battles`: the Rust env core's reproducible battles — the same rows every call, real rows of the right
width, and the compile parity fixture can be rebuilt from them (P6 of the poke-env retirement)."""
import numpy as np
import pytest

pytestmark = [pytest.mark.sim, pytest.mark.integration]


@pytest.fixture(scope="module")
def lib():
    from utils.rust_env import ffi
    from utils.rust_env.build import ensure_built

    ensure_built("selfcheck")
    return ffi.load(ffi.default_path("selfcheck"), nan_poison=True)


def test_two_calls_play_byte_identical_rows(lib):
    from utils.rust_env.fixture_battles import play_rows

    a = play_rows(3, seed=11, lib=lib)
    b = play_rows(3, seed=11, lib=lib)
    assert len(a) == len(b) == 3
    for (oa, ma), (ob, mb) in zip(a, b):
        assert oa.tobytes() == ob.tobytes() and ma.tobytes() == mb.tobytes()
    c = play_rows(3, seed=12, lib=lib)
    assert any(x[0].tobytes() != y[0].tobytes() for x, y in zip(a, c)), "the seed does not reach the battles"


def test_rows_are_real_training_rows(lib):
    from agents.battle.core_obs import obs_dim
    from utils.rust_env.fixture_battles import spread_rows

    OBS_DIM = obs_dim()

    obs, mask = spread_rows(24, n_battles=3, seed=5, lib=lib)
    assert obs.shape == (24, OBS_DIM) and obs.dtype == np.float32
    assert np.isfinite(obs).all(), "an unwritten (NaN-poisoned) cell reached a fixture row"
    assert mask.shape == (24, 11) and mask.any(axis=1).all()
    assert len({r.tobytes() for r in obs}) > 20, "the rows do not spread across the battles"


def test_the_compile_parity_fixture_rebuilds_from_the_core(lib, monkeypatch):
    """`compile_parity_fixture.build_rows` (the `--write` path) yields a fixture `load_parity_rows` accepts."""
    from agents.model import compile_parity_fixture as F
    from utils.rust_env import fixture_battles

    real = fixture_battles.play_rows
    monkeypatch.setattr(fixture_battles, "play_rows", lambda n, **kw: real(n, **{**kw, "lib": lib}))
    obs, mask = F.build_rows(n_rows=F.N_ROWS)
    assert obs.shape[0] == F.N_ROWS and mask.shape == (F.N_ROWS, 11)
    with np.load(F.FIXTURE_PATH) as z:
        assert z["obs"].shape[1] == obs.shape[1], "the committed fixture's width differs from the core's rows"


def test_self_play_answers_both_sides_with_the_policy(lib):
    """`self_play` (what `churn_probe.collect_probe_states` uses: a checkpoint against itself): p2 is EXTERNAL and the
    policy answers it — a policy that sees p2's decisions too, battles that end, and the same rows every call."""
    from utils.rust_env.fixture_battles import play_rows

    seen = []

    def lowest_legal(obs, mask):
        seen.append(len(mask))
        return mask.argmax(axis=1)

    a = play_rows(2, seed=3, policy=lowest_legal, self_play=True, lib=lib)
    b = play_rows(2, seed=3, policy=lowest_legal, self_play=True, lib=lib)
    assert [x[0].tobytes() for x in a] == [x[0].tobytes() for x in b]
    assert max(seen) >= 2, "p2's decisions never reached the policy (both sides open at once)"
    with pytest.raises(ValueError, match="self_play needs a policy"):
        play_rows(1, seed=3, self_play=True, lib=lib)
