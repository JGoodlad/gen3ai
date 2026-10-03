"""Unit tests for the churn probe's pure math (masked KL + roster grouping) — no models/bridge."""
import types

import numpy as np
import pytest

from agents.training.churn_probe import masked_kl


def _rows(dists, a=6):
    p = np.zeros((len(dists), a), np.float32)
    for i, d in enumerate(dists):
        p[i, :len(d)] = d
    return p


def test_masked_kl_identical_is_zero():
    p = _rows([[0.5, 0.3, 0.2], [0.9, 0.1]])
    mask = (p > 0).astype(np.float32)
    kl = masked_kl(p, p, mask)
    assert np.allclose(kl, 0.0, atol=1e-6)


def test_masked_kl_positive_when_different():
    pa = _rows([[0.9, 0.1]])
    pb = _rows([[0.5, 0.5]])
    mask = np.zeros_like(pa); mask[:, :2] = 1.0
    assert masked_kl(pa, pb, mask)[0] > 0.05


def test_masked_kl_asymmetric():
    pa = _rows([[0.99, 0.01]])
    pb = _rows([[0.5, 0.5]])
    mask = np.zeros_like(pa); mask[:, :2] = 1.0
    assert not np.isclose(masked_kl(pa, pb, mask)[0], masked_kl(pb, pa, mask)[0])


def test_masked_kl_ignores_illegal_remnants():
    """A numerically nonzero probability on an ILLEGAL action must not contribute."""
    pa = _rows([[0.6, 0.4, 1e-4]])
    pb = _rows([[0.6, 0.4, 1e-9]])
    mask = np.zeros_like(pa); mask[:, :2] = 1.0            # action 2 illegal
    assert np.isclose(masked_kl(pa, pb, mask)[0], 0.0, atol=1e-6)


def test_roster_keys_groups_by_our_species():
    """Rows with the same 6 our-side species (any slot order) share a key; different rosters don't."""
    import torch  # noqa: F401 — layout slicer needs torch
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    from agents.observation.constants import POKEMON_FULL_DIM
    from agents.model.features_extractor import slice_pokemon_categoricals  # noqa: F401
    from agents.training.churn_probe import roster_keys

    mappings = load_mappings()
    ek = Gen3ObservationEncoder(mappings).get_features_extractor_kwargs()
    layout, total = ek["layout"], ek["layout"]["total_dim"]
    sp_idx = None
    # plant species ids via the layout-driven slicer's own inverse: write id j at every slot's
    # species position (found by probing a marker through slice_pokemon_categoricals)
    probe = np.zeros((1, 12, POKEMON_FULL_DIM), np.float32)
    for cand in range(POKEMON_FULL_DIM):
        probe[:] = 0; probe[0, 0, cand] = 7
        got = slice_pokemon_categoricals(torch.tensor(probe), layout)["species_ids"][0, 0].item()
        if got == 7:
            sp_idx = cand
            break
    assert sp_idx is not None

    obs = np.zeros((3, total), np.float32)
    for slot in range(6):
        obs[0, slot * POKEMON_FULL_DIM + sp_idx] = slot + 1          # roster {1..6}
        obs[1, slot * POKEMON_FULL_DIM + sp_idx] = 6 - slot          # same roster, permuted
        obs[2, slot * POKEMON_FULL_DIM + sp_idx] = slot + 10         # different roster
    keys = roster_keys(obs, layout)
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]


# ---------------------------------------------------------------- device derivation (ai_v7_22 crash)

class _DeviceProbeModel:
    """Minimal model stub: params on a chosen device; records the obs device it is forwarded on."""

    def __init__(self, device):
        import torch as th

        class _Dist:
            pass

        class _Policy:
            def __init__(self, dev):
                self._p = th.nn.Parameter(th.zeros(1, device=dev))
                self.seen_devices = []
                # A real SB3 policy always has one, and `masked_action_probs` reads it to top the
                # obs dict up with the flag-gated Dict keys the extractor's forward demands
                # (`agents.model.extra_obs_keys`). A bare object is the honest stand-in for this
                # probe: no route attribute is set, so the registry asks for no extra key — which
                # is the configuration a device probe is about.
                self.features_extractor = types.SimpleNamespace()

            def parameters(self):
                return iter([self._p])

            def get_distribution(self, ob):
                self.seen_devices.append(ob["observation"].device)
                d = _Dist()
                inner = _Dist()
                inner.logits = th.zeros(ob["observation"].shape[0], ob["action_mask"].shape[1],
                                        device=ob["observation"].device)
                d.distribution = inner
                return d

        self.policy = _Policy(device)


def _run_device_probe(device):
    from agents.training.churn_probe import masked_action_probs
    m = _DeviceProbeModel(device)
    obs = np.zeros((5, 7), np.float32)
    mask = np.ones((5, 4), np.float32)
    probs = masked_action_probs(m, obs, mask)
    assert probs.shape == (5, 4)
    assert all(str(d).startswith(device) for d in m.policy.seen_devices)


def test_masked_action_probs_derives_device_from_model_cpu():
    """ai_v7_22 launch-crash regression: tensors must be built on the MODEL's device, not a
    caller-passed string — a cuda student + cpu-default teachers made any shared device wrong."""
    _run_device_probe("cpu")


def test_masked_action_probs_derives_device_from_model_cuda():
    import torch as th
    if not th.cuda.is_available():
        pytest.skip("no CUDA")
    _run_device_probe("cuda")
