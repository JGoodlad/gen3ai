"""`--leaf-head` — the swap changes the LEAF and nothing else, and every way it could lie REFUSES.

The load-bearing assertion is the last one: after a swap the policy's ACTION DISTRIBUTION is
byte-identical and only the win-prob readout moves. That is the whole licence for reading a
mirror cell against a contemporaneous control — if the swap touched pi, the unsearched side would
change too and the cell's 0.50 null would stop being structural.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from agents.model.aux_value_heads import WinProbHead
from main.search_dividend.leaf_head import (install_leaf_head, refuse_if_score_is_not_winprob)


class _FE(torch.nn.Module):
    """The two things `install_leaf_head` reaches for: a `win_head`, and a pi path beside it."""

    def __init__(self, with_head=True):
        super().__init__()
        self.win_head = WinProbHead() if with_head else None
        self.pi = torch.nn.Linear(128, 11)


class _Policy(torch.nn.Module):
    def __init__(self, with_head=True):
        super().__init__()
        self.features_extractor = _FE(with_head)


class _Model:
    def __init__(self, with_head=True):
        self.policy = _Policy(with_head)


def _save(tmp_path, sd, name="head.pt"):
    p = tmp_path / name
    torch.save(sd, p)
    return str(p)


def test_a_swapped_head_changes_the_win_prob_readout_and_NOTHING_about_pi(tmp_path):
    m = _Model()
    x = torch.randn(7, 128)
    pi_before = m.policy.features_extractor.pi(x).detach().numpy().copy()
    wp_before = m.policy.features_extractor.win_head(x).detach().numpy().copy()

    other = WinProbHead()
    with torch.no_grad():                    # make it unmistakably different
        for p in other.parameters():
            p.add_(1.0)
    info = install_leaf_head(m, _save(tmp_path, other.state_dict()))

    wp_after = m.policy.features_extractor.win_head(x).detach().numpy()
    pi_after = m.policy.features_extractor.pi(x).detach().numpy()
    assert not np.allclose(wp_before, wp_after), "the swap must move the win-prob readout"
    np.testing.assert_array_equal(pi_before, pi_after)   # the acting path is untouched
    np.testing.assert_allclose(wp_after, other(x).detach().numpy(), rtol=0, atol=0)
    assert len(info["leaf_head_sha1"]) == 16 and info["keys"] == sorted(other.state_dict())


@pytest.mark.parametrize("wrapper", [lambda sd: sd,
                                     lambda sd: {"state_dict": sd},
                                     lambda sd: {"win_head": sd},
                                     lambda sd: {"win_head": {"state_dict": sd}}])
def test_the_writer_s_wrappers_all_load(tmp_path, wrapper):
    m, other = _Model(), WinProbHead()
    install_leaf_head(m, _save(tmp_path, wrapper(other.state_dict())))
    x = torch.randn(3, 128)
    np.testing.assert_allclose(m.policy.features_extractor.win_head(x).detach().numpy(),
                               other(x).detach().numpy(), rtol=0, atol=0)


def test_a_checkpoint_with_no_win_head_REFUSES_rather_than_running_the_unmodified_cell(tmp_path):
    with pytest.raises(ValueError, match="nothing to replace"):
        install_leaf_head(_Model(with_head=False), _save(tmp_path, WinProbHead().state_dict()))


def test_a_head_with_different_keys_is_refused_not_coerced(tmp_path):
    sd = WinProbHead().state_dict()
    sd["net.99.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="key mismatch"):
        install_leaf_head(_Model(), _save(tmp_path, sd))


def test_a_head_of_the_wrong_WIDTH_is_refused(tmp_path):
    sd = {k: (torch.zeros(4, 4) if v.ndim == 2 else torch.zeros(4))
          for k, v in WinProbHead().state_dict().items()}
    with pytest.raises(ValueError, match="shape mismatch"):
        install_leaf_head(_Model(), _save(tmp_path, sd))


def test_a_missing_file_and_a_non_state_dict_both_refuse(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        install_leaf_head(_Model(), str(tmp_path / "nope.pt"))
    p = tmp_path / "blob.pt"
    torch.save([1, 2, 3], p)
    with pytest.raises(ValueError, match="not a state_dict"):
        install_leaf_head(_Model(), str(p))


def test_the_guard_is_on_the_resolved_SCORE_because_grid_never_reads_defensive_leaf(tmp_path):
    # A `grid` cell ranks on --score. A guard on --defensive-leaf would pass here while the
    # search was actually scoring on the scalar value head.
    refuse_if_score_is_not_winprob("win_prob")
    for bad in ("value", "auto"):
        with pytest.raises(ValueError, match="resolved --score"):
            refuse_if_score_is_not_winprob(bad)
    with pytest.raises(ValueError, match="resolved --score"):
        install_leaf_head(_Model(), _save(tmp_path, WinProbHead().state_dict()), score="value")


def test_the_cli_exposes_leaf_head_and_it_defaults_OFF():
    from main.search_dividend.__main__ import build_parser
    a = build_parser().parse_args(["m.zip"])
    assert a.leaf_head is None
    b = build_parser().parse_args(["m.zip", "--leaf-head", "/tmp/h.pt"])
    assert b.leaf_head == "/tmp/h.pt"
