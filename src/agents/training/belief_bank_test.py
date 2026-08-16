"""belief_bank — the fold's own contract: registry order, gating, coef scaling, and exact
equality with calling the loss functions directly (the byte-identity claim, unit-scale)."""
import pytest
import torch as th

from agents.training import belief_bank as bb


class _FakeExtractor:
    def __init__(self, stash):
        self._stash = stash

    def belief_supervision(self, key):
        return self._stash.get(key)


def _fixtures():
    B = 2
    stash = {
        "spread_belief": th.full((B, 6, 5), 200.0),
        "spread_nature_logits": th.zeros(B, 6, 25),
        "spread_ev": th.full((B, 6, 5), 100.0),
        "hp_type_logits": th.zeros(B, 6, 16),
    }
    stash["spread_nature_logits"][:, 0, 7] = 9.0
    stash["hp_type_logits"][:, 0, 3] = 5.0
    mask = th.zeros(B, 6)
    mask[:, 0] = 1.0
    obs = {
        "belief_spread": th.full((B, 6, 5), 250.0), "belief_spread_mask": mask,
        "belief_nature": th.full((B, 6), 7, dtype=th.long), "belief_nature_mask": mask,
        "belief_ev": th.full((B, 6, 5), 100.0), "belief_ev_mask": mask,
        "hp_type_label": th.full((B, 6), 3, dtype=th.long), "hp_type_mask": mask,
    }
    return _FakeExtractor(stash), obs


def test_compute_matches_direct_calls_in_registry_order():
    fe, obs = _fixtures()
    coefs = {"spread_belief_coef": 0.05, "hp_type_belief_coef": 0.07}
    out = bb.compute(fe, obs, coefs, gates={"spread": True, "hp_type": True},
                     site="revealed")
    assert [row.name for row, _, _ in out] == ["spread", "nature_ev", "hp_type"]
    assert [row.prefix for row, _, _ in out] == ["spread_", "natureev_", "hptype_"]

    d_spread = bb.spread_belief_loss(fe.belief_supervision("spread_belief"),
                                     obs["belief_spread"], obs["belief_spread_mask"])
    d_hp = bb.hp_type_belief_loss(fe.belief_supervision("hp_type_logits"),
                                  obs["hp_type_label"], obs["hp_type_mask"])
    assert th.equal(out[0][1], 0.05 * d_spread[0])       # coef-scaled, bit-equal
    assert th.equal(out[2][1], 0.07 * d_hp[0])
    assert out[0][2]["loss"] == pytest.approx(float(d_spread[0]))
    assert out[0][2]["mask_rate"] == pytest.approx(1 / 6)
    assert out[1][2]["nature_acc"] == 1.0 and out[2][2]["acc"] == 1.0


def test_gates_and_none_rows_skip():
    fe, obs = _fixtures()
    coefs = {"spread_belief_coef": 0.05, "hp_type_belief_coef": 0.07}
    only_hp = bb.compute(fe, obs, coefs, gates={"spread": False, "hp_type": True},
                         site="revealed")
    assert [row.name for row, _, _ in only_hp] == ["hp_type"]
    # a zero-scored minibatch (empty mask) drops the row rather than contributing NaN/0
    obs2 = dict(obs, hp_type_mask=th.zeros(2, 6))
    none_hp = bb.compute(fe, obs2, coefs, gates={"spread": False, "hp_type": True},
                         site="revealed")
    assert none_hp == []


def test_sites_partition_the_registry():
    """Every row belongs to exactly one site, and the three sites cover all six heads — the
    float-addition-order contract that lets one registry serve three call sites."""
    by_site = {}
    for row in bb.ROWS:
        by_site.setdefault(row.site, []).append(row.name)
    assert by_site == {
        "hidden_move": ["hidden_team", "move_belief"],
        "latent": ["move_latent"],
        "revealed": ["spread", "nature_ev", "hp_type"],
    }


def test_hidden_move_site_attr_param_and_loss_key():
    """The hidden-team row: 'attr' stash (last_belief_logits), 'param' arg (moves_weight),
    UNPREFIXED metrics with the historic `aux_loss` key. The move-belief row: 'attr' mode."""
    class _FE:
        def __init__(self):
            S, M = 30, 40
            self.last_belief_logits = {"species": th.randn(2, 6, S),
                                       "moves": th.randn(2, 6, M)}
            self.move_belief_mode = "revealed"
            self._ml = th.zeros(2, 6, M)

        def belief_supervision(self, key):
            return self._ml if key == "move_belief_logits" else None

    fe = _FE()
    sp = th.full((2, 6), -1, dtype=th.long)
    sp[0, 4] = 10
    mv = th.full((2, 6, 4), -1, dtype=th.long)
    mv[0, 4, 0] = 3
    km = th.full((2, 6, 4), -1, dtype=th.long)
    km[0, 0, 0] = 5
    obs = {"belief_species": sp, "belief_moves": mv, "known_moves": km}
    out = bb.compute(fe, obs,
                     coefs={"opp_belief_aux_coef": 0.1, "move_belief_coef": 0.05},
                     gates={"hidden_team": True, "move_belief": True},
                     site="hidden_move", params={"moves_weight": 1.0})
    assert [row.name for row, _, _ in out] == ["hidden_team", "move_belief"]
    ht_row, ht_term, ht_m = out[0]
    assert ht_row.prefix == "" and "aux_loss" in ht_m and "loss" not in ht_m
    direct = bb.belief_aux_loss(fe.last_belief_logits, sp, mv, moves_weight=1.0)
    assert th.equal(ht_term, 0.1 * direct[0])
    mb_row, mb_term, mb_m = out[1]
    assert mb_row.prefix == "move_" and "loss" in mb_m
    d_mb = bb.move_belief_loss(fe._ml, km, mv, "revealed")
    assert th.equal(mb_term, 0.05 * d_mb[0])


def test_aliases_still_resolve_on_the_ppo_class():
    pytest.importorskip("sb3_contrib")
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO as P

    assert P._spread_belief_loss is bb.spread_belief_loss
    assert P._nature_ev_belief_loss is bb.nature_ev_belief_loss
    assert P._hp_type_belief_loss is bb.hp_type_belief_loss
    assert P._belief_aux_loss is bb.belief_aux_loss
    assert P._move_belief_loss is bb.move_belief_loss
    assert P._move_belief_latent_loss is bb.move_belief_latent_loss
