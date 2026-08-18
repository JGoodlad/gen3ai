"""Unit tests for gen3_item_belief_v1 (v83) — the opponent's HIDDEN ITEM as a learned posterior.

The properties pinned here, in the order they matter:
  * the COLD-START contract — zero-init delta ⇒ posterior == the Smogon prior EXACTLY;
  * that the prior's Choice-Band column tracks the static ``SPECIES_CB_PRIOR`` (the table the op
    used to read) within the row-floor's renorm bound, so enabling is ~behavior-preserving at init;
  * the op seam — ``item_cb_prob=None`` is byte-identical to the pre-v83 static path, and a
    posterior that AGREES with the static table produces the same block;
  * the label builder + the BeliefBank's seventh row (loss, acc, None on empty mask);
  * the version machinery — v83 migration default + the check_compatible gate.
"""
import gymnasium as gym
import numpy as np
import pytest
import torch

from agents import gen3_data
from agents.model.damage_tables import build_item_prior, build_species_cb_prior
from agents.model.features_extractor import (
    Gen3FeaturesExtractor, ItemBelief, TEAM_SIZE,
)
from agents.observation.belief_labels import build_item_labels, zero_item_labels
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.model.model_version import MODEL_CONFIG_VERSION, _migrate_config
from agents.model import damage_op_test as _DT

_mp = load_mappings()
_layout = Gen3ObservationEncoder(_mp).get_layout()
_N_ITEMS = _layout["max_items"]
_N_SPECIES = _layout["max_species"]
_CB_NUM = gen3_data.items.get("choiceband").num


def _model(**kw):
    obs_space = gym.spaces.Box(0.0, 1.0, shape=(_layout["total_dim"],), dtype=np.float32)
    return Gen3FeaturesExtractor(obs_space, layout=_layout, mappings=_mp, **kw)


# ------------------------------------------------------------------ the head + its prior


def test_cold_start_posterior_equals_prior_exactly():
    """Zero-init delta ⇒ logits = log(prior) ⇒ softmax == prior (rows sum to 1). This is the
    semantic contract the identity-init sweep protects — not just numerical hygiene."""
    head = ItemBelief(_N_SPECIES, _N_ITEMS)
    tt = gen3_data.species.get("tyranitar")
    sk = gen3_data.species.get("skarmory")
    species = torch.tensor([[tt.num, sk.num] + [0] * 4])
    tokens = torch.randn(1, TEAM_SIZE, head.norm.normalized_shape[0])
    _, post = head(tokens, species)
    want = head.item_prior[species]
    assert torch.allclose(post, want, atol=1e-5), (
        f"cold-start posterior != prior (max|Δ|={float((post - want).abs().max()):.2e})")


def test_prior_cb_column_tracks_static_table():
    """The op's unrevealed branch swaps ``SPECIES_CB_PRIOR[s]`` for ``posterior[s, CB]``; at cold
    start those must agree to within the row floor's renorm (~0.6%), or enabling the flag is a
    bigger init change than documented."""
    ip = build_item_prior(_N_SPECIES, _N_ITEMS)
    cb = build_species_cb_prior(_N_SPECIES)
    carriers = cb > 0
    assert int(carriers.sum()) > 100, "CB carrier set collapsed — the priors data drifted"
    delta = (ip[:, _CB_NUM] - cb)[carriers].abs()
    assert float(delta.max()) < 0.01, (
        f"cold-start P(CB) drifted {float(delta.max()):.4f} from the static table — "
        "the row floor grew, or the CB column is misaligned")


def test_prior_gigo_anchors():
    """Two anchors straight from Smogon usage: Blissey ~always Leftovers; Tyranitar's CB share is
    real but minor. Either failing means the item-num axis or the priors source drifted."""
    ip = build_item_prior(_N_SPECIES, _N_ITEMS)
    bl = gen3_data.species.get("blissey").num
    lo = gen3_data.items.get("leftovers").num
    assert float(ip[bl, lo]) > 0.9
    tt = gen3_data.species.get("tyranitar").num
    assert 0.01 < float(ip[tt, _CB_NUM]) < 0.3
    assert torch.allclose(ip.sum(dim=1), torch.ones(_N_SPECIES), atol=1e-4)


# ------------------------------------------------------------------ the op seam


def _cb_ctx(op):
    """A ctx whose opp active (Tyranitar, unrevealed item) attacks our lone Skarmory."""
    tt = gen3_data.species.get("tyranitar")
    sk = gen3_data.species.get("skarmory")
    from agents.observation.types import TypeEncoder
    T = TypeEncoder.TYPE_TO_IDX
    return _DT._fake_ctx(
        op, attacker_num=tt.num, attacker_t1=T["ROCK"], attacker_t2=T["DARK"],
        defenders=[(sk.num, T["STEEL"], T["FLYING"])] + [(0, 0, 0)] * 5,
        hp_probs_active=[0.0] * 16)


def test_op_item_cb_prob_none_matches_static_and_agreeing_posterior():
    """The seam's two identities: None == the pre-v83 static path (same code line), and a supplied
    posterior whose CB column equals the static table produces the SAME block — proving the swap
    only ever moves the prior factor, never the gating."""
    op, layout = _DT._op_and_layout()
    ctx = _cb_ctx(op)
    logits = _DT._logits_hp_only(layout["max_moves"])
    base = op(ctx, logits, item_cb_prob=None)
    agree = op.SPECIES_CB_PRIOR[ctx.species_ids[:, TEAM_SIZE:]]      # [B,6] static P(CB) per opp slot
    swapped = op(ctx, logits, item_cb_prob=agree)
    assert torch.equal(base, swapped)


def test_op_item_cb_prob_moves_only_the_unrevealed_branch():
    """A posterior claiming CB=1 at the active slot must change the block when the item is
    UNREVEALED, and must be IGNORED once the item is revealed as not-CB (exactness stays 0/1)."""
    op, layout = _DT._op_and_layout()
    logits = _DT._logits_hp_only(layout["max_moves"])
    sure_cb = torch.zeros(1, TEAM_SIZE)
    sure_cb[:, 0] = 1.0
    ctx = _cb_ctx(op)
    assert not torch.equal(op(ctx, logits, item_cb_prob=None),
                           op(ctx, logits, item_cb_prob=sure_cb)), \
        "a certain-CB posterior did not move the unrevealed branch"
    ctx2 = _cb_ctx(op)
    lo = gen3_data.items.get("leftovers").num
    ctx2.item_ids[:, TEAM_SIZE] = lo                                 # revealed: NOT a Choice Band
    assert torch.equal(op(ctx2, logits, item_cb_prob=None),
                       op(ctx2, logits, item_cb_prob=sure_cb)), \
        "a revealed non-CB item must zero the belief's influence (exactness gate)"


# ------------------------------------------------------------------ the extractor wiring


def test_extractor_flag_builds_and_publishes():
    fe = _model(attend_unrevealed_opponents=True, move_belief_mode="revealed",
                  damage_op=True, item_belief=True)
    assert fe.item_belief_head is not None
    obs = {"observation": torch.zeros(2, _layout["total_dim"]),
           "action_mask": torch.ones(2, 11)}
    fe(obs)
    assert fe.last_item_logits.shape == (2, TEAM_SIZE, _N_ITEMS)
    assert "item_logits" in fe._belief_supervision
    off = _model(attend_unrevealed_opponents=True, move_belief_mode="revealed", damage_op=True)
    assert off.item_belief_head is None


def test_item_head_is_zero_init():
    """The generic end-of-__init__ sweep captures zero Linears by observation; assert membership so
    a re-init regression is caught by NAME here as well as by the sweep."""
    fe = _model(attend_unrevealed_opponents=True, move_belief_mode="revealed",
                  damage_op=True, item_belief=True)
    assert float(fe.item_belief_head.item_head.weight.abs().max()) == 0.0
    assert "item_belief_head.item_head" in fe._identity_init_zeroed


# ------------------------------------------------------------------ labels + bank row


def test_label_builder():
    ident = lambda s: s
    lab, msk = build_item_labels(
        ["tyranitar", "skarmory"], {"tyranitar": _CB_NUM, "skarmory": 0},
        species_known=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0], normalize=ident)
    assert lab[0] == _CB_NUM and msk[0] == 1.0
    assert lab[1] == 0 and msk[1] == 1.0, "'nothing' (num 0) is a CLASS, not PAD"
    assert msk[2:].sum() == 0
    lab0, msk0 = zero_item_labels()
    assert msk0.sum() == 0 and (lab0 == -1).all()


def test_bank_loss_and_row():
    from agents.training.belief_bank import item_belief_loss, ROWS
    row = [r for r in ROWS if r.name == "item"]
    assert len(row) == 1 and row[0].site == "revealed" and row[0].coef == "item_belief_coef"
    logits = torch.full((2, TEAM_SIZE, _N_ITEMS), -10.0)
    logits[:, 0, _CB_NUM] = 10.0
    label = torch.full((2, TEAM_SIZE), -1, dtype=torch.long)
    label[:, 0] = _CB_NUM
    mask = torch.zeros(2, TEAM_SIZE)
    mask[:, 0] = 1.0
    loss, metrics = item_belief_loss(logits, label, mask)
    assert metrics["acc"] == 1.0 and metrics["n_slots"] == 2
    assert float(loss) < 1e-3
    assert item_belief_loss(logits, label, torch.zeros(2, TEAM_SIZE)) is None


# ------------------------------------------------------------------ version machinery


def test_pre_floor_config_is_refused():
    """gen3_frame_deletion_v1 raised MIGRATION_FLOOR to 90, so this pre-floor config is now
    REFUSED rather than migrated — the floor's stated purpose ("refuses pre-floor configs outright
    instead of walking dead branches"). The assertion follows the behaviour: what must hold is that
    the old version is rejected with a diagnosis, not that a dead branch still defaults a field."""
    from agents.model.model_version import ModelVersionError
    with pytest.raises(ModelVersionError, match="PRE-GENERATION|floor"):
        _migrate_config({"config_version": 82})
    assert MODEL_CONFIG_VERSION >= 83


def test_check_compatible_gates_item_belief():
    from agents.model.model_version import ModelVersion, ModelVersionError
    import dataclasses
    b = ModelVersion.from_layout_and_policy_kwargs(_layout, {"features_extractor_kwargs": {}})
    a = dataclasses.replace(b, item_belief=True)
    with pytest.raises(ModelVersionError, match="item_belief"):
        a.check_compatible(b)
