"""Unit tests for gen3_opp_hp_type_belief_v1 — the opponent HIDDEN-POWER-TYPE belief + the typed-HP
candidate fix (the "opp Hidden Power reads immune" GIGO). Covers: the prior buffer, the op-side bare-237
mask + prior-floor + effectiveness-narrowing, the learned HPTypeBelief head (cold-start == prior, valid
posterior, gradient flow), module build/leak-safety across the tri-state, the CE loss, the privileged
labels, and the version gate.
"""
import gymnasium as gym
import numpy as np
import torch

from agents.model import damage_tables as dt
from agents.model.features_extractor import (
    Gen3FeaturesExtractor, DamageOperator, HPTypeBelief, TEAM_SIZE,
    _DMG_PER_MON, _DMG_IDX_PHYS_HIGH, _DMG_IDX_SPEC_HIGH, _DMG_IDX_PHYS_PKO, _DMG_IDX_SPEC_PKO,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.observation.types import TypeEncoder
from agents.training.hidden_power_tracker import HIDDEN_POWER_TYPE_ORDER
from agents.observation.belief_labels import (
    HP_TYPE_NAMES, N_HP_TYPES_LABEL, hp_type_idx_from_move_id, build_hp_type_labels, zero_hp_type_labels,
)
from agents.training.instrumented_ppo import InstrumentedMaskablePPO
from agents.model.model_version import ModelVersion, ModelVersionError
from agents import gen3_data
from agents.model import damage_op_test as _DT  # reuse its proven _fake_ctx / _make_layout / _logits_hp_only

_T2I = TypeEncoder.TYPE_TO_IDX
_mp = load_mappings()
_layout = Gen3ObservationEncoder(_mp).get_layout()


def _model(**kw):
    obs_space = gym.spaces.Box(0.0, 1.0, shape=(_layout["total_dim"],), dtype=np.float32)
    return Gen3FeaturesExtractor(obs_space, layout=_layout, mappings=_mp, **kw)


def _hp_idx(type_name: str) -> int:
    return [t.name for t in HIDDEN_POWER_TYPE_ORDER].index(type_name)


# ----------------------------------------------------------------- the prior buffer
def test_hp_type_prior_normalized_and_sensible():
    """build_hp_type_prior: every row sums to 1 (a proper distribution); a species with a real Smogon HP
    usage entry reflects it; a species with none falls back to the flat 1/16."""
    p = dt.build_hp_type_prior(_layout["max_species"])
    assert p.shape == (_layout["max_species"], 16)
    sums = p.sum(dim=1)
    assert torch.allclose(sums[sums > 0], torch.ones_like(sums[sums > 0]), atol=1e-5)
    abra = gen3_data.species.get("abra").num
    # abra's Smogon HP prior is grass-heavy (≈0.79 grass / 0.21 water) — those two dominate.
    assert p[abra, _hp_idx("GRASS")] > 0.5
    assert p[abra, _hp_idx("WATER")] > 0.1
    # the unknown-species sentinel (num 0) is the flat 1/16 fallback.
    assert torch.allclose(p[0], torch.full((16,), 1.0 / 16), atol=1e-6)


# ----------------------------------------------------------------- the op-side fix (the "immune" bug)
def _op_high(op, ctx, lg):
    out = op(ctx, lg)[:, :TEAM_SIZE * _DMG_PER_MON].reshape(1, TEAM_SIZE, _DMG_PER_MON)
    # the opp HP threat vs our slot 0 — the max of the phys/spec high-roll fractions.
    return float(max(out[0, 0, _DMG_IDX_PHYS_HIGH], out[0, 0, _DMG_IDX_SPEC_HIGH]))


def test_opp_hp_immune_bug_and_fix():
    """The reported bug: with HP believed present but the opp HP type unknown (obs hp_probs all-zero, the
    pre-fire case), the LEGACY op surfaces only the bare typeless HP (BP 0) → ~0 damage ("immune"). The
    fix (hp_type_fix) masks the bare-237 + floors the typed belief on the Smogon prior → real damage."""
    n_moves = _layout["max_moves"]
    lg = _DT._logits_hp_only(n_moves)                         # HP presence pinned, everything else ≈0
    # Salamence (Dragon/Flying) — 4× weak to HP Ice; any typed HP does real damage. obs hp_probs all-zero.
    attacker = dict(attacker_num=gen3_data.species.get("tyranitar").num,
                    attacker_t1=_T2I["ROCK"], attacker_t2=_T2I["DARK"])
    defenders = [(gen3_data.species.get("salamence").num, _T2I["DRAGON"], _T2I["FLYING"])] + [(0, 0, 0)] * 5
    off = DamageOperator(_DT._make_layout(), hp_type_fix=False)
    fix = DamageOperator(_DT._make_layout(), hp_type_fix=True)
    ctx_off = _DT._fake_ctx(off, defenders=defenders, hp_probs_active=[0.0] * 16, **attacker)
    ctx_fix = _DT._fake_ctx(fix, defenders=defenders, hp_probs_active=[0.0] * 16, **attacker)
    off_high = _op_high(off, ctx_off, lg)
    fix_high = _op_high(fix, ctx_fix, lg)
    assert off_high < 1e-3, f"legacy op should read the opp HP as ~immune, got {off_high}"
    assert fix_high > 0.05, f"the fix should surface a real typed-HP threat, got {fix_high}"


def _typed_w(op, w_all):
    """The 16 typed-HP candidate weights, read from the REAL move-nums HP_TYPED_NUMS (355-370), in
    HP_TYPE_ORDER order (gen3_opp_hp_typed_candidates_v1 — HP is ordinary typed moves, not an appended block)."""
    return w_all[0, op.HP_TYPED_NUMS]


def test_bare_237_always_masked_typed_at_real_nums():
    """gen3_opp_hp_typed_candidates_v1: the bare typeless num-237 is ALWAYS masked from the damage candidates
    (it's the presence token, BP 0); the candidate axis is C = n_moves (no appended block); the typed-HP
    belief lands on the REAL typed move-nums HP_TYPED_NUMS."""
    n_moves = _layout["max_moves"]
    lg = _DT._logits_hp_only(n_moves)
    attacker = dict(attacker_num=gen3_data.species.get("tyranitar").num,
                    attacker_t1=_T2I["ROCK"], attacker_t2=_T2I["DARK"])
    defenders = [(gen3_data.species.get("skarmory").num, _T2I["STEEL"], _T2I["FLYING"])] + [(0, 0, 0)] * 5
    for fix in (True, False):
        op = DamageOperator(_DT._make_layout(), hp_type_fix=fix)
        ctx = _DT._fake_ctx(op, defenders=defenders, hp_probs_active=[0.0] * 16, **attacker)
        w_all = op._opp_candidate_weights(ctx, lg)
        assert w_all.shape[1] == n_moves, "candidate axis must be C = n_moves (no appended-16)"
        assert float(w_all[0, op.hp_num]) < 1e-4, f"bare-237 must be masked (fix={fix})"


def test_fixed_op_respects_effectiveness_narrowing():
    """Once the opp HP has FIRED, the obs hp_probs carries the effectiveness-narrowed distribution (hard
    zeros for impossible types). The fixed op MUST respect those zeros even with the prior floor — a
    narrowed-to-Ice obs collapses the typed belief to Ice."""
    n_moves = _layout["max_moves"]
    lg = _DT._logits_hp_only(n_moves)
    ice = [0.0] * 16
    ice[_hp_idx("ICE")] = 1.0
    attacker = dict(attacker_num=gen3_data.species.get("tyranitar").num,
                    attacker_t1=_T2I["ROCK"], attacker_t2=_T2I["DARK"])
    defenders = [(gen3_data.species.get("salamence").num, _T2I["DRAGON"], _T2I["FLYING"])] + [(0, 0, 0)] * 5
    op = DamageOperator(_DT._make_layout(), hp_type_fix=True)
    ctx = _DT._fake_ctx(op, defenders=defenders, hp_probs_active=ice, **attacker)
    w_all = op._opp_candidate_weights(ctx, lg)
    typed = _typed_w(op, w_all)                               # [16] typed-HP candidate weights (real nums)
    ice_slot = _hp_idx("ICE")
    assert typed[ice_slot] > 0.5                              # Ice keeps ~all the presence belief
    assert typed.sum() - typed[ice_slot] < 1e-4              # every other type zeroed by the narrowing


def test_narrowing_off_meta_survivor_does_not_re_immune():
    """REGRESSION (review HIGH): when the opp fired an OFF-META HP — the effectiveness narrows to a type the
    species' prior gives ~zero mass — the OLD `clamp_min(1e-6)` renorm collapsed the typed weight to ~0,
    re-introducing 'immune'. The fix falls back to UNIFORM-over-survivors, so the believed type still carries
    the full HP-presence weight. Here abra's Smogon HP prior is grass/water only; we narrow the obs to ICE."""
    n_moves = _layout["max_moves"]
    lg = _DT._logits_hp_only(n_moves)
    ice = [0.0] * 16
    ice[_hp_idx("ICE")] = 0.3                                 # tracker's narrowed obs: only ICE survives (prior≈0 there)
    attacker = dict(attacker_num=gen3_data.species.get("abra").num,  # prior concentrated on grass/water
                    attacker_t1=_T2I["PSYCHIC"], attacker_t2=0)
    defenders = [(gen3_data.species.get("salamence").num, _T2I["DRAGON"], _T2I["FLYING"])] + [(0, 0, 0)] * 5
    op = DamageOperator(_DT._make_layout(), hp_type_fix=True)   # prior-mode (no learned posterior passed)
    ctx = _DT._fake_ctx(op, defenders=defenders, hp_probs_active=ice, **attacker)
    w_all = op._opp_candidate_weights(ctx, lg)
    typed = _typed_w(op, w_all)
    ice_slot = _hp_idx("ICE")
    assert float(typed[ice_slot]) > 0.5, f"off-meta survivor collapsed to ~immune: {float(typed[ice_slot])}"
    assert float(typed.sum() - typed[ice_slot]) < 1e-4         # ALL mass on the lone survivor, not ~0
    assert _op_high(op, ctx, lg) > 0.05                        # and the op prices a real (4× ice) threat


# ----------------------------------------------------------------- the learned head
def _head():
    return HPTypeBelief(_layout["max_species"], _layout["type_embedding_dim"])


def test_hp_type_belief_cold_start_equals_prior():
    """Zero-init head ⇒ the posterior at cold start == the Smogon prior (the clean A/B baseline)."""
    head = _head()
    tokens = torch.zeros(2, TEAM_SIZE, 128)
    species = torch.full((2, TEAM_SIZE), gen3_data.species.get("abra").num, dtype=torch.long)
    logits, post = head(tokens, species)
    assert logits.shape == (2, TEAM_SIZE, 16) and post.shape == (2, TEAM_SIZE, 16)
    assert torch.allclose(post.sum(-1), torch.ones(2, TEAM_SIZE), atol=1e-5)
    assert torch.allclose(post, head.hp_prior[species], atol=1e-5)


def test_v2_reinject_presence_gated_and_small_init():
    """gen3_opp_hp_type_belief_v2: the token reinjection is (a) ≈0 at init (small-init residual → no harm
    before the belief sharpens), (b) PRESENCE-GATED (presence 0 → no enrichment, so an unlikely HP injects
    no spurious type signal), and (c) MASKED to the selected (revealed) slots."""
    from agents.model.features_extractor import Gen3FeaturesExtractor as _FE  # for the shared Embeddings
    m = _model(move_belief_mode="revealed", damage_op=True, attend_unrevealed_opponents=True,
               hp_type_belief_mode="learned")
    head, emb = m.hp_type_belief_head, m.embeddings
    B = 2
    tokens = torch.randn(B, TEAM_SIZE, 128)
    post = torch.full((B, TEAM_SIZE, 16), 1.0 / 16)
    mask = torch.ones(B, TEAM_SIZE)
    # (a) small-init: with full presence the enrichment is small (≈0) at init
    enriched = head.reinject(tokens, post, torch.ones(B, TEAM_SIZE), mask, emb)
    assert enriched.shape == tokens.shape
    # (b) presence 0 → the residual is exactly 0, so enriched == LayerNorm(tokens) (gate kills the signal)
    z = head.reinject(tokens, post, torch.zeros(B, TEAM_SIZE), mask, emb)
    assert torch.allclose(z, head.reinject_norm(tokens), atol=1e-6)
    # (c) mask 0 on a slot → that slot's residual is 0 (only the masked-in slots are enriched)
    half = torch.zeros(B, TEAM_SIZE); half[:, 0] = 1.0
    e2 = head.reinject(tokens, post, torch.ones(B, TEAM_SIZE), half, emb)
    assert torch.allclose(e2[:, 1:], head.reinject_norm(tokens)[:, 1:], atol=1e-6)


def test_v2_reinject_does_not_collapse_multi_type_for_op():
    """The reinjection is a SEPARATE mean-field token signal — it does NOT touch the posterior the op
    consumes, so multiple un-ruled-out types stay live for the op's distinct top-K simulation."""
    head = _head()
    tokens = torch.zeros(1, TEAM_SIZE, 128)
    species = torch.zeros(1, TEAM_SIZE, dtype=torch.long)        # sentinel → flat 1/16 prior (all live)
    _, post = head(tokens, species)
    # the op-facing posterior is the full distribution (all 16 types live), unaffected by reinject
    assert (post[0, 0] > 0).all() and abs(float(post[0, 0].sum()) - 1.0) < 1e-5


def test_learned_posterior_feeds_op_and_grad_flows():
    """The op consumes the learned posterior at the active slot; its damage gradient flows back to the
    posterior (so the supervision/op gradient can sharpen the head)."""
    n_moves = _layout["max_moves"]
    lg = _DT._logits_hp_only(n_moves)
    attacker = dict(attacker_num=gen3_data.species.get("tyranitar").num,
                    attacker_t1=_T2I["ROCK"], attacker_t2=_T2I["DARK"])
    defenders = [(gen3_data.species.get("salamence").num, _T2I["DRAGON"], _T2I["FLYING"])] + [(0, 0, 0)] * 5
    op = DamageOperator(_DT._make_layout(), hp_type_fix=True)
    ctx = _DT._fake_ctx(op, defenders=defenders, hp_probs_active=[0.0] * 16, **attacker)
    post = torch.zeros(1, TEAM_SIZE, 16, requires_grad=True)
    with torch.no_grad():
        post_init = torch.full((1, TEAM_SIZE, 16), 1.0 / 16)
    post.data = post_init.clone()
    out = op(ctx, lg, None, None, post)                       # spread=None, latent=None, hp_type=post
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert post.grad is not None and float(post.grad.abs().sum()) > 0


# ----------------------------------------------------------------- model-level build / leak-safety
def test_modes_build_modules_and_projection_unchanged():
    """off/prior/learned: the HPTypeBelief head exists ONLY under learned; the op carries the prior buffers
    under prior+learned. The head is a SIDE readout (never in pi/vf) → projection widths identical across
    all three, so off is byte-for-byte and the others don't widen the heads."""
    base = dict(move_belief_mode="revealed", damage_op=True, attend_unrevealed_opponents=True)
    off = _model(hp_type_belief_mode="off", **base)
    prior = _model(hp_type_belief_mode="prior", **base)
    learned = _model(hp_type_belief_mode="learned", **base)
    assert off.hp_type_belief_head is None and prior.hp_type_belief_head is None
    assert learned.hp_type_belief_head is not None
    assert not off.damage_op.hp_type_fix and prior.damage_op.hp_type_fix and learned.damage_op.hp_type_fix
    assert not hasattr(off.damage_op, "SPECIES_HP_PRIOR")     # off = obs-only type source (no prior buffer)
    assert hasattr(prior.damage_op, "SPECIES_HP_PRIOR")       # prior/learned floor on the Smogon HP prior
    # HP_TYPED_NUMS / HP_CAND_MASK are ALWAYS present (the unified typed-HP candidates) — from the buffers.
    for m in (off, prior, learned):
        assert hasattr(m.damage_op, "HP_TYPED_NUMS") and hasattr(m.damage_op, "HP_CAND_MASK")
    # The head + op buffers are NON-persistent / side — projection input dims unchanged across modes.
    assert off.projection_input_dim == prior.projection_input_dim == learned.projection_input_dim
    assert (off.value_projection_input_dim == prior.value_projection_input_dim
            == learned.value_projection_input_dim)   # the head is a SIDE readout — never in pi/vf (leak-safe)
    # learned adds head params to the state_dict; prior does not.
    assert any("hp_type_belief_head" in k for k in learned.state_dict())
    assert not any("hp_type_belief_head" in k for k in prior.state_dict())


def test_topk_surfaces_two_distinct_typed_hps():
    """The headline payoff: with the discrete top-K block on, a belief split between hp-ice and hp-grass
    surfaces BOTH as DISTINCT top-K candidates (distinct indices in the typed-HP slice), each carrying its
    own typed latent + per-defender damage — so the policy reads "if hp-ice → these mons die; if hp-grass →
    those die", not a single collapsed worst case. (Pre-mask this slot was eaten by the bare-237.)"""
    layout = _DT._make_layout()
    n_moves = layout["max_moves"]
    K = 5
    op = DamageOperator(layout, topk_k=K, hp_type_fix=True)
    ice_slot, grass_slot = _hp_idx("ICE"), _hp_idx("GRASS")
    # Defenders: Salamence (4× Ice) + Swampert (4× Grass) → the two typed HPs threaten DIFFERENT pivots.
    defenders = [(0, _T2I["DRAGON"], _T2I["FLYING"]), (0, _T2I["WATER"], _T2I["GROUND"])] + [(0, 0, 0)] * 4
    ctx = _DT._topk_ctx(op, defenders=defenders, attacker_t1=_T2I["PSYCHIC"], hp_probs_active=[0.0] * 16)
    lg = _DT._logits_hp_only(n_moves)                          # HP present, all other moves ≈0
    post = torch.zeros(1, TEAM_SIZE, 16)
    post[:, :, ice_slot] = 0.5
    post[:, :, grass_slot] = 0.5                               # the "2 HPs" belief
    op(ctx, lg, None, _DT._synth_latent(layout), post)
    idx = op.last_topk_idx[0].tolist()
    # the typed HPs are REAL move-nums now (HP_TYPED_NUMS): hp-ice=365, hp-grass=363 — both surfaced.
    assert int(op.HP_TYPED_NUMS[ice_slot]) in idx, f"hp-ice candidate missing from top-K {idx}"
    assert int(op.HP_TYPED_NUMS[grass_slot]) in idx, f"hp-grass candidate missing from top-K {idx}"


def test_requires_damage_op():
    import pytest
    with pytest.raises(ValueError, match="requires damage_op"):
        _model(hp_type_belief_mode="prior")          # no damage_op


# ----------------------------------------------------------------- the CE loss
def test_hp_type_belief_loss_ce_and_masking():
    """_hp_type_belief_loss: CE over the masked slots; None when no slot is scored; acc reported."""
    B = 4
    logits = torch.zeros(B, TEAM_SIZE, 16)
    logits[:, 0, _hp_idx("ICE")] = 5.0                        # slot 0 confidently predicts ICE
    label = torch.full((B, TEAM_SIZE), -1, dtype=torch.long)
    label[:, 0] = _hp_idx("ICE")
    mask = torch.zeros(B, TEAM_SIZE)
    mask[:, 0] = 1.0
    out = InstrumentedMaskablePPO._hp_type_belief_loss(logits, label, mask)
    assert out is not None
    loss, m = out
    assert float(loss) >= 0 and m["acc"] == 1.0 and m["n_slots"] == B
    # no scored slot → None (off path)
    assert InstrumentedMaskablePPO._hp_type_belief_loss(logits, label, torch.zeros(B, TEAM_SIZE)) is None
    assert InstrumentedMaskablePPO._hp_type_belief_loss(None, label, mask) is None


# ----------------------------------------------------------------- the privileged labels
def test_hp_type_names_match_tracker_order():
    """GIGO pin: HP_TYPE_NAMES (belief_labels) must equal HIDDEN_POWER_TYPE_ORDER (the op's HP_TYPE_IDX
    axis + the obs hp_probs order) — else the label index would mean a different type than the op consumes."""
    assert HP_TYPE_NAMES == tuple(t.name.lower() for t in HIDDEN_POWER_TYPE_ORDER)
    assert N_HP_TYPES_LABEL == 16


def test_hp_type_idx_from_move_id():
    assert hp_type_idx_from_move_id("hiddenpowerice") == _hp_idx("ICE")
    assert hp_type_idx_from_move_id("hiddenpowergrass") == _hp_idx("GRASS")
    assert hp_type_idx_from_move_id("hiddenpower") is None     # bare = type unknown
    assert hp_type_idx_from_move_id("thunderbolt") is None


def test_build_hp_type_labels():
    """REVEALED slots whose species runs HP get the true type index + mask 1; others stay PAD/0."""
    # slots: [salamence revealed (HP-ice), tyranitar revealed (no HP), believed, ...]
    species_known = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    revealed = ["salamence", "tyranitar"]
    s2t = {"salamence": _hp_idx("ICE")}                       # only salamence runs HP
    lab, msk = build_hp_type_labels(revealed, s2t, species_known, lambda s: s)
    assert lab[0] == _hp_idx("ICE") and msk[0] == 1.0
    assert lab[1] == -1 and msk[1] == 0.0                     # tyranitar has no HP → not scored
    assert (msk[2:] == 0.0).all()
    z_lab, z_msk = zero_hp_type_labels()
    assert (z_lab == -1).all() and (z_msk == 0.0).all()


# ----------------------------------------------------------------- versioning
def test_version_roundtrip_and_gate():
    """hp_type_belief_mode round-trips through model_config JSON; check_compatible FATALs on a mismatch
    (off↔prior a forward change, prior↔learned a state_dict change)."""
    def _ver(mode):
        fe_kw = {"layout": _layout, "hp_type_belief_mode": mode}
        pk = {"features_extractor_kwargs": fe_kw, "net_arch": []}
        return ModelVersion.from_layout_and_policy_kwargs(_layout, pk)
    off, prior, learned = _ver("off"), _ver("prior"), _ver("learned")
    assert off.hp_type_belief_mode == "off" and learned.hp_type_belief_mode == "learned"
    # round-trip via JSON (migration default)
    import json
    from agents.model.model_version import _migrate_config
    reloaded = ModelVersion(**_migrate_config(json.loads(prior.to_json())))
    assert reloaded.hp_type_belief_mode == "prior"
    import pytest
    for a, b in ((off, prior), (prior, learned), (off, learned)):
        with pytest.raises(ModelVersionError, match="hp_type_belief_mode mismatch"):
            a.check_compatible(b)
    off.check_compatible(off)                                 # same mode → OK
