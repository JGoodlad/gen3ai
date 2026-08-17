"""Unit tests for gen3_typed_hp_belief_v1 — the opponent's Hidden Power reasoned about ONLY as the 16
discrete typed moves.

The composition `P(HP_t) = presence · P(type = t)` now happens once, in `HPTypeBelief.compose_typed_hp`,
right next to the move-belief head — so the posterior every consumer reads (the damage op, the top-K, the
BCE, the latent grading, the token reinjection, the prober) already carries Hidden Power at its 16 real
typed move-nums 355-370, with the bare typeless 237 driven hard off.

The properties pinned here, in the order they matter:
  * the CONSTRAINT — a revealed Hidden Power must exist as SOME type (Σ typed ≈ 1), structurally;
  * the two eliminations — moveset exhaustion and effectiveness narrowing ("discard what can't be true");
  * that the old "opp HP reads immune" GIGO cannot recur, in either of its two forms;
  * that the op does no HP reasoning of its own and both its call sites agree;
  * that the head is unconditional, leak-safe, and its gradient path is live.
"""
import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model import damage_tables as dt
from agents.model.features_extractor import (
    Gen3FeaturesExtractor, DamageOperator, HPTypeBelief, TEAM_SIZE,
    _DMG_PER_MON, _DMG_IDX_PHYS_HIGH, _DMG_IDX_SPEC_HIGH, _HP_PRESENCE_OFF_LOGIT, _REVEAL_LOGIT,
    HIDDEN_POWER_MOVE_NUM,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.observation.types import TypeEncoder
from agents.training.hidden_power_tracker import HIDDEN_POWER_TYPE_ORDER
from agents.observation.belief_labels import (
    HP_TYPE_NAMES, N_HP_TYPES_LABEL, hp_type_idx_from_move_id, build_hp_type_labels, zero_hp_type_labels,
)
from agents.training.instrumented_ppo import InstrumentedMaskablePPO
from agents.model.model_version import (
    ModelVersion, ModelVersionError, _migrate_config)
from agents import gen3_data
from agents.model import damage_op_test as _DT  # reuse its proven _fake_ctx / _make_layout / _logits_hp_only

_T2I = TypeEncoder.TYPE_TO_IDX
_mp = load_mappings()
_layout = Gen3ObservationEncoder(_mp).get_layout()
_N_MOVES = _layout["max_moves"]


def _model(**kw):
    obs_space = gym.spaces.Box(0.0, 1.0, shape=(_layout["total_dim"],), dtype=np.float32)
    return Gen3FeaturesExtractor(obs_space, layout=_layout, mappings=_mp, **kw)


def _hp_idx(type_name: str) -> int:
    return [t.name for t in HIDDEN_POWER_TYPE_ORDER].index(type_name)


def _head():
    return HPTypeBelief(_layout["max_species"], _layout["type_embedding_dim"])


def _raw_logits(presence_logit=_REVEAL_LOGIT, B=1):
    """Raw (un-composed) move-belief logits: Hidden Power PRESENT at the 237 presence channel, every
    other move ~absent. The shape `MoveBelief.move_logits` emits before the composition."""
    lg = torch.full((B, TEAM_SIZE, _N_MOVES), -10.0)
    lg[:, :, HIDDEN_POWER_MOVE_NUM] = presence_logit
    return lg


def _no_obs(B=1):
    """obs `hp_probs` for the opp slots when the opponent has NOT yet fired Hidden Power (all-zero — the
    state in which the pre-fix operator priced a revealed HP as nonexistent)."""
    return torch.zeros(B, TEAM_SIZE, 16)


def _move_ids(*revealed, B=1):
    """opp_move_ids [B,6,4] with `revealed` move-nums in slot 0's first positions (0 = unrevealed)."""
    ids = torch.zeros(B, TEAM_SIZE, 4, dtype=torch.long)
    for i, n in enumerate(revealed):
        ids[:, :, i] = n
    return ids


def _typed(logits, head):
    """The 16 typed-HP weights (probabilities) out of a composed posterior, in HP_TYPE_ORDER."""
    return torch.sigmoid(logits)[0, 0, head.HP_TYPED_NUMS]


# ------------------------------------------------------------------ the prior (unchanged, still pinned)
def test_hp_type_prior_normalized_and_sensible():
    """build_hp_type_prior: every row sums to 1 (a proper conditional P(type | has HP)); a species with a
    real Smogon HP usage entry reflects it; a species with none falls back to the flat 1/16."""
    p = dt.build_hp_type_prior(_layout["max_species"])
    assert p.shape == (_layout["max_species"], 16)
    sums = p.sum(dim=1)
    assert torch.allclose(sums[sums > 0], torch.ones_like(sums[sums > 0]), atol=1e-5)
    abra = gen3_data.species.get("abra").num
    assert p[abra, _hp_idx("GRASS")] > 0.5 and p[abra, _hp_idx("WATER")] > 0.1
    assert torch.allclose(p[0], torch.full((16,), 1.0 / 16), atol=1e-6)


def test_presence_prior_times_type_prior_reconstructs_typed_usage():
    """The factorisation is LOSSLESS on the data: the 237 PRESENCE channel holds Σ typed usage and
    `build_hp_type_prior` holds the conditional, so their product is each typed HP's own Smogon rate.
    That is why keying the presence prior at 237 costs no information."""
    prior = dt.build_move_prior_logits(_layout["max_species"], _N_MOVES)
    p_type = dt.build_hp_type_prior(_layout["max_species"])
    zap = gen3_data.species.get("zapdos").num
    presence = torch.sigmoid(prior[zap, HIDDEN_POWER_MOVE_NUM])
    usage = gen3_data.priors.moves("zapdos")
    for name in ("grass", "ice"):
        recon = float(presence * p_type[zap, _hp_idx(name.upper())])
        assert abs(recon - usage["hiddenpower" + name]) < 0.01, (name, recon)


def test_legality_gate_marks_typed_hp_legal():
    """A typed Hidden Power is legal iff the bare one is. `gen3_learnset.json` lists only `hiddenpower`
    (the type is an IV choice), so the legality gate used to drive all 16 typed nums to the `eps`
    IMPOSSIBLE floor for EVERY species — wrong data sitting in a live tensor."""
    gated = dt.build_move_prior_logits(_layout["max_species"], _N_MOVES, floor=0.02)
    zap = gen3_data.species.get("zapdos").num
    impossible = float(torch.logit(torch.tensor(1e-6)))
    for t in ("ICE", "GRASS", "FIRE"):
        num = int(dt._hp_typed_nums()[_hp_idx(t)])
        assert float(gated[zap, num]) > impossible + 1.0, f"typed HP {t} marked unlearnable"
    # A species that cannot learn Hidden Power at all keeps the impossible floor (the gate still bites).
    assert "hiddenpower" in gen3_data.learnset.get_legal_moves("zapdos")


# ------------------------------------------------------------------ THE CONSTRAINT
def test_revealed_hp_must_exist_as_some_type():
    """**The headline invariant the whole design exists to guarantee.** Once the opponent has been seen
    using Hidden Power, the belief may be unsure WHICH type it is, but it may never conclude there is no
    Hidden Power: `Σ_t P(HP_t) == presence`, and presence is reveal-pinned to ≈1.

    This is structural, not a penalty term — so it holds for ANY type posterior, including a pathological
    one. We assert it against a deliberately adversarial (near-one-hot, then near-uniform) head output."""
    head = _head()
    raw = _raw_logits()
    for post in (torch.full((1, TEAM_SIZE, 16), 1.0 / 16),                      # maximally unsure
                 torch.nn.functional.one_hot(torch.tensor(_hp_idx("ICE")), 16)  # maximally sure
                 .float().expand(1, TEAM_SIZE, 16)):
        typed_logits, presence = head.compose_typed_hp(raw, post, _no_obs(), _move_ids(HIDDEN_POWER_MOVE_NUM))
        total = float(_typed(typed_logits, head).sum())
        assert abs(total - float(presence[0, 0])) < 1e-4
        assert total > 0.99, f"a revealed Hidden Power vanished from the belief (Σ={total})"


def test_bare_237_is_hard_off_after_composition():
    """The typeless num is a belief bookkeeping channel, never a move. After composition it must be
    hard-off — but FINITE, so the multi-label BCE sees ~0 loss there rather than a NaN."""
    head = _head()
    typed_logits, _ = head.compose_typed_hp(
        _raw_logits(), torch.full((1, TEAM_SIZE, 16), 1.0 / 16), _no_obs(),
        _move_ids(HIDDEN_POWER_MOVE_NUM))
    assert float(typed_logits[0, 0, HIDDEN_POWER_MOVE_NUM]) == _HP_PRESENCE_OFF_LOGIT
    assert torch.isfinite(typed_logits).all()
    assert float(torch.sigmoid(typed_logits[0, 0, HIDDEN_POWER_MOVE_NUM])) < 1e-9


def test_non_hp_moves_pass_through_untouched():
    """The composition rewrites ONLY the 17 Hidden-Power channels; every other move's posterior is
    bit-identical (it must not perturb the rest of the moveset belief)."""
    head = _head()
    raw = _raw_logits()
    raw[:, :, gen3_data.moves.get("thunderbolt").num] = 3.0
    out, _ = head.compose_typed_hp(raw, torch.full((1, TEAM_SIZE, 16), 1.0 / 16), _no_obs(), _move_ids())
    touched = {HIDDEN_POWER_MOVE_NUM, *(int(n) for n in head.HP_TYPED_NUMS)}
    keep = torch.tensor([i for i in range(_N_MOVES) if i not in touched])
    assert torch.equal(out[..., keep], raw[..., keep])


# ------------------------------------------------------------------ "discard the ones that don't make sense"
def test_moveset_exhaustion_rules_hp_out():
    """Elimination 1: four moves revealed and none of them Hidden Power ⇒ it certainly has none, so the
    presence collapses to 0 and every typed weight with it — even though the 237 channel is still pinned
    'present' by the raw head. A certain fact overrides the belief."""
    head = _head()
    tb = gen3_data.moves.get("thunderbolt").num
    ids = _move_ids(tb, gen3_data.moves.get("icebeam").num,
                    gen3_data.moves.get("roar").num, gen3_data.moves.get("rest").num)
    out, presence = head.compose_typed_hp(
        _raw_logits(), torch.full((1, TEAM_SIZE, 16), 1.0 / 16), _no_obs(), ids)
    assert float(presence[0, 0]) == 0.0
    assert float(_typed(out, head).sum()) < 1e-6
    # …but four revealed moves INCLUDING Hidden Power leaves it fully present.
    ids_hp = _move_ids(tb, HIDDEN_POWER_MOVE_NUM,
                       gen3_data.moves.get("roar").num, gen3_data.moves.get("rest").num)
    _, presence_hp = head.compose_typed_hp(
        _raw_logits(), torch.full((1, TEAM_SIZE, 16), 1.0 / 16), _no_obs(), ids_hp)
    assert float(presence_hp[0, 0]) > 0.99


def test_effectiveness_narrowing_collapses_to_survivors():
    """Elimination 2: once Hidden Power has FIRED, the tracker's obs `hp_probs` hard-zeroes every type
    inconsistent with the observed effectiveness. Those zeros are CERTAIN physics, so the composed belief
    must put ALL of the presence mass on the survivors and exactly none elsewhere."""
    head = _head()
    obs = torch.zeros(1, TEAM_SIZE, 16)
    obs[:, :, _hp_idx("ICE")] = 1.0                                  # only ICE survived the narrowing
    out, _ = head.compose_typed_hp(_raw_logits(), torch.full((1, TEAM_SIZE, 16), 1.0 / 16), obs,
                                   _move_ids(HIDDEN_POWER_MOVE_NUM))
    typed = _typed(out, head)
    assert float(typed[_hp_idx("ICE")]) > 0.99
    assert float(typed.sum() - typed[_hp_idx("ICE")]) < 1e-5


def test_off_meta_survivor_does_not_re_immune():
    """REGRESSION: when the surviving type is one the head/prior gives ~zero mass (an off-meta Hidden
    Power), a naive renormalise collapses the typed weight to ~0 — silently re-immuning a move we have
    literally just been hit by. The fallback spreads UNIFORM over the survivors instead."""
    head = _head()
    obs = torch.zeros(1, TEAM_SIZE, 16)
    obs[:, :, _hp_idx("ICE")] = 0.3
    post = torch.zeros(1, TEAM_SIZE, 16)
    post[:, :, _hp_idx("GRASS")] = 1.0                               # the belief says GRASS; ICE is ~0
    out, _ = head.compose_typed_hp(_raw_logits(), post, obs, _move_ids(HIDDEN_POWER_MOVE_NUM))
    typed = _typed(out, head)
    assert float(typed[_hp_idx("ICE")]) > 0.99, "off-meta survivor collapsed back to ~immune"


def test_multiple_survivors_stay_live_and_distinct():
    """The belief is a DISTRIBUTION, not an argmax: two un-ruled-out types both keep real weight, so the
    op can simulate each separately ("if ice → these die; if grass → those do")."""
    head = _head()
    obs = torch.zeros(1, TEAM_SIZE, 16)
    obs[:, :, _hp_idx("ICE")] = 0.5
    obs[:, :, _hp_idx("GRASS")] = 0.5
    out, _ = head.compose_typed_hp(_raw_logits(), torch.full((1, TEAM_SIZE, 16), 1.0 / 16), obs,
                                   _move_ids(HIDDEN_POWER_MOVE_NUM))
    typed = _typed(out, head)
    assert float(typed[_hp_idx("ICE")]) > 0.4 and float(typed[_hp_idx("GRASS")]) > 0.4
    assert float(typed[_hp_idx("FIRE")]) < 1e-5


# ------------------------------------------------------------------ the GIGO that started this
def _op_high(op, ctx, lg):
    out = op(ctx, lg)[:, :TEAM_SIZE * _DMG_PER_MON].reshape(1, TEAM_SIZE, _DMG_PER_MON)
    return float(max(out[0, 0, _DMG_IDX_PHYS_HIGH], out[0, 0, _DMG_IDX_SPEC_HIGH]))


def test_revealed_hp_is_never_priced_as_immune_before_it_fires():
    """THE original bug, in the state that produced it: Hidden Power revealed but not yet fired, so the
    obs `hp_probs` is all-zero. The old op sourced the type from exactly that vector and multiplied the
    presence by it → 0 damage, rendered as 'immune'. With the type coming from the belief instead, a
    Salamence (4x weak to HP Ice) reads a real threat."""
    head = _head()
    op = DamageOperator(_DT._make_layout())
    attacker = dict(attacker_num=gen3_data.species.get("tyranitar").num,
                    attacker_t1=_T2I["ROCK"], attacker_t2=_T2I["DARK"])
    defenders = [(gen3_data.species.get("salamence").num, _T2I["DRAGON"], _T2I["FLYING"])] + [(0, 0, 0)] * 5
    ctx = _DT._fake_ctx(op, defenders=defenders, hp_probs_active=[0.0] * 16, **attacker)
    # Maximally unsure (uniform over 16 types): each candidate carries 1/16 of the presence, so the
    # threat is discounted — but it is emphatically NOT zero, which is the whole point.
    flat, _ = head.compose_typed_hp(_raw_logits(), torch.full((1, TEAM_SIZE, 16), 1.0 / 16),
                                    _no_obs(), _move_ids(HIDDEN_POWER_MOVE_NUM))
    assert _op_high(op, ctx, flat) > 0.02
    # …and once the belief concentrates on Ice, the 4x threat is priced at close to full weight.
    post = torch.zeros(1, TEAM_SIZE, 16); post[:, :, _hp_idx("ICE")] = 1.0
    sharp, _ = head.compose_typed_hp(_raw_logits(), post, _no_obs(), _move_ids(HIDDEN_POWER_MOVE_NUM))
    assert _op_high(op, ctx, sharp) > 10 * _op_high(op, ctx, flat)


def test_op_has_no_hp_type_source_of_its_own():
    """The op must be a plain CONSUMER now — no `hp_type_fix` switch, no `SPECIES_HP_PRIOR` buffer, and
    no `hp_type_belief` forward argument. Those were what let two call sites disagree."""
    import inspect
    op = DamageOperator(_DT._make_layout())
    assert not hasattr(op, "hp_type_fix") and not hasattr(op, "SPECIES_HP_PRIOR")
    assert "hp_type_belief" not in inspect.signature(op.forward).parameters
    assert "hp_type_belief" not in inspect.signature(op._opp_candidate_weights).parameters


def test_op_candidate_sites_agree():
    """REGRESSION (the divergence the old design allowed): `forward` used to be handed the learned
    posterior while `refine_candidates` was not, so the candidate-selection consumers priced Hidden
    Power off a different belief than the head block. Both now read the same composed posterior, so the
    typed candidates they select must carry identical weights."""
    head = _head()
    op = DamageOperator(_DT._make_layout())
    defenders = [(gen3_data.species.get("salamence").num, _T2I["DRAGON"], _T2I["FLYING"])] + [(0, 0, 0)] * 5
    ctx = _DT._fake_ctx(op, defenders=defenders, hp_probs_active=[0.0] * 16,
                        attacker_num=gen3_data.species.get("tyranitar").num,
                        attacker_t1=_T2I["ROCK"], attacker_t2=_T2I["DARK"])
    post = torch.zeros(1, TEAM_SIZE, 16)
    post[:, :, _hp_idx("ICE")] = 1.0
    composed, _ = head.compose_typed_hp(_raw_logits(), post, _no_obs(), _move_ids(HIDDEN_POWER_MOVE_NUM))
    w = op._opp_candidate_weights(ctx, composed)
    topk_idx, topk_w = op.refine_candidates(ctx, composed)
    ice_num = int(head.HP_TYPED_NUMS[_hp_idx("ICE")])
    assert float(w[0, ice_num]) > 0.99
    assert ice_num in topk_idx[0].tolist()
    assert torch.allclose(topk_w[0], w[0, topk_idx[0]])


def test_topk_surfaces_two_distinct_typed_hps():
    """The payoff: a belief split between hp-ice and hp-grass surfaces BOTH as distinct top-K candidates
    at their real move-nums, each with its own typed latent + per-defender damage."""
    layout = _DT._make_layout()
    head = _head()
    op = DamageOperator(layout, topk_k=5, matrices_incoming=True)
    defenders = [(0, _T2I["DRAGON"], _T2I["FLYING"]), (0, _T2I["WATER"], _T2I["GROUND"])] + [(0, 0, 0)] * 4
    ctx = _DT._topk_ctx(op, defenders=defenders, attacker_t1=_T2I["PSYCHIC"], hp_probs_active=[0.0] * 16)
    post = torch.zeros(1, TEAM_SIZE, 16)
    post[:, :, _hp_idx("ICE")] = 0.5
    post[:, :, _hp_idx("GRASS")] = 0.5
    composed, _ = head.compose_typed_hp(_raw_logits(), post, _no_obs(), _move_ids(HIDDEN_POWER_MOVE_NUM))
    op(ctx, composed, None, _DT._synth_latent(layout))
    idx = op.last_topk_idx[0].tolist()
    assert int(head.HP_TYPED_NUMS[_hp_idx("ICE")]) in idx, f"hp-ice missing from top-K {idx}"
    assert int(head.HP_TYPED_NUMS[_hp_idx("GRASS")]) in idx, f"hp-grass missing from top-K {idx}"


# ------------------------------------------------------------------ the head
def test_hp_type_belief_cold_start_equals_prior():
    """Zero-init head ⇒ the posterior at cold start == the Smogon prior."""
    head = _head()
    tokens = torch.zeros(2, TEAM_SIZE, 128)
    species = torch.full((2, TEAM_SIZE), gen3_data.species.get("abra").num, dtype=torch.long)
    logits, post = head(tokens, species)
    assert logits.shape == (2, TEAM_SIZE, 16) and post.shape == (2, TEAM_SIZE, 16)
    assert torch.allclose(post.sum(-1), torch.ones(2, TEAM_SIZE), atol=1e-5)
    assert torch.allclose(post, head.hp_prior[species], atol=1e-5)


def test_grad_flows_to_both_factors():
    """The composition is differentiable in BOTH halves, so the damage gradient (and the move BCE)
    sharpen the type head AND the presence channel — one posterior, one gradient path."""
    head = _head()
    raw = _raw_logits().requires_grad_(True)
    post = torch.full((1, TEAM_SIZE, 16), 1.0 / 16, requires_grad=True)
    out, _ = head.compose_typed_hp(raw, post, _no_obs(), _move_ids(HIDDEN_POWER_MOVE_NUM))
    out[0, 0, head.HP_TYPED_NUMS].sum().backward()
    assert post.grad is not None and float(post.grad.abs().sum()) > 0
    assert raw.grad is not None and float(raw.grad[0, 0, HIDDEN_POWER_MOVE_NUM].abs()) > 0


def test_reinject_presence_gated_and_masked():
    """The token reinjection is (a) presence-GATED (an unlikely HP injects no spurious type signal) and
    (b) masked to the selected slots."""
    m = _model(move_belief_mode="revealed", damage_op=True, attend_unrevealed_opponents=True)
    head, emb = m.hp_type_belief_head, m.embeddings
    B = 2
    tokens = torch.randn(B, TEAM_SIZE, 128)
    post = torch.full((B, TEAM_SIZE, 16), 1.0 / 16)
    mask = torch.ones(B, TEAM_SIZE)
    z = head.reinject(tokens, post, torch.zeros(B, TEAM_SIZE), mask, emb)
    assert torch.allclose(z, head.reinject_norm(tokens), atol=1e-6)
    half = torch.zeros(B, TEAM_SIZE); half[:, 0] = 1.0
    e2 = head.reinject(tokens, post, torch.ones(B, TEAM_SIZE), half, emb)
    assert torch.allclose(e2[:, 1:], head.reinject_norm(tokens)[:, 1:], atol=1e-6)


def test_move_reinjection_soft_embeds_typed_rows_not_the_typeless_one():
    """`MoveBelief.reinject_moves` soft-embeds `Σ_m P(m)·move_emb[m]`. Fed the COMPOSED posterior it must
    pull on the TYPED Hidden Power rows (355-370) and not on the typeless 237 row — otherwise the token
    would carry a type-less HP signal whose latent/attr rows are deliberately all-zero."""
    m = _model(move_belief_mode="revealed", damage_op=True, attend_unrevealed_opponents=True)
    head, mb, emb = m.hp_type_belief_head, m.move_belief, m.embeddings
    post = torch.zeros(1, TEAM_SIZE, 16)
    post[:, :, _hp_idx("ICE")] = 1.0
    composed, _ = head.compose_typed_hp(_raw_logits(), post, _no_obs(), _move_ids(HIDDEN_POWER_MOVE_NUM))
    w = emb.move_embedding.weight.clone()
    ice_num = int(head.HP_TYPED_NUMS[_hp_idx("ICE")])
    tokens = torch.zeros(1, TEAM_SIZE, 128)
    mask = torch.ones(1, TEAM_SIZE, dtype=torch.bool)
    base = mb.reinject_moves(tokens, mask, emb.move_embedding, composed).clone()
    with torch.no_grad():                       # perturb ONLY the typed-ice row → the reinjection moves
        emb.move_embedding.weight[ice_num] += 5.0
    assert not torch.allclose(base, mb.reinject_moves(tokens, mask, emb.move_embedding, composed), atol=1e-6)
    with torch.no_grad():                       # restore, then perturb the TYPELESS row → nothing moves
        emb.move_embedding.weight.copy_(w)
        emb.move_embedding.weight[HIDDEN_POWER_MOVE_NUM] += 5.0
    assert torch.allclose(base, mb.reinject_moves(tokens, mask, emb.move_embedding, composed), atol=1e-6)


# ------------------------------------------------------------------ model-level build / leak-safety
def test_head_is_unconditional_under_a_move_belief():
    """gen3_typed_hp_belief_v1: there is no 'off'. The head exists whenever there is a move belief to
    compose with — and it no longer requires the damage op, since the composition lives in the belief and
    reaches the token reinjection / BCE / prober even with no operator at all."""
    with_op = _model(move_belief_mode="revealed", damage_op=True, attend_unrevealed_opponents=True)
    no_op = _model(move_belief_mode="revealed", attend_unrevealed_opponents=True)
    none = _model()
    assert with_op.hp_type_belief_head is not None
    assert no_op.hp_type_belief_head is not None, "the head must not depend on the damage operator"
    assert none.hp_type_belief_head is None, "no move belief ⇒ nothing to compose presence from"


def test_head_is_a_side_readout_never_in_pi_vf():
    """Leak-safety + shape: the HP-type posterior enriches the opp TOKEN but is never concatenated into
    the projection inputs, so the head cannot widen (or leak into) pi/vf."""
    base = dict(move_belief_mode="revealed", damage_op=True, attend_unrevealed_opponents=True)
    m = _model(**base)
    with torch.no_grad():
        pi, vf = m.eval()({"observation": torch.rand(2, _layout["total_dim"])})
    assert m.last_hp_type_logits is not None and m.last_hp_type_logits.shape == (2, TEAM_SIZE, 16)
    assert pi.shape[1] == m.projection.out_features and torch.isfinite(pi).all()


def test_end_to_end_posterior_is_typed():
    """The integration claim: after a real forward, the stashed `last_move_belief_logits` — the tensor
    EVERY downstream consumer reads — has the typeless channel hard-off. Nothing downstream of the
    composition can see a typeless Hidden Power."""
    m = _model(move_belief_mode="revealed", move_prior_fusion=True, damage_op=True,
               attend_unrevealed_opponents=True).eval()
    with torch.no_grad():
        m({"observation": torch.rand(3, _layout["total_dim"])})
    lg = m.last_move_belief_logits
    assert lg is not None
    assert torch.all(lg[..., HIDDEN_POWER_MOVE_NUM] == _HP_PRESENCE_OFF_LOGIT)


# ------------------------------------------------------------------ the CE loss + labels (unchanged)
def test_hp_type_belief_loss_ce_and_masking():
    B = 4
    logits = torch.zeros(B, TEAM_SIZE, 16)
    logits[:, 0, _hp_idx("ICE")] = 5.0
    label = torch.full((B, TEAM_SIZE), -1, dtype=torch.long)
    label[:, 0] = _hp_idx("ICE")
    mask = torch.zeros(B, TEAM_SIZE)
    mask[:, 0] = 1.0
    out = InstrumentedMaskablePPO._hp_type_belief_loss(logits, label, mask)
    assert out is not None
    loss, m = out
    assert float(loss) >= 0 and m["acc"] == 1.0 and m["n_slots"] == B
    assert abs(m["mask_rate"] - 1.0 / TEAM_SIZE) < 1e-6   # uniform coverage: 1 of 6 slots/sample
    assert InstrumentedMaskablePPO._hp_type_belief_loss(logits, label, torch.zeros(B, TEAM_SIZE)) is None
    assert InstrumentedMaskablePPO._hp_type_belief_loss(None, label, mask) is None


def test_hp_type_names_match_tracker_order():
    """GIGO pin: HP_TYPE_NAMES == HIDDEN_POWER_TYPE_ORDER == the composition's HP_TYPED_NUMS order."""
    assert HP_TYPE_NAMES == tuple(t.name.lower() for t in HIDDEN_POWER_TYPE_ORDER)
    assert N_HP_TYPES_LABEL == 16
    for j, t in enumerate(HIDDEN_POWER_TYPE_ORDER):
        assert dt._hp_typed_nums()[j] == gen3_data.moves.get("hiddenpower" + t.name.lower()).num


def test_hp_type_idx_from_move_id():
    assert hp_type_idx_from_move_id("hiddenpowerice") == _hp_idx("ICE")
    assert hp_type_idx_from_move_id("hiddenpowergrass") == _hp_idx("GRASS")
    assert hp_type_idx_from_move_id("hiddenpower") is None
    assert hp_type_idx_from_move_id("thunderbolt") is None


def test_build_hp_type_labels():
    species_known = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    revealed = ["salamence", "tyranitar"]
    s2t = {"salamence": _hp_idx("ICE")}
    lab, msk = build_hp_type_labels(revealed, s2t, species_known, lambda s: s)
    assert lab[0] == _hp_idx("ICE") and msk[0] == 1.0
    assert lab[1] == -1 and msk[1] == 0.0
    assert (msk[2:] == 0.0).all()
    z_lab, z_msk = zero_hp_type_labels()
    assert (z_lab == -1).all() and (z_msk == 0.0).all()


# ------------------------------------------------------------------ versioning
def test_hp_type_belief_mode_is_gone_and_migrated_away():
    """v52 DELETED `hp_type_belief_mode` from the schema. With MIGRATION_FLOOR the v52 POP branch
    is gone too: a pre-v67 config carrying the stale key is refused with the clear pre-generation
    error — still never a bare TypeError out of `cls(**data)`."""
    assert "hp_type_belief_mode" not in ModelVersion.__dataclass_fields__
    stale = {"config_version": 38, "hp_type_belief_mode": "learned", "hp_type_belief_coef": 0.05}
    with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
        _migrate_config(dict(stale))


# ------------------------------------------------------------------ the `flat` ablation (v53)
def _ab(mode, **kw):
    return _model(move_belief_mode="revealed", move_prior_fusion=True, damage_op=True,
                  attend_unrevealed_opponents=True, hp_belief_mode=mode, **kw)


def test_flat_ablation_drops_the_head_but_keeps_typed_hp():
    """gen3_hp_belief_ablation_v1. The variable is HOW the 16 typed channels are produced, NOT whether
    Hidden Power is typed: `flat` still reasons over the discrete typed moves, it just predicts them
    independently from the move head with no presence×type factorisation and no HPTypeBelief."""
    composed, flat = _ab("composed"), _ab("flat")
    assert composed.hp_type_belief_head is not None
    assert flat.hp_type_belief_head is None
    assert any("hp_type_belief_head" in k for k in composed.state_dict())
    assert not any("hp_type_belief_head" in k for k in flat.state_dict())
    # The head is a side readout either way → the projection widths must NOT differ, so the ablation
    # is not confounded by a capacity change.
    assert composed.projection_input_dim == flat.projection_input_dim
    assert composed.value_projection_input_dim == flat.value_projection_input_dim


def test_flat_still_masks_the_typeless_channel():
    """The 237 mask is NOT an arm of the ablation — 237 carries BP 0, so leaving it a live damage
    candidate is the original "opp Hidden Power reads immune" bug. BOTH modes must kill it."""
    for mode in ("composed", "flat"):
        m = _ab(mode).eval()
        with torch.no_grad():
            m({"observation": torch.rand(3, _layout["total_dim"])})
        assert torch.all(m.last_move_belief_logits[..., HIDDEN_POWER_MOVE_NUM]
                         == _HP_PRESENCE_OFF_LOGIT), f"{mode} left the typeless HP live"


def test_flat_leaves_the_typed_channels_to_the_move_head():
    """Under `flat` the 16 typed channels are exactly what the move head predicted — untouched by any
    composition. Concretely: perturbing the MOVE head's typed-HP row moves the posterior there, and the
    composition's constraint is NOT applied (the typed weights need not sum to the presence)."""
    from agents.model.features_extractor import mask_typeless_hp
    m = _ab("flat")
    raw = _raw_logits()
    ice_num = int(dt._hp_typed_nums()[_hp_idx("ICE")])
    raw[:, :, ice_num] = 4.0                                  # the head likes HP-Ice
    out = mask_typeless_hp(raw)
    assert float(torch.sigmoid(out[0, 0, ice_num])) > 0.98    # passed straight through
    assert float(out[0, 0, HIDDEN_POWER_MOVE_NUM]) == _HP_PRESENCE_OFF_LOGIT
    # …and every non-HP channel is bit-identical, as for the composed path
    keep = torch.tensor([i for i in range(_N_MOVES) if i != HIDDEN_POWER_MOVE_NUM])
    assert torch.equal(out[..., keep], raw[..., keep])


def test_flat_has_no_reveal_constraint_and_composed_does():
    """The measurable difference, stated as the experiment: with HP REVEALED and the move head silent
    on every typed channel, `composed` still guarantees Σ typed ≈ 1 (the constraint) while `flat`
    leaves it near 0 (the head simply hasn't learned it yet). That gap IS what the ablation prices."""
    head = _head()
    from agents.model.features_extractor import mask_typeless_hp
    raw = _raw_logits()                                       # 237 pinned revealed, typed all at -10
    ids = _move_ids(HIDDEN_POWER_MOVE_NUM)
    composed, _ = head.compose_typed_hp(raw, torch.full((1, TEAM_SIZE, 16), 1.0 / 16), _no_obs(), ids)
    flat = mask_typeless_hp(raw)
    assert float(_typed(composed, head).sum()) > 0.99         # constraint holds
    assert float(_typed(flat, head).sum()) < 0.01             # no constraint — HP effectively vanishes


def test_hp_belief_mode_is_version_gated():
    """STRUCTURAL string toggle: a resume that flips it must FATAL (it adds/removes the head AND
    changes the forward), and v53 migrates an older config to the 'composed' default."""
    def _ver(mode):
        pk = {"features_extractor_kwargs": {"layout": _layout, "hp_belief_mode": mode},
              "net_arch": []}
        return ModelVersion.from_layout_and_policy_kwargs(_layout, pk)
    composed, flat = _ver("composed"), _ver("flat")
    assert composed.hp_belief_mode == "composed" and flat.hp_belief_mode == "flat"
    with pytest.raises(Exception, match="hp_belief_mode mismatch"):
        composed.check_compatible(flat)
    composed.check_compatible(composed)
    # the v53 'composed' default branch is pre-floor (MIGRATION_FLOOR): a v52 config is refused.
    with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
        _migrate_config({"config_version": 52})


def test_invalid_hp_belief_mode_raises():
    with pytest.raises(ValueError, match="hp_belief_mode must be one of"):
        _ab("nonsense")
