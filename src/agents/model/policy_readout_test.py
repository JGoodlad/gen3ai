"""`--policy-readout {tower,trunk}` (gen3_policy_readout_trunk_v1, config v138; architecture audit F2).

The claim: under ``trunk`` the flat SB3 policy tower — the extractor's ``pre_proj_norm`` / ``projection``
(1177→512) and ``mlp_extractor.policy_net`` (512→512→512, tanh) — is GONE from the policy, and the pointer
head's decision context is one learned query over every refined trunk token (`pools.PolicyStateQuery`);
each legal action is still scored from its OWN token by the same equivariant scorer. ``tower`` (production)
is byte-identical to the build without the flag: that is pinned by the K9 learner golden's two entries
(`learner_golden_test`, `learner_golden_fixed_mass_test`), unchanged by this unit.

Every test here builds the REAL production surface through the training path (`learner_golden.build_learner`
on `production_args()` + the lever; the M1 rule: an invariant asserted only on a bare module is not one), and
each FAILS on revert of what it pins:

* the tower is retired EXACTLY (state_dict, parameter count by formula, optimizer coverage, widths);
* every surviving non-lever parameter starts from the SAME bytes as the ``tower`` build (the private-seed
  ``IsolatedLinear`` build and the retire-after-ortho-init order — revert either and dozens of groups move);
* the policy features ARE the state query's read, and the actor branch is the identity on them;
* the state query is INVARIANT to key order and blind to a masked key;
* permuting our switch slots / our move slots permutes the logits (equivariance through the trunk readout);
* masked actions: illegal entries masked, legal mass sums to 1, an unresolved move slot scores exactly 0,
  and the T2 `DecisionModule` equals the policy's own sb3 path;
* one fresh-build eager CPU update under the K6 freeze: finite losses, the new parts move, nothing lazy;
* the version gate: recorded, migrated (a pre-v138 config is ``tower``), and a mismatch refused.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch as th

from agents.action.constants import MOVE_START, N_MOVE_SLOTS
from agents.model.arch_constants import D_MODEL, POINTER_HIDDEN, TRUNK_POINTER_HIDDEN
from agents.model.pools import PolicyStateQuery
from agents.observation.constants import TEAM_SIZE
from agents.training import learner_golden as L

_CACHE: Dict[str, Any] = {}
#: The tower's parameter-name prefixes (one per alias SB3 registers the shared extractor under).
_TOWER_PREFIXES = tuple(f"{a}.{m}." for a in ("features_extractor", "pi_features_extractor",
                                              "vf_features_extractor")
                        for m in ("pre_proj_norm", "projection")) + ("mlp_extractor.policy_net.",)
_QUERY_PREFIXES = tuple(f"{a}.policy_query." for a in ("features_extractor", "pi_features_extractor",
                                                         "vf_features_extractor"))


def _args(mode: str) -> Any:
    from main.train.production_args import production_args
    a = production_args()
    a.policy_readout = mode
    return a


def _learner(mode: str) -> Any:
    """The production-surface learner of ``mode`` with the NAME-keyed perturbation (so a parameter both
    modes hold gets the same noise, and the pointer scorers are not exactly zero). Cached: read-only."""
    if mode not in _CACHE:
        _CACHE[mode] = L.build_learner(args=_args(mode), perturb_keyed=True)
    return _CACHE[mode]


def _rows() -> Tuple[th.Tensor, th.Tensor]:
    """The K9 golden buffer's 64 REAL production rows and their action masks."""
    with np.load(L.BUFFER_PATH) as z:
        obs = th.as_tensor(z["obs:observation"].reshape(-1, z["obs:observation"].shape[-1]))
        mask = th.as_tensor(z["action_masks"].reshape(-1, z["action_masks"].shape[-1])) > 0.5
    return obs, mask


def _count(sd: Dict[str, th.Tensor], keys: Any) -> int:
    return sum(int(sd[k].numel()) for k in keys)


# ------------------------------------------------------------------------------- the build
def test_tower_builds_no_trunk_part_and_keeps_its_tower():
    pol = _learner("tower").policy
    fe = pol.features_extractor
    assert fe.policy_readout == "tower" and fe.policy_query is None
    assert isinstance(fe.projection, th.nn.Linear) and isinstance(fe.pre_proj_norm, th.nn.LayerNorm)
    assert [type(m).__name__ for m in pol.mlp_extractor.policy_net] == ["Linear", "Tanh", "Linear", "Tanh"]
    assert pol.pointer_head.ctx_proj.in_features == pol.mlp_extractor.latent_dim_pi == 512
    assert pol.pointer_head.ctx_proj.out_features == POINTER_HIDDEN
    assert not any(k.startswith(_QUERY_PREFIXES) for k in pol.state_dict())


def test_trunk_retires_the_tower_exactly_and_the_count_follows():
    """The parameter-count claim, by FORMULA from the tower's own widths (never a pasted number): the
    trunk build = the tower build − (pre_proj_norm + projection + policy_net) + the state query + the
    pointer scorers' widening — and nothing else differs."""
    tw, tk = _learner("tower").policy, _learner("trunk").policy
    sd_w, sd_k = tw.state_dict(), tk.state_dict()
    only_w = set(sd_w) - set(sd_k)
    only_k = set(sd_k) - set(sd_w)
    assert only_w and all(k.startswith(_TOWER_PREFIXES) for k in only_w), sorted(only_w)
    assert only_k and all(k.startswith(_QUERY_PREFIXES) for k in only_k), sorted(only_k)
    # Retired: the extractor half and SB3's actor branch.
    fe = tk.features_extractor
    assert fe.projection is None and fe.pre_proj_norm is None and fe.policy_query is not None
    assert len(tk.mlp_extractor.policy_net) == 0 and tk.mlp_extractor.latent_dim_pi == D_MODEL
    # No 64-wide squeeze of the context: the scorer keeps the trunk width.
    ptr = tk.pointer_head
    assert (ptr.ctx_proj.in_features, ptr.ctx_proj.out_features) == (D_MODEL, TRUNK_POINTER_HIDDEN)
    assert ptr.move_score.in_features == ptr.switch_score.in_features == TRUNK_POINTER_HIDDEN

    pin = tw.features_extractor.projection.in_features
    proj_out = tw.features_extractor.projection.out_features
    tower = (pin * proj_out + proj_out) + 2 * pin + sum(
        int(m.weight.numel() + m.bias.numel()) for m in tw.mlp_extractor.policy_net
        if isinstance(m, th.nn.Linear))
    query = 3 * (D_MODEL * D_MODEL + D_MODEL) + D_MODEL + 2 * D_MODEL
    ptr_w = sum(int(p.numel()) for p in tw.pointer_head.parameters())
    ptr_k = sum(int(p.numel()) for p in tk.pointer_head.parameters())
    n_w = sum(int(p.numel()) for p in tw.parameters())
    n_k = sum(int(p.numel()) for p in tk.parameters())
    assert n_k == n_w - tower + query + (ptr_k - ptr_w), (n_w, n_k, tower, query, ptr_w, ptr_k)
    assert tower > 1_000_000, f"the retired tower holds {tower} parameters — not the audited ~1.13M"
    # The optimizer holds exactly the live parameters (none of the retired ones, all of the new ones).
    opt = {id(p) for g in tk.optimizer.param_groups for p in g["params"]}
    assert opt == {id(p) for p in tk.parameters()}


def test_every_surviving_non_lever_parameter_starts_from_the_tower_bytes():
    """The query is built from a PRIVATE seed out of `IsolatedLinear`s, and the tower is retired only
    AFTER SB3's orthogonal re-init: so every parameter both modes hold — outside the pointer head, whose
    widths are the lever — starts from identical bytes. Revert either half and the global stream shifts."""
    sd_w = _learner("tower").policy.state_dict()
    sd_k = _learner("trunk").policy.state_dict()
    shared = [k for k in sd_w if k in sd_k and not k.startswith("pointer_head.")]
    assert len(shared) > 100
    moved = [k for k in shared if not th.equal(sd_w[k], sd_k[k])]
    assert not moved, f"{len(moved)} shared parameters start from different bytes: {moved[:8]}"


# ------------------------------------------------------------------------------- the forward
def _captured_query_inputs(pol: Any, obs: th.Tensor) -> Tuple[Tuple[Any, ...], th.Tensor]:
    """One extractor forward; returns the state query's positional inputs and the policy features."""
    seen = []
    h = pol.features_extractor.policy_query.register_forward_pre_hook(lambda m, a: seen.append(a))
    try:
        with th.no_grad():
            pi, _vf = pol.extract_features({"observation": obs})
    finally:
        h.remove()
    assert len(seen) == 1
    return seen[0], pi


def test_trunk_policy_features_are_the_state_query_read():
    pol = _learner("trunk").policy
    obs, _ = _rows()
    (tokens, pad, klp), pi = _captured_query_inputs(pol, obs)
    assert pi.shape == (obs.shape[0], D_MODEL)
    with th.no_grad():
        assert th.equal(pi, pol.features_extractor.policy_query(tokens, pad, klp))
        assert th.equal(pol.mlp_extractor.forward_actor(pi), pi)       # the actor branch is the identity
    fe = pol.features_extractor
    # Every refined trunk token is a key: 12 team + global + every extra seat (+ the belief pool's K)
    # (+ X5's OTHER_species token: the opponent-belief family is on in production since the version break).
    n_seats = fe.entity_seats.n_seats + (32 if fe.history_events is not None else 0)
    k = int(fe.hidden_opp_belief.k) if fe.hidden_opp_belief is not None else 0
    assert fe.hypothesis_builder is not None                       # PRECONDITION: production is X5
    assert tokens.shape[1] == 2 * TEAM_SIZE + 1 + n_seats + k + 1
    assert not bool(pad[:, 2 * TEAM_SIZE].any()), "the global token is never masked"


def test_state_query_is_invariant_to_key_order_and_blind_to_a_masked_key():
    g = th.Generator().manual_seed(3)
    q = PolicyStateQuery()
    with th.no_grad():
        for p in q.parameters():
            p.add_(th.randn(p.shape, generator=g) * 0.1)
    tok = th.randn(5, 9, D_MODEL, generator=g)
    pad = th.zeros(5, 9, dtype=th.bool)
    pad[:, 4] = True
    klp = th.randn(5, 9, generator=g).clamp(max=0.0)
    with th.no_grad():
        base = q(tok, pad, klp)
        perm = th.randperm(9, generator=g)
        assert th.allclose(q(tok[:, perm], pad[:, perm], klp[:, perm]), base, atol=1e-5, rtol=0)
        tok2 = tok.clone()
        tok2[:, 4] = 1e3 * th.randn(5, D_MODEL, generator=g)
        klp2 = klp.clone()
        klp2[:, 4] = -7.0
        assert th.equal(q(tok2, pad, klp2), base), "a masked key reached the read"
        assert not th.allclose(q(tok2, th.zeros_like(pad), klp), base), "the mask is load-bearing"


def _logits(pol: Any, ctx: th.Tensor, inputs: Any) -> th.Tensor:
    tok_req, valid, team, mcells, scells = inputs
    with th.no_grad():
        return pol.pointer_head(ctx, tok_req, valid, team, mcells, scells)  # type: ignore[no-any-return]


def test_trunk_logits_are_equivariant_under_permuting_switch_slots_and_move_slots():
    """Through the trunk readout: re-ordering the refined trunk's OUR-TEAM tokens (and their switch cells)
    permutes the switch logits; re-ordering the E3 move seats (and their tokens / cells / validity)
    permutes the move logits; the context — a set read — does not move. Real production rows."""
    pol = _learner("trunk").policy
    obs, _ = _rows()
    (tokens, pad, klp), pi = _captured_query_inputs(pol, obs)
    inputs = tuple(pol.features_extractor.last_pointer_inputs)
    base = _logits(pol, pi, inputs)
    assert float(base[:, :TEAM_SIZE].std()) > 0.0 and float(base[:, MOVE_START:MOVE_START + 4].std()) > 0.0
    q = pol.features_extractor.policy_query
    tok_req, valid, team, mcells, scells = inputs

    sig = th.tensor([3, 0, 5, 1, 4, 2])
    order = th.cat([sig, th.arange(TEAM_SIZE, tokens.shape[1])])
    with th.no_grad():
        ctx = q(tokens[:, order], pad[:, order], None if klp is None else klp[:, order])
    assert th.allclose(ctx, pi, atol=1e-5, rtol=0)
    got = _logits(pol, ctx, (tok_req, valid, team[:, sig], mcells, scells[:, sig]))
    assert th.allclose(got[:, :TEAM_SIZE], base[:, sig], atol=1e-5, rtol=0)
    assert th.allclose(got[:, TEAM_SIZE:], base[:, TEAM_SIZE:], atol=1e-5, rtol=0)

    pm = th.tensor([2, 0, 3, 1])
    e3 = 2 * TEAM_SIZE + 1
    order = th.cat([th.arange(e3), e3 + pm, th.arange(e3 + N_MOVE_SLOTS, tokens.shape[1])])
    with th.no_grad():
        ctx = q(tokens[:, order], pad[:, order], None if klp is None else klp[:, order])
    assert th.allclose(ctx, pi, atol=1e-5, rtol=0)
    got = _logits(pol, ctx, (tok_req[:, pm], valid[:, pm], team, mcells[:, pm], scells))
    mv = slice(MOVE_START, MOVE_START + N_MOVE_SLOTS)
    assert th.allclose(got[:, mv], base[:, mv][:, pm], atol=1e-5, rtol=0)
    assert th.allclose(got[:, :TEAM_SIZE], base[:, :TEAM_SIZE], atol=1e-5, rtol=0)


def test_trunk_masked_action_handling_and_the_T2_decision_forward():
    from agents.inference.service.decision import HUGE_NEG, DecisionModule, policy_reference
    pol = _learner("trunk").policy
    obs, mask = _rows()
    assert bool((~mask).any()), "the rows must hold illegal actions for this to test masking"
    with th.no_grad():
        pi, _ = pol.extract_features({"observation": obs})
        logp = pol.masked_logp(pol.mlp_extractor.forward_actor(pi), mask)
    assert bool((logp[~mask] <= HUGE_NEG / 2).all())
    assert th.allclose(logp.exp().masked_fill(~mask, 0.0).sum(-1), th.ones(obs.shape[0]), atol=1e-5)
    # An unresolved request slot scores EXACTLY 0 (the valid gate), never a score of a zero token.
    tok_req, valid, team, mcells, scells = tuple(pol.features_extractor.last_pointer_inputs)
    dead = valid.clone()
    dead[:, 1] = 0.0
    raw = _logits(pol, pi, (tok_req, dead, team, mcells, scells))
    assert bool((raw[:, MOVE_START + 1] == 0.0).all())
    # T2: the served decision forward equals the policy's own sb3 path, -inf exactly where illegal.
    dm = DecisionModule(pol).eval()
    with th.no_grad():
        lp_dm, v_dm = dm(obs, mask)
    lp_ref, v_ref = policy_reference(pol, obs, mask)
    assert th.equal(th.isinf(lp_dm), ~mask) and th.equal(th.isinf(lp_ref), ~mask)
    assert th.allclose(lp_dm[mask], lp_ref[mask], atol=1e-6, rtol=0)
    assert th.allclose(v_dm, v_ref, atol=1e-6, rtol=0)


# ------------------------------------------------------------------------------- one update
def test_trunk_fresh_build_one_cpu_update_under_the_K6_freeze():
    from agents.training.learner_lifecycle import LearnerFreeze, declare_learner_startup
    model = L.build_learner(args=_args("trunk"), perturb_keyed=True)
    declare_learner_startup(model)
    freeze = LearnerFreeze(model)
    try:
        out = L.compute(model, before_train=lambda: freeze.freeze("policy_readout_test"))
        freeze.check("after one trunk update")
    finally:
        freeze.release()
    assert out["losses"] and all(np.isfinite(v) for v in out["losses"].values()), out["losses"]
    moved = {g for g in out["group_sha256"] if out["group_sha256"][g] != out["init_group_sha256"][g]}
    for g in ("features_extractor.policy_query", "pointer_head.ctx_proj", "pointer_head.move_proj",
              "pointer_head.switch_proj", "features_extractor.team_transformer"):
        assert g in moved, f"{g} did not move in one update (moved: {sorted(moved)})"
    assert not any(g.startswith(("features_extractor.projection", "mlp_extractor.policy_net"))
                   for g in out["group_sha256"])
    assert all(bool(th.isfinite(p).all()) for p in model.policy.parameters())


# ------------------------------------------------------------------------------- the version gate
def test_recorded_migrated_and_a_mismatch_is_refused():
    """The worker's rebuild surface records the mode; a RECORDED mode migrates through verbatim; a pre-v138
    config (whose v138 branch defaulted ``tower``) predates the X5 version break's MIGRATION_FLOOR and is
    REFUSED; a resume or a frozen opponent of the other mode is REFUSED, naming the field."""
    import dataclasses

    from agents.model.model_version import MODEL_CONFIG_VERSION, ModelVersionError, _migrate_config
    assert _version("trunk").policy_readout == "trunk" and _version("tower").policy_readout == "tower"
    for mode in ("trunk", "tower"):
        rec = _migrate_config(dataclasses.asdict(_version(mode)))
        assert rec["policy_readout"] == mode and rec["config_version"] == MODEL_CONFIG_VERSION
    with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
        _migrate_config({"config_version": 137})
    with pytest.raises(ModelVersionError, match="policy_readout mismatch"):
        _version("trunk").check_compatible(_version("tower"))
    _version("trunk").check_compatible(_version("trunk"))


def _version(mode: str) -> Any:
    """The `ModelVersion` a launch of ``mode`` records, through the surface an eval / self-play worker uses."""
    from agents.model.snapshot import current_model_version
    from agents.observation.state_encoder import load_mappings
    from main.train.run_io import _run_arch_toggles
    return current_model_version(load_mappings(), **_run_arch_toggles(_args(mode)))
