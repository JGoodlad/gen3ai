"""Unit tests for the POINTER-NATIVE ACTION HEAD (v51, gen3_pointer_native_v1).

The claim: the pointer head IS the action head — no flat positional `action_net` exists. Each action
is scored from the token of the entity it selects (move logit k ← REQUEST-slot-k move token ⊕ its op
cells; switch logit j ← our-team token j ⊕ its incoming/OAX cells; struggle ← the latent_pi context),
which makes two defect classes structurally impossible rather than defended:
  * **F2** — switch logits read from a permutation-INVARIANT CLS pool, so a bench mon's own token
    could never reach its own switch logit.
  * **the ordering bug class** — the extractor reads a mon's moves SORTED BY ID while the action space
    uses REQUEST order; the permutation happens once, by move-num identity.

Load-bearing tests here:
  * the PERMUTATION on a SCRAMBLED order (an alphabetical moveset would pass even if it were a no-op);
  * `pointer_cells` offset parity vs `decode_damage_block` (the SoT layout mirror) with EVERY optional
    op block enabled, so a future append that shifts the OAX tail is caught;
  * the REAL-POLICY suite (M1 rule: invariants asserted only on a bare module are not invariants) —
    uniform-over-legal at init, funnel consistency, no `action_net.*` params, exact optimizer
    coverage, save→load logit identity.
"""
import inspect
import math

import dataclasses
import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.action.constants import ACTION_SPACE_SIZE, MOVE_START, N_MOVE_SLOTS, STRUGGLE, SWITCH_START
from agents.model.damage_op import DamageOperator, decode_damage_block
from agents.model.features_extractor import (
    Gen3FeaturesExtractor, MOVE_NET_HIDDEN, D_MODEL, PointerNativeActionHead,
    _request_order_move_tokens,
)
from agents.model.model_version import (
    ARCH_SIGNATURE, MODEL_CONFIG_VERSION, ModelVersion, _migrate_config,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_mappings = load_mappings()
_layout = Gen3ObservationEncoder(_mappings).get_layout()
_SIG = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)

# The op-enabled toggle set (damage_op pulls in move_belief revealed + the unmask flag; outgoing +
# OAX are the two cell sources the pointer head consumes).
_OP_TOGGLES = dict(attend_unrevealed_opponents=True, move_belief_mode="revealed",
                   damage_op=True, damage_outgoing=True, damage_matrices_outgoing_all=True)


def _make(**kw):
    space = gym.spaces.Box(0.0, 1.0, shape=(_layout["total_dim"],), dtype=np.float32)
    return Gen3FeaturesExtractor(space, layout=_layout, mappings=_mappings,
                                 **{k: v for k, v in kw.items() if k in _SIG})


def _obs(batch=4, seed=5):
    return {"observation": torch.rand(batch, _layout["total_dim"],
                                      generator=torch.Generator().manual_seed(seed))}


# ------------------------------------------------------- the permutation (the crux)
def _ctx(active_idx, sorted_ids, req_ids, tok):
    import types
    B = len(active_idx)
    c = types.SimpleNamespace(batch_size=B, device=torch.device("cpu"),
                              our_active_idx=torch.tensor(active_idx),
                              all_move_ids=torch.zeros(B, 12, N_MOVE_SLOTS, dtype=torch.long),
                              our_active_req_move_ids=torch.zeros(B, N_MOVE_SLOTS))
    for b, a in enumerate(active_idx):
        c.all_move_ids[b, a] = torch.tensor(sorted_ids[b])
        c.our_active_req_move_ids[b] = torch.tensor(req_ids[b], dtype=torch.float32)
    return c, tok


def test_permutation_maps_request_slots_by_move_num_not_position():
    """A SCRAMBLED request order must be resolved by move-num identity. Token value == its sorted slot,
    so the output reads back the sorted slot each request slot resolved to."""
    D = 3
    tok = torch.zeros(1, 12, N_MOVE_SLOTS, D)
    for s in range(N_MOVE_SLOTS):
        tok[0, 2, s] = float(s)
    ctx, tok = _ctx([2], [[10, 20, 30, 40]], [[30., 10., 40., 20.]], tok)
    out, valid = _request_order_move_tokens(tok, ctx)
    assert [float(out[0, k, 0]) for k in range(4)] == [2.0, 0.0, 3.0, 1.0]
    assert valid[0].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_permutation_zeroes_unresolved_slots():
    """A mon with <4 moves (or a forced-Struggle decision) has request slots that resolve to nothing:
    they must be zeroed AND marked invalid, so the head contributes exactly 0 there rather than
    scoring a garbage vector into a live logit."""
    D = 2
    tok = torch.zeros(1, 12, N_MOVE_SLOTS, D)
    for s in range(N_MOVE_SLOTS):
        tok[0, 0, s] = float(10 + s)
    ctx, tok = _ctx([0], [[7, 8, 0, 0]], [[8., 7., 0., 0.]], tok)
    out, valid = _request_order_move_tokens(tok, ctx)
    assert [float(out[0, k, 0]) for k in range(4)] == [11.0, 10.0, 0.0, 0.0]
    assert valid[0].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_permutation_is_identity_on_an_already_sorted_moveset():
    """The degenerate case that would hide a broken permutation — pinned so a regression can't pass
    by only ever being tested on alphabetical movesets."""
    D = 2
    tok = torch.zeros(1, 12, N_MOVE_SLOTS, D)
    for s in range(N_MOVE_SLOTS):
        tok[0, 1, s] = float(s)
    ctx, tok = _ctx([1], [[5, 6, 7, 8]], [[5., 6., 7., 8.]], tok)
    out, _ = _request_order_move_tokens(tok, ctx)
    assert [float(out[0, k, 0]) for k in range(4)] == [0.0, 1.0, 2.0, 3.0]


# ------------------------------------------------------- pointer_cells vs decode_damage_block (SoT)
def test_pointer_cells_match_decode_damage_block_with_every_block_enabled():
    """The offset test: build the op with EVERY optional block between the outgoing block and the OAX
    tail enabled (omx + imx; the lean top-K is suppressed by matrices_incoming), fill a random row, and
    require each pointer cell to equal the `decode_damage_block` field it claims to be. A future block
    appended before OAX (or a reordering) that shifts an offset fails here, not in a trained run."""
    op = DamageOperator(_layout, outgoing=True, topk_k=5, matrices_outgoing=True,
                        matrices_incoming=True, matrices_outgoing_all=True)
    row = torch.rand(2, op.out_dim, generator=torch.Generator().manual_seed(3))
    move_cells, switch_cells = op.pointer_cells(row)
    assert tuple(move_cells.shape) == (2, 4, op.pointer_move_cell_dim) == (2, 4, 16)
    assert tuple(switch_cells.shape) == (2, 6, op.pointer_switch_cell_dim) == (2, 6, 15 + 18)
    for b in range(2):
        d = decode_damage_block(row[b], outgoing=True, topk_k=0, matrices_outgoing=True,
                                matrices_incoming_k=op.matrices_incoming_k,
                                matrices_outgoing_all=True)
        for k in range(4):
            cell = move_cells[b, k]
            mv, st = d["outgoing"]["moves"][k], d["status_landing"][k]
            assert [float(x) for x in cell[:4]] == [mv["low"], mv["high"], mv["crit"], mv["pko"]]
            assert float(cell[4]) == st["p_land"] and float(cell[5]) == st["known"]
            assert [float(x) for x in cell[6:16]] == list(d["outgoing"]["secondary"][k].values())
        for j in range(6):
            cell = switch_cells[b, j]
            inc = d["incoming"][j]
            assert [float(x) for x in cell[:5]] == list(inc["phys"].values())
            assert [float(x) for x in cell[5:10]] == list(inc["spec"].values())
            assert float(cell[10]) == inc["p_outspeed"] and float(cell[11]) == inc["provenance"]
            cb = d["choice_band"]
            assert float(cell[12]) == cb["phys_high_cb"][j]
            assert float(cell[13]) == cb["phys_pko_cb"][j]
            assert float(cell[14]) == cb["p_cb"]                      # shared, broadcast per mon
            atk = d["outgoing_matrix_all"]["attackers"][j]
            flat = [atk["moves"][k][c] for k in range(4) for c in ("low", "high", "crit", "pko")]
            assert [float(x) for x in cell[15:31]] == flat
            assert float(cell[31]) == atk["p_outspeed"] and float(cell[32]) == atk["alive"]


def test_pointer_cell_dims_track_the_toggle_set():
    assert DamageOperator(_layout).pointer_move_cell_dim == 0
    assert DamageOperator(_layout).pointer_switch_cell_dim == 15
    assert DamageOperator(_layout, outgoing=True).pointer_move_cell_dim == 16
    assert DamageOperator(_layout, matrices_outgoing_all=True).pointer_switch_cell_dim == 33


# ------------------------------------------------------- the extractor stash (unconditional)
def test_extractor_always_stashes_pointer_inputs_baseline():
    """No toggle set: the stash exists with WIDTH-0 cells (the head's Linears are built narrower —
    a missing op block must never silently zero-pad a learned weight)."""
    fe = _make().eval()
    with torch.no_grad():
        fe(_obs(batch=3))
    tok, valid, team, mcells, scells = fe.last_pointer_inputs
    assert tuple(tok.shape) == (3, N_MOVE_SLOTS, MOVE_NET_HIDDEN[1])
    assert tuple(valid.shape) == (3, N_MOVE_SLOTS)
    assert tuple(team.shape) == (3, 6, D_MODEL)
    assert tuple(mcells.shape) == (3, 4, 0) and tuple(scells.shape) == (3, 6, 0)
    assert fe.pointer_move_cell_dim == 0 and fe.pointer_switch_cell_dim == 0


def test_extractor_stashes_op_cells_when_the_op_is_on():
    fe = _make(**_OP_TOGGLES).eval()
    with torch.no_grad():
        fe(_obs(batch=2))
    _, _, _, mcells, scells = fe.last_pointer_inputs
    assert tuple(mcells.shape) == (2, 4, 16)
    assert tuple(scells.shape) == (2, 6, 33)
    assert fe.pointer_move_cell_dim == 16 and fe.pointer_switch_cell_dim == 33


# ------------------------------------------------------- the REAL policy (the M1 rule)
_POLICY_CACHE = {}


def _real_policy(key="baseline", **toggles):
    """Construct through the SAME path training uses — MaskablePPO -> _build() — and cache (the
    build runs real dummy forwards; the read-only tests share one instance)."""
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from agents.model.policy import Gen3DualHeadMaskablePolicy

    enc = Gen3ObservationEncoder(_mappings)

    class _Env(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Dict(
                {"observation": gym.spaces.Box(0.0, 1.0, (enc.dimension,), np.float32)})
            self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)

        def reset(self, **kw):
            return {"observation": np.zeros(enc.dimension, np.float32)}, {}

        def step(self, a):
            return {"observation": np.zeros(enc.dimension, np.float32)}, 0.0, True, False, {}

        def action_masks(self):
            return np.ones(ACTION_SPACE_SIZE, bool)

    ek = enc.get_features_extractor_kwargs()
    kw = {**ek, **{k: v for k, v in toggles.items() if k in _SIG}}
    torch.manual_seed(0)
    model = MaskablePPO(
        Gen3DualHeadMaskablePolicy, DummyVecEnv([lambda: _Env()]),
        n_steps=8, batch_size=8, n_epochs=1, device="cpu",
        policy_kwargs={"features_extractor_class": Gen3FeaturesExtractor,
                       "features_extractor_kwargs": kw,
                       "net_arch": dict(pi=[64], vf=[64])},
    )
    _POLICY_CACHE[key] = (model, enc)
    return model, enc


def _policy_obs(enc, batch=3, seed=7):
    return {"observation": torch.rand(batch, enc.dimension,
                                      generator=torch.Generator().manual_seed(seed))}


def test_no_flat_action_net_exists_and_calling_it_raises():
    model, _ = _real_policy()
    pol = model.policy
    assert isinstance(pol.pointer_head, PointerNativeActionHead)
    keys = pol.state_dict().keys()
    assert any(k.startswith("pointer_head.") for k in keys)
    assert not any(k.startswith("action_net.") for k in keys), "flat head params survived"
    with pytest.raises(RuntimeError, match="gen3_pointer_native_v1"):
        pol.action_net(torch.zeros(1, 64))


def test_cold_start_policy_is_uniform_over_legal():
    """Zero-init scorers ⇒ ALL logits identical ⇒ after masking, uniform over the LEGAL subset —
    the correct fresh-run init (there is no flat head for this to be 'byte-identical' to)."""
    model, enc = _real_policy()
    pol = model.policy
    obs = _policy_obs(enc)
    with torch.no_grad():
        dist = pol.get_distribution(obs)
        probs = dist.distribution.probs
    assert torch.allclose(probs, torch.full_like(probs, 1.0 / ACTION_SPACE_SIZE), atol=1e-6)
    mask = np.zeros((3, ACTION_SPACE_SIZE), bool)
    mask[:, [0, 6, 7]] = True                                     # 3 legal actions
    with torch.no_grad():
        dist = pol.get_distribution(obs, action_masks=mask)
        probs = dist.distribution.probs
    assert torch.allclose(probs[:, [0, 6, 7]], torch.full((3, 3), 1.0 / 3.0), atol=1e-6)
    assert float(probs[:, [1, 2, 3, 4, 5, 8, 9, 10]].abs().max()) == 0.0


def test_logit_funnel_consistency_across_the_three_sites():
    """forward / evaluate_actions / get_distribution must produce the same log-probs for the same
    obs — the PPO-ratio correctness property the single funnel guarantees."""
    model, enc = _real_policy()
    pol = model.policy
    obs = _policy_obs(enc)
    with torch.no_grad():
        actions, _values, logp = pol.forward(obs, deterministic=True)
        _v2, logp2, _ent = pol.evaluate_actions(obs, actions)
        dist = pol.get_distribution(obs)
        logp3 = dist.log_prob(actions)
    assert torch.allclose(logp, logp2, atol=1e-6)
    assert torch.allclose(logp, logp3, atol=1e-6)


def test_optimizer_covers_exactly_the_live_params():
    """The _build surgery rebuilds the optimizer: the pointer head's params must be IN (or they
    silently never train) and the deleted flat Linear's must be GONE (or dead params ride every
    checkpoint's optimizer state)."""
    model, _ = _real_policy()
    pol = model.policy
    opt_ids = {id(p) for g in pol.optimizer.param_groups for p in g["params"]}
    assert all(id(p) in opt_ids for p in pol.pointer_head.parameters())
    assert len(opt_ids) == sum(1 for _ in pol.parameters())


def test_gradient_reaches_the_pointer_scorers():
    """At the uniform init the scorers are the gradient frontier (zero weights block flow further
    upstream) — a log_prob loss is asymmetric at uniform, so their grads must be nonzero.

    NOTE the move scorer is deliberately NOT asserted here: on random [0,1) obs every request
    move-id reads 0, so every request slot is UNRESOLVED and `move_valid` gates the move logits
    (and their gradient) to exactly 0 — the valid-gate doing its job. The move-path gradient is
    pinned on the bare module below, where validity is controlled."""
    model, enc = _real_policy()
    pol = model.policy
    pol.optimizer.zero_grad(set_to_none=True)
    dist = pol.get_distribution(_policy_obs(enc))
    loss = -dist.log_prob(torch.tensor([0, 1, 10])).sum()
    loss.backward()
    for name in ("switch_score", "struggle_score"):
        lin = getattr(pol.pointer_head, name)
        assert lin.weight.grad is not None and float(lin.weight.grad.abs().sum()) > 0, name
    mv = pol.pointer_head.move_score.weight.grad
    assert mv is None or float(mv.abs().sum()) == 0.0, \
        "move grads flowed through UNRESOLVED request slots — the valid gate is broken"
    pol.optimizer.zero_grad(set_to_none=True)


def test_move_gradient_flows_when_request_slots_resolve():
    """Bare-module complement of the policy-level grad test: with valid=1 the move scorer must get
    gradient, and it must flow back into the move tokens AND the op cells (the lossless per-action
    physics route is differentiable end-to-end)."""
    head = PointerNativeActionHead(move_token_dim=MOVE_NET_HIDDEN[1], d_model=D_MODEL,
                                   ctx_dim=32, move_cell_dim=16, switch_cell_dim=33)
    g = torch.Generator().manual_seed(2)
    ctx_vec = torch.rand(2, 32, generator=g)
    tok = torch.rand(2, 4, MOVE_NET_HIDDEN[1], generator=g, requires_grad=True)
    cells = torch.rand(2, 4, 16, generator=g, requires_grad=True)
    team = torch.rand(2, 6, D_MODEL, generator=g)
    scells = torch.rand(2, 6, 33, generator=g)
    valid = torch.ones(2, 4)
    with torch.no_grad():                                    # un-zero the scorer so gradient can pass it
        head.move_score.weight.add_(1.0)
    logits = head(ctx_vec, tok, valid, team, cells, scells)
    logits[:, MOVE_START:MOVE_START + 4].sum().backward()
    assert head.move_score.weight.grad is not None
    assert tok.grad is not None and float(tok.grad.abs().sum()) > 0
    assert cells.grad is not None and float(cells.grad.abs().sum()) > 0


def test_save_load_roundtrip_preserves_the_logits(tmp_path):
    """MaskablePPO.load reconstructs the policy via __init__ → _build (fresh head) → load_state_dict;
    the reloaded policy must produce identical logits (pins the surgery against the load path)."""
    from sb3_contrib import MaskablePPO
    model, enc = _real_policy()
    pol = model.policy
    with torch.no_grad():
        for p in pol.pointer_head.parameters():                    # make the logits non-degenerate
            p.add_(torch.randn(p.shape, generator=torch.Generator().manual_seed(11)) * 0.05)
    obs = _policy_obs(enc)
    with torch.no_grad():
        before = pol.get_distribution(obs).distribution.logits.clone()
    path = tmp_path / "ptr_native.zip"
    model.save(path)
    reloaded = MaskablePPO.load(path, device="cpu")
    with torch.no_grad():
        after = reloaded.policy.get_distribution(obs).distribution.logits
    assert torch.allclose(before, after, atol=0), "reloaded logits differ"
    assert float(before.std()) > 0, "degenerate check — logits were still uniform"


# ------------------------------------------------------- versioning
def test_version_constants_and_migration():
    assert MODEL_CONFIG_VERSION >= 51
    assert ARCH_SIGNATURE == "gen3_pointer_native_v1"
    assert "pointer_head" not in {f.name for f in dataclasses.fields(ModelVersion)}
    migrated = _migrate_config({"config_version": 49, "pointer_head": True})
    assert "pointer_head" not in migrated and migrated["config_version"] >= 51
