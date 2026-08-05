"""Unit tests for the edge-family ablation audit (the Stage-2 verdict instrument).

Load-bearing invariants: zeroing an ALREADY-zero (init-state) family measures KL exactly 0;
randomizing ONE family's map makes ITS ablation KL > 0 while the untouched families stay 0;
weights are restored bitwise after every ablation."""
import inspect

import gymnasium as gym
import numpy as np
import torch

from agents.model.edge_ablation_audit import _collect_states, audit
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_mappings = load_mappings()
_layout = Gen3ObservationEncoder(_mappings).get_layout()
_SIG = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)

_TOGGLES = dict(attend_unrevealed_opponents=True, move_belief_mode="both",
                move_belief_prefuse=True, move_belief_single_compute=True,
                damage_op=True, damage_outgoing=True, move_latent=True,
                damage_op_prefuse=True, move_prior_fusion=True,
                entity_topk_seats=5, edge_bias_families="d1,d3,v")


class _AuditPolicy(torch.nn.Module):
    """A minimal policy shim over a bare extractor + tiny heads — the audit only calls
    `get_distribution` and `predict_values`, so this keeps the test fast (no MaskablePPO build).
    The REAL-policy path is exercised by the CLI against real checkpoints."""

    def __init__(self, fe):
        super().__init__()
        self.features_extractor = fe
        torch.manual_seed(0)
        self.pi_head = torch.nn.Linear(512, 11)
        self.vf_head = torch.nn.Linear(512, 1)

    def get_distribution(self, obs):
        pi, _ = self.features_extractor(obs)
        import types
        d = types.SimpleNamespace()
        d.distribution = types.SimpleNamespace(logits=self.pi_head(pi))
        return d

    def predict_values(self, obs):
        _, vf = self.features_extractor(obs)
        return self.vf_head(vf)


def _fixture(batch=6):
    space = gym.spaces.Box(0.0, 1.0, shape=(_layout["total_dim"],), dtype=np.float32)
    fe = Gen3FeaturesExtractor(space, layout=_layout, mappings=_mappings,
                               **{k: v for k, v in _TOGGLES.items() if k in _SIG}).eval()
    pol = _AuditPolicy(fe).eval()
    obs = torch.rand(batch, _layout["total_dim"],
                     generator=torch.Generator().manual_seed(7)).numpy()
    masks = np.ones((batch, 11), dtype=bool)
    masks[:, 10] = False                                   # struggle illegal in the fixture
    return pol, obs, masks


def test_zero_init_families_measure_exactly_zero():
    pol, obs, masks = _fixture()
    rep = audit(pol, obs, masks, batch=4)
    for fam, r in rep.items():
        if fam in ("concat", "concat_cells"):
            continue                                     # the op arms measure a LIVE block
        assert r["kl_mean"] == 0.0 and r["flip_rate"] == 0.0 and r["dv_mean"] == 0.0, fam


def test_op_concat_arms_measure_a_live_block_and_restore():
    """The op-concat arms: on an UNTRAINED net the damage block is real physics feeding random
    projection weights, so zeroing it at the assembler must register (kl > 0) while the zero-init
    edge families still read exactly 0 — and the audit's own bitwise-baseline assert guarantees
    the hook/patch restore. concat_cells additionally zeroes the pointer cells, so it can only be
    >= concat on the policy KL."""
    pol, obs, masks = _fixture()
    rep = audit(pol, obs, masks, batch=4)
    assert "concat" in rep and "concat_cells" in rep
    assert rep["concat"]["kl_mean"] > 0.0, "zeroing a live concat block must register"
    assert rep["concat_cells"]["kl_mean"] >= rep["concat"]["kl_mean"] * 0.99
    assert rep["d1"]["kl_mean"] == 0.0, "edge families stay isolated from the op arms"


def test_randomized_family_is_isolated_and_restored():
    pol, obs, masks = _fixture()
    eb = pol.features_extractor.edge_bias
    with torch.no_grad():
        eb.v_map.weight.normal_(0, 0.5)
        eb.v_map.bias.normal_(0, 0.5)
    w_before = eb.v_map.weight.detach().clone()
    rep = audit(pol, obs, masks, batch=4)
    assert rep["v"]["kl_mean"] > 0.0, "the randomized family must register"
    assert rep["d1"]["kl_mean"] == 0.0 and rep["d3"]["kl_mean"] == 0.0, "untouched families stay 0"
    assert rep["all"]["kl_mean"] > 0.0
    assert torch.equal(eb.v_map.weight, w_before), "weights must be restored bitwise"


def test_collect_states_recovers_masks_from_logits(tmp_path):
    obs = np.random.rand(5, _layout["total_dim"]).astype(np.float32)
    logits = np.zeros((5, 11), dtype=np.float32)
    logits[:, 7:] = -1e9                                   # actions 7+ masked in the trace
    np.savez(tmp_path / "x_states.npz", obs=obs, logits=logits)
    o, m = _collect_states([str(tmp_path / "*_states.npz")], max_states=10)
    assert o.shape == (5, _layout["total_dim"])
    assert m[:, :7].all() and not m[:, 7:].any()
