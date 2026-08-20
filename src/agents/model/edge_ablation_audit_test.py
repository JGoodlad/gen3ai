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
                damage_op=True, damage_outgoing=True, move_latent=True,
                move_prior_fusion=True,
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
        if fam == "concat_cells":
            continue                                     # the op arm measures a LIVE block
        assert r["kl_mean"] == 0.0 and r["flip_rate"] == 0.0 and r["dv_mean"] == 0.0, fam


def test_the_op_cells_arm_is_present_isolated_and_restores():
    """`concat_cells` — the surviving op arm — must be present, isolated from the edge families,
    and bitwise-restoring (the audit's own baseline assert guarantees the last).

    On an UNTRAINED net its policy KL is 0 by construction: the pointer head's cell input columns
    are ZERO-INIT (identity-at-init, M1-guarded), so the op's cells contribute nothing to the
    logits until they train. Its effect exists only on a trained checkpoint — gen-14 reads KL
    0.5682 / flips 0.3105, the largest policy dependence in that report — so this test asserts the
    MECHANISM (present, fires, isolated, restores) and leaves the magnitude to the real audit.

    Its twin `concat` is DELETED. It zeroed the assembler's trailing positional argument, which
    stopped being the op concat at v61 and became `seed_rows` at v76 — so for three generations it
    reported the multi-seed critic readout under the name of a block that no longer existed. The
    critic-route wave then deleted that readout too. **An arm whose subject is gone re-points at
    whatever occupies the slot; it does not go quiet.**"""
    pol, obs, masks = _fixture()
    rep = audit(pol, obs, masks, batch=4)
    assert "concat_cells" in rep
    assert "concat" not in rep, "the `concat` arm has no subject — it must not report a row"
    assert set(rep["concat_cells"]) == {"kl_mean", "kl_p95", "flip_rate", "dv_mean"}
    assert rep["concat_cells"]["kl_mean"] == 0.0, (
        "the pointer cell columns are zero-init, so an UNTRAINED net cannot move pi through them")
    assert rep["d1"]["kl_mean"] == 0.0, "edge families stay isolated from the op arm"


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
    o, m, coverage = _collect_states([str(tmp_path / "*_states.npz")], max_states=10)
    assert o.shape == (5, _layout["total_dim"])
    assert m[:, :7].all() and not m[:, 7:].any()
    # gen3_audit_state_sampler_v1: the stratified sampler's coverage rides the return so the
    # report can prove its spread (full behavioural pins live in audit_states_test.py).
    assert coverage["n_states"] == 5 and coverage["sampler"].startswith("stratified")


def test_content_only_arm_excludes_the_shared_bias_constant():
    """gen3_content_only_ablation_v1 (2026-08-19): families writing one seat block share
    bit-identical, permanently-tied BIAS vectors (input-independent term, shared zero init,
    same gradient forever), so the legacy full ablation charges every family for one shared
    constant — 97% of c5's and 70% of c3's historical pooled KL was that artifact. The
    content-only arm (zero weight, KEEP bias) measures the family's information alone: give
    a family a nonzero bias but a ZERO weight and full ablation must register while
    content-only reads exactly 0."""
    pol, obs, masks = _fixture()
    eb = pol.features_extractor.edge_bias
    with torch.no_grad():
        eb.v_map.bias.normal_(0, 0.5)          # constant offset only — no information
    b_before = eb.v_map.bias.detach().clone()
    rep = audit(pol, obs, masks, batch=4)
    assert rep["v"]["kl_mean"] > 0.0, "full ablation must charge the bias constant"
    assert rep["v"]["content"]["kl_mean"] == 0.0, (
        "content-only must NOT charge a family for its input-independent bias")
    assert rep["v"]["content"]["flip_rate"] == 0.0
    assert torch.equal(eb.v_map.bias, b_before), "bias must be restored bitwise"


def test_content_arm_registers_real_information():
    pol, obs, masks = _fixture()
    eb = pol.features_extractor.edge_bias
    with torch.no_grad():
        eb.v_map.weight.normal_(0, 0.5)        # real input-dependent content
    rep = audit(pol, obs, masks, batch=4)
    assert rep["v"]["content"]["kl_mean"] > 0.0, "content-only must register a live weight"
    assert rep["all"]["content"]["kl_mean"] > 0.0
