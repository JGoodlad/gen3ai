"""critic_route_audit — the instrument's own gates.

The audit exists to run ONCE per generation on a trained checkpoint, so its failure mode is
silent staleness: a hook that stops matching its argument measures nothing while producing a
plausible report. These tests pin (a) every arm FIRES on a live-route policy, (b) the zero-init
routes read exactly zero at init (threat's W_inj, the pointer-side KL), (c) the fail-loud path
raises when a hook never matches, and (d) the arms the critic-route deletion wave RETIRED stay
retired — an arm whose subject is gone does not report a null, it re-points.
"""
import numpy as np
import pytest

pytest.importorskip("sb3_contrib")

from agents.model.critic_route_audit import audit
from agents.model.identity_init_test import _build_real_policy


@pytest.fixture(scope="module")
def model_and_enc():
    return _build_real_policy(opp_belief_cls_k=6, attend_unrevealed_opponents=True,
                              value_threat_inject=True, value_entity_pool=True,
                              opp_intent=True, entity_topk_seats=6, damage_topk_k=6,
                              damage_matrices_incoming=True)


def _states(enc, n=16):
    rng = np.random.default_rng(0)
    return (rng.random((n, enc.dimension), dtype=np.float32),
            np.ones((n, 11), dtype=bool))


def test_every_arm_fires_and_reports(model_and_enc):
    model, enc = model_and_enc
    obs, masks = _states(enc)
    rep = audit(model.policy, obs, masks, batch=8)
    # The arm set is CONDITIONAL on the build, deliberately, and the two ways it varies are
    # different in kind:
    #   * `vr_*` arms exist per ENABLED value route (the generic registry-keyed arm), so this
    #     fixture — which enables value_entity_pool, the ONE surviving seam member — gets exactly
    #     one. That is the property that stops the arm set drifting from the route set.
    #   * `frames` is absent because gen3_frame_deletion_v1 DELETED its subject at v90. The arm
    #     raises if asked directly and the runner skips it with a printed reason; what it must
    #     never do is appear here with a 0.0, which would read as "the frames were worthless"
    #     rather than "the frames are gone".
    #   * `seed`, `intent_reduce` and the `hidden_opp_{both,pi,vf}` SPLIT are absent for a
    #     stronger reason still: the critic-route deletion wave deleted their subjects, and an
    #     arm that outlives its subject re-points at whatever is left rather than going quiet
    #     (see `edge_ablation_audit`'s deleted `concat` arm for the case where that happened).
    assert set(rep) == {"threat", "hidden_opp", "entity_pool", "nmr", "all_off",
                        "vr_value_entity_pool"}
    assert "frames" not in rep, "an unmeasurable arm must be ABSENT, never a fabricated row"
    for arm, row in rep.items():
        assert set(row) == {"kl_mean", "kl_p95", "flip_rate", "dv_mean"}


def test_zero_init_routes_read_zero_at_init(model_and_enc):
    """At init: threat's W_inj is zero (M1-guarded) so its arm is a strict no-op, and the
    pointer scorers are zero-init so NO arm can move the policy distribution yet — any
    nonzero here means an identity-at-init contract broke upstream."""
    model, enc = model_and_enc
    obs, masks = _states(enc)
    rep = audit(model.policy, obs, masks, batch=8)
    assert rep["threat"]["dv_mean"] == 0.0
    # the v80 entity pool's out projection is zero-init, so its arm reads a strict no-op cold
    assert rep["entity_pool"]["dv_mean"] == 0.0
    # nmr and hidden_opp ride straight into the POLICY projection (no zero-init between), and
    # since the wave deleted both of their vf halves their |dV| is now structurally ZERO — a
    # nonzero reading on either would mean a concat leaked back onto the critic.
    assert rep["nmr"]["dv_mean"] == 0.0
    assert rep["hidden_opp"]["dv_mean"] == 0.0
    for arm, row in rep.items():
        assert row["kl_mean"] == 0.0 and row["flip_rate"] == 0.0, (arm, row)


def test_the_deleted_vf_routes_stay_deleted(model_and_enc):
    """The wave's standing claim, asserted as an arm-set property.

    `seed`, `intent_reduce` and the `hidden_opp_vf` split measured dV **0.0000 bit-exact** on two
    consecutive end-of-run audits, and their subjects are deleted. Re-adding an arm here without
    re-adding a subject would produce a row that reads as a measurement of something; re-adding a
    SUBJECT without re-running the audit that condemned it is the thing this guards against."""
    model, enc = model_and_enc
    obs, masks = _states(enc)
    rep = audit(model.policy, obs, masks, batch=8)
    for gone in ("seed", "intent_reduce", "hidden_opp_vf", "hidden_opp_pi", "hidden_opp_both",
                 "vr_value_clock", "vr_value_intent", "vr_intent_threshold_value",
                 "vr_intent_value_reduce"):
        assert gone not in rep, f"arm {gone!r} is back without a subject"
    fe = model.policy.features_extractor
    assert getattr(fe.assembler, "seed_readout", None) is None
    assert list(fe.assembler.parameters()) == []


def test_event_seats_arm_fires_on_a_history_events_build():
    """The H-B usage-audit arm: present only when the seats are built; masking them MOVES the
    forward even at init (the seat projection is not zero-init — this is a source ablation,
    not a zero-init route), which is exactly what makes it a usage read on a trained run."""
    model, enc = _build_real_policy(history_events=True)
    obs, masks = _states(enc)
    rep = audit(model.policy, obs, masks, batch=8)
    assert "event_seats" in rep
    assert rep["event_seats"]["kl_mean"] > 0.0 or rep["event_seats"]["dv_mean"] > 0.0


def test_the_assembler_arms_bind_their_argument_by_NAME_not_by_position():
    """The positional-binding sweep's fix for `nmr` / `hidden_opp`.

    Both patch a POSITIONAL pre-hook argument, so both need an index — but `_assert_fired` only
    catches an argument that DISAPPEARS. An argument INSERTED before the subject shifts the
    occupant while every marker still fires, which is the `concat` failure exactly: the arm
    keeps printing numbers, for the wrong tensor, under the old name. Resolving the index from
    the live signature makes an insertion move the index WITH the subject and a rename raise.
    """
    from agents.model.critic_route_audit import _arg_index
    from agents.model.features_extractor import ProjectionAssembler

    asm = ProjectionAssembler({})
    assert _arg_index(asm.forward, "ctx") == 4
    assert _arg_index(asm.forward, "hidden_opp_belief") == 5

    # an argument inserted ahead of the subject moves the index rather than the meaning
    def _widened(our, their, active, value, brand_new, ctx, hidden_opp_belief=None):
        return None
    assert _arg_index(_widened, "ctx") == 5
    assert _arg_index(_widened, "hidden_opp_belief") == 6

    with pytest.raises(RuntimeError, match="seed_rows.*not an argument"):
        _arg_index(asm.forward, "seed_rows")


def test_a_dead_hook_fails_loud():
    """The staleness guard: a marker that never fired must RAISE with the arm named, never
    return a plausible zero report."""
    from agents.model.critic_route_audit import _assert_fired

    class _Cold:
        fired = False

    with pytest.raises(RuntimeError, match="threat.*never matched"):
        _assert_fired("threat", [_Cold()])
    _assert_fired("ok", [{"fired": True}])   # dict-style marker, fired: no raise
