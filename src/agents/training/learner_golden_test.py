"""K9(a) THE LEARNER GOLDEN as a routine test (``learner_golden`` module docs).

One eager ``train()`` at the production loss surface on the committed buffer must reproduce the
recorded post-update parameter bytes and every pinned loss EXACTLY, under this interpreter's torch
build. Unmarked (routine tier): CPU, one thread, ~4 s for the update. The golden's learner is the
PRODUCTION recipe's, so since 2026-10-14 it trains its trunk rounds in bf16 (``recipe.fresh.trunk_precision``);
the ``--trunk-precision fp32`` opt-out update is pinned beside it (``fp32_opt_out``). It NEVER records — a
failure is fixed by finding what changed the update, or, when the change is intended, by

    python -m agents.training.learner_golden record --reason "..."

under every torch build that has an entry (``gen3ai_torch28`` and ``gen3ai_torch28_cu128`` today).
"""
from __future__ import annotations

import pytest

from agents.training import learner_golden as L


def test_one_update_reproduces_the_recorded_golden_exactly():
    diffs, now = L.check()
    assert not diffs, (
        "the learner golden MOVED — what one PPO update computes changed:\n  " + "\n  ".join(diffs)
        + "\nIf that is intended, re-record deliberately (module docs): "
          "python -m agents.training.learner_golden record --reason \"...\" under EVERY torch build.")
    # the production loss surface is live, not vacuously matched: every production term logged a loss
    # (X5: the hidden-team belief is the set BCE, the intent is the flat pointer — `belief/aux_loss` and
    # `opp_intent/alpha_loss` / `beta_loss` were the deleted blob path's)
    for k in ("win_prob/loss", "belief/set_aux_loss", "belief/move_loss", "belief/movelatent_loss",
              "belief/spread_loss", "belief/natureev_loss", "belief/hptype_loss", "belief/item_loss",
              "opp_intent/flat_loss", "train/policy_gradient_loss", "train/entropy_loss"):
        assert k in now["losses"], f"{k} was not logged — a production term did not fold on the pinned buffer"


# LINCHPIN (ship gate, 2026-10-19): kept in the routine gate at any cost — designs/ops/testing.md "The SHIP GATE"
@pytest.mark.linchpin
def test_the_production_golden_is_the_bf16_update_and_the_fp32_opt_out_is_pinned_beside_it():
    """The bf16 ADOPTION (2026-10-14): `production_args()` carries `recipe.fresh.trunk_precision` bf16, so the
    golden's learner has bf16 trunk rounds (not vacuously fp32), and the `--trunk-precision fp32` update is a
    SECOND recorded block — which holds the bytes the golden pinned while fp32 was production, so a pass here
    proves the adoption left the fp32 path alone. The two updates must differ (bf16 is a real change)."""
    from agents.model.trunk_precision import trunk_precision_of
    model = L.build_learner()
    assert model.trunk_precision == "bf16" and trunk_precision_of(model.policy) == "bf16"
    fp32_model = L.build_learner(trunk_precision="fp32")
    assert fp32_model.trunk_precision == "fp32" and trunk_precision_of(fp32_model.policy) == "fp32"
    diffs, now32 = L.check_fp32(fp32_model)
    assert not diffs, "the fp32 OPT-OUT update MOVED:\n  " + "\n  ".join(diffs)
    e = L.entry()
    assert now32["post_params_sha256"] == e["fp32_opt_out"]["post_params_sha256"] != e["post_params_sha256"]
    assert now32["init_params_sha256"] == e["init_params_sha256"], "the two precisions share one INIT"


def test_the_golden_has_teeth_an_advantage_normalisation_flip_fails_it():
    """Change ONE detail of what the update computes (per-micro-batch advantage normalisation off) and
    the golden must name the losses and parameter groups that moved."""
    entry = L.load_golden()["entries"].get(L.torch_key())
    if entry is None:
        pytest.fail(f"no golden recorded for torch {L.torch_key()} (see test above)")
    model = L.build_learner()
    model.normalize_advantage = False
    diffs = L.diff(entry, L.compute(model))
    assert any(d.startswith("post_params_sha256") for d in diffs), diffs
    assert any("train/policy_gradient_loss" in d for d in diffs), diffs
    assert not any(d.startswith("the INIT moved") for d in diffs), "the perturbation must be in the update only"


def test_a_missing_torch_build_fails_it_never_records(monkeypatch):
    before = L.GOLDEN_PATH.read_bytes()
    monkeypatch.setattr(L, "torch_key", lambda: "0.0.0+never")
    with pytest.raises(L.LearnerGoldenError, match="no learner golden recorded for torch 0.0.0"):
        L.check()
    assert L.GOLDEN_PATH.read_bytes() == before


def test_the_default_slot_is_the_break_s_one_deliberate_re_record():
    """The X5 version break MOVED the fixed_mass arm's committed entry into the default slot verbatim (part 1,
    init `47c4c5fd…` / post `70a26bfc…`, buffer `4b48eaf0…`), then re-recorded it ONCE at the end of the break
    (parts 2/4/5 moved the init and the update, part 3 appended the OBS-FACTS block to the observation, so the
    seed-18 buffer was rebuilt): ONE history row carrying the break's reason, after the moved-verbatim row. Then
    v145 (`gen3_mon_tied_gain_v1`, the op's out_gain tied across our team slots) re-recorded it once more: its
    init moved ONLY in `damage_op` (the out_gain shape, so its name-keyed perturbation), and the buffer was rebuilt
    because the seeded learner's behaviour log-probs follow that perturbation. Then v151 (`gen3_toxic_stage_scale_v1`:
    the `g` ledger's toxic decoder) re-recorded the UPDATE once more on the SAME buffer and init (the buffer's toxic
    cells still carry the old `min(n, 8) / 8` scale, which the new decoder reads on the new scale).
    Then the END STATE's ADOPTION as production (2026-10-12) re-recorded it once more: the learner became the
    end-state network (init `60cb6e1c…` -> `d9c69eca…`, post `5171d83a…` -> `2618d640…`), and the buffer was rebuilt
    (`66a14392…` -> `06585756…`) because its behaviour log-probs follow the seeded learner — at RUN SEED 19, seed 18's
    end-state rollout holding no OTHER_move-dead row (`RECORD_RUN_SEED`'s comment).
    Then `gen3_learnset_prevo_chain_v1` (2026-10-14: the move prior's legality table gains 1500 legal pairs, the
    pre-evolution chain + Smeargle's Sketch pool; 754 mask cells flip legal, 1238 prior cells move) kept the init
    (`d9c69eca…`) and moved the update (post `2618d640…` -> `b132b440…`, identical under +cu126 and +cu128), and the
    buffer was rebuilt (`06585756…` -> `448715cc…`) at the SAME run seed 19 because its behaviour log-probs follow the
    move prior (coverage unchanged: OTHER_move live 58 / dead 6).
    Then the bf16 TRUNK's ADOPTION as production (2026-10-14: the strength A/B PASSED, `recipe.fresh.trunk_precision`
    bf16) kept the init (`d9c69eca…`) and the buffer (`448715cc…`) and moved the PRODUCTION update to the bf16 one
    (post `b132b440…` -> `2dba416a…`, identical under +cu126 and +cu128); the fp32 update (`b132b440…`, UNCHANGED)
    became the `fp32_opt_out` block.
    Then the LEGACY DELETION's U3 (config v157, 2026-10-15) re-recorded it once more: the seven per-action blocks and
    the critic's threat projection, constructed and retired before, no longer draw, so the FRESH-INIT STREAM moved
    (D-LD7; init `d9c69eca…` -> `c44c7e52…`, post `2dba416a…` -> `25debac1…`, fp32 opt-out `b132b440…` -> `3375befb…`,
    identical under +cu126 and +cu128), and the buffer was rebuilt (`448715cc…` -> `a782af44…`) because its behaviour
    log-probs follow the seeded learner — at RUN SEED 22 (`RECORD_RUN_SEED`'s comment: 19 / 20 missed the
    OTHER_move-dead coverage under the new learner, 21 failed K9(b)).
    Then T62 (`gen3_tau_fp64_solve_v1`, 2026-10-19: `hypothesis_set.fixed_size_tau` solves its constraint in float64
    and casts only τ back) kept the init (`c44c7e52…`) and the buffer (`a782af44…`) and moved the update by τ's fp32
    rounding (post `25debac1…` -> `bd79d729…`, fp32 opt-out `3375befb…` -> `d614ec22…`, identical under +cu126 and
    +cu128; the fp32 losses move ~1e-8 relative, the bf16 ones ~1e-5).
    FAILS if anyone re-records the golden without a new reason row, or restores an arm block."""
    g = L.load_golden()
    e = g["entries"]["2.8.0+cu126"]
    # The +cu128 build (env gen3ai_torch28_cu128, 2026-10-12) keeps its own entry and its own history rows; the history
    # below is the +cu126 build's (the row's `torch` field says whose).
    hist = [h for h in g["history"] if h["torch"] == "2.8.0+cu126"]
    cu128 = g["entries"].get("2.8.0+cu128")
    if cu128 is not None:
        assert [h["torch"] for h in g["history"]].count("2.8.0+cu128") == 5   # adoption + learnset + bf16 + U3 + T62
        assert cu128["post_params_sha256"] == e["post_params_sha256"] and cu128["init_params_sha256"] == e["init_params_sha256"], \
            "the CPU update is byte-identical across the two CUDA builds (verified 2026-10-12, bf16 2026-10-14): it no longer is"
        assert cu128["fp32_opt_out"]["post_params_sha256"] == e["fp32_opt_out"]["post_params_sha256"]
    assert e["init_params_sha256"] == "c44c7e52301adc82d3070fecc7b21c88e053aa351070811ea1ae43c46f775e3d"
    assert e["post_params_sha256"] == "bd79d7291ea9cb8693f9dea6f0611441b31d636452bf746eabdfdae742979d84"
    assert e["fp32_opt_out"]["post_params_sha256"] == "d614ec22cbcd15b911674bffa42bab34ac1af2e00f4f43a2257b4120eeb1f215"
    assert e["fp32_opt_out"]["init_params_sha256"] == e["init_params_sha256"]
    assert g["buffer_sha256"] == "a782af4451acec3f9403098980844493ba9f9df589501cfd60c3dea00e25cfd0"
    assert g["perturb"] == "name_keyed" and "arms" not in g
    assert L.RECORD_RUN_SEED == 22
    assert "MOVED VERBATIM" in hist[-9]["reason"]
    brk = hist[-8]["reason"]
    assert brk.startswith("gen3_x5_version_break_v1 (config v144)") and "obs-facts appended (2845)" in brk
    assert sum("gen3_x5_version_break_v1 (config v144)" in h["reason"] for h in hist) == 1
    assert hist[-7]["reason"].startswith("gen3_mon_tied_gain_v1 (config v145)")
    assert sum(h["reason"].startswith("gen3_mon_tied_gain_v1") for h in hist) == 1
    assert hist[-6]["reason"].startswith("gen3_toxic_stage_scale_v1 (config v151)")
    assert sum(h["reason"].startswith("gen3_toxic_stage_scale_v1") for h in hist) == 1
    assert hist[-5]["reason"].startswith("the END STATE adopted as production (closing test PASS 2026-10-12)")
    assert sum(h["reason"].startswith("the END STATE adopted as production") for h in hist) == 1
    assert hist[-4]["reason"].startswith("gen3_learnset_prevo_chain_v1")
    assert sum(h["reason"].startswith("gen3_learnset_prevo_chain_v1") for h in hist) == 1
    assert hist[-3]["reason"].startswith("the bf16 trunk adopted as production")
    assert sum(h["reason"].startswith("the bf16 trunk adopted as production") for h in hist) == 1
    assert hist[-2]["reason"].startswith("legacy deletion U3 (D-LD7, config v157)")
    assert sum(h["reason"].startswith("legacy deletion U3") for h in hist) == 1
    assert hist[-1]["reason"].startswith("T62 (gen3_tau_fp64_solve_v1)")
    assert sum(h["reason"].startswith("T62 (gen3_tau_fp64_solve_v1)") for h in hist) == 1
    assert hist[-1]["fp32_opt_out_post_params_sha256"] == e["fp32_opt_out"]["post_params_sha256"]


def test_record_requires_a_reason():
    with pytest.raises(SystemExit):
        L.main(["record"])
    with pytest.raises(SystemExit):
        L.record("   ")


def test_the_golden_recipe_is_READ_from_the_production_recipe_and_matches_the_recorded_one():
    """The golden's recipe knobs come from K10(a)'s block (and sb3's defaults for the knobs no launch
    sets), never a hand copy; only the shape + the documented LR are overrides. A drift — the
    production recipe moved, or someone hand-edited a value back in — fails HERE, naming the knob,
    before the parameter hashes do."""
    import inspect

    from sb3_contrib import MaskablePPO

    from main.train.recipe_surface import production_recipe
    r = L.golden_recipe()
    prod = production_recipe()
    for k in L._FROM_RECIPE:
        assert r[k] == prod[k], (k, r[k], prod[k])
    sig = inspect.signature(MaskablePPO.__init__).parameters
    for k in L._FROM_SB3_DEFAULTS:
        assert r[k] == sig[k].default, k
    assert set(L.GOLDEN_OVERRIDES) == {"n_epochs", "batch_size", "grad_accum_steps", "learning_rate"}
    recorded = L.load_golden().get("recipe")
    assert recorded == r, ("the recorded golden was computed under a different recipe — re-record "
                           f"deliberately: {recorded} vs {r}")
