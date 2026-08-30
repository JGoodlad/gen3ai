"""ai_v12 ADVERSARIAL REVIEW (R1) — the INTERSECTIONS between the five build-wave landings.

Every flag below was individually tested by the wave that shipped it. What this file pins is what
no single wave could see: the compositions a launch actually types. It exists because of the M2
lesson (`value_from_dist`) — every flag in a tail had its own test, the INTERSECTION had none, and
an entire critic chain was orphaned for four generations while every suite stayed green.

Five groups, each naming the pair it crosses:

1. **CLEAN WORLD × the DISTRIBUTIONAL CRITIC.** `--victory-value` (wave A, v105) made the return
   SCALE a flag for the first time. `--value-dist-*`'s support is a *separate* flag, fixed at
   launch, and under `--no-use-popart` — which is what the registered clean/sparse arms run
   (ledger 2d38a4a) — the two live in the SAME raw units and nothing compared them. Production
   carries `value_from_dist=True` with a support of [−12, +12], so the clean arm's ±1 returns land
   inside ~4 of 51 atoms: a critic quantized to ~0.5 on a ±1 scale, feeding GAE, silently.

2. **CLEAN WORLD × the TIMEOUT terminal.** The wave-A guard warns on ONE side of the ordering
   (draw better than a loss). A launch that types `--victory-value 1.0` and forgets
   `--draw-penalty` keeps the −35 default, i.e. a timeout 35× a clean loss — the "1 TERMINAL"
   claim is then false and no metric names it.

3. **CLEAN WORLD × `train/pbrs_reward_share`.** With every hand term off, the unshaped stream is
   terminal-only, and `raw_absmean == 0` on a rollout that ends no episode. The metric then
   published `0.0` — a perfect-looking score for the case where the shaping is 100% of the reward.
   The project's own rule (wave C's `train/q_winprob_loss`) is ABSENT-never-zero.

4. **v105 × v106 × v107 STACKED.** The three migrations landed as three commits an hour apart.
   This runs the whole chain on a fabricated v104 config AND on every REAL archived config in the
   models root, and asserts the result is a fully-populated, constructible `ModelVersion`.

5. **THE Q HEAD × the DISTILL FOLDS (stash clobbering).** Wave C added `q_winprob_*` to `cf_any_on`,
   so a Q-head-only run now runs the cf sample+forward on minibatches where it never ran before —
   and that forward CLOBBERS `last_value_pooled`, the FitNets hint the exploiter-distillation term
   reads. It is correct today only because of statement ORDER inside one function, which is exactly
   the kind of fact that is true until someone moves a block.

Run:
    python -m pytest src/agents/training/ai_v12_intersection_test.py -q
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import os

import numpy as np
import pytest

from agents.model.model_version import ModelVersion
from agents.model.model_version.constants import MODEL_CONFIG_VERSION
from agents.model.model_version.migrations import _migrate_config
from main.train_rl_agent import build_parser

#: The registered clean-world reward set (kept spelled the same as `clean_world_config_test`).
CLEAN_REWARD = ["--no-hand-shaping", "--victory-value", "1.0", "--draw-penalty", "-1.0"]
#: The production distributional critic, verbatim off `ai_v9_72_R3SELF_0828/model_config.json`.
PROD_DIST = ["--value-dist-mode", "shaping", "--value-dist-bins", "51",
             "--value-dist-vmin", "-12", "--value-dist-vmax", "12", "--value-from-dist"]

#: The two warning BANNERS, quoted from the guards so a rename breaks the test rather than
#: silently making it vacuous. Deliberately distinctive: the launch prints hundreds of lines of
#: flag help, several of which contain the bare words "SUPPORT" and "SCALE".
_SUPPORT = "VALUE-DIST SUPPORT"
_SCALE = "TERMINAL SCALE"


def _resolve(argv):
    from main.train.config import resolve_config
    parser = build_parser()
    return resolve_config(parser.parse_args(["--steps", "1", "--debug", *argv]), parser)


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 1. CLEAN WORLD × value_from_dist — the atom support and the return scale are ONE question
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_a_pm1_terminal_under_a_pm30_atom_support_with_no_popart_WARNS(capsys):
    """THE M2 INTERSECTION. `--victory-value 1.0` puts every raw return in [−1, +1]; the support
    the production critic carries is [−12, +12] over 51 atoms (Δ = 0.48). With PopArt OFF the
    target is the RAW return (`ppo._vd_target`), so the whole return range collapses into ~4 atoms
    and, under `--value-from-dist`, that quantized E[Z] IS the critic feeding GAE.

    A warning, not a refusal: the operator may be deliberately sizing a wide support for a later
    reward change. But it must be SAID at launch, because nothing downstream distinguishes a
    resolution-starved critic from a well-fitted one — `value_dist/mean_abs_err` looks BETTER as
    the support widens."""
    _resolve([*CLEAN_REWARD, *PROD_DIST, "--no-use-popart"])
    out = capsys.readouterr().out
    assert _SUPPORT in out, out
    assert "--value-dist-vmax" in out and "atom" in out


def test_the_same_support_with_POPART_ON_does_not_warn(capsys):
    """ANTI-VACUITY, and the reason the guard is conditioned on PopArt at all: under PopArt the CE
    target is `popart.normalize(returns)`, so the support lives in units of standard deviations and
    the raw terminal magnitude says nothing about whether it fits. Every historical run is here."""
    _resolve([*CLEAN_REWARD, *PROD_DIST, "--use-popart", "--clip-range-vf", "none"])
    assert _SUPPORT not in capsys.readouterr().out


def test_the_HISTORICAL_pm30_terminal_in_a_pm40_support_does_not_warn(capsys):
    """The guard must not fire on a correctly-sized pairing — otherwise it is noise and gets
    ignored, which is worse than not shipping it."""
    _resolve(["--value-dist-mode", "shaping", "--value-dist-bins", "51",
              "--value-dist-vmin", "-40", "--value-dist-vmax", "40", "--no-use-popart"])
    assert _SUPPORT not in capsys.readouterr().out


def test_a_terminal_LARGER_than_the_support_warns_too(capsys):
    """The other direction of the same defect, and the more destructive one: HL-Gauss absorbs
    out-of-support mass into the EDGE atoms, so a ±35 draw penalty against a ±12 support means the
    critic literally cannot represent that outcome — it saturates, and `pit_mean` is the only
    tell."""
    _resolve(["--value-dist-mode", "shaping", "--value-dist-bins", "51",
              "--value-dist-vmin", "-12", "--value-dist-vmax", "12", "--no-use-popart"])
    out = capsys.readouterr().out
    assert _SUPPORT in out, out


def test_no_value_dist_head_means_no_opinion(capsys):
    _resolve([*CLEAN_REWARD, "--no-use-popart"])
    assert _SUPPORT not in capsys.readouterr().out


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 2. CLEAN WORLD × the TIMEOUT terminal — the guard covered one side of the ordering
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_a_pm1_terminal_with_the_INHERITED_minus35_draw_penalty_warns(capsys):
    """`--victory-value 1.0` alone leaves `--draw-penalty` at its −35 default. The wave-A ORDERING
    guard passes it (−35 IS worse than a loss, which is the ordering it checks), so the launch is
    silent — while the reward stream is dominated 35:1 by an outcome the arm exists to make rare.
    A run in that state is not the clean world; it is a stall-avoidance objective wearing its
    label."""
    _resolve(["--victory-value", "1.0"])
    out = capsys.readouterr().out
    assert _SCALE in out, out
    assert "draw-penalty" in out


def test_the_registered_clean_set_is_silent_on_BOTH_guards(capsys):
    """`--victory-value 1.0 --draw-penalty -1.0` is exactly `draw = loss`; neither guard may fire,
    or the launch this whole wave exists for starts by printing two warnings it should ignore."""
    _resolve(CLEAN_REWARD)
    out = capsys.readouterr().out
    assert _SCALE not in out and "ORDERING" not in out, out


def test_the_HISTORICAL_pairing_is_silent(capsys):
    """±30 with −35 is a 1.17× ratio — the composition every generation through gen-15 trained
    under. A guard that fires on it would be a guard nobody reads."""
    _resolve([])
    assert _SCALE not in capsys.readouterr().out


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 3. CLEAN WORLD × `train/pbrs_reward_share` — ABSENT, never a flattering zero
# ──────────────────────────────────────────────────────────────────────────────────────────────

def _pbrs_model_and_buffer(raw_reward: float):
    """The REAL `apply_winprob_pbrs` over `winprob_pbrs_test`'s fakes (a genuine SB3
    `DictRolloutBuffer`), with the UNSHAPED reward stream set by hand."""
    import torch as th

    from agents.training import winprob_pbrs_test as W
    n_steps, n_envs = 4, 2
    buf = W._make_buffer(n_steps, n_envs)
    buf.rewards[:] = raw_reward
    policy = W._FakePolicy()
    model = W._FakeModel(buf, policy, coef=1.0)
    model._last_obs = {"observation": np.zeros((n_envs, 1), dtype=np.float32)}
    model._last_episode_starts = np.zeros(n_envs, dtype=np.float32)
    th.manual_seed(0)
    return model, buf


def test_an_all_zero_unshaped_stream_publishes_a_NaN_share_not_a_zero():
    """The clean arm's unshaped stream is TERMINAL-ONLY, so a rollout that ends no episode has
    `mean|r| == 0` exactly — and the shaping is then 100% of the reward. Publishing `0.0` there
    reads as "the shaping is negligible": the single most misleading number this metric could
    produce, in precisely the arm it was built to watch.

    The project's own rule, applied one wave later to `train/q_winprob_loss`: *"a defaulted 0.0
    would be a perfect score for a head that trained on nothing"* — ABSENT, never zero."""
    from agents.training.winprob_pbrs import apply_winprob_pbrs
    model, buf = _pbrs_model_and_buffer(0.0)
    m = apply_winprob_pbrs(model, buf)
    assert "reward_share" in m
    assert np.isnan(m["reward_share"]), m["reward_share"]


def test_a_real_unshaped_stream_still_reports_a_finite_share():
    """ANTI-VACUITY: the ordinary path must be untouched."""
    from agents.training.winprob_pbrs import apply_winprob_pbrs
    model, buf = _pbrs_model_and_buffer(2.0)
    m = apply_winprob_pbrs(model, buf)
    assert np.isfinite(m["reward_share"])


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 4. v105 × v106 × v107 STACKED — three migrations, one chain, on configs that really exist
# ──────────────────────────────────────────────────────────────────────────────────────────────

_WAVE_KEYS = {
    # wave A (v105)
    "hand_shaping": True, "pbrs_material": True, "pbrs_belief": True, "victory_value": 30.0,
    "win_prob_pbrs_source": None,
    # wave D (v106)
    "progress_decision_tense": False, "progress_switch_freeze": False,
    # wave C (v107)
    "q_winprob_mode": "none", "q_winprob_coef": 0.0, "q_winprob_onpolicy_coef": 0.0,
}


def _fabricated_v104() -> dict:
    """A v104 config: every `ModelVersion` field at its default, MINUS this wave's ten keys,
    stamped one version back. The last config shape that existed before the wave, so it is the
    exact input the chain must handle.

    The required (default-less) fields are the weight-shape block; they are filled with the
    current `ARCH_SIGNATURE`'s own values so the dict is a plausible checkpoint record rather than
    a bag of zeros."""
    from agents.model.model_version.constants import ARCH_SIGNATURE
    filled = {}
    for f in dataclasses.fields(ModelVersion):
        if f.default is not dataclasses.MISSING:
            filled[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:      # pragma: no cover - none today
            filled[f.name] = f.default_factory()
        elif f.name == "arch_signature":
            filled[f.name] = ARCH_SIGNATURE
        elif "hidden" in f.name or f.name == "net_arch":
            filled[f.name] = [128, 128]
        else:
            filled[f.name] = 64
    for k in _WAVE_KEYS:
        filled.pop(k, None)
    filled["config_version"] = 104
    return filled


def test_the_three_migrations_stack_on_a_v104_config():
    out = _migrate_config(_fabricated_v104())
    assert out["config_version"] == MODEL_CONFIG_VERSION == 107
    for k, want in _WAVE_KEYS.items():
        assert k in out, f"v10x migration left {k} unset"
        assert out[k] == want, (k, out[k], want)


def test_the_chain_leaves_a_v104_config_CONSTRUCTIBLE():
    """A migration that fills a dict but produces something `ModelVersion(**d)` refuses is a
    migration that only looks complete."""
    out = _migrate_config(_fabricated_v104())
    mv = ModelVersion(**out)
    assert mv.config_version == MODEL_CONFIG_VERSION


def test_the_chain_runs_on_EVERY_real_archived_config_of_this_generation():
    """The fabricated case shares its author's assumptions; the archive does not. Every
    current-generation `model_config.json` on this box goes through the whole chain and must come
    out with every field present and constructible.

    SKIPS when there is no models archive (another machine, CI) — `main_models_dir` returns None
    there, and a test that silently passes on an empty set is the failure this docstring names."""
    from utils.paths import main_models_dir
    root = main_models_dir()
    if root is None:
        pytest.skip("no models archive on this box")
    seen = 0
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "model_config.json")
        if not os.path.isfile(path):
            continue
        try:
            cfg = json.load(open(path))
        except Exception:                                          # noqa: BLE001 - a truncated file
            continue
        if int(cfg.get("config_version", 0)) < 96:                 # pre-generation: refused by design
            continue
        out = _migrate_config(dict(cfg))
        assert out["config_version"] == MODEL_CONFIG_VERSION, name
        missing = [f.name for f in dataclasses.fields(ModelVersion) if f.name not in out]
        assert not missing, f"{name}: {missing}"
        ModelVersion(**out)
        seen += 1
    if seen == 0:
        pytest.skip("no current-generation configs in the archive")


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 5. THE Q HEAD × THE DISTILL FOLDS — the cf forward clobbers the FitNets hint
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_the_distill_hint_is_read_BEFORE_the_cf_forward_clobbers_the_stash():
    """Wave C put `q_winprob_on` / `q_onpolicy_on` into `cf_any_on`, so `_cf_sample_and_forward()`
    — an extractor forward over a DIFFERENT batch of rows — now runs on minibatches where no cf
    readout is configured at all. That forward overwrites `features_extractor.last_value_pooled`,
    which is the FitNets hint `--distill-value-feat-coef` matches against.

    It is correct today purely because `_s_vfeat` is captured EARLIER in `train()` than the cf
    block. That is a statement-ORDER fact, invisible to every behavioural test (the two never
    disagree unless a block moves), and the ledger already records one stash-clobber of exactly
    this shape. Pinned as source order, the same idiom `entry_source()` scans use."""
    from agents.training.instrumented_ppo import ppo as _ppo
    src = inspect.getsource(_ppo.InstrumentedMaskablePPO.train)
    hint = src.index("features_extractor.last_value_pooled")
    cf = src.index("_cf_sample_and_forward()")
    assert hint < cf, (
        "the FitNets hint is now read AFTER the cf forward — under --q-winprob-coef the "
        "distillation would match the teacher against the COUNTERFACTUAL batch's pooled features")


def test_the_cf_twin_fold_is_also_before_it():
    """The same order, stated for the twin heads, whose own comment claims it. A claim in a
    comment beside code nothing checks is how the first one shipped."""
    from agents.training.instrumented_ppo import ppo as _ppo
    src = inspect.getsource(_ppo.InstrumentedMaskablePPO.train)
    assert src.index("_cf_twin_onpolicy_terms") < src.index("_cf_sample_and_forward()")
