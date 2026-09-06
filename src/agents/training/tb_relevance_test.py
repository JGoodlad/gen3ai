"""`gen3_tb_relevance_v1` — A TAG WHOSE SOURCE IS ABSENT IS NOT EMITTED.

The `--critic winprob` era (`designs/ai_v12/design_winprob_only_critic.md`) does not change any tag
NAME, but it removes the SOURCE behind several of them — PopArt is refused, the shaping composition
is one terminal indicator, and the scalar critic *is* the win-prob head. Four families then went on
publishing, as confident constants and byte-identical duplicates, on the first arm
(`models/ai_v12_01_winprob_critic`, 205 tags over its first hours):

| family | what it published | why it is content-free |
|---|---|---|
| `train/scaffolding_{gauge,rho,n}` | ρ = 1.0, gauge = 5.5e-13, both flat | V IS `sigmoid(win_prob_logit)`; a rank gauge between a quantity and itself is a tautology |
| `grad/win_prob_{share,norm_shared,policy_cosine}` | equal to `grad/value_*` to the last bit | the critic loss IS the win-prob BCE — the SAME tensor, reported twice AND double-counted in the shared denominator |
| `win_prob/{brier,acc}_contested`, `contested_{frac,label_mean}`, `brier_material`, `skill_vs_material` | copies of the pooled tags, plus a flat 1.0 and a flat 0.25 | `win_margin` is a MATERIAL-potential by-product, identically 0.0 with no material PBRS term, so nothing stratifies |
| `reward/{bias_refund,class_refund}_*` | six flat zeros | the refund is the BIAS class's mechanism and the composition has no bias term |

The rule these tests pin is one rule, applied four times: **the gate is on the SOURCE, never on the
value**, and turning a source off must leave a GAP in the curve rather than a confident number. Each
test therefore comes in a pair — the degenerate case is silent, the live case is unchanged — because
a gate that fires too widely is the same defect pointing the other way.

The two `slow` smokes at the bottom are the end-to-end statement: a real `--critic winprob` debug run
writes none of the NOISE tags and all of the LIVE ones, and a real `shaped` run's tag set is
BYTE-IDENTICAL to what it was before any of this landed (measured 2026-09-06: 172 tags, diff empty).

Run:
    pytest src/agents/training/tb_relevance_test.py -q
    pytest src/agents/training/tb_relevance_test.py -q -m slow      # the two real runs
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
import torch as th

from agents.training.grad_balance import grad_balance_metrics
from agents.training.instrumented_ppo.value_terms import ValueTerms
from agents.training.reward_term_stats import (
    reward_term_metrics,
    term_class_map,
    tracked_terms,
)
from agents.training.scaffolding import live_gauge_metrics
from utils.paths import repo_path, src_path

# ═══════════════════════════════════════════════════ G1 — grad_balance: value counted ONCE ══


def _trunk_and_terms():
    """A two-parameter 'shared trunk' plus three loss terms built on it, with a live graph."""
    th.manual_seed(0)
    w = th.nn.Parameter(th.randn(4, 3))
    b = th.nn.Parameter(th.randn(3))
    x = th.randn(8, 4)
    h = x @ w + b
    return [w, b], h


def test_an_aux_term_that_is_the_value_term_is_reported_once():
    """`--critic winprob` passes ONE tensor as both the critic loss and `aux_terms['win_prob']`."""
    shared, h = _trunk_and_terms()
    policy = (h ** 2).mean()
    value = (h.abs()).mean() * 3.0          # the critic term…
    other = (h ** 4).mean() * 0.1
    out = grad_balance_metrics(policy, value, shared,
                               aux_terms={"win_prob": value, "hp_type": other})  # …and the aux
    assert "grad/win_prob_share" not in out
    assert "grad/win_prob_norm_shared" not in out
    assert "grad/win_prob_policy_cosine" not in out
    # The shares still partition to 1 — but so did they WHILE double-counting, so that identity
    # alone proves nothing. The denominator itself is checked in the next test.
    assert out["grad/policy_share"] + out["grad/value_share"] + out["grad/aux_share"] == pytest.approx(1.0)
    assert out["grad/aux_share"] > 0.0      # `other` is still an aux


def test_the_denominator_no_longer_double_counts_the_critic_gradient():
    """THE DEFECT, arithmetically: every `grad/*_share` was deflated by the critic's own norm.

    Reproduces the published shape of `models/ai_v12_01_winprob_critic` — policy 0.4555, value
    0.1380, win_prob 0.1380 (the same number), hp_type 0.0723, move_latent 0.0039, published
    policy_share 0.5639 — where the true share over a de-duplicated denominator is 0.680.
    """
    shared, h = _trunk_and_terms()
    policy = (h ** 2).mean()
    value = (h.abs()).mean() * 3.0
    other = (h ** 4).mean() * 0.1
    out = grad_balance_metrics(policy, value, shared,
                               aux_terms={"win_prob": value, "hp_type": other})
    n_pi = out["grad/policy_norm_shared"]
    n_vf = out["grad/value_norm_shared"]
    n_aux = out["grad/hp_type_norm_shared"]
    assert out["grad/policy_share"] == pytest.approx(n_pi / (n_pi + n_vf + n_aux), rel=1e-6)
    # The pre-fix denominator would have carried n_vf twice; assert we are NOT on that number.
    deflated = n_pi / (n_pi + 2 * n_vf + n_aux)
    assert out["grad/policy_share"] > deflated


def test_a_distinct_aux_term_is_still_reported():
    """The negative control: identity, not value equality — a separate tensor stays a term."""
    shared, h = _trunk_and_terms()
    policy = (h ** 2).mean()
    value = (h.abs()).mean() * 3.0
    twin = (h.abs()).mean() * 3.0           # numerically identical, a DIFFERENT tensor
    out = grad_balance_metrics(policy, value, shared, aux_terms={"win_prob": twin})
    assert "grad/win_prob_share" in out
    assert out["grad/win_prob_norm_shared"] == pytest.approx(out["grad/value_norm_shared"])


# ══════════════════════════════════════ G2 — the scaffolding gauge against itself ══


def test_gauge_is_silent_when_V_is_a_monotone_map_of_the_logit():
    """`--critic winprob`: V = sigmoid(z), so ρ = 1 by construction and there is nothing to read."""
    rng = np.random.default_rng(0)
    z = rng.normal(size=512)
    v = 1.0 / (1.0 + np.exp(-z))            # exactly what `_critic_value` returns in that mode
    assert live_gauge_metrics(v, z) == {}


def test_gauge_is_silent_for_any_monotone_map_not_just_the_sigmoid():
    rng = np.random.default_rng(1)
    z = rng.normal(size=256)
    assert live_gauge_metrics(np.exp(z), z) == {}
    assert live_gauge_metrics(3.0 * z - 7.0, z) == {}


def test_gauge_still_reads_two_genuinely_different_readouts():
    """The negative control: a `shaped` run's V and win-prob head order states differently."""
    rng = np.random.default_rng(2)
    z = rng.normal(size=512)
    v = z + rng.normal(size=512)            # correlated but not a monotone map
    out = live_gauge_metrics(v, z)
    assert set(out) == {"scaffolding_gauge", "scaffolding_rho", "scaffolding_n"}
    assert 0.0 < out["scaffolding_rho"] < 1.0
    assert out["scaffolding_n"] == 512.0


def test_gauge_absence_is_still_reported_for_no_head_and_no_rows():
    """The pre-existing silences are untouched — 'no head' and 'no rows' both leave a gap."""
    assert live_gauge_metrics(None, None) == {}
    assert live_gauge_metrics(np.array([]), np.array([])) == {}


def test_a_tiny_monotone_sample_is_a_COINCIDENCE_and_is_still_published():
    """The gate claims SAMENESS, so it needs enough rows for agreement to mean something.

    Two unrelated readouts agree on the order of n distinct values with probability 1/n! — 1-in-6
    at n=3. Suppressing there would silence a real (if thin) reading; a live rollout carries ~1e5
    paired rows, so the floor costs the production path nothing. This is the case the NaN-safety
    test in `scaffolding_test.py` exercises.
    """
    v = np.array([1.0, 2.0, 5.0])
    z = np.array([0.1, 0.2, 0.5])
    out = live_gauge_metrics(v, z)
    assert out["scaffolding_n"] == 3.0
    assert out["scaffolding_gauge"] == pytest.approx(0.0)


# ═══════════════════════════════════ G3 — a CONSTANT margin cannot stratify ══

_CONTESTED_KEYS = ("contested_frac", "brier_contested", "acc_contested",
                   "contested_label_mean", "brier_material", "skill_vs_material")


def _win_prob_metrics(margin):
    th.manual_seed(3)
    logits = th.randn(64, 1)
    target = (th.rand(64, 1) < 0.5).float()
    mask = th.ones(64, 1)
    out = ValueTerms._win_prob_loss(logits, target, mask, margin)
    assert out is not None
    return out[1]


def test_a_constant_zero_margin_publishes_no_contested_split():
    """No material PBRS term ⇒ `_last_material_margin` never moves ⇒ margin is identically 0."""
    m = _win_prob_metrics(th.zeros(64, 1))
    for k in _CONTESTED_KEYS:
        assert k not in m, f"{k} published against a margin with no spread"
    assert {"loss", "acc", "brier", "pred_mean", "label_mean", "coverage"} <= set(m)


def test_a_constant_NONZERO_margin_also_publishes_nothing():
    """The gate is on SPREAD, not on the value — a pinned non-zero margin stratifies just as little."""
    m = _win_prob_metrics(th.full((64, 1), 0.4))
    for k in _CONTESTED_KEYS:
        assert k not in m


def test_a_real_margin_publishes_the_whole_contested_family():
    """The negative control: a `shaped` run's margin varies and every tag comes back."""
    th.manual_seed(4)
    m = _win_prob_metrics(th.rand(64, 1) * 2.0 - 1.0)
    for k in _CONTESTED_KEYS:
        assert k in m, f"{k} missing on a margin that DOES vary"
    assert 0.0 < m["contested_frac"] < 1.0


def test_an_absent_margin_is_unchanged():
    m = _win_prob_metrics(None)
    for k in _CONTESTED_KEYS:
        assert k not in m


# ══════════════════════════ G4 — the refund is the BIAS class's mechanism ══

_TERMINAL_ONLY = {"terminal": 1, "pbrs": 0, "bias": 0,
                  "terminal_terms": ["win_loss"], "pbrs_terms": [], "bias_terms": []}
_PRODUCTION = {"terminal": 1, "pbrs": 7, "bias": 1,
               "terminal_terms": ["win_loss"],
               "pbrs_terms": [f"p{i}" for i in range(7)],
               "bias_terms": ["no_progress_tax"]}


def test_a_composition_with_no_bias_term_does_not_track_the_refund():
    assert "bias_refund" not in tracked_terms(_TERMINAL_ONLY)
    assert "refund" not in term_class_map(_TERMINAL_ONLY).values()


def test_the_production_composition_still_tracks_the_refund():
    """The negative control — a shaped run's tag set must not move."""
    assert tracked_terms(_PRODUCTION)[-1] == "bias_refund"
    assert term_class_map(_PRODUCTION)["bias_refund"] == "refund"


def test_the_count_and_the_term_list_agree_and_either_alone_suffices():
    """`reward_class_composition` emits both; a hand-built census may carry only one."""
    assert "bias_refund" in tracked_terms({"terminal_terms": ["w"], "bias_terms": ["t"]})
    assert "bias_refund" not in tracked_terms({"terminal_terms": ["w"], "bias": 0})


def test_no_refund_curves_are_exported_for_a_terminal_only_run():
    """End to end through the exporter: six flat zeros become six absent tags."""
    tc = term_class_map(_TERMINAL_ONLY)
    merged = {"n": 100, "total_sum": 53.0, "total_abs_sum": 53.0, "residual_abs_sum": 0.0,
              "sum": {"win_loss": 53.0}, "abs": {"win_loss": 53.0}}
    out = reward_term_metrics(merged, tc)
    assert not [k for k in out if "refund" in k]
    assert out["win_loss_abs_share"] == pytest.approx(1.0)
    assert out["untracked_abs_mean"] == 0.0          # the GIGO guard is untouched


def test_an_unexpected_refund_now_reaches_the_GIGO_guard():
    """Dropping the refund from the tracked set STRENGTHENS `untracked_abs_mean`.

    A `bias_refund` that somehow became non-zero with no bias term in the census is no longer
    absorbed into a curve nobody reads — it shows up as an untracked residual, which is what that
    scalar exists to say.
    """
    tc = term_class_map(_TERMINAL_ONLY)
    merged = {"n": 100, "total_sum": 63.0, "total_abs_sum": 63.0, "residual_abs_sum": 10.0,
              "sum": {"win_loss": 53.0}, "abs": {"win_loss": 53.0}}
    out = reward_term_metrics(merged, tc)
    assert out["untracked_abs_mean"] == pytest.approx(0.1)


# ═══════════════════════════════════════════ THE SMOKES — real runs, real tfevents ══

#: Emitted on a `--critic winprob` run before `gen3_tb_relevance_v1` and content-free there.
NOISE_TAGS = (
    "train/scaffolding_gauge", "train/scaffolding_rho", "train/scaffolding_n",
    "grad/win_prob_share", "grad/win_prob_norm_shared", "grad/win_prob_policy_cosine",
    "win_prob/brier_contested", "win_prob/acc_contested", "win_prob/contested_frac",
    "win_prob/contested_label_mean", "win_prob/brier_material", "win_prob/skill_vs_material",
    "reward/bias_refund_mean", "reward/bias_refund_abs_mean", "reward/bias_refund_abs_share",
    "reward/class_refund_mean", "reward/class_refund_abs_mean", "reward/class_refund_abs_share",
)

#: The reader's floor on EITHER critic mode — tags that must survive whatever the gates do.
LIVE_TAGS = (
    "train/explained_variance", "train/value_loss", "train/return_mean", "train/return_std",
    "train/approx_kl", "train/clip_fraction", "train/entropy_loss", "train/grad_norm",
    "win_prob/brier", "win_prob/ece", "win_prob/mce", "win_prob/acc", "win_prob/coverage",
    "win_prob/pred_mean", "win_prob/label_mean", "win_prob/start_gap",
    "signal/adv_raw_mean", "signal/adv_raw_std", "signal/outcome_win_rate",
    "signal/outcome_entropy", "signal/draw_rate",
    "reward/total_mean", "reward/untracked_abs_mean", "reward/n_decisions",
    "grad/policy_share", "grad/value_share", "grad/value_policy_logratio",
    "rollout/ep_len_mean", "rollout/ep_rew_mean",
)

#: CONDITIONAL, and the condition is the CRITIC MODE itself — the DEPLOYED value's own Murphy
#: split, which only exists when the deployed value is the win-prob head (`ppo.py`'s
#: `if critic_winprob:`). Present on a winprob run (where `critic_resolution` is the G1 PRIMARY
#: meter) and correctly ABSENT on a shaped one, where the critic is a different readout entirely.
CRITIC_MODE_TAGS = (
    "win_prob/critic_resolution", "win_prob/critic_reliability", "win_prob/critic_skill",
    "win_prob/critic_brier", "win_prob/critic_uncertainty", "win_prob/critic_base_rate",
    "win_prob/critic_ece", "win_prob/critic_mce", "win_prob/critic_n",
    "win_prob/critic_decomp_residual",
)

_WINPROB_ARGV = ["--critic", "winprob", "--no-hand-shaping", "--terminal-indicator",
                 "--victory-value", "1.0", "--draw-penalty", "0"]
_SHAPED_ARGV = ["--win-prob-mode", "read_only"]


def _run_smoke(tmp_path, name, extra):
    """One short CPU `--debug` run into `tmp_path/models/<name>`; returns its scalar tag set.

    Runs in a SCRATCH cwd, never the checkout: `--run-name X` writes to a cwd-relative
    ``models/X``, and in the main checkout that is the read-only run archive. `TeamLoader`'s
    ``base_dir`` is cwd-relative too (``"data/teams"``), so the scratch dir carries symlinks to the
    checkout's `data/` and `deps/` — the two directories the trainer reaches for by relative path.
    """
    for d in ("data", "deps"):
        link = tmp_path / d
        if not link.exists():
            link.symlink_to(repo_path(d))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_path()) + os.pathsep + env.get("PYTHONPATH", "")
    argv = [sys.executable, str(src_path("main", "train_rl_agent.py")),
            "--debug", "--steps", "3000", "--n-steps", "256", "--batch-size", "128",
            "--n-epochs", "2", "--run-name", name, *extra]
    proc = subprocess.run(argv, cwd=str(tmp_path), env=env, capture_output=True, text=True)
    assert proc.returncode == 0, f"smoke failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(str(tmp_path / "models" / name / "tb"), size_guidance={"scalars": 0})
    acc.Reload()
    return set(acc.Tags()["scalars"])


@pytest.mark.slow
def test_a_winprob_run_emits_no_noise_tag_and_every_live_tag(tmp_path):
    tags = _run_smoke(tmp_path, "tbrel_wp", _WINPROB_ARGV)
    leaked = sorted(t for t in NOISE_TAGS if t in tags)
    assert not leaked, f"content-free tags published on a winprob run: {leaked}"
    missing = sorted(t for t in LIVE_TAGS + CRITIC_MODE_TAGS if t not in tags)
    assert not missing, f"the gate swallowed tags that are LIVE on a winprob run: {missing}"


@pytest.mark.slow
def test_a_shaped_run_keeps_every_tag_the_gates_touch(tmp_path):
    """The shaped tag set must not move: every gated family has its source ON here.

    Byte-identical was verified by measurement (2026-09-06: 172 tags, empty diff before/after);
    this pins the half that could regress, which is a gate firing where its source is live.
    """
    tags = _run_smoke(tmp_path, "tbrel_shaped", _SHAPED_ARGV)
    for tag in ("train/scaffolding_gauge", "train/scaffolding_rho", "train/scaffolding_n",
                "win_prob/brier_contested", "win_prob/acc_contested", "win_prob/contested_frac",
                "win_prob/contested_label_mean", "win_prob/brier_material",
                "win_prob/skill_vs_material",
                "reward/bias_refund_mean", "reward/bias_refund_abs_mean",
                "reward/class_refund_mean", "reward/class_refund_abs_mean"):
        assert tag in tags, f"{tag} lost on a SHAPED run — a gate fired where its source is live"
    missing = sorted(t for t in LIVE_TAGS if t not in tags)
    assert not missing, f"shaped run missing: {missing}"
    # The critic's own Murphy split belongs to the OTHER mode, and its absence here is the
    # CONDITIONAL half of the contract — not something the gates took away.
    assert not [t for t in CRITIC_MODE_TAGS if t in tags]


# ════════════════════════════════════════════ the census stays honest ══


def test_the_census_table_carries_an_era_column():
    """The training leaf's census is the list this pass classified; the column must survive."""
    leaf = repo_path("src", "agents", "training", "CLAUDE.md").read_text()
    assert "gen3_tb_relevance_v1" in leaf
    assert "What to watch on a WIN-PROB run" in leaf
