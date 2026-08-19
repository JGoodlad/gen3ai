"""Gates for `--intent-label-bot-weight` (`gen3_intent_label_bot_weight_v1`).

The knob discounts the opponent-intent (α/β) supervision on rows whose opponent was a heuristic
BOT — bots play strategies that are not the meta, and the self-play ramp makes early intent
supervision bot-DOMINATED, so the head can imprint on a decision tree before it ever faces a
player. Four properties carry the whole design, and each fails silently if it is not pinned:

1. **The default is BIT-identical**, not merely close — a weight of 1.0 must take the original
   unweighted `cross_entropy` call, because a "harmless" re-spelling of the reduction is exactly
   how a no-op flag stops being a no-op.
2. **The weight actually differentiates** — hand-computed against a mixed batch, so a plumbing
   error that drops the weight on the floor cannot pass.
3. **It COMPOSES with the masks** rather than colliding with them — a masked row is dropped first
   and the weight multiplies only survivors, so no weight can resurrect a masked label and the
   denominator stays the supervised-row COUNT (not Σw, which would make a 100%-bot batch identical
   to an unweighted one — i.e. do nothing in exactly the regime this exists for).
4. **`label_bot_frac` is emitted** — the exposure number the weight is chosen off.
"""
import pytest
import torch
import torch.nn.functional as F

from agents.model.opp_intent import (
    INTENT_IGNORE,
    OPP_CLASS_BOT,
    intent_label_weights,
    intent_losses,
)


def _alpha(n: int = 6, k: int = 4, seed: int = 0):
    """A deterministic (logits, target) alpha pair with all rows supervised."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, k, generator=g), torch.randint(0, k, (n,), generator=g)


def _classes(codes) -> torch.Tensor:
    return torch.tensor(codes, dtype=torch.long).reshape(-1, 1)


# ------------------------------------------------------------------ 1. the default is a no-op

@pytest.mark.parametrize("codes", [[0] * 6, [1] * 6, [0, 1, 0, 2, 3, 1]])
def test_weight_one_is_bit_identical_to_the_unweighted_loss(codes):
    """W=1.0 must reproduce the pre-flag loss EXACTLY — same bits, on every opponent mix."""
    logits, tgt = _alpha()
    reference = float(F.cross_entropy(logits, tgt, ignore_index=INTENT_IGNORE))
    loss, _ = intent_losses(logits, tgt, None, None, opp_class=_classes(codes),
                            bot_label_weight=1.0)
    assert float(loss) == reference          # bit equality, not approx


def test_the_flag_defaults_to_one_so_an_unset_run_is_unchanged():
    logits, tgt = _alpha()
    a, _ = intent_losses(logits, tgt, None, None, opp_class=_classes([0] * 6))
    b, _ = intent_losses(logits, tgt, None, None, opp_class=_classes([0] * 6),
                         bot_label_weight=1.0)
    assert float(a) == float(b) == float(F.cross_entropy(logits, tgt, ignore_index=INTENT_IGNORE))


def test_no_opp_class_means_no_weighting_at_any_weight():
    """A run whose env does not emit `opp_class` cannot be weighted — and must not silently
    weight everything as if it were a bot."""
    logits, tgt = _alpha()
    loss, _ = intent_losses(logits, tgt, None, None, opp_class=None, bot_label_weight=0.25)
    assert float(loss) == float(F.cross_entropy(logits, tgt, ignore_index=INTENT_IGNORE))


# ------------------------------------------------------------------ 2. it differentiates

def test_a_mixed_batch_matches_the_hand_computed_weighted_mean():
    """THE arithmetic gate: Σ w_i·ce_i / n_sup, with w=0.25 on the bot rows and 1.0 elsewhere."""
    logits, tgt = _alpha(n=6)
    codes = [0, 1, 0, 2, 3, 1]                        # 2 bot rows, 4 non-bot
    per = F.cross_entropy(logits, tgt, reduction="none")
    w = torch.tensor([0.25 if c == OPP_CLASS_BOT else 1.0 for c in codes])
    expected = float((per * w).sum() / 6)

    loss, _ = intent_losses(logits, tgt, None, None, opp_class=_classes(codes),
                            bot_label_weight=0.25)
    assert float(loss) == pytest.approx(expected, abs=1e-6)


def test_an_all_bot_batch_is_scaled_down_rather_than_left_alone():
    """The regime the knob exists for. Normalising by Σw instead of n_sup would make this batch
    identical to the unweighted one — i.e. the flag would do NOTHING during the bots-only ramp."""
    logits, tgt = _alpha()
    plain = float(F.cross_entropy(logits, tgt, ignore_index=INTENT_IGNORE))
    loss, _ = intent_losses(logits, tgt, None, None, opp_class=_classes([0] * 6),
                            bot_label_weight=0.25)
    assert float(loss) == pytest.approx(0.25 * plain, abs=1e-6)


def test_non_bot_classes_are_never_discounted():
    """pool / stable / exploiter keep weight 1.0 — only `bot` is the meta-mismatch case."""
    logits, tgt = _alpha()
    plain = float(F.cross_entropy(logits, tgt, ignore_index=INTENT_IGNORE))
    for code in (1, 2, 3):
        loss, _ = intent_losses(logits, tgt, None, None, opp_class=_classes([code] * 6),
                                bot_label_weight=0.0)
        assert float(loss) == pytest.approx(plain, abs=1e-6)


def test_zero_trains_on_no_bot_rows_at_all():
    logits, tgt = _alpha()
    loss, _ = intent_losses(logits, tgt, None, None, opp_class=_classes([0] * 6),
                            bot_label_weight=0.0)
    assert float(loss) == pytest.approx(0.0, abs=1e-7)
    # ... and the gradient is gone with it, not merely small.
    logits = logits.clone().requires_grad_(True)
    loss, _ = intent_losses(logits, tgt, None, None, opp_class=_classes([0] * 6),
                            bot_label_weight=0.0)
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_the_weight_reaches_the_gradient_proportionally():
    """A halved weight halves the gradient on a bot row and leaves a pool row untouched."""
    logits, tgt = _alpha(n=2)
    codes = _classes([OPP_CLASS_BOT, 1])

    def _grad(w):
        x = logits.clone().requires_grad_(True)
        loss, _ = intent_losses(x, tgt, None, None, opp_class=codes, bot_label_weight=w)
        loss.backward()
        return x.grad.clone()

    g1, g_half = _grad(1.0), _grad(0.5)
    assert torch.allclose(g_half[0], 0.5 * g1[0], atol=1e-6)   # the bot row
    assert torch.allclose(g_half[1], g1[1], atol=1e-6)         # the pool row


# ------------------------------------------------------------------ 3. mask composition

def test_a_masked_row_stays_masked_at_every_weight():
    """The mask runs FIRST. A masked bot row contributes nothing at w=1, w=0.25 or w=5.0, and the
    denominator counts only the surviving rows."""
    logits, tgt = _alpha(n=4)
    tgt = tgt.clone()
    tgt[0] = INTENT_IGNORE                            # a MASKED bot row
    codes = _classes([0, 0, 1, 1])

    per = F.cross_entropy(logits[1:], tgt[1:], reduction="none")
    for w in (1.0, 0.25, 5.0):
        wv = torch.tensor([w, 1.0, 1.0])              # row1 is bot, rows 2-3 are pool
        expected = float((per * wv).sum() / 3)        # denominator = 3 SURVIVORS, not 4
        loss, m = intent_losses(logits, tgt, None, None, opp_class=codes, bot_label_weight=w)
        assert float(loss) == pytest.approx(expected, abs=1e-6)
        assert m["opp_intent/alpha_n_supervised"] == 3.0


def test_a_masked_row_receives_no_gradient_however_it_is_weighted():
    logits, tgt = _alpha(n=4)
    tgt = tgt.clone()
    tgt[0] = INTENT_IGNORE
    x = logits.clone().requires_grad_(True)
    loss, _ = intent_losses(x, tgt, None, None, opp_class=_classes([0, 0, 1, 1]),
                            bot_label_weight=3.0)
    loss.backward()
    assert torch.count_nonzero(x.grad[0]) == 0
    assert torch.count_nonzero(x.grad[1:]) > 0


def test_an_entirely_masked_batch_contributes_nothing():
    logits, tgt = _alpha(n=3)
    tgt = torch.full_like(tgt, INTENT_IGNORE)
    loss, m = intent_losses(logits, tgt, None, None, opp_class=_classes([0, 0, 0]),
                            bot_label_weight=0.25)
    assert float(loss) == 0.0
    assert m["opp_intent/alpha_n_supervised"] == 0.0
    assert "opp_intent/label_bot_frac" not in m       # nothing was scored ⇒ no exposure to report


# ------------------------------------------------------------------ beta takes the same rule

def test_beta_is_weighted_by_the_same_per_row_vector():
    """β's supervised subset is only the voluntary switches, so its bot SHARE differs from α's —
    but the per-ROW weight cannot, since both grade the same opponent on the same decision."""
    g = torch.Generator().manual_seed(3)
    b_logits = torch.randn(4, 6, generator=g)
    b_tgt = torch.tensor([2, INTENT_IGNORE, 0, 4])    # row 1 is not a switch ⇒ masked
    codes = _classes([0, 0, 1, 0])

    sup = b_tgt != INTENT_IGNORE
    per = F.cross_entropy(b_logits[sup], b_tgt[sup], reduction="none")
    w = torch.tensor([0.25, 1.0, 0.25])               # surviving rows 0, 2, 3
    expected = float((per * w).sum() / 3)

    loss, _ = intent_losses(None, None, b_logits, b_tgt, opp_class=codes,
                            bot_label_weight=0.25)
    assert float(loss) == pytest.approx(expected, abs=1e-6)


def test_beta_alone_at_weight_one_is_bit_identical():
    g = torch.Generator().manual_seed(4)
    b_logits = torch.randn(5, 6, generator=g)
    b_tgt = torch.tensor([1, INTENT_IGNORE, 3, 0, INTENT_IGNORE])
    ref = float(F.cross_entropy(b_logits, b_tgt, ignore_index=INTENT_IGNORE))
    loss, _ = intent_losses(None, None, b_logits, b_tgt,
                            opp_class=_classes([0, 0, 1, 0, 1]), bot_label_weight=1.0)
    assert float(loss) == ref


def test_alpha_and_beta_are_both_weighted_when_both_are_present():
    """The total is the SUM of two independently weighted terms, not one weighted once."""
    a_logits, a_tgt = _alpha(n=4)
    g = torch.Generator().manual_seed(5)
    b_logits = torch.randn(4, 6, generator=g)
    b_tgt = torch.tensor([2, INTENT_IGNORE, 0, 4])
    codes = _classes([0, 0, 0, 0])

    total, _ = intent_losses(a_logits, a_tgt, b_logits, b_tgt, opp_class=codes,
                             bot_label_weight=0.25)
    la, _ = intent_losses(a_logits, a_tgt, None, None, opp_class=codes, bot_label_weight=0.25)
    lb, _ = intent_losses(None, None, b_logits, b_tgt, opp_class=codes, bot_label_weight=0.25)
    assert float(total) == pytest.approx(float(la) + float(lb), abs=1e-6)


# ------------------------------------------------------------------ 4. the exposure metric

def test_label_bot_frac_is_the_bot_share_of_the_SUPERVISED_alpha_rows():
    logits, tgt = _alpha(n=8)
    tgt = tgt.clone()
    tgt[0] = INTENT_IGNORE                            # a masked BOT row must not count either way
    codes = _classes([0, 0, 0, 1, 1, 1, 1, 1])
    _, m = intent_losses(logits, tgt, None, None, opp_class=codes)
    assert m["opp_intent/label_bot_frac"] == pytest.approx(2 / 7)


def test_label_bot_frac_is_emitted_even_with_the_weight_off():
    """The decision to SET the weight is made off this number, so it cannot be gated on it."""
    logits, tgt = _alpha()
    _, m = intent_losses(logits, tgt, None, None, opp_class=_classes([0] * 6),
                         bot_label_weight=1.0)
    assert m["opp_intent/label_bot_frac"] == pytest.approx(1.0)


def test_label_bot_frac_is_absent_without_opp_class():
    logits, tgt = _alpha()
    _, m = intent_losses(logits, tgt, None, None, opp_class=None)
    assert "opp_intent/label_bot_frac" not in m


def test_the_existing_stratified_metrics_are_untouched():
    """The whole `_bot`/`_pool` family must read exactly as it did before the weight existed —
    they measure the head, and a weighted LOSS must not move an accuracy."""
    logits, tgt = _alpha(n=8)
    codes = _classes([0, 0, 0, 0, 1, 1, 1, 1])
    _, off = intent_losses(logits, tgt, None, None, opp_class=codes, bot_label_weight=1.0)
    _, on = intent_losses(logits, tgt, None, None, opp_class=codes, bot_label_weight=0.25)
    shared = set(off) & set(on)
    moved = {k for k in shared if k != "opp_intent/alpha_loss" and off[k] != on[k]}
    assert not moved, f"the weight moved a diagnostic it must not touch: {sorted(moved)}"
    assert "opp_intent/alpha_acc_pool" in shared      # the gate metric is actually in this set


# ------------------------------------------------------------------ the weight-vector helper

def test_the_helper_returns_none_for_the_cases_that_take_the_original_call():
    like = torch.zeros(3, 2)
    assert intent_label_weights(None, 0.25, like) is None      # nothing to key on
    assert intent_label_weights(_classes([0, 1, 0]), 1.0, like) is None   # the no-op weight


def test_the_helper_marks_only_the_bot_code():
    like = torch.zeros(4, 2)
    w = intent_label_weights(_classes([0, 1, 2, 3]), 0.25, like)
    assert w.tolist() == [0.25, 1.0, 1.0, 1.0]


# ------------------------------------------------------------------ the flag itself

def test_the_cli_flag_defaults_to_none_so_a_flagless_resume_can_inherit():
    """`_resolve` fills a `None` from the saved config; a hard default would OVERWRITE a value the
    run was already training with on every launcher restart."""
    from main.train_rl_agent import build_parser

    args = build_parser().parse_args([])
    assert args.intent_label_bot_weight is None


def test_a_negative_weight_is_refused():
    """Negative would train alpha/beta to be MAXIMALLY wrong about bots — the opposite of the
    flag's meaning. Training-only, so the `main()` parser check is the ONLY gate there is; pinned
    at the source, since that check lives inside the async entry point and cannot be called."""
    import inspect

    from main import train_rl_agent

    src = inspect.getsource(train_rl_agent.main)
    assert "args.intent_label_bot_weight is not None and args.intent_label_bot_weight < 0.0" in src
    assert "--intent-label-bot-weight must be >= 0" in src


@pytest.mark.parametrize("value", ["0", "0.25", "1", "2.5"])
def test_the_flag_parses_the_whole_intended_range(value):
    from main.train_rl_agent import build_parser

    args = build_parser().parse_args(["--intent-label-bot-weight", value])
    assert args.intent_label_bot_weight == float(value)


def test_the_weight_is_recorded_and_inherited_on_a_flagless_resume():
    """The `td_aux_coef` class exactly: recorded on ModelVersion for provenance AND so
    `train_rl_agent`'s `_resolve` (a `getattr(saved_version, name, default)`) reads it back when a
    launcher restart forwards no flag. A weight that is NOT a ModelVersion field would silently
    revert to 1.0 — i.e. OFF — on every 3-hour restart."""
    import json

    from agents.model.model_version import (MODEL_CONFIG_VERSION, ModelVersion, _migrate_config)
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    pk = {"net_arch": [512, 512]}
    v = ModelVersion.from_layout_and_policy_kwargs(layout, pk, intent_label_bot_weight=0.25)
    assert v.intent_label_bot_weight == 0.25
    assert ModelVersion(**json.loads(v.to_json())).intent_label_bot_weight == 0.25

    # NOT gated by check_compatible — a frozen eval / pool / distill opponent never runs a loss,
    # so comparing it there would be a false rejection that breaks league play.
    other = ModelVersion.from_layout_and_policy_kwargs(layout, pk, intent_label_bot_weight=1.0)
    other.check_compatible(v)

    # The v97 migration IS reachable (unlike v92's): the field is new at 97 and MIGRATION_FLOOR is
    # 96, so a v96 config sits AT the floor and genuinely lacks it. Absent ⇒ 1.0 = OFF, which is
    # what every pre-v97 checkpoint trained under.
    old = json.loads(v.to_json())
    old.pop("intent_label_bot_weight")
    old["config_version"] = 96
    migrated = _migrate_config(old)
    assert migrated["intent_label_bot_weight"] == 1.0
    assert migrated["config_version"] == MODEL_CONFIG_VERSION
