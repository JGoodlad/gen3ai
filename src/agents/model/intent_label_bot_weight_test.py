"""Gates for `--intent-label-bot-weight` (`gen3_intent_label_bot_weight_v1`).

The knob discounts the opponent-intent supervision (the X5 FLAT pointer's cross-entropy,
`agents.training.instrumented_ppo.flat_intent_fold`) on rows whose opponent was a heuristic BOT —
bots play strategies that are not the meta, and the self-play ramp makes early intent supervision
bot-DOMINATED, so the head can imprint on a decision tree before it ever faces a player. Four
properties carry the whole design, and each fails silently if it is not pinned:

1. **The default is BIT-identical** — a weight of 1.0 takes the unweighted reduction, exactly the
   term a run without `opp_class` computes.
2. **The weight actually differentiates** — hand-computed against a mixed batch, so a plumbing
   error that drops the weight on the floor cannot pass.
3. **It COMPOSES with the masks** rather than colliding with them — a masked row is dropped first
   and the weight multiplies only survivors, so no weight can resurrect a masked label and the
   denominator stays the supervised-row COUNT (not Σw, which would make a 100%-bot batch identical
   to an unweighted one — i.e. do nothing in exactly the regime this exists for).
4. **`label_bot_frac` is emitted** — the exposure number the weight is chosen off.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from agents.model.flat_intent import FlatIntentInputs, flat_width, slot_col
from agents.model.opp_intent import OPP_CLASS_BOT
from agents.training.instrumented_ppo.flat_intent_fold import flat_intent_fold_tensors

K, M, S, TEAM = 6, 400, 400, 6

#: (kind, num, switch_slot, switch_species) rows: three seat moves, a revealed switch, and a
#: NON-choice (kind 2), which the fold masks.
_ROWS = [(0, 89, -100, 0), (0, 57, -100, 0), (0, 237, -100, 0), (1, 0, 2, 0), (0, 57, -100, 0),
         (2, 0, -100, 0)]
#: their flat targets (the masked row's is unused)
_TGT = [0, 1, 2, slot_col(K, 2), 1, -1]


def _fi(B: int) -> FlatIntentInputs:
    """Seats [89, 57, 237, 0, 0, 0] with the first 3 on; slot 0 is the active; slots 1-5 live."""
    seat_nums = torch.tensor([[89, 57, 237, 0, 0, 0]]).expand(B, -1).clone()
    seat_on = torch.tensor([[True, True, True, False, False, False]]).expand(B, -1).clone()
    live = torch.ones(B, flat_width(K), dtype=torch.bool)
    live[:, 3:K] = False
    live[:, slot_col(K, 0)] = False
    return FlatIntentInputs(k=K, live=live, cand_ids=torch.zeros(B, flat_width(K), dtype=torch.long),
                            log_pi=torch.zeros(B, flat_width(K)), seat_nums=seat_nums, seat_on=seat_on,
                            hp_seat=torch.zeros(B, K, dtype=torch.bool), beyond=torch.zeros(B, M, dtype=torch.bool),
                            slot_species=torch.zeros(B, TEAM, dtype=torch.long),
                            slot_is_hypothesis=torch.zeros(B, TEAM, dtype=torch.bool),
                            in_tail=torch.zeros(B, S, dtype=torch.bool))



def _fold(logits, codes=None, w=1.0, rows=_ROWS):
    kind = torch.tensor([r[0] for r in rows])
    num = torch.tensor([r[1] for r in rows])
    slot = torch.tensor([r[2] for r in rows])
    sp = torch.tensor([r[3] for r in rows])
    oc = None if codes is None else torch.tensor(codes, dtype=torch.long).reshape(-1, 1)
    return flat_intent_fold_tensors(logits, _fi(len(rows)), kind=kind, num=num, switch_slot=slot,
                                    switch_species=sp, opp_class=oc, intent_coef=1.0,
                                    bot_label_weight=w)


def _logits(seed: int = 0, rows=_ROWS) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(len(rows), flat_width(K), generator=g)


def _per_row_ce(logits: torch.Tensor) -> torch.Tensor:
    """The supervised rows' CE, by hand (the masked last row excluded)."""
    sup = logits[:-1]
    return F.cross_entropy(sup, torch.tensor(_TGT[:-1]), reduction="none")


# ------------------------------------------------------------------ 1. the default is a no-op

@pytest.mark.parametrize("codes", [[0] * 6, [1] * 6, [0, 1, 0, 2, 3, 1]])
def test_weight_one_is_bit_identical_to_the_unweighted_loss(codes):
    x = _logits()
    assert torch.equal(_fold(x, codes, 1.0).intent_term, _fold(x, None, 1.0).intent_term)


def test_no_opp_class_means_no_weighting_at_any_weight():
    x = _logits()
    assert torch.equal(_fold(x, None, 0.25).intent_term, _fold(x, None, 1.0).intent_term)


# ------------------------------------------------------------------ 2. it differentiates

def test_a_mixed_batch_matches_the_hand_computed_weighted_mean():
    x = _logits()
    codes = [0, 1, 0, 2, 3, 0]
    w = torch.tensor([0.25, 1.0, 0.25, 1.0, 1.0])
    want = float((_per_row_ce(x) * w).sum() / 5)
    assert float(_fold(x, codes, 0.25).intent_term) == pytest.approx(want, rel=1e-6)


def test_an_all_bot_batch_is_scaled_down_rather_than_left_alone():
    """The denominator is the supervised-row COUNT: a 100%-bot batch at 0.25 is 0.25x the loss."""
    x = _logits()
    full = float(_fold(x, [OPP_CLASS_BOT] * 6, 1.0).intent_term)
    quarter = float(_fold(x, [OPP_CLASS_BOT] * 6, 0.25).intent_term)
    assert quarter == pytest.approx(0.25 * full, rel=1e-6)


@pytest.mark.parametrize("code", [1, 2, 3])
def test_non_bot_classes_are_never_discounted(code):
    x = _logits()
    assert torch.equal(_fold(x, [code] * 6, 0.25).intent_term, _fold(x, [code] * 6, 1.0).intent_term)


def test_zero_trains_on_no_bot_rows_at_all():
    x = _logits().requires_grad_(True)
    _fold(x, [0, 1, 0, 1, 1, 1], 0.0).intent_term.backward()
    assert float(x.grad[0].abs().sum()) == 0.0 and float(x.grad[2].abs().sum()) == 0.0
    assert float(x.grad[1].abs().sum()) > 0.0


def test_the_weight_reaches_the_gradient_proportionally():
    codes = [0, 1, 1, 1, 1, 1]
    grads = []
    for w in (1.0, 0.5):
        x = _logits().requires_grad_(True)
        _fold(x, codes, w).intent_term.backward()
        grads.append(x.grad[0].clone())
    assert torch.allclose(grads[1], 0.5 * grads[0], rtol=1e-6, atol=0.0)


# ------------------------------------------------------------------ 3. mask composition

@pytest.mark.parametrize("w", [0.0, 0.25, 1.0, 2.5])
def test_a_masked_row_receives_no_gradient_however_it_is_weighted(w):
    x = _logits().requires_grad_(True)
    out = _fold(x, [0] * 6, w)
    out.intent_term.backward()
    assert float(x.grad[-1].abs().sum()) == 0.0
    assert float(out.metrics["opp_intent/flat_n_supervised"][0]) == 5.0


def test_an_entirely_masked_batch_contributes_nothing():
    rows = [(2, 0, -100, 0)] * 3
    x = _logits(rows=rows).requires_grad_(True)
    out = _fold(x, [0] * 3, 0.25, rows=rows)
    assert float(out.intent_term.detach()) == 0.0 and not bool(out.intent_present)
    out.intent_term.backward()
    assert float(x.grad.abs().sum()) == 0.0


def test_a_nan_reaches_the_term_only_from_a_supervised_row():
    """K9(c) fail-closed: a masked row's logits are replaced BEFORE the softmax, so a NaN there
    cannot poison the term — and a NaN in a SUPERVISED row must still reach it."""
    for w in (1.0, 0.25):
        x = _logits()
        x[-1, 0] = float("nan")                          # the masked (non-choice) row
        assert math.isfinite(float(_fold(x, [0] * 6, w).intent_term))
        x = _logits()
        x[1, 3] = float("nan")                           # a supervised row
        assert math.isnan(float(_fold(x, [0] * 6, w).intent_term))


# ------------------------------------------------------------------ 4. the exposure metric

def test_label_bot_frac_is_the_bot_share_of_the_SUPERVISED_rows():
    m = _fold(_logits(), [0, 1, 0, 1, 1, 0], 1.0).metrics      # the bot row 5 is masked
    v, wt = m["opp_intent/label_bot_frac"]
    assert float(wt) == 1.0 and float(v) == pytest.approx(2 / 5)


def test_label_bot_frac_is_absent_without_opp_class():
    assert "opp_intent/label_bot_frac" not in _fold(_logits(), None, 1.0).metrics


# ------------------------------------------------------------------ the flag itself

def test_the_cli_flag_defaults_to_none_so_a_flagless_resume_can_inherit():
    """`_resolve` fills a `None` from the saved config; a hard default would OVERWRITE a value the
    run was already training with on every launcher restart."""
    from main.train_rl_agent import build_parser

    args = build_parser().parse_args([])
    assert args.intent_label_bot_weight is None


def test_a_negative_weight_is_refused():
    """Negative would train alpha/beta to be MAXIMALLY wrong about bots — the opposite of the
    flag's meaning. Training-only, so the CLI validation is the ONLY gate there is; pinned at the
    source, since the check sits mid-way through a function that resolves a whole run."""
    import inspect

    from main.train.config import resolve_config

    src = inspect.getsource(resolve_config)
    assert "args.intent_label_bot_weight is not None and args.intent_label_bot_weight < 0.0" in src
    assert "--intent-label-bot-weight must be >= 0" in src


@pytest.mark.parametrize("value", ["0", "0.25", "1", "2.5"])
def test_the_flag_parses_the_whole_intended_range(value):
    from main.train_rl_agent import build_parser

    args = build_parser().parse_args(["--intent-label-bot-weight", value])
    assert args.intent_label_bot_weight == float(value)


def test_the_weight_is_recorded_and_inherited_on_a_flagless_resume():
    """The training-only provenance class exactly: recorded on ModelVersion for provenance AND so
    `train_rl_agent`'s `_resolve` (a `getattr(saved_version, name, default)`) reads it back when a
    launcher restart forwards no flag. A weight that is NOT a ModelVersion field would silently
    revert to 1.0 — i.e. OFF — on every 3-hour restart."""
    import json

    from agents.model.model_version import (MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError,
                                            _migrate_config)
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    pk = {"net_arch": [512, 512]}
    v = ModelVersion.from_layout_and_policy_kwargs(layout, pk, intent_label_bot_weight=0.25)
    assert v.intent_label_bot_weight == 0.25
    assert ModelVersion(**json.loads(v.to_json())).intent_label_bot_weight == 0.25

    # NOT gated by check_compatible — a frozen eval / pool opponent never runs a loss,
    # so comparing it there would be a false rejection that breaks league play.
    other = ModelVersion.from_layout_and_policy_kwargs(layout, pk, intent_label_bot_weight=1.0)
    other.check_compatible(v)

    # THE v97 MIGRATION LEG IS NOW UNREACHABLE, and asserting the refusal is the honest form (the
    # v92 precedent). gen3_event_record_v2 bumped ARCH_SIGNATURE and raised
    # MIGRATION_FLOOR to 121 in the same commit, so v97's `setdefault("intent_label_bot_weight",
    # 1.0)` sits below the floor (archived verbatim in `_migrate_config`'s v97–v120 history). A
    # config that lacks the field is a pre-generation one and is REFUSED with a diagnosis.
    old = json.loads(v.to_json())
    old.pop("intent_label_bot_weight")
    old["config_version"] = 96
    with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
        _migrate_config(old)

    # The surviving property: every config at the floor RECORDS the weight explicitly (so the
    # flagless-resume read-back above is what matters), a fresh one records 1.0 = OFF, and a
    # recorded value passes through the current migration untouched.
    fresh = json.loads(ModelVersion.from_layout_and_policy_kwargs(layout, pk).to_json())
    assert fresh["config_version"] == MODEL_CONFIG_VERSION >= 121
    assert fresh["intent_label_bot_weight"] == 1.0
    migrated = _migrate_config(json.loads(v.to_json()))
    assert migrated["intent_label_bot_weight"] == 0.25
    assert migrated["config_version"] == MODEL_CONFIG_VERSION
