"""Public-value (V_pub) unit tests — the shared feature definition, the replay parser, the frozen
artifact round-trip, the LiveView fold, and the aux loss (gen3_pubval_aux_v1)."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agents.training.pubval import (
    PUBVAL_FEATURE_NAMES, PUBVAL_N_FEATURES, PubSide, PubValModel, features,
    parse_replay_log, pub_side_from_live,
)
from agents.training.instrumented_ppo import InstrumentedMaskablePPO


# ── features() — the single definition ────────────────────────────────────────

def _side(**kw):
    base = dict(alive=6, active_hp=1.0, known_hp=1.0, revealed=1, spikes=0, statusc=0, boost=0.0)
    base.update(kw)
    return PubSide(**base)


def test_feature_vector_order_and_math():
    me = _side(alive=5, active_hp=0.5, known_hp=2.5, revealed=4, spikes=2, statusc=1, boost=2.0)
    them = _side(alive=6, active_hp=1.0, known_hp=3.0, revealed=3, spikes=0, statusc=0, boost=-1.0)
    f = features(me, them, turn=25, weather="Sandstorm")
    assert f.shape == (PUBVAL_N_FEATURES,) and len(PUBVAL_FEATURE_NAMES) == PUBVAL_N_FEATURES
    exp = [-1.0, -0.5, -0.5, 2.0, 1.0, 3.0, 1.0,      # diffs (alive/active_hp/known_hp/spikes/statusc/boost/revealed)
           5.0, 6.0, 0.5, 1.0,                        # absolutes
           0.5,                                       # turn clock 25/50
           0.0, 1.0, 0.0, 0.0, 0.0]                   # weather one-hot (sandstorm)
    np.testing.assert_allclose(f, np.array(exp, np.float32), atol=1e-6)


def test_turn_clock_saturates_and_weather_normalizes():
    f = features(_side(), _side(), turn=200, weather=None)
    assert f[11] == 1.0                                # min(turn,50)/50 saturates
    assert f[12] == 1.0 and f[13:].sum() == 0          # None → 'none'
    f2 = features(_side(), _side(), turn=1, weather="RainDance")
    assert f2[14] == 1.0 and f2[12] == 0.0             # protocol-case string normalized


def test_symmetric_state_features_antisymmetric_diffs():
    a, b = _side(alive=4, known_hp=2.0), _side(alive=6, known_hp=4.5)
    fa, fb = features(a, b, 10, "none"), features(b, a, 10, "none")
    np.testing.assert_allclose(fa[:7], -fb[:7], atol=1e-6)   # the 7 diffs mirror exactly


# ── parse_replay_log — the corpus parser ──────────────────────────────────────

_FIXTURE = """
|player|p1|Alice||1600
|player|p2|Bob||1500
|rated|
|switch|p1a: Zap|Zapdos|100/100
|switch|p2a: Tar|Tyranitar|100/100
|-weather|Sandstorm
|turn|1
|move|p1a: Zap|Thunderbolt|p2a: Tar
|-damage|p2a: Tar|60/100
|-status|p2a: Tar|par
|-boost|p1a: Zap|spa|1
|-sidestart|p2: Bob|Spikes
|turn|2
|switch|p1a: Ske|Skarmory|100/100
|-damage|p1a: Ske|88/100
|turn|3
|move|p2a: Tar|Crunch|p1a: Ske
|-damage|p1a: Ske|0 fnt
|faint|p1a: Ske
|-curestatus|p2a: Tar|par
|turn|4
|win|Alice
"""


def test_parser_fixture_end_to_end():
    positions, winner, ratings, is_rated = parse_replay_log(_FIXTURE)
    assert winner == 0 and ratings == (1600, 1500) and is_rated
    assert [t for (t, *_ ) in positions] == [1, 2, 3, 4]

    t1_p1, t1_p2 = positions[0][1], positions[0][2]
    assert t1_p1 == PubSide(alive=6, active_hp=1.0, known_hp=1.0, revealed=1,
                            spikes=0, statusc=0, boost=0.0)
    assert positions[0][3] == "Sandstorm"

    # turn 2: Tar damaged + paralyzed, Zap boosted +1 spa, spikes on p2's side.
    t2_p1, t2_p2 = positions[1][1], positions[1][2]
    assert t2_p2.active_hp == pytest.approx(0.6) and t2_p2.statusc == 1 and t2_p2.spikes == 1
    assert t2_p1.boost == 1.0

    # turn 3: p1 switched (boosts reset, 2 revealed), Skarmory chipped by spikes.
    t3_p1 = positions[2][1]
    assert t3_p1.boost == 0.0 and t3_p1.revealed == 2
    assert t3_p1.known_hp == pytest.approx(1.0 + 0.88)

    # turn 4: Skarmory fainted (alive 5, its hp out of known_hp); Tar's status cured.
    t4_p1, t4_p2 = positions[3][1], positions[3][2]
    assert t4_p1.alive == 5 and t4_p1.known_hp == pytest.approx(1.0) and t4_p1.active_hp == 0.0
    assert t4_p2.statusc == 0


def test_parser_faint_clears_status_and_cureteam():
    log = """
|player|p1|A||1500
|player|p2|B||1500
|rated|
|switch|p1a: X|Snorlax|100/100
|switch|p2a: Y|Blissey|100/100
|-status|p1a: X|tox
|-status|p2a: Y|brn
|turn|1
|faint|p1a: X
|-cureteam|p2a: Y
|turn|2
|win|B
"""
    positions, winner, *_ = parse_replay_log(log)
    assert winner == 1
    t2_p1, t2_p2 = positions[1][1], positions[1][2]
    assert t2_p1.statusc == 0        # faint ends the status count (parity with live 'fnt')
    assert t2_p2.statusc == 0        # Heal Bell / Aromatherapy clears the side


def test_parser_unrated_or_no_winner_flagged():
    positions, winner, _, is_rated = parse_replay_log("|player|p1|A||\n|turn|1\n")
    assert winner is None and not is_rated


# ── PubValModel — artifact round-trip ─────────────────────────────────────────

def _toy_model():
    return PubValModel(mu=np.zeros(PUBVAL_N_FEATURES), sd=np.ones(PUBVAL_N_FEATURES),
                       w=np.r_[1.0, np.zeros(PUBVAL_N_FEATURES - 1)], b=0.0,
                       feature_names=PUBVAL_FEATURE_NAMES, meta={"n_games": 1})


def test_model_predict_and_roundtrip(tmp_path):
    m = _toy_model()
    f = np.zeros(PUBVAL_N_FEATURES, np.float32)
    assert m.predict(f) == pytest.approx(0.5)          # zero logit
    f[0] = 2.0                                          # alive_diff +2 → sigmoid(2)
    assert m.predict(f) == pytest.approx(1 / (1 + np.exp(-2.0)))
    p = tmp_path / "art.json"
    p.write_text(__import__("json").dumps(m.to_json()))
    m2 = PubValModel.load(str(p))
    assert m2.predict(f) == pytest.approx(m.predict(f)) and m2.meta["n_games"] == 1


def test_model_load_missing_is_instructive(tmp_path):
    with pytest.raises(FileNotFoundError, match="pubval_calibration"):
        PubValModel.load(str(tmp_path / "nope.json"))


def test_model_rejects_stale_feature_names():
    with pytest.raises(ValueError, match="stale"):
        PubValModel(mu=np.zeros(PUBVAL_N_FEATURES), sd=np.ones(PUBVAL_N_FEATURES),
                    w=np.zeros(PUBVAL_N_FEATURES), b=0.0,
                    feature_names=("bogus",) * PUBVAL_N_FEATURES)


def test_committed_artifact_loads_and_is_sane():
    """The repo's data/gen3_pubval.json must load through the runtime path and behave like a
    calibrated value: more alive mons for me ⇒ higher P(win)."""
    m = PubValModel.load()
    assert m.meta.get("auc_test", 0) > 0.70            # the POC-validated signal level
    up = features(_side(alive=6, known_hp=5.0), _side(alive=3, known_hp=1.0), 30, "none")
    down = features(_side(alive=3, known_hp=1.0), _side(alive=6, known_hp=5.0), 30, "none")
    assert m.predict(up) > 0.6 > 0.4 > m.predict(down)


# ── pub_side_from_live — the LiveView fold ────────────────────────────────────

def _mon(revealed=True, fainted=False, hp=1.0, status=None, boosts=None):
    return SimpleNamespace(revealed=revealed, fainted=fainted, hp_fraction=hp,
                           status=status, boosts=boosts or {})


def test_pub_side_from_live_semantics():
    active = _mon(hp=0.5, status="par", boosts={"atk": 2, "spe": -1})
    side = SimpleNamespace(
        mons=[active,
              _mon(hp=0.8, status="tox"),
              _mon(fainted=True, hp=0.0, status="fnt"),      # fainted: not alive, not statusc
              _mon(revealed=False),                          # unrevealed OWN mon: alive, no known_hp
              _mon(revealed=False), _mon(revealed=False)],
        active=active,
        side_conditions={"spikes": 2},
    )
    s = pub_side_from_live(side)
    assert s == PubSide(alive=5, active_hp=0.5, known_hp=pytest.approx(1.3), revealed=3,
                        spikes=2, statusc=2, boost=1.0)


def test_pub_side_from_live_no_active():
    side = SimpleNamespace(mons=[_mon()], active=None, side_conditions={})
    s = pub_side_from_live(side)
    assert s.active_hp == 0.0 and s.boost == 0.0


# ── _pubval_loss — the aux loss ───────────────────────────────────────────────

def test_pubval_loss_math_and_metrics():
    logits = torch.tensor([[0.0], [2.0]])
    target = torch.tensor([[0.5], [0.9]])
    mask = torch.ones(2, 1)
    loss, m = InstrumentedMaskablePPO._pubval_loss(logits, target, mask)
    expect = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.reshape(-1), target.reshape(-1))
    assert loss.item() == pytest.approx(expect.item())
    assert m["target_mean"] == pytest.approx(0.7)
    assert m["mae"] == pytest.approx((abs(0.5 - 0.5) + abs(torch.sigmoid(torch.tensor(2.0)).item() - 0.9)) / 2, abs=1e-6)
    assert m["coverage"] == 1.0


def test_pubval_loss_mask_excludes():
    logits = torch.tensor([[0.0], [100.0]])                 # the 2nd row is garbage…
    target = torch.tensor([[0.5], [0.0]])
    mask = torch.tensor([[1.0], [0.0]])                     # …but masked out
    loss, m = InstrumentedMaskablePPO._pubval_loss(logits, target, mask)
    expect = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(0.0), torch.tensor(0.5))
    assert loss.item() == pytest.approx(expect.item())
    assert m["coverage"] == pytest.approx(0.5)


def test_pubval_loss_none_guards():
    assert InstrumentedMaskablePPO._pubval_loss(None, torch.zeros(1, 1), torch.ones(1, 1)) is None
    assert InstrumentedMaskablePPO._pubval_loss(torch.zeros(1, 1), None, torch.ones(1, 1)) is None
    assert InstrumentedMaskablePPO._pubval_loss(
        torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1)) is None   # zero-mask minibatch


def test_pubval_loss_gradient_flows():
    logits = torch.zeros(3, 1, requires_grad=True)
    loss, _ = InstrumentedMaskablePPO._pubval_loss(logits, torch.full((3, 1), 0.8), torch.ones(3, 1))
    loss.backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0
