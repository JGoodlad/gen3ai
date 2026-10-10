"""Unit tests for the representation probe battery (`gen3_probe_battery_v1`): deterministic folds BY BATTLE, the
ridge probe, the Hewitt & Liang control task's selectivity, the deterministic bank selection, the facts' boundary
exclusions (rule 8), and the refusal of ``models/`` as an output."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from main.probe_battery import bank as B
from main.probe_battery import facts as F
from main.probe_battery import probes as P
from main.probe_battery import report as R
from main.probe_battery.capture import parse_spec


# ------------------------------------------------------------------------------------------------- folds
def test_folds_are_by_battle_and_independent_of_row_order():
    battles = [f"g{i % 37}" for i in range(500)]
    f1 = P.battle_folds(battles)
    # every row of a battle shares its fold
    for b in set(battles):
        assert len({int(f1[i]) for i, x in enumerate(battles) if x == b}) == 1
    # reversing the rows moves no battle's fold (a function of the id, not the position)
    rev = list(reversed(battles))
    f2 = P.battle_folds(rev)
    assert [int(x) for x in f2] == [int(x) for x in reversed(f1)]
    # pinned: the fold is sha256(id)[:8] mod 5
    assert int(P.battle_folds(["L1xS1#0"])[0]) == int(__import__("hashlib").sha256(b"L1xS1#0").hexdigest()[:8], 16) % 5
    assert set(int(x) for x in f1) == {0, 1, 2, 3, 4}


# ------------------------------------------------------------------------------------------------- probe
def _species_world(n=6000, n_species=400, d=64, seed=0):
    rng = np.random.default_rng(seed)
    sp = rng.integers(0, n_species, n)
    emb = rng.normal(size=(n_species, d))           # the representation knows the species (a lookup)
    sig = rng.normal(size=n)                        # a state fact, independent of the species
    w = rng.normal(size=d)
    X = emb[sp] + 0.05 * rng.normal(size=(n, d))
    X[:, 0] += 3.0 * sig                            # ...and carries the state fact on one axis
    battles = [f"b{i // 30}" for i in range(n)]
    return X.astype(np.float32), sp, sig, w, battles


def test_ridge_oof_reads_a_linear_fact_and_not_noise():
    X, sp, sig, _, battles = _species_world()
    folds = P.battle_folds(battles)
    pred = P.ridge_oof(X, np.stack([sig, np.random.default_rng(9).normal(size=len(sig))], 1), folds)
    assert P.r2(sig, pred[:, 0]) > 0.85
    assert P.r2(np.random.default_rng(9).normal(size=len(sig)), pred[:, 1]) < 0.05


def test_control_task_makes_a_species_memorising_probe_unselective():
    """With more species (400) than dimensions (64) the probe can memorise species only PARTLY: a fact that is a
    function of the species alone reads about as well as its CONTROL (a random function of the same species) —
    selectivity ~0 — while a genuine state fact reads far above its control."""
    X, sp, sig, _, battles = _species_world()
    folds = P.battle_folds(battles)
    valid = np.ones(len(sp), bool)
    species_fact = (np.random.default_rng(3).normal(size=400))[sp]           # determined by the species
    c_species = P.control_labels(species_fact, valid, sp, "c", "control:species_fact")
    c_state = P.control_labels(sig, valid, sp, "c", "control:state_fact")
    pred = P.ridge_oof(X, np.stack([species_fact, c_species, sig, c_state], 1), folds)
    sel_species = P.r2(species_fact, pred[:, 0]) - P.r2(c_species, pred[:, 1])
    sel_state = P.r2(sig, pred[:, 2]) - P.r2(c_state, pred[:, 3])
    assert 0.1 < P.r2(c_species, pred[:, 1]) < 0.9    # the probe memorises species PARTLY → a control scores
    assert abs(sel_species) < 0.15                    # so a species-determined fact is NOT selective
    assert sel_state > 0.5                            # a state fact is


def test_control_labels_keep_the_marginal_and_are_a_function_of_the_key():
    y = np.array([0.0, 1.0] * 500)
    key = np.arange(1000) % 50
    c = P.control_labels(y, np.ones(1000, bool), key, "b", "control:x")
    for k in range(50):
        assert len(set(c[key == k].tolist())) == 1
    assert set(np.unique(c).tolist()) <= {0.0, 1.0}
    c2 = P.control_labels(y, np.ones(1000, bool), key, "b", "control:x")
    assert np.array_equal(c, c2)                      # seeded by the fact's name: deterministic


def test_species_lookup_baseline_reads_a_species_fact_and_not_a_state_fact():
    X, sp, sig, _, battles = _species_world()
    folds = P.battle_folds(battles)
    species_fact = (np.random.default_rng(3).normal(size=400))[sp]
    valid = np.ones(len(sp), bool)
    assert P.species_lookup(species_fact, valid, sp, folds, "c") > 0.8
    assert P.species_lookup(sig, valid, sp, folds, "c") < 0.05


def test_auc_by_ranks():
    assert P.auc(np.array([0, 0, 1, 1.0]), np.array([0.1, 0.2, 0.3, 0.4])) == 1.0
    assert P.auc(np.array([0, 1, 0, 1.0]), np.array([1.0, 1.0, 1.0, 1.0])) == 0.5


# ------------------------------------------------------------------------------------------------- refusal
def test_refuses_models_as_an_output(tmp_path, monkeypatch):
    arch = tmp_path / "archive_models"
    arch.mkdir()
    monkeypatch.setenv("GEN3AI_MODELS_DIR", str(arch))
    with pytest.raises(B.BatteryError):
        B.refuse_models_output(arch / "probe_out")
    # a checkout's own models/ (a worktree twin) is refused too
    co = tmp_path / "checkout"
    (co / "src").mkdir(parents=True)
    with pytest.raises(B.BatteryError):
        B.refuse_models_output(co / "models" / "x")
    B.refuse_models_output(tmp_path / "elsewhere")   # anything else is fine


def test_spec_parsing():
    assert parse_spec("L1=/a/b.zip") == ("L1", "/a/b.zip", None)
    assert parse_spec("Lr2=/a/b.zip@rand7") == ("Lr2", "/a/b.zip", 7)
    with pytest.raises(B.BatteryError):
        parse_spec("L1=/a/run_dir")
    with pytest.raises(B.BatteryError):
        parse_spec("/a/b.zip")


# ---------------------------------------------------------------------------------------------- selection
def _dec(i, arch, turn, legal=3):
    mons = [{"species": f"s{k}", "fainted": False} for k in range(6)]
    return {"game": f"g{i // 10}", "side": i % 2, "n": i, "mask": [1] * legal + [0] * (11 - legal),
            "archetype": arch, "phase": "opening" if turn <= 3 else "midgame",
            "V": {"turn": turn, "ours": {"mons": mons, "team_size": 6}, "opp": {"mons": mons, "team_size": 6}}}


def test_selection_is_deterministic_balanced_and_drops_single_legal():
    decs = [_dec(i, B.ARCHETYPES[i % 5], 1 + i % 20, legal=1 if i % 17 == 0 else 3) for i in range(2000)]
    a = B.select(decs, target=500, bank_seed=11)
    b = B.select(list(decs), target=500, bank_seed=11)
    assert a == b and len(a) == 500
    assert all(sum(decs[i]["mask"]) >= 2 for i in a)
    from collections import Counter

    assert set(Counter(decs[i]["archetype"] for i in a).values()) == {100}
    assert B.select(decs, target=500, bank_seed=12) != a          # the seed is the only thing that moves it


# ------------------------------------------------------------------------------------------------- facts
def _mon(sp, *, types, stats, hp=300, mx=300, boosts=None, moves=(), item="leftovers", ability="", status=None,
         active=False, vol=None):
    return {"species": sp, "types": list(types), "stats": dict(stats, hp=mx), "current_hp": hp, "max_hp": mx,
            "hp_fraction": hp / mx, "boosts": boosts or {}, "item": item, "ability": ability, "status": status,
            "active": active, "fainted": False, "volatiles": vol or {}, "status_counter": 0,
            "moves": [{"id": m, "move_id": m, "current_pp": 10, "max_pp": 10} for m in moves]}


ST = {"atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200}


def _board(ours, theirs, *, our_spikes=0, their_spikes=0, seen_moves=()):
    seen = copy.deepcopy(theirs)
    seen["moves"] = [{"id": m, "move_id": m, "current_pp": 10, "max_pp": 10} for m in seen_moves]
    bench = _mon("blissey", types=["normal"], stats=ST, moves=("softboiled",))
    flyer = _mon("skarmory", types=["steel", "flying"], stats=ST, moves=("spikes",))
    V = {"turn": 5, "weather": {"weather": None, "is_permanent": False, "turns_active": 0},
         "ours": {"team_size": 6, "active": ours["species"], "mons": [ours, bench, flyer],
                  "side_conditions": {"spikes": our_spikes} if our_spikes else {}},
         "opp": {"team_size": 6, "active": theirs["species"], "mons": [seen],
                 "side_conditions": {"spikes": their_spikes} if their_spikes else {}}}
    W = {"turn": 5, "weather": V["weather"], "ours": {"team_size": 6, "active": theirs["species"], "mons": [theirs],
                                                     "side_conditions": V["opp"]["side_conditions"]},
         "opp": {"team_size": 6, "active": ours["species"], "mons": [], "side_conditions": {}}}
    return {"game": "g", "side": 0, "n": 0, "V": V, "W": W, "legal": {}, "opp_legal": {}, "winner": "p1",
            "our_slots": [ours["species"], "blissey", "skarmory"], "opp_choice": "move roar", "wish": [False, False]}


def _facts(d):
    facts, vals, valid, _ = F.extract([d])
    return {f.name: (float(vals[0, j]) if valid[0, j] else None) for j, f in enumerate(facts)}


def test_speed_tie_is_excluded_and_order_read_from_true_stats():
    a = _mon("a", types=["water"], stats=ST, active=True)
    b = _mon("b", types=["water"], stats=ST)
    assert _facts(_board(a, b))["we_move_first"] is None                     # a tie is never decided by chance
    b2 = _mon("b", types=["water"], stats=dict(ST, spe=150))
    assert _facts(_board(a, b2))["we_move_first"] == 1.0
    b3 = _mon("b", types=["water"], stats=dict(ST, spe=150), boosts={"spe": 2})   # +2: 150 -> 300
    assert _facts(_board(a, b3))["we_move_first"] == 0.0


def test_ko_band_excludes_rows_near_the_boundary():
    a = _mon("a", types=["ground"], stats=ST, active=True, moves=("earthquake",))
    big = _facts(_board(a, _mon("b", types=["electric"], stats=ST, hp=10, mx=300)))
    assert big["we_can_ko"] == 1.0
    tank = _facts(_board(a, _mon("b", types=["water"], stats=ST, hp=300, mx=300)))
    assert tank["we_can_ko"] == 0.0
    ratio = tank["our_best_dmg"]
    hp_at_boundary = int(round(300 * ratio))                                 # max roll == HP: inside the band
    edge = _facts(_board(a, _mon("b", types=["water"], stats=ST, hp=hp_at_boundary, mx=300)))
    assert edge["we_can_ko"] is None
    immune = _facts(_board(a, _mon("b", types=["flying"], stats=ST)))
    assert immune["our_best_dmg"] == 0.0


def test_phazing_facts():
    me = _mon("a", types=["water"], stats=ST, active=True, moves=("roar", "surf"))
    foe = _mon("b", types=["water"], stats=ST, boosts={"spa": 2}, moves=("calmmind", "whirlwind"))
    d = _board(me, foe, their_spikes=2, seen_moves=("calmmind",))
    f = _facts(d)
    assert f["OA_phazer"] == 1.0 and f["TA_phazer_true"] == 1.0 and f["TA_phazer_revealed"] == 0.0
    assert f["TA_hidden_phazer"] == 1.0
    assert f["phaze_value"] == 1.0 and f["phaze_value_tool"] == 1.0
    f2 = _facts(_board(me, foe, their_spikes=0, seen_moves=("calmmind",)))
    assert f2["phaze_value"] == 0.0                                           # no Spikes: no chip on the drag
    me_b = _mon("a", types=["water"], stats=ST, active=True, moves=("calmmind",), boosts={"spa": 1})
    f3 = _facts(_board(me_b, foe, seen_moves=("whirlwind",)))
    assert f3["phaze_threat"] == 1.0 and f3["phaze_threat_believed"] == 1.0
    assert f3["TA_hidden_phazer"] is None                                     # revealed: not a hidden-belief row


def test_entry_chip_respects_flying_and_levitate():
    me = _mon("a", types=["water"], stats=ST, active=True)
    foe = _mon("b", types=["water"], stats=ST)
    d = _board(me, foe, our_spikes=3)
    assert _facts(d)["entry_chip_OB"] == pytest.approx(1 / 4)                 # Blissey, grounded
    d["our_slots"] = [me["species"], "skarmory", "blissey"]
    assert _facts(d)["entry_chip_OB"] == 0.0                                  # Skarmory flies


# ------------------------------------------------------------------------------------------------- report
def test_welch_and_classify():
    g = R.welch([0.5, 0.52, 0.48, 0.5], [0.3, 0.31, 0.29, 0.3])
    assert g["diff"] == pytest.approx(-0.2, abs=1e-6) and g["ci"][1] < 0
    row = {"kind": "c", "species_lookup": 0.9, "arms": {"legacy": {"final": {"mean": 0.3}, "final_minus_input": {"mean": -0.2, "ci": [-0.3, -0.1]},
                               "random": {"mean": 0.25}, "selectivity": {"mean": 0.05},
                               "input": {"mean": 0.5}, "L1": {"mean": 0.4}}},
           "gap": {"diff": -0.1, "ci": [-0.15, -0.05]}}
    tier, flags = R.classify(row, "legacy", "static")
    assert tier == "poor"                                 # final 0.3 < 0.5, and static significantly worse
    assert any("LOSES" in f for f in flags) and any("random" in f for f in flags)
    assert any("static WORSE" in f for f in flags) and any("selective" in f for f in flags)
    good = copy.deepcopy(row)
    good["arms"]["legacy"].update(final={"mean": 0.9}, input={"mean": 0.95}, random={"mean": 0.85})
    good["gap"] = {"diff": 0.0, "ci": [-0.01, 0.01]}
    t2, f2 = R.classify(good, "legacy", "static")
    assert t2 == "easy" and any("random" in f for f in f2)      # a random-net match is a NOTE, never the tier
    good["gap"] = {"diff": -0.1, "ci": [-0.15, -0.05]}
    assert R.classify(good, "legacy", "static")[0] == "poor"   # an arm significantly worse is a hunt candidate


def test_a_near_constant_feature_cannot_blow_up_a_held_out_fold():
    """A column that is (nearly) constant on the training folds but not on a held-out one is zeroed, never scaled by
    its tiny sd (a random network's critic pool has such columns; scaled, they gave R² of −300)."""
    rng = np.random.default_rng(5)
    n = 2000
    battles = [f"b{i // 20}" for i in range(n)]
    folds = P.battle_folds(battles)
    sig = rng.normal(size=n)
    X = np.stack([sig + 0.1 * rng.normal(size=n), rng.normal(size=n)], 1)
    bad = np.where(folds == 0, 50.0, 3e-5 * rng.normal(size=n))      # constant-ish except in fold 0
    X = np.concatenate([X, bad[:, None]], 1).astype(np.float32)
    pred = P.ridge_oof(X, sig[:, None], folds)
    assert P.r2(sig, pred[:, 0]) > 0.9


# ------------------------------------------------------------------------------------------- the round replica
@pytest.mark.parametrize("kind", ["BiasedEncoderLayer", "IdentityInitRound"])
def test_the_depth_replica_is_the_round_and_its_variants_mean_what_they_say(kind):
    """`pin_worker._layer_forward` must BE each trunk round (post-LN and the pre-LN identity-init round that
    `--trunk-layers 3/4` append) before any variant is read; `lens` is the identity, and for a pre-LN round zeroing
    both residual updates is the identity too (a post-LN round keeps its two LayerNorms)."""
    import torch

    from agents.model.team_transformer import BiasedEncoderLayer
    from agents.model.trunk_depth import IdentityInitRound
    from main.probe_battery import pin_worker as W

    g = torch.Generator().manual_seed(3)
    L = BiasedEncoderLayer() if kind == "BiasedEncoderLayer" else IdentityInitRound()
    with torch.no_grad():
        for p in L.parameters():                     # an identity-init round's zero matrices would hide its branch
            p.copy_(0.2 * torch.randn(p.shape, generator=g))
    L.eval()
    x = torch.randn(2, 9, 128, generator=g)
    bias = torch.zeros(2, int(L.n_heads), 9, 9)
    with torch.no_grad():
        ref = L(x, bias)
        assert torch.allclose(W._layer_forward(L, x, bias, mode="full"), ref, atol=1e-5)
        assert torch.equal(W._layer_forward(L, x, bias, mode="lens"), x)
        zu = W._layer_forward(L, x, bias, mode="zero_update")
        if kind == "IdentityInitRound":
            assert torch.equal(zu, x)
        else:
            assert torch.allclose(zu, L.norm2(L.norm1(x)), atol=1e-6)
        off = W._layer_forward(L, x, bias, mode="full", head_mask=torch.tensor([1.0, 0.0, 1.0, 1.0]))
        assert not torch.allclose(off, ref, atol=1e-4)            # a head ablation moves the output
