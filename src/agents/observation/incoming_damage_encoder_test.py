"""Unit tests for the incoming-damage BATTLE-EXTRACTION (incoming_damage_encoder).

The pure belief MATH is pinned in incoming_damage_test.py; this pins the candidate-building glue:
the FIX-2 widenings (Return/Frustration pricing, Hidden Power typed expansion, the broadened
floor/cap so super-effective coverage survives). Uses real gen3_data (species/moves/priors) with
tiny duck-typed opponents — no battle, no torch.
"""
from types import SimpleNamespace as NS

from agents.enums import PokemonType as PT
from agents.observation import incoming_damage as inc
from agents.observation import incoming_damage_encoder as enc
from agents.training.hidden_power_tracker import HiddenPowerTracker


def _channels(opp_moves, species, hp_tracker=None):
    return enc._candidates(NS(moves={m: object() for m in opp_moves}), species, hp_tracker)


def test_return_and_frustration_priced():
    # Return/Frustration read base_power 0 in the dex (power is computed live from happiness) — they
    # were silently dropped. Now priced at the competitive max (102 BP), Normal → physical channel.
    for mv in ("return", "frustration"):
        phys, _ = _channels([mv], "snorlax")
        priced = [c for c in phys if c.power == 102 and c.move_type == PT.NORMAL]
        assert priced, f"{mv} should yield a 102-BP Normal candidate, got {phys}"
        assert priced[0].fixed_dmg is None   # variable-power, NOT a fixed-damage move (full formula)


def test_revealed_hidden_power_expands_to_typed_coverage():
    # A revealed bare `hiddenpower` (dex BP 0 → previously a silent zero) expands into per-type
    # candidates priced from the typed dex variants (~70 BP), typed from the species HP prior.
    # Zapdos: HP Grass / HP Ice dominate → both Special-channel coverage must appear.
    _, spec = _channels(["hiddenpower"], "zapdos")
    hp = {c.move_type: c for c in spec if c.power == 70}
    assert PT.ICE in hp and PT.GRASS in hp, f"expected HP Ice + HP Grass, got {list(hp)}"
    assert all(0.0 < c.p_in_set <= 1.0 for c in hp.values())
    # HP Grass is the most likely Zapdos HP type → highest p_in_set among the expansion
    assert hp[PT.GRASS].p_in_set >= hp[PT.ICE].p_in_set


def test_revealed_hidden_power_no_dist_is_silent_but_safe():
    # A species with no HP prior and no tracker obs → no typed HP candidates (graceful, not a crash).
    cands = enc._hidden_power_candidates("missingno_nonexistent_species", None, 1.0)
    assert cands == []


def test_revealed_hp_tracker_narrows_type():
    # Once the HP tracker has observed an effectiveness tier, the expansion follows the NARROWED
    # distribution, not the broad prior. Observing HP at 2× on a Dragon/Flying mon rules out every
    # type but Ice (the 4× coverage), so the expansion collapses to a single HP Ice candidate at P=1.
    tr = HiddenPowerTracker(_priors={"zapdos": {"ice": 0.5, "grass": 0.5}})
    dragon_flyer = NS(type_1=PT.DRAGON, type_2=PT.FLYING, ability=None, status=None, species="x")
    assert tr.is_feasible(2.0, dragon_flyer)
    tr.observe("zapdos", 2.0, dragon_flyer)
    cands = enc._hidden_power_candidates("zapdos", tr, 1.0)
    assert len(cands) == 1 and cands[0].move_type == PT.ICE
    assert abs(cands[0].p_in_set - 1.0) < 1e-6   # normalised: HP is in the set, type pinned to Ice


def test_widened_floor_and_cap_admit_more_se_coverage():
    # FIX-2c: the broadened floor (0.12→0.05) + cap (4→6) let low-usage coverage survive into the
    # per-species pool, so the per-defender max can surface a super-effective option. Tyranitar runs
    # many coverage moves (Rock/Ground/Fire/Ice/Dark/HP) → strictly more candidates than the old
    # gate. Monotone in both knobs; restore + clear the per-species cache so other tests are clean.
    enc._prior_candidates.cache_clear()
    new_phys, new_spec = enc._prior_candidates("tyranitar")
    n_new = len(new_phys) + len(new_spec)
    saved_floor, saved_cap = enc._PRIOR_MOVE_MIN_P, enc._MAX_CANDIDATES_PER_CHANNEL
    try:
        enc._PRIOR_MOVE_MIN_P, enc._MAX_CANDIDATES_PER_CHANNEL = 0.12, 4
        enc._prior_candidates.cache_clear()
        old_phys, old_spec = enc._prior_candidates("tyranitar")
        n_old = len(old_phys) + len(old_spec)
    finally:
        enc._PRIOR_MOVE_MIN_P, enc._MAX_CANDIDATES_PER_CHANNEL = saved_floor, saved_cap
        enc._prior_candidates.cache_clear()
    assert n_new > n_old, f"widened gate should admit more candidates: new={n_new} old={n_old}"
    assert all(len(ch) <= enc._MAX_CANDIDATES_PER_CHANNEL for ch in (new_phys, new_spec))


def test_offensive_tail_quantile_is_raised():
    # FIX-1: the offensive-stat tail used for the KO magnitude is the high (max-EV+) percentile, so a
    # genuine OHKO from a strong but <15%-usage set isn't averaged out. Guard the constant + that the
    # 0.95 tail is >= the old 0.85 tail for a species with a spread distribution.
    assert enc._OFFENSIVE_TAIL_Q >= 0.95
    dist = inc.percentile  # alias for clarity
    from agents import gen3_data
    spe = gen3_data.priors.stat_distribution("tyranitar", "atk")
    assert dist(spe, 0.95) >= dist(spe, 0.85)
