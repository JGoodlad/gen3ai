"""Pure unit tests for the incoming-damage belief math (no battle, no torch)."""
from agents.enums import PokemonType as PT, Status
from agents.observation import incoming_damage as inc


def test_gen3_damage_max_known_calc():
    # core = (42*100*300//200)//50 + 2 = (6300)//50 + 2 = 128; ×(2.0 eff × 1.5 stab)=3 → 384.
    assert inc.gen3_damage_max(100, 300, 200, stab=True, type_eff=2.0) == 384
    # immunity / zero power → 0
    assert inc.gen3_damage_max(100, 300, 200, stab=True, type_eff=0.0) == 0
    assert inc.gen3_damage_max(0, 300, 200, stab=False, type_eff=1.0) == 0
    # screen halves; burn halves physical
    full = inc.gen3_damage_max(100, 300, 200, stab=False, type_eff=1.0)
    assert inc.gen3_damage_max(100, 300, 200, stab=False, type_eff=1.0, screen=True) == int(full * 0.5)
    assert inc.gen3_damage_max(100, 300, 200, stab=False, type_eff=1.0, burned=True) == int(full * 0.5)


def test_p_ko_roll_integration():
    assert inc.p_ko(400, 300) == 1.0          # low roll (340) already KOs
    assert inc.p_ko(200, 300) == 0.0          # high roll can't reach
    # 350 HP vs 400 max: rolls 88..100 reach it → 13/16
    assert abs(inc.p_ko(400, 350) - 13 / 16) < 1e-9
    assert inc.p_ko(400, 0) == 1.0            # already dead
    assert inc.p_ko(0, 100) == 0.0


def test_p_outspeed_over_distribution():
    dist = [(250, 0.6), (310, 0.4)]            # 60% slow set, 40% fast set
    assert abs(inc.p_outspeed(300, dist) - 0.6) < 1e-9
    # opp paralysis quarters their speed → we always outspeed
    assert inc.p_outspeed(300, dist, opp_para=True) == 1.0
    # our paralysis quarters ours → we never outspeed these
    assert inc.p_outspeed(300, dist, our_para=True) == 0.0
    # exact tie → ½ (gen3 coin flip)
    assert inc.p_outspeed(250, [(250, 1.0)]) == 0.5
    # unknown opponent speed → 0.5 (max uncertainty)
    assert inc.p_outspeed(300, []) == 0.5
    # +1 boost (×1.5) flips a losing tier
    assert inc.p_outspeed(220, [(300, 1.0)], our_boost=2) == 1.0  # 220×2=440 > 300


def test_percentile_and_mean():
    d = [(100, 0.5), (200, 0.5)]
    assert inc.percentile(d, 0.5) == 100
    assert inc.percentile(d, 0.9) == 200
    assert inc.weighted_mean(d) == 150.0
    assert inc.percentile([], 0.5) is None and inc.weighted_mean([]) is None


def _zapdos(hp_remaining=200):
    return inc.Defender(def_stat=200, spd_stat=240, hp_remaining=hp_remaining, hp_max=290, spe=300,
                        type1=PT.ELECTRIC, type2=PT.FLYING, ability=None, status=None)


def _mence_threat(**kw):
    base = dict(types=(PT.DRAGON, PT.FLYING), atk_tail=350.0, atk_mean=320.0,
                spa_tail=250.0, spa_mean=230.0, spe_dist=[(280, 1.0)],
                phys=[inc.Candidate(PT.ROCK, 75, 0.5)],          # Rock Slide, 50% prior
                spec=[inc.Candidate(PT.ICE, 95, 0.4)])           # Ice Beam, 40% prior
    base.update(kw)
    return inc.AttackerThreat(**base)


def test_compute_block_flags_rock_slide_on_flyer():
    blk = inc.compute_team_block([_zapdos()], _mence_threat(), n_slots=6)
    phys_exp, spec_exp, phys_pko, spec_pko, outspeed = blk[:inc.PER_MON]
    assert phys_pko > 0.0          # Rock Slide 2× on a chipped Flyer → real KO chance (×0.5 prior)
    assert phys_exp > 0.0
    assert outspeed == 1.0          # spe 300 > opp 280
    assert blk.shape[0] == 6 * inc.PER_MON + inc.RECOVERY
    # a FULL-HP Zapdos: Rock Slide is only a 2HKO → phys_pko should be 0 there (chip still real)
    full = inc.compute_team_block([_zapdos(hp_remaining=290)], _mence_threat(), n_slots=6)
    assert full[2] == 0.0 and full[0] > 0.0


def test_fixed_damage_respects_immunity():
    ghost = inc.Defender(def_stat=200, spd_stat=200, hp_remaining=200, hp_max=200, spe=200,
                         type1=PT.GHOST, type2=None, ability=None, status=None)
    thr = _mence_threat(phys=[inc.Candidate(PT.FIGHTING, 0, 1.0, fixed_dmg=100)], spec=[])
    blk = inc.compute_team_block([ghost], thr, n_slots=6)
    assert blk[2] == 0.0   # phys_pko: Seismic Toss (Fighting) is 0× on a Ghost → no threat
    # vs a non-Ghost, the same fixed 100 vs 200 HP is not a KO but is real chip
    norm = inc.Defender(def_stat=200, spd_stat=200, hp_remaining=200, hp_max=200, spe=200,
                        type1=PT.WATER, type2=None, ability=None, status=None)
    blk2 = inc.compute_team_block([norm], thr, n_slots=6)
    assert blk2[0] == 100 / 200    # phys_exp = 100/200 fixed fraction


def test_substitute_zeroes_ko():
    d = inc.Defender(def_stat=150, spd_stat=150, hp_remaining=250, hp_max=250, spe=100,
                     type1=PT.NORMAL, type2=None, ability=None, status=None, has_sub=True)
    blk = inc.compute_team_block([d], _mence_threat(), n_slots=6)
    assert blk[0] == 0.0 and blk[2] == 0.0   # behind a Sub → no KO/dmg this turn


def test_no_attacker_returns_zeros():
    blk = inc.compute_team_block([_zapdos()], None, n_slots=6)
    assert blk.shape[0] == 6 * inc.PER_MON + inc.RECOVERY and blk.sum() == 0.0


def test_recovery_scalars_tail():
    thr = _mence_threat(recovery_rate=0.35, cures_status=0.35, recovery_known=1.0)
    blk = inc.compute_team_block([_zapdos()], thr, n_slots=6)
    assert tuple(blk[-3:]) == (0.35, 0.35, 1.0)


def test_boost_mult_table():
    assert inc.boost_mult(0) == 1.0
    assert inc.boost_mult(1) == 1.5
    assert inc.boost_mult(2) == 2.0
    assert abs(inc.boost_mult(-1) - 2 / 3) < 1e-9
    assert inc.boost_mult(99) == inc.boost_mult(6)   # clamp


def _skarmory(hp_remaining=334):
    # 252 HP Skarmory: high Def, low SpD; Steel/Flying.
    return inc.Defender(def_stat=337, spd_stat=176, hp_remaining=hp_remaining, hp_max=334, spe=176,
                        type1=PT.STEEL, type2=PT.FLYING, ability=None, status=None)


def test_explosion_halves_defense():
    # Metagross Explosion (Normal, 250 BP) vs a wall: pricing it as an ordinary physical move
    # under-reads the KO; halving Def (the Gen-3 mechanic) lifts the belief. Same move/atk, the
    # only difference is the halves_defense flag.
    d = _skarmory()
    plain = inc._channel_threat([inc.Candidate(PT.NORMAL, 250, 1.0)], d, 405.0, 360.0,
                                a=_mence_threat(types=(PT.NORMAL,)), screen=False, is_phys=True)
    boom = inc._channel_threat([inc.Candidate(PT.NORMAL, 250, 1.0, halves_defense=True)], d,
                               405.0, 360.0, a=_mence_threat(types=(PT.NORMAL,)),
                               screen=False, is_phys=True)
    assert boom[0] >= plain[0]      # P(KO) no lower with the Def-halve …
    assert boom[1] > plain[1]       # … and strictly more expected chip


def test_sandstorm_boosts_rock_spd():
    # Special hit vs a Rock-type under Sandstorm reads LESS damage than in clear weather (the
    # ×1.5 SpD). Tyranitar (Rock/Dark) on the special channel.
    ttar = inc.Defender(def_stat=256, spd_stat=236, hp_remaining=341, hp_max=341, spe=236,
                        type1=PT.ROCK, type2=PT.DARK, ability=None, status=None)
    surf = [inc.Candidate(PT.WATER, 95, 1.0)]
    clear = inc._channel_threat(surf, ttar, 350.0, 320.0,
                                a=_mence_threat(types=(PT.WATER,), weather=None),
                                screen=False, is_phys=False)
    sand = inc._channel_threat(surf, ttar, 350.0, 320.0,
                               a=_mence_threat(types=(PT.WATER,), weather="Sandstorm"),
                               screen=False, is_phys=False)
    assert sand[1] < clear[1]       # sand-boosted SpD → less expected damage
    # the SpD boost is special-only: a physical hit ignores it
    phys = [inc.Candidate(PT.GROUND, 100, 1.0)]
    pclear = inc._channel_threat(phys, ttar, 350.0, 320.0,
                                 a=_mence_threat(types=(PT.GROUND,), weather=None),
                                 screen=False, is_phys=True)
    psand = inc._channel_threat(phys, ttar, 350.0, 320.0,
                                a=_mence_threat(types=(PT.GROUND,), weather="Sandstorm"),
                                screen=False, is_phys=True)
    assert psand[1] == pclear[1]


def test_paralysis_status_enum_folds_into_outspeed():
    # Defender.status is the raw Status enum; PAR quarters our Speed → we drop the speed tie/lead.
    fast = inc.Defender(def_stat=200, spd_stat=200, hp_remaining=200, hp_max=200, spe=300,
                        type1=PT.NORMAL, type2=None, ability=None, status=None)
    para = inc.Defender(def_stat=200, spd_stat=200, hp_remaining=200, hp_max=200, spe=300,
                        type1=PT.NORMAL, type2=None, ability=None, status=Status.PAR)
    thr = _mence_threat(spe_dist=[(280, 1.0)])
    healthy = inc.compute_team_block([fast], thr, n_slots=6)
    slowed = inc.compute_team_block([para], thr, n_slots=6)
    assert healthy[4] == 1.0        # 300 > 280 → always first
    assert slowed[4] == 0.0         # 300×0.25 = 75 < 280 → never first
