"""The MATERIAL MARGIN — the normalized material balance of a board, in [−1, 1].

It is the `win_margin` training-only obs key (`Gen3Env._merge_training_labels`), read by the win-prob
head's closeness-stratified metrics (`instrumented_ppo/value_terms.py`: `contested_frac`,
`brier_contested`, `skill_vs_material`). It is NOT a reward term.

It was born as a by-product of the material PBRS potential Φ_mat, and when the shaped reward path
was deleted (program_rust_core §4 M3 row, 2026-09-26) the potential went and the margin stayed —
moved here verbatim, with `mat_alive_weight` frozen at the value every run since the flag existed
trained with (1.25, production's), so the obs key is byte-identical (`reward_golden_test`'s
`win_margin` column is recorded across the deletion).

    margin = clamp(Φ, ±B) / B,   Φ = MAT_HP_WEIGHT·(Σ our_hp − Σ opp_hp)
                                    + MAT_ALIVE_WEIGHT·(n_alive_ours − n_alive_opp),
                                 B = (MAT_HP_WEIGHT + MAT_ALIVE_WEIGHT)·6

summed over the **declared team size** — unrevealed opponent mons count as full-HP-alive, so the
margin is ≈0 at turn 0 and has no opp-reveal discontinuities. A pure function of the board.
"""
from agents.observation.constants import TEAM_SIZE as _TEAM_SIZE

#: Weight of one full-HP mon's HP in the material balance.
MAT_HP_WEIGHT = 2.0
#: Weight of one mon's ALIVE bit. The value of the deleted `--mat-alive-weight` flag's default,
#: which every run trained with (production recorded 1.25).
MAT_ALIVE_WEIGHT = 1.25


def material_margin(live) -> float:
    """The normalized material margin of `live` (a `LiveView`), in [−1, 1]. 0.0 when our team is
    empty (mock / standalone paths; production always has all 6 from turn 1)."""
    our = live.ours
    opp = live.opp
    if not our.mons:
        return 0.0
    our_hp = sum(float(m.hp_fraction) for m in our.mons[:_TEAM_SIZE] if not m.fainted)
    our_alive = sum(1 for m in our.mons[:_TEAM_SIZE] if not m.fainted)
    # Opp: revealed mons carry their real HP/alive; unrevealed declared slots count full-HP-alive.
    opp_team_size = getattr(opp, "team_size", None) or _TEAM_SIZE
    opp_team_size = min(int(opp_team_size), _TEAM_SIZE)
    opp_revealed = list(opp.mons[:_TEAM_SIZE])
    opp_hp = sum(float(m.hp_fraction) for m in opp_revealed if not m.fainted)
    opp_alive = sum(1 for m in opp_revealed if not m.fainted)
    n_unrevealed = max(0, opp_team_size - len(opp_revealed))
    opp_hp += n_unrevealed * 1.0
    opp_alive += n_unrevealed
    phi = MAT_HP_WEIGHT * (our_hp - opp_hp) + MAT_ALIVE_WEIGHT * (our_alive - opp_alive)
    bound = MAT_HP_WEIGHT * _TEAM_SIZE + MAT_ALIVE_WEIGHT * _TEAM_SIZE
    clamped = max(-bound, min(phi, bound))
    return clamped / bound if bound > 0 else 0.0
