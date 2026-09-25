"""The divergence taxonomy the readiness gate is read by: only core-vs-training-input keys are
CUTOVER; an unknown key is never waved through."""
from main.rust_core_cutover import verdict as VD


def test_every_core_vs_python_key_is_a_cutover_blocker():
    for s, k in (("E", "MOVE.value"), ("T", "window.rows"), ("O", "our_team moves+4"),
                 ("V", "[core][SIM-FACT] ours.boosts"), ("V", "[core] mask"),
                 ("V", "[ALIGN] reading decided on a `wait` request"), ("N", "reward"),
                 ("V", "REFUSED: x"), ("V", "something new")):
        assert VD.category(s, k) == VD.CUTOVER, (s, k)


def test_reading_and_view_road_keys_are_reported_not_blocking():
    assert VD.category("V", "[BOARD] ours.[V15] moves[pp]") == VD.READING
    assert VD.category("V", "[TRUTH] opp.item") == VD.READING
    assert VD.category("V", "[SIM-FACT] ours.moves") == VD.READING
    assert VD.category("V", "[PRESENTATION/V4-volatiles] opp.volatiles") == VD.VIEW_ROAD


def test_census_counts_battles_and_fields_per_key():
    recs = [{"label": "a", "classes": {"V": {"[SIM-FACT] ours.moves": 3}}},
            {"label": "b", "classes": {"V": {"[SIM-FACT] ours.moves": 2}, "O": {"x": 1}}}]
    c = VD.census(recs)
    assert c[VD.READING]["V [SIM-FACT] ours.moves"] == {"battles": 2, "fields": 5, "example": "a"}
    assert c[VD.CUTOVER]["O x"]["battles"] == 1


def test_a_slice_n_difference_after_a_phantom_step_is_named_not_merged():
    assert VD.category("N", "observation board +2 [after-phantom]") == VD.CUTOVER_F1
    assert VD.category("N", "observation board +2") == VD.CUTOVER
    assert VD.category("N", "observation opp_team recency+0 [after-phantom]") == VD.CUTOVER_F1
    # a field the phantom record cannot move is NOT explained by F1, phantom or not
    assert VD.category("N", "observation our_team moves+4 [after-phantom]") == VD.CUTOVER
    assert VD.category("N", "win_target [after-phantom]") == VD.CUTOVER
