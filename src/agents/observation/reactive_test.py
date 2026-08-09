"""Unit tests for `reactive.py` — post-`gen3_entity_rehome_v1` this module is the lean
BOARD block (5 raw scalars + the request-order active-move ID block). The Hidden-Power /
ability-prior expected-effectiveness helpers and the 288-dim matchup matrices are DELETED —
that physics lives GPU-side in the DamageOperator / D-V edge families, with their own gates
(`damage_op_probe_fuzz_test.py` is the authoritative physics oracle).

What remains to pin here is the `_request_slot_moves` resolver (action-order alignment).
"""
# The per-move obs features (base power vec[0:4], type multiplier vec[4:8], the move-effect block)
# must be filled in REQUEST-slot order — slot i ↔ action logit 6+i, disabled moves KEPT — so they
# align with the action mask / mapper (which index legal.move_slots). `_request_slot_moves` is the
# resolver that guarantees it; the bridge `move_alignment_fuzz_test.py` validates the full encoded
# vector against the live protocol (incl. real Choice-locks / Disable). These pin the helper
# deterministically — the exact shift the old `enumerate(battle.available_moves)` introduced.
# ---------------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402
from .reactive import _request_slot_moves  # noqa: E402


def _lm(move_id):
    """A minimal LegalMove-like request slot (only .id is read by the resolver)."""
    return SimpleNamespace(id=move_id)


def test_request_slots_keep_disabled_in_request_order():
    """A disabled NON-LAST move slot must stay at its request index — not get the next
    available move shifted into it (the available_moves bug). Slot 'b' is disabled, so
    poke-env's available_moves drops it; the resolver must still place b at index 1."""
    moves = {mid: SimpleNamespace(id=mid, base_power=bp)
             for mid, bp in [("a", 40), ("b", 80), ("c", 120), ("d", 60)]}
    active = SimpleNamespace(moves=moves)
    # available_moves is the SHIFTED list (b dropped) — proves we do NOT use it when legal is present.
    battle = SimpleNamespace(active_pokemon=active,
                             available_moves=[moves["a"], moves["c"], moves["d"]])
    legal = SimpleNamespace(move_slots=[_lm("a"), _lm("b"), _lm("c"), _lm("d")])

    resolved = _request_slot_moves(battle, legal)
    assert [m.id for m in resolved] == ["a", "b", "c", "d"]          # request order, disabled kept
    assert [m.base_power for m in resolved] == [40, 80, 120, 60]      # b's 80 stays at index 1


def test_request_slots_resolve_typed_hidden_power():
    """The request id is the bare 'hiddenpower'; the typed Move (hiddenpowerfire, carrying the
    real effectiveness) lives on the moveset under its typed key — resolve to it, like poke-env."""
    typed_hp = SimpleNamespace(id="hiddenpowerfire", base_power=70)
    moves = {"earthquake": SimpleNamespace(id="earthquake", base_power=100),
             "hiddenpowerfire": typed_hp}
    active = SimpleNamespace(moves=moves)
    battle = SimpleNamespace(active_pokemon=active, available_moves=[])
    legal = SimpleNamespace(move_slots=[_lm("earthquake"), _lm("hiddenpower")])

    resolved = _request_slot_moves(battle, legal)
    assert resolved[0].id == "earthquake"
    assert resolved[1] is typed_hp                                   # bare hiddenpower → typed Move


def test_request_slots_unresolvable_slot_is_none():
    """A request id absent from the moveset (and not Hidden Power) resolves to None — the encoder
    then leaves that slot at its neutral default rather than mis-attributing another move."""
    moves = {"earthquake": SimpleNamespace(id="earthquake", base_power=100)}
    active = SimpleNamespace(moves=moves)
    battle = SimpleNamespace(active_pokemon=active, available_moves=[])
    legal = SimpleNamespace(move_slots=[_lm("earthquake"), _lm("metronome")])

    resolved = _request_slot_moves(battle, legal)
    assert resolved[0].id == "earthquake"
    assert resolved[1] is None


def test_request_slots_fallback_to_available_moves_without_legal():
    """legal is None (unit-test / plain-Battle path) → fall back to available_moves order,
    byte-identical to the legacy behaviour (the trainee obs path always threads a real legal)."""
    am = [SimpleNamespace(id="a"), SimpleNamespace(id="b")]
    battle = SimpleNamespace(active_pokemon=SimpleNamespace(moves={}), available_moves=am)

    assert _request_slot_moves(battle, legal=None) == am
