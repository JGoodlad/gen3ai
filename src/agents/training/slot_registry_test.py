import pytest
from agents.training.slot_registry import SlotRegistry


def test_assigns_slots_in_order():
    reg = SlotRegistry()
    assert reg.assign("tyranitar") == 0
    assert reg.assign("skarmory") == 1
    assert reg.assign("swampert") == 2


def test_stable_on_repeat():
    reg = SlotRegistry()
    reg.assign("tyranitar")
    reg.assign("skarmory")
    assert reg.assign("tyranitar") == 0
    assert reg.assign("skarmory") == 1


def test_get_returns_none_for_unknown():
    reg = SlotRegistry()
    assert reg.get("tyranitar") is None
    reg.assign("tyranitar")
    assert reg.get("tyranitar") == 0


def test_snapshot_is_copy():
    reg = SlotRegistry()
    reg.assign("tyranitar")
    reg.assign("skarmory")
    snap = reg.snapshot()
    assert snap == {"tyranitar": 0, "skarmory": 1}
    snap["tyranitar"] = 99
    assert reg.assign("tyranitar") == 0


def test_overflow_raises():
    reg = SlotRegistry()
    for i in range(6):
        reg.assign(f"mon{i}")
    with pytest.raises(RuntimeError, match="overflow"):
        reg.assign("mon6")


def test_reset_clears_all():
    reg = SlotRegistry()
    reg.assign("tyranitar")
    reg.assign("skarmory")
    reg.reset()
    assert reg.get("tyranitar") is None
    assert reg.assign("tyranitar") == 0


def test_reorder_yields_different_slots():
    reg1 = SlotRegistry()
    reg1.assign("skarmory")
    reg1.assign("tyranitar")

    reg2 = SlotRegistry()
    reg2.assign("tyranitar")
    reg2.assign("skarmory")

    assert reg1.get("skarmory") == 0
    assert reg2.get("skarmory") == 1
