"""The LABEL columns are the inventory's ``core`` rows, generated, never hand-copied (M5 Lane C).

``columns.py`` builds one column per ``core`` row of ``label_inventory.LABELS`` (named by its
``Gen3Env`` key, its dtype, ``(N, SIDES, *shape)``), and the family map the spec's ``labels``
declaration is checked against. The inventory is pinned against ``Gen3Env`` by
``agents/training/rust_env_label_inventory_test.py``; this pins the columns to the inventory, so the
chain Gen3Env → inventory → columns → Rust has no unpinned link.
"""
import json

from utils.rust_env import columns as C
from utils.rust_env import label_inventory as LI
from utils.rust_env import protocol as P


def test_one_column_per_core_label_with_its_dtype_and_shape():
    by_name = {c.name: c for c in C.COLUMNS}
    core = [r for r in LI.LABELS if r.rust == "core"]
    assert core, "no core labels — the gate would be vacuous"
    for r in core:
        c = by_name[r.key]
        assert (c.dtype, c.dims, c.direction, c.owner) == (r.dtype, ("N", "SIDES") + r.shape, "out", "C"), r.key
    for r in LI.LABELS:
        if r.rust != "core":
            assert r.key not in by_name, f"{r.key} is {r.rust}: the core has no column for it"
    assert C.shapes(3, 10)["belief_moves"] == (3, 2, 6, 4)
    assert C.numpy_dtype(by_name["belief_species"]) == "<i8"


def test_the_family_map_partitions_the_core_columns_and_is_rendered():
    keys = [k for ks in C.LABEL_FAMILIES.values() for k in ks]
    assert sorted(keys) == sorted(r.key for r in LI.LABELS if r.rust == "core")
    assert len(keys) == len(set(keys))
    assert set(C.LABEL_NOT_CORE) == {f for f, rows in LI.families().items() if rows[0].rust != "core"}
    text = C.render()
    for fam, ks in C.LABEL_FAMILIES.items():
        assert f'("{fam}", &[' + ", ".join(f"super::col::{k.upper()}" for k in ks) + "])" in text, fam
    for fam, why in C.LABEL_NOT_CORE.items():
        assert f'("{fam}", "{why}")' in text, fam


def test_the_spec_declares_label_families_and_the_schema_covers_them():
    s = json.loads(P.spec_json(n=1, threads=1, teams=["a"], names=("x", "y"), decision_tense=False,
                               switch_freeze=False, turn_limit=None, refusal_budget=0, bank_dir=None,
                               labels=("belief",)))
    assert s["labels"] == ["belief"] and tuple(s) == P.SPEC_KEYS
    assert "labelfamily belief belief_species belief_moves known_moves" in C.schema_text()
