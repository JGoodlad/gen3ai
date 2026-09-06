"""Unit tests for `belief_tables` — the module contract of the `gen3_belief_tables_split_v1` cut.

The per-table SEMANTICS are tested beside the head each prior feeds (`spread_belief_test.py`,
`hp_type_belief_test.py`, `item_belief_test.py`) — that is where a prior's meaning is checkable, and
splitting those out would scatter each head's specification. What lives HERE is what only the SPLIT
can break:

  1. the RE-EXPORT surface — `damage_tables` still resolves every moved name, to the SAME object;
  2. the one-way LAYERING — `belief_tables` imports nothing from `damage_tables`, which is the whole
     reason the two modules layer rather than cycle;
  3. the `state_dict` invariant — the relocated tables are `persistent=False` buffers, so they
     contribute ZERO `state_dict` keys and a relocation cannot move a key that does not exist;
  4. bit-for-bit identity between each head's REGISTERED buffer and a fresh call of the constructor,
     through BOTH import paths.

(The one-off before/after proof that the cut changed nothing — all 236 `state_dict` keys, all 236
tensors and all 80 buffers byte-identical on a seeded production build at the parent commit vs this
one — was `tmp/belief_tables_equiv_probe.py`. A permanent test pins the INVARIANT; a refactor's
equivalence is a one-time measurement.)
"""
import ast
import inspect

import pytest
import torch

from agents import gen3_data
from agents.model import belief_tables as bt
from agents.model import damage_tables as dt

_N_SPECIES = 600

# Every name the split moved. `damage_tables` must still resolve all of them (historical import
# paths live in `belief_heads`, `gen3_env`, the prober and four test modules).
_MOVED = (
    "SPREAD_STAT_COLS", "N_SPREAD_STATS", "_SPREAD_BASE_IDX",
    "N_NATURES", "_NATURE_PRIOR_FLOOR",
    "build_opp_spread_prior", "build_nature_mult", "build_species_nature_prior",
    "build_species_ev_prior", "build_species_base_stats", "invert_nature_evs",
    "build_hp_type_prior", "build_item_prior",
)


# ------------------------------------------------------------------- 1. the re-export surface
@pytest.mark.parametrize("name", _MOVED)
def test_damage_tables_still_re_exports_every_moved_name(name):
    """`from agents.model.damage_tables import <name>` must keep working, and resolve to the SAME
    object — an accidental second definition would give two tables that silently disagree."""
    assert hasattr(bt, name), f"{name} is missing from belief_tables"
    assert hasattr(dt, name), (
        f"damage_tables no longer re-exports {name} — every historical "
        f"`from agents.model.damage_tables import {name}` breaks")
    assert getattr(dt, name) is getattr(bt, name), f"{name} is a COPY, not a re-export"


def test_no_moved_name_is_still_defined_in_damage_tables():
    """The re-export must be an import, not a leftover definition. Two definitions of a
    data-derived table is the shape where a fix lands in one copy and the other keeps shipping."""
    tree = ast.parse(inspect.getsource(dt))
    defined = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(t.id for t in node.targets if isinstance(t, ast.Name))
    assert defined.isdisjoint(_MOVED), f"still DEFINED in damage_tables: {defined & set(_MOVED)}"


# ------------------------------------------------------------------- 2. the one-way layering
def test_belief_tables_never_imports_damage_tables():
    """The edge runs ONE way: `damage_tables` imports from here (its `build_damage_buffers`
    registers SPECIES_SPREAD_PRIOR and NATURE_MULT for the op), so an import back would be a
    cycle. Python would resolve it only for whichever module was imported first — importing
    `belief_tables` first would raise, and nothing in the normal import order exercises that.
    Checked at the AST so a deferred function-level import counts too."""
    tree = ast.parse(inspect.getsource(bt))
    offenders = [
        f"line {n.lineno}: {ast.unparse(n)}"
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        and "damage_tables" in (getattr(n, "module", None) or
                                " ".join(a.name for a in n.names))
    ]
    assert not offenders, (
        "belief_tables imports damage_tables — that closes a cycle:\n  " + "\n  ".join(offenders))


# ------------------------------------------------------------------- 3./4. the production build
@pytest.fixture(scope="module")
def production():
    """The literal production-config extractor, built the way `delivery_graph` builds it."""
    from agents.model.delivery_graph import build_extractor
    fe, _cfg, layout = build_extractor()
    return fe, layout


# The buffer each relocated constructor produces, and how to rebuild it from the layout.
_BUFFERS = {
    "spread_belief.spread_prior":     ("build_opp_spread_prior", ("max_species",)),
    "spread_belief.nature_logprior":  ("build_species_nature_prior", ("max_species",)),
    "spread_belief.ev_prior":         ("build_species_ev_prior", ("max_species",)),
    "spread_belief.nature_mult":      ("build_nature_mult", ()),
    "spread_belief.base_nonhp":       ("build_species_base_stats", ("max_species",)),
    "hp_type_belief_head.hp_prior":   ("build_hp_type_prior", ("max_species",)),
    "item_belief_head.item_prior":    ("build_item_prior", ("max_species", "max_items")),
}


def test_no_relocated_table_contributes_a_state_dict_key(production):
    """THE key-move guard, and it holds structurally rather than by comparison: every relocated
    table registers `persistent=False` (data-derived and recomputable, never a saved weight), so it
    is absent from `state_dict` by construction — a relocation cannot move a key that does not
    exist. The buffers must still be PRESENT on the module tree; both halves are asserted, because
    "absent from state_dict" is also what a table that failed to build looks like."""
    fe, _layout = production
    keys = set(fe.state_dict().keys())
    buffers = dict(fe.named_buffers())
    for name in _BUFFERS:
        assert name in buffers, f"{name} is not registered on the production extractor"
        assert name not in keys, (
            f"{name} appears in state_dict — it must be registered persistent=False")


def test_every_relocated_table_is_bit_for_bit_its_constructor(production):
    """Each head's REGISTERED buffer equals a fresh call of the constructor, byte for byte, through
    BOTH import paths (`belief_tables` and the `damage_tables` hub). This is what a re-export test
    alone cannot say: that the hub resolves to a function producing the same tensor the live model
    is actually holding."""
    fe, layout = production
    buffers = dict(fe.named_buffers())
    for name, (fn_name, arg_keys) in _BUFFERS.items():
        args = [int(layout[k]) for k in arg_keys]
        want = buffers[name]
        for module in (bt, dt):
            got = getattr(module, fn_name)(*args)
            assert got.shape == want.shape, f"{name} via {module.__name__}: shape drift"
            assert torch.equal(got, want), (
                f"{name} does not equal {module.__name__}.{fn_name}{tuple(args)} bit-for-bit")


# ------------------------------------------------------------------- the moved forme-safety row
# gen3_species_formes_v1: the num-indexed tables must stay BASE-FORM. Moved here verbatim with
# `build_species_base_stats` (its siblings covering `build_species_types` / `build_move_prior_logits`
# stay in `damage_tables_test.py`, where those builders still live).
#
# Every buffer here is `table[species.num] = …`, and an alternate FORME shares its base's
# national-dex num (Deoxys-Speed / Unown-B / Castform-Sunny all landed in the species data
# once the port needed to construct gen3 randbats teams). Iterating `species.raw()` would be
# last-write-wins, so a forme would silently redefine the base's stats at that num — a
# plausible-but-false number fed to the model, invisible to any shape check. Every builder
# therefore iterates `species.base_form_ids()`; this pins that.

def test_base_stats_table_holds_the_BASE_forme():
    base = bt.build_species_base_stats(_N_SPECIES)
    deoxys = gen3_data.species.get("deoxys")
    row = [float(deoxys.base_stats[s]) for s in bt.SPREAD_STAT_COLS]
    assert base[deoxys.num].tolist() == row            # NOT Deoxys-Speed's 95/90/95/90/180
    assert base[deoxys.num][bt.SPREAD_STAT_COLS.index("spe")].item() == 150.0
