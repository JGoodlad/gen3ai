"""Unit tests for `belief_tables` + `dex_ids` — the module contract of the two split rounds
(`gen3_belief_tables_split_v1` then `gen3_dex_ids_split_v1`, both 2026-09-06).

The per-table SEMANTICS are tested beside the head each prior feeds (`spread_belief_test.py`,
`hp_type_belief_test.py`, `item_belief_test.py`, `move_prior_fusion_test.py`,
`species_prior_fusion_test.py`, `damage_tables_test.py`) — that is where a prior's meaning is
checkable, and splitting those out would scatter each head's specification. What lives HERE is what
only the SPLIT can break:

  1. the RE-EXPORT surface — `damage_tables` still resolves every moved name, to the SAME object;
  2. the one-way LAYERING — `damage_tables` → `belief_tables` → `dex_ids`, and never back, which is
     the whole reason the three modules layer rather than cycle;
  3. the `state_dict` invariant — the relocated tables are `persistent=False` buffers, so they
     contribute ZERO `state_dict` keys and a relocation cannot move a key that does not exist;
  4. bit-for-bit identity between each head's REGISTERED buffer and a fresh call of the constructor,
     through BOTH import paths.

(The one-off before/after proof that each cut changed nothing — all 236 `state_dict` entries and all
80 buffers byte-identical on a seeded production build at the parent commit vs the cut, and every
moved definition executable-AST identical — is
`designs/research_state/measurements/dex_ids_split_2026-09-06/equivalence_probe.py`, re-runnable
against any `--baseline`. A permanent test pins the INVARIANT; a refactor's equivalence is a
one-time measurement.)
"""
import ast
import inspect
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from agents import gen3_data
from agents.model import belief_tables as bt
from agents.model import damage_tables as dt
from agents.model import dex_ids as dx

_N_SPECIES = 600

# THE LAYERING, declared once. The scans below read this tuple rather than a hand-written test per
# edge, so a fourth module means adding an entry — not remembering to write another scan.
#
#     damage_tables → belief_tables → dex_ids,   and   damage_tables → dex_ids
#
# Strictly descending: a module may import only from those BELOW it in this tuple.
_LAYERS = (dt, bt, dx)

# Every name round 1 moved into `belief_tables`, and round 2 into `dex_ids`, keyed by its new OWNER.
# `damage_tables` must still resolve all of them — the historical import paths live in
# `belief_heads`, `t0_species`, `extractor_build`, `snapshot`, `gen3_env`, `main.train.config`,
# `flag_registry`, the prober and nine test modules.
_MOVED_TO = {
    bt: (
        "SPREAD_STAT_COLS", "N_SPREAD_STATS", "_SPREAD_BASE_IDX",
        "N_NATURES", "_NATURE_PRIOR_FLOOR",
        "build_opp_spread_prior", "build_nature_mult", "build_species_nature_prior",
        "build_species_ev_prior", "build_species_base_stats", "invert_nature_evs",
        "build_hp_type_prior", "build_item_prior",
        # gen3_dex_ids_split_v1 — the MOVE prior and the team-composition SPECIES prior, with the
        # floor constants that define what "illegal" vs "legal but unobserved" means.
        "_PRIOR_FLOOR", "_ILLEGAL_PROB", "_MIN_PRIOR_FLOOR",
        "sanitize_historical_move_floor", "build_move_prior_logits",
        "_SPECIES_PRIOR_FLOOR", "_SPECIES_CLAUSE_PROB", "SPECIES_CLAUSE_LOGIT",
        "_COOCCUR_LIFT_CLAMP", "build_species_cooccur_prior",
    ),
    dx: (
        # gen3_dex_ids_split_v1 — the dex-IDENTITY facts BOTH the physics and the beliefs key on.
        "HIDDEN_POWER_NUM", "_belief_num", "_hp_typed_nums",
        "_USAGE_PRIOR_FLOOR", "build_species_usage_prior",
    ),
}
_MOVED = tuple(name for names in _MOVED_TO.values() for name in names)


# ------------------------------------------------------------------- 1. the re-export surface
@pytest.mark.parametrize("name", _MOVED)
def test_damage_tables_still_re_exports_every_moved_name(name):
    """`from agents.model.damage_tables import <name>` must keep working, and resolve to the SAME
    object as the module that now OWNS it — an accidental second definition would give two tables
    that silently disagree."""
    owner = next(m for m, names in _MOVED_TO.items() if name in names)
    assert hasattr(owner, name), f"{name} is missing from {owner.__name__}"
    assert hasattr(dt, name), (
        f"damage_tables no longer re-exports {name} — every historical "
        f"`from agents.model.damage_tables import {name}` breaks")
    assert getattr(dt, name) is getattr(owner, name), f"{name} is a COPY, not a re-export"


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


def test_no_name_is_defined_in_two_of_the_three_layers():
    """The generalization of the test above, across the whole layering: exactly one owner per name.
    A constant re-declared in a lower layer rather than imported from it is the same "a fix lands in
    one copy" hazard — and re-declaring a floor is precisely the shortcut a later split round is
    tempted by, since it looks like it removes a dependency."""
    owners: dict = {}
    for module in _LAYERS:
        tree = ast.parse(inspect.getsource(module))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            else:
                names = []
            for n in names:
                owners.setdefault(n, []).append(module.__name__)
    dupes = {n: where for n, where in owners.items() if len(where) > 1}
    assert not dupes, f"defined in more than one layer: {dupes}"


# ------------------------------------------------------------------- 2. the one-way layering
@pytest.mark.parametrize("rank", range(1, len(_LAYERS)))
def test_the_import_edge_only_ever_points_DOWN_the_layering(rank):
    """`damage_tables` → `belief_tables` → `dex_ids`, and never back.

    The higher layers are real CONSUMERS — `build_damage_buffers` registers SPECIES_SPREAD_PRIOR,
    NATURE_MULT and SPECIES_USAGE_PRIOR for the op, and `build_move_prior_logits` calls
    `_belief_num` / `_hp_typed_nums` — so an import back would close a cycle. Python resolves a
    cycle only for whichever module was imported FIRST: importing `dex_ids` (or `belief_tables`)
    first would raise, and nothing in the normal import order exercises that, which is exactly why
    this is checked structurally instead of by importing. Checked at the AST so a deferred
    function-level import counts too."""
    module = _LAYERS[rank]
    forbidden = {m.__name__.rsplit(".", 1)[-1] for m in _LAYERS[:rank]}
    offenders = []
    for n in ast.walk(ast.parse(inspect.getsource(module))):
        if not isinstance(n, (ast.Import, ast.ImportFrom)):
            continue
        parts = set((getattr(n, "module", None) or "").split("."))
        parts.update(p for a in n.names for p in a.name.split("."))
        if forbidden & parts:
            offenders.append(f"line {n.lineno}: {ast.unparse(n)}")
    assert not offenders, (
        f"{module.__name__} imports UP the layering ({', '.join(sorted(forbidden))}) — that closes "
        "a cycle:\n  " + "\n  ".join(offenders))


def test_a_cold_import_of_the_lowest_layer_leaves_the_ones_above_unloaded():
    """The layering EXECUTED rather than read off the AST — a fresh interpreter that imports only
    `dex_ids` must end with neither `belief_tables` nor `damage_tables` in `sys.modules`. The AST
    scan above is the precise guard (it also sees a deferred import this could miss); this one
    would still fail if the edge ever arrived through import machinery an AST cannot read."""
    src_root = Path(dx.__file__).resolve().parents[2]      # …/src, the import root of THIS tree
    probe = ("import sys, agents.model.dex_ids; "
             "print([m for m in ('agents.model.belief_tables', 'agents.model.damage_tables') "
             "if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(src_root),
                              "HOME": str(Path.home()), "OMP_NUM_THREADS": "1"})
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "[]", (
        f"importing dex_ids dragged in {out.stdout.strip()} — the edge is not one-way")


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
    # gen3_dex_ids_split_v1 — the op's own consumer of a relocated builder, and the two priors the
    # second round moved. `move_belief.move_prior_logits` is the one whose constructor takes the
    # run's `--move-candidate-floor`, so it is rebuilt from the live module's own default.
    "damage_op.SPECIES_USAGE_PRIOR":  ("build_species_usage_prior", ("max_species",)),
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
    BOTH import paths (the owning module and the `damage_tables` hub). This is what a re-export test
    alone cannot say: that the hub resolves to a function producing the same tensor the live model
    is actually holding."""
    fe, layout = production
    buffers = dict(fe.named_buffers())
    for name, (fn_name, arg_keys) in _BUFFERS.items():
        args = [int(layout[k]) for k in arg_keys]
        want = buffers[name]
        owner = next(m for m, names in _MOVED_TO.items() if fn_name in names)
        for module in (owner, dt):
            got = getattr(module, fn_name)(*args)
            assert got.shape == want.shape, f"{name} via {module.__name__}: shape drift"
            assert torch.equal(got, want), (
                f"{name} does not equal {module.__name__}.{fn_name}{tuple(args)} bit-for-bit")


def test_the_move_prior_the_model_holds_is_the_relocated_builders_output(production):
    """`move_belief.move_prior_logits` gets its own row: unlike every buffer above, its constructor
    takes the run's `--move-candidate-floor`, so rebuilding it means reading the floor the built
    extractor recorded rather than passing a default. It is also the one relocated table whose
    builder reaches DOWN a layer (`_belief_num` / `_hp_typed_nums` in `dex_ids`), which makes it the
    table that would break first if the second split round had copied those instead of importing
    them."""
    fe, layout = production
    buffers = dict(fe.named_buffers())
    name = "move_belief.move_prior_logits"
    assert name in buffers, "the production config does not build a move belief"
    assert name not in set(fe.state_dict().keys()), f"{name} must be persistent=False"
    floor = float(fe.move_candidate_floor)
    for module in (bt, dt):
        got = module.build_move_prior_logits(int(layout["max_species"]), int(layout["max_moves"]),
                                             floor=floor)
        assert torch.equal(got, buffers[name]), (
            f"{name} does not equal {module.__name__}.build_move_prior_logits bit-for-bit "
            f"at floor={floor}")


# ------------------------------------------------------------------- the moved forme-safety row
# gen3_species_formes_v1: the num-indexed tables must stay BASE-FORM. Moved here verbatim with
# `build_species_base_stats` (its sibling covering `build_species_types` stays in
# `damage_tables_test.py`, where that builder still lives; `build_move_prior_logits`' own coverage
# is in `move_prior_fusion_test.py` and `damage_tables_test.py`, both of which reach it through the
# `damage_tables` hub and so are unaffected by the move).
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


def test_species_usage_prior_holds_the_BASE_forme():
    """The same forme rule on the table the SECOND round moved, and the one where a forme collision
    would be hardest to see: a usage share is a plausible number at any num, so a last-write-wins
    row reads as data rather than as a defect. Deoxys' three formes share num 386."""
    prior = dx.build_species_usage_prior(_N_SPECIES)
    deoxys = gen3_data.species.get("deoxys")
    speed = gen3_data.species.get("deoxysspeed")
    assert speed is not None and speed.num == deoxys.num       # the collision is real
    assert float(prior[deoxys.num]) > 0.0
    assert abs(float(prior.sum()) - 1.0) < 1e-5                # normalized over base forms only
