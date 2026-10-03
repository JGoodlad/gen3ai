"""The trainee's spaces without an env, and the DELETED Python env core (deletion pass U2 / U3).

1. ``trainee_spaces`` builds ``(observation_space, action_space)`` from the args alone — the production
   argv's 23 keys, every label gate opening exactly its keys.
2. **The Python env core is GONE** (deletion pass U3, 2026-10-02): ``Gen3Env`` / ``wrappers`` /
   ``env_factory`` / ``async_vec_env`` / ``bridge_session`` do not exist, and no module in the tree
   imports one. (Until U3 this file held the runtime proof that a production launch never LOADED them
   and the static list of every remaining importer with its deletion unit; the list is empty now.)
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import io

import pytest

from utils.paths import src_path

#: The Python env core — what deletion pass R1 deleted. A file by one of these names coming back, or
#: any module importing one, fails here.
PY_CORE = ("agents.training.gen3_env", "agents.training.wrappers", "agents.training.async_vec_env",
           "main.train.env_factory", "utils.bridge.bridge_session")

def _resolved(*argv: str):
    from main.train.config import resolve_config
    from main.train.parser import build_parser

    p = build_parser()
    a = p.parse_args(["--steps", "1", *argv])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        resolve_config(a, p)
    return a


#: The production argv's observation keys (``spaces.Dict`` orders them itself, by key).
PRODUCTION_KEYS = [
    "observation", "action_mask", "belief_species", "belief_moves", "known_moves",
    "opp_action_kind", "opp_action_num", "opp_switch_slot", "opp_switch_species", "opp_class",
    "belief_spread", "belief_spread_mask", "belief_nature", "belief_nature_mask", "belief_ev",
    "belief_ev_mask", "hp_type_label", "hp_type_mask", "item_label", "item_mask", "win_target",
    "win_mask", "win_margin",
]


def test_the_production_argv_declares_its_keys_in_order():
    from agents.training.trainee_spaces import N_ACTIONS, trainee_spaces

    obs, act = trainee_spaces(_resolved("--arch", "production"))
    assert list(obs.spaces) == sorted(PRODUCTION_KEYS)
    assert act.n == N_ACTIONS == 11
    assert obs["observation"].shape[0] > 0 and obs["action_mask"].shape == (11,)


def test_the_bare_argv_carries_only_the_win_prob_labels():
    from agents.training.trainee_spaces import trainee_spaces

    obs, _ = trainee_spaces(_resolved("--allow-nonproduction-arch", "--allow-nonproduction-recipe"))
    assert set(obs.spaces) >= {"observation", "action_mask", "win_target", "win_mask", "win_margin",
                               "opp_class"}


@pytest.mark.parametrize("gate,keys", [
    ("fork_pg_mask", {"fork_pg_m"}),
])
def test_each_gate_opens_exactly_its_keys(gate, keys):
    from agents.observation.schema import build_schema
    from agents.observation.state_encoder import get_observation_encoder, load_mappings
    from agents.training.trainee_spaces import LabelGates, trainee_observation_space

    layout = get_observation_encoder(load_mappings()).get_layout()
    vec = build_schema(layout).gym_space()
    off = LabelGates(*([False] * len(LabelGates._fields)))
    base = set(trainee_observation_space(layout, vec, off).spaces)
    on = trainee_observation_space(layout, vec, off._replace(**{gate: True}))
    assert set(on.spaces) - base == keys


def test_the_derived_gates():
    from agents.training.trainee_spaces import label_gates

    g = label_gates(move_belief_mode="both")
    assert g.belief_labels and g.known_moves            # the move belief needs the belief labels
    assert not label_gates(move_belief_mode="unrevealed").known_moves


def test_the_python_env_core_modules_do_not_exist():
    for m in PY_CORE:
        assert importlib.util.find_spec(m) is None, f"{m} is back — the Python env core was deleted (U3)"


def _importers() -> dict:
    root = src_path()
    found: dict = {}
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if rel.startswith(("poke_env/", "rust_sim/", "rust_env/")) or "/target/" in rel \
                or rel == "agents/training/trainee_spaces_test.py":
            continue
        for n in ast.walk(ast.parse(p.read_text())):
            names = []
            if isinstance(n, ast.ImportFrom) and n.module:
                names = [n.module] + [f"{n.module}.{a.name}" for a in n.names]
            elif isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            hit = set(PY_CORE) & set(names)
            if hit:
                found.setdefault(rel, set()).update(hit)
    return found


def test_no_module_imports_the_deleted_python_env_core():
    found = _importers()
    assert not found, (f"module(s) import the deleted Python env core: {sorted(found)} — the Rust env core "
                       "is the only core (deletion pass U3); read spaces / class tables / the eval team "
                       "table from agents.training.trainee_spaces, opponent_classes, eval_teams")


def test_the_model_packages_class_table_agrees_with_the_declared_one():
    from agents.model import opp_intent
    from agents.training import opponent_classes as oc

    assert opp_intent.OPP_CLASS_NAMES == oc.OPP_CLASS_NAMES
    assert opp_intent.OPP_CLASS_BOT == oc.OPP_CLASS_BOT
    assert len(oc.OPP_CLASS_NAMES) == oc.N_OPP_CLASSES
