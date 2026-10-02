"""The trainee's spaces without an env, and the Python env core's production dependents (deletion pass U2).

1. ``trainee_spaces`` builds ``(observation_space, action_space)`` from the args alone — the production
   argv's 23 keys, every label gate opening exactly its keys.
2. **The Python env core has NO production dependents.** A FRESH interpreter that resolves a production
   argv, builds its spaces, its opponent plan and its eval team table, and imports every module the Rust
   core's launch reads, never loads ``Gen3Env`` / ``wrappers`` / ``env_factory`` / ``async_vec_env`` /
   ``bridge_session`` (RUNTIME); and every non-test module that names one of them is DECLARED below with
   the deletion unit that removes it (STATIC) — a new production importer fails here.
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import subprocess
import sys

import pytest

from utils.paths import repo_root, src_path

#: The Python env core — what deletion pass R1 deletes.
PY_CORE = ("agents.training.gen3_env", "agents.training.wrappers", "agents.training.async_vec_env",
           "main.train.env_factory", "utils.bridge.bridge_session")

#: Every non-test module allowed to import the Python env core, and why. Each goes in the deletion
#: pass (designs/ops/deletion_pass_manifest.md) — none is on the Rust core's launch path.
ALLOWED_IMPORTERS = {
    # the Python core's OWN branches inside shared modules: lazy, reached only on --env-core python
    "main.train_rl_agent": "R1 — the python branch (lazy imports), U3",
    "agents.training.instrumented_ppo.rollout_probes": "R4 — the python branch (lazy import), U4",
    # Python-vs-Rust oracle harnesses (D3: retired without a banked run)
    "agents.training.rust_env_opponents_parity": "R10, U3",
    "agents.training.rust_rollout.parity": "R10, U3",
    "main.rust_core_cutover.driver": "R10, U3",
    "main.rust_core_cutover.envs": "R10, U3",
    "main.rust_core_m5.production": "R10, U3",
    "main.rust_core_m5.slice_n": "R10, U3",
    "main.rust_core_m5.throughput": "R10, U3",
    "utils.rust_env.bot_corpus": "R10 — the bot-corpus re-record, U3",
    # benchmarks of the Python bridge session itself
    "utils.bridge.bridge_heap_growth_benchmark": "R1 (bridge_session), U3",
    "utils.bridge.bridge_impl_throughput_benchmark": "R1 (bridge_session), U3",
    "utils.bridge.bridge_vs_websocket_latency_benchmark": "R1 (bridge_session), U3",
}


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


#: The production-path probe, run in a FRESH interpreter so nothing an earlier test imported counts.
_PROBE = r"""
import contextlib, io, json, sys
from main.train.config import resolve_config
from main.train.parser import build_parser
import main.train_rl_agent, main.train.rust_env_setup, main.checkargs
import main.train.parser.eval_subprocess, main.ops.value_sidecar_read
import agents.training.rust_rollout.build, agents.training.rust_rollout.collector
import agents.training.rust_env_opponents, agents.training.selfplay_callback
import agents.training.rust_eval.build, agents.training.instrumented_ppo.ppo
from agents.training.trainee_spaces import trainee_spaces
from agents.training.rust_eval.build import eval_builders
from agents.training.opponent_classes import OPP_CLASS_NAMES
p = build_parser()
a = p.parse_args(["--steps", "1", "--arch", "production"])
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    resolve_config(a, p)
assert a.env_core == "rust", a.env_core
trainee_spaces(a)
eval_builders(None, [])
print(json.dumps(sorted(m for m in sys.modules if m in %r)))
""" % (PY_CORE,)


def test_a_production_launch_path_never_loads_the_python_env_core():
    out = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=600,
                         cwd=str(repo_root()), env={**os.environ, "PYTHONPATH": str(src_path())})
    assert out.returncode == 0, out.stderr[-3000:]
    loaded = json.loads(out.stdout.strip().splitlines()[-1])
    assert loaded == [], f"the Rust core's launch path loaded the Python env core: {loaded}"


def _importers() -> dict:
    root = src_path()
    found: dict = {}
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if rel.startswith(("poke_env/", "rust_sim/", "rust_env/")) or rel.endswith("_test.py") \
                or "/target/" in rel:
            continue
        mod = rel[:-3].replace("/", ".")
        if mod in PY_CORE:
            continue
        for n in ast.walk(ast.parse(p.read_text())):
            names = []
            if isinstance(n, ast.ImportFrom) and n.module:
                names = [n.module] + [f"{n.module}.{a.name}" for a in n.names]
            elif isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            hit = set(PY_CORE) & set(names)
            if hit:
                found.setdefault(mod, set()).update(hit)
    return found


def test_every_python_env_core_importer_is_declared_with_its_deletion_unit():
    found = _importers()
    undeclared = sorted(set(found) - set(ALLOWED_IMPORTERS))
    assert not undeclared, (
        f"new importer(s) of the Python env core {undeclared}: the Rust core's launch must not depend "
        "on it (deletion pass U2) — read the constant / builder from its env-free home "
        "(agents.training.trainee_spaces, opponent_classes, eval_teams)")
    stale = sorted(set(ALLOWED_IMPORTERS) - set(found))
    assert not stale, f"declared importers that no longer import the Python env core: {stale} — drop them"


def test_the_model_packages_class_table_agrees_with_the_declared_one():
    from agents.model import opp_intent
    from agents.training import opponent_classes as oc

    assert opp_intent.OPP_CLASS_NAMES == oc.OPP_CLASS_NAMES
    assert opp_intent.OPP_CLASS_BOT == oc.OPP_CLASS_BOT
    assert len(oc.OPP_CLASS_NAMES) == oc.N_OPP_CLASSES
