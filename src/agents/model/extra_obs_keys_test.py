"""The DECLARED extra-obs-key registry, and the crash it exists to make unrepeatable.

THE DEFECT THIS FILE PINS (`ai_v12_14_ladder_truevalue`, launched at 377a5aa1, child dead two
minutes in at env init, exit 1). Two independently-correct facts had never been exercised
together:

* `gen3_value_true_team_v1`'s value route (DELETED, deletion pass L2) read its own Dict key,
  `opp_true_team`, and **RAISED** on a missing one — deliberately, because a silent skip reads
  exactly like a route that learned nothing (the gen-12 dead-tail bug).
* `gen3_forkserver_preload_v1` traced the compiled extractor once, on a hand-built **one-key**
  obs, so 48 workers could inherit the graph.

The fix was not "add the key at four sites" — it is that the (attribute -> key, shape) mapping is
DECLARED once in `extra_obs_keys` and every synthetic-obs caller builds from it. **The table is
EMPTY today** (the only row left with its lever), so the real extractor's forward reads
`obs["observation"]` alone; these tests hold the MECHANISM — the drift gate over the forward's
source, the registry's builders on a stand-in row, and the "no site hand-builds its dict" pins —
so the next obs-key flag cannot reintroduce the crash.
"""
import ast
import inspect
import types

import torch

from agents.model import extra_obs_keys as EOK
from agents.model.extra_obs_keys import (
    BASE_OBS_KEYS, EXTRA_OBS_KEYS, ExtraObsKey, required_extra_obs_keys, synthetic_obs,
    zero_extra_obs,
)


def _fe(**cfg):
    """A serverless extractor from an arch-kwargs dict, and its layout (what the deleted forkserver
    preload built — deletion pass U3 / R6 — kept here as the test's own fixture)."""
    import gymnasium as gym
    import numpy as np

    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    fe = Gen3FeaturesExtractor(space, layout=layout, mappings=mappings,
                               **{k: v for k, v in cfg.items() if k in sig})
    fe.eval()
    return fe, layout


def preload_trace_obs(fe, layout):
    """The synthetic obs a compile trace runs on: `observation` plus every declared flag-gated key."""
    return synthetic_obs(fe, int(layout["total_dim"]))


def scan_obs_key_reads(module):
    """``(found, unresolved)`` — every ``obs.get(X)`` / ``obs[X]`` in `module`'s source, with each key
    resolved to a string (a literal or a module-level string constant), else left unresolved."""
    tree = ast.parse(inspect.getsource(module))
    found, unresolved = set(), []

    def _keyname(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            val = getattr(module, node.id, None)
            if isinstance(val, str):
                return val
        return None

    for node in ast.walk(tree):
        arg = None
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "obs" and node.args):
            arg = node.args[0]
        elif (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == "obs"):
            arg = node.slice
        if arg is None:
            continue
        name = _keyname(arg)
        if name is None:
            unresolved.append(ast.dump(arg))
        else:
            found.add(name)
    return found, unresolved


# ------------------------------------------------------------------ the declaration itself
def test_the_registry_is_EMPTY_and_every_declared_row_names_a_REAL_attribute():
    """Today no flag adds a Dict key to the extractor's forward. If a row is ever added it must
    name an attribute the flag builds (and which is None without it), or
    `required_extra_obs_keys` answers a different question than the seam asks."""
    assert EXTRA_OBS_KEYS == (), "a row was added: extend this test with its flag round-trip"
    for entry in EXTRA_OBS_KEYS:                                # pragma: no cover - empty today
        off, _ = _fe()
        on, _ = _fe(**{entry.flag: True})
        assert getattr(off, entry.attr, "missing") is None
        assert getattr(on, entry.attr, None) is not None


def test_the_AST_of_the_forward_reads_no_UNDECLARED_obs_key():
    """THE DRIFT GATE. Every `obs.get(X)` / `obs[X]` in `extractor_forward` must resolve to a key
    this registry declares (or a base key). A new route that reads a new key therefore cannot
    reach a launch without a row here — which is the whole mechanism, since the synthetic-obs
    callers build from the rows and nothing else."""
    from agents.model import extractor_forward as EF

    found, unresolved = scan_obs_key_reads(EF)
    assert not unresolved, (
        "extractor_forward reads an obs key this gate cannot resolve to a literal — spell the key "
        f"out as a module-level constant so the registry can be checked against it: {unresolved}")
    declared = set(BASE_OBS_KEYS) | {e.key for e in EXTRA_OBS_KEYS}
    undeclared = found - declared
    assert not undeclared, (
        f"extractor_forward reads obs key(s) {sorted(undeclared)} that agents.model.extra_obs_keys "
        "does not declare. Every synthetic-obs caller (the forkserver preload, the round-trip "
        "smoke, compile_opponents, warmstart) builds its dict from that table, "
        "so an undeclared key is a crash at the next launch that turns the flag on.")
    # …and the converse: a row for a key nothing reads would make every caller build dead weight.
    assert {e.key for e in EXTRA_OBS_KEYS} <= found


def test_the_AST_scanner_is_NOT_vacuous():
    """The gate above is empty-handed today (`found` is ∅), so its scanner needs its own control: it
    must FIND a read in a module that has one, in all three spellings, and report an unresolvable
    key rather than skip it."""
    mod = types.ModuleType("scanner_probe")
    mod.__dict__["_K"] = "via_constant"
    src = (
        "def f(obs, name):\n"
        "    a = obs.get('literal_get')\n"
        "    b = obs['literal_subscript']\n"
        "    c = obs.get(_K)\n"
        "    d = obs.get(name)\n"
    )
    import linecache
    fname = "<scanner_probe>"
    linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
    exec(compile(src, fname, "exec"), mod.__dict__)
    mod.__file__ = fname
    found, unresolved = scan_obs_key_reads(mod)
    assert found == {"literal_get", "literal_subscript", "via_constant"}
    assert len(unresolved) == 1


# ------------------------------------------------------------------ the builders, on a stand-in row
def test_the_registry_builders_honour_a_declared_row(monkeypatch):
    """`required_extra_obs_keys` / `zero_extra_obs` / `synthetic_obs` on a STAND-IN row: the key is
    demanded exactly when the extractor attribute is non-None, at the declared shape and batch."""
    row = ExtraObsKey(key="x_probe_key", attr="x_probe_attr", shape=(3, 2), flag="x_probe")
    monkeypatch.setattr(EOK, "EXTRA_OBS_KEYS", (row,))
    on = types.SimpleNamespace(x_probe_attr=object())
    off = types.SimpleNamespace(x_probe_attr=None)
    assert [e.key for e in required_extra_obs_keys(on)] == ["x_probe_key"]
    assert required_extra_obs_keys(off) == ()
    obs = synthetic_obs(on, 7, batch=5, action_mask=True)
    assert set(obs) == {"observation", "action_mask", "x_probe_key"}
    assert tuple(obs["x_probe_key"].shape) == (5, 3, 2) and float(obs["x_probe_key"].abs().sum()) == 0.0
    assert obs["observation"].shape == (5, 7)
    assert set(synthetic_obs(off, 7)) == {"observation"}
    assert set(zero_extra_obs(on)) == {"x_probe_key"} and zero_extra_obs(off) == {}


# ------------------------------------------------------------------ (a) the forward runs
def test_the_synthetic_preload_obs_FORWARDS():
    """The obs the preload traces on runs the real forward (eager, not compiled: the failure this
    pins is in the obs DICT, not in the codegen)."""
    fe, layout = _fe()
    obs = preload_trace_obs(fe, layout)
    assert set(obs) == {"observation"}
    with torch.no_grad():
        pi, vf = fe(obs)
    assert pi.shape[0] == vf.shape[0] == 1


def test_the_batch_and_device_arguments_are_honoured():
    """A compile warm-up runs at a real batch and `warmstart` tops up a chunk of REAL obs — both
    would produce a broadcast error, not a crash, if the block came back at batch 1."""
    fe, layout = _fe()
    obs = synthetic_obs(fe, int(layout["total_dim"]), batch=5, action_mask=True)
    assert obs["observation"].shape[0] == 5
    assert obs["action_mask"].shape == (5, 11)
    with torch.no_grad():
        pi, _ = fe(obs)
    assert pi.shape[0] == 5


def test_the_round_trip_smoke_and_the_two_warmups_all_build_from_the_registry():
    """The three TRAINING-RUN synthetic-obs sites are the ones whose crash costs a GPU-hour. Pinned
    by SOURCE because there is no cheap way to run `compile_opponents` / `warmstart` / the round-trip
    smoke in a unit test — and a source check is exactly strong enough for the claim, which is 'this
    site does not hand-build its dict any more'. (The learner compile's site went with the
    extractor-only gate, K1 2026-10-02: the regions' gate and prewarm build every key of the
    policy's OBSERVATION SPACE — `compile_trainer._prewarm_obs` — so no key can be missing there.)"""
    from agents.model import compile_opponents
    from agents.training import warmstart
    from main.train import lifecycle

    for mod, fn in ((compile_opponents, "zero_extra_obs"),
                    (warmstart, "zero_extra_obs"), (lifecycle, "synthetic_obs")):
        assert fn in inspect.getsource(mod), (
            f"{mod.__name__} no longer builds its synthetic obs from agents.model.extra_obs_keys "
            "— it is one obs-key flag away from the ai_v12_14_ladder_truevalue crash")
