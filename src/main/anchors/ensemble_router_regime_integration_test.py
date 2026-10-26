"""The ENSEMBLE peer's regime instrument (``metamon:PastaMittens``, the Kakuna router), run in Metamon's interpreter.

An ensemble router is not an ``amago`` ``Agent`` and has no argmax, so the argmax probe can neither see nor
verify it: before this instrument every ensemble unit read ``regime_verified_decisions = false`` and counted
nothing (``measurements/metamon_strongest_2026-10-25/PREREG.md`` precondition 3). The router is greedy-equivalent
because it is DETERMINISTIC, so the instrument checks exactly that per call: every RNG state byte-equal before
and after. Each test here fails if the instrument is reverted.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap

import pytest

from main.anchors.config import load_config
from main.anchors.windowed_attention_integration_test import _SCRIPT, _metamon_or_skip, _run
from utils.paths import src_path

_ENSEMBLE_MODULE = "metamon.rl.experimental.ensemble.ensemble"


def _run_light(cfg, body: str) -> dict:
    """``metamon_side`` loaded in THIS repo's interpreter (it needs only the stdlib), with NO ``import metamon``: the
    package's ``__init__`` pulls torch + amago (7-13 s of imports beside a live run, the cost that made the two real-
    class tests below 12-22 s). The body gets ``ms`` and ``ens_dir`` (the checkout's ensemble package directory)."""
    code = textwrap.dedent(f"""
        import importlib.util, json, sys, types
        spec = importlib.util.spec_from_file_location("metamon_side", {str(_SCRIPT)!r})
        ms = importlib.util.module_from_spec(spec); spec.loader.exec_module(ms)
        ens_dir = {str(cfg.checkout / "metamon" / "rl" / "experimental" / "ensemble")!r}
    """) + textwrap.dedent(body)
    proc = subprocess.run([sys.executable, "-c", code], env={"PYTHONPATH": str(src_path()), "CUDA_VISIBLE_DEVICES": "",
                                                              "PATH": "/usr/bin:/bin"},
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])

pytestmark = pytest.mark.integration


def test_install_names_a_router_class_that_the_checkout_defines() -> None:
    """The cheap routine half of the real-class test below. ``RegimeProbe.install`` imports the router inside a
    ``try/except ImportError`` that falls back to NOT instrumenting it, so a renamed class or module would silently
    return every ensemble unit to ``regime_verified_decisions = false``. Read both sides as SOURCE, no import: the name
    ``install`` imports, and the class (with ``get_actions``) the checkout's module defines."""
    cfg = _metamon_or_skip()
    install = next(n for n in ast.walk(ast.parse(_SCRIPT.read_text()))
                   if isinstance(n, ast.FunctionDef) and n.name == "install")
    imported = [(n.module, a.name) for n in ast.walk(install) if isinstance(n, ast.ImportFrom) and n.module
                and n.module.startswith("metamon.rl.experimental.ensemble") for a in n.names]
    assert imported == [(_ENSEMBLE_MODULE, "HeuristicRouterEnsemblePolicy")], imported
    path = cfg.checkout.joinpath(*_ENSEMBLE_MODULE.split(".")).with_suffix(".py")
    cls = next((n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == "HeuristicRouterEnsemblePolicy"), None)
    assert cls is not None, f"{path} defines no HeuristicRouterEnsemblePolicy"
    assert any(isinstance(n, ast.FunctionDef) and n.name == "get_actions" for n in cls.body), \
        "the router class has no get_actions of its own to wrap (it would wrap an inherited one)"


@pytest.mark.slow  # 12.8 s beside a live run: imports torch + amago + the real ensemble module
def test_install_instruments_the_real_router_class() -> None:
    cfg = _metamon_or_skip()
    out = _run(cfg, """
        from metamon.rl.experimental.ensemble.ensemble import HeuristicRouterEnsemblePolicy as R
        before = R.get_actions
        ms.RegimeProbe(verify=True).install()
        print(json.dumps({"wrapped": R.get_actions is not before,
                          "name": R.get_actions.__qualname__}))
    """)
    assert out["wrapped"] and "wrap_router" in out["name"], out


def test_a_deterministic_router_reads_1_and_a_sampling_one_does_not() -> None:
    """Each RNG the router could draw from fails the check; another THREAD's Python draw does not (the live
    smoke's 1-in-23 false alarm, 2026-10-25: poke-env's loop thread shares ``random``'s state)."""
    cfg = _metamon_or_skip()
    out = _run(cfg, """
        import random, threading, torch, numpy as np

        def run(draw):
            class Toy:
                def get_actions(self, obs, sample=True):
                    if draw == "torch":
                        torch.rand(1)
                    elif draw == "python":
                        random.random()
                    elif draw == "numpy":
                        np.random.rand()
                    elif draw == "other_thread":   # poke-env's loop thread drawing while the router computes
                        t = threading.Thread(target=lambda: [random.choice([1, 2]) for _ in range(3)])
                        t.start(); t.join()
                    return 0, None
            probe = ms.RegimeProbe(verify=True)
            ms.wrap_router(Toy, probe)
            for _ in range(5):
                Toy().get_actions({}, sample=False)
            rep = probe.report()
            return {"rate": rep["router_no_rng_rate"], "n": rep["router_checked"],
                    "kw": rep["sample_kwarg_values"], "argmax": rep.get("argmax_match_rate"),
                    "greedy_ok": ms.decisions_verified("greedy", rep.get("argmax_match_rate"), rep["n_decisions"],
                                                       rep["router_no_rng_rate"], rep["router_checked"])}
        print(json.dumps({d: run(d) for d in ("none", "torch", "python", "numpy", "other_thread")}))
    """)
    want = {"rate": 1.0, "n": 5, "kw": [False], "argmax": None, "greedy_ok": True}
    assert out["none"] == want and out["other_thread"] == want, out
    for d in ("torch", "python", "numpy"):
        assert out[d]["rate"] == 0.0 and out[d]["greedy_ok"] is False, (d, out[d])


@pytest.mark.parametrize("regime, rate, n_dec, router_rate, n_router, ok", [
    ("greedy", None, 40, 1.0, 40, True),       # a deterministic router in greedy
    ("greedy", None, 40, 0.975, 40, False),    # one call drew a random number
    ("t1", None, 40, 1.0, 40, False),          # a router cannot sample: t1 is never verified
    ("greedy", 1.0, 40, None, 0, True),        # a single policy: the argmax rule, unchanged
    ("t1", 0.647, 40, None, 0, True),
    ("greedy", None, 0, None, 0, False),       # nothing checked is never a pass
])
def test_the_rule(regime, rate, n_dec, router_rate, n_router, ok) -> None:
    cfg = _metamon_or_skip()
    out = _run(cfg, f"""
        print(json.dumps(ms.decisions_verified({regime!r}, {rate!r}, {n_dec!r}, {router_rate!r}, {n_router!r})))
    """)
    assert out is ok


def test_the_pin_check_on_the_checkouts_preset_file() -> None:
    """The cheap routine half of the real-model test below: ``check_preset`` on a stand-in model whose members are the
    preset file's own entries (built from the JSON, not by importing ``metamon``, which is 13 s of torch + amago). The
    pin passes; a wrong pin and a wrong member list are each refused with the pin named."""
    cfg = _metamon_or_skip()
    name, sha = load_config().opponent("metamon").preset("PastaMittens")
    out = _run_light(cfg, f"""
        import dataclasses
        stub = types.ModuleType("metamon.rl.experimental.ensemble"); stub.__file__ = ens_dir + "/__init__.py"
        for m in ("metamon", "metamon.rl", "metamon.rl.experimental"):
            sys.modules[m] = types.ModuleType(m)
        sys.modules["metamon.rl.experimental.ensemble"] = stub
        presets = json.load(open(ens_dir + "/ensemble_presets.json"))
        Spec = dataclasses.make_dataclass("Spec", [("model_name", str), ("checkpoint", object), ("gxe", float),
                                                   ("proposer_bias", float), ("judge_bias", float),
                                                   ("shortlist_k", int), ("proposal_roles", tuple),
                                                   ("action_temperature", float)])
        def model_of(preset):
            members = [Spec(**{{**e, "proposal_roles": tuple(e["proposal_roles"])}}) for e in presets[preset]]
            return type("Ens", (), {{"MEMBER_SPECS": members}})()
        other = next(k for k in presets if k != {name!r})
        out = {{"good": ms.check_preset(model_of({name!r}), {name!r}, {sha!r})}}
        for label, (model, pin) in {{"wrong_pin": (model_of({name!r}), "0" * 64),
                                     "wrong_members": (model_of(other), {sha!r})}}.items():
            try:
                ms.check_preset(model, {name!r}, pin)
                out[label] = "played"
            except SystemExit as exc:
                out[label] = str(exc)
        print(json.dumps(out))
    """)
    assert out["good"] == {"preset": name, "preset_sha256": sha}, out
    assert "the pin is" in out["wrong_pin"] and "the pin is" in out["wrong_members"], out


@pytest.mark.slow  # 22 s beside a live run: builds the two real nickname ensembles through `metamon.rl.pretrained`
def test_the_pinned_preset_passes_and_a_wrong_pin_is_refused() -> None:
    cfg = _metamon_or_skip()
    name, sha = load_config().opponent("metamon").preset("PastaMittens")
    out = _run(cfg, f"""
        from metamon.rl.pretrained import get_pretrained_model
        m = get_pretrained_model("PastaMittens")
        good = ms.check_preset(m, {name!r}, {sha!r})
        out = {{"good": good}}
        for label, (model, pin) in {{"wrong_pin": (m, "0" * 64),
                                     "wrong_members": (get_pretrained_model("Exeggcute"), {sha!r})}}.items():
            try:
                ms.check_preset(model, {name!r}, pin)
                out[label] = "played"
            except SystemExit as exc:
                out[label] = str(exc)
        print(json.dumps(out))
    """)
    assert out["good"] == {"preset": name, "preset_sha256": sha}, out
    assert "the pin is" in out["wrong_pin"] and "the pin is" in out["wrong_members"], out
    assert json.dumps(out)  # the record is JSON-serialisable for the peer report
