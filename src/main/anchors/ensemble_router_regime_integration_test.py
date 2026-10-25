"""The ENSEMBLE peer's regime instrument (``metamon:PastaMittens``, the Kakuna router), run in Metamon's interpreter.

An ensemble router is not an ``amago`` ``Agent`` and has no argmax, so the argmax probe can neither see nor
verify it: before this instrument every ensemble unit read ``regime_verified_decisions = false`` and counted
nothing (``measurements/metamon_strongest_2026-10-25/PREREG.md`` precondition 3). The router is greedy-equivalent
because it is DETERMINISTIC, so the instrument checks exactly that per call: every RNG state byte-equal before
and after. Each test here fails if the instrument is reverted.
"""
from __future__ import annotations

import json

import pytest

from main.anchors.config import load_config
from main.anchors.windowed_attention_integration_test import _metamon_or_skip, _run

pytestmark = pytest.mark.integration


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
