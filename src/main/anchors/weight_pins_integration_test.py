"""The peer's weight pins, run in Metamon's own interpreter against Metamon's real cache (offline, no network, no server).

``install_weight_pins`` is what ``metamon_side.main`` calls before the model is built: it forces the hub library
offline, replaces ``huggingface_hub.hf_hub_download`` by the pinned downloader and hashes every pinned file. Here the
REAL ``metamon.rl.pretrained`` model then resolves its checkpoint through the replacement, and a wrong sha, a revision
that is not in the cache and an online inheritance are each REFUSED. SKIPS with a named reason where Metamon is absent.
"""
from __future__ import annotations

import json
import textwrap

import pytest

from main.anchors.config import load_config
from main.anchors.windowed_attention_integration_test import _metamon_or_skip, _run

pytestmark = [pytest.mark.integration, pytest.mark.slow]  # ship gate: imports torch + metamon, a few seconds

_BODY = """
    import types
    from metamon.rl.pretrained import get_pretrained_model
    def args(**kw):
        base = dict(hf_repo={repo!r}, hf_revision={rev!r}, weights_json=json.dumps({weights!r}))
        base.update(kw)
        return types.SimpleNamespace(**base)
    def refused(**kw):
        try:
            ms.install_weight_pins(args(**kw))
        except SystemExit as exc:
            return str(exc)
        return None
"""


def _setup(agent: str = "SmallRL") -> tuple:
    cfg = _metamon_or_skip()
    pin = load_config().metamon.pin(agent)
    return cfg, textwrap.dedent(_BODY.format(repo=pin.repo, rev=pin.revision, weights=pin.weights)), pin


def test_the_real_model_loads_its_checkpoint_only_through_the_pinned_offline_downloader() -> None:
    cfg, body, pin = _setup()
    out = _run(cfg, body + textwrap.dedent("""
        import os
        os.environ["HF_HUB_OFFLINE"] = "0"           # an online caller: the script's import-time force must have won
        record = ms.install_weight_pins(args())
        m = get_pretrained_model("SmallRL")
        path = m.get_path_to_checkpoint(40)
        print(json.dumps({"record": record, "path": path}))
    """))
    rec = out["record"]
    assert rec["hf_hub_offline"] is True and rec["hf_revision"] == pin.revision
    assert rec["weights_verified"] == pin.weights
    assert rec["weights_loaded"] == list(pin.weights)
    assert f"snapshots/{pin.revision}/" in out["path"], "the checkpoint resolves inside the PINNED snapshot"


def test_a_wrong_sha_a_missing_revision_and_an_unpinned_load_are_each_refused() -> None:
    cfg, body, pin = _setup()
    rel = next(iter(pin.weights))
    out = _run(cfg, body + textwrap.dedent(f"""
        res = {{"sha": refused(weights_json=json.dumps({{{rel!r}: "0" * 64}})),
                "revision": refused(hf_revision="b" * 40),
                "moving": refused(hf_revision="main")}}
        ms.install_weight_pins(args())
        m = get_pretrained_model("SmallRL")
        try:
            m.get_path_to_checkpoint(39)             # a real file upstream, but NOT the pinned one
            res["unpinned"] = None
        except Exception as exc:
            res["unpinned"] = type(exc).__name__ + ": " + str(exc)
        print(json.dumps(res))
    """))
    assert "hashes to" in out["sha"] and "REFUSING TO PLAY" in out["sha"], out
    assert "not in the local cache" in out["revision"], out
    assert "40-hex" in out["moving"], out
    assert "not a pinned weight file" in out["unpinned"], out
    assert json.dumps(out)
