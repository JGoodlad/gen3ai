"""Registered human-trained models (``utils.human_models``): the load against the committed stub, the smoke and
retired statuses, the marker and the throwing guard, the digest ``init`` copies against. Synthetic models only
(``utils.human_models_testkit``). Each test names the revert it fails on."""
import json
import shutil

import pytest

from utils import human_models as HM
from utils.human_models_testkit import TEST_HUMAN_MODEL, TEST_ROWSET, install_human_model, tiny_state


def test_load_returns_the_verified_identity(monkeypatch, tmp_path):
    d = install_human_model(monkeypatch, tmp_path)
    got = HM.load(TEST_HUMAN_MODEL)
    stub = HM.stub(TEST_HUMAN_MODEL)
    assert got["id"] == TEST_HUMAN_MODEL and got["rowset"] == TEST_ROWSET and got["human_data"] is True
    assert got["zip_path"] == str(d / f"{TEST_HUMAN_MODEL}.zip") and got["n_trained_keys"] == 2
    for k in HM.IDENTITY_FIELDS:
        assert got[k] == stub[k]
    assert stub["status"] == "measurement" and stub["human_data"] is True and stub["held_out"]["report"]["nll"] == 1.0
    assert HM.trained_keys(TEST_HUMAN_MODEL) == ["a.bias", "a.weight"]
    state = HM.read_policy_state(got["zip_path"])
    assert got["trained_keys_sha256"] == HM.state_digest(state, ["a.weight", "a.bias"])


def test_an_unregistered_or_missing_model_is_refused(monkeypatch, tmp_path):
    d = install_human_model(monkeypatch, tmp_path, register=False)
    with pytest.raises(HM.HumanModelError, match="not registered"):
        HM.load(TEST_HUMAN_MODEL)
    HM.register(d)
    shutil.rmtree(d)
    with pytest.raises(HM.HumanModelError, match="is missing"):
        HM.load(TEST_HUMAN_MODEL)


@pytest.mark.parametrize("name", [f"{TEST_HUMAN_MODEL}.zip", HM.CONFIG_NAME, HM.MANIFEST_NAME])
def test_a_file_changed_after_registration_is_refused(monkeypatch, tmp_path, name):
    """Fails on revert of the stub comparison in ``load`` (a swapped checkpoint would train under the old id)."""
    d = install_human_model(monkeypatch, tmp_path)
    p = d / name
    if name == HM.MANIFEST_NAME:
        man = json.loads(p.read_text())
        man["seed"] = 2
        p.write_text(json.dumps(man, indent=1))
    else:
        p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(HM.HumanModelError, match="does not match its committed stub"):
        HM.load(TEST_HUMAN_MODEL)


def test_a_smoke_model_loads_only_under_debug_and_registers_only_as_smoke(monkeypatch, tmp_path):
    """Fails on revert of either smoke rule (a model trained under recipe overrides would enter a real run)."""
    d = install_human_model(monkeypatch, tmp_path, smoke=True)
    assert HM.stub(TEST_HUMAN_MODEL)["status"] == HM.SMOKE_STATUS
    with pytest.raises(HM.HumanModelError, match="SMOKE model"):
        HM.load(TEST_HUMAN_MODEL)
    assert HM.load(TEST_HUMAN_MODEL, debug=True)["status"] == HM.SMOKE_STATUS
    with pytest.raises(HM.HumanModelError, match="registers as 'smoke' only"):
        HM.register(d, "measurement")


def test_a_retired_model_and_an_unknown_status_are_refused(monkeypatch, tmp_path):
    d = install_human_model(monkeypatch, tmp_path, status="retired")
    with pytest.raises(HM.HumanModelError, match="RETIRED"):
        HM.load(TEST_HUMAN_MODEL, debug=True)
    with pytest.raises(HM.HumanModelError, match="is not one of"):
        HM.register(d, "production")


def test_register_takes_only_the_stores_own_directory(monkeypatch, tmp_path):
    d = install_human_model(monkeypatch, tmp_path, register=False)
    other = tmp_path / "elsewhere" / TEST_HUMAN_MODEL
    shutil.copytree(d, other)
    with pytest.raises(HM.HumanModelError, match="is not the store's directory"):
        HM.register(other)
    assert HM.registry() == {}


def test_register_rewrites_the_tracked_registry_in_its_own_format(monkeypatch, tmp_path):
    """The registry is a tracked file: a registration changes the ``models`` block and nothing else (indent 1,
    escaped non-ASCII, the trailing newline), so the commit that carries a stub is a one-block diff."""
    d = install_human_model(monkeypatch, tmp_path, register=False)
    before = json.dumps({"doc": "\u00a75", "models": {}, "models_schema": "x"}, indent=1) + "\n"
    HM.registry_path().write_text(before)
    HM.register(d)
    after = HM.registry_path().read_text()
    doc = json.loads(after)
    assert after.endswith("}\n") and "\\u00a75" in after
    assert list(doc) == ["doc", "models", "models_schema"] and list(doc["models"]) == [TEST_HUMAN_MODEL]
    doc["models"] = {}
    assert json.dumps(doc, indent=1) + "\n" == before


def test_the_marker_finds_a_human_model_by_the_store_and_by_its_manifest(monkeypatch, tmp_path):
    """THE THROWING GUARD's two rules. Fails on revert of either (a copy outside the store, or a store file with its
    manifest deleted, would pass as an ordinary checkpoint)."""
    d = install_human_model(monkeypatch, tmp_path)
    zp = d / f"{TEST_HUMAN_MODEL}.zip"
    assert "lies under the human-model store" in HM.marker_of(str(zp))
    assert "lies under the human-model store" in HM.marker_of(str(d))
    (d / HM.MANIFEST_NAME).unlink()
    assert "lies under the human-model store" in HM.marker_of(str(zp))        # the store rule alone
    copy = tmp_path / "run" / "checkpoints"
    install_human_model(monkeypatch, tmp_path, model_id="hclone_u")
    shutil.copytree(HM.model_dir("hclone_u"), copy)
    assert "marks it as a human-trained model" in HM.marker_of(str(copy / "hclone_u.zip"))   # the manifest rule alone
    assert "marks it as a human-trained model" in HM.marker_of(str(copy))
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "final_model.zip").write_bytes(b"")
    assert HM.marker_of(str(plain / "final_model.zip")) is None and HM.marker_of(None) is None
    HM.refuse_human_model(str(plain / "final_model.zip"), "a test")
    with pytest.raises(HM.HumanModelError, match="only through --human-data"):
        HM.refuse_human_model(str(zp), "a test")
    assert issubclass(HM.HumanModelError, ValueError)        # the stable-opponent path's FATAL catch


def test_the_state_digest_reads_the_named_tensors_bytes():
    import torch as th

    s = tiny_state()
    full, part = HM.state_digest(s), HM.state_digest(s, ["a.weight", "a.bias"])
    assert full != part and part == HM.state_digest(s, ["a.bias", "a.weight"])      # order-free
    t = {k: v.clone() for k, v in s.items()}
    t["b.bias"][0] += 1
    assert HM.state_digest(t, ["a.weight", "a.bias"]) == part and HM.state_digest(t) != full
    t["a.bias"][0] = th.nextafter(t["a.bias"][0], th.tensor(9.0))
    assert HM.state_digest(t, ["a.weight", "a.bias"]) != part                       # one ulp is another digest
