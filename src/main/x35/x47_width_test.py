"""The growth descriptors on a WIDTH growth (``main.x35.x47_report``; N4's E1 / E2 and REVIEW M1's two nulls,
``designs/endstate/design_n4_capacity_screen.md`` §5). Real toy networks (``trunk_ffn_width_test``'s builder: T*'s
architecture, CPU, one thread, fp32 eager) grown 256 -> 1024 by ``growth.apply_growth`` in tmp dirs, plus constructed
state dicts for the arithmetic.

Each test fails on the revert it names:
* the zeroed twin (``growth_zeroer`` -> the ``trunk_ffn_dim`` row's ``zero_new`` hook) restores the PARENT's logits and
  values bit for bit after the branch is perturbed — fails if the hook is a no-op or misses a branch;
* E1 on a freshly grown toy, through ``run_norms`` and the run's own ``model_config.json``, is exactly 0.0 on every
  round and prefix — fails if the width reader folds the branch into the base map (the old hazard) or is undispatched;
* the REFUSALS (exit 2, never an empty reading): a run declaring no growth, a field no reader handles (B-KFFN), a
  reader removed from ``NORM_READERS`` — fails if the 2026-10-31 hazard (exit 0, ``reading {}``) comes back;
* the E1 null's arithmetic by hand and its refusal of a wide state; the E2 null's revert copies T*'s ``linear2``
  (weight and bias) and nothing else.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch

from agents.model import growth as G
from main.x35 import x47 as X
from main.x35 import x47_report as XR
from utils.torch_state_guard import torch_globals

FIELDS = {"trunk_ffn_dim": {"parent": 256, "child": 1024}}
PRE = "features_extractor.team_transformer."


@pytest.fixture(scope="module")
def rows() -> Tuple[np.ndarray, np.ndarray]:
    from agents.model.growth_test import _rows
    return _rows()


def _parent(rows: Tuple[np.ndarray, np.ndarray]) -> Any:
    from agents.model.trunk_ffn_width_test import _parent as parent
    return parent(rows)


def _fwd(model: Any, rows: Tuple[np.ndarray, np.ndarray]) -> Dict[str, np.ndarray]:
    model.policy.set_training_mode(False)
    return G.forward_record(model.policy, *rows)


def _save_run(model: Any, run, growth: Any = None) -> None:
    """A run dir as the trainer leaves one: ``final_model.zip`` and ``model_config.json`` (with ``growth``)."""
    import dataclasses

    from agents.model.snapshot import recorded_model_version, save_model_snapshot

    run.mkdir(parents=True)
    model.save(str(run / "final_model"))
    version = recorded_model_version(model)
    if growth is not None:
        version = dataclasses.replace(version, growth=growth)
    save_model_snapshot(str(run), version, git_hash="test")


# ------------------------------------------------------------------------------------------ E2's zeroed twin
def test_zeroing_the_width_branch_restores_the_parents_function_bit_for_bit(rows) -> None:
    with torch_globals(num_threads=1):
        model = _parent(rows)
        parent = _fwd(model, rows)
        G.apply_growth(model, FIELDS, rows=rows)
        with torch.no_grad():                                 # a "trained" branch: both maps moved off their init
            for n, p in model.policy.named_parameters():
                if ".ffn_extra." in n:
                    p.add_(0.05 * torch.randn_like(p))
        moved = _fwd(model, rows)
        assert not np.array_equal(moved["logits"], parent["logits"])
        n = XR.growth_zeroer(FIELDS)(model.policy)
        back = _fwd(model, rows)
    assert n >= 2                                             # the toy's two rounds (tied extractors counted once)
    assert np.array_equal(back["logits"], parent["logits"]) and np.array_equal(back["values"], parent["values"])


def test_the_zeroer_refuses_a_growth_without_a_zero_init_block() -> None:
    assert G.GROWABLE["trunk_ffn_by_kind"].zero_new is None                   # copies, not a zero-init block
    with pytest.raises(ValueError, match="no zero_new hook"):
        XR.growth_zeroer({"trunk_ffn_by_kind": {"parent": "off", "child": "on"}})
    with pytest.raises(ValueError, match="no zero_new hook"):
        XR.growth_zeroer({})
    with pytest.raises(ValueError, match="found no new block"):              # declared, but nothing to zero
        XR.growth_zeroer(FIELDS)(torch.nn.Linear(2, 2))


# ------------------------------------------------------------------------------------------ E1
def test_e1_on_a_freshly_grown_toy_is_exactly_zero_and_the_cli_reads_it(rows, tmp_path, capsys) -> None:
    with torch_globals(num_threads=1):
        model = _parent(rows)
        report = G.apply_growth(model, FIELDS, rows=rows)
        _save_run(model, tmp_path / "grown", growth={"fields": FIELDS, **report})
    assert X._main(["norms", "--run", str(tmp_path / "grown")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["reader"] == "trunk_ffn_dim" and out["last_node"] == "final"
    norms = out["series"][-1]["norms"]
    assert PRE in norms
    ratios = [r["ratio"] for p in norms.values() for r in p["rounds"].values()]
    assert ratios and all(x == 0.0 for x in ratios)
    assert all(r["linear2"] > 0.0 for p in norms.values() for r in p["rounds"].values())
    assert out["reading"] == {"transformer_layers.0": XR.NEAR_IDENTITY, "transformer_layers.1": XR.NEAR_IDENTITY}


def test_e1_reads_the_branch_apart_from_the_base_map() -> None:
    sd = {}
    for j in range(2):
        sd[f"{PRE}transformer_layers.{j}.linear2.weight"] = torch.full((4, 4), 2.0)              # norm 8
        sd[f"{PRE}transformer_layers.{j}.ffn_extra.linear2.weight"] = torch.full((4, 12), 0.04 * (j + 1))
    n = XR.width_norms(sd)
    r0, r1 = n[PRE]["rounds"]["transformer_layers.0"], n[PRE]["rounds"]["transformer_layers.1"]
    assert r0["linear2"] == pytest.approx(8.0) and r0["ffn_extra_linear2"] == pytest.approx(0.04 * 48 ** 0.5)
    assert r0["ratio"] == pytest.approx(0.04 * 48 ** 0.5 / 8) and r1["ratio"] == pytest.approx(0.08 * 48 ** 0.5 / 8)
    assert XR.width_reading(n) == {"transformer_layers.0": XR.ENGAGED, "transformer_layers.1": XR.ENGAGED}
    with pytest.raises(ValueError, match="not a width-grown"):
        XR.width_norms({k: v for k, v in sd.items() if "ffn_extra" not in k})
    with pytest.raises(ValueError, match="not both"):                         # one round without its branch
        XR.width_norms({k: v for k, v in sd.items() if k != f"{PRE}transformer_layers.1.ffn_extra.linear2.weight"})


# ------------------------------------------------------------------------------------------ the refusals
def _cfg(run, fields: Any) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "final_model.zip").write_bytes(b"z")
    (run / "model_config.json").write_text(json.dumps({"growth": {"fields": fields} if fields else None}))


def test_the_descriptors_refuse_an_undeclared_or_unhandled_growth(tmp_path, capsys, monkeypatch) -> None:
    """The 2026-10-31 hazard: ``norms`` on a width-grown toy exited 0 with ``reading {}``. Now every growth the
    readers do not handle exits 2 BEFORE any load (the loader would fail on these 1-byte zips)."""
    _cfg(tmp_path / "plain", None)
    _cfg(tmp_path / "kffn", {"trunk_ffn_by_kind": {"parent": "off", "child": "on"}})
    _cfg(tmp_path / "two", {"trunk_layers": {"parent": 2, "child": 4}, **FIELDS})
    (tmp_path / "noconfig").mkdir()
    for run in ("plain", "kffn", "two", "noconfig"):
        assert X._main(["norms", "--run", str(tmp_path / run)]) == 2, run
        assert "norms REFUSED" in capsys.readouterr().err
    for run in ("plain", "kffn", "noconfig"):
        assert X._main(["policy-diff", "--run", str(tmp_path / run), "--bank", str(tmp_path / "nobank")]) == 2, run
        assert "policy-diff REFUSED" in capsys.readouterr().err
    # a declared width growth whose reader is withdrawn is refused, never read by another field's reader
    _cfg(tmp_path / "wide", FIELDS)
    monkeypatch.setattr(XR, "NORM_READERS", {"trunk_layers": XR.NORM_READERS["trunk_layers"]})
    assert X._main(["norms", "--run", str(tmp_path / "wide")]) == 2
    assert "norms REFUSED" in capsys.readouterr().err


def test_a_depth_run_with_no_new_round_is_an_empty_reading_and_refused(tmp_path) -> None:
    run = tmp_path / "d"
    _cfg(run, {"trunk_layers": {"parent": 2, "child": 4}})
    sd = {f"{PRE}transformer_layers.0.out_proj.weight": torch.ones(2, 2),
          f"{PRE}transformer_layers.0.linear2.weight": torch.ones(2, 2)}
    with pytest.raises(ValueError, match="empty reading"):
        XR.run_norms(str(run), loader=lambda f: sd)


# ------------------------------------------------------------------------------------------ REVIEW M1's nulls
def test_the_e1_null_is_the_base_drift_scaled_to_the_branch_by_hand() -> None:
    parent = {f"{PRE}transformer_layers.0.linear2.weight": torch.ones(4, 4)}                    # norm 4
    final = {f"{PRE}transformer_layers.0.linear2.weight": torch.full((4, 4), 1.1)}              # drift norm 0.4
    out = XR.width_drift_null(final, parent, ffn_dim=1024)
    row = out["rounds"][PRE]["transformer_layers.0"]
    assert out["scale"] == pytest.approx(3 ** 0.5)
    assert row["drift"] == pytest.approx(0.1) and row["e1_null"] == pytest.approx(0.1 * 3 ** 0.5)
    assert XR.width_drift_null(parent, parent, ffn_dim=1024)["rounds"][PRE]["transformer_layers.0"]["drift"] == 0.0
    wide = dict(final, **{f"{PRE}transformer_layers.0.ffn_extra.linear2.weight": torch.zeros(4, 12)})
    with pytest.raises(ValueError, match="NARROW control"):
        XR.width_drift_null(wide, parent, ffn_dim=1024)
    with pytest.raises(ValueError, match="must exceed"):
        XR.width_drift_null(final, parent, ffn_dim=256)


def test_the_e2_null_reverts_every_base_linear2_and_nothing_else(rows) -> None:
    with torch_globals(num_threads=1):
        model = _parent(rows)
        tstar = copy.deepcopy(model.policy.state_dict())
        with torch.no_grad():
            for p in model.policy.parameters():
                p.add_(0.01 * torch.randn_like(p))
        moved = copy.deepcopy(model.policy.state_dict())
        n = XR.revert_base_linear2(tstar)(model.policy)
        after = model.policy.state_dict()
    l2 = [k for k in after if XR._BASE_L2.match(k.rsplit(".", 1)[0])]
    assert n >= 2 and l2 and any(k.endswith(".bias") for k in l2)
    assert all(torch.equal(after[k], tstar[k]) for k in l2)
    assert all(torch.equal(after[k], moved[k]) for k in after if k not in l2)
    with torch_globals(num_threads=1):
        G.apply_growth(model, FIELDS, rows=rows)
        with pytest.raises(ValueError, match="NARROW control"):
            XR.revert_base_linear2(tstar)(model.policy)
