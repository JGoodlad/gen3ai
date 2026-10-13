"""THE DIAGNOSTIC-FORK CONSENT (`main.train.diagnostic_fork`; `designs/endstate/design_information_levers.md` §6).

`--oracle-reveal` is resume-immutable; a FORK may change it from its parent's recorded spec only under
`--diagnostic-fork-consent "<reason>"`. Pinned here, each refusal by name and the stamp end to end:

* no consent: nothing changes (the loader's `check_oracle_reveal` refuses a flip exactly as before);
* a consented FORK with a real change is STAMPED (schema, reason, date, parent / child spec records, parent model);
* refused: an empty reason, no `--model`, a SAME-RUN resume that flips the spec, a consent with no spec change, a
  fork INTO a registry-named run; a same-run RESTART of a consented fork (its original argv still carries the flag)
  passes silently;
* the loader's gate accepts exactly the change the stamp records and nothing else;
* the stamp lands in the child's immutable lineage block (`run_io._run_lineage`), is read back for the run and every
  descendant (`agents.training.lineage.diagnostic_fork_stamps`), shown by `main.lineage`, `main.sidecar_audit` and
  `main.checkargs`, and `main.baselines` refuses (set) / flags (check) a run that carries it.
"""
from __future__ import annotations

import dataclasses
import json
import types

import pytest

from agents.model.model_version import ModelVersion, ModelVersionError
from agents.model.snapshot import save_model_snapshot
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from main.train import diagnostic_fork as DF


@pytest.fixture(scope="module")
def version():
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    return ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})


def _run(archive, name, version, reveal="off", lineage=None):
    d = archive / name
    (d / "checkpoints").mkdir(parents=True)
    (d / "final_model.zip").write_bytes(b"zip-" + name.encode())
    (d / "checkpoints" / "checkpoint_100_steps.zip").write_bytes(b"ckpt-" + name.encode())
    save_model_snapshot(str(d), dataclasses.replace(version, oracle_reveal=reveal), git_hash="t")
    if lineage is not None:
        meta = json.loads((d / "metadata.json").read_text())
        meta["lineage"] = lineage
        (d / "metadata.json").write_text(json.dumps(meta))
    return d


def _fork(parent, child, consent="knock-in: does the network convert told moves?", requested="ki_moves"):
    return DF.decide(consent=consent, model_path=str(parent / "final_model.zip"), model_dir=str(child),
                     requested=requested, now="2026-10-12T00:00:00+00:00")


def test_without_the_consent_nothing_is_decided_and_the_flip_stays_refused(run_archive, version):
    parent = _run(run_archive, "rb_parent", version)
    assert DF.decide(consent=None, model_path=str(parent / "final_model.zip"), model_dir=str(run_archive / "c"),
                     requested="ki_moves") is None
    with pytest.raises(ModelVersionError, match="oracle_reveal mismatch"):
        version.check_oracle_reveal("ki_moves")


def test_a_consented_fork_with_a_real_change_is_stamped(run_archive, version):
    parent = _run(run_archive, "rb_parent", version)
    stamp = _fork(parent, run_archive / "rb_hib_moves_s3101")
    assert stamp["schema"] == DF.SCHEMA and stamp["reason"].startswith("knock-in")
    assert stamp["date"] == "2026-10-12T00:00:00+00:00"
    assert stamp["parent_spec"]["label"] == "off" and stamp["parent_spec"]["levers"] == []
    assert stamp["child_spec"]["label"] == "moves:seen@0.95" and stamp["child_spec"]["hash"]
    assert stamp["parent_model"].endswith("rb_parent/final_model.zip")
    assert "off -> moves:seen@0.95" in DF.describe(stamp)
    # a consented fork from a REVEAL parent back to off is a change too
    rev = _run(run_archive, "rb_rev", version, reveal="species")
    assert _fork(rev, run_archive / "rb_back", requested="off")["child_spec"]["label"] == "off"


@pytest.mark.parametrize("consent", ["", "   "])
def test_an_empty_reason_is_refused_by_the_parser_and_the_decision(consent):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError, match="non-empty reason"):
        DF.consent_reason(consent)
    with pytest.raises(DF.DiagnosticForkRefused, match="non-empty reason"):
        DF.decide(consent=consent, model_path="x.zip", model_dir="y", requested="ki_moves")


def test_the_parser_carries_the_reason_and_refuses_an_empty_one(capsys):
    from main.train.parser import build_parser

    p = build_parser()
    assert p.parse_args([]).diagnostic_fork_consent is None
    assert p.parse_args(["--diagnostic-fork-consent", " why "]).diagnostic_fork_consent == "why"
    with pytest.raises(SystemExit):
        p.parse_args(["--diagnostic-fork-consent", ""])
    assert "non-empty reason" in capsys.readouterr().err


def test_a_consent_without_a_model_is_refused():
    with pytest.raises(DF.DiagnosticForkRefused, match="without --model"):
        DF.decide(consent="r", model_path=None, model_dir="x", requested="ki_moves")


def test_a_same_run_resume_that_flips_is_refused_even_with_the_consent(run_archive, version):
    run = _run(run_archive, "rb_same", version)
    for ckpt in (run / "checkpoints" / "checkpoint_100_steps.zip", run / "final_model.zip"):
        with pytest.raises(DF.DiagnosticForkRefused, match="SAME-RUN resume can never change"):
            DF.decide(consent="r", model_path=str(ckpt), model_dir=str(run), requested="ki_moves")


def test_a_consent_with_no_change_is_refused_as_meaningless(run_archive, version):
    parent = _run(run_archive, "rb_parent", version, reveal="full")
    for same in ("full", "species:unseen,moves:both,item:both,ability:both,spread:both"):
        with pytest.raises(DF.DiagnosticForkRefused, match="meaningless"):
            _fork(parent, run_archive / "rb_child", requested=same)


def test_a_fork_into_a_registry_named_run_is_refused_and_a_registered_parent_is_allowed(run_archive, version,
                                                                                       monkeypatch):
    parent = _run(run_archive, "rb_parent", version)
    monkeypatch.setattr(DF, "_registry_runs", lambda: {"rb_prod_child": ["production"], "rb_parent": ["closing"]})
    with pytest.raises(DF.DiagnosticForkRefused, match=r"registry names that run \(production\)"):
        _fork(parent, run_archive / "rb_prod_child")
    assert _fork(parent, run_archive / "rb_free_child")["child_spec"]["label"] == "moves:seen@0.95"


def test_the_registry_lookup_reads_the_committed_registry():
    from agents.training import baselines as reg

    runs = DF._registry_runs()
    prod = reg.get("production")
    assert "production" in runs[prod.run]
    assert DF.registry_names_for_run(f"/x/models/{prod.run}") == sorted(runs[prod.run])


def test_a_restart_of_a_consented_fork_passes_and_one_without_a_recorded_consent_is_meaningless(run_archive, version):
    parent = _run(run_archive, "rb_parent", version)
    stamp = _fork(parent, run_archive / "x")
    child = _run(run_archive, "rb_hib_ctrl", version, reveal="moves:seen@0.95",
                 lineage={"schema": "x", "role": "fork", "fork_parent": None, "diagnostic_fork": stamp})
    ckpt = str(child / "checkpoints" / "checkpoint_100_steps.zip")
    assert DF.decide(consent="r", model_path=ckpt, model_dir=str(child), requested="moves:seen@0.95") is None
    plain = _run(run_archive, "rb_plain", version)
    with pytest.raises(DF.DiagnosticForkRefused, match="meaningless"):
        DF.decide(consent="r", model_path=str(plain / "final_model.zip"), model_dir=str(plain), requested="off")


def test_the_loader_gate_accepts_exactly_the_recorded_change(run_archive, version):
    stamp = _fork(_run(run_archive, "rb_parent", version), run_archive / "c")
    version.check_oracle_reveal("ki_moves", diagnostic_fork=stamp)
    version.check_oracle_reveal("moves:seen@0.95", diagnostic_fork=stamp)
    with pytest.raises(ModelVersionError, match="covers exactly the change"):
        version.check_oracle_reveal("item:seen@0.95", diagnostic_fork=stamp)
    with pytest.raises(ModelVersionError, match="covers exactly the change"):
        dataclasses.replace(version, oracle_reveal="species").check_oracle_reveal("ki_moves", diagnostic_fork=stamp)
    with pytest.raises(ModelVersionError, match="oracle_reveal mismatch"):
        version.check_oracle_reveal("ki_moves")


def test_the_stamp_lands_in_the_lineage_block_and_every_reader_shows_it(run_archive, version, capsys):
    from agents.training.lineage import diagnostic_fork_stamps
    from main import lineage as LN
    from main import sidecar_audit as SA
    from main.train.run_io import _run_lineage

    parent = _run(run_archive, "rb_parent", version)
    child_dir = run_archive / "rb_hib_moves_s3101"
    stamp = _fork(parent, child_dir)
    args = types.SimpleNamespace(exploiter=None, _diagnostic_fork=stamp)
    block = _run_lineage(args, str(child_dir), model_path=str(parent / "final_model.zip"), fork_step=0)
    assert block["diagnostic_fork"] == stamp and block["fork_parent"]["run_name"] == "rb_parent"
    # a same-run restart writes no block (the immutable one stands)
    assert _run_lineage(args, str(child_dir), model_path=str(child_dir / "checkpoints" / "c.zip"), fork_step=0) is None
    child = _run(run_archive, "rb_hib_moves_s3101", version, reveal="moves:seen@0.95", lineage=block)
    own = diagnostic_fork_stamps(str(child))
    assert len(own) == 1 and own[0]["self"] and own[0]["reason"] == stamp["reason"]
    # a plain fork of the stamped run is a diagnostic lineage too (its ancestor's stamp)
    from agents.training.lineage import build_lineage
    grand_block = build_lineage(model_path=str(child / "final_model.zip"), model_dir=str(run_archive / "rb_grand"))
    grand = _run(run_archive, "rb_grand", version, reveal="moves:seen@0.95", lineage=grand_block)
    anc = diagnostic_fork_stamps(str(grand))
    assert [(s["run"], s["self"]) for s in anc] == [("rb_hib_moves_s3101", False)]
    assert diagnostic_fork_stamps(str(parent)) == []
    # main.lineage
    text = LN.render(LN.read_run(str(child)))
    assert "DIAGNOSTIC FORK (this run)" in text and "off -> moves:seen@0.95" in text
    assert "DIAGNOSTIC FORK (ancestor rb_hib_moves_s3101)" in LN.render(LN.read_run(str(grand)))
    # main.sidecar_audit
    rec = SA.audit_run(str(child))
    assert rec["diagnostic_fork"] and rec["diagnostic_fork"][0]["self"]
    SA.print_run(rec, verbose=False)
    assert "diagnostic fork" in capsys.readouterr().out


def test_the_baselines_registry_refuses_and_flags_a_stamped_run(run_archive, version, monkeypatch):
    from agents.training import baselines as reg
    from main import baselines as MB

    parent = _run(run_archive, "rb_parent", version)
    stamp = _fork(parent, run_archive / "rb_hib_moves_s3101")
    _run(run_archive, "rb_hib_moves_s3101", version, reveal="moves:seen@0.95",
         lineage={"schema": "x", "role": "fork", "fork_parent": None, "diagnostic_fork": stamp})
    with pytest.raises(SystemExit, match="DIAGNOSTIC-FORK consent"):
        MB.build_entry("knockin", "rb_hib_moves_s3101/final_model.zip", reason="r", purpose="p", notes=None,
                       kind=None, set_on=None, registry=None)
    assert MB.diagnostic_fork_refusal(str(run_archive / "rb_parent")) is None
    fake = {"knockin": types.SimpleNamespace(run="rb_hib_moves_s3101"), "plain": types.SimpleNamespace(run="rb_parent")}
    monkeypatch.setattr(reg, "names", lambda registry=None: list(fake))
    monkeypatch.setattr(reg, "get", lambda name, registry=None: fake[name])
    found = MB.diagnostic_fork_findings(None)
    assert [(f.level, f.name) for f in found] == [("error", "knockin")]


def test_checkargs_prints_the_stamp_and_the_refusal(run_archive, version, capsys):
    from main.checkargs import _print_diagnostic_fork

    parent = _run(run_archive, "rb_parent", version)
    ns = types.SimpleNamespace(diagnostic_fork_consent="knock-in", model=str(parent / "final_model.zip"),
                               oracle_reveal="moves:seen@0.95", run_dir=str(run_archive / "rb_new"), run_name=None)
    assert _print_diagnostic_fork(ns) is None
    out = capsys.readouterr().out
    assert "diagnostic fork" in out and "off -> moves:seen@0.95" in out and "STAMPED" in out
    ns.oracle_reveal = "off"
    assert "meaningless" in _print_diagnostic_fork(ns)
    assert "REFUSED" in capsys.readouterr().out
    assert _print_diagnostic_fork(types.SimpleNamespace(diagnostic_fork_consent=None)) is None
