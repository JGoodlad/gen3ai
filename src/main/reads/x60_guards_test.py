"""``main.reads.x60_guards`` (X60-T PREREG §23.4, §24 AM-X60T-3): the filled guards artefact the builder writes is the
one rule ``x60`` reads through to a verdict; G-F's cells are the ON x OFF finals in seed order; G-E's block names its
readers' targets by sha256; the re-decision spec built from the REGISTERED look-2 spec parses at main, imports every
cell look 2 played (never transitively), re-plays only the determinism unit, pins every final by its look-2 sha256 and
changes nothing else. Synthetic states, no game."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from main.reads import rules_x60 as X
from main.reads import x60_guards as XG
from main.reads.rules_x60_look2_test import look2_spec
from main.reads.rules_x60_test import OFF8, ON8, SEED, SHA, _decide, _gc, _gf_cell, _look2
from utils.hashing import sha256_file


def _state(tmp_path: Path, spec: Dict[str, Any], shas: Dict[str, str], sources: Dict[str, str]) -> Path:
    from main.reads import layout as LY

    st = tmp_path / "look2_state"
    st.mkdir()
    LY.spec_file(st).write_text(XG.render(spec))
    LY.models_file(st).write_text(json.dumps({lab: {"label": lab, "sha256": shas[lab], "source": sources[lab]}
                                              for lab in spec["models"]}))
    return st


def _synthetic_spec() -> Dict[str, Any]:
    return {"read": "x60t_synthetic", "models": {lab: {"seed": SEED[lab]} for lab in ON8 + OFF8},
            "decision": {"rule": "x60", "params": {"look": 2, "on": list(reversed(ON8)), "off": OFF8}}}


def test_g_f_cells_are_every_on_final_against_every_off_final_in_seed_order(tmp_path):
    spec = _synthetic_spec()
    st = _state(tmp_path, spec, SHA, {lab: f"/models/rb_{lab}/final_model.zip" for lab in ON8 + OFF8})
    s, m = XG.state_read(st)
    cells = XG.gf_cells(s, m)
    assert len(cells) == 64 and cells[0] == ["/models/rb_on_s3501/final_model.zip", "/models/rb_c0_s3501/final_model.zip"]
    assert cells[8][0].endswith("rb_on_s3502/final_model.zip") and cells[63][1].endswith("rb_c0_s3508/final_model.zip")
    assert XG.main(["gf-cells", "--state", str(st), "--out", str(tmp_path / "c.json")]) == 0
    assert json.loads((tmp_path / "c.json").read_text()) == cells


def test_a_state_with_an_unresolved_final_refuses(tmp_path):
    spec = _synthetic_spec()
    st = _state(tmp_path, spec, {**SHA, OFF8[6]: ""}, {lab: "/x" for lab in ON8 + OFF8})
    s, m = XG.state_read(st)
    with pytest.raises(XG.GuardsError, match="not resolved"):
        XG.gf_cells(s, m)
    assert XG.main(["gf-cells", "--state", str(st), "--out", str(tmp_path / "c.json")]) == 2


def test_g_f_block_names_a_cell_of_another_final_and_holds_the_registered_floor():
    spec = _synthetic_spec()
    ms = {lab: {"sha256": SHA[lab], "source": "/x"} for lab in ON8 + OFF8}
    cells = [_gf_cell(r, c, 0) for r in ON8 for c in OFF8]
    blk = XG.gf_block_from_cells(cells, spec, ms, family="f", request="q")
    assert blk["inconclusive"] == [] and blk["min_pairs"] == 500 and blk["rows"] == ON8 and blk["cols"] == OFF8
    assert blk["h_pp"] == [[50.0] * 8] * 8
    stray = _gf_cell(ON8[0], OFF8[0], 0, sha={**SHA, ON8[0]: "a" * 64})
    blk = XG.gf_block_from_cells(cells + [stray], spec, ms, family="f", request="q")
    assert blk["h_pp"] is None and any("outside" in w for w in blk["inconclusive"])
    assert [c["row"] for c in blk["cells"]][-1] is None


def _ge_models(tmp_path: Path, monkeypatch) -> Path:
    from main.x35 import ge as GE
    from main.x35 import kit as K

    models = tmp_path / "models"
    for arm in ("ge_x60t", "ge_c0"):
        d = models / GE.target_run(arm, 3501)
        d.mkdir(parents=True)
        (d / "final_model.zip").write_bytes(f"final of {arm}".encode())
    monkeypatch.setattr(K, "_final_step", lambda p: 300_078_185)
    return models


def _reader(models: Path, arm: str, wins: int) -> Any:
    from agents.training import best_response_gap as E
    from main.x35 import ge as GE

    rd = models / GE.reader_run(arm, 3501)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "final_model.zip").write_bytes(b"r")
    step = 300_078_185
    pts = [E.SeriesPoint(step=step + 2_000_000 * (i + 1), wins=wins // 3, games=100, post_fork=True) for i in range(3)]
    return E.ExploiterRun(
        run=rd.name, run_dir=str(rd), target_run=GE.target_run(arm, 3501),
        target_file=str(models / GE.target_run(arm, 3501) / "final_model.zip"), target_step=step,
        target_pins_own_teams=False, fork_step=step, num_timesteps=step + 5_000_064, budget=5_000_064,
        dose_rate=1.0e-8, lr_median=5.6e-5, teams=[], archetype="offense", membership="exact",
        regime={"exploiter_keep_bots": False, "cycle_games": 100}, series=pts, lineage_derived=False, dose_source="tb")


def test_the_r2_block_hashes_each_readers_target_and_the_rule_reads_it(tmp_path, monkeypatch):
    models = _ge_models(tmp_path, monkeypatch)
    on, off = _reader(models, "ge_x60t", 90), _reader(models, "ge_c0", 93)
    runs = {on.run: on, off.run: off}
    r2 = XG.r2_block(str(models), reader=lambda rd, ts: runs[Path(rd).name])
    want = {s: hashlib.sha256(f"final of {a}".encode()).hexdigest() for s, a in (("on", "ge_x60t"), ("off", "ge_c0"))}
    assert {s: t["sha256"] for s, t in r2["targets"].items()} == want
    assert r2["readers"]["on"]["run"] == "rb_x35_ge_x60t_s3501" and r2["readers"]["off"]["run"] == "rb_x35_ge_c0_s3501"
    # the rule reads G-E only when those targets are the read's own seed-3501 finals
    doc = {"seeds": SEED, "sides": {"on": ON8, "off": OFF8}, "finals": {**SHA, ON8[0]: want["on"], OFF8[0]: want["off"]}}
    assert X._guard_e(doc, r2).status == "PASS"
    assert X._guard_e(dict(doc, finals=SHA), r2).status == "INCONCLUSIVE"


def test_a_gc_input_that_is_not_a_pooled_record_refuses(tmp_path):
    f = tmp_path / "gc.json"
    f.write_text(json.dumps({"b1": [0, -1, 1], "b2": [0, -1, 1]}))
    with pytest.raises(XG.GuardsError, match="not a x35_gc_pooled_v1"):
        XG.gc_block(f)


def test_the_built_artefact_is_what_rule_x60_reads_through_to_adopt(tmp_path):
    spec = _synthetic_spec()
    st = _state(tmp_path, spec, SHA, {lab: "/x" for lab in ON8 + OFF8})
    s, m = XG.state_read(st)
    g_f = XG.gf_block_from_cells([_gf_cell(r, c, (i * 7 + j * 3) % 9 - 4) for i, r in enumerate(ON8)
                                  for j, c in enumerate(OFF8)], s, m, family="f", request="q")
    r2 = {"on": [30, 200], "off": [31, 200], "stat": "pooled",
          "readers": {"on": {"run": "rb_x35_ge_x60t_s3501"}, "off": {"run": "rb_x35_ge_c0_s3501"}},
          "targets": {"on": {"sha256": SHA[ON8[0]]}, "off": {"sha256": SHA[OFF8[0]]}}}
    doc = XG.artefact(st, s, m, g_f=g_f, r2=r2, g_c=_gc())
    assert doc["schema"] == X.GUARDS_SCHEMA and doc["status"] == X.PLAYED
    assert doc["look2_state"]["spec_sha256"] == sha256_file(st / "spec.json")
    art = tmp_path / "guards.json"
    art.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    cells, p, rspec = _look2(home_wins=95)
    p["guards"] = {"path": str(art), "sha256": sha256_file(art)}
    dec = _decide(cells, p, rspec)
    assert {n: g["status"] for n, g in dec["guards"].items()} == {"G-F": "PASS", "G-E": "PASS", "G-C": "PASS"}
    assert dec["decision"]["verdict"] == X.ADOPT


# ------------------------------------------------------------------------------------------------ the re-decision
def _look2_state_models(spec: Dict[str, Any]) -> Dict[str, Any]:
    from utils.paths import main_models_dir

    root = main_models_dir() or Path("/nonexistent/models")
    out = {}
    for lab, m in spec["models"].items():
        ref = m["ref"]
        run = ref["run"] if ref["kind"] == "final" else str(Path(ref["path"]).parent)
        out[lab] = {"label": lab, "sha256": ref.get("sha256") or hashlib.sha256(lab.encode()).hexdigest(),
                    "source": str(Path(root) / Path(run).relative_to("models") / "final_model.zip")}
    return out


def _redecide(guards_sha: str = "1" * 64) -> Dict[str, Any]:
    spec = look2_spec()
    return XG.redecide_spec(spec, _look2_state_models(spec), state="~/gen3ai_archive/x60t_read/look2",
                            spec_sha256="2" * 64, guards={"path": "g.json", "sha256": guards_sha},
                            read="x60t_look2_guards_test")


def test_the_redecision_spec_parses_at_main_and_passes_rule_x60s_look2_check():
    from main.reads import spec as SP

    got = SP.parse(_redecide())
    assert got.rule == "x60" and got.params["look"] == 2 and got.pin == look2_spec()["pin"]
    assert got.params["guards"] == {"path": "g.json", "sha256": "1" * 64}


def test_the_redecision_imports_every_cell_look2_played_and_replays_only_the_determinism_unit():
    old, new = look2_spec(), _redecide()
    det = old["decision"]["params"]["determinism_guard"]["group"]
    for lane in ("anchor", "s"):
        cover_old, cover_new = set(), set()
        for g, d in old[lane].items():
            cfg = json.dumps({k: v for k, v in d.items() if k not in ("models", "import", "family")}, sort_keys=True)
            cover_old |= {(cfg, m) for m in d["models"]}
        for g, d in new[lane].items():
            cfg = json.dumps({k: v for k, v in d.items() if k not in ("models", "import", "family")}, sort_keys=True)
            cover_new |= {(cfg, m) for m in d["models"]}
            if lane == "anchor" and g == det:
                assert "import" not in d and d == old[lane][g]                  # played, as registered
                continue
            imp = d["import"]
            assert imp["models"] == d["models"]                                   # nothing else plays
            src = old[lane].get(imp["group"])
            if imp["state"] == "~/gen3ai_archive/x60t_read/look2":
                assert imp["spec_sha256"] == "2" * 64
                # never transitive: only models look 2 itself PLAYED in that group
                assert not set(imp["models"]) & set(((src or {}).get("import") or {}).get("models") or ())
            else:
                assert imp == old[lane][g]["import"]                              # an earlier import, kept as is
        assert cover_new == cover_old, lane
    assert set(new["anchor"]) == set(old["anchor"]) | {"primary_l2"}
    assert new["anchor"]["primary_l2"]["models"] == [f"x60t_s{s}" for s in range(3504, 3509)] + \
        [f"c0_s{s}" for s in range(3506, 3509)]
    fams = [d["family"] for d in new["s"].values()]
    assert len(set(fams)) == len(fams) and not set(fams) & {d["family"] for d in old["s"].values()}


def test_the_redecision_pins_every_final_by_its_look2_sha256_and_changes_nothing_else():
    old, new = look2_spec(), _redecide()
    models = _look2_state_models(old)
    for lab, m in new["models"].items():
        assert m["ref"]["kind"] == "pinned" and m["ref"]["sha256"] == models[lab]["sha256"]     # FINDING 24
        assert {k: v for k, v in m.items() if k != "ref"} == {k: v for k, v in old["models"][lab].items() if k != "ref"}
    for lab in ("c0_s3506", "x60t_s3508"):
        assert old["models"][lab]["ref"]["kind"] == "final"
    po, pn = old["decision"]["params"], new["decision"]["params"]
    assert {k: v for k, v in pn.items() if k not in ("primary", "contrast", "guards")} == \
        {k: v for k, v in po.items() if k not in ("primary", "contrast", "guards")}
    assert pn["primary"]["groups"] == ["primary", "primary_l2", "primary_c0a", "primary_c0b"]
    assert pn["contrast"]["groups"] == ["primary", "primary_l2", "h_contrast", "d1"]
    assert {k: v for k, v in pn["contrast"].items() if k != "groups"} == \
        {k: v for k, v in po["contrast"].items() if k != "groups"}
    assert {k: v for k, v in new.items() if k not in ("read", "note", "models", "anchor", "s", "decision")} == \
        {k: v for k, v in old.items() if k not in ("read", "note", "models", "anchor", "s", "decision")}


def test_the_redecide_cli_refuses_a_spec_that_is_not_the_states(tmp_path):
    spec = look2_spec()
    models = _look2_state_models(spec)
    st = tmp_path / "look2"
    st.mkdir()
    (st / "spec.json").write_text(XG.render(spec))
    (st / "models.json").write_text(json.dumps(models))
    g = tmp_path / "guards.json"
    g.write_text("{}")
    other = tmp_path / "other.json"
    other.write_text(XG.render(dict(spec, note="edited")))
    base = ["redecide-spec", "--state", str(st), "--guards", str(g), "--read", "x60t_look2_guards_t"]
    assert XG.main(base + ["--spec", str(other), "--out", str(tmp_path / "o.json")]) == 2
    assert XG.main(base + ["--spec", str(st / "spec.json"), "--out", str(tmp_path / "o.json")]) == 0
    out = json.loads((tmp_path / "o.json").read_text())
    assert out["decision"]["params"]["guards"] == {"path": str(g), "sha256": sha256_file(g)}
    assert out["anchor"]["primary_l2"]["import"]["spec_sha256"] == sha256_file(st / "spec.json")
