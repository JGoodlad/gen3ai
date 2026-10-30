"""X59-S's ≥ 1500 follow-up (``followup.py``, PREREG Amendment 7): the three selections (the rating filter, the
seeded account-level mix, the S-S prefix), their equal counts and step budgets, the registration the trainer
enforces, the window's steps and budget, its stop rules, and the in-house read."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from main.human_data import followup as FU
from main.human_data import rowset as RS
from main.human_data import student as ST

TRAIN = (RS.SHARD, RS.SHARD, 7000)
N_FIX = 20000


def _rows(rng, k, rating, acct, act_seed):
    mask = np.zeros((k, 11), np.int8)
    mask[:, [0, 1, 6, 7, 8]] = 1
    ms = np.zeros((k, 11), np.int8)
    ms[:, [6, 7, 8]] = 1
    act = rng.choice([0, 6, 7, 8], k).astype(np.int8)
    return {"rows": rng.normal(size=(k, 4)).astype(np.float32), "mask": mask, "act": act,
            "hidden": np.zeros(k, bool), "cons": mask.copy(), "moveset": ms,
            "rid": rng.integers(0, 2 ** 62, k), "bkey": rng.integers(0, 500, k), "acct": acct.astype(np.int64),
            "seat": np.zeros(k, np.int8), "k": np.zeros(k, np.int16), "turn": np.zeros(k, np.int16),
            "forced": np.zeros(k, bool), "rating": rating.astype(np.int16), "quit_dist": np.full(k, -1, np.int16),
            "end": np.zeros(k, np.int8)}


def _rowset(d: Path, seed: int, hi_rows: int, extra: dict) -> None:
    """A parent rowset of :data:`TRAIN` shards whose rows hold exactly ``hi_rows`` actors rated >= 1500."""
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True)
    n = sum(TRAIN)
    hi = np.zeros(n, bool)
    hi[rng.choice(n, hi_rows, replace=False)] = True
    rating = np.where(hi, 1600, 1400)
    acct = rng.integers(0, 60, n)
    recs, at = [], 0
    for i, k in enumerate(TRAIN):
        recs.append(RS._write_shard(d, f"train_{i:04d}", _rows(rng, k, rating[at:at + k], acct[at:at + k], i)))
        at += k
    v = RS._write_shard(d, "val", _rows(rng, 300, np.full(300, 1400), rng.integers(1000, 1010, 300), 9))
    man = {"rowset": f"t{seed}", "train_shards": recs, "val": v, "train_rows": n, "val_rows": 300, "train_hidden": 0,
           "filter_spec": {}, "row_profile": {"rating_ge1500_rows": hi_rows}, **extra}
    (d / "rows_manifest.json").write_text(json.dumps(man))


@pytest.fixture()
def parents(tmp_path):
    r2, ss = tmp_path / "r2", tmp_path / "ss"
    _rowset(r2, 1, N_FIX, {})
    _rowset(ss, 2, 0, {"schema": "gen3_x59s_selfplay_rows_v1", "pstar_sha256": "PSTAR"})
    reg = {"pass": True, "rows_manifest_sha256": RS.S_sha(r2 / "rows_manifest.json"),
           "selfplay_manifest_sha256": RS.S_sha(ss / "rows_manifest.json")}
    return r2, ss, tmp_path / "fu", reg


def _train(rows: Path, fields=("rid", "acct", "rating")):
    man, _ = RS.load_manifest(rows)
    return {f: np.concatenate([RS.load_shard(rows, r, (f,))[f] for r in man["train_shards"]]) for f in fields}


def test_the_mix_takes_whole_accounts_in_the_seeded_order_and_trims_only_the_last():
    rng = np.random.default_rng(0)
    acct = rng.integers(0, 40, 5000)
    m, info = FU.mix_mask(acct, 1234, seed=FU.MIX_SEED)
    assert int(m.sum()) == 1234
    accts, counts = np.unique(acct, return_counts=True)
    order = accts[np.random.default_rng(FU.MIX_SEED).permutation(len(accts))]
    taken = order[:info["accounts_taken"]]
    whole, last = taken[:-1], taken[-1]
    for a in whole:                                    # every account before the last: all of its rows
        assert m[acct == a].all()
    assert not m[~np.isin(acct, taken)].any()          # nothing from an account after it
    lm = m[acct == last]
    assert 0 < lm.sum() <= len(lm) and lm[:lm.sum()].all() and not lm[lm.sum():].any()   # its FIRST rows
    m2, _ = FU.mix_mask(acct, 1234, seed=FU.MIX_SEED)
    assert np.array_equal(m, m2)                       # a pure function of the rows and the seed
    assert not np.array_equal(m, FU.mix_mask(acct, 1234, seed=FU.MIX_SEED + 1)[0])


def test_the_three_arms_have_equal_counts_and_budgets_and_register(parents):
    r2, ss, root, preg = parents
    out = {arm: FU.derive(arm, r2, ss, root, n=N_FIX) for arm in FU.ARMS}
    P = _train(r2)
    H = _train(FU.rows_dir(root, "h1500"))
    assert np.array_equal(H["rid"], P["rid"][P["rating"] >= 1500])            # r2's order, exactly the ≥ 1500 rows
    M = _train(FU.rows_dir(root, "hmix"))
    pm, _ = FU.mix_mask(P["acct"], N_FIX)
    assert np.array_equal(M["rid"], P["rid"][pm])
    S, Sp = _train(FU.rows_dir(root, "sscut")), _train(ss)
    assert np.array_equal(S["rid"], Sp["rid"][:N_FIX])                       # S-S's first N rows
    for arm, o in out.items():
        assert o["train_rows"] == N_FIX
        man, _ = RS.load_manifest(FU.rows_dir(root, arm))
        assert [s["rows"] for s in man["train_shards"]] == [RS.SHARD, N_FIX - RS.SHARD]   # re-cut, remainder last
        vpar = RS.load_manifest(ss if arm == "sscut" else r2)[0]["val"]["sha256"]
        assert man["val"]["sha256"] == vpar == RS.S_sha(FU.rows_dir(root, arm) / "val.npz")
        assert FU.verify(arm, r2, ss, root, n=N_FIX)["pass"]
    assert len({json.dumps(o["followup"]["step_budget"], sort_keys=True) for o in out.values()}) == 1
    reg = FU.register(root, r2, ss, {**preg}, n=N_FIX)
    assert reg["pass"], reg["problems"]
    assert reg["steps_all_students"] == 15 * reg["step_budget"][5901]
    with pytest.raises(FU.FollowupError, match="never rewritten"):
        FU.derive("h1500", r2, ss, root, n=N_FIX)


def test_verify_catches_a_selection_that_moved(parents):
    r2, ss, root, _ = parents
    FU.derive("hmix", r2, ss, root, n=N_FIX)
    p = FU.rows_dir(root, "hmix") / "rows_manifest.json"
    man = json.loads(p.read_text())
    man["followup"]["content_sha256"] = "0" * 64
    p.write_text(json.dumps(man))
    assert not FU.verify("hmix", r2, ss, root, n=N_FIX)["pass"]


def test_a_rating_count_other_than_the_registered_n_stops(parents):
    r2, ss, root, _ = parents
    with pytest.raises(FU.FollowupError, match="STOP"):
        FU.derive("h1500", r2, ss, root, n=N_FIX + 1)


def test_register_refuses_parents_other_than_amendment_2b(parents):
    r2, ss, root, preg = parents
    for arm in FU.ARMS:
        FU.derive(arm, r2, ss, root, n=N_FIX)
    bad = FU.register(root, r2, ss, {**preg, "selfplay_manifest_sha256": "x"}, n=N_FIX)
    assert not bad["pass"] and any("S-S" in p for p in bad["problems"])


def test_the_trainer_takes_each_followup_arm_only_on_its_own_registered_rows():
    reg = {"pass": True, "rows_1500_manifest_sha256": "A", "rows_mix_manifest_sha256": "B",
           "selfplay_cut_manifest_sha256": "C"}
    ST.check_registration(reg, "human_1500", "A", None)
    ST.check_registration(reg, "human_mix", "B", None)
    ST.check_registration(reg, "selfplay_cut", "C", None)
    with pytest.raises(ST.StudentError, match="VOID"):
        ST.check_registration(reg, "human_mix", "A", None)           # S-H≥1500's rows under the mix's name
    with pytest.raises(ST.StudentError, match="VOID"):
        ST.check_registration(reg, "human", "A", None)               # a primary arm on the follow-up's registration
    z = {"act": np.array([6, -1], np.int8), "hidden": np.array([False, True]),
         "mask": np.array([[1] * 11] * 2, np.int8), "moveset": np.array([[0] * 6 + [1] * 4 + [0]] * 2, np.int8),
         "cons": np.array([[1] * 11] * 2, np.int8)}
    for arm in ST.FOLLOWUP_ARMS:                                     # each trains on its rows' own labels
        assert np.array_equal(ST.targets(arm, z, None), ST.targets("human", z, None))


def test_train_step_continues_only_on_exit_3_and_stops_the_window_otherwise(tmp_path):
    marker = tmp_path / "NO_GO"
    calls = []

    def runner(codes):
        def run(argv):
            calls.append(list(argv))
            return codes.pop(0)
        return run

    assert FU.train_step("hmix", 5902, tmp_path, tmp_path / "reg.json", str(marker), run=runner([3, 3, 0])) == 0
    assert len(calls) == 3 and not marker.exists()
    a = calls[0]
    assert a[a.index("--arm") + 1] == "human_mix" and a[a.index("--device") + 1] == "cuda"
    assert a[a.index("--out") + 1].endswith("students/s_hmix_5902")
    for codes in ([1], [3, 86], [3] * FU.MAX_CHUNKS):
        calls.clear()
        assert FU.train_step("h1500", 5901, tmp_path, tmp_path / "r", str(marker), run=runner(list(codes))) == 3
        assert marker.exists() and len(calls) == min(len(codes), FU.MAX_CHUNKS)
        marker.unlink()

    def boom(argv):
        raise RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError):
        FU.train_step("sscut", 5903, tmp_path, tmp_path / "r", str(marker), run=boom)
    assert "CUDA out of memory" in marker.read_text()


def test_the_preflight_and_the_gate_write_the_marker_on_a_failure(parents, tmp_path):
    r2, ss, root, preg = parents
    for arm in FU.ARMS:
        FU.derive(arm, r2, ss, root, n=N_FIX)
    regp = tmp_path / "reg.json"
    regp.write_text(json.dumps(FU.register(root, r2, ss, preg, n=N_FIX)))
    marker = tmp_path / "NO_GO"
    assert FU.preflight(root, regp, "tok", str(marker), token_valid=lambda f: (True, "ok")) == 0
    assert not marker.exists() and (root / "students").is_dir()
    assert FU.preflight(root, regp, "tok", str(marker), token_valid=lambda f: (False, "expired")) == 3
    assert "expired" in marker.read_text()
    marker.unlink()
    reg = json.loads(regp.read_text())
    reg["rows_mix_manifest_sha256"] = "other"
    regp.write_text(json.dumps(reg))
    assert FU.preflight(root, regp, "tok", str(marker), token_valid=lambda f: (True, "ok")) == 3
    assert "hmix" in marker.read_text()
    marker.unlink()
    pf = tmp_path / "pf.json"
    assert FU.gate(pf, str(marker)) == 3 and marker.exists()          # no result: HOLD
    marker.unlink()
    pf.write_text(json.dumps({"pass": False, "checks": {"export": {"pass": False}, "init": {"pass": True}}}))
    assert FU.gate(pf, str(marker)) == 3 and "export" in marker.read_text()
    marker.unlink()
    pf.write_text(json.dumps({"pass": True, "checks": {}}))
    assert FU.gate(pf, str(marker)) == 0 and not marker.exists()


def test_the_window_item_is_the_registered_order_with_the_computed_budget():
    reg = {"steps_all_students": 15 * 1584}
    it = FU.window_item(worktree="/wt", require_commit="abc", python="/py", root="/arch/fu", job_dir="/job",
                        token_file="/job/tok", reg=reg)
    assert it["budget_min"] == 140.0                    # 23,760 x 225.6 ms = 89.34 min; x 1.5 + 5 = 139.007, up
    assert it["on_fail"] == "hold" and it["preempt_marker"] == "/job/NO_GO_X59FU"
    names = [s["name"] for s in it["steps"]]
    assert names[:3] == ["preflight", "cuda_preflight", "preflight_gate"] and names[-1] == "check"
    assert names[3:-1] == [f"student_{FU.ARMS[a][1]}_{s}" for a, s in FU.jobs()]
    assert names[3:6] == ["student_s_h1500_5901", "student_s_hmix_5901", "student_s_scut_5901"]
    gpu = {s["name"]: s for s in it["steps"] if s["gpu"]}
    assert set(gpu) == {"cuda_preflight", *names[3:-1]}
    assert all(gpu[n]["mem_gb"] == 10 for n in names[3:-1]) and gpu["cuda_preflight"]["mem_gb"] == 16
    for s in it["steps"]:                               # arm_chain.md §2: nothing wraps itself
        assert not any("gpu_lock" in x or "mem_cap" in x for x in s["argv"])
    cp = gpu["cuda_preflight"]["argv"]
    assert cp[cp.index("--arm") + 1] == "human_1500" and cp[cp.index("--rows") + 1] == "/arch/fu/h1500/rows"
    assert all(s["argv"][s["argv"].index("--marker") + 1] == "/job/NO_GO_X59FU"
               for s in it["steps"] if "--marker" in s["argv"])


def test_the_cuda_preflight_cli_runs_the_named_arm(monkeypatch, tmp_path):
    from main.human_data import cuda_preflight as CP

    seen = {}

    def fake(rows, scratch, device, reg, **kw):
        seen.update(kw)
        return {"pass": True}
    monkeypatch.setattr(CP, "preflight", fake)
    (tmp_path / "reg.json").write_text("{}")
    assert CP.main(["--rows", str(tmp_path), "--registration", str(tmp_path / "reg.json"), "--scratch",
                    str(tmp_path / "s"), "--out", str(tmp_path / "o.json"), "--arm", "human_1500",
                    "--students", "15"]) == 0
    assert seen == {"arm": "human_1500", "students": 15}


def test_the_inhouse_read_classes_each_pairing_on_one_engine(tmp_path, monkeypatch):
    from main.human_data import x59s_ops as XO

    side = {f"{FU.ARMS[a][1]}_{s}": {"zip_path": f"/z/{a}_{s}.zip"} for a, s in FU.jobs()}
    cells = FU.inhouse_cells({"students": side})
    assert len(cells) == 2 * 5 * 3 and all(c["player"].startswith("/z/h1500_") for c in cells)
    argv = XO.inhouse_argv(cells[0], tmp_path / "ledger", "/py", label=FU.INHOUSE_LABEL)
    assert argv[argv.index("--label") + 1] == "x59s_fu1500_inhouse" and argv[argv.index("--pairs") + 1] == "200"
    assert XO.inhouse_argv(cells[0], tmp_path, "/py")[XO.inhouse_argv(cells[0], tmp_path, "/py").index("--label") + 1] \
        == "x59s_inhouse"
    score = {"rating": 0.62, "data": 0.50}
    for c in cells:
        (tmp_path / f"{c['name']}.log").write_text(json.dumps({"pairs": 200, "score": score[c["contrast"]]
                                                               + 0.002 * (c["seed"] - 5903)}) + "\n")
    monkeypatch.setattr(XO, "ledger_engines", lambda ledger: ["env=a/1"])
    res = FU.inhouse_read(tmp_path, {"students": side})
    assert res["classes"]["rating"]["class"] == "HIGHER-RATED-BETTER"
    assert res["classes"]["data"]["class"] == "NO-DIFFERENCE-AT-4PP"
    monkeypatch.setattr(XO, "ledger_engines", lambda ledger: ["env=a/1", "env=b/2"])
    assert FU.inhouse_read(tmp_path, {"students": side})["classes"]["rating"]["class"] is None
