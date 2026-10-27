"""FR-2's read (the HOM's trace twin): the class with rule 8's band, the (battle, seat, k) pairing, the JOINT battle
bootstrap, the viewer and sha refusals and the resumable cache, on a synthetic extraction (no model)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

from main.human_data.x59s_read import FR2_CLASSES
from main.x50_r1 import read_fr2 as F

D_OBS = 3


def test_the_class_strings_are_the_x59s_reads():
    assert {F.fr2_class(1.0, 2.0), F.fr2_class(-1.0, 0.0), F.fr2_class(0.0, 1.0)} == set(FR2_CLASSES)


def test_the_class_reads_the_ci_against_005_with_the_1e9_band():
    assert F.fr2_class(0.05 + 2e-9, 0.3) == "ARTEFACT"
    assert F.fr2_class(0.05 + 5e-10, 0.3) == "UNCLEAR"          # within 1e-9 of the bar: UNCLEAR
    assert F.fr2_class(0.05, 0.3) == "UNCLEAR"
    assert F.fr2_class(-0.3, 0.05 - 2e-9) == "NO-ARTEFACT"
    assert F.fr2_class(-0.3, 0.05 - 5e-10) == "UNCLEAR"
    assert F.fr2_class(-0.3, 0.05) == "UNCLEAR"
    assert F.fr2_class(0.01, 0.2) == "UNCLEAR"                  # straddles the bar


# ------------------------------------------------------------------------------------------------ the fixture

def _rows(battle: int, seat: int, ks: List[int], mode: int, oa: List[int], nl: List[int]) -> Dict[str, np.ndarray]:
    n = len(ks)
    mask = np.zeros((n, 11), np.int8)
    for i, m in enumerate(nl):
        mask[i, :m] = 1
    return {"rows": np.full((n, D_OBS), float(battle), np.float32), "mask": mask,
            "label": np.zeros((n, 4), np.int64), "hp": np.zeros(n, bool), "forced": np.zeros(n, bool),
            "placeholder": np.zeros(n, bool), "turn": np.zeros(n, np.int32), "k": np.asarray(ks, np.int32),
            "battle": np.full(n, battle, np.int32), "seat": np.full(n, seat, np.int8),
            "mode": np.full(n, mode, np.int8), "rating_labelled": np.full(n, -1, np.int32),
            "rating_viewer": np.full(n, -1, np.int32), "own_action": np.asarray(oa, np.int8),
            "n_legal": np.asarray(nl, np.int8)}


def _cat(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _write_extraction(root: Path, viewer_seat_is_trainee: bool = False,
                      seat_stamp: Optional[str] = "opponent") -> Tuple[Path, Dict[str, np.ndarray]]:
    """One part of 4 battles (2 vs a sentinel, 2 vs a bot). The trainee is p1 (seat 0); the rows are the viewer's.
    The true team is read twice: mode 1 (real requests) and mode 3 (synthesised), with different labels."""
    tr = root / "traces"
    tr.mkdir()
    battles = []
    for b, opp in enumerate(("sentinel_0", "sentinel_1", "heuristic", "staller")):
        d = root / "eval_traces" / opp
        d.mkdir(parents=True, exist_ok=True)
        s = d / f"g{b}_summary.json"
        s.write_text(json.dumps({"meta": {"trace_source": {"trainee_side": "p1"}}}))
        battles.append(str(s))
    seat = 0 if viewer_seat_is_trainee else 1
    parts = []
    for b in range(4):
        # decisions 0..4: 0-2 usable in both modes, 3 has no true label, 4 has one legal action in mode 2;
        # battle 3's decision 5 exists only in the true modes (its completed seat was excluded at that key)
        parts.append(_rows(b, seat, [4, 2, 0, 1, 3], 2, [1, 1, 2, 1, 0], [1, 3, 3, 3, 3]))
        tk = [0, 1, 2, 3, 4] + ([5] if b == 3 else [])
        nl = [3, 3, 3, 3, 3] + ([3] if b == 3 else [])
        parts.append(_rows(b, seat, tk, 1, [0, 1, 2, -1, 0] + ([1] if b == 3 else []), nl))
        parts.append(_rows(b, seat, tk, 3, [1, 2, 3, -1, 1] + ([2] if b == 3 else []), nl))
    z = _cat(parts)
    np.savez_compressed(tr / "traces_0000.npz", **z)
    meta = {"part": "traces_0000", "battles": battles, "part_size": 500, "counts": {}, "complete": True, "commit": "x"}
    if seat_stamp is not None:                                   # R1's trainee-seat parts carry no stamp
        meta["seat"] = seat_stamp
    (tr / "traces_0000.json").write_text(json.dumps(meta))
    return tr, z


def _fake(shift: float) -> F.Forward:
    """A deterministic per-row NLL: the action index / 10 + the battle (the row's value) / 100 + ``shift``."""
    def fwd(rows: np.ndarray, mask: np.ndarray, act: np.ndarray) -> np.ndarray:
        return act.astype(np.float64) / 10 + rows[:, 0].astype(np.float64) / 100 + shift
    return fwd


def test_the_pairing_is_by_battle_seat_k_and_counts_every_drop(tmp_path):
    _, z = _write_extraction(tmp_path)
    for true_mode in (1, 3):
        ic, it, drops = F.pair_part(z, true_mode)
        keys_c = list(zip(z["battle"][ic], z["seat"][ic], z["k"][ic]))
        keys_t = list(zip(z["battle"][it], z["seat"][it], z["k"][it]))
        assert keys_c == keys_t                                 # aligned, despite mode 2's shuffled order
        assert sorted(set(keys_c)) == [(b, 1, k) for b in range(4) for k in range(3)]
        assert (z["mode"][ic] == 2).all() and (z["mode"][it] == true_mode).all()
        assert dict(drops) == {"no_own_action_label": 4, "fewer_than_2_legal": 4, "only_true": 1}
    with pytest.raises(F.Fr2Error):
        F.pair_part(z, 2)


def test_a_duplicated_decision_is_no_pairing(tmp_path):
    _, z = _write_extraction(tmp_path)
    dup = _cat([z, {k: v[:1] for k, v in z.items()}])
    with pytest.raises(F.Fr2Error, match="twice"):
        F.pair_part(dup, 1)


def test_the_read_on_a_synthetic_extraction_and_its_cache_resumes(tmp_path):
    tr, _ = _write_extraction(tmp_path)
    calls = {"n": 0}

    def counting(shift: float) -> F.Forward:
        inner = _fake(shift)

        def fwd(r, m, a):
            calls["n"] += 1
            return inner(r, m, a)
        return fwd

    fw = {"hom": counting(0.0), "pstar": counting(0.5)}
    tags = {"hom": "hom_aaa", "pstar": "pstar_bbb"}
    res = F.read(tr, fw, tags, tmp_path / "cache", "opponent")
    assert res["counts"]["decisions_paired"] == 12 and res["counts"]["battles"] == 4
    assert res["counts"]["decisions_paired_vs_sentinel"] == 6 and res["counts"]["decisions_paired_vs_bot"] == 6
    # completed labels (k 0..2) are 2, 1, 1; true labels 0, 1, 2: the gap is the label difference / 10, per model
    assert res["gap"]["hom"]["mean"] == pytest.approx((4 / 3 - 1) / 10)
    assert res["gap"]["pstar"]["mean"] == pytest.approx((4 / 3 - 1) / 10)
    assert res["D"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert res["counts"]["true_rows"] == "full (mode 1)"
    assert calls["n"] == 2
    # the synthesised true rows (mode 3, labels 1, 2, 3) are read only when asked for: a different gap
    syn = F.read(tr, fw, {"hom": "hom_s", "pstar": "pstar_s"}, tmp_path / "cache", "opponent", "own_true")
    assert syn["gap"]["hom"]["mean"] == pytest.approx((4 / 3 - 2) / 10)
    calls["n"] = 2
    again = F.read(tr, fw, tags, tmp_path / "cache", "opponent")
    assert calls["n"] == 2 and again["D"] == res["D"]           # every chunk from the cache
    with pytest.raises(F.Fr2Error, match="stale cache"):
        bad = tmp_path / "cache" / "hom_aaa_traces_0000_0000.npz"
        c = dict(np.load(bad))
        c["keys"] = c["keys"][::-1].copy()
        np.savez(bad, **c)
        F.read(tr, fw, tags, tmp_path / "cache", "opponent")


def test_rows_of_the_other_seat_refuse_the_read(tmp_path):
    tr, _ = _write_extraction(tmp_path, viewer_seat_is_trainee=True, seat_stamp=None)
    fw = {"hom": _fake(0.0), "pstar": _fake(0.0)}
    with pytest.raises(F.Fr2Error, match="not the opponent's seat"):
        F.read(tr, fw, {"hom": "h", "pstar": "p"}, tmp_path / "cache", "opponent")
    res = F.read(tr, fw, {"hom": "h", "pstar": "p"}, tmp_path / "cache", "trainee")
    assert res["counts"]["read_rows_by_viewer"] == {"trainee": 41}


def test_a_part_stamped_with_another_seat_refuses_the_read(tmp_path):
    tr, _ = _write_extraction(tmp_path, seat_stamp="trainee")
    with pytest.raises(F.Fr2Error, match="stamped seat"):
        F.read(tr, {"hom": _fake(0.0), "pstar": _fake(0.0)}, {"hom": "h", "pstar": "p"}, tmp_path / "cache",
               "opponent")


def test_the_bootstrap_draws_one_battle_sample_for_both_models_and_both_modes():
    # a huge per-battle level shared by all four populations, and D = 0.2 on every decision: D's interval is a
    # point ONLY if one draw of battles serves every population (independent draws would scatter it widely)
    rng = np.random.default_rng(0)
    battle = np.repeat(np.arange(40), 5)
    lvl = rng.normal(0.0, 10.0, 40)[battle]
    noise = rng.normal(0.0, 0.01, len(battle))
    nll = {"hom_c": lvl + 0.3 + noise, "hom_t": lvl + noise, "pstar_c": lvl + 0.1 + noise, "pstar_t": lvl + noise}
    res = F.twin(nll, battle)
    lo, hi = res["D"]["ci95"]
    assert res["D"]["mean"] == pytest.approx(0.2)
    assert lo == pytest.approx(0.2, abs=1e-9) and hi == pytest.approx(0.2, abs=1e-9)
    assert F.fr2_class(lo, hi) == "ARTEFACT"
    # each gap alone carries the battle level's spread only through the shared draw: also a point
    assert res["gap"]["hom"]["ci95"][1] - res["gap"]["hom"]["ci95"][0] < 1e-9
    # the shared draw is the registered seed's: the same input, the same interval
    assert F.twin(nll, battle)["D"]["ci95"] == res["D"]["ci95"]


def _zip(p: Path, payload: bytes) -> Path:
    p.write_bytes(payload)
    return p


def test_a_sha_mismatch_refuses_before_any_forward(tmp_path, monkeypatch):
    import hashlib

    hom = _zip(tmp_path / "hom.zip", b"hom")
    pst = _zip(tmp_path / "pstar.zip", b"pstar")
    sha = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (hom, pst)}
    monkeypatch.setattr(F, "PSTAR_SHA256", sha[pst])
    side = tmp_path / "hom_sidecar.json"
    side.write_text(json.dumps({"hom_sha256": sha[hom], "init_sha256": sha[pst]}))
    assert F.verify_inputs(hom, side, pst)["hom"] == sha[hom]
    side.write_text(json.dumps({"hom_sha256": "0" * 64, "init_sha256": sha[pst]}))
    with pytest.raises(F.Fr2Error, match="not the frozen HOM"):
        F.verify_inputs(hom, side, pst)
    side.write_text(json.dumps({"hom_sha256": sha[hom], "init_sha256": "1" * 64}))
    with pytest.raises(F.Fr2Error, match="init"):
        F.verify_inputs(hom, side, pst)
    side.write_text(json.dumps({"hom_sha256": sha[hom], "init_sha256": sha[pst]}))
    monkeypatch.setattr(F, "PSTAR_SHA256", "2" * 64)
    with pytest.raises(F.Fr2Error, match="P\\*'s"):
        F.verify_inputs(hom, side, pst)
    # the CLI refuses (exit 2) and writes nothing
    out = tmp_path / "out"
    assert F.main(["--traces", str(tmp_path), "--hom", str(hom), "--pstar", str(pst), "--out", str(out)]) == 2
    assert not out.exists()


def test_the_registered_p_star_sha_is_the_briefs():
    assert F.PSTAR_SHA256 == "814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed"
