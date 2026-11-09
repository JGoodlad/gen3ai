"""B1's label bank: π′, the keys, outcome-blindness, and — on the frame probe's one real miss (S′-LIVE
``pstar_s__forum72__340__g10``, our second decision inside request update 24 after an ``[Unavailable choice]``) — the
ordered rebuild and the identity guard (PREREG §9: "the identity guard on a constructed mis-ordered log")."""
from __future__ import annotations

import builtins
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from main.x65 import bank as B

FIXTURE = Path(__file__).parent / "testdata" / "misordered_cell"
#: an S′-HONEST searched game (``pstar_s__forum72__756__g11``, its chunks dropped): 8 searched decisions, 5 exclusions
#: (one of them a re-score sample), 2 searched re-score samples
HONEST = Path(__file__).parent / "testdata" / "honest_cell"
#: the probe's miss: our decision n 23 (turn 23), the log handed to the search held 46 commands
MISS_N, MISS_AT = 23, 46


def _recs():
    return [json.loads(x) for x in (FIXTURE / "search_decisions.jsonl").read_text().splitlines()]


# ------------------------------------------------------------------------------------------------ pure
def test_pi_prime_is_the_pikl_softmax_over_the_legal_actions():
    q = {"1": 0.40, "3": 0.50, "7": 0.45}
    lp = {"1": -0.5, "3": -2.0, "7": -1.2}
    p = B.pi_prime(q, lp, lam=0.015)
    s = np.array([lp[k] + q[k] / 0.015 for k in ("1", "3", "7")])
    e = np.exp(s - s.max())
    assert np.allclose(p[[1, 3, 7]], e / e.sum(), rtol=0, atol=1e-15)
    assert p.sum() == pytest.approx(1.0, abs=1e-15)
    assert (p[[0, 2, 4, 5, 6, 8, 9, 10]] == 0).all()
    assert int(np.argmax(p)) == 3                     # Q / λ dominates: the mode is S′'s pick
    with pytest.raises(B.BankError):
        B.pi_prime({"1": 0.1}, {"2": -0.1})


def test_inner_validation_is_a_function_of_the_deal_alone():
    import hashlib

    deals = [f"{i:020x}" for i in range(4000)]
    got = [B.inner_val(d) for d in deals]
    assert got == [hashlib.sha256(d.encode()).digest()[0] % 10 == 0 for d in deals]
    assert 0.08 < np.mean(got) < 0.12                 # 26 of 256 first-byte values


def test_the_builder_refuses_every_result_bearing_file(tmp_path):
    for name in ("games.jsonl", "summary.json", "forced_ends.jsonl", "peer_acceptor_report.json"):
        p = tmp_path / name
        p.write_text("{}")
        with pytest.raises(B.OutcomeBlindError):
            B.read_json(p)
        with pytest.raises(B.OutcomeBlindError):
            B.read_jsonl(p)


def test_counted_units_keep_only_the_counted_units_structure(tmp_path):
    rec = {"rule": "searched_play", "void": None, "inputs": {"raw": "pstar", "searched": "pstar_s", "deals": [
        {"deal": "forum72:300:0", "game": 0, "panel": "forum72", "unit_seed": 300,
         "raw": {"status": "counted", "unit": "pstar__forum72__300__g00"},
         "searched": {"status": "exhausted", "unit": "pstar_s__forum72__300__g00"}}]},
           "units": [{"games": [["x", 1, "t", "win", 9]]}]}
    (tmp_path / "record.json").write_text(json.dumps(rec))
    assert B.counted_units(tmp_path / "record.json") == [
        {"label": "pstar", "unit": "pstar__forum72__300__g00", "panel": "forum72", "unit_seed": 300, "game": 0}]


def test_the_capture_reader_never_keeps_the_chunks_and_stops_at_the_end(tmp_path):
    cell = tmp_path / "c"
    (cell / "capture").mkdir(parents=True)
    start = {"formatid": "gen3ou", "seed": [1, 2, 3, 4], "p1": {}, "p2": {}, "core_obs": {"sides": ["p1"]}}
    (cell / "capture" / "b.json").write_text(json.dumps({"seed": [1, 2, 3, 4], "commands": [
        "START " + json.dumps(start), "CHOOSE p1 move a", "CHOOSE p2 move b", "FORCELOSE p2", "END"],
        "chunks": [["p1", "|win|someone"]]}))
    st, cmds = B.read_capture(cell)
    assert cmds == ["CHOOSE p1 move a", "CHOOSE p2 move b"] and B.our_side(st) == "p1"


# ------------------------------------------------------------------------------------------------ the core
@pytest.mark.sim
@pytest.mark.integration
def test_the_naive_prefix_misses_the_probe_frame():
    """The hazard B1 removes (the frame probe's one miss): a naive ``cmds[:at]`` holds the opponent's update-24
    choice instead of our own refused first choice, and its replayed row is not the live frame."""
    from main.anchors.input_log import frame_sha256
    from main.value_search import worlds as W

    start, cmds = B.read_capture(FIXTURE)
    rec = _recs()[MISS_N]
    assert int(rec["at"]) == MISS_AT and rec["searched"] and cmds[MISS_AT - 1].startswith("CHOOSE p1 ")
    truth = W.prepare_root(B.naive_cut(start, cmds, MISS_AT), MISS_AT, "p2", rec["key"])
    assert frame_sha256(truth.ref[-1]) != rec["frame_sha256"]


@pytest.mark.sim
@pytest.mark.integration
def test_the_ordered_rebuild_matches_every_recorded_frame_including_the_miss():
    rows, st = B.game_rows(FIXTURE, "pstar_s")
    recs = _recs()
    assert st["drop"] == {} and st["labels_kept"] == st["labels_recorded"] == sum(1 for r in recs if r["searched"])
    assert st["frame_ok"] == st["frame_checked"] == sum(1 for r in recs if r.get("frame_sha256"))
    assert st["log_sha_bad"] == st["at_bad"] == st["turn_bad"] == st["xcheck_bad"] == 0
    miss = [r for r in rows if r["n"] == MISS_N]
    assert len(miss) == 1 and miss[0]["label"] and miss[0]["at"] == MISS_AT and miss[0]["update"] == 24
    assert [r["n"] for r in rows] == list(range(len(recs)))
    assert all(r["turn"] == int(rec["turn"]) for r, rec in zip(rows, recs))


@pytest.mark.sim
@pytest.mark.integration
def test_the_identity_guard_drops_a_misordered_log(monkeypatch):
    """A mis-ordered log (the naive prefix, substituted for the ordered cut) gives the miss a wrong row: the guard
    DROPS that label row and counts it; it never reaches the bank."""
    real = B.honest_cut

    def misordered(start, cmds, upd, side, p, u):
        return B.naive_cut(start, cmds, len(real(start, cmds, upd, side, p, u)["cmds"]))

    monkeypatch.setattr(B, "honest_cut", misordered)
    rows, st = B.game_rows(FIXTURE, "pstar_s")
    assert not any(r["n"] == MISS_N for r in rows)
    assert st["labels_kept"] == st["labels_recorded"] - 1
    lost = {k: v for k, v in st["drop"].items() if k.startswith("label:")}
    # the naive prefix opens decision 22, not 23: rebuild_row's decision-index check removes it (REVIEW R-m1)
    assert lost == {"label:rebuild:BankError": 1}, st["drop"]


@pytest.mark.sim
@pytest.mark.integration
def test_a_wrong_rebuilt_row_reaches_the_frame_guard_and_is_dropped(monkeypatch):
    """The identity guard through the REAL path (REVIEW R-m1): the row rebuilt for the miss differs by one byte from
    the live frame, so its digest differs and the label row is dropped as ``frame_sha256``."""
    import base64

    real = B.rebuild_row

    def one_byte_off(log, side, k, key):
        r = real(log, side, k, key)
        if k == MISS_N:
            raw = bytearray(base64.b64decode(r["b64"]))
            raw[100] ^= 0x01
            r["b64"] = base64.b64encode(bytes(raw)).decode()
        return r

    monkeypatch.setattr(B, "rebuild_row", one_byte_off)
    rows, st = B.game_rows(FIXTURE, "pstar_s")
    assert not any(r["n"] == MISS_N for r in rows)
    assert {k: v for k, v in st["drop"].items() if k.startswith("label:")} == {"label:frame_sha256": 1}


def _edited_cell(tmp_path, edit):
    """A copy of the fixture cell whose decision records ``edit`` changed in place (``edit(recs)``)."""
    cell = tmp_path / "cell"
    shutil.copytree(FIXTURE, cell)
    recs = _recs()
    edit(recs)
    (cell / "search_decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    return cell


def _searched(recs, nth=0):
    return [r for r in recs if r["searched"]][nth]


@pytest.mark.sim
@pytest.mark.integration
@pytest.mark.parametrize("what,reason", [("q_key", "legal_mask"), ("played", "played"), ("key", "key"),
                                         ("deal", "key"), ("turn", "turn")])
def test_each_alignment_guard_drops_its_label_row(tmp_path, what, reason):
    """REVIEW R-M3 / R-m10: a label row whose legal set is not ``q``'s keys, whose recorded ``played`` is not the
    rebuilt command's index, whose key / deal or turn is not the rebuilt one, is dropped and counted by name."""
    target = {}

    def edit(recs):
        r = _searched(recs, 5)
        target["n"] = int(r["n"])
        legal = sorted(int(a) for a in r["q"])
        if what == "q_key":
            extra = next(a for a in range(11) if a not in legal)
            r["q"][str(extra)] = 0.5
            r["log_pi"][str(extra)] = -9.0
        elif what == "played":
            r["played"] = next(a for a in legal if a != int(r["played"]))
        elif what == "key":
            r["key"] = r["key"] + "x"
        elif what == "deal":
            r["deal"] = "0" * 20
        else:
            r["turn"] = int(r["turn"]) + 1

    rows, st = B.game_rows(_edited_cell(tmp_path, edit), "pstar_s")
    assert st["drop"].get("label:" + reason) == 1, st["drop"]
    assert st["labels_kept"] == st["labels_recorded"] - 1
    assert not any(r["n"] == target["n"] for r in rows)


def test_the_pstar_check_matches_ties_and_mismatches():
    """REVIEW R-m10: P*'s greedy action is checked against the recorded one; two logits within TIE_EPS are excluded
    (never passed by chance), a clear mismatch is 0."""
    mask = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    rows = [{"mask": mask, "arm": 0, "played": 2, "pi_argmax": -1},
            {"mask": mask, "arm": 1, "played": 1, "pi_argmax": 3},
            {"mask": mask, "arm": 0, "played": 1, "pi_argmax": -1},
            {"mask": mask, "arm": 0, "played": 2, "pi_argmax": -1},
            {"mask": [0, 0, 1] + [0] * 8, "arm": 0, "played": 2, "pi_argmax": -1}]
    lg = np.full((5, 11), -50.0)
    lg[0, [1, 2, 3]] = [0.1, 0.9, 0.2]                      # P* plays 2 = played: 1
    lg[1, [1, 2, 3]] = [0.1, 0.2, 0.9]                      # searched arm: P* plays 3 = pi_argmax: 1
    lg[2, [1, 2, 3]] = [0.1, 0.9, 0.2]                      # P* plays 2, the game played 1: 0
    lg[3, [1, 2, 3]] = [0.1, 0.9, 0.9 - B.TIE_EPS / 2]      # a near-tie: 2, excluded
    lg[4, 2] = 0.0                                          # one legal action, the one played: 1
    assert B.pstar_check(rows, lg) == [1, 1, 0, 2, 1]


def test_more_than_half_a_percent_of_a_panel_s_label_rows_dropped_voids_the_bank():
    """REVIEW R-M3: PREREG §10's first VOID, exactly at its bar (5 of 1,000 is not more than 0.5 %)."""
    def shards(lost):
        return [{"panel": "forum72", "label": "pstar_s", "rows": 2000, "labels": 1000 - lost,
                 "pstar_mismatch": {"label": 0, "anchor": 0}, "pstar_ties": 0,
                 "games": [{"labels_recorded": 1000, "drop": {"label:frame_sha256": lost, "frame_sha256": lost}}]},
                {"panel": "p72", "label": "pstar_s", "rows": 2000, "labels": 1000, "games": [{"labels_recorded": 1000}],
                 "pstar_mismatch": {"label": 0, "anchor": 0}, "pstar_ties": 0}]
    assert B.summarize(shards(5))["void"] is None
    got = B.summarize(shards(6))
    assert got["void"] and "forum72" in got["void"][0] and got["per_panel"]["forum72"]["label_drop_frac"] == 0.006
    assert got["per_panel"]["p72"]["label_drop_frac"] == 0


@pytest.mark.sim
@pytest.mark.integration
def test_a_row_p_star_disagrees_with_is_dropped_from_the_shard(tmp_path, monkeypatch):
    """REVIEW R-m10: a row whose P* greedy action is not the recorded one never reaches the shard; it is counted
    (a label row's loss counts toward the VOID share)."""
    state = tmp_path / "state"
    cells = state / "anchor" / "u" / "pstar_s" / "cells"
    cells.mkdir(parents=True)
    shutil.copytree(FIXTURE, cells / "pstar_s__forum72__340__g10")
    recs = _recs()
    bad_n = int(_searched(recs, 3)["n"])

    def fake_logits(model_zip, rows, threads):
        lg = np.full((len(rows), 11), -50.0)
        for i, r in enumerate(rows):
            want = r["played"] if r["arm"] == 0 else r["pi_argmax"]
            legal = [a for a, m in enumerate(r["mask"]) if m]
            pick = want if r["n"] != bad_n else next(a for a in legal if a != want)
            lg[i, pick] = 1.0
        return lg

    monkeypatch.setattr(B, "pstar_logits", fake_logits)
    out = tmp_path / "bank"
    out.mkdir()
    rec = B.build_shard(state, out, "pstar_s", "forum72", 340,
                        [{"unit": "pstar_s__forum72__340__g10", "game": 10}], tmp_path / "pstar.zip")
    assert rec["pstar_mismatch"] == {"label": 1, "anchor": 0} and rec["rows"] == len(recs) - 1
    z = np.load(out / f"{rec['name']}.npz")
    assert bad_n not in z["n"].tolist() and (z["pstar_check"] == 1).all()
    summ = B.summarize([rec])["per_panel"]["forum72"]
    assert summ["label_drops"]["pstar_check"] == 1 and summ["labels_kept"] == summ["labels_recorded"] - 1


@pytest.mark.sim
@pytest.mark.integration
def test_a_corrupted_frame_digest_is_dropped_and_counted(tmp_path):
    cell = tmp_path / "cell"
    shutil.copytree(FIXTURE, cell)
    recs = _recs()
    recs[MISS_N - 1]["frame_sha256"] = "0" * 64
    (cell / "search_decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    rows, st = B.game_rows(cell, "pstar_s")
    assert st["drop"].get("label:frame_sha256") == 1
    assert not any(r["n"] == MISS_N - 1 for r in rows)
    del recs[MISS_N - 2]["frame_sha256"]                      # a label row with no digest is never kept either
    (cell / "search_decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    rows, st = B.game_rows(cell, "pstar_s")
    assert st["drop"].get("label:no_frame_sha256") == 1
    assert not any(r["n"] == MISS_N - 2 for r in rows)


@pytest.mark.sim
@pytest.mark.integration
def test_the_bank_builds_checks_and_never_opens_a_result_file(tmp_path, monkeypatch):
    state = tmp_path / "state"
    cells = state / "anchor" / "u" / "pstar_s" / "cells"
    cells.mkdir(parents=True)
    shutil.copytree(FIXTURE, cells / "pstar_s__forum72__340__g10")
    (cells / "pstar_s__forum72__340__g10" / "games.jsonl").write_text("THE RESULT\n")
    (cells / "pstar_s__forum72__340__g10" / "summary.json").write_text("{}")
    rec = {"rule": "searched_play", "void": None, "inputs": {
        "raw": "pstar", "searched": "pstar_s", "expected_search": "r0s4_rolls@10", "engine_registered": B.E4, "deals": [
        {"deal": "forum72:340:10", "game": 10, "panel": "forum72", "unit_seed": 340,
         "raw": {"status": "exhausted", "unit": "pstar__forum72__340__g10"},
         "searched": {"status": "counted", "unit": "pstar_s__forum72__340__g10"}}]}}
    (state / "record.json").write_text(json.dumps(rec))
    opened = []
    real_open = builtins.open

    def spy(file, *a, **k):
        opened.append(Path(str(file)).name)
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    out = tmp_path / "bank"
    from main.anchors import server

    monkeypatch.setattr(server, "engine_stamp", lambda: "port=x/1;env=y/1")
    with pytest.raises(B.BankError, match="not the sprime_live source's"):   # a bank is built on its games' engine
        B.build(state, tmp_path / "other", workers=1, model=None, allow_dirty=True)
    monkeypatch.setattr(server, "engine_stamp", lambda: B.E4)
    with pytest.raises(B.BankError, match="not the sprime_honest source's"):  # S′-HONEST played on E6, not E4
        B.build(state, tmp_path / "other", workers=1, model=None, allow_dirty=True, source="sprime_honest")
    # the engine check itself is the line above; here it is E4 whatever this worktree's env stamp says (an
    # untracked env/Cargo.lock is part of that stamp: REVIEW R-m12)
    monkeypatch.setattr(server, "engine_stamp", lambda: B.E4)
    man = B.build(state, out, workers=1, model=None, allow_dirty=True)
    monkeypatch.setattr(builtins, "open", real_open)
    assert not set(opened) & B.FORBIDDEN
    assert man["void"] is None and man["label_rows"] == sum(1 for r in _recs() if r["searched"])
    pp = man["per_panel"]["forum72"]
    assert pp["labels_recorded"] == pp["labels_kept"] and pp["label_drop_frac"] == 0
    sha = (out / "bank_manifest.sha256").read_text().strip()
    man["dirty"] = False
    (out / "bank_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    sha = B.sha256_file(out / "bank_manifest.json")
    assert B.check(out, sha)["label_rows"] == man["label_rows"]
    assert B.check(out, sha, source="sprime_live")["source"] == man["source"] == "sprime_live"
    with pytest.raises(B.BankError, match="not 'sprime_honest'"):    # an honest trainer never reads a leaky bank
        B.check(out, sha, source="sprime_honest")
    with pytest.raises(B.BankError):
        B.check(out, "f" * 64)
    z = B.load_rows(out, ["forum72"])
    assert z["obs"].shape == (len(_recs()), 2845) and int(z["label"].sum()) == man["label_rows"]
    lab = z["label"]
    assert np.allclose(z["pi_prime"][lab].sum(1), 1.0) and (z["pi_prime"][~lab] == 0).all()
    assert ((z["pi_prime"] > 0) <= (z["mask"] == 1)).all()


# ------------------------------------------------------------------------------------------------ the honest source
def _honest_recs():
    return [json.loads(x) for x in (HONEST / "search_decisions.jsonl").read_text().splitlines()]


def test_an_honest_view_holds_bank_fields_only_never_a_hidden_one():
    recs = _honest_recs()
    assert any("species_hit" in r for r in recs)              # the live records carry the truth-scored diagnostic
    for r in recs:
        v = B.label_view(r, "sprime_honest")
        assert set(v) <= B.RECORD_FIELDS and not set(v) & B.HIDDEN_INFO_FIELDS


def test_a_hidden_information_field_can_never_become_a_bank_field(monkeypatch):
    """The refusal itself: a field computed against the true team, listed as a bank field, stops every view."""
    monkeypatch.setattr(B, "RECORD_FIELDS", B.RECORD_FIELDS | {"species_hit"})
    for source in B.SOURCES:
        with pytest.raises(B.HiddenInfoError, match="hidden-information field"):
            B.label_view(_honest_recs()[12], source)


def test_an_honest_bank_refuses_an_unclassified_field():
    r = dict(_honest_recs()[12], true_species=["Tyranitar"])
    with pytest.raises(B.HiddenInfoError, match="unclassified"):
        B.label_view(r, "sprime_honest")
    assert "true_species" not in B.label_view(r, "sprime_live")     # X65's leaky source: projected, never banked


@pytest.mark.parametrize("edit", [
    lambda r: r.update(search="r0s4_rolls@10"),
    lambda r: r.update(leak={"attribution_miss": 3, "partial": ["spread:seen"], "residual": ["counters:seen"]}),
    lambda r: r["leak"].update(builder="leaky"),
    lambda r: r["leak"].update(partial=["item:seen"]),
    lambda r: r["leak"].update(residual=["counters:seen"]),
    lambda r: r["config"].update(reveal="on"),
    lambda r: r["config"].update(pikl_lambda=0.03),
], ids=["preset", "leaky_block", "builder", "partial", "residual", "reveal", "lambda"])
def test_an_honest_bank_refuses_a_record_not_from_a_leak_honest_search(edit):
    r = json.loads(json.dumps(_honest_recs()[12]))
    assert r["searched"]
    B.label_view(r, "sprime_honest")
    edit(r)
    with pytest.raises(B.HiddenInfoError):
        B.label_view(r, "sprime_honest")


def test_counted_units_refuse_a_record_of_another_source(tmp_path):
    ins = {"raw": "pstar", "searched": "pstar_s", "deals": [], "expected_search": "r0s4_rolls@10",
           "engine_registered": B.E4}
    (tmp_path / "record.json").write_text(json.dumps({"rule": "searched_play", "void": None, "inputs": ins}))
    assert B.counted_units(tmp_path / "record.json", "sprime_live") == []
    with pytest.raises(B.BankError, match="not the sprime_honest source's"):
        B.counted_units(tmp_path / "record.json", "sprime_honest")


@pytest.mark.sim
@pytest.mark.integration
def test_the_honest_cell_banks_every_label_and_no_exclusion_as_a_label():
    """S′-HONEST's records unchanged (roadmap FINDING 6): every searched decision is a verified label row; every
    exclusion is an anchor row whose recorded frame is checked; no row is dropped."""
    recs = _honest_recs()
    rows, st = B.game_rows(HONEST, "pstar_s", "sprime_honest")
    n_s = sum(1 for r in recs if r["searched"])
    excl = {int(r["n"]) for r in recs if r.get("why") == "excluded"}
    assert n_s == 8 and len(excl) == 5
    assert st["drop"] == {} and st["labels_kept"] == st["labels_recorded"] == n_s and len(rows) == len(recs)
    assert st["frame_ok"] == st["frame_checked"] == n_s + len(excl)
    assert st["log_sha_bad"] == st["at_bad"] == st["turn_bad"] == st["xcheck_bad"] == st["rescore_log_bad"] == 0
    assert st["mode_bad"] == 0
    assert {r["n"] for r in rows if r["label"]} == {int(r["n"]) for r in recs if r["searched"]}
    assert all(r["frame_checked"] and not r["label"] for r in rows if r["n"] in excl)
    for r in rows:
        if r["label"]:
            assert r["leak"]["builder"] in B.HONEST_BUILDERS and not r["leak"]["partial"]


@pytest.mark.sim
@pytest.mark.integration
def test_hidden_and_ignored_fields_never_shape_a_row(tmp_path):
    """Moving every truth-scored and telemetry field of every record leaves the rows bit for bit."""
    def move(recs):
        for r in recs:
            if "species_hit" in r:
                r["species_hit"] = [1.0] * len(r["species_hit"])
            if "world_teams" in r.get("honest", {}):
                r["honest"]["world_teams"] = []
            for k in ("root_v", "cpu_s", "wall_s"):
                if k in r:
                    r[k] = 0

    cell = tmp_path / "cell"
    shutil.copytree(HONEST, cell)
    recs = _honest_recs()
    move(recs)
    (cell / "search_decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    a, _ = B.game_rows(HONEST, "pstar_s", "sprime_honest")
    b, _ = B.game_rows(cell, "pstar_s", "sprime_honest")
    for rows in (a, b):
        for r in rows:
            r.update(arm=1, game=11, unit_seed=756, pstar_check=1)
    za, zb = B._arrays(a), B._arrays(b)
    assert all(np.array_equal(za[k], zb[k], equal_nan=za[k].dtype.kind == "f") for k in B.FIELDS)


@pytest.mark.sim
@pytest.mark.integration
def test_an_honest_build_refuses_s_prime_live_s_leaky_records():
    with pytest.raises(B.HiddenInfoError):
        B.game_rows(FIXTURE, "pstar_s", "sprime_honest")
