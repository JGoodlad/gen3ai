"""``eval_ledger.reader`` + ``cells`` + ``audit`` — the declared read and everything it guarantees or refuses: the
declaration itself, the request scopes, ``as_of``, ``selection="exclude"``, one regime per call, flags, the
DUPLICATE refusal (a planted duplicate batch), a family read across its looks with per-cell INCONCLUSIVE (X5 §7.4's
registered read), the audit's cross-record invariants and ``verify``."""
from __future__ import annotations

import datetime as _dt
import json

import pytest

from agents.training import eval_ledger as L
from agents.training.eval_ledger import audit as AU
from agents.training.eval_ledger import schema as S
from agents.training.eval_ledger import store as ST
from agents.training.eval_ledger import testkit as K

HOST = "testhost"

OWN = L.ReaderDecl(name="t.own", purposes=L.ALL_PURPOSES, regime=L.RegimeFilter(), requests="own",
                   selection="include", flags_ok=frozenset(), inference="conditional")
ANY = L.ReaderDecl(name="t.any", purposes=L.ALL_PURPOSES, regime=L.RegimeFilter(), requests="any",
                   selection="include", flags_ok=frozenset(), inference="conditional")
ESTIMATE = L.ReaderDecl(name="t.estimate", purposes=frozenset({"audit", "promotion"}), regime=L.RegimeFilter(),
                        requests="any", selection="exclude", flags_ok=frozenset(), inference="conditional")
FAMILY = L.ReaderDecl(name="t.x5", purposes=frozenset({"ab"}),
                      regime=L.RegimeFilter(protocol="gen3_eval_protocol_v1_h2h", mirrored=True), requests="family",
                      selection="include", flags_ok=frozenset(), inference="across_runs", decision_kind="ab_verdict")


class Box:
    """A ledger root with a hand-moved clock and two writers on one host."""

    def __init__(self, root):
        self.root, self.clock = root, K.Clock()
        self.w = L.LedgerWriter(root, "h2h", clock=self.clock, host=HOST, pid=11, alive=lambda _p: True)

    def open(self, rid="r1", **kw):
        args = dict(kind="adhoc", purpose="audit", spec={})
        args.update(kw)
        return self.w.open_request(rid, **args)

    def put(self, req, batch=0, p=K.SHA_A, o=K.SHA_B, reg=None, advance=60, **kw):
        self.clock.advance(advance)
        reg = reg or K.regime()
        c = self.w.claim(req["request_id"], batch=batch, player=p, opponent=o, regime_id=reg["regime_id"],
                         expected_wall_s=60)
        row = K.make_row(self.w.next_row_id(), request=req, batch=batch, p=p, o=o, reg=reg,
                         ts=self.clock.iso(), purpose=req["purpose"],
                         cycle_seed=kw.pop("cycle_seed", hash((req["request_id"], batch, p, o)) % 10 ** 9), **kw)
        self.w.append_row(row, c)
        return row


@pytest.fixture
def box(tmp_path):
    return Box(tmp_path / "ledger")


# ------------------------------------------------------------------------------------------------ the declaration
@pytest.mark.parametrize("kw,needle", [
    (dict(purposes=frozenset({"study"})), "purposes"),
    (dict(purposes=frozenset()), "purposes"),
    (dict(purposes={"ab"}), "purposes"),                                  # a set is not a frozen declaration
    (dict(requests="mine"), "requests"),
    (dict(selection="sometimes"), "selection"),
    (dict(flags_ok=frozenset({"made_up"})), "flags_ok"),
    (dict(inference="vibes"), "inference"),
    (dict(requests="family"), "group-sequential"),                         # a family read without its kind
    (dict(requests="family", decision_kind="promotion"), "group-sequential"),
    (dict(decision_kind="ab_verdict"), "only a family read"),
    (dict(regime=L.RegimeFilter(protocol="gen3_eval_protocol_v9")), "protocol"),
])
def test_a_declaration_that_is_not_one_is_refused(kw, needle):
    base = dict(name="t", purposes=frozenset({"ab"}), regime=L.RegimeFilter(), requests="own", selection="include",
                flags_ok=frozenset(), inference="conditional")
    base.update(kw)
    with pytest.raises(L.ReaderDeclError, match=needle):
        L.ReaderDecl(**base)


def test_a_call_outside_its_declaration_is_refused(box):
    with pytest.raises(L.ReaderDeclError, match="request_id"):
        L.read(OWN, root=box.root)
    with pytest.raises(L.ReaderDeclError, match="DECLARE"):
        L.read(ANY, root=box.root, request_id="r1")
    with pytest.raises(L.ReaderDeclError, match="outside the declaration"):
        L.read(ESTIMATE, root=box.root, purposes=["ab"])
    with pytest.raises(L.ReaderDeclError, match="ReaderDecl"):
        L.read({"name": "dict"}, root=box.root)  # type: ignore[arg-type]
    with pytest.raises(L.ReaderDeclError, match="aware ISO-8601"):
        L.read(ANY, root=box.root, as_of="2026-10-03T12:00:00")         # naive
    assert len(L.read(ANY, root=box.root / "nowhere")) == 0


# ------------------------------------------------------------------------------------------------ scopes + regimes
def test_own_reads_one_request_and_any_reads_them_all(box):
    r1, r2 = box.open("r1"), box.open("r2")
    box.put(r1, 0)
    box.put(r1, 1)
    box.put(r2, 0)
    assert [r["request"]["batch"] for r in L.read(OWN, root=box.root, request_id="r1")] == [0, 1]
    assert len(L.read(ANY, root=box.root)) == 3
    assert len(L.read(ANY, root=box.root, players=[K.SHA_C])) == 0


def test_one_regime_per_call_and_read_by_regime_splits(box):
    r1, r2 = box.open("r1"), box.open("r2")
    box.put(r1, 0)
    box.put(r2, 0, reg=K.regime(turn_limit=100))
    with pytest.raises(L.MixedRegimeError, match="one regime per call"):
        L.read(ANY, root=box.root)
    by = L.read_by_regime(ANY, root=box.root)
    assert sorted(len(v) for v in by.values()) == [1, 1] and all(v.regime_id == k for k, v in by.items())
    assert len(L.read(ANY, root=box.root, regime_id=K.regime()["regime_id"])) == 1


def test_flags_are_opt_in(box, tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    ST.append_line(legacy / "ledger.20261003T120000Z-1.jsonl", K.make_v1_row())
    with_flag = L.ReaderDecl(name="t.legacy", purposes=L.ALL_PURPOSES, regime=L.RegimeFilter(), requests="any",
                             selection="include", flags_ok=frozenset({"digest_unrecorded"}), inference="conditional")
    assert len(L.read(ANY, root=legacy)) == 0, "a v1 row lacks its digest: a reader that did not opt in sees none"
    got = L.read(with_flag, root=legacy)
    assert len(got) == 1 and got.rows[0]["regime"]["v1_id"]


# ------------------------------------------------------------------------------------------------ duplicates
def test_a_planted_duplicate_batch_is_refused_by_every_read_and_by_the_audit(box):
    r1 = box.open("r1")
    row = box.put(r1, 0)
    assert AU.audit(box.root).ok
    dup = dict(row, row_id="planted-1-h2h:0")
    ST.append_line(box.root / "rows" / "h2h" / "ledger.planted-1-h2h.jsonl", dup)
    with pytest.raises(L.DuplicateBatchError, match="batch key"):
        L.read(OWN, root=box.root, request_id="r1")
    with pytest.raises(L.DuplicateBatchError):
        L.read(ANY, root=box.root, players=[K.SHA_C])       # even a read that would not select it
    rep = AU.audit(box.root)
    assert not rep.ok and any("DUPLICATE batch key" in p for p in rep.problems)


def test_the_same_seed_block_under_another_request_is_a_duplicate_too(box):
    r1, r2 = box.open("r1"), box.open("r2")
    row = box.put(r1, 0, cycle_seed=5)
    forged = K.make_row("planted-2-h2h:0", request=r2, batch=0, cycle_seed=5, ts=row["ts"])
    ST.append_line(box.root / "rows" / "h2h" / "ledger.planted-2-h2h.jsonl", forged)
    with pytest.raises(L.DuplicateBatchError, match="seed block"):
        L.read(ANY, root=box.root)


def test_a_copied_shard_is_a_duplicate_row_id(box):
    r1 = box.open("r1")
    box.put(r1, 0)
    src = box.w.rows_path
    (box.root / "rows" / "h2h" / "ledger.copy-1-h2h.jsonl").write_text(src.read_text())
    with pytest.raises(L.DuplicateBatchError, match="row_id"):
        L.read(ANY, root=box.root)


def test_a_correction_supersedes_and_is_not_a_duplicate(box):
    r1 = box.open("r1")
    row = box.put(r1, 0)
    fix = dict(row, row_id="fix-1-h2h:0", supersedes=row["row_id"], ts=box.clock.iso(120), run="corrected")
    ST.append_line(box.root / "rows" / "h2h" / "ledger.fix-1-h2h.jsonl", fix)
    got = L.read(ANY, root=box.root)
    assert [r["run"] for r in got] == ["corrected"]
    before = L.read(ANY, root=box.root, as_of=box.clock.iso(60))
    assert [r["run"] for r in before] == ["study"], "as_of predates the correction"


# ------------------------------------------------------------------------------------------------ as_of + selection
def test_as_of_restricts_rows_and_decisions(box):
    r1 = box.open("r1", purpose="promotion", kind="sprt")
    box.put(r1, 0)
    t_mid = box.clock.iso()
    box.put(r1, 1)
    assert len(L.read(OWN, root=box.root, request_id="r1", as_of=t_mid)) == 1
    assert len(L.read(OWN, root=box.root, request_id="r1")) == 2
    assert len(L.read(OWN, root=box.root, request_id="r1", as_of=_dt.datetime(2020, 1, 1, tzinfo=_dt.timezone.utc))) == 0


def test_selection_exclude_drops_the_rows_that_selected_the_node(box):
    sprt = box.open("sprt1", purpose="promotion", kind="sprt")
    sel = box.put(sprt, 0)
    other = box.open("matrix1", purpose="audit", kind="matrix_dense")
    kept = box.put(other, 0, cycle_seed=77)
    consumed = L.read(OWN, root=box.root, request_id="sprt1")
    box.clock.advance(60)
    t_before = box.clock.iso(-1)
    box.w.append_decision(kind="promotion", subject=K.SHA_A, consumed=consumed, verdict="PROMOTE",
                          rule="sprt_promotion", rule_version="1", request_id="sprt1")
    got = L.read(ESTIMATE, root=box.root, players=[K.SHA_A])
    assert [r["row_id"] for r in got] == [kept["row_id"]] and got.excluded == (sel["row_id"],)
    assert len(L.read(ESTIMATE, root=box.root, players=[K.SHA_A], as_of=t_before)) == 2, \
        "before the decision existed, its rows were not yet 'selecting'"
    inc = L.ReaderDecl(name="t.inc", purposes=frozenset({"audit", "promotion"}), regime=L.RegimeFilter(),
                       requests="any", selection="include", flags_ok=frozenset(), inference="conditional")
    assert len(L.read(inc, root=box.root)) == 2


# ------------------------------------------------------------------------------------------------ X5 §7.4
def test_a_family_read_across_looks_with_per_cell_inconclusive(box):
    box.w.register_family("x5_ab", decision_kind="ab_verdict", rule="design_x5 §7.4 v1",
                          protocol="gen3_eval_protocol_v1_h2h", commit="abc123")
    x5 = [K.SHA_A, K.SHA_B]
    blob = [K.SHA_C, K.SHA_D]
    look1 = box.open("x5_look1", kind="ab_cell", purpose="ab", family="x5_ab")
    for i, (p, o) in enumerate([(x5[0], blob[0])]):
        box.put(look1, 0, p=p, o=o, pc=(1, 2, 4, 2, 3))
    look2 = box.open("x5_look2", kind="ab_cell", purpose="ab", family="x5_ab")
    for p, o in [(x5[0], blob[1]), (x5[1], blob[0]), (x5[1], blob[1])]:
        box.put(look2, 0, p=p, o=o, pc=(1, 2, 4, 2, 3))
    # a cell whose games mostly aborted (3 voided pairs, 6 aborted of 6 + 24 attempted = 20 % ... and one over 25 %)
    bad = box.open("x5_look2b", kind="ab_cell", purpose="ab", family="x5_ab")
    box.put(bad, 0, p=x5[0], o=blob[0], pc=(0, 0, 3, 0, 0), voided=2, aborted=4, cycle_seed=999)
    unrelated = box.open("adhoc")
    box.put(unrelated, 0, cycle_seed=1234)

    got = L.read(FAMILY, root=box.root, family="x5_ab")
    assert {r["request"]["id"] for r in got} == {"x5_look1", "x5_look2", "x5_look2b"}
    looks = L.looks(got, min_pairs=10)
    assert [rid for rid, _ in looks] == ["x5_look1", "x5_look2", "x5_look2b"], "looks in the order opened"
    flat = [c for _rid, cs in looks for c in cs]
    ok = [c for c in flat if c.verdict == "OK"]
    bad_cells = [c for c in flat if c.verdict == "INCONCLUSIVE"]
    assert len(ok) == 4 and len(bad_cells) == 1
    reasons = bad_cells[0].reasons
    assert any("aborted 4 of 10" in r for r in reasons) and any("< the registered 10" in r for r in reasons)
    # X5 §7.4's statistic is expressible from the cells: h_ij per (X5_i, blob_j) cell, delta-hat = mean(h) - 50
    h = {(c.player, c.opponent): 100 * c.score for c in ok}
    delta_hat = sum(h.values()) / len(h) - 50
    assert len(h) == 4 and delta_hat == pytest.approx(100 * (0 * 1 + 1 * 2 + 2 * 4 + 3 * 2 + 4 * 3) / (4 * 12) - 50)
    with pytest.raises(L.InferenceScopeError, match="conditional"):
        L.pooled_pairs(got)                       # declared across_runs: a pooled pentanomial is the wrong error bar


def test_a_family_read_needs_the_family_registered_for_its_decision_kind(box):
    with pytest.raises(L.ReaderDeclError, match="not registered"):
        L.read(FAMILY, root=box.root, family="x5_ab")
    box.w.register_family("plat", decision_kind="plateau", rule="§8 v1", protocol="gen3_eval_protocol_v1_h2h")
    with pytest.raises(L.ReaderDeclError, match="registered for 'plateau'"):
        L.read(FAMILY, root=box.root, family="plat")


def test_a_cell_is_inconclusive_strictly_above_a_quarter_aborted(box):
    """4 x aborted > attempted, in integers: EXACTLY 25 % is not INCONCLUSIVE; one more aborted game is. No input
    sits within a rounding error of the boundary (standing rule 8)."""
    at = box.open("at_limit")
    box.put(at, 0, pc=(0, 0, 3, 0, 0), voided=2, aborted=2, cycle_seed=1)     # 2 of 8 attempted = 25 %
    over = box.open("over_limit")
    box.put(over, 0, pc=(0, 0, 3, 0, 0), voided=2, aborted=3, cycle_seed=2)   # 3 of 9 attempted > 25 %
    v = {c.request_id: c for c in L.cells(L.read(ANY, root=box.root))}
    assert v["at_limit"].attempted == 8 and v["at_limit"].verdict == "OK"
    assert v["over_limit"].verdict == "INCONCLUSIVE" and "aborted 3 of 9" in v["over_limit"].reasons[0]


def test_pooled_pairs_reads_one_mirrored_edge_conditionally(box):
    r1 = box.open("r1")
    box.put(r1, 0, pc=(0, 0, 2, 0, 0))
    box.put(r1, 1, pc=(0, 0, 0, 0, 2))
    est = L.pooled_pairs(L.read(OWN, root=box.root, request_id="r1"))
    assert est["pair_counts"] == [0, 0, 2, 0, 2] and est["n_pairs"] == 4 and est["score"] == pytest.approx(0.75)
    box.put(r1, 2, o=K.SHA_C)
    with pytest.raises(L.MixedRegimeError, match="ONE edge"):
        L.pooled_pairs(L.read(OWN, root=box.root, request_id="r1"))


# ------------------------------------------------------------------------------------------------ audit + verify
def test_the_audit_reports_cross_record_problems(box):
    r1 = box.open("r1")
    row = box.put(r1, 0)
    rep = AU.audit(box.root)
    assert rep.ok, rep.problems
    assert rep.counts["rows"] == 1 and rep.counts["requests"] == 1 and rep.counts["live_claims"] == 0
    # a row whose request was never opened, written behind the writer's back
    ghost = K.make_row("ghost-1-h2h:0", request={**r1, "request_id": "ghost"}, cycle_seed=4242, ts=row["ts"])
    ST.append_line(box.root / "rows" / "h2h" / "ledger.ghost-1-h2h.jsonl", ghost)
    # a correction naming a row that does not exist
    stray = dict(K.make_row("stray-1-h2h:0", cycle_seed=31337), supersedes="nope:0")
    ST.append_line(box.root / "rows" / "h2h" / "ledger.stray-1-h2h.jsonl", stray)
    probs = AU.audit(box.root).problems
    assert any("never opened" in p for p in probs) and any("supersedes nope:0" in p for p in probs)


def test_a_requested_row_without_its_claim_fails_the_audit(box):
    r1 = box.open("r1")
    unclaimed = K.make_row("unclaimed-1-h2h:0", request=r1, ts=box.clock.iso(60))
    ST.append_line(box.root / "rows" / "h2h" / "ledger.unclaimed-1-h2h.jsonl", unclaimed)
    assert any("no `row` event" in p for p in AU.audit(box.root).problems)


def test_verify_rechecks_a_decisions_rows_and_digest(box):
    r1 = box.open("r1", purpose="promotion", kind="sprt")
    box.put(r1, 0)
    d = box.w.append_decision(kind="promotion", subject=K.SHA_A, consumed=L.read(OWN, root=box.root, request_id="r1"),
                              verdict="PROMOTE", rule="sprt", rule_version="1", request_id="r1")
    v = AU.verify(box.root, d["decision_id"])
    assert v.ok and v.verdict_rederived is None and "no rule registered" in v.note
    assert AU.audit(box.root).ok
    # tamper: a correction changes a consumed row -> the decision's rows are no longer live
    row = L.read(OWN, root=box.root, request_id="r1").rows[0]
    fix = dict(row, row_id="fix-1-h2h:0", supersedes=row["row_id"], ts=box.clock.iso(30), run="edited")
    ST.append_line(box.root / "rows" / "h2h" / "ledger.fix-1-h2h.jsonl", fix)
    assert any("not live" in p for p in AU.audit(box.root).problems)
    assert AU.verify(box.root, d["decision_id"]).ok, "as of its own as_of, the decision's rows were live"
    assert not AU.verify(box.root, "nope").ok


def test_the_cli_audits_and_shows(box, capsys):
    from main import eval_ledger as CLI

    box.put(box.open("r1"), 0)
    assert CLI.main(["audit", "--root", str(box.root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert CLI.main(["show", "--root", str(box.root)]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["rows"] == 1 and s["requests_open"] == ["r1"] and s["by_schema_origin"] == {"v2": 1}
    assert CLI.main(["request-close", "r1", "--root", str(box.root)]) == 0
    capsys.readouterr()
    assert CLI.main(["family-register", "--root", str(box.root), "--id", "f", "--decision-kind", "ab_verdict",
                     "--rule", "r", "--protocol", "gen3_eval_protocol_v1_h2h"]) == 0
    assert S.validate_event(json.loads(capsys.readouterr().out)) == []
    ST.append_line(box.root / "rows" / "h2h" / "ledger.bad-1-h2h.jsonl", {"schema": "nope"})
    assert CLI.main(["audit", "--root", str(box.root)]) == 1
