"""H-TEAMS (``hteams.py``): the seat rules, both panel nets (m4 at >= 3; m5's R72 never excluding its source), the
account cap, the duplicate rule, R72's draw, the chain check, the parity report and the finalized corpus's guards.

Every fixture is SYNTHETIC (invented sets and ids): no human replay, team or account name is in this file.
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from main import heldout_panels as HP
from main.human_data import hteams as H
from main.human_data import rowset as RS

MONS = [
    ("Snorlax", "Leftovers", "Thick Fat", ["Body Slam", "Earthquake", "Rest", "Sleep Talk"]),
    ("Skarmory", "Leftovers", "Keen Eye", ["Spikes", "Roar", "Drill Peck", "Protect"]),
    ("Blissey", "Leftovers", "Natural Cure", ["Seismic Toss", "Soft-Boiled", "Toxic", "Aromatherapy"]),
    ("Gengar", "Leftovers", "Levitate", ["Thunderbolt", "Ice Punch", "Giga Drain", "Explosion"]),
    ("Tyranitar", "Leftovers", "Sand Stream", ["Rock Slide", "Earthquake", "Pursuit", "Dragon Dance"]),
    ("Salamence", "Choice Band", "Intimidate", ["Double-Edge", "Earthquake", "Rock Slide", "Hidden Power [Flying]"]),
    ("Zapdos", "Leftovers", "Pressure", ["Thunderbolt", "Hidden Power [Ice]", "Toxic", "Rest"]),
    ("Metagross", "Leftovers", "Clear Body", ["Meteor Mash", "Earthquake", "Explosion", "Agility"]),
]


def paste(idx, swap=None):
    """A synthetic six-mon paste of ``MONS[idx]``; ``swap`` = {position: new last move}."""
    blocks = []
    for pos, k in enumerate(idx):
        sp, item, ab, moves = MONS[k]
        moves = list(moves)
        if swap and pos in swap:
            moves[-1] = swap[pos]
        blocks.append(f"{sp} @ {item}\nAbility: {ab}\nEVs: 252 HP / 4 Def / 252 SpD\nCareful Nature\n"
                      + "\n".join(f"- {m}" for m in moves))
    return "\n\n".join(blocks)


# ------------------------------------------------------------------------------------------------ seats

def test_seats_keep_actor_ge_1500_not_flagged_and_group_by_battle():
    keep = [{"rel": "d/a.log", "sha": "s", "n": 1, "seats": ["p1", "p2"], "rating": {"p1": 1500, "p2": 1499},
             "acct": {"p1": "acctA", "p2": "acctB"}},
            {"rel": "d/b.log", "sha": "s", "n": 1, "seats": ["p2"], "rating": {"p2": 1700}, "acct": {"p2": "flag"}}]
    c = collections.Counter()
    got = H.select_seats(keep, {RS.key64("flag")}, c)
    assert [(b["rel"], b["seats"]) for b in got] == [("d/a.log", ["p1"])]
    assert c["drop:rating_lt_1500"] == 1 and c["drop:flagged_account"] == 1 and c["seat:kept_in_r2_battle"] == 3


def test_a_revealed_panel_core_on_either_seat_drops_the_whole_battle_and_six_seen_is_required():
    rec = {"reveals": {"p1": {f"m{i}": ([], None, None) for i in range(6)}, "p2": {"x": ([], None, None)}}}
    clean = {"p1": {"forum72": 2}, "p2": {"forum72": 0}}
    assert H.judge_battle(rec, ["p1", "p2"], clean) == {"p1": "ok", "p2": "not_six_seen"}
    near = {"p1": {"forum72": 0}, "p2": {"p72": 3}}                  # the OTHER seat pilots a panel core
    assert H.judge_battle(rec, ["p1"], near) == {"p1": "panel_near_revealed"}


# ------------------------------------------------------------------------------------------------ nets (m4, m5)

@pytest.fixture
def fake_panels(monkeypatch):
    """Three panels: ``hold`` (held out), ``home`` (in-pool) and ``r72x`` (practised, drawn from THIS corpus)."""
    from main.anchors.team_overlap import Reference, parse_items

    texts = {"hold": [("h1", paste([0, 1, 2, 3, 4, 5]))], "home": [("o1", paste([2, 3, 4, 5, 6, 7]))],
             "r72x": [("r1", paste([0, 1, 6, 7, 3, 5], swap={4: "Taunt", 5: "Swords Dance"}))]}
    reg = {"panels": {"hold": {"held_out_from": ["training"]}, "home": {"held_out_from": []},
                      "r72x": {"held_out_from": [], "drawn_from_corpus": H.ROWSET_ID}}}
    monkeypatch.setattr(HP, "load_registry", lambda: reg)
    monkeypatch.setattr(HP, "_reference", lambda name: Reference(parse_items(texts[name])))
    return texts


def test_m5_a_panel_drawn_from_this_corpus_never_excludes_its_source(fake_panels):
    panels, skipped = H.net_panels()
    assert panels == ["hold", "home"] and skipped == ["r72x"]
    # a team IDENTICAL to the practised panel's team stays (on revert to "every panel" it would leave)
    dropped, _ = H.apply_nets([("t", fake_panels["r72x"][0][1])])
    assert dropped == {}


def test_m4_three_shared_sets_with_any_registered_panel_drops_and_the_exact_set_guard_fires_first(fake_panels):
    three_home = paste([6, 7, 2, 0, 1, 3], swap={3: "Curse", 4: "Toxic", 5: "Taunt"})     # 3 home, 1 hold
    two_home = paste([6, 7, 2, 0, 1, 3], swap={2: "Counter", 3: "Curse", 4: "Toxic", 5: "Taunt"})
    four_hold = paste([0, 1, 2, 3, 6, 7])
    dropped, per = H.apply_nets([("a", three_home), ("b", two_home), ("c", four_hold)])
    assert dropped == {"a": "m4_near3", "c": "exact_set_guard"}
    assert per["exact_set_guard:hold"] == 1 and per["m4_near3:home"] == 1


# ------------------------------------------------------------------------------------------------ cap, dedupe, R72

def test_the_account_cap_keeps_at_most_five_per_account_deterministically():
    by = {"a": [f"a{i}" for i in range(9)], "b": ["b0", "b1"]}
    got = H.cap_accounts(by)
    assert len([x for x in got if x.startswith("a")]) == 5 and {"b0", "b1"} <= set(got)
    assert got == H.cap_accounts({"b": ["b1", "b0"], "a": list(reversed(by["a"]))})
    assert got != H.cap_accounts(by, seed=H.CAP_SEED + 1)


def test_duplicate_six_mon_sets_leave_in_id_order():
    t = paste([0, 1, 2, 3, 4, 5])
    keep, dup = H.dedupe([("z", t), ("a", t), ("m", paste([0, 1, 2, 3, 4, 6]))])
    assert keep == ["a", "m"] and dup == 1


def test_r72_takes_one_team_per_account_seeded():
    rows = [{"id": f"t{a}{k}", "curator": f"acct{a}", "text": "x"} for a in range(100) for k in range(3)]
    got = H.draw_r72(rows)
    assert len(got) == 72 and len({r["curator"] for r in got}) == 72
    assert [r["id"] for r in got] == [r["id"] for r in H.draw_r72(list(reversed(rows)))]
    with pytest.raises(H.HTeamsError, match="accounts"):
        H.draw_r72(rows[:3 * 71])


# ------------------------------------------------------------------------------------------------ parity, ids

def test_parity_classes_match_any_and_all_of():
    from utils.showdown_id import to_id_str

    ids = H._mon_ids("Snorlax (M) @ Leftovers\nAbility: Insomnia\n- Rest\n- Hidden Power [Ghost]", to_id_str)
    assert ("species", "snorlax") in ids and ("move", "hiddenpower") in ids
    cls = {"any_of": [["move", "furycutter"]], "all_of_groups": [[["move", "rest"], ["ability", "insomnia"]]]}
    assert H.mon_matches(ids, cls)
    assert not H.mon_matches(ids - {("ability", "insomnia")}, cls)


def test_the_committed_parity_spec_parses_and_reports_per_class():
    rep = H.parity_report([paste([0, 1, 2, 3, 4, 5])])
    assert rep["mons_total"] == 6 and set(rep["mons_by_class"]) == set(rep["teams_by_class"])
    assert rep["mons_by_class"]["bf7_pursuit_lastmove"] == 1          # the Tyranitar's Pursuit


def test_pack_classes_name_the_defect_fields():
    d = ["mon 2 (Snorlax) happiness: Showdown plays 0, our pack 255", "mon 3 (Zapdos) ivs: Showdown plays a, b"]
    assert H.pack_classes(d) == ["happiness", "ivs"]


def test_fill_counts_keep_only_c_game_counters():
    got = H.fill_counts({"c_attempts_0": 3, "row_drop:one_legal": 9, "seat_refused:GuardError": 1, "c_x": 0})
    assert got == {"c_attempts_0": 3, "seat_refused:GuardError": 1}


# ------------------------------------------------------------------------------------------------ the chain

def _chain_fixture(tmp_path):
    x59s, store = tmp_path / "x59s", tmp_path / "store"
    for d in (x59s / H.GRANDPARENT / "rows", x59s / H.PARENT / "rows", x59s / "kakuna_audit_v1", store):
        d.mkdir(parents=True)
    body = b"a.log\tsha\t1\n"
    with gzip.open(store / f"{H.SNAPSHOT}.tsv.gz", "wb") as fh:
        fh.write(body)
    msha = hashlib.sha256(body).hexdigest()
    (x59s / H.GRANDPARENT / "keep.jsonl.gz").write_bytes(gzip.compress(b"{}\n", mtime=0))
    r1m = x59s / H.GRANDPARENT / "rows" / "rows_manifest.json"
    r1m.write_text("{}")
    r1sha = H._sha(r1m)
    (x59s / H.PARENT / "rows" / "rows_manifest.json").write_text(json.dumps({
        "parent_rowset": H.GRANDPARENT, "parent_rows_manifest_sha256": r1sha, "manifest_sha256": msha,
        "scan_keep_sha256": H._sha(x59s / H.GRANDPARENT / "keep.jsonl.gz")}))
    (x59s / "kakuna_audit_v1" / "drop_list.json").write_text('{"drop_accounts": []}')
    reg = {"snapshots": {H.SNAPSHOT: {"manifest_sha256": msha, "files": 1, "population": "spectator"}},
           "rowsets": {H.GRANDPARENT: {"rows_manifest_sha256": r1sha},
                       H.PARENT: {"rows_manifest_sha256": H._sha(x59s / H.PARENT / "rows" / "rows_manifest.json"),
                                  "derived": {"drop_list_sha256": H._sha(x59s / "kakuna_audit_v1" / "drop_list.json")}}}}
    regp = tmp_path / "reg.json"
    regp.write_text(json.dumps(reg))
    return x59s, regp, store


def test_the_chain_verifies_and_refuses_a_moved_link(tmp_path):
    x59s, regp, store = _chain_fixture(tmp_path)
    got = H.verify_chain(x59s, regp, store)
    assert got["chain"] == [H.SNAPSHOT, H.GRANDPARENT, H.PARENT, H.ROWSET_ID]
    (x59s / "kakuna_audit_v1" / "drop_list.json").write_text('{"drop_accounts": [1]}')
    with pytest.raises(H.HTeamsError, match="drop list"):
        H.verify_chain(x59s, regp, store)


def test_the_root_refuses_a_checkout(tmp_path):
    from utils.paths import repo_root

    with pytest.raises(H.HTeamsError, match="never goes under git"):
        H._guard_root(Path(repo_root()) / "x")
    H._guard_root(tmp_path / "ok")


# ------------------------------------------------------------------------------------------------ finalize

def test_finalize_writes_a_corpus_the_readers_load_with_human_data_true(tmp_path, monkeypatch):
    from utils.team_corpora import CORPORA, CorpusError, load_dir, read_dir

    work = tmp_path / "work"
    work.mkdir()
    last = ["Surf", "Ice Beam", "Psychic", "Thunder", "Fire Blast", "Flamethrower", "Toxic", "Protect"]
    cands = [{"id": f"hr3_{k:02d}", "acct": "acct0" if k < 11 else "acct1", "rating": 1600 + k, "attempt": 0,
              "text": paste([1, 2, 3, 4, 5, 6], swap={5: last[k - 4]}),
              "ids": {"species": [], "moves": [], "items": [], "abilities": []}} for k in range(4, 12)]
    H._write_jsonl_gz(work / "candidates.jsonl.gz", cands)
    H._write_json(work / "filter.json", {
        "stages": [["seats", 20], ["completed", 12]], "candidates_sha256": H._sha(work / "candidates.jsonl.gz"),
        "completion_counts": {}, "completion_ident": {}, "hp_repair": {}, "counts": {},
        "validator_failures_by_reason": {}, "nets": {}, "engine_main": {}})
    H._write_json(work / "pin_check.json", {"dropped": {"hr3_04": "pin_pack_differs"}, "pin": "f" * 40,
                                            "pack_identity": {"after": 7}, "engine": {}, "after": 7})
    H._write_json(work / "seats.json", {"chain": {"chain": []}, "counts": {}})
    man = H.run_finalize(tmp_path)
    c = load_dir(tmp_path, H.TIER, H.ROWSET_ID, human_data_ok=True)
    with pytest.raises(CorpusError, match="HUMAN DATA"):           # F-M7: no plain reader takes it
        load_dir(tmp_path, H.TIER, H.ROWSET_ID)
    _, rows, _, _ = read_dir(tmp_path)
    assert man["human_data"] is True and H.ROWSET_ID not in CORPORA
    assert "hr3_04" not in c.ids and len(c) == 6 == man["tiers"][H.TIER]["n"]      # 7 left, acct0 capped 6 -> 5
    assert collections.Counter(r["curator"] for r in rows) == {"acct0": H.ACCOUNT_CAP, "acct1": 1}
    assert man["stages"][-1] == ["account_cap (final)", 6] and man["account_cap_dropped"] == 1
