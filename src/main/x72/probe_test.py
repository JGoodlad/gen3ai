"""X72's probe (``main.x72.probe`` and its by-path player): every restated value against X71's FROZEN rule and reader,
the statistic against X71's two banked verdicts and against the frozen ``classify`` at five targets, X71's banked
cells and every way they stop verifying, the player's calls against the frozen playing function, the plan over
synthetic run dirs, and the driver with a recorded runner. No game is played and nothing touches ``models/``."""
from __future__ import annotations

import ast
import importlib.util
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.training import best_response_gap as E
from agents.training import untaught_meter as um
from main.x72 import finals as F
from main.x72 import probe as P
from main.x72 import probe_play as PP
from main.x72.finals_test import DIR, train_round, world  # noqa: F401  (world is a fixture)
from utils.hashing import sha256_file
from utils.paths import repo_path

X71 = repo_path("designs/research_state/measurements/x71_exploitability_probe_2026-11-16")
RECORD = X71 / "record"
COMMIT = (RECORD / "read" / "play_commit").read_text().strip()


@pytest.fixture(scope="module")
def reader():
    """X71's FROZEN reader, loaded by path; ``reader.R`` is its frozen rule. Never edited: a test patches a copy
    of a table in memory (``monkeypatch.setitem``) to ask the frozen code about a fifth target."""
    before = list(sys.path)
    spec = importlib.util.spec_from_file_location("x72t_frozen_read_x71", X71 / "read_x71.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = before
    return mod


@pytest.fixture()
def five(reader, monkeypatch):
    """The frozen rule asked about FIVE targets: a third read whose arm is ``plat`` (patched in memory)."""
    R = reader.R
    monkeypatch.setitem(R.TARGETS, P.TAG, (P.TAG, "the read final of lineage 1", "X72"))
    monkeypatch.setitem(R.READS, 3, (P.TAG,))
    monkeypatch.setitem(R.READ_TARGETS, 3, P.TARGETS)
    return R


def cells_of(rates, g0=-0.025, **edit):
    """Exact-count cells: target tag -> (responder 7111's gap, 7112's); ``edit[tag]`` replaces a target's cells."""
    out = {}
    for t, (a, b) in rates.items():
        out[t] = {"k0": {"wins": round((0.5 + g0) * P.G0), "finished": P.G0, "attempted": P.G0},
                  "r7111": {"wins": round((0.5 + a) * P.G), "finished": P.G, "attempted": P.G},
                  "r7112": {"wins": round((0.5 + b) * P.G), "finished": P.G, "attempted": P.G}}
    out.update(edit)
    return out


BASE = {t: (0.065, 0.065) for t in P.TARGETS}


def same(a, b) -> bool:
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


# ------------------------------------------------------------------------------------- the restated tables
def test_the_restated_tables_are_the_frozen_x71_rule_and_reader(reader):
    R = reader.R
    assert (P.SEEDS, P.L_RESP, P.PIN, P.FORK_LR, P.RECIPE) == (R.SEEDS, R.L_RESP, R.PIN, R.FORK_LR, R.RECIPE)
    assert P.PIN == F.TRAIN_PIN and P.LINEAGE == 7341 == min(F.SEEDS)
    assert (P.READ_SEED, P.G, P.G0, P.N_TEAMS) == (R.READ_SEED, R.G, R.G0, R.N_TEAMS)
    assert (P.TIMEOUT_BAR, P.RULE8, P.BAR, P.Z_RESPONDED) == (R.TIMEOUT_BAR, R.RULE8, R.BAR, R.Z_RESPONDED)
    assert P.CONTROL == R.CONTROL and P.BANKED == R.READ_TARGETS[2] and P.TARGETS == P.BANKED + ("plat",)
    assert (P.REGIME, P.X71_CHUNK_SCHEMA) == (reader.REGIME, reader.CHUNK_SCHEMA)
    assert P.CELLS == ("k0", "r7111", "r7112")
    assert all(P.run_name(s) == R.RUN_FMT.format(tag="plat", seed=s) for s in P.SEEDS)
    assert P.mult(5) == R.mult(5) and round(P.mult(5), 3) == 2.571
    driver = (X71 / "drive_read_x71.sh").read_text()
    assert f"GB=${{X71_MEM_GB:-{P.MEM_GB}}}" in driver and f"BOUND=${{X71_CHUNK_MAX_S:-{P.CHUNK_MAX_S}}}" in driver
    assert (RECORD.relative_to(repo_path()).as_posix(), P.PREREG) == (P.X71_RECORD, F.PREREG)


def test_the_registration_words_the_probe_as_this_module_restates_it():
    text = " ".join((DIR / "PREREG.md").read_text().split())
    for words in ("the read final of lineage 1 (the lowest seed, chosen now, by rule)",
                  "seeds 7111 and 7112, runs `rb_x35_x71_plat_s7111` / `s7112`, at pin `9e97cad5`",
                  "greedy against greedy, seed 20261020; the target against itself 8,000 games, each responder's "
                  "final against the target 4,000",
                  "**D = L(`plat`) − L(`c0`)**, with X71's banked cells of `c0`, `x67`, `swap` and `x70` reused and "
                  "never re-played",
                  "pooled over the FIVE targets, multiplier t(0.975, 5) = 2.571",
                  "`MORE-EXPLOITABLE` (lower end > +0.05 pp), `LESS-EXPLOITABLE` (upper end < −0.05 pp), "
                  "`NOT-MORE-BY-BAR` (upper end < +5 − 0.05 pp), `UNRESOLVED`, `INCONCLUSIVE`; rule 8 at 0.05 pp",
                  "Also reported, no class: L(`plat`) − L(`swap`)"):
        assert words in text, words
    assert P.PIN.startswith("9e97cad5") and (P.RULE8, P.BAR, P.REPORTED) == (0.0005, 0.05, "swap")


def test_the_responder_argv_is_the_frozen_rules_on_the_fifth_target_and_the_registrations_block(five):
    for s in P.SEEDS:
        mine = P.responder_argv(s, "/a/r3/T/final_model.zip", 360_061_000, teams="a,b,c,d,e")
        assert mine == five.responder_argv("plat", s, "/a/r3/T/final_model.zip", 360_061_000, teams="a,b,c,d,e")
        block = (DIR / "PREREG.md").read_text().split("## 10.")[1].split("```")[1].split()
        flags = [t for t in block if t.startswith("--")]
        assert flags == [t for t in mine if t.startswith("--")][:-1]                     # the block's flags, in order
        typed = {t: block[i + 1] for i, t in enumerate(block[:-1])
                 if t.startswith("--") and not block[i + 1].startswith(("--", "<"))}
        assert typed == {"--fork-lr": "5.6e-05", "--behaviour-check": "warn", "--checkpoint-every-steps": "1000000",
                         "--device": "cuda", "--ent-coef": "0.05", "--magnet-kl-coef": "0"}
        assert all(mine[mine.index(k) + 1] == v for k, v in typed.items())
        assert mine[mine.index("--model") + 1] == mine[mine.index("--exploiter") + 1] == "/a/r3/T/final_model.zip"
        assert mine[mine.index("--seed") + 1] == str(s) and mine[mine.index("--trainee-teams") + 1] == "a,b,c,d,e"
        assert mine[mine.index("--steps") + 1] == str(360_061_000 + 5_000_000)
        assert mine[-2:] == ["--run-name", f"rb_x35_x71_plat_s{s}"]
    with pytest.raises(P.ProbeRefused):
        P.responder_argv(7341, "/a/T.zip", 1, teams="a,b,c,d,e")


def test_the_default_trainee_teams_are_the_files_x71s_cells_were_played_on(monkeypatch):
    monkeypatch.chdir(repo_path())
    argv = P.responder_argv(7111, "/a/T.zip", 1)
    assert argv[argv.index("--trainee-teams") + 1] == ",".join(P.banked()["team_paths"])


# ------------------------------------------------------------------------------------------------ the statistic
@pytest.mark.parametrize("read, targets, arms", [(1, ("c0", "x67", "swap"), ("x67", "swap")),
                                                 (2, ("c0", "x67", "swap", "x70"), ("x70",))])
def test_classify_reproduces_x71s_banked_verdicts_field_for_field(read, targets, arms):
    banked = json.loads((RECORD / "read" / f"x71_verdict_read{read}.json").read_text())
    got = P.classify(P.banked()["cells"], targets, arms)
    assert set(got) == {"targets", "bar", "rule8", "games", "cls", "why", "s2r", "sigma_r", "mult", "control_L",
                        "control_L_lo", "arms", "near_boundary"}
    for key, value in got.items():
        assert same(value, banked[key]), key
    assert got["cls"] == banked["cls"] and all(c != "INCONCLUSIVE" for c in got["cls"].values())


def _random_cells(rng: random.Random):
    out = {}
    for t in P.TARGETS:
        out[t] = {}
        for cell, games in (("k0", P.G0), ("r7111", P.G), ("r7112", P.G)):
            n = games - (rng.choice((0, 0, 0, 3, 40, games // 3)) if rng.random() < 0.4 else 0)
            p = (0.47 if cell == "k0" else 0.47 + rng.choice((0.0, 0.03, 0.06, 0.09, 0.15))) + rng.uniform(-0.03, 0.03)
            out[t][cell] = {"wins": min(n, round(p * n)), "finished": n, "attempted": games}
        if rng.random() < 0.04:
            del out[t][rng.choice(P.CELLS)]
    return out


def test_classify_at_five_targets_is_the_frozen_classify_asked_about_a_fifth_target(five):
    rng = random.Random(7205)
    seen = set()
    for _ in range(600):
        cells = _random_cells(rng)
        mine, frozen = P.classify(cells), five.classify(cells, 3)
        assert {k: v for k, v in frozen.items() if k not in ("schema", "read")} == mine
        seen.add((mine["cls"]["plat"], mine["why"].split(":")[0][:11]))
    assert {c for c, _ in seen} == {"MORE-EXPLOITABLE", "LESS-EXPLOITABLE", "NOT-MORE-BY-BAR", "UNRESOLVED",
                                    "INCONCLUSIVE"}
    assert ("INCONCLUSIVE", "the control") in seen and len([s for s in seen if s[0] == "INCONCLUSIVE"]) >= 2


@pytest.mark.parametrize("edit, cls", [
    ({}, "NOT-MORE-BY-BAR"),
    ({"plat": (0.165, 0.165)}, "MORE-EXPLOITABLE"),
    ({"plat": (-0.035, -0.035)}, "LESS-EXPLOITABLE"),
    ({"plat": (0.090, 0.090)}, "UNRESOLVED"),
    ({"plat": (0.245, 0.085)}, "UNRESOLVED"),              # +10 pp with a 16 pp split: the spread widens it
    ({"c0": (-0.025, -0.025), "plat": (0.165, 0.165)}, "INCONCLUSIVE"),     # the control did not respond
    ({"x67": (0.165, 0.165)}, "NOT-MORE-BY-BAR"),          # a bystander's gap is not the arm's
])
def test_the_classes_at_five_targets(five, edit, cls):
    cells = cells_of(dict(BASE, **edit))
    got = P.classify(cells)
    assert got["cls"] == {"plat": cls} == five.classify(cells, 3)["cls"]
    assert same(got["targets"], five.classify(cells, 3)["targets"])


def test_a_missing_or_unfinished_cell_of_any_of_the_five_targets_is_inconclusive(five):
    for tag in P.TARGETS:
        for cell in P.CELLS:
            cells = cells_of(BASE)
            del cells[tag][cell]
            got = P.classify(cells)
            assert got["cls"] == {"plat": "INCONCLUSIVE"} and f"{tag}/{cell}: missing" in got["why"]
            assert "arms" not in got and "s2r" not in got and P.reported(got) is None
            games = P.G0 if cell == "k0" else P.G
            lost = cells_of(BASE)
            lost[tag][cell] = {"wins": 10, "finished": games * 3 // 4 - 1, "attempted": games}     # > 25 % lost
            assert P.classify(lost)["cls"] == {"plat": "INCONCLUSIVE"} == five.classify(lost, 3)["cls"]
            kept = cells_of(BASE)
            kept[tag][cell] = {"wins": round(0.5 * games * 3 // 4), "finished": games * 3 // 4, "attempted": games}
            assert P.classify(kept)["cls"]["plat"] != "INCONCLUSIVE" or tag == "c0"      # exactly 25 % is read
    with pytest.raises(P.ProbeRefused):
        P.classify(cells_of(BASE, plat={"k0": {"wins": 9, "finished": 8, "attempted": 8}}))
    assert P.classify({t: c for t, c in cells_of(BASE).items() if t != "plat"})["cls"] == {"plat": "INCONCLUSIVE"}


def test_an_end_within_rule_8_of_a_boundary_satisfies_neither_side(five):
    """Walk the arm's responders up one win at a time: every interval end that comes within 0.05 pp of 0 or of the
    bar is recorded and classed as if it had not crossed (deterministic: exact counts, no draw)."""
    hit = {}
    for wins in range(round(0.52 * P.G), round(0.66 * P.G)):
        cells = cells_of(BASE)
        cells["plat"]["r7111"]["wins"] = cells["plat"]["r7112"]["wins"] = wins
        got = P.classify(cells)
        assert got["near_boundary"] == five.classify(cells, 3)["near_boundary"]
        a = got["arms"]["plat"]
        for note in got["near_boundary"]:
            if "upper end" in note and note.endswith("+0.05"):
                assert abs(a["hi"] - P.BAR) <= P.RULE8 and got["cls"]["plat"] == "UNRESOLVED"
                hit["bar"] = True
            if "lower end" in note:
                assert abs(a["lo"]) <= P.RULE8 and got["cls"]["plat"] == "UNRESOLVED"
                hit["zero"] = True
        if not got["near_boundary"]:
            want = ("MORE-EXPLOITABLE" if a["lo"] > 0 else "LESS-EXPLOITABLE" if a["hi"] < 0
                    else "NOT-MORE-BY-BAR" if a["hi"] < P.BAR else "UNRESOLVED")
            assert got["cls"]["plat"] == want
    assert hit == {"bar": True, "zero": True}


def test_the_interval_is_the_registered_arithmetic_and_the_swap_difference_is_reported_without_a_class():
    cells = cells_of(dict(BASE, plat=(0.115, 0.115), swap=(0.085, 0.085)))
    got = P.classify(cells)
    p0, pc, pa, ps = 0.475, 0.565, 0.615, 0.585
    v0 = p0 * (1 - p0) / P.G0

    def var(p):
        return p * (1 - p) / P.G / 2 + v0

    se = (var(pa) + var(pc)) ** 0.5
    a = got["arms"]["plat"]
    assert got["s2r"] == 0.0 and got["mult"] == P.mult(5)
    assert (round(a["D"], 12), round(a["se"], 12)) == (0.05, round(se, 12))
    assert (round(a["lo"], 12), round(a["hi"], 12)) == (round(0.05 - P.mult(5) * se, 12),
                                                        round(0.05 + P.mult(5) * se, 12))
    r = P.reported(got)
    se_r = (var(pa) + var(ps)) ** 0.5
    assert "NO class" in r["what"] and set(r) == {"what", "D", "se", "lo", "hi"}
    assert (round(r["D"], 12), round(r["se"], 12)) == (0.03, round(se_r, 12))
    assert (round(r["lo"], 12), round(r["hi"], 12)) == (round(0.03 - P.mult(5) * se_r, 12),
                                                        round(0.03 + P.mult(5) * se_r, 12))
    # the responder spread is pooled over the FIVE targets: a bystander's split widens the arm's interval
    split = P.classify(cells_of(dict(BASE, plat=(0.115, 0.115), x70=(0.145, -0.015))))
    assert split["s2r"] > 0 and split["arms"]["plat"]["se"] > a["se"]
    e = [split["targets"][t]["e"] for t in P.TARGETS]
    assert split["s2r"] == max(0.0, sum(e) / 5) / 2.0


# ------------------------------------------------------------------------------------------- X71's banked cells
def test_the_banked_cells_are_x71s_record_verified():
    b = P.banked()
    verdict = json.loads((RECORD / "read" / "x71_verdict_read2.json").read_text())
    assert b["cells"] == verdict["cells"] and b["by_team"] == verdict["by_team"]
    assert b["transport"] == verdict["transport"] == "rust_eval" and b["play_commit"] == COMMIT
    assert sorted(b["chunks"]) == sorted(f"read/ledger/{t}__{c}.json" for t in P.BANKED for c in P.CELLS)
    assert all(sha256_file(RECORD / rel) == sha for rel, sha in b["chunks"].items())
    assert len(b["teams"]) == 5 == len(b["team_paths"]) and b["cells"]["c0"]["k0"]["attempted"] == P.G0
    assert round(100 * P.classify(b["cells"], P.BANKED, ("x70",))["control_L"], 2) == 5.66      # the PREREG's +5.66 pp


def test_the_teams_x71_played_on_are_this_checkouts_files():
    b = P.banked()
    assert [[t.index, t.key, t.sha1] for t in um.team_slices(b["team_paths"], prefix="X")] == b["teams"]


@pytest.fixture()
def record(tmp_path):
    """A scratch copy of the files of X71's record the probe reads."""
    rec = tmp_path / "x71_record"
    shutil.copytree(RECORD / "read" / "ledger", rec / "read" / "ledger")
    for name in ("x71_verdict_read2.json", "plan_read2.json", "play_commit"):
        shutil.copy(RECORD / "read" / name, rec / "read" / name)
    shutil.copy(RECORD / "SHA256SUMS", rec / "SHA256SUMS")
    return rec


def _relist(rec: Path, rel: str) -> None:
    lines = [(f"{sha256_file(rec / rel)}  {rel}" if ln.split()[1] == rel else ln)
             for ln in (rec / "SHA256SUMS").read_text().splitlines()]
    (rec / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def test_a_banked_record_that_changed_is_refused(record):
    assert P.banked(record)["cells"] == P.banked()["cells"]
    rel = "read/ledger/c0__r7111.json"
    chunk = json.loads((record / rel).read_text())
    key = sorted(chunk["counts"])[0]
    chunk["counts"][key]["wins"] += 1
    (record / rel).write_text(json.dumps(chunk, sort_keys=True) + "\n")
    with pytest.raises(P.ProbeRefused, match="c0__r7111.json: no longer matches X71's SHA256SUMS"):
        P.banked(record)
    _relist(record, rel)                     # even re-listed, the cells are no longer the verdict's
    with pytest.raises(P.ProbeRefused, match="do not sum to the cells X71's read-2 verdict was classed on"):
        P.banked(record)


@pytest.mark.parametrize("field, value, why", [
    ("seed", 1, "is not X71's chunk of this cell"), ("regime", "sampled", "is not X71's chunk"),
    ("games_per_team", 400, "is not X71's chunk"), ("cell", "r7112", "is not X71's chunk"),
    ("schema", "x72_probe_chunk_v1", "is not X71's chunk"), ("teams", [], "not the ledger's one team list")])
def test_a_banked_chunk_that_is_not_x71s_cell_is_refused(record, field, value, why):
    rel = "read/ledger/x70__r7111.json"
    chunk = json.loads((record / rel).read_text())
    chunk[field] = value
    (record / rel).write_text(json.dumps(chunk, sort_keys=True) + "\n")
    _relist(record, rel)
    with pytest.raises(P.ProbeRefused, match=why):
        P.banked(record)


def test_a_missing_or_unlisted_banked_file_is_refused(record):
    (record / "read" / "ledger" / "swap__k0.json").unlink()
    with pytest.raises(P.ProbeRefused, match="swap__k0.json: missing from X71's record"):
        P.banked(record)
    shutil.copy(RECORD / "read" / "ledger" / "swap__k0.json", record / "read" / "ledger" / "swap__k0.json")
    sums = [ln for ln in (record / "SHA256SUMS").read_text().splitlines() if "play_commit" not in ln]
    (record / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    with pytest.raises(P.ProbeRefused, match="read/play_commit: not listed in X71's SHA256SUMS"):
        P.banked(record)


# ------------------------------------------------------------------------------------------------- the player
class _Meter:
    """The meter's playing surface, recorded: what ``resolve_ref`` and ``play_cells`` were called with."""

    def __init__(self, monkeypatch, wins=3):
        self.calls = []
        monkeypatch.setattr(um, "resolve_ref", self.resolve_ref)
        monkeypatch.setattr(um, "play_cells", self.play_cells)
        monkeypatch.setattr(um, "cells_transport", lambda cells: "rust_eval")
        self.wins = wins

    def resolve_ref(self, ref, **kw):
        self.calls.append(("resolve_ref", ref, kw))
        return SimpleNamespace(label=kw["label"], path=ref)

    def play_cells(self, refs, teams, opponent, **kw):
        log = kw.pop("game_log")
        self.calls.append(("play_cells", [r.path for r in refs], [t.key for t in teams], opponent.path, kw))
        n = kw["games_per_team"]
        for t in teams:
            log[(refs[0].label, t.key)] = [{"game": i, "result": "win"} for i in range(n)]
        return {refs[0].label: {t.key: um.Cell(wins=self.wins, losses=n - self.wins, finished=n, attempted=n,
                                               transport="rust_eval") for t in teams}}


def _want(tmp_path, cell="r7111", games=4):
    ref, target = tmp_path / "resp.zip", tmp_path / "target.zip"
    ref.write_bytes(b"responder")
    target.write_bytes(b"target")
    b = P.banked()
    return {"schema": P.CHUNK_SCHEMA, "target_tag": "plat", "cell": cell, "round": 2, "path": str(ref),
            "sha256": sha256_file(ref), "target": str(target), "target_sha256": sha256_file(target),
            "team_paths": b["team_paths"], "teams": b["teams"], "seed": P.READ_SEED, "games_per_team": games,
            "regime": P.REGIME, "play_commit": COMMIT}


def test_the_player_plays_exactly_the_frozen_readers_playing_function(reader, monkeypatch, tmp_path):
    w = _want(tmp_path)
    teams = um.team_slices(w["team_paths"], prefix="X")
    frozen = _Meter(monkeypatch)
    cells, rows, transport = reader.play_chunk(w["path"], w["target"], teams, w["games_per_team"], label="plat_r7111")
    mine = _Meter(monkeypatch)
    body = PP.play(w, um=um)
    assert mine.calls == frozen.calls and len(mine.calls) == 3
    assert mine.calls[2][4] == {"games_per_team": 4, "seed": 20261020, "stochastic": False}
    assert body == dict(w, counts={k: PP.asdict(c) for k, c in cells.items()}, transport=transport, game_rows=rows)
    assert sorted(body["counts"]) == sorted(t[1] for t in w["teams"]) and body["counts"]["X_9eb3abdc"]["wins"] == 3


@pytest.mark.parametrize("edit, why", [
    (lambda w: w["teams"][2].__setitem__(2, "0" * 40), "team files are"),
    (lambda w: w.__setitem__("teams", w["teams"][::-1]), "team files are"),
    (lambda w: w.__setitem__("sha256", "0" * 64), "planned 000000000000"),
    (lambda w: w.__setitem__("target_sha256", "f" * 64), "planned ffffffffffff")])
def test_the_player_refuses_other_teams_or_files_before_any_game(monkeypatch, tmp_path, edit, why):
    w = _want(tmp_path)
    edit(w)
    meter = _Meter(monkeypatch)
    with pytest.raises(PP.PlayRefused, match=why):
        PP.play(w, um=um)
    assert meter.calls == []


def test_the_player_refuses_a_meter_or_a_commit_that_is_not_the_play_checkouts(monkeypatch, tmp_path):
    meter = _Meter(monkeypatch)
    with pytest.raises(PP.PlayRefused, match="the meter in use is"):
        PP.play(_want(tmp_path), checkout=str(tmp_path), um=um)
    with pytest.raises(PP.PlayRefused, match="not the planned playing commit " + "0" * 40):
        PP.play(dict(_want(tmp_path), play_commit="0" * 40), checkout=str(repo_path()), um=um)
    assert meter.calls == []
    head = subprocess.run(["git", "-C", str(repo_path()), "rev-parse", "HEAD"], capture_output=True, text=True)
    body = PP.play(dict(_want(tmp_path), play_commit=head.stdout.strip()), checkout=str(repo_path()), um=um)
    assert len(meter.calls) == 3 and body["transport"] == "rust_eval"


def test_the_player_banks_a_chunk_once_and_imports_nothing_of_this_checkouts_tools(monkeypatch, tmp_path, capsys):
    w = _want(tmp_path)
    (tmp_path / "want.json").write_text(json.dumps(w))
    monkeypatch.setattr(PP, "play", lambda want, checkout=None: dict(want, counts={"X_a": {"wins": 2, "finished": 3}},
                                                                     transport="rust_eval", game_rows={}))
    out = tmp_path / "ledger" / "plat__r7111.json"
    argv = ["--want", str(tmp_path / "want.json"), "--out", str(out), "--checkout", str(tmp_path)]
    assert PP.main(argv) == 0 and json.loads(out.read_text())["counts"] == {"X_a": {"wins": 2, "finished": 3}}
    assert "banked plat__r7111.json: 2/3" in capsys.readouterr().out and not out.with_suffix(".tmp").exists()
    before = out.read_bytes()
    assert PP.main(argv) == PP.EXIT_REFUSED == P.PLAYER_REFUSED_RC and out.read_bytes() == before
    tree = ast.parse(Path(PP.__file__).read_text())
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert {m for m in mods if m.split(".")[0] in ("main", "utils", "agents")} == {"agents.training"}


# ------------------------------------------------------------------------------------------- the plan's records
def _responder(models: Path, seed: int, t: F.Final, **edit) -> Path:
    d = models / P.run_name(seed)
    d.mkdir(parents=True, exist_ok=True)
    (d / "final_model.zip").write_bytes(f"responder {seed}".encode())
    if not edit.get("final", True):
        (d / "final_model.zip").unlink()
    md = {"num_timesteps": edit.get("step", t.step + P.L_RESP + 19_000), "ent_coef": edit.get("ent", 0.05),
          "magnet_kl_coef": edit.get("magnet", 0.0), "cli_args": {"seed": edit.get("cli_seed", seed)},
          "pin_history": [{"git_hash": p} for p in edit.get("pins", [P.PIN])],
          "lineage": {"role": "exploiter", "fork_parent": {"sha256": edit.get("parent", t.sha256)}}}
    (d / "metadata.json").write_text(json.dumps(md))
    rec = {"target_file": edit.get("target_file", t.path), "fork_step": edit.get("fork_step", t.step),
           "archetype": edit.get("archetype", "offense"), "teams": edit.get("teams", P.banked()["team_paths"])}
    (d / "exploiter_record.json").write_text(json.dumps(rec))
    return d


def _read_exploiter(ref, _teamsets):
    f = Path(ref) / "exploiter_record.json"
    if not f.is_file():
        raise E.RunReadError(f"{Path(ref).name}: no record")
    return SimpleNamespace(run=Path(ref).name, membership="exact", **json.loads(f.read_text()))


@pytest.fixture()
def probe_world(world, monkeypatch):  # noqa: F811
    """Round 2 archived, lineage 1's copy the target, its two responders finished: ``(models, root, target)``."""
    models, root = world
    train_round(models, 2, pins=[F.TRAIN_PIN] * 2)
    F.archive(2, root=root, models_dir=models)
    monkeypatch.setattr(E, "read_exploiter", _read_exploiter)
    monkeypatch.setattr(E, "load_teamsets", lambda path=None: {})
    t = F.load_final(2, P.LINEAGE, root=root, models_dir=models)
    for s in P.SEEDS:
        _responder(models, s, t)
    return models, root, t


def test_the_plan_names_the_archive_copy_the_two_finished_responders_and_x71s_teams(probe_world):
    models, root, t = probe_world
    p = P.plan(2, archive_root=root, models_dir=models)
    assert t.source == "archive" and t.path == str(root / "r2" / "rb_x72_swapplat_s7341" / "final_model.zip")
    assert (p["target"], p["target_step"], p["target_sha256"], p["round"]) == (t.path, t.step, t.sha256, 2)
    assert p["cells"]["k0"] == {"path": t.path, "sha256": t.sha256, "games_per_team": 1600}
    for s in P.SEEDS:
        f = models / f"rb_x35_x71_plat_s{s}" / "final_model.zip"
        assert p["cells"][f"r{s}"] == {"path": str(f), "sha256": sha256_file(f), "games_per_team": 800}
    b = P.banked()
    assert (p["teams"], p["team_paths"], p["play_commit"], p["rehearsal"]) == (b["teams"], b["team_paths"], COMMIT,
                                                                               False)
    w = P.want(p, "r7112")
    assert w["schema"] == "x72_probe_chunk_v1" and (w["seed"], w["regime"]) == (20261020, "greedy-vs-greedy")
    assert set(w) >= {"schema", "target_tag", "cell", "path", "sha256", "target", "target_sha256", "teams", "seed",
                      "games_per_team", "regime"}                      # X71's chunk identity, every field
    assert sum(P.want(p, c)["games_per_team"] * 5 for c in P.CELLS) == 16_000


@pytest.mark.parametrize("edit, why", [
    ({"final": False}, "no final_model.zip"),
    ({"step": 5_000_000}, r"not \+5,000,000"),
    ({"target_file": "/elsewhere/final_model.zip"}, "recorded target '/elsewhere/final_model.zip' is not"),
    ({"fork_step": 300_061_272}, "forked at 300061272, not at the target's step"),
    ({"archetype": "stall"}, "trainee teams are 'stall'"),
    ({"ent": 0.0}, "recorded ent_coef 0.0, not the typed 0.05"),
    ({"ent": None}, "recorded ent_coef None"),
    ({"magnet": 0.05}, "recorded magnet_kl_coef 0.05, not 0"),
    ({"cli_seed": 7111}, "recorded seed 7111, not 7112"),
    ({"pins": [P.PIN, "0" * 40]}, "trained at pin"),
    ({"pins": []}, "trained at pin"),
    ({"parent": "0" * 64}, "forked from a file with sha256 000000000000"),
    ({"teams": ["data/teams/sample/other.txt"] * 5}, "are not the files X71's cells were played on"),
])
def test_a_responder_that_is_not_the_registered_one_is_refused_as_x71s_reader_refuses_it(probe_world, reader, edit,
                                                                                         why):
    models, root, t = probe_world
    rd = _responder(models, 7112, t, **edit)
    with pytest.raises(P.ProbeRefused, match=why) as exc:
        P.plan(2, archive_root=root, models_dir=models)
    assert exc.value.code == P.EXIT_REFUSED
    mine = P.check_responder(rd, t, 7112, _read_exploiter(str(rd), {}))
    frozen = reader.check_responder(rd, Path(t.path), t.step, 7112, _read_exploiter(str(rd), {}))
    assert [m for m in mine if "forked from a file" not in m] == frozen
    assert bool(mine) != ("teams" in edit)        # the team list is the plan's check, not the record's


def test_the_plan_refuses_round_0_a_missing_responder_and_a_copy_that_does_not_verify(probe_world):
    models, root, t = probe_world
    with pytest.raises(P.ProbeRefused, match="when the read round is 0 no probe is played"):
        P.plan(0, archive_root=root, models_dir=models)
    with pytest.raises(P.ProbeRefused, match="the round is not archived"):
        P.plan(3, archive_root=root, models_dir=models)
    shutil.rmtree(models / "rb_x35_x71_plat_s7111")
    with pytest.raises(P.ProbeRefused, match="rb_x35_x71_plat_s7111: no such run"):
        P.plan(2, archive_root=root, models_dir=models)
    Path(t.path).write_bytes(b"another file")
    with pytest.raises(P.ProbeRefused, match="no longer match"):
        P.plan(2, archive_root=root, models_dir=models)
    for bad in (0, 800, 4000):
        with pytest.raises(P.ProbeRefused, match="--rehearsal-games"):
            P.plan(2, archive_root=root, models_dir=models, rehearsal_games=bad)


@pytest.mark.parametrize("error", [E.RunReadError, E.SeriesError])
def test_a_responder_whose_record_does_not_read_is_a_named_refusal_never_a_traceback(probe_world, monkeypatch,
                                                                                      error):
    """Found by execution: the meter's reader raises ``SeriesError`` when the run's evals name another target than
    its lineage does (a stand-in's did); the plan must refuse by name, exit 3, whatever the reader's error is."""
    models, root, _t = probe_world

    def unreadable(ref, _teamsets):
        raise error(f"{Path(ref).name}: no 'ext_rb_x72_swapplat_s7341' cycle")

    monkeypatch.setattr(E, "read_exploiter", unreadable)
    with pytest.raises(P.ProbeRefused, match="rb_x35_x71_plat_s7111: no 'ext_rb_x72_swapplat_s7341' cycle") as exc:
        P.plan(2, archive_root=root, models_dir=models)
    assert exc.value.code == P.EXIT_REFUSED and "rb_x35_x71_plat_s7112" in str(exc.value)
    assert P.main(["plan", "--round", "2", "--archive-root", str(root), "--models-dir", str(models)]) == 3


def test_the_record_check_is_x71s_g6_on_a_responder_that_may_still_be_training(probe_world):
    models, _root, t = probe_world
    assert P.record_check(7111, models_dir=models) == (0, [])
    (models / "rb_x35_x71_plat_s7111" / "final_model.zip").unlink()          # a running responder has no final
    assert P.record_check(7111, models_dir=models) == (0, [])
    _responder(models, 7112, t, ent=0.0, magnet=0.05)
    code, probs = P.record_check(7112, models_dir=models)
    assert code == 3 and len(probs) == 2 and "ent_coef 0.0" in probs[0] and "magnet_kl_coef 0.05" in probs[1]
    (models / "rb_x35_x71_plat_s7112" / "metadata.json").unlink()
    assert P.record_check(7112, models_dir=models)[0] == 5
    with pytest.raises(P.ProbeRefused):
        P.record_check(7341, models_dir=models)


# ------------------------------------------------------------------------------------------------- the driver
class _Runner:
    """``subprocess.run`` recorded: git answers for the play checkout, and the player's command banks a chunk whose
    win rate is ``rates[cell]`` (``rc[cell]`` is its exit code instead; ``lost[cell]`` unfinished games a team)."""

    def __init__(self, head=COMMIT, dirty="", rates=None, rc=None, lost=None, write=True):
        self.head, self.dirty, self.rates, self.rc, self.lost, self.write = head, dirty, rates or {}, rc or {}, \
            lost or {}, write
        self.played = []

    def __call__(self, argv, **kw):
        if argv[0] == "git":
            return SimpleNamespace(returncode=0, stdout=(self.head + "\n") if "rev-parse" in argv else self.dirty)
        w = json.loads(Path(argv[argv.index("--want") + 1]).read_text())
        self.played.append((w["cell"], argv, kw))
        rc = self.rc.get(w["cell"], 0)
        if rc == 0 and self.write:
            n = w["games_per_team"]
            fin = n - self.lost.get(w["cell"], 0)
            wins = round(self.rates.get(w["cell"], 0.475 if w["cell"] == "k0" else 0.5316) * fin)
            counts = {t[1]: {"wins": wins, "losses": fin - wins, "finished": fin, "attempted": n, "ties": 0}
                      for t in w["teams"]}
            Path(argv[argv.index("--out") + 1]).write_text(json.dumps(dict(
                w, counts=counts, transport="rust_eval", game_rows={})))
        return SimpleNamespace(returncode=rc)


@pytest.fixture()
def venue(probe_world, tmp_path):
    models, root, _t = probe_world
    play = tmp_path / "play"
    (play / "src").mkdir(parents=True)

    def drive(run, work="work", **kw):
        return P.drive(2, play, tmp_path / work, archive_root=root, models_dir=models, run=run, py="PY",
                       cpus="5-7", **kw)

    return SimpleNamespace(drive=drive, play=play, work=tmp_path / "work", tmp=tmp_path, models=models, root=root)


def test_drive_plays_the_three_cells_in_rule_5bs_shape_and_writes_the_verdict_once(venue):
    run = _Runner()
    assert venue.drive(run) == 0
    assert [c for c, _a, _k in run.played] == ["k0", "r7111", "r7112"]
    for cell, argv, kw in run.played:
        out = venue.work / "ledger" / f"plat__{cell}.json"
        assert argv == ["nice", "-n", "19", "taskset", "-c", "5-7", str(repo_path("scripts", "ops", "mem_cap.sh")),
                        "--name", f"x72_probe_plat__{cell}", "--max-run-s", "7200", "8", "PY", "-P",
                        str(Path(P.__file__).with_name("probe_play.py")), "--want",
                        str(venue.work / "ledger" / f"want_{cell}.json"), "--out", str(out), "--checkout",
                        str(venue.play)]
        assert "timeout" not in argv and kw["cwd"] == str(venue.play)
        env = kw["env"]
        assert (env["PYTHONPATH"], env["CUDA_VISIBLE_DEVICES"], env["OMP_NUM_THREADS"]) == (
            str(venue.play / "src"), "", "1") and "GEN3AI_TEST_ALLOW_GPU" not in env
    doc = json.loads((venue.work / P.VERDICT).read_text())
    b = P.banked()
    assert doc["schema"] == "x72_probe_verdict_v1" and doc["round"] == 2 and doc["prereg"] == F.PREREG
    assert {t: doc["cells"][t] for t in P.BANKED} == b["cells"] and sorted(doc["cells"]) == sorted(P.TARGETS)
    assert doc["cells"]["plat"]["k0"] == {"wins": 3800, "finished": 8000, "attempted": 8000, "ties": 0}
    assert doc["cells"]["plat"]["r7111"]["attempted"] == 4000 == doc["cells"]["plat"]["r7112"]["attempted"]
    assert doc["cls"] == {"plat": "NOT-MORE-BY-BAR"} and round(doc["mult"], 3) == 2.571
    assert same(doc, dict(P.classify(doc["cells"]), **{k: doc[k] for k in (
        "schema", "prereg", "round", "cells", "reported", "by_team", "transport", "regime", "plan", "banked")}))
    assert doc["banked"] == {"record": P.X71_RECORD, "play_commit": COMMIT, "chunks": b["chunks"]}
    assert doc["reported"]["what"].endswith("NO class") and doc["plan"]["target_sha256"] == doc["plan"]["cells"][
        "k0"]["sha256"]
    # a second drive plays nothing, writes nothing, and answers with the banked verdict
    before = (venue.work / P.VERDICT).read_bytes()
    again = _Runner(rates={"r7111": 0.9})
    assert venue.drive(again) == 0 and again.played == [] and (venue.work / P.VERDICT).read_bytes() == before
    assert "DONE before" in (venue.work / "run.log").read_text()


@pytest.mark.parametrize("rates, lost, cls, code", [
    ({"r7111": 0.60, "r7112": 0.60}, {}, "MORE-EXPLOITABLE", 0),
    ({"r7111": 0.45, "r7112": 0.45}, {}, "LESS-EXPLOITABLE", 0),
    ({"r7111": 0.557, "r7112": 0.557}, {}, "UNRESOLVED", 0),
    ({}, {"k0": 401}, "INCONCLUSIVE", 6),              # > 25 % of the target's own games did not finish
])
def test_drive_returns_the_class_the_cells_earn(venue, rates, lost, cls, code):
    assert venue.drive(_Runner(rates=rates, lost=lost)) == code
    doc = json.loads((venue.work / P.VERDICT).read_text())
    assert doc["cls"] == {"plat": cls} and (doc["reported"] is None) == (cls == "INCONCLUSIVE")
    assert venue.drive(_Runner()) == code                     # the banked verdict's code, nothing replayed


def test_drive_resumes_after_a_failed_cell_and_never_replays_a_banked_one(venue):
    first = _Runner(rc={"r7111": 1})
    assert venue.drive(first) == 4 and [c for c, _a, _k in first.played] == ["k0", "r7111"]
    assert not (venue.work / P.VERDICT).exists()
    k0 = (venue.work / "ledger" / "plat__k0.json").read_bytes()
    second = _Runner(rates={"k0": 0.9})
    assert venue.drive(second) == 0 and [c for c, _a, _k in second.played] == ["r7111", "r7112"]
    assert (venue.work / "ledger" / "plat__k0.json").read_bytes() == k0


@pytest.mark.parametrize("runner, code", [
    (dict(rc={"k0": 89}), 8), (dict(rc={"k0": 3}), 3), (dict(rc={"k0": 137}), 4), (dict(write=False), 8)])
def test_a_cell_that_fails_refuses_or_banks_nothing_stops_the_read(venue, runner, code):
    run = _Runner(**runner)
    assert venue.drive(run) == code and len(run.played) == 1 and not (venue.work / P.VERDICT).exists()


def test_drive_holds_at_any_commit_but_x71s_and_on_a_dirty_play_checkout(venue):
    for run in (_Runner(head="333a73ddda7829b9ccdf9e17ef04624eb40edeba"), _Runner(dirty=" M src/agents/x.py\n")):
        assert venue.drive(run) == 5 and run.played == [] and not (venue.work / "ledger").exists()
    log = (venue.work / "run.log").read_text()
    assert f"X71's banked cells were played at {COMMIT}" in log and "tracked changes under src/ or data/" in log


def test_drive_holds_on_a_work_dir_inside_a_checkout_or_models_and_on_another_plans_dir(venue):
    for inside in (venue.models / "probe", venue.play / "probe", repo_path("x72_probe_work_never_made")):
        run = _Runner()
        assert P.drive(2, venue.play, inside, archive_root=venue.root, models_dir=venue.models, run=run, py="PY",
                       cpus="5-7") == 5
        assert run.played == [] and not inside.exists()
    assert venue.drive(_Runner(rc={"r7112": 1})) == 4
    resp = venue.models / "rb_x35_x71_plat_s7112" / "final_model.zip"
    resp.write_bytes(b"another responder file")
    run = _Runner()
    assert venue.drive(run) == 5 and run.played == []                     # the plan changed under the work dir
    assert "is another plan" in (venue.work / "run.log").read_text()


def test_a_chunk_banked_for_another_read_is_refused_never_replayed_over(venue):
    assert venue.drive(_Runner(rc={"r7111": 1})) == 4
    f = venue.work / "ledger" / "plat__k0.json"
    chunk = json.loads(f.read_text())
    chunk["seed"] = 1
    f.write_text(json.dumps(chunk))
    run = _Runner()
    assert venue.drive(run) == 3 and run.played == [] and json.loads(f.read_text())["seed"] == 1


def test_a_cell_played_on_another_transport_or_with_another_count_is_refused(venue, probe_world):
    models, root, _t = probe_world
    assert venue.drive(_Runner(rc={"r7112": 1})) == 4
    p = P.plan(2, archive_root=root, models_dir=models)
    f = venue.work / "ledger" / "plat__r7112.json"
    good = json.loads((venue.work / "ledger" / "plat__r7111.json").read_text())
    body = dict(P.want(p, "r7112"), counts=good["counts"], transport="node", game_rows={})
    f.write_text(json.dumps(body))
    with pytest.raises(P.ProbeRefused, match="played on 'node'"):
        P.verdict(p, venue.work / "ledger")
    key = sorted(good["counts"])[0]
    short = dict(good["counts"], **{key: dict(good["counts"][key], attempted=799)})
    f.write_text(json.dumps(dict(body, counts=short, transport="rust_eval")))
    with pytest.raises(P.ProbeRefused, match="3999 games attempted, planned 4000"):
        P.verdict(p, venue.work / "ledger")
    f.unlink()
    with pytest.raises(P.ProbeRefused, match="plat__r7112.json is not banked"):
        P.verdict(p, venue.work / "ledger")


def test_a_rehearsal_never_yields_a_verdict_and_its_cells_are_never_collected(venue, probe_world):
    models, root, _t = probe_world
    run = _Runner()
    assert venue.drive(run, rehearsal_games=2) == 0
    assert [(c, a[a.index("--max-s") + 1], "--max-run-s" in a) for c, a, _k in run.played] == [
        ("k0", "600", False), ("r7111", "600", False), ("r7112", "600", False)]
    assert all(str(venue.work / "rehearsal_ledger") in a[a.index("--out") + 1] for _c, a, _k in run.played)
    got = json.loads((venue.work / "rehearsal.json").read_text())
    assert "cls" not in got and got["cells"]["k0"]["attempted"] == 20 and got["cells"]["r7111"]["attempted"] == 10
    assert not (venue.work / P.VERDICT).exists() and not (venue.work / "ledger").exists()
    chunk = json.loads((venue.work / "rehearsal_ledger" / "plat__k0.json").read_text())
    assert chunk["schema"] == "x72_probe_rehearsal_chunk_v1" != P.CHUNK_SCHEMA
    rehearsal = P.plan(2, archive_root=root, models_dir=models, rehearsal_games=2)
    with pytest.raises(P.ProbeRefused, match="a rehearsal has no verdict"):
        P.verdict(rehearsal, venue.work / "rehearsal_ledger")
    with pytest.raises(P.ProbeRefused, match="banked for another read"):          # nor read as the real cells
        P.verdict(P.plan(2, archive_root=root, models_dir=models), venue.work / "rehearsal_ledger")
    real = _Runner()
    assert venue.drive(real) == 0 and len(real.played) == 3                      # the read still plays all three


def test_one_lane_a_work_dir(venue):
    import fcntl

    venue.work.mkdir()
    with open(venue.work / "lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = _Runner()
        assert venue.drive(run) == 5 and run.played == []
    assert venue.drive(_Runner()) == 0


# ------------------------------------------------------------------------------------------------------ the CLI
def test_the_cli_prints_a_responders_argv_and_judges_its_record(world, capsys, monkeypatch, tmp_path):  # noqa: F811
    models, root = world
    train_round(models, 2, pins=[F.TRAIN_PIN] * 2)
    F.archive(2, root=root, models_dir=models)
    t = F.load_final(2, P.LINEAGE, root=root, models_dir=models)
    _responder(models, 7111, t)
    monkeypatch.chdir(tmp_path)                                    # the CLI finds the committed teamsets itself
    common = ["--archive-root", str(root), "--models-dir", str(models)]
    assert P.main(["argv", "--round", "2", "--seed", "7112", *common]) == 0
    argv = capsys.readouterr().out.split()
    assert argv == P.responder_argv(7112, t.path, t.step, teams=",".join(P.banked()["team_paths"]))
    assert argv[argv.index("--exploiter") + 1] == t.path == argv[argv.index("--model") + 1]
    assert P.main(["argv", "--round", "0", "--seed", "7111", *common]) == 3
    assert P.main(["record-check", "--seed", "7111", "--models-dir", str(models)]) == 0
    assert "the record holds the typed recipe (G6)" in capsys.readouterr().out
    assert P.main(["record-check", "--seed", "7111", "--models-dir", str(tmp_path)]) == 5
    with pytest.raises(SystemExit):
        P.main(["argv", "--round", "2", "--seed", "7341", *common])
