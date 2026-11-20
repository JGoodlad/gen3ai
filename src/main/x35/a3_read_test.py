"""X35-A3's READ (PREREG ``x35_a3_main_exploiters_2026-11-16`` §5, §9, §11.1 items 1-3): the six reader kit arms'
committed files (argvs, transcripts, the draft chain items, the recorded plan and kit checks), the read's driver
``read_a3.py`` on synthetic run directories with a recorded player, the registered classes at their boundaries
THROUGH the driver, and the four head-to-head crosses' plan and reader on fixture cells.

Nothing here launches, plays a game or touches ``models/``. A boundary case sits a whole game (or half a point)
away from its boundary, never within a rounding error of it; the rule's own tie handling is ``a3_rule``'s
self-check (``a3_test.py``).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from main.x35 import batch as B
from main.x35 import ge
from main.x35 import kit as K

from utils.paths import repo_path

DIR = repo_path("designs/research_state/measurements/x35_a3_main_exploiters_2026-11-16")
PIN = "f4b25238def57900c0305bb2a3a5a8f9edaef285"
SEEDS = (3501, 3502, 3503)
#: reader arm -> (the side's key in the read, the target run's format, the reader run's)
ARMS = {"ge_a3": ("a3", "rb_x35_a3mx_s{}", "rb_x35_ge_a3mx_s{}"), "ge_c0": ("c0", "rb_x35_c0_s{}", "rb_x35_ge_c0_s{}")}
#: the three C0 finals' steps as their records hold them (the banked argvs' ``--steps`` minus 5M)
C0_STEPS = {3501: 300_061_940, 3502: 300_061_805, 3503: 300_061_491}
VOID = ["[STABLE]", "with the heuristic bots MIXED IN", "target pilots ITS OWN pinned team", "[Exploiter] FATAL",
        "[ModelVersion] FATAL"]


def _banked_models_root() -> str:
    """The models directory the banked argvs were written against, READ from the first one and never typed here
    (``utils/home_path_literal_gate_test.py``)."""
    argv = (DIR / "argv_ge_c0_s3501.txt").read_text().split()
    root, run, name = argv[argv.index("--model") + 1].rsplit("/", 2)
    assert (run, name) == ("rb_x35_c0_s3501", "final_model.zip") and root.startswith("/") and root.endswith("/models")
    return root


MODELS = _banked_models_root()


@pytest.fixture(scope="module")
def reader():
    """``read_a3.py`` (it imports ``a3_rule`` from its own directory as ``reader.R``)."""
    before = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("x35_a3_frozen_read", DIR / "read_a3.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path[:] = before
        sys.modules.pop("a3_rule", None)


# ------------------------------------------------------------------------------ the six readers' committed files
def test_the_six_readers_are_one_per_final_and_the_driver_names_the_kits_runs(reader):
    assert reader.R.SEEDS == tuple(str(s) for s in SEEDS) and reader.PIN == PIN
    for arm, (side, t_fmt, r_fmt) in ARMS.items():
        assert ge.target_seeds(arm) == SEEDS
        for s in SEEDS:
            assert reader.SIDES[side] == (t_fmt, r_fmt)
            assert ge.target_run(arm, s) == t_fmt.format(s) and K.run_name(arm, s) == r_fmt.format(s)
    assert reader.READER_SEED == ge.READER_SEED == 3531 and reader.L_READER == ge.L_READER == 5_000_000
    assert str(reader.FORK_LR) == ge.READER_FORK_LR
    assert [u for _, _, u in reader.units()] == [f"{side}_s{s}" for side in ("a3", "c0") for s in SEEDS]
    assert (reader.R.GAMES, reader.N_TEAMS, reader.REGIME) == (4_000, 5, "greedy-vs-greedy")


@pytest.mark.parametrize("seed", SEEDS)
def test_the_kit_builds_the_banked_reader_argvs_token_for_token(seed):
    """``ge_c0`` on the real final's step; ``ge_a3`` on the declared placeholder (the arm's registered floor)."""
    real = {"path": f"{MODELS}/rb_x35_c0_s{seed}/final_model.zip", "step": C0_STEPS[seed]}
    assert " ".join(ge.build_reader_argv("ge_c0", seed, real)) == \
        (DIR / f"argv_ge_c0_s{seed}.txt").read_text().strip()
    ph = ge.placeholder_target("ge_a3", seed, MODELS, 300_000_170)
    argv = ge.build_reader_argv("ge_a3", seed, ph)
    assert " ".join(argv) == (DIR / f"argv_ge_a3_PLACEHOLDER_s{seed}.txt").read_text().strip()
    assert argv[argv.index("--model") + 1] == argv[argv.index("--exploiter") + 1] == ph["path"]
    assert argv[argv.index("--steps") + 1] == "305000170" and argv[argv.index("--seed") + 1] == "3531"
    assert "--no-self-play" in argv and "--fork-lr-freeze" in argv and "--stable-opponents" not in argv
    assert argv[argv.index("--trainee-teams") + 1] == ge.offense_teams()


@pytest.mark.parametrize("seed", SEEDS)
def test_the_transcripts_are_these_argvs_and_stop_only_where_they_must(seed):
    """A C0 reader's argv passes the PIN's parser and its dry run stops only at the registering shell's two
    refusals (the hidden card, nice 19). An A3 reader's argv cannot be checked before its target exists: its one
    refusal is the missing checkpoint."""
    check = (DIR / f"checkargs_ge_c0_s{seed}.txt").read_text()
    dry = (DIR / f"dryrun_ge_c0_s{seed}.txt").read_text()
    n_flags = sum(t.startswith("--") for t in (DIR / f"argv_ge_c0_s{seed}.txt").read_text().split())
    assert check.splitlines()[0] == f"checked {n_flags} flags from --argv" and n_flags == 12
    assert "argv validated against PINNED parser @f4b25238" in check and "this command still launches" in check
    assert f"resolved against the FORK PARENT: {MODELS}/rb_x35_c0_s{seed}/model_config.json" in check
    assert "REFUSED" not in check
    assert [ln.split(":")[0].strip() for ln in dry.splitlines() if "REFUSED" in ln] == ["cuda devices",
                                                                                         "inherited nice"]
    assert f"role        : FORK of {MODELS}/rb_x35_c0_s{seed}" in dry
    assert f"--steps {C0_STEPS[seed] + 5_000_000:,} vs checkpoint at {C0_STEPS[seed]:,}" in dry
    assert "→ +5,000,000 steps" in dry and "11.22 GiB REQUIRED" in dry
    assert f"{MODELS}/rb_x35_ge_c0_s{seed}   [would be created]" in dry
    ph = (DIR / f"checkargs_ge_a3_PLACEHOLDER_s{seed}.txt").read_text()
    assert ph.splitlines()[0] == "checked 12 flags from --argv" and "the PINNED commit f4b25238's" in ph
    refused = [ln for ln in ph.splitlines() if "REFUSED" in ln]
    assert len(refused) == 1 and "checkpoint file not found" in refused[0]
    assert f"rb_x35_a3mx_s{seed}/final_model.zip" in refused[0]


def test_the_draft_reader_items_are_the_kits_runs_at_the_side_pin_with_the_registered_banner():
    doc = json.loads((DIR / "chain_items_a3_readers.json").read_text())
    assert doc["x_pin"]["sha"] == PIN and doc["x_backend"]["kind"] == "x35"
    assert "GO" not in doc and "go" not in doc and "NO GO" in doc["x_what"]
    items = doc["items"]
    assert [(i["arm"], i["seed"]) for i in items] == [(a, s) for a in ("ge_c0", "ge_a3") for s in SEEDS]
    for it in items:
        target = ge.target_run(it["arm"], it["seed"])
        assert it["kind"] == "arm" and it["pin"] == PIN and it["run"] == K.run_name(it["arm"], it["seed"])
        assert it["thin"] == "planner" and it["flags"] == ["--thinning"] and it["disk_need_gib"] == 12.0
        assert it["on_void"] == "stop" and it["on_throughput_fail"] == "page"
        assert it["throughput"] == {"min_steps_per_s": 1000, "within_min": 60, "first_line_within_min": 90}
        assert it["go_by"].startswith("<orchestrator, date>") and "<sha>" in it["go_by"]
        b = it["banner"]
        assert b["expect"] == (f"[EXPLOITER] training vs ext_{target} as the SOLE opponent every episode "
                               "(temp 1; no self-play/pool/bots)")
        assert b["void"] == VOID and b["within_min"] == 30 and not any(v in b["expect"] for v in b["void"])
        assert "read_a3.py" in it["x_registration"] and "[STABLE]" in it["x_registration"]


def test_the_recorded_reader_checks_and_plan_say_what_the_prereg_says():
    checks = (DIR / "kit_checks_a3_readers.txt").read_text()
    for arm, (_side, t_fmt, _r) in ARMS.items():
        for s in SEEDS:
            assert (f"stage-check {arm} s{s}: ALLOWED: {arm} s{s}: a reader of X35-A3's deciding read, allowed by "
                    f"the A3 registration (PREREG §5.1, §11.1 item 1; reads {t_fmt.format(s)})") in checks
        assert f"REFUSED: seed 3504 is not registered for {arm} ([3501, 3502, 3503])" in checks
    assert checks.count(": ALLOWED: ") == 6
    assert "REFUSED: seed 3502 is not registered for ge_a1 ([3501])" in checks
    assert "stage-check ge_a1 s3501: REFUSED: G-E reader of a1" in checks      # a batch reader keeps its GAIN rule
    assert "REFUSED: rb_x35_a3mx_s3501: no final_model.zip under" in checks    # the LAUNCH's describe stays strict
    assert '"mode": "TARGET-PLACEHOLDER"' in checks
    plan = (DIR / "arm_chain_plan_a3_readers.txt").read_text()
    for s in SEEDS:
        assert (f"arm:rb_x35_ge_c0_s{s}: the kit agrees — FORK of {MODELS}/rb_x35_c0_s{s}, +5,000,000 steps, "
                f"--steps {C0_STEPS[s] + 5_000_000:,}\n") in plan
        assert (f"arm:rb_x35_ge_a3mx_s{s}: the kit agrees — FORK of {MODELS}/rb_x35_a3mx_s{s}, +5,000,000 steps, "
                "--steps 305,000,170; its target has no final yet: the argv is built from a declared PLACEHOLDER") \
            in plan
    assert plan.index("\nPLAN OK\nrc=0\n") < plan.index("PLAN REFUSED (2 problem(s))")
    assert "REFUSED: seed 3504 is not registered for ge_a3 ([3501, 3502, 3503])" in plan


def test_the_reads_build_files_are_committed_and_no_go_file_is():
    for name in ("read_a3.py", "cross/plan_cross.py", "cross/read_cross.py", "cross/drive_cross.sh",
                 "cross/banked_files_check.txt", "chain_items_a3_readers.json", "kit_checks_a3_readers.txt",
                 "arm_chain_plan_a3_readers.txt"):
        assert (DIR / name).is_file(), name
    assert not [p for p in DIR.rglob("*") if p.name.upper().startswith("GO")]
    assert not list(DIR.rglob("verdict*.json")) and not list(DIR.rglob("gf.json"))      # no read was played
    banked = (DIR / "cross" / "banked_files_check.txt").read_text()
    assert "banked files OK: c0 [3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508]; pstar ['pstar']; R [" in banked
    assert "(16 files pass every guard)" in banked


# -------------------------------------------------------------------------------- the driver on synthetic runs
TEAMS = ge.offense_teams().split(",")
T_STEP = 300_061_000


def _write_run(root: Path, name: str, *, steps: int, cli=None, pin=PIN) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "final_model.zip").write_bytes(name.encode())
    (d / "metadata.json").write_text(json.dumps({"num_timesteps": steps, "cli_args": cli or {},
                                                 "pin_history": [{"git_hash": pin}]}))
    return d


@pytest.fixture()
def world(reader, tmp_path, monkeypatch):
    """A synthetic ``models/`` with the six targets and their six clean readers; ``read_exploiter`` reads each
    reader's record from a table the test may edit."""
    from agents.training import best_response_gap as E

    models = tmp_path / "models"
    records, banked = {}, {}
    for side, seed, _unit in reader.units():
        t_fmt, r_fmt = reader.SIDES[side]
        t = _write_run(models, t_fmt.format(seed), steps=T_STEP + int(seed))
        if side == "c0":
            banked[seed] = hashlib.sha256((t / "final_model.zip").read_bytes()).hexdigest()
        name = r_fmt.format(seed)
        _write_run(models, name, steps=T_STEP + int(seed) + 5_000_000 + 400,
                   cli={"seed": 3531, "fork_lr": 5.6e-05, "fork_lr_freeze": True, "stable_opponents": None})
        # the launcher records the target's step as a STRING (read from a real exploiter's record at this pin)
        records[name] = dict(target_file=str(t / "final_model.zip"), target_step=str(T_STEP + int(seed)),
                             fork_step=T_STEP + int(seed), archetype="offense", membership="exact",
                             teams=list(TEAMS), pooled=(300, 512), end=(160, 256))

    def read_exploiter(ref, _teamsets):
        r = records[Path(ref).name]
        end = SimpleNamespace(wins=r["end"][0], games=r["end"][1], step=T_STEP + 5_000_000)
        return SimpleNamespace(run=Path(ref).name, target_file=r["target_file"], target_step=r["target_step"],
                               fork_step=r["fork_step"], archetype=r["archetype"], membership=r["membership"],
                               teams=r["teams"], pooled=lambda: r["pooled"], endpoint=lambda: end)

    monkeypatch.setattr(E, "read_exploiter", read_exploiter)
    monkeypatch.setattr(E, "load_teamsets", lambda path=None: {})
    monkeypatch.setattr(reader, "banked_c0_sha256", lambda: dict(banked))
    return SimpleNamespace(models=models, records=records, ledger=str(tmp_path / "ledger"), tmp=tmp_path)


def _player(calls, wins=None, finished=None):
    """A recorded player. ``wins[label]`` is the cell's TOTAL wins over its five teams (default half), spread as
    evenly as whole games allow; ``finished[label]`` the games a team finishes (default all)."""
    from agents.training import untaught_meter as um

    def play(ref_path, target_path, teams, games_per_team, *, label):
        calls.append((label, ref_path, target_path, games_per_team, [t.key for t in teams]))
        fin = (finished or {}).get(label, games_per_team)
        total = (wins or {}).get(label, fin * len(teams) // 2)
        got = {}
        for i, t in enumerate(teams):
            w = total // len(teams) + (1 if i < total % len(teams) else 0)
            got[t.key] = um.Cell(wins=w, losses=fin - w, finished=fin, attempted=games_per_team,
                                 transport="rust_eval")
        return got, {t.key: [] for t in teams}, "rust_eval"
    return play


def _wins(a3: int, c0: int, k0_a3: int = 2_000, k0_c0: int = 2_000) -> dict:
    out = {}
    for s in SEEDS:
        out.update({f"a3_s{s}_r": a3, f"c0_s{s}_r": c0, f"a3_s{s}_k0": k0_a3, f"c0_s{s}_k0": k0_c0})
    return out


def _guard(tmp: Path, cross: str, value=None) -> str:
    """A guard file as ``cross/read_cross.py`` writes it: a uniform G-F matrix / R1 score (pp), or, with no value,
    the file of a cross that could not be read."""
    doc = {"cross": cross, "inconclusive": [] if value is not None else ["missing cell aaaa vs bbbb"]}
    if value is not None and cross == "gf":
        doc["gf_h_pp"] = [[value] * 8] * 3
    if value is not None and cross == "r1":
        doc["r1_on"] = {str(s): value for s in SEEDS}
    f = tmp / f"{cross}_{value}.json"
    f.write_text(json.dumps(doc))
    return str(f)


def _played(reader, world, wins, finished=None) -> None:
    reader.play(world.ledger, str(world.models), say=lambda _s: None, player=_player([], wins, finished))


def test_the_plan_admits_the_six_registered_readers_and_names_twelve_cells(reader, world):
    p = reader.plan(str(world.models))
    assert sorted(p) == sorted(u for _, _, u in reader.units())
    for unit, t in p.items():
        side, seed = unit.split("_s")
        assert t["target"].endswith(f"{reader.SIDES[side][0].format(seed)}/final_model.zip")
        assert t["target_step"] == T_STEP + int(seed) and t["teams"] == TEAMS and len(t["target_sha256"]) == 64
        assert list(t["cells"]) == ["r", "k0"]
        assert t["cells"]["k0"] == {"path": t["target"], "games": 4_000, "sha256": t["target_sha256"]}
        assert t["cells"]["r"]["path"].endswith(f"{reader.SIDES[side][1].format(seed)}/final_model.zip")
        assert t["cells"]["r"]["games"] == 4_000
        assert t["cells"]["r"]["in_run"] == {"pooled": [300, 512], "endpoint": [160, 256, T_STEP + 5_000_000]}
    assert reader.main(["plan", "--models-dir", str(world.models)]) == 0


def _spoil(world, name, fn) -> None:
    f = world.models / name / "metadata.json"
    doc = json.loads(f.read_text())
    fn(doc)
    f.write_text(json.dumps(doc))


@pytest.mark.parametrize("how,needle", [
    ("short", "not +5,000,000"), ("seed", "recorded seed 3532, not 3531"), ("pin", "trained at pin(s)"),
    ("target", "is not"), ("tstep", "recorded target step"), ("fork", "forked at"), ("teams", "not the offense pin"),
    ("lr", "recorded fork rate"), ("thaw", "recorded fork rate"), ("stable", "carries --stable-opponents"),
    ("nofinal", "a reader is read FINISHED only"), ("gone", "no such run"), ("t_short", "not P* + 20M"),
    ("t_pin", "trained at pin(s)"), ("t_gone", "no final_model.zip under"), ("c0_other", "whose S is banked"),
    ("teams_differ", "differ from the first reader's")])
def test_the_plan_refuses_each_way_a_reader_or_a_target_can_be_another_file(reader, world, monkeypatch, how, needle):
    """PREREG §9's record guards: every single departure is named, on the A3 side's second unit."""
    name, target = "rb_x35_ge_a3mx_s3502", "rb_x35_a3mx_s3502"
    cli = lambda **kw: (lambda d: d["cli_args"].update(kw))  # noqa: E731
    if how == "short":
        _spoil(world, name, lambda d: d.update(num_timesteps=T_STEP + 3502 + 4_999_999))
    elif how == "seed":
        _spoil(world, name, cli(seed=3532))
    elif how == "pin":
        _spoil(world, name, lambda d: d["pin_history"].append({"git_hash": "9e97cad5709da21a1f9d4d8846096c87eb17b1d0"}))
    elif how == "target":
        world.records[name]["target_file"] = str(world.models / "rb_x35_a3mx_s3501" / "final_model.zip")
    elif how == "tstep":
        world.records[name]["target_step"] = str(T_STEP)
    elif how == "fork":
        world.records[name]["fork_step"] = 280_000_170
    elif how == "teams":
        world.records[name].update(archetype="balance")
    elif how == "lr":
        _spoil(world, name, cli(fork_lr=1e-4))
    elif how == "thaw":
        _spoil(world, name, cli(fork_lr_freeze=False))
    elif how == "stable":
        _spoil(world, name, cli(stable_opponents=["mx_s3511=/m/a.zip"]))
    elif how == "nofinal":
        (world.models / name / "final_model.zip").unlink()
    elif how == "gone":
        (world.models / name).rename(world.models / "elsewhere")
    elif how == "t_short":
        _spoil(world, target, lambda d: d.update(num_timesteps=300_000_169))
        world.records[name].update(target_step="300000169", fork_step=300_000_169)
    elif how == "t_pin":
        _spoil(world, target, lambda d: d["pin_history"][0].update(git_hash="0" * 40))
    elif how == "t_gone":
        (world.models / target / "final_model.zip").unlink()
    elif how == "c0_other":
        monkeypatch.setattr(reader, "banked_c0_sha256", lambda: {"3501": "0" * 64})
    elif how == "teams_differ":
        world.records[name]["teams"] = list(reversed(TEAMS))
    with pytest.raises(reader.A3ReadError, match=re.escape(needle)):
        reader.plan(str(world.models))
    assert reader.main(["plan", "--models-dir", str(world.models)]) == 2
    with pytest.raises(reader.A3ReadError):                        # and no game is played on it
        reader.play(world.ledger, str(world.models), player=_player([]))
    assert not Path(world.ledger).exists()


def test_the_real_banked_c0_shas_are_the_three_finals_g1_read(reader):
    """The C0 targets are the files whose S is R1's banked side: three sha256, read from G1's record."""
    got = reader.banked_c0_sha256()
    assert sorted(got) == ["3501", "3502", "3503"] and all(len(v) == 64 for v in got.values())
    assert len(set(got.values())) == 3


def test_play_banks_one_cell_at_a_time_the_deciding_cells_first_resumes_and_never_replays(reader, world):
    calls: list = []
    play = _player(calls)
    said: list = []
    res = reader.play(world.ledger, str(world.models), max_chunks=5, say=said.append, player=play)
    assert res == {"chunks": 12, "banked": 5, "new": 5, "partial": True}
    res = reader.play(world.ledger, str(world.models), say=said.append, player=play)
    assert res == {"chunks": 12, "banked": 12, "new": 7, "partial": False}
    units = [u for _, _, u in reader.units()]
    assert [c[0] for c in calls] == [f"{u}_r" for u in units] + [f"{u}_k0" for u in units]
    assert {c[3] for c in calls} == {800} and all(len(c[4]) == 5 for c in calls)
    for label, ref, target, _g, _teams in calls:
        side, seed = label.split("_")[0], label.split("_")[1][1:]
        t_fmt, r_fmt = reader.SIDES[side]
        assert Path(target).parent.name == t_fmt.format(seed)                  # the opponent is the unit's target
        assert Path(ref).parent.name == (r_fmt if label.endswith("_r") else t_fmt).format(seed)
    assert sorted(p.name for p in Path(world.ledger).iterdir()) == sorted(
        f"{u}__{c}.json" for u in units for c in ("r", "k0"))
    n = len(calls)
    assert reader.play(world.ledger, str(world.models), say=said.append, player=play)["new"] == 0
    assert len(calls) == n and len(said) == 12
    one = json.loads((Path(world.ledger) / "a3_s3501__r.json").read_text())
    assert (one["seed"], one["regime"], one["games_per_team"]) == (reader.READ_SEED, "greedy-vs-greedy", 800)
    assert reader.READ_SEED == 20261020                             # R0's reader's seed, the one X71's read uses


def test_a_cell_banked_for_another_file_is_refused_not_replayed(reader, world):
    calls: list = []
    play = _player(calls)
    reader.play(world.ledger, str(world.models), say=lambda _s: None, player=play)
    (world.models / "rb_x35_ge_c0_s3503" / "final_model.zip").write_bytes(b"another file")
    n = len(calls)
    with pytest.raises(reader.A3ReadError, match="banked for another read"):
        reader.play(world.ledger, str(world.models), say=lambda _s: None, player=play)
    assert len(calls) == n


def test_the_ledger_is_refused_inside_models(reader, world):
    with pytest.raises(reader.A3ReadError, match="OUTSIDE models/"):
        reader.play(str(world.models / "_ledger" / "a3"), str(world.models), player=_player([]))
    assert not (world.models / "_ledger").exists()


# -------------------------------------------------------------------- the registered classes, through the driver
#: C0's readers win 2,400 of 4,000 (gap +10 pp). With no spread across finals each side's variance is its binomial
#: floor, so the 90 % half-width on E is 1.376 pp: 2,344 A3 wins (E = -1.400) FALLS, 2,345 (E = -1.375) does not;
#: 2,456 (E = +1.400) is MORE-EXPLOITABLE, 2,455 is not. A uniform G-F matrix has half-width 0.783 pp against the
#: bar -2: 47.0 FAILs, 47.5 holds. R1 against the banked (57.52, 56.60, 56.81): S = 53 is a LOSS, S = 54 is not.
CASES = [
    # a3 wins, gf, r1 -> outcome, labels
    (2_344, 47.5, 54.0, "ROBUSTNESS-GAIN", []),
    (2_200, 50.0, 57.0, "ROBUSTNESS-GAIN", []),
    (2_344, 47.0, None, "TRADE", []),                              # the gap falls, a head-to-head LOSS
    (2_344, 47.5, 53.0, "TRADE", []),                              # the gap falls, a population-score LOSS
    (2_345, 47.5, None, "NO-FALL", []),                            # one game the other side of the boundary
    (2_345, 47.0, None, "NO-FALL", []),                            # G-F is reported, it does not class a NO-FALL
    (2_400, 50.0, None, "NO-FALL", []),
    (2_455, 50.0, None, "NO-FALL", []),
    (2_456, 50.0, None, "NO-FALL", ["MORE-EXPLOITABLE"]),
    (2_600, 44.0, None, "NO-FALL", ["MORE-EXPLOITABLE"]),
]


@pytest.mark.parametrize("a3,gf,r1,outcome,labels", CASES)
def test_the_driver_ends_in_the_registered_class_on_each_side_of_every_boundary(reader, world, a3, gf, r1, outcome,
                                                                               labels):
    _played(reader, world, _wins(a3, 2_400))
    out = world.tmp / "out" / "verdict.json"
    doc = reader.read_verdict(world.ledger, str(out), _guard(world.tmp, "gf", gf),
                              _guard(world.tmp, "r1", r1) if r1 is not None else None, models_dir=str(world.models))
    assert (doc["outcome"], doc["labels"]) == (outcome, labels), doc["reasons"]
    e = doc["e"]
    assert e["e_pp"] == pytest.approx((a3 - 2_400) / 40.0) and e["near_boundary"] == []
    assert min(abs(e["interval90_pp"][0]), abs(e["interval90_pp"][1])) > 1e-3      # rule 8: clear of the boundary
    assert e["falls"] == (a3 <= 2_344) and e["more_exploitable"] == (a3 >= 2_456)
    assert doc["g_f"]["fail"] == (gf <= 47.0) and abs(doc["g_f"]["interval95_pp"][1] + 2.0) > 0.2
    assert ("r1" in doc) == (r1 is not None)
    if r1 is not None:
        assert doc["r1"]["loss"] == (r1 <= 53.0) and abs(doc["r1"]["interval975_pp"][1] + 2.0) > 0.2
    # the same input through the frozen rule directly: the driver adds nothing to the class
    again = reader.R.decide(doc["inputs"])
    assert (again["outcome"], again["labels"], again["reasons"]) == (doc["outcome"], doc["labels"], doc["reasons"])
    assert json.loads(out.read_text())["outcome"] == outcome
    assert doc["inputs"]["readers"]["a3"]["3502"] == [a3, 4_000, 4_000]
    assert doc["cells"]["c0_s3501"]["r"] == {"wins": 2_400, "finished": 4_000, "attempted": 4_000, "ties": 0}
    assert (doc["regime"], doc["seed"], doc["transport"]) == ("greedy-vs-greedy", 20261020, "rust_eval")


def test_r1_is_played_only_when_owed_and_a_read_that_owes_it_writes_nothing(reader, world):
    """PREREG §5.2: R1 is played only when the gap FALLS and G-F does not FAIL. The read without it writes no
    verdict and exits 4; the read with it classes."""
    _played(reader, world, _wins(2_344, 2_400))
    out = world.tmp / "verdict.json"
    gf, r1 = _guard(world.tmp, "gf", 47.5), _guard(world.tmp, "r1", 54.0)
    base = ["read", "--ledger", world.ledger, "--models-dir", str(world.models), "--out", str(out)]
    assert reader.read_verdict(world.ledger, str(out), gf, models_dir=str(world.models)) == {"r1_owed": True}
    assert reader.main(base + ["--gf", gf]) == 4 and not out.exists()
    assert reader.main(base + ["--gf", gf, "--r1", r1]) == 0
    assert json.loads(out.read_text())["outcome"] == "ROBUSTNESS-GAIN"
    assert reader.main(base + ["--gf", gf, "--r1", r1]) == 2       # a verdict is written ONCE


@pytest.mark.parametrize("a3,gf", [(2_345, 47.5), (2_344, 47.0), (2_456, 50.0)])
def test_an_r1_that_was_not_owed_is_refused(reader, world, a3, gf):
    _played(reader, world, _wins(a3, 2_400))
    out = world.tmp / "verdict.json"
    with pytest.raises(reader.A3ReadError, match="R1 is played only when the gap FALLS and G-F does not FAIL"):
        reader.read_verdict(world.ledger, str(out), _guard(world.tmp, "gf", gf), _guard(world.tmp, "r1", 50.0),
                            models_dir=str(world.models))
    assert not out.exists()


def test_a_missing_guard_the_class_needs_is_inconclusive_and_one_it_does_not_need_is_a_reason(reader, world):
    """PREREG §5.4: "a guard the class needs is missing or malformed" is INCONCLUSIVE. A cross that could not be
    read carries no matrix / no S."""
    _played(reader, world, _wins(2_344, 2_400))
    kw = dict(models_dir=str(world.models))
    doc = reader.read_verdict(world.ledger, str(world.tmp / "v1.json"), _guard(world.tmp, "gf"), **kw)
    assert doc["outcome"] == "INCONCLUSIVE" and "G-F" in doc["reasons"][0] and "gf_h_pp" not in doc["inputs"]
    doc = reader.read_verdict(world.ledger, str(world.tmp / "v2.json"), _guard(world.tmp, "gf", 47.5),
                              _guard(world.tmp, "r1"), **kw)
    assert doc["outcome"] == "INCONCLUSIVE" and "R1" in doc["reasons"][0] and "r1_on" not in doc["inputs"]
    assert doc["guards"]["r1"]["inconclusive"] == ["missing cell aaaa vs bbbb"]


def test_a_no_fall_with_an_unreadable_g_f_is_still_a_no_fall_with_the_reason(reader, world):
    _played(reader, world, _wins(2_400, 2_400))
    doc = reader.read_verdict(world.ledger, str(world.tmp / "v.json"), _guard(world.tmp, "gf"),
                              models_dir=str(world.models))
    assert doc["outcome"] == "NO-FALL" and len(doc["reasons"]) == 1 and "always played" in doc["reasons"][0]


def test_a_reader_with_more_than_a_quarter_of_its_games_unfinished_is_inconclusive(reader, world):
    """A timeout is never an outcome: 599 of a team's 800 games finished is 25.1 % unfinished (600 is 25 %, the
    bar itself, and stays readable)."""
    _played(reader, world, _wins(1_700, 2_400), finished={"a3_s3502_r": 599})
    doc = reader.read_verdict(world.ledger, str(world.tmp / "v.json"), _guard(world.tmp, "gf", 50.0),
                              models_dir=str(world.models))
    assert doc["outcome"] == "INCONCLUSIVE" and "readers.a3.3502: 1005 of 4000 games unfinished" in doc["reasons"][0]
    assert "unreadable" in doc["reported"]["k0"] and doc["reported"]["k0"]["unreadable"] != []


def test_a_reader_at_the_unfinished_bar_itself_is_read(reader, world):
    wins = dict(_wins(1_700, 2_400), a3_s3502_r=1_275)             # the same 42.5 % of the 3,000 it finished
    _played(reader, world, wins, finished={"a3_s3502_r": 600})
    doc = reader.read_verdict(world.ledger, str(world.tmp / "v.json"), _guard(world.tmp, "gf", 50.0),
                              _guard(world.tmp, "r1", 57.0), models_dir=str(world.models))
    assert doc["outcome"] == "ROBUSTNESS-GAIN" and doc["inputs"]["readers"]["a3"]["3502"] == [1_275, 3_000, 4_000]


def test_an_unrepaired_void_is_inconclusive_with_no_game_read(reader, world):
    out = world.tmp / "verdict.json"
    assert reader.main(["read", "--void", "rb_x35_ge_a3mx_s3502: a [STABLE] line, re-run VOID too", "--out",
                        str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["outcome"] == "INCONCLUSIVE" and "e" not in doc and "cells" not in doc
    assert doc["reasons"] == ["VOID, not repaired (PREREG §9): rb_x35_ge_a3mx_s3502: a [STABLE] line, re-run VOID too"]
    assert not Path(world.ledger).exists()
    assert reader.main(["read", "--void", " ", "--out", str(world.tmp / "v2.json")]) == 2


def test_the_read_refuses_an_unplayed_ledger_a_wrong_guard_file_and_no_g_f(reader, world):
    out = world.tmp / "verdict.json"
    gf = _guard(world.tmp, "gf", 50.0)
    kw = dict(models_dir=str(world.models))
    with pytest.raises(reader.A3ReadError, match="not banked"):
        reader.read_verdict(world.ledger, str(out), gf, **kw)
    _played(reader, world, _wins(2_400, 2_400))
    with pytest.raises(reader.A3ReadError, match="G-F is always played"):
        reader.read_verdict(world.ledger, str(out), None, **kw)
    with pytest.raises(reader.A3ReadError, match="is the 'r1' cross's, not the 'gf' cross's"):
        reader.read_verdict(world.ledger, str(out), _guard(world.tmp, "r1", 57.0), **kw)
    assert not out.exists()


def test_a_cell_short_of_its_registered_games_is_refused(reader, world):
    _played(reader, world, _wins(2_400, 2_400))
    f = Path(world.ledger) / "c0_s3502__r.json"
    doc = json.loads(f.read_text())
    doc["counts"][sorted(doc["counts"])[0]]["attempted"] -= 1
    f.write_text(json.dumps(doc))
    with pytest.raises(reader.A3ReadError, match="games attempted"):
        reader.read_verdict(world.ledger, str(world.tmp / "v.json"), _guard(world.tmp, "gf", 50.0),
                            models_dir=str(world.models))


def test_the_k0_decomposition_is_reported_beside_the_class_and_never_enters_it(reader, world):
    """PREREG §5.5 item 1: E_learned = E - (mean k0 gap of A3 - mean k0 gap of C0). Here E = -5 pp and the A3
    targets' copies on the offense pin score 4 pp lower than C0's, so only -1 pp of the fall is learned."""
    _played(reader, world, _wins(2_200, 2_400, k0_a3=1_840, k0_c0=2_000))
    doc = reader.read_verdict(world.ledger, str(world.tmp / "v.json"), _guard(world.tmp, "gf", 50.0),
                              _guard(world.tmp, "r1", 57.0), models_dir=str(world.models))
    k0 = doc["reported"]["k0"]
    assert doc["outcome"] == "ROBUSTNESS-GAIN" and doc["e"]["e_pp"] == pytest.approx(-5.0)
    assert k0["k0_gaps_pp"] == {"a3": [pytest.approx(-4.0)] * 3, "c0": [pytest.approx(0.0)] * 3}
    assert k0["e_learned_pp"] == pytest.approx(-1.0) and "no class" in k0["note"]
    lo, hi = k0["interval90_pp"]
    assert lo < -1.0 < hi and (hi - lo) > (doc["e"]["interval90_pp"][1] - doc["e"]["interval90_pp"][0])
    assert "k0" not in json.dumps(doc["inputs"])                    # the rule never sees the k0 cells


# ------------------------------------------------------------------------------- the four head-to-head crosses
@pytest.fixture(scope="module")
def plan():
    return runpy.run_path(str(DIR / "cross" / "plan_cross.py"))


@pytest.fixture()
def cross():
    return runpy.run_path(str(DIR / "cross" / "read_cross.py"))


@pytest.fixture(scope="module")
def rule():
    return runpy.run_path(str(DIR / "a3_rule.py"))


def test_the_cross_plans_constants_are_the_kits_the_rules_and_the_registered_cells(plan, cross, rule):
    arms = plan["ARMS"]
    assert plan["SIDE_PIN"] == PIN and plan["FORK_STEP"] == 280_000_170
    for name, kit_arm in (("mx", "a3_mx"), ("a3", "a3"), ("c0", "c0")):
        fmt, seeds, _floor, _role = arms[name]
        assert [fmt.format(s) for s in seeds] == [K.run_name(kit_arm, s) for s in seeds], name
    assert arms["mx"][1] == (3511, 3512, 3513) and arms["a3"][1] == arms["c0x3"][1] == SEEDS
    assert len(arms["c0"][1]) == rule["C0_FINALS"] == 8
    assert (arms["mx"][2], arms["a3"][2], arms["c0"][2]) == (285_000_170, 300_000_170, 300_000_170)
    assert (arms["mx"][3], arms["a3"][3], arms["c0"][3]) == ("exploiter", "fork", "fork")
    assert plan["CROSSES"] == {"gate": ("mx", ("pstar",), rule["GATE_PAIRS"]), "gf": ("a3", ("c0",), rule["GF_PAIRS"]),
                               "r1": ("a3", ("R",), 500), "read2": ("mx", ("a3", "c0x3"), 500)}
    assert {k: v[2] for k, v in cross["CROSSES"].items()} == {k: v[2] for k, v in plan["CROSSES"].items()}
    assert cross["CROSSES"]["gf"][3] == "gf.json" and cross["CROSSES"]["r1"][3] == "r1.json"
    assert len({v[0] for v in cross["CROSSES"].values()}) == 4 == len({v[1] for v in cross["CROSSES"].values()})
    call = json.loads((plan["REFPOP"]).read_text())
    assert plan["PSTAR_SHA256"] == call["members"][0]["sha256"] and call["members"][0]["label"] == "x26_pstar"
    assert len(call["members"]) == 7
    assert json.loads(plan["G1_RESULTS"].read_text())["refpop_content_sha256"] == call["content_sha256"]


def _cross_run(root: Path, plan: dict, name: str, seed: int, role: str, last: int) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps({
        "pin_history": [{"git_hash": PIN, "first_step": plan["FORK_STEP"], "last_step": last}],
        "lineage": {"role": role, "fork_parent": {"sha256": plan["PSTAR_SHA256"]}},
        "cli_args": {"seed": seed}, "torch_version": "2.8.0", "init_num_threads": 1, "num_timesteps": last}))
    return run


def test_the_cross_plan_admits_a_clean_run_and_refuses_each_way_it_can_be_another(plan, tmp_path, monkeypatch):
    """PREREG §9 for a cross's files: one pin entry at the side pin from P*'s step, the arm's length, the role its
    record must name (an exploiter for phase 1, a fork for phase 2), its seed, K9(b) PASS."""
    monkeypatch.setattr(B, "k9b_read", lambda run: SimpleNamespace(verdict="PASS", reasons=[], fires=[]))
    check = plan["check_run"]
    mx = _cross_run(tmp_path / "ok", plan, "rb_x35_mx_s3511", 3511, "exploiter", 285_000_400)
    a3 = _cross_run(tmp_path / "ok", plan, "rb_x35_a3mx_s3501", 3501, "fork", 300_061_000)
    assert check(mx, 3511, 285_000_170, "exploiter")[0] == [] and check(a3, 3501, 300_000_170, "fork")[0] == []
    assert check(a3, 3501, 300_000_170, "fork")[1]["torch_version"] == "2.8.0"
    # an exploiter is not a fold-in fork, and a fork is not an exploiter
    assert any("is not a FORK of P*" in p for p in check(mx, 3511, 285_000_170, "fork")[0])
    assert any("is not a EXPLOITER of P*" in p for p in check(a3, 3501, 300_000_170, "exploiter")[0])

    def one(tag, fn, needle):
        bad = _cross_run(tmp_path / tag, plan, "rb_x35_a3mx_s3501", 3501, "fork", 300_061_000)
        doc = json.loads((bad / "metadata.json").read_text())
        fn(doc)
        (bad / "metadata.json").write_text(json.dumps(doc))
        got = check(bad, 3501, 300_000_170, "fork")[0]
        assert any(needle in g for g in got), (tag, got)

    one("pin", lambda d: d["pin_history"][0].update(git_hash="0" * 40), "is not one entry at f4b25238")
    one("two", lambda d: d["pin_history"].append(dict(d["pin_history"][0])), "is not one entry at f4b25238")
    one("start", lambda d: d["pin_history"][0].update(first_step=0), "not the fork step")
    one("short", lambda d: d["pin_history"][0].update(last_step=299_999_999), "last_step 299999999 <")
    one("parent", lambda d: d["lineage"]["fork_parent"].update(sha256="0" * 64), "is not a FORK of P*")
    one("seed", lambda d: d["cli_args"].update(seed=3502), "seed 3502 != 3501")
    monkeypatch.setattr(B, "k9b_read", lambda run: SimpleNamespace(verdict="FIRE", reasons=[], fires=["approx_kl"]))
    assert check(a3, 3501, 300_000_170, "fork")[0] == ["K9(b) FIRE: approx_kl"]


def test_a_cross_is_refused_while_its_runs_do_not_exist(plan, tmp_path, monkeypatch, capsys):
    """No A3 run exists at this build: planning G-F over an empty models directory names every missing final."""
    monkeypatch.setitem(plan["main"].__globals__, "MODELS", tmp_path)
    monkeypatch.setattr("sys.argv", ["plan_cross.py", "--cross", "gf", "--out", str(tmp_path / "c.json"),
                                     "--shas", str(tmp_path / "s.json")])
    assert plan["main"]() == 3
    err = capsys.readouterr().err
    assert err.count("no final_model.zip") == 11 and "rb_x35_a3mx_s3503: no final_model.zip" in err
    assert not (tmp_path / "c.json").exists() and not (tmp_path / "s.json").exists()


def _cell(row: str, col: str, score_pp: float, n_pairs: int, verdict: str = "OK"):
    return SimpleNamespace(player=row, opponent=col, player_id=row[:6], opponent_id=col[:6], w=0, l=0, d=0, aborted=0,
                           pairs=(0, 0, n_pairs, 0, 0), verdict=verdict, reasons=("aborted pairs",), n_pairs=n_pairs,
                           score=score_pp / 100.0)


def _shas(cross: str, pairs: int, rows, cols, col_labels=None) -> dict:
    return {"cross": cross, "pairs": pairs, "rows": rows, "cols": cols, "row_seeds": list(SEEDS)[:len(rows)],
            "col_seeds": col_labels or list(range(len(cols))), "row_runs": [f"run_{r}" for r in rows],
            "col_runs": [f"run_{c}" for c in cols]}


ROWS = ["a" * 64, "b" * 64, "c" * 64]


def test_the_gate_cross_reads_exploits_from_the_three_scores_against_p_star(cross, rule):
    shas = _shas("gate", 1000, ROWS, ["p" * 64])
    doc = cross["result"](shas, [_cell(r, "p" * 64, s, 1000) for r, s in zip(ROWS, (54.0, 55.0, 56.0))], rule)
    assert doc["scores_pp"] == pytest.approx([54.0, 55.0, 56.0]) and doc["exploits"] is True
    assert doc["gate"] == rule["phase1_gate"](doc["scores_pp"]) and doc["inconclusive"] == []
    flat = cross["result"](shas, [_cell(r, "p" * 64, 50.0, 1000) for r in ROWS], rule)
    assert flat["exploits"] is False


def test_the_g_f_cross_hands_the_rule_its_three_by_eight_and_refuses_a_cell_off_its_pair_count(cross, rule):
    cols = [f"{i}" * 64 for i in range(8)]
    shas = _shas("gf", 1000, ROWS, cols)
    doc = cross["result"](shas, [_cell(r, c, 47.0, 1000) for r in ROWS for c in cols], rule)
    assert doc["gf_h_pp"] == [[pytest.approx(47.0)] * 8] * 3 and doc["g_f"]["fail"] is True
    assert doc["g_f"] == rule["guard_f"](doc["gf_h_pp"])
    assert cross["result"](shas, [_cell(r, c, 47.5, 1000) for r in ROWS for c in cols], rule)["g_f"]["fail"] is False
    for n, needle in ((999, "999 completed pairs < 1000"), (1001, "1001 completed pairs > the registered 1000")):
        cells = [_cell(r, c, 50.0, 1000) for r in ROWS for c in cols]
        cells[5] = _cell(ROWS[0], cols[5], 50.0, n)
        bad = cross["result"](shas, cells, rule)
        assert "gf_h_pp" not in bad and bad["h_pp"] is None and any(needle in x for x in bad["inconclusive"]), n
    short = cross["result"](shas, [_cell(r, c, 50.0, 1000) for r in ROWS for c in cols][:-1], rule)
    assert "gf_h_pp" not in short and any("missing cell" in x for x in short["inconclusive"])
    aborted = [_cell(r, c, 50.0, 1000) for r in ROWS for c in cols]
    aborted[0] = _cell(ROWS[0], cols[0], 50.0, 1000, verdict="INCONCLUSIVE")
    assert "gf_h_pp" not in cross["result"](shas, aborted, rule)


def test_the_r1_cross_is_the_mean_of_seven_member_cells_a_fork_against_c0s_banked_scores(cross, rule):
    cols = [f"{i}" * 64 for i in range(7)]
    labels = ["x26_pstar", "x26_270m", "x26_080m", "x26_140m", "x26_210m", "production", "tstar"]
    shas = _shas("r1", 500, ROWS, cols, labels)
    per = {ROWS[0]: 57.0, ROWS[1]: 54.0, ROWS[2]: 55.5}
    cells = [_cell(r, c, per[r] + (j - 3), 500) for r in ROWS for j, c in enumerate(cols)]
    doc = cross["result"](shas, cells, rule)
    assert doc["r1_on"] == {"3501": pytest.approx(57.0), "3502": pytest.approx(54.0), "3503": pytest.approx(55.5)}
    assert doc["r1_off_banked"] == rule["c0_population_scores"]() and doc["r1"]["loss"] is False
    assert sorted(doc["per_member_pp"]["3501"]) == sorted(labels)
    low = cross["result"](shas, [_cell(r, c, 53.0, 500) for r in ROWS for c in cols], rule)
    assert low["r1"]["loss"] is True
    bad = cross["result"](shas, [_cell(r, c, 55.0, 500) for r in ROWS for c in cols][:-1], rule)
    assert "r1_on" not in bad and bad["inconclusive"]


def test_read_2_is_a_reported_three_by_six_with_no_class(cross, rule):
    cols = [f"{i}" * 64 for i in range(6)]
    doc = cross["result"](_shas("read2", 500, ROWS, cols),
                          [_cell(r, c, 50.0 + j, 500) for r in ROWS for j, c in enumerate(cols)], rule)
    assert doc["row_means_pp"] == [pytest.approx(52.5)] * 3
    assert doc["col_means_pp"] == [pytest.approx(50.0 + j) for j in range(6)]
    assert "no class" in doc["note"] and not {"outcome", "g_f", "r1", "gate"} & set(doc)


def _row(request: str, device: str, backend: str) -> dict:
    return {"request": {"id": request}, "compute": {"device": device, "backend": backend}}


@pytest.mark.parametrize("rows,device,rc", [([], "cuda", 0), ([("cpu", "eager")], "cpu", 0),
                                            ([("cuda", "graph")], "cuda", 0), ([("cpu", "eager")], "cuda", 7),
                                            ([("cuda", "graph")], "cpu", 7)])
def test_a_cross_is_finished_on_the_device_it_began_on(cross, monkeypatch, tmp_path, capsys, rows, device, rc):
    shas = tmp_path / "shas.json"
    shas.write_text(json.dumps({"cross": "gf", "pairs": 1000}))
    served = [_row("x35_a3_forks_vs_c0", d, b) for d, b in rows] + [_row("x35_a3_forks_vs_r", "cpu", "eager")]
    monkeypatch.setitem(cross["main"].__globals__, "_family_rows", lambda family: served)
    monkeypatch.setattr("sys.argv", ["read_cross.py", "--venue-check", str(shas), device])
    assert cross["main"]() == rc
    got = capsys.readouterr()
    assert ("one venue a cross" in got.err) == (rc == 7) and ("venue OK" in got.out) == (rc == 0), got


def test_the_cross_reader_refuses_a_mixed_venue_and_a_second_read_before_it_writes(cross, monkeypatch, tmp_path,
                                                                                  capsys):
    shas = tmp_path / "shas.json"
    shas.write_text(json.dumps({"cross": "gf", "pairs": 1000}))
    out = tmp_path / "result"
    out.mkdir()
    mixed = [_row("x35_a3_forks_vs_c0", "cpu", "eager"), _row("x35_a3_forks_vs_c0", "cuda", "graph")]
    monkeypatch.setitem(cross["main"].__globals__, "_family_rows", lambda family: mixed)
    monkeypatch.setattr("sys.argv", ["read_cross.py", str(shas), str(out), str(DIR / "a3_rule.py")])
    assert cross["main"]() == 7
    assert "more than one device/backend" in capsys.readouterr().err and list(out.iterdir()) == []
    (out / "gf.json").write_text("{}")
    assert cross["main"]() == 6
    assert "a cross is read ONCE" in capsys.readouterr().err and [p.name for p in out.iterdir()] == ["gf.json"]


def test_the_cross_reader_writes_the_file_the_read_takes(cross, monkeypatch, tmp_path, capsys, reader):
    """End to end on fixture cells: ``read_cross.py`` writes ``gf.json``, and it is the file ``read_a3.py read --gf``
    accepts (its ``cross`` and its ``gf_h_pp``)."""
    import main.h2h.cross as X

    cols = [f"{i}" * 64 for i in range(8)]
    shas = tmp_path / "shas.json"
    shas.write_text(json.dumps(_shas("gf", 1000, ROWS, cols)))
    out = tmp_path / "result"
    rows = [_row("x35_a3_forks_vs_c0", "cuda", "graph")] * 24
    monkeypatch.setitem(cross["main"].__globals__, "_family_rows", lambda family: rows)
    monkeypatch.setattr(X, "look_cells", lambda root, family, request, protocol: [
        _cell(r, c, 47.5, 1000) for r in ROWS for c in cols])
    monkeypatch.setattr("sys.argv", ["read_cross.py", str(shas), str(out), str(DIR / "a3_rule.py")])
    assert cross["main"]() == 0
    capsys.readouterr()
    doc = json.loads((out / "gf.json").read_text())
    assert doc["cross"] == "gf" and doc["played_on"] == [{"device": "cuda", "backend": "graph", "rows": 24}]
    assert (doc["family"], doc["request"], doc["n_cells_read"]) == ("x35_a3_gf_h2h", "x35_a3_forks_vs_c0", 24)
    got = reader.rule_input({u: {"r": {"wins": 2_000, "finished": 4_000, "attempted": 4_000}}
                             for _, _, u in reader.units()}, doc, None)
    assert got["gf_h_pp"] == doc["gf_h_pp"] and reader.R.decide(got)["outcome"] == "NO-FALL"
    assert sorted(p.name for p in out.iterdir()) == ["cells.json", "gf.json", "h.json"]


def test_the_cross_driver_plays_each_cross_as_registered_and_checks_the_venue_first():
    text = (DIR / "cross" / "drive_cross.sh").read_text()
    assert "PIN=333a73ddda7829b9ccdf9e17ef04624eb40edeba" in text                # PREREG §11.1 item 3's play checkout
    for cross_name, pairs in (("gate", 1000), ("gf", 1000), ("r1", 500), ("read2", 500)):
        assert re.search(rf"^\s*{cross_name}\)\s.*PAIRS={pairs}\b", text, re.M), cross_name
    register = text.index("family-register")
    check, play = text.index('--venue-check "$W/shas.json" "$DEV"'), text.index('PLAY=("$PY" -m main.h2h play-many')
    assert register < check < play
    assert "--seed 0" in text and "--oracle-reveal-mode off" in text and "gen3_eval_protocol_v1_h2h" in text
    assert "gpu_lock.sh" in text and "mem_cap.sh" in text and "core_split agent-cpus" in text
    assert "GEN3AI_TEST_ALLOW_GPU" not in text and "pkill" not in text
