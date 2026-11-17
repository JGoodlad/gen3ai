"""X71 (PREREG ``x71_exploitability_probe_2026-11-16``): the exploitability probe of the entropy arms -- the committed
registration files: the responder argv and its validation transcripts, the rule and its hand cases, the design
simulation (reproduced), the reader's plan / play / read on synthetic run directories with a recorded player, and
what the PREREG says about each.

Nothing here launches, plays a game or touches ``models/``.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils.paths import repo_path

DIR = repo_path("designs/research_state/measurements/x71_exploitability_probe_2026-11-16")
PIN = "9e97cad5709da21a1f9d4d8846096c87eb17b1d0"
#: tag -> (the target run, its final step): the three targets that existed at registration
EXISTING = {"c0": ("rb_x35_c0_s3501", 300_061_940), "x67": ("rb_x35_x67ent_s6701", 300_021_355),
            "swap": ("rb_x35_ce3dswap_s7321", 300_061_272)}
SEEDS = (7111, 7112)


def _banked_models_root() -> str:
    """The models directory the registration's argvs were written against, READ from the banked record (the first
    committed argv's ``--model``) and never typed here: an absolute home path as a value is correct on one machine
    only (``utils/paths_test.py``). Every other banked argv and transcript is then checked against this one root."""
    argv = (DIR / "argv_x71_c0_s7111.txt").read_text().split()
    root, run, name = argv[argv.index("--model") + 1].rsplit("/", 2)
    assert (run, name) == ("rb_x35_c0_s3501", "final_model.zip") and root.startswith("/") and root.endswith("/models")
    return root


MODELS = _banked_models_root()


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"x71_frozen_{name}", DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def reader():
    """``read_x71.py`` (it imports ``x71_rule`` from its own directory as ``reader.R``)."""
    before = list(sys.path)
    try:
        return _load("read_x71")
    finally:
        sys.path[:] = before
        sys.modules.pop("x71_rule", None)


@pytest.fixture(scope="module")
def rule(reader):
    return reader.R


# ------------------------------------------------------------------------------------------- the argv and its checks
@pytest.mark.parametrize("tag", sorted(EXISTING))
@pytest.mark.parametrize("seed", SEEDS)
def test_the_rule_builds_the_banked_argv_token_for_token(rule, tag, seed):
    run, step = EXISTING[tag]
    argv = rule.responder_argv(tag, seed, f"{MODELS}/{run}/final_model.zip", step)
    assert " ".join(argv) == (DIR / f"argv_x71_{tag}_s{seed}.txt").read_text().strip()
    assert argv[argv.index("--steps") + 1] == str(step + 5_000_000)
    assert argv[argv.index("--model") + 1] == argv[argv.index("--exploiter") + 1]
    assert argv[argv.index("--run-name") + 1] == f"rb_x35_x71_{tag}_s{seed}"


def test_every_responder_types_the_control_recipe_whatever_its_target_trained_with(rule):
    for tag in rule.TARGETS:
        argv = rule.responder_argv(tag, 7111, "/m/T/final_model.zip", 300_000_000, teams="a,b,c,d,e")
        assert argv[argv.index("--ent-coef") + 1] == "0.05"
        assert argv[argv.index("--magnet-kl-coef") + 1] == "0"
        assert "--no-self-play" in argv and "--fork-lr-freeze" in argv
        assert argv[argv.index("--fork-lr") + 1] == "5.6e-05"


def test_the_targets_are_each_arms_lowest_seed_and_the_seeds_and_pin_are_the_registered_ones(rule):
    assert {t: v[1] for t, v in rule.TARGETS.items()} == {
        "c0": "rb_x35_c0_s3501", "x67": "rb_x35_x67ent_s6701", "swap": "rb_x35_ce3dswap_s7321",
        "x70": "rb_x35_x70ent0_s7331"}
    assert rule.SEEDS == SEEDS and rule.PIN == PIN and rule.L_RESP == 5_000_000
    assert rule.READS == {1: ("x67", "swap"), 2: ("x70",)}
    assert rule.READ_TARGETS == {1: ("c0", "x67", "swap"), 2: ("c0", "x67", "swap", "x70")}
    assert rule.TARGET_SHA256["x70"] is None and all(len(rule.TARGET_SHA256[t]) == 64 for t in EXISTING)
    assert (rule.G, rule.G0, rule.N_TEAMS, rule.READ_SEED, rule.BAR) == (4_000, 8_000, 5, 20261020, 0.05)


@pytest.mark.parametrize("tag,seed", [("c0", 7101), ("c0", 3501), ("x67", 7113), ("p", 7111)])
def test_the_rule_refuses_an_unregistered_responder(rule, tag, seed):
    with pytest.raises(rule.X71Error):
        rule.run_name(tag, seed)


@pytest.mark.parametrize("tag", sorted(EXISTING))
@pytest.mark.parametrize("seed", SEEDS)
def test_the_transcripts_are_this_argvs_and_stop_only_at_the_registering_shells_two_refusals(tag, seed):
    argv = (DIR / f"argv_x71_{tag}_s{seed}.txt").read_text().strip()
    check = (DIR / f"checkargs_x71_{tag}_s{seed}.txt").read_text()
    dry = (DIR / f"dryrun_x71_{tag}_s{seed}.txt").read_text()
    assert f'--argv "{argv}" --pin {PIN}' in check.splitlines()[0]
    assert check.rstrip().endswith("[exit 0]")
    assert "this command still launches" in check
    assert f"--pin-commit {PIN} {argv}" in dry.splitlines()[0]
    assert dry.rstrip().endswith("[exit 3]")
    refused = [ln.split(":")[0].strip() for ln in dry.splitlines() if "REFUSED" in ln]
    assert refused == ["cuda devices", "inherited nice"]
    run, _step = EXISTING[tag]
    assert f"role        : FORK of {MODELS}/{run}" in dry
    assert "→ +5,000,000 steps" in dry and "7.91 GiB REQUIRED" in dry
    assert f"rb_x35_x71_{tag}_s{seed}   [would be created]" in dry
    assert f"{MODELS}/rb_x35_x71" not in dry          # the dry run resolved under a scratch models dir


# ------------------------------------------------------------------------------------------------------- the rule
def test_the_rules_self_check_passes_and_is_what_was_banked(rule):
    assert rule.self_check() == []
    assert (DIR / "selfcheck.log").read_text().strip() == "self-check OK"


def _cells(rule, gaps, g0=-0.025):
    return rule._cells(gaps, g0)


def test_the_multiplier_is_the_t_quantile_at_the_reads_number_of_targets(rule):
    assert round(rule.mult(3), 3) == 3.182 and round(rule.mult(4), 3) == 2.776
    base = {"c0": (0.065, 0.065), "x67": (0.095, 0.095), "swap": (0.065, 0.065)}
    v = rule.classify(_cells(rule, base), 1)
    # +3 pp with no responder spread: above zero under 1.96 (half-width 2.2 pp), not under t(3) (3.5 pp)
    assert v["cls"]["x67"] == "UNRESOLVED"
    assert v["arms"]["x67"]["D"] - 1.96 * v["arms"]["x67"]["se"] > rule.RULE8
    assert v["arms"]["x67"]["lo"] < 0
    # the same target in read 2 (four targets, t(4)) is classed by X70's read only for x70
    v2 = rule.classify(_cells(rule, dict(base, x70=(0.065, 0.065))), 2)
    assert set(v2["cls"]) == {"x70"} and v2["mult"] == rule.mult(4)


def test_responder_disagreement_enters_the_interval_and_game_noise_alone_does_not(rule):
    tight = rule.classify(_cells(rule, {"c0": (0.065, 0.065), "x67": (0.165, 0.165), "swap": (0.065, 0.065)}), 1)
    split = rule.classify(_cells(rule, {"c0": (0.065, 0.065), "x67": (0.245, 0.085), "swap": (0.065, 0.065)}), 1)
    assert tight["s2r"] == 0.0 and tight["cls"]["x67"] == "MORE-EXPLOITABLE"
    assert split["sigma_r"] > 0.05 and split["cls"]["x67"] == "UNRESOLVED"
    assert abs(split["arms"]["x67"]["D"] - tight["arms"]["x67"]["D"]) < 1e-12
    # the pooled spread widens EVERY arm's interval, not only the arm that split
    assert split["arms"]["swap"]["se"] > tight["arms"]["swap"]["se"]


def test_the_control_must_respond_and_every_cell_must_be_readable(rule):
    flat = rule.classify(_cells(rule, {"c0": (-0.025, -0.025), "x67": (0.165, 0.165), "swap": (0.065, 0.065)}), 1)
    assert set(flat["cls"].values()) == {"INCONCLUSIVE"} and "did not respond" in flat["why"]
    c = _cells(rule, {"c0": (0.065, 0.065), "x67": (0.065, 0.065), "swap": (0.065, 0.065)})
    c["c0"]["k0"] = dict(c["c0"]["k0"], finished=5_000, wins=2_400)        # 37.5 % of 8,000 unfinished
    out = rule.classify(c, 1)
    assert set(out["cls"].values()) == {"INCONCLUSIVE"} and "did not finish" in out["why"]
    c["c0"]["k0"] = dict(c["c0"]["k0"], finished=6_400, wins=3_040)        # 20 %: under the bar, read
    assert "INCONCLUSIVE" not in rule.classify(c, 1)["cls"].values()
    with pytest.raises(rule.X71Error):
        rule.classify(c, 3)


def test_the_bar_bounds_only_an_interval_wholly_below_it(rule):
    base = {"c0": (0.065, 0.065), "x67": (0.065, 0.065), "swap": (0.075, 0.075)}
    v = rule.classify(_cells(rule, base), 1)
    assert v["cls"] == {"x67": "NOT-MORE-BY-BAR", "swap": "NOT-MORE-BY-BAR"}
    assert v["arms"]["swap"]["hi"] < rule.BAR - rule.RULE8
    v = rule.classify(_cells(rule, dict(base, swap=(0.085, 0.085))), 1)
    assert v["cls"]["swap"] == "UNRESOLVED" and v["arms"]["swap"]["hi"] > rule.BAR


# -------------------------------------------------------------------------------------------- the design simulation
def test_the_banked_power_log_is_what_the_rule_simulates_from_r0s_banked_curve(rule):
    curve = (DIR / rule.R0_CURVE).resolve()
    assert curve.is_file()
    assert rule.simulate_report(str(curve)).rstrip("\n") == (DIR / "power.log").read_text().rstrip("\n")
    a = rule.r0_anchor(str(curve))
    assert (round(100 * a["g0"], 2), round(100 * a["learned5"], 2), round(100 * a["headroom"], 2)) == (-2.5, 9.0, 11.08)
    assert a["plateau_points"] == 7 and a["games_per_point"] == 400


def _rows(block: str):
    """``{(sigma_r pp, D pp): (MORE, NOT-MORE-BY-BAR, half-width)}`` of one table of ``power.log``."""
    out = {}
    for ln in block.splitlines():
        m = re.match(r"\s*([\d.]+)pp\s+\+([\d.]+)pp \|\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+[\d.]+ \|\s+([\d.]+) pp", ln)
        if m:
            out[(float(m[1]), float(m[2]))] = (m[3], m[4], m[5])
    return out


def test_the_preregs_power_table_is_the_logs(rule):
    text = (DIR / "PREREG.md").read_text()
    blocks = (DIR / "power.log").read_text().split("== ")
    for n_t, label in ((3, "**read 1** (3 targets) "), (4, "**read 2** (4 targets) ")):
        (block,) = [b for b in blocks if b.startswith("THE REGISTERED DESIGN") and f"{n_t} targets" in b]
        rows = _rows(block)
        assert len(rows) == 20
        for s in (0, 1, 2, 3):
            cells = " | ".join(rows[(float(s), d)][0] for d in (0.0, 2.5, 5.0, 7.5, 10.0))
            width = rows[(float(s), 0.0)][2]
            line = f"| {label if s == 0 else ''}{s} pp | {cells} | {width} pp |"
            assert text.count(line) == 1, line
    r1 = _rows([b for b in blocks if b.startswith("THE REGISTERED DESIGN") and "3 targets" in b][0])
    assert " / ".join(r1[(float(s), 0.0)][1] for s in (0, 1, 2, 3)) in text
    # the false-MORE ceiling the PREREG quotes for the registered multiplier, and the 1.96 comparison's
    reg = [b for b in blocks if b.startswith("THE REGISTERED DESIGN")]
    assert max(float(v[0]) for b in reg for (s, d), v in _rows(b).items() if d == 0.0) == 0.021
    z = _rows([b for b in blocks if "MULT = 1.96" in b][0])
    assert (z[(2.0, 0.0)][0], z[(3.0, 0.0)][0]) == ("0.056", "0.059")


def test_the_prereg_names_what_the_rule_registers(rule):
    text = (DIR / "PREREG.md").read_text()
    for needle in (PIN, "7111", "7112", "20261020", "rb_x35_x71_{c0,x67,swap,x70}_s{7111,7112}",
                   "--ent-coef 0.05 --magnet-kl-coef 0", "MORE-EXPLOITABLE", "LESS-EXPLOITABLE", "NOT-MORE-BY-BAR",
                   "UNRESOLVED", "INCONCLUSIVE", "8,000", "4,000", "3.182", "2.776", "≈ 8.4 GPU-h", "UNVERIFIED",
                   "OWED, not built here", "NO RESPONDER HAS\nTRAINED"):
        assert needle in text, needle
    for tag, (_arm, run, _lever) in rule.TARGETS.items():
        assert f"| `{tag}` |" in text and f"`{run}`" in text
    for tag, (_run, step) in EXISTING.items():
        assert f"{step:,}" in text
    assert "5.6 % and 5.9 %" in text          # the 1.96 comparison, as the log has it (above)
    assert not [p.name for p in DIR.iterdir() if p.name.startswith(("verdict", "GO"))]


# ------------------------------------------------------------------------------------------------------ the reader
TEAMS = ["data/teams/sample/9eb3abdc52876a63.txt", "data/teams/sample/ac17a9dde5.txt",
         "data/teams/sample/a185b2d193.txt", "data/teams/sample/9ba039ba8a.txt", "data/teams/sample/b904dbe059.txt"]
T_STEP = 300_000_000


def _write_run(root: Path, name: str, *, steps: int, ent=0.05, magnet=None, seed=None, pin=PIN, body=b"") -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "final_model.zip").write_bytes(body or name.encode())
    md = {"num_timesteps": steps, "ent_coef": ent, "cli_args": {"seed": seed}, "pin_history": [{"git_hash": pin}]}
    if magnet is not None:
        md["magnet_kl_coef"] = magnet
    (d / "metadata.json").write_text(json.dumps(md))
    return d


@pytest.fixture()
def world(reader, tmp_path, monkeypatch):
    """A synthetic ``models/`` with the four targets and their eight clean responders, and ``read_exploiter`` reading
    each responder's record from a table the test may edit."""
    from agents.training import best_response_gap as E

    from main.x35 import ge

    assert ge.offense_teams().split(",") == TEAMS
    R = reader.R
    models = tmp_path / "models"
    records = {}
    for tag, (_arm, run, _lever) in R.TARGETS.items():
        t = _write_run(models, run, steps=T_STEP)
        for s in R.SEEDS:
            name = R.run_name(tag, s)
            _write_run(models, name, steps=T_STEP + R.L_RESP + 61_000, seed=s)
            records[name] = dict(target_file=str(t / "final_model.zip"), fork_step=T_STEP, archetype="offense",
                                 membership="exact", teams=list(TEAMS), pooled=(300, 512), end=(160, 256))

    def read_exploiter(ref, _teamsets):
        r = records[Path(ref).name]
        end = SimpleNamespace(wins=r["end"][0], games=r["end"][1], step=T_STEP + R.L_RESP)
        return SimpleNamespace(run=Path(ref).name, target_file=r["target_file"], fork_step=r["fork_step"],
                               archetype=r["archetype"], membership=r["membership"], teams=r["teams"],
                               pooled=lambda: r["pooled"], endpoint=lambda: end)

    monkeypatch.setattr(E, "read_exploiter", read_exploiter)
    monkeypatch.setattr(E, "load_teamsets", lambda path=None: {})
    monkeypatch.setattr(R, "TARGET_SHA256", {t: None for t in R.TARGETS})
    return SimpleNamespace(models=models, records=records, R=R, ledger=str(tmp_path / "ledger"))


def _player(calls, rates):
    """A recorded player: every team's cell wins ``rates[label]`` of its games (default one half)."""
    from agents.training import untaught_meter as um

    def play(ref_path, target_path, teams, games_per_team, *, label):
        calls.append((label, ref_path, target_path, games_per_team, [t.key for t in teams]))
        w = round(rates.get(label, 0.5) * games_per_team)
        got = {t.key: um.Cell(wins=w, losses=games_per_team - w, finished=games_per_team, attempted=games_per_team,
                              transport="rust_eval") for t in teams}
        return got, {t.key: [] for t in teams}, "rust_eval"
    return play


def test_the_plan_admits_the_registered_responders_and_names_their_cells(reader, world):
    p = reader.plan(1, str(world.models))
    assert sorted(p) == ["c0", "swap", "x67"]
    for tag, t in p.items():
        assert list(t["cells"]) == ["k0", "r7111", "r7112"]
        assert t["cells"]["k0"] == {"path": t["target"], "games": 8_000, "sha256": t["target_sha256"]}
        assert t["cells"]["r7111"]["games"] == 4_000
        assert t["cells"]["r7111"]["path"].endswith(f"rb_x35_x71_{tag}_s7111/final_model.zip")
        assert t["cells"]["r7112"]["in_run"] == {"pooled": [300, 512], "endpoint": [160, 256, T_STEP + 5_000_000]}
        assert t["teams"] == TEAMS and t["target_step"] == T_STEP
    assert sorted(reader.plan(2, str(world.models))) == ["c0", "swap", "x67", "x70"]


def _spoil(world, name, **md):
    f = world.models / name / "metadata.json"
    doc = json.loads(f.read_text())
    doc.update(md)
    f.write_text(json.dumps(doc))


@pytest.mark.parametrize("how,needle", [
    ("short", "not +5,000,000"), ("ent", "recorded ent_coef 0.0125"), ("magnet", "recorded magnet_kl_coef 0.05"),
    ("seed", "recorded seed 7112, not 7111"), ("pin", "trained at pin(s)"), ("target", "is not"),
    ("fork", "forked at"), ("teams", "not the offense pin"), ("nofinal", "no final_model.zip"),
    ("gone", "no such run"), ("sha", "is not the registered final")])
def test_the_plan_refuses_each_way_a_responder_can_be_another_run(reader, world, monkeypatch, how, needle):
    R, name = world.R, "rb_x35_x71_x67_s7111"
    if how == "short":
        _spoil(world, name, num_timesteps=T_STEP + 4_999_999)
    elif how == "ent":
        _spoil(world, name, ent_coef=0.0125)
    elif how == "magnet":
        _spoil(world, name, magnet_kl_coef=0.05)
    elif how == "seed":
        _spoil(world, name, cli_args={"seed": 7112})
    elif how == "pin":
        _spoil(world, name, pin_history=[{"git_hash": PIN}, {"git_hash": "f4b25238def57900c0305bb2a3a5a8f9edaef285"}])
    elif how == "target":
        world.records[name]["target_file"] = str(world.models / "rb_x35_c0_s3501" / "final_model.zip")
    elif how == "fork":
        world.records[name]["fork_step"] = T_STEP - 1_000_000
    elif how == "teams":
        world.records[name].update(archetype="balance", membership="exact")
    elif how == "nofinal":
        (world.models / name / "final_model.zip").unlink()
    elif how == "gone":
        (world.models / name).rename(world.models / "elsewhere")
    elif how == "sha":
        monkeypatch.setattr(R, "TARGET_SHA256", dict(R.TARGET_SHA256, x67="0" * 64))
    with pytest.raises(R.X71Error, match=re.escape(needle)):
        reader.plan(1, str(world.models))
    assert reader.main(["plan", "1", "--models-dir", str(world.models)]) == 2


def test_play_banks_one_cell_at_a_time_resumes_and_never_replays(reader, world):
    calls: list = []
    rates = {"c0_r7111": 0.565, "c0_r7112": 0.565, "x67_r7111": 0.665, "x67_r7112": 0.665,
             "swap_r7111": 0.565, "swap_r7112": 0.565, "c0_k0": 0.475, "x67_k0": 0.475, "swap_k0": 0.475}
    play = _player(calls, rates)
    said: list = []
    res = reader.play(1, world.ledger, str(world.models), max_chunks=4, say=said.append, player=play)
    assert res == {"chunks": 9, "banked": 4, "new": 4, "partial": True}
    res = reader.play(1, world.ledger, str(world.models), say=said.append, player=play)
    assert res == {"chunks": 9, "banked": 9, "new": 5, "partial": False}
    assert [c[0] for c in calls] == [f"{t}_{c}" for t in ("c0", "x67", "swap") for c in ("k0", "r7111", "r7112")]
    assert {c[3] for c in calls if c[0].endswith("k0")} == {1_600}
    assert {c[3] for c in calls if not c[0].endswith("k0")} == {800}
    assert all(len(c[4]) == 5 for c in calls)
    assert all(c[2].endswith("/final_model.zip") and Path(c[2]).parent.name == world.R.TARGETS[c[0].split("_")[0]][1]
               for c in calls)
    assert sorted(p.name for p in Path(world.ledger).iterdir()) == sorted(
        f"{t}__{c}.json" for t in ("c0", "x67", "swap") for c in ("k0", "r7111", "r7112"))
    # a third pass plays nothing
    n = len(calls)
    assert reader.play(1, world.ledger, str(world.models), say=said.append, player=play)["new"] == 0
    assert len(calls) == n
    # read 2 reuses read 1's nine cells and plays only X70's three
    res = reader.play(2, world.ledger, str(world.models), say=said.append, player=play)
    assert res == {"chunks": 12, "banked": 12, "new": 3, "partial": False}
    assert [c[0] for c in calls[n:]] == ["x70_k0", "x70_r7111", "x70_r7112"]


def test_a_cell_banked_for_another_file_is_refused_not_replayed(reader, world):
    calls: list = []
    play = _player(calls, {})
    reader.play(1, world.ledger, str(world.models), say=lambda _s: None, player=play)
    (world.models / "rb_x35_x71_swap_s7112" / "final_model.zip").write_bytes(b"another file")
    n = len(calls)
    with pytest.raises(world.R.X71Error, match="banked for another read"):
        reader.play(1, world.ledger, str(world.models), say=lambda _s: None, player=play)
    assert len(calls) == n


def test_the_ledger_is_refused_inside_models(reader, world):
    with pytest.raises(world.R.X71Error, match="OUTSIDE models/"):
        reader.play(1, str(world.models / "_ledger" / "x71"), str(world.models), player=_player([], {}))
    assert not (world.models / "_ledger").exists()


def test_the_read_classes_the_banked_cells_once_and_refuses_an_unplayed_or_second_read(reader, world, tmp_path):
    out = tmp_path / "out" / "verdict_read1.json"
    with pytest.raises(world.R.X71Error, match="not banked"):
        reader.read_verdict(1, world.ledger, str(out), str(world.models))
    assert not out.exists()
    rates = {"c0_r7111": 0.565, "c0_r7112": 0.565, "x67_r7111": 0.665, "x67_r7112": 0.665,
             "swap_r7111": 0.565, "swap_r7112": 0.565, "c0_k0": 0.475, "x67_k0": 0.475, "swap_k0": 0.475}
    reader.play(1, world.ledger, str(world.models), say=lambda _s: None, player=_player([], rates))
    doc = reader.read_verdict(1, world.ledger, str(out), str(world.models))
    assert doc["cls"] == {"x67": "MORE-EXPLOITABLE", "swap": "NOT-MORE-BY-BAR"}
    assert doc["cells"]["x67"]["r7111"] == {"wins": 2_660, "finished": 4_000, "attempted": 4_000, "ties": 0}
    assert doc["cells"]["c0"]["k0"] == {"wins": 3_800, "finished": 8_000, "attempted": 8_000, "ties": 0}
    assert abs(doc["arms"]["x67"]["D"] - 0.10) < 1e-12 and doc["transport"] == "rust_eval"
    assert doc["regime"] == "greedy-vs-greedy" and len(doc["by_team"]["x67"]["k0"]) == 5
    # the same cells through the frozen rule directly: the reader adds nothing to the class
    assert world.R.classify(doc["cells"], 1)["cls"] == doc["cls"]
    assert json.loads(out.read_text())["cls"] == doc["cls"]
    with pytest.raises(world.R.X71Error, match="written ONCE"):
        reader.read_verdict(1, world.ledger, str(out), str(world.models))
    assert reader.main(["read", "1", "--ledger", world.ledger, "--out", str(out), "--models-dir",
                        str(world.models)]) == 2


def test_a_cell_short_of_its_registered_games_is_refused(reader, world, tmp_path):
    reader.play(1, world.ledger, str(world.models), say=lambda _s: None, player=_player([], {}))
    f = Path(world.ledger) / "x67__r7112.json"
    doc = json.loads(f.read_text())
    key = sorted(doc["counts"])[0]
    doc["counts"][key]["attempted"] -= 1
    f.write_text(json.dumps(doc))
    with pytest.raises(world.R.X71Error, match="games attempted"):
        reader.read_verdict(1, world.ledger, str(tmp_path / "v.json"), str(world.models))
