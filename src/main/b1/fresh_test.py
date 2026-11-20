"""The fresh-cell kit (``main.b1.fresh``): the arm file, the argvs, the lever check, the per-run and cross-run guards,
the plan, the play commands and the read, on run folders built the way the trainer leaves them."""
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from main.b1 import arms as A
from main.b1 import cross as X
from main.b1 import fresh as F
from main.b1 import fresh_rule as FR
from main.b1 import look as K
from main.b1.look_test import LEVERS, make_run

PIN = F.B1_LAUNCH_PIN
NEW_PIN = "c" * 40
FRAME = "designs/research_state/measurements/fresh_cell_frame_2026-11-17"
Q5N = {"schema": F.ARM_SCHEMA, "card": "q5n", "pin": PIN, "banked_arm": "C", "lever": ["--n-epochs", "5", "--lr", "6e-4"],
       "declared_keys": ["lr", "n_epochs"],
       "fresh": [{"seed": s, "run": f"rb_q5n_s{s}"} for s in (5201, 5202, 5203)]}
#: The cards' draft arm files and the first-seed argv each card recorded at the pin (FRAME's folder).
CARDS = [("q5n", "argv_q5n_s5201.txt", ["lr", "n_epochs"]), ("q5w", "argv_q5w_s5204.txt", ["lr", "n_epochs"]),
         ("q6n", "argv_q6n_s5211.txt", ["allow_nonproduction_arch", "policy_readout"]),
         ("q6w", "argv_q6w_s5214.txt", ["policy_readout"])]


def spec_of(**over: Any) -> F.ArmSpec:
    return F.arm_from_doc({**Q5N, **over})


#: What a launcher argv resolves to, for the guards' tests: every typed flag over the defaults below (the real
#: parsers are held by the lever-check tests against the cards' own argvs).
DEFAULTS: Dict[str, Any] = {"d_model": 128, "trunk_ffn_dim": 256, "team_mix": None, "team_mix_sides": None,
                            "n_epochs": 10, "lr": 0.0003, "policy_readout": "tower", "allow_nonproduction_arch": False}


def _value(tok: str) -> Any:
    for kind in (int, float):
        try:
            return kind(tok)
        except ValueError:
            pass
    return tok


def stub_resolver(argvs: Sequence[Sequence[str]], extra: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    out = []
    for argv in argvs:
        ns, toks = dict(DEFAULTS), list(argv)
        for i, t in enumerate(toks):
            if t.startswith("--"):
                key = t[2:].replace("-", "_")
                ns[key] = True if i + 1 == len(toks) or toks[i + 1].startswith("--") else _value(toks[i + 1])
        if extra and "--n-epochs" in argv:
            ns.update(extra)
        out.append(ns)
    return out


def _stub_k9b(run_dir: Path, upto: Optional[int]) -> Dict[str, Any]:
    return {"schema": K.K9B_SCHEMA, "run": run_dir.name, "verdict": "PASS", "upto": upto, "fires": [], "reasons": []}


def make_cell_run(models: Path, spec: F.ArmSpec, run: F.Run, *, argv: Optional[List[str]] = None,
                  cli: Optional[Dict[str, Any]] = None, pin: Optional[str] = None,
                  steps: Tuple[int, ...] = (50_000_081, 100_000_044), control: bool = False, **over: Any) -> Path:
    """A fresh (or own-control) run folder as the trainer leaves it, launched by its registered argv."""
    pin = pin or spec.pin
    d = models / run.run
    (d / "checkpoints").mkdir(parents=True)
    argv = argv or (F.control_argv(spec, 50_000_000, run) if control else F.fresh_argv(spec, run, 50_000_000))
    ns = stub_resolver([argv])[0]
    history = [{"git_hash": pin, "pin_source": "pin_commit", "first_step": 0, "last_step": 100_050_000, "torch": "2.8"}]
    md: Dict[str, Any] = {
        "cli_args": {"seed": run.seed, "ent_coef": 0.0125, **LEVERS[spec.banked_arm],
                     "n_epochs": ns["n_epochs"], "lr": ns["lr"],
                     **(cli or {})},
        "original_command": " ".join(["/x/src/main/launcher/__main__.py", *argv]), "ent_coef": 0.0125,
        "pin_history": history, "lineage": {"role": "fresh", "fork_parent": None},
        "torch_version": "2.8.0+cu126", "init_num_threads": 8, **over}
    (d / "metadata.json").write_text(json.dumps(md))
    for n in steps:
        z = d / "checkpoints" / f"checkpoint_{n}_steps.zip"
        z.write_bytes(f"{run.run}@{n}".encode())
        z.with_suffix(".json").write_text(json.dumps({"num_timesteps": n, "ent_coef": 0.0125, "git_hash": pin,
                                                      "pin_history": [{**history[0], "last_step": 43_210_987}]}))
    return d


def make_banked(models: Path, arm: str = "C", **kw: Any) -> None:
    for seed in A.SEEDS[K.KIT_ARM[arm]]:
        make_run(models, arm, seed, pin=PIN, steps=(50_000_081, 100_000_044), **kw)


def b1_look_of(models: Path, look: str = "50M", arm: str = "C") -> Dict[str, Any]:
    """B1's own look.json for ``arm``, as its kit writes one: the runs' look files and their sha256."""
    from utils.hashing import sha256_file

    out = []
    for seed in A.SEEDS[K.KIT_ARM[arm]]:
        run = A.run_name(K.KIT_ARM[arm], seed)
        z, n, _ = K.find_checkpoint(models / run, F.LOOK_STEPS[look])
        out.append({"run": run, "steps": n, "sha256": sha256_file(str(z)), "checkpoint": str(z)})
    return {"look": look, "pin": PIN, "survivor": None, "runs": {arm: out}}


@pytest.fixture()
def world(tmp_path):
    models = tmp_path / "models"
    spec = spec_of()
    make_banked(models)
    for r in spec.fresh:
        make_cell_run(models, spec, r)
    return models, spec


def plan(models: Path, spec: F.ArmSpec, look: str = "50M", **kw: Any):
    kw.setdefault("b1_look", b1_look_of(models, look) if spec.own_control is None else None)
    return F.plan(models, spec, look, kw.pop("resolver", stub_resolver), k9b=_stub_k9b, **kw)


# ------------------------------------------------------------------------------------------- the arm file
def test_a_cards_arm_file_loads_and_every_malformed_one_is_refused_by_name():
    s = spec_of()
    assert (s.card, s.banked_arm, s.lever, s.declared_keys) == ("q5n", "C", ("--n-epochs", "5", "--lr", "6e-4"),
                                                                ("lr", "n_epochs"))
    assert F.banked_runs(s) == tuple(F.Run(x, A.run_name("c", x)) for x in (5101, 5102, 5110))
    assert F.banked_runs(s, replace=5102)[1] == F.Run(5109, "rb_b1c_s5109")
    bad = [({"pin": NEW_PIN}, "not B1's launch pin"),
           ({"lever": ["--seed", "1"]}, "already types"), ({"lever": ["--model", "x.zip"]}, "no lever may"),
           ({"lever": ["--ent-coef=0.05"]}, "already types"), ({"lever": []}, "non-empty list"),
           ({"declared_keys": ["seed", "lr"]}, "none of"), ({"declared_keys": []}, "non-empty list"),
           ({"fresh": Q5N["fresh"][:2]}, "2 fresh runs against a banked cell of 3"),
           ({"fresh": [{"seed": 5101, "run": "rb_q5n_s5101"}] + Q5N["fresh"][1:]}, "BOUNDARY-1's own"),
           ({"fresh": [{"seed": 5201, "run": "rb_q5n_s5209"}] + Q5N["fresh"][1:]}, "ending in _s5201"),
           ({"fresh": Q5N["fresh"][:1] * 3}, "appears twice"), ({"card": "Q5-N"}, "lower-case"),
           ({"banked_arm": "Br"}, "not one of"), ({"extra": 1}, "unknown fields"), ({"schema": "x"}, "not a")]
    for over, why in bad:
        with pytest.raises(F.FreshError, match=why):
            F.arm_from_doc({**Q5N, **over})
    own = spec_of(pin=NEW_PIN, own_control=[{"seed": s, "run": f"rb_q5c_s{s}"} for s in (5221, 5222, 5223)])
    assert F.banked_runs(own) == own.own_control
    with pytest.raises(F.FreshError, match="an own-control cell has none"):
        F.banked_runs(own, replace=5102)


@pytest.mark.parametrize("card,record,keys", CARDS)
def test_each_cards_arm_file_builds_the_argv_the_card_recorded_token_for_token(card, record, keys):
    """FRAME §2 item 2's construction, held against the four argvs the cards recorded at the pin."""
    from utils.paths import repo_path

    s = F.load_arm(repo_path(f"{FRAME}/fresh_arm_{card}.json"))
    assert s.card == card and s.pin == PIN and list(s.declared_keys) == keys and s.own_control is None
    want = shlex.split(repo_path(f"{FRAME}/{record}").read_text())
    assert F.fresh_argv(s, s.fresh[0], 50_000_000) == want
    assert F.fresh_argv(s, s.fresh[0], 100_000_000) == [("100000000" if t == "50000000" else t) for t in want]
    assert F.resume_argv(s, s.fresh[0]) == ["--restart-interval-hours", "6", "--pin-commit", PIN, "--model",
                                            f"models/{s.fresh[0].run}/final_model.zip", "--steps", "100000000",
                                            "--device", "cuda", "--log-level", "periodic"]


# ------------------------------------------------------------------------------- a lost fresh run (PREREG §4)
#: The registered cards' folders: each holds the PREREG and the arm file it registered.
REGISTERED = {"q6n": "q6n_policy_readout_fresh_2026-11-20", "q5n": "q5n_epochs_fresh_2026-11-20"}


def registered_doc(card: str) -> Dict[str, Any]:
    from utils.paths import repo_path

    return json.loads(repo_path(f"designs/research_state/measurements/{REGISTERED[card]}/fresh_arm_{card}.json").read_text())


def amended(doc: Dict[str, Any], row: int, seed: int, run: Optional[str] = None) -> Dict[str, Any]:
    """The registered amendment: the lost row's seed and run name, and nothing else."""
    fresh = [dict(r) for r in doc["fresh"]]
    fresh[row] = {"seed": seed, "run": run or f"rb_{doc['card']}_s{seed}"}
    return {**doc, "fresh": fresh}


def test_the_kits_registered_cells_are_the_committed_arm_files_and_the_preregs_spares():
    from utils.paths import repo_path

    assert set(F.REGISTERED_CELLS) == set(REGISTERED)
    for card, (seeds, spare) in F.REGISTERED_CELLS.items():
        doc = registered_doc(card)
        spec = F.arm_from_doc(doc)
        assert tuple(r.seed for r in spec.fresh) == seeds and spec.replaced_fresh is None
        assert spare not in seeds and doc["prereg"].endswith(f"{REGISTERED[card]}/PREREG.md")
        prereg = repo_path(doc["prereg"]).read_text()
        assert f"spare {spare}" in prereg and f"`rb_{card}_s{spare}`" in prereg


@pytest.mark.parametrize("card", sorted(REGISTERED))
@pytest.mark.parametrize("row", [0, 1, 2])
def test_the_registered_amendment_puts_the_spare_in_the_lost_row_and_builds_its_argv(tmp_path, card, row):
    seeds, spare = F.REGISTERED_CELLS[card]
    doc = registered_doc(card)
    path = tmp_path / f"fresh_arm_{card}.json"
    path.write_text(json.dumps(amended(doc, row, spare), indent=1))
    was, now = F.arm_from_doc(doc), F.load_arm(path)
    assert now.replaced_fresh == seeds[row] and now.fresh[row] == F.Run(spare, f"rb_{card}_s{spare}")
    assert [r for i, r in enumerate(now.fresh) if i != row] == [r for i, r in enumerate(was.fresh) if i != row]
    assert (now.pin, now.lever, now.declared_keys, now.banked_arm) == (was.pin, was.lever, was.declared_keys,
                                                                       was.banked_arm)
    swap = {str(seeds[row]): str(spare), f"rb_{card}_s{seeds[row]}": f"rb_{card}_s{spare}"}
    for steps in A.PHASE_STEPS:                      # from step 0, the lost row's argv with the spare's seed and name
        assert F.fresh_argv(now, now.fresh[row], steps) == [swap.get(t, t) for t in F.fresh_argv(was, was.fresh[row], steps)]
    assert F.find_run(now, f"rb_{card}_s{spare}") == ("fresh", now.fresh[row])
    with pytest.raises(F.FreshError, match="not a fresh or own-control run"):
        F.find_run(now, f"rb_{card}_s{seeds[row]}")
    assert F.main(["describe", "--arm", str(path)]) == 0


@pytest.mark.parametrize("card", sorted(REGISTERED))
def test_a_registered_card_takes_one_replacement_by_its_spare_and_nothing_else(card):
    seeds, spare = F.REGISTERED_CELLS[card]
    doc = registered_doc(card)
    second = amended(amended(doc, 0, spare), 1, spare + 1)             # a second loss: there is no second spare
    other = amended(doc, 1, spare - 1)                                 # a seed that is not the registered spare
    moved = {**doc, "fresh": [doc["fresh"][1], doc["fresh"][0], doc["fresh"][2]]}
    for bad in (second, other, moved):
        with pytest.raises(F.FreshError, match="at most ONE row replaced by its spare"):
            F.arm_from_doc(bad)
    with pytest.raises(F.FreshError, match="not its registered run names"):
        F.arm_from_doc(amended(doc, 2, spare, run=f"rb_{card}x_s{spare}"))
    draft = F.arm_from_doc({**Q5N, "card": "q9z", "fresh": [{"seed": s, "run": f"rb_q9z_s{s}"} for s in (1, 2, 3)]})
    assert draft.replaced_fresh is None                                # a card nothing registered is not held


def test_a_replaced_fresh_run_plans_like_any_fresh_run_and_the_look_records_it(world):
    models, spec = world
    first, _ = plan(models, spec)
    assert first is not None and first["replaced_fresh_seed"] is None
    now = F.arm_from_doc(amended({**Q5N}, 1, 5209), sha256="amended")
    spare = now.fresh[1]
    doc, probs = plan(models, now)
    assert doc is None and probs == [f"rb_q5n_s5209: no run folder under {models}"]     # the lost run is not read
    make_cell_run(models, now, spare)
    doc, probs = plan(models, now)
    assert probs == [] and doc is not None
    assert doc["replaced_fresh_seed"] == 5202 and doc["replaced_seed"] is None and doc["arm_sha256"] == "amended"
    assert [f["run"] for f in doc["runs"]["fresh"]] == ["rb_q5n_s5201", "rb_q5n_s5209", "rb_q5n_s5203"]
    assert len(doc["cells"]) == 9 and not any("rb_q5n_s5202" in z for cell in doc["cells"] for z in cell)
    assert doc["names"] == first["names"]                              # one family and one request a card and look
    assert doc["arm"]["fresh"][1] == {"seed": 5209, "run": "rb_q5n_s5209"}


def test_a_spare_run_that_is_not_its_own_argv_from_step_zero_is_refused(tmp_path):
    """The replacement is "from step 0 with the spare seed": the lost row's argv, or a fork of the lost run, fails."""
    for k, (kw, why) in enumerate([
            ({"argv": F.fresh_argv(spec_of(), spec_of().fresh[1], 50_000_000)},
             "original_command is not the registered argv of rb_q5n_s5209"),
            ({"lineage": {"role": "fork", "fork_parent": "models/rb_q5n_s5202/checkpoints/x.zip"}}, "fork")]):
        models = tmp_path / f"m{k}"
        now = F.arm_from_doc(amended({**Q5N}, 1, 5209))
        make_banked(models)
        for r in now.fresh:
            make_cell_run(models, now, r, **(kw if r.seed == 5209 else {}))
        doc, probs = plan(models, now)
        assert doc is None and any("rb_q5n_s5209" in p and why in p for p in probs), probs


# -------------------------------------------------------------------------------------------- the lever check
def test_the_lever_check_passes_the_cards_levers_with_the_real_parser_and_fails_an_undeclared_key():
    """THIS checkout's parser (the plan's is the pin's, :func:`main.b1.fresh.pin_tree_resolver`): each card's lever
    moves exactly its declared keys at both targets (the cards' diff records, FRAME's folder); declaring less, or
    more, fails."""
    q5 = F.lever_check(spec_of(), F.in_process_resolver)
    assert q5["ok"] and q5["problems"] == []
    for t in q5["targets"].values():
        assert all(sorted(r["diff"]) == ["lr", "n_epochs", "run_name", "seed"] for r in t["runs"].values())
    assert q5["expected"]["rb_q5n_s5201"]["n_epochs"] == 5 and q5["expected"]["rb_q5n_s5201"]["lr"] == 0.0006
    assert q5["banked_levers"] == LEVERS["C"]
    from utils.paths import repo_path

    for card, _record, keys in CARDS:
        got = F.lever_check(F.load_arm(repo_path(f"{FRAME}/fresh_arm_{card}.json")), F.in_process_resolver)
        assert got["ok"], (card, got["problems"])
        assert got["banked_levers"] == LEVERS["W" if card.endswith("w") else "C"]
    short = F.lever_check(spec_of(declared_keys=["n_epochs"]), F.in_process_resolver)
    assert not short["ok"] and len(short["problems"]) == 6
    assert "differs from the control's in ['lr', 'n_epochs'] beyond its seed and name, not ['n_epochs']" \
        in short["problems"][0]
    wide = F.lever_check(spec_of(declared_keys=["lr", "n_epochs", "clip_range"]), F.in_process_resolver)
    assert not wide["ok"]


def test_the_pin_resolver_refuses_a_tree_that_is_not_clean_at_the_pin(tmp_path):
    with pytest.raises(F.FreshError, match="not a git checkout"):
        F.pin_tree_resolver(tmp_path, PIN)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"],
                   check=True)
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    with pytest.raises(F.FreshError, match="not the pin"):
        F.pin_tree_resolver(tmp_path, PIN)
    assert F.tree_problems(tmp_path, head) == []
    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    assert F.tree_problems(tmp_path, head) == [f"{tmp_path}: src/ or data/ is dirty (or unreadable)"]


# --------------------------------------------------------------------------------------------------- the plan
def test_a_clean_cell_plans_nine_cells_fresh_against_banked_with_the_cards_names(world):
    models, spec = world
    doc, probs = plan(models, spec)
    assert probs == [] and doc is not None
    assert [f["run"] for f in doc["runs"]["fresh"]] == [r.run for r in spec.fresh]
    assert [f["run"] for f in doc["runs"]["banked"]] == ["rb_b1c_s5101", "rb_b1c_s5102", "rb_b1c_s5110"]
    assert len(doc["cells"]) == 9 and all("rb_q5n_" in p and "rb_b1c_" in o for p, o in doc["cells"])
    assert all(f["steps"] == 50_000_081 for s in FR.SIDES for f in doc["runs"][s])
    assert doc["names"]["home72"] == {"family": "b1_fresh_q5n_h2h_home72", "request": "b1fresh_q5n_50m_home72",
                                      "pairs": 1000, "rule": F.family_rule("q5n", "home72", PIN)}
    assert doc["banked_identity"]["checked"] is True and doc["lever_check"]["ok"]
    assert all("_cli" not in f for s in FR.SIDES for f in doc["runs"][s])
    doc100, _ = plan(models, spec, "100M")
    assert doc100 is not None and doc100["names"]["p72"]["request"] == "b1fresh_q5n_100m_p72"


@pytest.mark.parametrize("plant,why", [
    # launched WITHOUT the lever (B1's own argv at a fresh seed and name): the argv guard refuses it
    (lambda m, s, r: make_cell_run(m, s, r, argv=F.control_argv(s, 50_000_000, r), cli={"n_epochs": 5}),
     "original_command is not the registered argv of rb_q5n_s5202"),
    # launched with the lever, but its last process ran ten epochs (a resume that dropped it)
    (lambda m, s, r: make_cell_run(m, s, r, cli={"n_epochs": 10}), "cli_args.n_epochs 10 is not its argv's 5"),
    # trained at another commit
    (lambda m, s, r: make_cell_run(m, s, r, pin="d" * 40), "not one entry at 15136b61"),
    # a knob nobody declared moved in its last process
    (lambda m, s, r: make_cell_run(m, s, r, cli={"clip_range": 0.3}), "differs from rb_b1c_s5101's in ['clip_range']"),
    # another torch
    (lambda m, s, r: make_cell_run(m, s, r, torch_version="2.9.0"), "torch_version differs across the runs"),
    # a fork, not a fresh run
    (lambda m, s, r: make_cell_run(m, s, r, lineage={"role": "fork", "fork_parent": "x"}), "not a FRESH run"),
])
def test_a_fresh_run_that_is_not_exactly_the_lever_refuses_the_plan(tmp_path, plant, why):
    models = tmp_path / "models"
    spec = spec_of()
    make_banked(models)
    for r in spec.fresh:
        (plant if r.seed == 5202 else make_cell_run)(models, spec, r)
    doc, probs = plan(models, spec)
    assert doc is None and any(why in p for p in probs), probs


def test_a_lever_that_moves_an_undeclared_key_or_a_missing_or_foreign_b1_look_refuses(world):
    models, spec = world
    doc, probs = plan(models, spec, resolver=lambda a: stub_resolver(a, {"clip_range": 0.3}))
    assert doc is None and probs[0].startswith("lever check: rb_q5n_s5201 @ 50,000,000")
    doc, probs = plan(models, spec, b1_look=None)
    assert doc is None and "no B1 look.json given" in probs[0]
    other = b1_look_of(models)
    other["runs"]["C"][1]["sha256"] = "0" * 64
    doc, probs = plan(models, spec, b1_look=other)
    assert doc is None and probs == ["rb_b1c_s5102: its 50M file is step 50000081 sha "
                                     f"{doc_sha(models, 'rb_b1c_s5102')[:12]}, B1 read step 50000081 sha 000000000000"]
    doc, probs = plan(models, spec, b1_look=b1_look_of(models, "100M"))
    assert doc is None and "not B1's full 50M look" in probs[0]


def doc_sha(models: Path, run: str) -> str:
    from utils.hashing import sha256_file

    return sha256_file(str(K.find_checkpoint(models / run, 50_000_000)[0]))


def test_a_banked_run_is_held_to_b1s_own_guard(tmp_path):
    """The banked runs go through :func:`main.b1.look.check_run` itself: one launched by another argv is refused."""
    models = tmp_path / "models"
    spec = spec_of()
    make_banked(models)
    bad = models / "rb_b1c_s5110" / "metadata.json"
    md = json.loads(bad.read_text())
    md["original_command"] = md["original_command"].replace("--ent-coef 0.0125", "--ent-coef 0.05")
    bad.write_text(json.dumps(md))
    for r in spec.fresh:
        make_cell_run(models, spec, r)
    doc, probs = plan(models, spec)
    assert doc is None and probs == [f"rb_b1c_s5110: original_command is not the registered argv of arm C seed 5110: "
                                     f"{' '.join(shlex.split(md['original_command'])[1:])!r}"]


def test_an_own_control_cell_at_a_new_pin_is_guarded_like_the_fresh_runs_without_the_lever(tmp_path):
    models = tmp_path / "models"
    spec = spec_of(pin=NEW_PIN, own_control=[{"seed": s, "run": f"rb_q5c_s{s}"} for s in (5221, 5222, 5223)])
    for r in spec.fresh:
        make_cell_run(models, spec, r)
    for r in spec.own_control:
        make_cell_run(models, spec, r, control=True)
    doc, probs = plan(models, spec)
    assert probs == [] and doc is not None and doc["own_control"] and not doc["banked_identity"]["checked"]
    assert [f["run"] for f in doc["runs"]["banked"]] == ["rb_q5c_s5221", "rb_q5c_s5222", "rb_q5c_s5223"]
    # a control run that carries the lever is not a control
    (models / "rb_q5c_s5222").rename(models / "gone")
    make_cell_run(models, spec, spec.own_control[1], argv=F.fresh_argv(spec, spec.own_control[1], 50_000_000))
    doc, probs = plan(models, spec)
    assert doc is None and any("original_command is not the registered argv of rb_q5c_s5222" in p for p in probs)


def test_the_guard_override_exists_for_tests_and_the_rehearsal_alone(world):
    """The plan's ``guard`` / ``cross`` / ``step`` keywords are the tests' (the CLI has no flag for them): a guard
    that passes everything plans runs the registered guard refuses."""
    models, spec = world
    (models / "rb_q5n_s5203" / "metadata.json").write_text(json.dumps({}))
    assert plan(models, spec)[0] is None
    k = F.lever_check(spec, stub_resolver)
    ok = F.default_guard(spec, k, _stub_k9b)

    def lenient(side: str, run_dir: Path, run: F.Run, step: int):
        if run.run == "rb_q5n_s5203":
            return [], dict(ok(side, models / "rb_q5n_s5202", run, step)[1], run=run.run)
        return ok(side, run_dir, run, step)

    doc, probs = plan(models, spec, guard=lenient, cross=lambda s, f: [])
    assert probs == ["two runs' checkpoints have the same sha256"]  # every other guard was waived


def test_the_kits_ledger_read_is_the_h2h_family_read_under_its_own_name():
    assert F.FRESH_READ.name == "main.b1.fresh.family"
    assert {k: v for k, v in vars(F.FRESH_READ).items() if k != "name"} == \
        {k: v for k, v in vars(K.LOOK_READ).items() if k != "name"}


def test_the_cli_has_no_flag_for_the_tests_overrides(capsys):
    with pytest.raises(SystemExit):
        F.main(["plan", "--help"])
    text = capsys.readouterr().out
    assert "--b1-look" in text and not any(f in text for f in ("--guard", "--step", "--cross", "--k9b", "--pairs"))


# --------------------------------------------------------------------------------------------------- play
def test_play_runs_drive_looks_commands_in_the_pin_checkout_and_holds_on_a_failed_panel(world, tmp_path):
    models, spec = world
    doc, _ = plan(models, spec)
    work = tmp_path / "w"
    work.mkdir()
    (work / "cells_plan.json").write_text(json.dumps(doc["cells"]))
    cmds = F.play_commands(doc, work, tmp_path, "cpu", py="PY", ledger=tmp_path / "L", agent_cpus="1-2,9-10")
    assert [(p, w) for p, w, _ in cmds] == [(p, w) for p in X.PANELS for w in ("family-register", "play-many")]
    fam, game = cmds[0][2], cmds[1][2]
    assert fam[:4] == ["PY", "-m", "main.eval_ledger", "family-register"] and fam[fam.index("--commit") + 1] == PIN
    assert fam[fam.index("--root") + 1] == str(tmp_path / "L")
    assert game[:6] == ["nice", "-n", "19", "taskset", "-c", "1-2,9-10"] and game[7] == "24"
    i = game.index("main.h2h")
    assert game[i + 1:i + 6] == ["play-many", "--cells", str(work / "cells_plan.json"), "--pairs", "1000"]
    for flag, val in (("--seed", "0"), ("--purpose", "ab"), ("--oracle-reveal-mode", "off"), ("--batch-pairs", "500"),
                      ("--request", "b1fresh_q5n_50m_home72"), ("--family", "b1_fresh_q5n_h2h_home72"),
                      ("--threads", "2"), ("--out", str(tmp_path / "L"))):
        assert game[game.index(flag) + 1] == val
    cuda = F.play_commands(doc, work, tmp_path, "cuda", py="PY", max_s=600)[3][2]
    assert cuda[0].endswith("gpu_lock.sh") and cuda[1].endswith("mem_cap.sh") and cuda[2:5] == ["--max-s", "600", "32"]
    assert cuda[cuda.index("--pairs") + 1] == "500" and "--out" not in cuda and "--threads" not in cuda
    small = F.play_commands(doc, work, tmp_path, "cuda", py="PY", pairs={p: 2 for p in X.PANELS})[1][2]
    assert small[small.index("--pairs") + 1] == "2" and small[small.index("--batch-pairs") + 1] == "2"

    (work / "look.json").write_text(json.dumps(doc))
    assert F.play(work, tmp_path / "nowhere", "cpu", runner=lambda *a: 0) == 5
    pin_dir = tmp_path / "pin"
    subprocess.run(["git", "init", "-q", str(pin_dir)], check=True)
    subprocess.run(["git", "-C", str(pin_dir), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
                    "--allow-empty", "-m", "x"], check=True)
    assert F.play(work, pin_dir, "cpu", runner=lambda *a: 0) == 5  # not at the plan's pin
    head = subprocess.run(["git", "-C", str(pin_dir), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    (work / "look.json").write_text(json.dumps(dict(doc, pin=head)))
    seen: List[List[str]] = []

    def runner(argv, env, cwd, log):
        seen.append(list(argv))
        assert env["PYTHONPATH"] == str(pin_dir / "src") and env["CUDA_VISIBLE_DEVICES"] == "" and cwd == pin_dir
        return 1 if "play-many" in argv and "forum72" in argv else 0

    assert F.play(work, pin_dir, "cpu", runner=runner) == 4
    assert [a[2] for a in seen[:2]] == ["main.sidecar_audit", "main.checkargs"]
    assert sum("play-many" in a for a in seen) == 3  # home72, heldout72_hpfix, then forum72 fails and play stops
    assert "HOLD: play-many failed on forum72" in (work / "run.log").read_text()


# --------------------------------------------------------------------------------------------------- the read
@pytest.mark.integration
def test_the_read_maps_real_ledger_rows_to_the_cross_and_rechecks_from_its_inputs(world, tmp_path, monkeypatch):
    """The whole read path on a real count ledger (``main.h2h``'s row builder, B1's declared read, the row checks):
    at 2 pairs a cell the registered read is INCONCLUSIVE (short), and with the tests' pairs it classes the cross.
    The team bootstrap reproduces the statistic from the rows' team tallies before it resamples."""
    from main.b1.look_test import _write_look

    models, spec = world
    doc, _ = plan(models, spec)
    root = tmp_path / "ledger"
    _write_look(root, doc, commit=PIN)
    inp, v = F.read_ledger(root, doc)
    assert v["decision"]["outcome"] == FR.INCONCLUSIVE and len(v["decision"]["reasons"]) == 36
    assert all("2 completed pairs < the registered" in r for r in v["decision"]["reasons"])
    two = {p: 2 for p in X.PANELS}
    inp, v = F.read_ledger(root, doc, pairs=two)
    assert v["decision"]["outcome"] == FR.STOPPED_AT_50M and v["views"]["N"]["estimate_pp"] == 0.0
    cells, rows = F.ledger_rows(root, doc)
    rep = F.bootstrap(doc, cells, rows, draws=200, pairs=two)
    assert rep["status"] == "REPORTED" and rep["outcome_registered"] == FR.STOPPED_AT_50M
    assert rep["views"]["N"]["estimate_pp"] == 0.0 and rep["teams"] == {p: 4 for p in X.PANELS}

    # the CLI: read writes inputs + verdict, recheck re-decides from inputs alone, a tampered verdict fails it
    work = tmp_path / "w"
    work.mkdir()
    (work / "look.json").write_text(json.dumps(doc))
    monkeypatch.setenv("GEN3AI_MODELS_DIR", str(models))
    assert F.main(["read", "--work", str(work), "--ledger", str(root)]) == 0
    assert json.loads((work / "result" / "verdict.json").read_text())["decision"]["outcome"] == FR.INCONCLUSIVE
    assert F.main(["recheck", "--work", str(work)]) == 0
    v = json.loads((work / "result" / "verdict.json").read_text())
    v["decision"]["outcome"] = X.WIN
    (work / "result" / "verdict.json").write_text(json.dumps(v))
    assert F.main(["recheck", "--work", str(work)]) == 3
    assert F.main(["bootstrap", "--work", str(work), "--ledger", str(root)]) == 3  # the registered read is short


def test_the_cli_plan_refuses_without_a_clean_pin_tree_and_describe_prints_the_argvs(tmp_path, capsys):
    arm = tmp_path / "fresh_arm.json"
    arm.write_text(json.dumps(Q5N))
    assert F.main(["describe", "--arm", str(arm)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 9 and out[0] == f"python -m main.launcher {shlex.join(F.fresh_argv(spec_of(), spec_of().fresh[0], 50_000_000))}"
    assert out[2].endswith("# phase 2") and "--model models/rb_q5n_s5201/final_model.zip" in out[2]
    assert F.main(["plan", "--arm", str(arm), "--look", "50M", "--pin-tree", str(tmp_path), "--out",
                   str(tmp_path / "w")]) == 2
    assert "not a git checkout" in capsys.readouterr().err and not (tmp_path / "w").exists()
