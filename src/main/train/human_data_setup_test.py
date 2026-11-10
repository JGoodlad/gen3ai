"""``--human-data`` at the trainer's surfaces (``human_data_v1``; ``designs/endstate/human_data_architecture.md`` §6):
the parser, the resume / fork refusal, one lever per run, the ``teams`` clause resolving to the team-mix path (sides
both, the whole corpus), the ``init`` clause copying a registered clone's trained tensors into a fresh learner, the
``opp`` clause deriving the stable-opponent flags, the null doses, off-means-off, one entry point per human-trained
model, the lineage keys, ``main.checkargs`` and the launcher's restart argv.

Pure unit tests over a SYNTHETIC human corpus and a SYNTHETIC human-trained model in ``tmp_path``
(``agents.training.team_mix_testkit``, ``utils.human_models_testkit``); the Node validator is mocked where a builder
is built. Each test names the revert it fails on.
"""
import json
from types import SimpleNamespace
from unittest import mock

import pytest

from agents.info_levers import human_data as HD
from agents.training.matchup_spec import MatchupSpec
from agents.training.team_mix_testkit import TEST_HUMAN_CORPUS, clear_corpus_cache, install_human_corpus
from main.train import human_data_setup as HS
from main.train import matchup_setup as MS
from utils.human_models_testkit import TEST_HUMAN_MODEL, TEST_ROWSET, install_human_model, tiny_state
from utils.teambuilder_test import TEAM_A, TEAM_B

CLAUSE = f"teams:{TEST_HUMAN_CORPUS}@0.5"
INIT = f"init:{TEST_HUMAN_MODEL}@1"
OPP = f"opp:{TEST_HUMAN_MODEL}@0.2"


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_corpus_cache()
    yield
    clear_corpus_cache()


@pytest.fixture
def human(monkeypatch, tmp_path):
    return install_human_corpus(monkeypatch, tmp_path, [TEAM_B, TEAM_A])


@pytest.fixture
def clone(monkeypatch, tmp_path):
    """A registered synthetic human-trained model whose TRAINED tensors are ``a.weight`` and ``a.bias``."""
    return install_human_model(monkeypatch, tmp_path)


@pytest.fixture
def all_valid():
    with mock.patch("utils.bridge.team_validator.validate_teams_locally",
                    side_effect=lambda fmt, ts: [{"valid": True} for _ in ts]):
        yield


def _args(**kw):
    base = dict(trainee_team=None, trainee_teams=None, exploiter=None, self_play=False, bot_weights=None,
                exploiter_keep_bots=False, exploiter_bot_fraction=0.5, stable_opponent_temp=1.0,
                team_mix=None, team_mix_sides=None, model=None, run_dir=None, run_name=None, human_data=None,
                stable_opponents=None, stable_opponent_selfplay_share=0.2, debug=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _run(tmp_path, name, cli, lineage=None):
    run = tmp_path / name
    (run / "checkpoints").mkdir(parents=True)
    ckpt = run / "checkpoints" / "ckpt_100.zip"
    ckpt.write_bytes(b"")
    meta = {"cli_args": cli}
    if lineage is not None:
        meta["lineage"] = lineage
    (run / "metadata.json").write_text(json.dumps(meta))
    return run, ckpt


# ── the parser ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_parser_takes_the_clause_as_typed_and_defaults_to_untyped():
    from main.train.parser import build_parser

    assert build_parser().parse_args([]).human_data is None
    assert build_parser().parse_args(["--human-data", "teams:hr_x.r1@.50"]).human_data == "teams:hr_x.r1@.50"


@pytest.mark.parametrize("value", ["on", "teams:hr_x@2", "judge:hom@1", "teams:hr_x", "bc:hr_spec_v1.r2@0.5",
                                   "kl:hom@0.1", "init:hclone_t@0.5", "opp:hclone_t@2"])
def test_parser_refuses_a_value_outside_the_grammar(value):
    """Fails on revert of the ``type=`` syntax check."""
    from main.train.parser import build_parser

    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(["--human-data", value])
    assert e.value.code == 2


# ── off means off ──────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("typed", [None, "off"])
def test_off_opens_no_corpus_path_and_leaves_the_matchup_unchanged(monkeypatch, typed):
    """OFF MEANS OFF: no human corpus is read, the spec carries no mix and its hash is a no-flag spec's. Fails on
    revert of the ``lever != 'teams'`` early return in ``team_mix_from_args``."""
    from utils.team_corpora import human as H

    def _boom(*a, **k):
        raise AssertionError("off opened a human corpus path")

    from utils import human_models as HM

    monkeypatch.setattr(H, "load", _boom)
    monkeypatch.setattr(H, "_measure", _boom)
    monkeypatch.setattr(HM, "load", _boom)
    monkeypatch.setattr(HM, "registry", _boom)
    d = HS.decide(_args(human_data=typed))
    assert d.refusal is None and d.record == HD.record(HD.parse("off"))
    assert d.lines == [HD.stamp_line(d.record)] and "lever=off rowset=- rows_sha=-" in d.lines[0]
    base = MatchupSpec.from_args(SimpleNamespace(**{k: v for k, v in vars(_args()).items() if k != "human_data"}))
    spec = MatchupSpec.from_args(_args(human_data=typed))
    assert spec.trainee_teams.mix is None and spec.opponent_teams.mix is None
    assert spec.spec_hash() == base.spec_hash()


# ── teams resolves to the team-mix path, sides both, the whole corpus ────────────────────────────────────────

def test_teams_resolves_to_a_both_sides_mix_over_the_whole_corpus(human):
    """Fails on revert of the ``teams`` branch in ``team_mix_from_args`` (the clause would stamp and train nothing)."""
    spec = MatchupSpec.from_args(_args(human_data=CLAUSE))
    for side in (spec.trainee_teams, spec.opponent_teams):
        assert side.mix is not None and side.mix.corpus == TEST_HUMAN_CORPUS and side.mix.tier == "ALL"
        assert side.mix.n == 2 and side.mix.share == 0.5
    assert spec.eval_trainee_teams.mix is None          # eval stays on the pool
    # exactly `--team-mix <corpus>@ALL:0.5 --team-mix-sides both` over the same verified corpus
    from agents.training.matchup_spec import TeamMix
    from utils.team_corpora import human as H
    assert spec.trainee_teams.mix == TeamMix.from_corpus(H.load(TEST_HUMAN_CORPUS), 0.5)


def test_decide_stamps_the_corpus_identity(human):
    from utils.team_corpora import human as H

    d = HS.decide(_args(human_data=CLAUSE))
    s = H.stub(TEST_HUMAN_CORPUS)
    assert d.refusal is None and d.record["canonical"] == CLAUSE and d.record["rowset"] == TEST_HUMAN_CORPUS
    assert d.record["corpus"]["data_sha256"] == s["data_sha256"]
    assert d.record["corpus"]["manifest_sha256"] == s["manifest_sha256"] and d.record["corpus"]["human_data"]
    assert d.lines[0].startswith(f"[HUMAN-DATA] lever=teams rowset={TEST_HUMAN_CORPUS} rows_sha={s['data_sha256']}")
    args, lines = _args(human_data=CLAUSE), []
    assert HS.resolve(args, emit=lines.append) == d.record and args._human_data == d.record and lines == d.lines


def test_the_null_dose_loads_and_stamps_but_never_draws(human, all_valid):
    """``@0``: the corpus is verified and stamped, the spec records a 0-share mix, and the builders install NO mix
    (no corpus team in the team table, no RNG consumed): OFF's draws exactly. Fails on revert of the builder's
    ``mix_prob > 0`` gate (a 0-share mix would still widen the team table)."""
    import random

    from utils.teambuilder import Gen3Teambuilder

    clause = f"teams:{TEST_HUMAN_CORPUS}@0"
    d = HS.decide(_args(human_data=clause))
    assert d.refusal is None and "NULL-DOSE" in d.lines[0] and d.record["corpus"]["n"] == 2
    spec = MatchupSpec.from_args(_args(human_data=clause))
    assert spec.trainee_teams.mix.share == 0.0 and spec.opponent_teams.mix.share == 0.0
    pool = [TEAM_A]
    on, off = spec.opponent_teams.build(pool, []), Gen3Teambuilder(pool)
    assert on.mix_packed_teams == () and "mix_packed_teams" not in vars(on)
    on._rng, off._rng = random.Random(7), random.Random(7)
    assert [on.yield_team() for _ in range(5)] == [off.yield_team() for _ in range(5)]


# ── the refusals ───────────────────────────────────────────────────────────────────────────────────────────────

def test_teams_beside_a_team_mix_is_refused(human):
    """One team lever per run, including a --team-mix a same-run resume restored. Fails on revert of the check."""
    d = HS.decide(_args(human_data=CLAUSE, team_mix="w_forum_v1@ABC:0.5"))
    assert d.record is None and "one team lever per run" in d.refusal


def test_an_unregistered_rowset_or_a_missing_corpus_is_refused(human, monkeypatch, tmp_path):
    assert "not registered" in HS.decide(_args(human_data="teams:hr_never.r9@0.5")).refusal
    monkeypatch.setenv("GEN3AI_ARCHIVE_DIR", str(tmp_path / "nowhere"))
    clear_corpus_cache()
    assert "is missing" in HS.decide(_args(human_data=CLAUSE)).refusal


def test_startup_refusal_is_fatal_config(clone):
    from main.exit_codes import TrainExitCode, exit_code_for

    with pytest.raises(HS.HumanDataRefused) as e:
        MS.build_matchup_and_opponents(_args(human_data="init:hclone_never@1"))
    assert exit_code_for(e.value) == int(TrainExitCode.FATAL_CONFIG)


@pytest.mark.parametrize("cli, lineage", [
    ({"human_data": CLAUSE}, None),
    ({"_human_data": {"canonical": CLAUSE}}, None),
    ({"human_data": "off"}, {"human_data": {"canonical": CLAUSE}}),
    ({}, {"human_data_ancestor": {"run": "rb_parent", "record": {"canonical": CLAUSE}}}),
])
def test_a_resume_or_fork_of_a_clause_run_refuses_unless_typed(tmp_path, cli, lineage):
    """THE RESUME / FORK REFUSAL: a source carrying a clause anywhere (the last process, the typed text, the lineage,
    an ancestor) refuses an untyped --human-data on a same-run resume AND a fork; typing the clause or ``off`` passes.
    Fails on revert of the ``typed is None and src`` refusal (a resume would silently drop the lever)."""
    run, ckpt = _run(tmp_path, "rb_x60t", cli, lineage)
    for run_dir in (str(run), str(tmp_path / "rb_fork")):
        d = HS.decide(_args(model=str(ckpt), run_dir=run_dir))
        assert d.record is None and "must TYPE --human-data" in d.refusal
        ok = HS.decide(_args(model=str(ckpt), run_dir=run_dir, human_data="off"))
        assert ok.refusal is None and not HD.is_on(ok.record) and "the lever changes here" in ok.lines[-1]


def test_a_source_without_a_clause_needs_nothing_typed(tmp_path):
    run, ckpt = _run(tmp_path, "rb_x26_like", {"self_play": True, "human_data": None,
                                               "_human_data": HD.record(HD.parse("off"))})
    d = HS.decide(_args(model=str(ckpt), run_dir=str(run)))
    assert d.refusal is None and not HD.is_on(d.record) and len(d.lines) == 1


def test_lineage_keys_follow_the_lever_down_every_fork(tmp_path):
    rec = HD.record(HD.parse(CLAUSE), {"data_sha256": "d" * 64})
    assert HS.lineage_keys(_args(_human_data=HD.record(HD.parse("off"))), None) == {}
    assert HS.lineage_keys(_args(_human_data=rec), None) == {"human_data": rec}
    run, ckpt = _run(tmp_path, "rb_x60t", {"_human_data": rec})
    off = HD.record(HD.parse("off"))
    child = HS.lineage_keys(_args(_human_data=off), str(ckpt))
    assert child == {"human_data_ancestor": {"run": "rb_x60t", "record": rec}}
    run2, ckpt2 = _run(tmp_path, "rb_child", {"_human_data": off}, lineage=child)
    assert HS.lineage_keys(_args(_human_data=off), str(ckpt2)) == child      # the grandchild still names it
    assert HS.source_clause(str(ckpt2))["canonical"] == CLAUSE


def test_run_lineage_writes_the_keys_into_a_new_block(tmp_path):
    """The seam: ``run_io._run_lineage`` puts the clause into the immutable block of a fresh run."""
    from main.train.run_io import _run_lineage

    rec = HD.record(HD.parse(CLAUSE), {"data_sha256": "d" * 64})
    block = _run_lineage(_args(_human_data=rec, exploiter=None), str(tmp_path / "rb_new"), model_path=None,
                         fork_step=0)
    assert block["role"] == "fresh" and block["human_data"] == rec


# ── the offline surfaces and the launcher's restart ─────────────────────────────────────────────────────────

def test_checkargs_prints_the_decision_and_refuses_an_unregistered_model(capsys, clone):
    from main import checkargs

    rc = checkargs.main(["--argv", "--arch production --self-play --human-data opp:hclone_never@0.2"])
    out = capsys.readouterr().out
    assert "human data" in out and "REFUSED" in out and "not registered" in out and rc != 0
    rc = checkargs.main(["--argv", f"--arch production --self-play --human-data {OPP}"])
    out = capsys.readouterr().out
    assert f"human data                     : {OPP}" in out and "opp: derives --stable-opponents" in out
    checkargs.main(["--argv", "--arch production --self-play"])
    assert "human data                     : off (no human data)" in capsys.readouterr().out


def test_checkargs_refuses_an_untyped_resume_of_a_clause_run(tmp_path):
    run, ckpt = _run(tmp_path, "rb_x60t", {"human_data": CLAUSE})
    lines, refusal = HS.report_lines(_args(model=str(ckpt), run_dir=str(run)))
    assert refusal and "must TYPE --human-data" in lines[0]


def test_the_launchers_restart_argv_re_passes_the_typed_clause():
    """A 6-h restart re-launches the RESUME-role argv: ``--human-data`` is not fresh-only, so the typed clause rides
    every restart and the refusal never trips. Fails if the flag ever joins ``fresh_only_flags``."""
    from main.launcher.checkpoint import resume_child_args
    from main.train.combination_checks import fresh_only_flags

    assert "--human-data" not in fresh_only_flags()
    for argv in (["--arch", "production", "--human-data", CLAUSE, "--steps", "9"],
                 ["--model", "/m/p/checkpoints/a.zip", f"--human-data={CLAUSE}", "--run-name", "rb_x60t_s3501"]):
        out, _ = resume_child_args(argv, "/m/rb_x60t_s3501/checkpoints/c.zip", "/m/rb_x60t_s3501")
        typed = out[out.index("--human-data") + 1] if "--human-data" in out else \
            next(t.split("=", 1)[1] for t in out if t.startswith("--human-data="))
        assert typed == CLAUSE and out[out.index("--model") + 1] == "/m/rb_x60t_s3501/checkpoints/c.zip"


# ── init: a fresh learner starts from the clone ────────────────────────────────────────────────────────────

class _Net:
    """A learner-shaped stand-in: ``.policy`` is a module whose state dict has the testkit's four tensors."""

    def __init__(self, seed=7):
        import torch as th

        th.manual_seed(seed)
        self.policy = th.nn.ModuleDict({"a": th.nn.Linear(4, 3), "b": th.nn.Linear(3, 2)})


def test_init_copies_exactly_the_trained_tensors_into_a_fresh_learner(clone):
    """THE INIT LEVER. The clone's trained tensors land bit for bit; every other parameter keeps the run's own fresh
    init; the record carries what was copied. Fails on revert of the copy in ``apply_init`` (the run would stamp the
    lever and train from scratch) and of the ``trained_keys`` restriction (it would also overwrite ``b``)."""
    import torch as th

    args, lines = _args(human_data=INIT), []
    rec = HS.resolve(args, emit=lines.append)
    assert rec["lever"] == "init" and rec["model"] == TEST_HUMAN_MODEL and rec["rowset"] == TEST_ROWSET
    assert lines[0].startswith(f"[HUMAN-DATA] lever=init rowset={TEST_ROWSET} rows_sha={'r' * 64} ")
    assert "takes 2 trained tensors" in lines[1] and args.stable_opponents is None
    net, src = _Net(), tiny_state()
    before = {k: v.clone() for k, v in net.policy.state_dict().items()}
    out = HS.apply_init(net, args, emit=lines.append)
    after = net.policy.state_dict()
    assert out == {"model": TEST_HUMAN_MODEL, "tensors": 2, "parameters": 15, "of_tensors": 4}
    assert args._human_data["init_applied"] == out and "copied 2 of 4 tensors" in lines[-1]
    for k in ("a.weight", "a.bias"):
        assert th.equal(after[k], src[k]) and not th.equal(after[k], before[k])
    for k in ("b.weight", "b.bias"):
        assert th.equal(after[k], before[k]) and not th.equal(after[k], src[k])


def test_the_init_null_dose_and_every_other_clause_copy_nothing(clone, human):
    """``init:<model>@0`` verifies and stamps the model and leaves the learner's bytes alone; so do ``off``, a
    ``teams`` clause and an ``opp`` clause. Fails on revert of ``apply_init``'s lever / null-dose gate."""
    import torch as th

    for typed in (f"init:{TEST_HUMAN_MODEL}@0", None, CLAUSE, OPP):
        args, lines = _args(human_data=typed, self_play=True), []
        HS.resolve(args, emit=lines.append)
        net = _Net()
        before = {k: v.clone() for k, v in net.policy.state_dict().items()}
        assert HS.apply_init(net, args) is None
        assert all(th.equal(v, before[k]) for k, v in net.policy.state_dict().items())
        assert ("NULL-DOSE" in lines[0]) == (typed is not None and typed.endswith("@0"))
    assert args._human_data.get("init_applied") is None


def test_init_refuses_a_network_the_clone_does_not_fit(clone):
    """Another width is another network: refused, never a partial copy. Fails on revert of the shape check."""
    import torch as th

    args = _args(human_data=INIT)
    HS.resolve(args, emit=lambda _l: None)
    net = _Net()
    net.policy["a"] = th.nn.Linear(4, 5)
    before = {k: v.clone() for k, v in net.policy.state_dict().items()}
    with pytest.raises(HS.HumanDataRefused, match="absent from this run's network or have another shape"):
        HS.apply_init(net, args)
    assert all(th.equal(v, before[k]) for k, v in net.policy.state_dict().items())


def test_init_refuses_a_checkpoint_whose_trained_tensors_are_not_the_manifests(monkeypatch, tmp_path):
    """Fails on revert of the digest check before the copy (the manifest names what the training moved)."""
    install_human_model(monkeypatch, tmp_path, trained_sha="0" * 64)
    args = _args(human_data=INIT)
    HS.resolve(args, emit=lambda _l: None)
    with pytest.raises(HS.HumanDataRefused, match="are not the ones its manifest recorded"):
        HS.apply_init(_Net(), args)


def test_init_is_a_fresh_runs_lever(clone, tmp_path):
    """On ``--model`` the weights come from the checkpoint: ``init`` typed on a source that does not carry it is
    refused; re-typed to keep the source's own clause (a restart, a resume) it passes and copies nothing. Fails on
    revert of either half (a fork would stamp an init it never applied; a restart would re-apply the clone over the
    trained weights)."""
    import torch as th

    plain, ckpt = _run(tmp_path, "rb_plain", {"self_play": True})
    d = HS.decide(_args(model=str(ckpt), run_dir=str(plain), human_data=INIT))
    assert d.record is None and "init is a FRESH run's lever" in d.refusal
    run, ckpt2 = _run(tmp_path, "rb_r9a", {"human_data": INIT})
    args, lines = _args(model=str(ckpt2), run_dir=str(run), human_data=INIT), []
    HS.resolve(args, emit=lines.append)
    assert "nothing is copied" in lines[1]
    net = _Net()
    before = {k: v.clone() for k, v in net.policy.state_dict().items()}
    assert HS.apply_init(net, args) is None
    assert all(th.equal(v, before[k]) for k, v in net.policy.state_dict().items())


# ── opp: the frozen clone joins the self-play opponents ────────────────────────────────────────────────────

def test_opp_derives_the_stable_opponent_flags(clone):
    """THE OPP LEVER wraps the one cross-run opponent path. Fails on revert of the derivation in ``resolve`` (the
    clause would stamp and install no opponent)."""
    args, lines = _args(human_data=OPP, self_play=True), []
    rec = HS.resolve(args, emit=lines.append)
    zp = str(clone / f"{TEST_HUMAN_MODEL}.zip")
    assert args.stable_opponents == f"{zp}:{TEST_HUMAN_MODEL}" and args.stable_opponent_selfplay_share == 0.2
    assert rec["lever"] == "opp" and rec["human_model"]["zip_path"] == zp
    assert f"derives --stable-opponents {zp}:{TEST_HUMAN_MODEL} --stable-opponent-selfplay-share 0.2" in lines[1]
    assert HS.allowed_stable_zip(args) == zp
    other = _args(human_data=f"opp:{TEST_HUMAN_MODEL}@0.35", self_play=True)
    HS.resolve(other, emit=lambda _l: None)
    assert other.stable_opponent_selfplay_share == 0.35


def test_the_opp_null_dose_installs_no_opponent(clone):
    """Fails on revert of the null-dose branch (a 0-share clause would still add an eval opponent)."""
    args, lines = _args(human_data=f"opp:{TEST_HUMAN_MODEL}@0", self_play=True), []
    HS.resolve(args, emit=lines.append)
    assert args.stable_opponents is None and args.stable_opponent_selfplay_share == 0.2
    assert "NULL-DOSE" in lines[0] and len(lines) == 1 and HS.allowed_stable_zip(args) is None


def test_opp_owns_the_stable_opponent_flags(clone):
    """One stable-opponent lever per run; the flags a same-run restart restores (equal to the derived ones) pass.
    Fails on revert of ``_stable_of``'s refusals (a second opponent would dilute the clause's share unrecorded) or
    of its equality pass (every 6-h restart of an ``opp`` run would die FATAL_CONFIG)."""
    from main.train.recipe_surface import TYPED_ATTR

    zp = str(clone / f"{TEST_HUMAN_MODEL}.zip")
    d = HS.decide(_args(human_data=OPP, self_play=True, stable_opponents="models/rb_x26_s3501"))
    assert d.record is None and "one stable-opponent lever per run" in d.refusal
    d = HS.decide(_args(human_data=f"opp:{TEST_HUMAN_MODEL}@0", self_play=True, stable_opponents="models/rb_x26"))
    assert d.record is None and "also at the null dose" in d.refusal
    typed = {TYPED_ATTR: frozenset({"stable_opponent_selfplay_share"})}
    d = HS.decide(_args(human_data=OPP, self_play=True, stable_opponent_selfplay_share=0.5, **typed))
    assert d.record is None and "the clause's share IS that flag" in d.refusal
    restart = _args(human_data=OPP, self_play=True, stable_opponents=f"{zp}:{TEST_HUMAN_MODEL}",
                    stable_opponent_selfplay_share=0.2, **{TYPED_ATTR: frozenset({"stable_opponents",
                                                                                  "stable_opponent_selfplay_share"})})
    d = HS.decide(restart)
    assert d.refusal is None and d.stable == (f"{zp}:{TEST_HUMAN_MODEL}", 0.2)


# ── one entry point per human-trained model ────────────────────────────────────────────────────────────────

def test_an_unregistered_or_smoke_model_is_refused(monkeypatch, tmp_path):
    install_human_model(monkeypatch, tmp_path, smoke=True)
    assert "not registered" in HS.decide(_args(human_data="init:hclone_never@1")).refusal
    for clause in (INIT, OPP):
        assert "SMOKE model" in HS.decide(_args(human_data=clause, self_play=True)).refusal
        assert HS.decide(_args(human_data=clause, self_play=True, debug=True)).refusal is None


def test_a_human_trained_model_is_never_resumed_forked_or_handed_to_another_flag(clone):
    """THE THROWING GUARD at its three callers. Fails on revert of any one: ``--model`` on the clone's zip, a
    ``--stable-opponents`` / ``--exploiter`` naming it, or the allowance leaking to a zip the clause did not derive."""
    from agents.model.model_version import ModelVersion
    from agents.training.fixed_opponent_pool import resolve_stable_opponents
    from utils.human_models import HumanModelError

    zp = str(clone / f"{TEST_HUMAN_MODEL}.zip")
    d = HS.decide(_args(model=zp))
    assert d.record is None and "is never resumed or forked" in d.refusal
    assert "is never resumed or forked" in HS.decide(_args(model=zp, human_data="off")).refusal
    for allow in (None, str(clone / "another.zip")):
        with pytest.raises(HumanModelError, match="a stable-opponent / exploiter target"):
            resolve_stable_opponents(f"{zp}:x", None, allow_human_model=allow)

    class _Past(Exception):
        pass

    def _past(_path):
        raise _Past()

    with mock.patch.object(ModelVersion, "from_json_file", side_effect=_past):
        with pytest.raises(_Past):                       # the clause's own zip passes the guard
            resolve_stable_opponents(f"{zp}:x", None, allow_human_model=zp)


def test_the_trainer_wires_both_model_levers_at_their_one_site():
    """Static, on the two call sites (the levers' functions are tested above; this is the WIRING). ``init``: the fresh
    build's path calls ``apply_init`` once, after the learner is constructed and before the env core copies its
    weights (``_start_rust_env``); the resume path never does. ``opp``: the stable-opponent resolve passes the
    clause's zip as the allowance, and ``--exploiter``'s resolve of the same function passes none."""
    from utils.paths import src_path

    build = src_path("main", "train", "model_build.py").read_text()
    call = "_apply_human_init(model, args, version, emit=emit)"
    assert build.count(call) == 1
    fresh = build.index("model = construct_fresh_learner(args, env, policy_kwargs)")
    assert fresh < build.index(call) < build.index("_start_rust_env(env, model)", fresh)

    matchup = src_path("main", "train", "matchup_setup.py").read_text()
    assert matchup.count("allow_human_model=") == 1
    stable = matchup.index("resolve_stable_opponents(\n                args.stable_opponents")
    exploiter = matchup.index("resolve_stable_opponents(args.exploiter")
    assert stable < matchup.index("allow_human_model=allowed_stable_zip(args)") < exploiter
