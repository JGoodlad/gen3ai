"""The trainer, ``main.h2h``, ``main.plateau``, ``main.anchors`` (P3) and the offline meters run with ``poke_env``
IMPOSSIBLE (P1 of the poke-env retirement, ``T27``).

**Why this exists.** The import gate (``poke_env_import_gate_test.py``) counts files that NAME ``poke_env``; it cannot
say whether an entry point's import CLOSURE — or its RUN-TIME path — reaches one. Before P1 the trainer loaded 36
poke-env modules at import (``agents.gen3_data`` -> ``agents.enums`` re-exported the fork's classes; the eval facade
imported the player classes; the training roster was a list of poke-env bot classes), and ``main.h2h`` / ``main.plateau``
loaded 0 at import but 36 at run time (the survey's A-F7). A ``sys.modules`` look after an import therefore proves only
half; these tests install ``utils.poke_env_blocker`` FIRST (every ``import poke_env[.x]`` raises AND is recorded, so a
``try: … except ImportError`` cannot hide it) and then import or RUN the entry point:

1. ``test_every_entry_point_imports_without_poke_env`` — a fresh interpreter imports every entry point of the list
   below (+ builds the production argv through the trainer's own parser) with the blocker installed; no module, no
   attempt.
2. ``test_the_blocker_has_teeth`` — the same blocker, asked for ``poke_env``, raises and records, and a swallowed
   ``ImportError`` still lands in the record (without this a blocker that always passed would read as green).
3. ``test_a_head_to_head_edge_plays_with_poke_env_blocked`` (``sim``) — a real ``main.h2h`` edge (two perturbed-fresh
   production-architecture checkpoints built through the trainer's parser, the Rust eval engine, a ledger row) with
   the blocker installed: the whole RUN-TIME path, the half an import look cannot see.
4. ``test_a_debug_smoke_with_eval_runs_with_poke_env_blocked`` (``slow``) — the root CLAUDE.md ``--debug`` smoke WITH
   two in-process eval cycles, entered through ``poke_env_blocker.main`` (the trainer's own module name split so no
   argv carries it).
5. ``main.anchors`` (P3 + P6, 2026-10-07) — its import closure is in the list below (the whole package: the legacy
   poke-env client was deleted in P6); ``test_a_rust_bot_our_side_plays_a_battle_with_poke_env_blocked`` (``sim``)
   plays a real battle with a ``bot:`` our-side (the Rust bot on the live client) under the blocker; the full
   RUN-TIME read against Metamon (the core slot, the live client on the front end and on Node, a Rust-bot our-side)
   runs under the same blocker in ``src/main/anchors/anchors_integration_test.py`` (``slow``, its verdicts banked in
   ``designs/ops/slow_tier_status.json``), which needs the Metamon checkout and so lives beside the tool.

6. ``test_every_prober_command_and_the_web_app_run_with_poke_env_blocked`` (``sim``, P5) — the PROBER on real Rust-eval
   core traces and a current-architecture checkpoint: every JSON-CLI command (model-free and model-loading) and the
   web app's views RUN with the blocker installed — ``replay-counterfactual`` too since P6 (the Rust play-out,
   a model and a self-model opponent); the commands that still need poke-env are the closed list
   :data:`PROBER_POKE_ENV_COMMANDS`, EMPTY since P6, and the test pins that nothing else reaches poke-env.

A new import of ``poke_env`` on any of these paths fails here with the importing file and line. Fix the import (the
data facade, ``utils.showdown_id``, ``utils.team_packing``, ``agents.enums`` are poke-env-free); never allowlist it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from utils.contention import scale_timeout
from utils.paths import repo_path, repo_root, src_root

#: Every entry point whose import closure must stay poke-env-free. The trainer first (its module name is split so
#: this file's own text does not carry it), then the head-to-head and the plateau meter, the launcher and the offline
#: meters, then the shared packages the data / eval / team paths hang off.
ENTRY_POINTS = (
    "main.train_rl" "_agent",
    "main.train.parser",
    "main.h2h",
    "main.h2h.cli",
    "main.plateau",
    "main.anchors",
    "main.anchors.cli",
    "main.anchors.core_side",
    "main.anchors.runner",
    "main.anchors.server",
    # P6: the whole anchors package — our side as a live client, and the Rust bot's reader
    "main.anchors.session",
    "main.anchors.live_side",
    "main.live.bot_reader",
    "utils.rust_env.bot_reader_bin",
    "utils.bridge.ws_frontend",
    # P6 slice 3: live play (the Rust client is the only client), P4's gate (c) harness and its bot peer, and the
    # pre-session drift gate (both reader checks on the Rust reader)
    "main.play",
    "main.live.client",
    "main.live.gate_peer",
    "main.live.master_series",
    "main.live.replay_scan",
    "main.live.effect_scan",
    "main.ladder_drift_scan",
    "main.launcher",
    "main.checkargs",
    "main.elo",
    "main.critic_gate",
    "main.untaught_meter",
    "main.best_response_gap",
    "main.eval_ledger",
    "agents.training.snapshot_ladder",
    "agents.training.eval_callback",
    "agents.gen3_data",
    "agents.model.snapshot",
    "agents.observation.state_encoder",
    "agents.battle.live_view",
    "utils.teambuilder",
    "utils.team_sources",
    "agents.training.team_archetypes",
    "agents.training.species_priors",
    # the PROBER (P5): the JSON CLI, the web front end and the engine behind both
    "main.prober.query",
    "main.prober.web",
    "main.prober.web.app",
    "main.prober.session",
    "main.prober.engine",
    "main.prober.model",
    "main.prober.core_trace",
    "main.prober.core_walk",
    "main.prober.core_recorder",
    "main.prober.falsifier",
    "main.prober.lookahead",
    "main.prober.better_line",
    "main.prober.forensics",
)


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_root())
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _run(code: str, *, cwd, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=str(cwd), env=_env(),
                          capture_output=True, text=True, timeout=scale_timeout(timeout))


@pytest.mark.integration
def test_every_entry_point_imports_without_poke_env():
    code = f"""
        import importlib, sys
        from utils import poke_env_blocker as B
        B.install()
        for name in {list(ENTRY_POINTS)!r}:
            importlib.import_module(name)
        # the trainer's own parser, on the production argv (what a fresh checkpoint build and `checkargs` run)
        from main.train.production_args import production_args
        production_args()
        loaded = sorted(k for k in sys.modules if k == "poke_env" or k.startswith("poke_env."))
        assert not loaded, loaded
        assert not B.ATTEMPTS, B.report()
        print("OK", len(sys.modules))
    """
    r = _run(code, cwd=repo_root(), timeout=240)
    assert r.returncode == 0 and "OK" in r.stdout, (r.stdout[-2000:], r.stderr[-4000:])


@pytest.mark.integration
def test_the_blocker_has_teeth():
    code = """
        import sys
        from utils import poke_env_blocker as B
        B.install()
        try:
            import poke_env.battle.pokemon
        except B.PokeEnvBlocked:
            pass
        else:
            raise SystemExit("poke_env imported under the blocker")
        # a caller that SWALLOWS the ImportError is still recorded
        try:
            from poke_env.player import Player
        except ImportError:
            pass
        assert [n for n, _ in B.ATTEMPTS] == ["poke_env", "poke_env"], B.ATTEMPTS
        assert not [k for k in sys.modules if k.startswith("poke_env")]
        print("OK")
    """
    r = _run(code, cwd=repo_root(), timeout=120)
    assert r.returncode == 0 and "OK" in r.stdout, (r.stdout[-2000:], r.stderr[-4000:])


# ---- the RUN-TIME paths ------------------------------------------------------------------------------------------

_EDGE = """
    import tempfile
    from pathlib import Path
    from utils import poke_env_blocker as B
    B.install()
    from agents.training.rust_rollout.testkit import build_selfcheck
    build_selfcheck()
    from agents.training.rust_eval import offline as PAR
    from main.h2h import play as PL
    tmp = Path(tempfile.mkdtemp(prefix="p1_h2h_"))
    with PAR.declared_torch_state(1):
        trainee, sentinels, _cfg = PAR.build_models(tmp / "run_h2h_test", n_sentinels=1)
    compute = PL.Compute(device="cpu", backend="eager", n_envs=8, threads=2, torch_threads=2, front="ffi",
                         profile="selfcheck")
    stats = PL.play_edge(str(tmp / "ledger"), PL.resolve_player(trainee), PL.resolve_player(sentinels[0]), pairs=2,
                         batch_pairs=2, schedule_seed=3, run_label="p1-blocked", compute=compute, emit=lambda _m: None)
    assert not B.ATTEMPTS, B.report()
    print("EDGE-OK", sorted(stats)[:3])
"""


@pytest.fixture(scope="module")
def h2h_edge_blocked():
    """One tiny real edge, played in a child with poke-env blocked (a MODULE fixture: the tier budget is per call)."""
    return _run(_EDGE, cwd=repo_root(), timeout=900)


@pytest.mark.sim
@pytest.mark.integration
def test_a_head_to_head_edge_plays_with_poke_env_blocked(h2h_edge_blocked):
    r = h2h_edge_blocked
    assert r.returncode == 0 and "EDGE-OK" in r.stdout, (r.stdout[-2000:], r.stderr[-4000:])


_BOT_SLOT = """
    import asyncio, os, tempfile
    from types import SimpleNamespace
    from utils import poke_env_blocker as B
    B.install()
    os.environ["GEN3AI_LIVE_HALT_FILE"] = os.path.join(tempfile.mkdtemp(prefix="p6_halt_"), "halt.json")
    from main.anchors import live_side
    from main.anchors.server import InProcessFrontEnd, pick_port
    from main.anchors.session import OurSideState
    from main.live.client import ClientConfig, LiveClient
    from main.live.policy import RandomPolicy
    from utils.team_sources import packed_teams

    async def series():
        teams = packed_teams("pool")
        front = InProcessFrontEnd(pick_port([9500, 9599]), seed_base=61_009)
        await front.__aenter__()
        state = OurSideState()
        plan = SimpleNamespace(server_uri=front.uri, battle_format="gen3ou", connect_timeout_s=60.0,
                               progress_timeout_s=300.0, our_side="bot:staller_v2", forfeit_turn_limit=250,
                               team_seed=3, bot_seed=17)
        ours = live_side.build_client(plan, "ours_challenge", username="p6ours", team_spec={"kind": "pool"},
                                      state=state)
        opp = LiveClient(ClientConfig(uri=front.uri, username="p6opp", forfeit_turn_limit=250),
                         policy=RandomPolicy(5), team_fn=lambda: teams[9])
        try:
            await opp.connect()
            await ours.connect()
            acc = asyncio.ensure_future(opp.accept("p6ours", 1))
            await asyncio.wait_for(ours.challenge("p6opp", 1), timeout=600)
            await asyncio.wait_for(acc, timeout=120)
        finally:
            await ours.close()
            await opp.close()
            await front.__aexit__(None, None, None)
        return state

    st = asyncio.run(series())
    assert len(st.records) == 1 and st.records[0].n_decisions > 0, st.records
    assert not B.ATTEMPTS, B.report()
    print("BOT-OK", st.records[0].result, st.records[0].turns, st.records[0].n_decisions)
"""


@pytest.mark.sim
@pytest.mark.integration
def test_a_rust_bot_our_side_plays_a_battle_with_poke_env_blocked():
    """P6: a ``bot:`` our-side of ``main.anchors`` — the Rust port of the roster bot on ``main.live``'s client over
    the ``bot_reader`` session — plays a real battle on the in-process front end with the blocker installed (the
    checkpoint's live path shares every module but the policy; the core slot is the slow read's)."""
    r = _run(_BOT_SLOT, cwd=repo_root(), timeout=900)
    assert r.returncode == 0 and "BOT-OK" in r.stdout, (r.stdout[-2000:], r.stderr[-4000:])


_RUNNER = ("import sys; from utils import poke_env_blocker as B; "
           "sys.exit(B.main(['main.train_rl' '_agent'] + sys.argv[1:]))")


@pytest.mark.slow
@pytest.mark.integration
def test_a_debug_smoke_with_eval_runs_with_poke_env_blocked(tmp_path):
    """The root CLAUDE.md smoke with two BLOCKING eval cycles (`--debug-eval --eval-freq 4000`), CPU, in a scratch cwd
    (the trainer reaches `data/` and `deps/` by relative path; its run lands under `$GEN3AI_MODELS_DIR`)."""
    for d in ("data", "deps"):
        link = tmp_path / d
        if not link.exists():
            link.symlink_to(repo_path(d))
    (tmp_path / "models").mkdir(exist_ok=True)
    log = tmp_path / "attempts.log"
    env = _env()
    env["GEN3AI_MODELS_DIR"] = str(tmp_path / "models")
    env["GEN3AI_POKE_ENV_BLOCK_LOG"] = str(log)
    argv = [sys.executable, "-c", _RUNNER, "--debug", "--steps", "10000", "--debug-eval", "--eval-freq", "4000",
            "--run-name", "p1_blocked_smoke"]
    proc = subprocess.run(argv, cwd=str(tmp_path), env=env, capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=scale_timeout(1500))
    assert proc.returncode == 0, f"smoke failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-4000:]}"
    assert "Training complete" in proc.stdout and "[EVAL] step 4,000" in proc.stdout, proc.stdout[-2000:]
    assert not log.exists() or not log.read_text().strip(), log.read_text()


# ---- the PROBER (P5): every JSON-CLI command and the web front end, RUN with poke-env blocked -------------------

#: The prober commands that still need poke-env: NONE since P6 (`replay-counterfactual` plays the rest of a battle
#: on the Rust core, `gen3_cf_core_playout_v1`). Kept as a closed, EMPTY list: a command that comes to need poke-env
#: again must be named here, in review, rather than slip through.
PROBER_POKE_ENV_COMMANDS: tuple = ()

_PROBER = r"""
import json, sys, traceback
from utils import poke_env_blocker as B
B.install()
from main.prober import query

run, battle, decide = sys.argv[1], sys.argv[2], int(sys.argv[3])
from main.prober.session import ProbeSession
short = next(row["short_id"] for row in ProbeSession(run).battles() if row["id"] == battle)
from main.prober.core_trace import load_summary
_acts = list(load_summary(battle)["invocations"][decide]["actions"].values())
sub = str(next(i for i, a in enumerate(_acts) if a["valid"]))
P = query._build_parser()
rr = ["--seeds", "2", "--alts", "1", "--worst", "1"]
cmds = [["summary", run], ["list", run], ["scan", run], ["scan", run, "--metric", "td_residual"],
        ["awareness", run, "--outcome", "all"], ["loops", run], ["triage", run], ["switch-vs-info", run],
        ["decision-table", run], ["turns", battle], ["overview", battle],
        *[["find", battle, c] for c in query._FIND_CRITERIA],
        ["analyze", battle, str(decide)],
        ["--impl", "rust", "falsify", battle, "--inv", str(decide), "--seeds", "2", "--alts", "1"],
        ["--impl", "rust", "lookahead", battle, "--inv", str(decide)],
        ["--impl", "rust", "better-line", battle, str(decide), "--depth", "2", "--beam", "2", "--top-k", "2",
         "--interior-opponent", "self"],
        ["--impl", "rust", "falsify-scan", run, "--outcome", "loss", "--limit", "1", *rr, "--concurrency", "1"],
        ["--impl", "rust", "calibration", run, "--outcome", "loss", "--limit", "1", *rr, "--concurrency", "1"],
        ["probe", run, "is_faster", "--max-decisions", "60"],
        ["history-saliency", run, "--max-decisions", "10"],
        ["--impl", "rust", "replay-counterfactual", battle, str(decide), sub, "--rollouts", "2", "--narrate"],
        ["--impl", "rust", "replay-counterfactual", battle, str(decide), sub, "--opponent-source", "self"]]
out = {}
for argv in cmds:
    key = " ".join(a for a in argv if not a.startswith("/"))
    try:
        res = query._run(P.parse_args(argv))
        out[key] = {"ok": True, "error": res.get("error") if isinstance(res, dict) else None}
    except Exception as e:
        out[key] = {"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}

from fastapi.testclient import TestClient
from main.prober.web.app import create_app
pages = {}
with TestClient(create_app(run, open_access=True, impl="rust")) as c:
    q = f"battle={short}&inv={decide}"
    for path in ("/api/run", "/api/battles", f"/api/battle-turns?battle={short}", f"/api/analyze?{q}",
                 "/api/scan", "/api/triage", "/", "/battles", f"/battle?battle={short}", f"/analyze?{q}",
                 f"/partials/analyze?{q}", "/scan", "/triage"):
        r = c.get(path)
        pages[path] = r.status_code
out_attempts = list(B.ATTEMPTS)

# the declared exceptions (none since P6): each reaches poke-env, and nothing else does
exc = {}
print("PROBER-RESULT " + json.dumps({"commands": out, "pages": pages, "attempts": [n for n, _ in out_attempts],
                                     "report": B.report() if out_attempts else "", "exceptions": exc}))
"""


@pytest.fixture(scope="module")
def prober_run(tmp_path_factory):
    """A real Rust-eval CORE-TRACE run (games played by the Rust env core, written by `write_core_trace` — the
    prober integration test's builder) plus a seeded perturbed fresh production checkpoint as its eval snapshot, so
    the model-loading commands load a model at the CURRENT architecture."""
    import shutil

    from agents.training.rust_eval.offline import build_models, declared_torch_state
    from main.prober.core_trace_integration_test import FEATURES, _cargo, _play_core_games
    from utils.rust_env import ffi

    crate = src_root() / "rust_env"
    benv = dict(os.environ, CARGO_TARGET_DIR=str(crate / "target"))
    r = subprocess.run([_cargo(), "build", "--lib", *FEATURES, "--manifest-path", str(crate / "Cargo.toml")],
                       env=benv, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, f"building the cdylib failed:\n{r.stderr[-4000:]}"
    lib = ffi.load(ffi.default_path("selfcheck"), nan_poison=True)
    run_dir = tmp_path_factory.mktemp("prober_blocked_run")
    paths = _play_core_games(lib, str(run_dir), 3, turn_limit=250, seed=21, tag="s0")
    with declared_torch_state(1):
        trainee, _sent, cfg = build_models(tmp_path_factory.mktemp("prober_model"), n_sentinels=0)
    shutil.copy(cfg, run_dir / "model_config.json")
    # the NEAREST tier: a periodic checkpoint at the traces' step
    os.makedirs(run_dir / "checkpoints")
    shutil.copy(trainee, run_dir / "checkpoints" / "checkpoint_1000_steps.zip")
    from main.prober.core_trace import load_summary

    for sp in paths:
        invs = load_summary(sp)["invocations"]
        pick = [i for i, inv in enumerate(invs) if inv["phase"] == "move_selection" and inv["turn"] >= 2
                and sum(a["valid"] for a in inv["actions"].values()) >= 2]
        if pick:
            return str(run_dir), sp, pick[0]
    pytest.fail("no core trace with a usable move_selection decision")


@pytest.mark.slow  # 127 s on a quiet box (2026-10-08, P6 slice 1 ship): run per P6 slice explicitly
@pytest.mark.sim
@pytest.mark.integration
def test_every_prober_command_and_the_web_app_run_with_poke_env_blocked(prober_run):
    """P5: the JSON CLI's every command bar the declared exception, and the web front end's views, RUN — not just
    import — with poke-env blocked, on real core traces and a current-architecture checkpoint."""
    run_dir, battle, decide = prober_run
    r = subprocess.run([sys.executable, "-c", _PROBER, run_dir, battle, str(decide)], cwd=str(repo_root()),
                       env=_env(), capture_output=True, text=True, timeout=scale_timeout(1500))
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("PROBER-RESULT ")), None)
    assert r.returncode == 0 and line, (r.stdout[-3000:], r.stderr[-4000:])
    res = json.loads(line[len("PROBER-RESULT "):])
    assert not res["attempts"], res["report"]
    failed = {k: v for k, v in res["commands"].items() if not v["ok"]}
    # `awareness` reads the deleted distributional value head; on a run that never had one it raises by design
    assert set(failed) <= {"awareness --outcome all"}, failed
    for must in ("turns", "overview", "analyze", "--impl rust falsify", "--impl rust lookahead",
                 "--impl rust better-line", "--impl rust replay-counterfactual"):
        hits = [k for k in res["commands"] if k.startswith(must)]
        assert hits and all(res["commands"][k]["error"] is None for k in hits), (must, res["commands"])
    assert all(code == 200 for code in res["pages"].values()), res["pages"]
    assert res["exceptions"] == {name: "blocked" for name in PROBER_POKE_ENV_COMMANDS}, res["exceptions"]
