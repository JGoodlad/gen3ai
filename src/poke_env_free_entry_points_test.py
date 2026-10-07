"""The trainer, ``main.h2h``, ``main.plateau`` and the offline meters run with ``poke_env`` IMPOSSIBLE (P1 of the
poke-env retirement, ``T27``).

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

A new import of ``poke_env`` on any of these paths fails here with the importing file and line. Fix the import (the
data facade, ``utils.showdown_id``, ``utils.team_packing``, ``agents.enums`` are poke-env-free); never allowlist it.
"""
from __future__ import annotations

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
    "agents.battle.battle_event",
    "utils.teambuilder",
    "utils.team_sources",
    "agents.training.team_archetypes",
    "agents.training.species_priors",
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
    from agents.training.rust_eval import parity as PAR
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
