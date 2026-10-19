"""Offline SEARCH, ``reconstruction``, ``better_line`` / ``lookahead`` and the prober's search-shaped commands RUN
with Node IMPOSSIBLE (owner 2026-10-10: "delete node").

**Why this exists.** Until 2026-10-10 every one of these paths ran Node BY DEFAULT: ``SearchSession`` /
``replay_battle`` / ``reroll_turn`` / ``reroll_many`` spawned ``search_driver.js`` / ``replay_driver.js`` unless a
caller passed ``impl="rust"``, the prober's ``ProbeSession`` and ``query --impl`` defaulted to ``node``, and the
packed-team decode behind ``team_details`` dumped Showdown's alias table with a ``node -e`` child (falling back
SILENTLY to an empty table when node was absent). The Node drivers and the ``impl`` selector are deleted; the
decode reads ``data/pokemon/gen3_move_aliases.json``. A grep cannot prove a RUN-TIME path no longer reaches Node
(a default argument, a fallback or a helper three calls down can), so this test RUNS the paths in a fresh
interpreter with :mod:`utils.node_blocker` installed — every ``node`` spawn (any path) and every ``open`` under
``deps/pokemon-showdown`` raises AND is recorded, and a ``node`` TRIPWIRE is first on the child's ``PATH`` so a
grandchild that runs it fails too. Revert any one of the deletions and the attempt is recorded here.

Out of scope, on purpose (owner, same day): TEAM VALIDATION (``utils.teambuilder`` -> ``validate_team.js``) stays a
Node subprocess, so the fixture's battles are built OUTSIDE the blocker; the dev / test oracles (the differential
gate ladder, the A/B fuzzers, the websocket front end's byte-identity test) keep Node by design.

1. ``test_the_node_blocker_has_teeth`` — the blocker refuses a ``node`` spawn and a submodule read, records a
   SWALLOWED attempt, and the PATH tripwire fails a child's ``node`` and logs it (a blocker that always passed would
   read as green).
2. ``test_a_typed_impl_flag_is_refused_with_its_reason`` — ``query`` / ``web`` / ``harvest`` ``--impl`` and the
   websocket front end's ``--impl node`` are refused WITH the reason, never ``unrecognized arguments``.
3. ``test_search_replay_and_the_prober_run_with_node_impossible`` (``sim``) — real Rust-core traces, then under the
   blocker: the three replay primitives, a warm ``SearchSession``, ``better_line_decision`` (depth 2) and
   ``lookahead_decision`` on a deterministic fake critic, ``team_details`` (the alias decode), and the JSON CLI's
   model-free search commands (``falsify``, ``falsify-scan``, ``calibration``).

The model-LOADING commands (``lookahead`` / ``better-line`` / ``replay-counterfactual`` through the CLI and the
web app) run under the same blocker in ``poke_env_free_entry_points_test.py``'s prober test (``slow``; it needs a
current-architecture checkpoint).
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


def _env(tmp_path, extra=None) -> dict:
    from utils import node_blocker as NB

    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_root())
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.update(extra or {})
    return NB.shim_env(tmp_path / "shim", tmp_path / "node_attempts.log", env)


def _run(code: str, tmp_path, *args, timeout: float = 120, extra=None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code), *map(str, args)], cwd=str(repo_root()),
                          env=_env(tmp_path, extra), capture_output=True, text=True, timeout=scale_timeout(timeout))


def _log(tmp_path) -> str:
    p = tmp_path / "node_attempts.log"
    return p.read_text() if p.exists() else ""


@pytest.mark.integration
def test_the_node_blocker_has_teeth(tmp_path):
    code = f"""
        import subprocess, sys
        from utils import node_blocker as NB
        NB.install()
        for argv in (["node", "-e", "1"], ["/usr/bin/node", "--version"], "nodejs -v"):
            try:
                subprocess.run(argv, shell=isinstance(argv, str), capture_output=True)
            except NB.NodeBlocked:
                pass
            else:
                raise SystemExit(f"spawned {{argv!r}} under the blocker")
        try:
            open({str(repo_path("deps", "pokemon-showdown", "package.json"))!r}).read()
        except NB.NodeBlocked:
            pass
        else:
            raise SystemExit("read the submodule under the blocker")
        # a caller that SWALLOWS the refusal is still recorded
        try:
            subprocess.run(["node", "x.js"])
        except Exception:
            pass
        assert len(NB.ATTEMPTS) == 5, NB.report()
        # a CHILD (which the hook cannot see) hits the PATH tripwire
        r = subprocess.run([sys.executable, "-c", "import subprocess, sys; "
                            "sys.exit(subprocess.run(['node', '-e', '1']).returncode)"], capture_output=True)
        assert r.returncode == NB.SHIM_EXIT, r
        print("TEETH-OK")
    """
    r = _run(code, tmp_path)
    assert r.returncode == 0 and "TEETH-OK" in r.stdout, (r.stdout[-2000:], r.stderr[-3000:])
    log = _log(tmp_path)
    assert log.count("\n") == 6 and "PATH tripwire" in log, log


@pytest.mark.integration
@pytest.mark.parametrize("argv", [
    ["-m", "main.prober.query", "--impl", "rust", "summary", "x"],
    ["-m", "main.prober.query", "--impl=node", "summary", "x"],
    ["-m", "main.prober.web", "--impl", "rust", "x"],
    ["-m", "main.harvest", "--impl", "rust"],
    ["-m", "utils.bridge.ws_frontend", "--port", "9611", "--impl", "node"],
])
def test_a_typed_impl_flag_is_refused_with_its_reason(tmp_path, argv):
    r = subprocess.run([sys.executable, *argv], cwd=str(repo_root()), env=_env(tmp_path), capture_output=True,
                       text=True, timeout=scale_timeout(120))
    assert r.returncode != 0, (r.stdout[-1500:], r.stderr[-1500:])
    assert "2026-10-10" in r.stderr and "delete node" in r.stderr, r.stderr[-1500:]
    assert "unrecognized arguments" not in r.stderr
    assert not _log(tmp_path), _log(tmp_path)


_CHILD = r"""
import json, sys, traceback
import numpy as np
from utils import node_blocker as NB
NB.install()

run_dir, battle, decide, outcome = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
from main.prober.core_trace import load_summary
from main.prober.better_line import better_line_decision
from main.prober.lookahead import lookahead_decision
from main.prober import query
from utils.bridge.reconstruction import ReconstructionRecord, replay_battle, reroll_many, reroll_turn
from utils.bridge.search_session import SearchSession


class SumModel:
    # V(s) = obs.sum() — deterministic, model-free (better_line_integration_test's fake critic)
    def value(self, obs, mask):
        return float(np.asarray(obs, dtype=np.float64).sum())

    def values_batch(self, obs, masks):
        return np.asarray(obs, dtype=np.float64).reshape(len(obs), -1).sum(1)

    def _probs(self, obs, mask):
        x = np.asarray(obs, dtype=np.float64)
        m = np.asarray(mask, dtype=float)
        sc = np.where(m > 0, np.array([x[(i * 97) % len(x)] for i in range(len(m))]), -1e9)
        p = np.exp(sc - sc.max())
        return p / p.sum()

    def action_dist(self, obs, mask):
        p = self._probs(obs, mask)
        return p, np.log(p + 1e-12)

    def action_probs_batch(self, obs, masks):
        return np.stack([self._probs(o, m) for o, m in zip(np.asarray(obs), np.asarray(masks))])

    def win_prob_at(self, obs, mask):
        return None


rec = ReconstructionRecord.load(battle.replace("_summary.json", "_reconstruction.json"))
summary = load_summary(battle)
with np.load(battle.replace("_summary.json", "_states.npz")) as z:
    npz = {k: z[k] for k in z.files}
turn = int(summary["invocations"][decide]["turn"])
out = {}


def step(name, fn):
    try:
        res = fn()
        out[name] = {"ok": True, "error": res.get("error") if isinstance(res, dict) else None}
    except BaseException as e:
        out[name] = {"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}


step("replay_battle", lambda: replay_battle(rec) and None)
step("reroll_turn", lambda: reroll_turn(rec, turn, seeds=["original"]) and None)
step("reroll_many", lambda: reroll_many(rec, turn, [{"p1_action": "recorded", "p2_action": "recorded",
                                                     "seed": "original", "label": 0}]) and None)


def _warm_session():
    with SearchSession() as ss:
        ss.open_root(turn, record=rec)


step("SearchSession", _warm_session)
step("team_details", lambda: [rec.team_details(s) for s in ("p1", "p2")] and None)
step("better_line_decision", lambda: better_line_decision(SumModel(), rec, summary, npz, decide, depth=2, beam=2,
                                                          top_k=2, opp_model=SumModel()))
step("lookahead_decision", lambda: lookahead_decision(SumModel(), rec, summary, npz, decide))
P = query._build_parser()
rr = ["--seeds", "2", "--alts", "1", "--worst", "1"]
for argv in (["falsify", battle, "--inv", str(decide), "--seeds", "2", "--alts", "1"],
             ["falsify-scan", run_dir, "--outcome", outcome, "--limit", "1", *rr, "--concurrency", "1"],
             ["calibration", run_dir, "--outcome", outcome, "--limit", "1", *rr, "--concurrency", "1"]):
    step("query " + argv[0], lambda argv=argv: query._run(P.parse_args(argv)))
print("NODE-FREE-RESULT " + json.dumps({"steps": out, "attempts": [w for w, _ in NB.ATTEMPTS],
                                        "report": NB.report()}))
"""


@pytest.fixture(scope="module")
def core_run(tmp_path_factory):
    """Real Rust-core games written as core traces (built OUTSIDE the blocker: the pool's teams go through the
    Node team validator, which stays — see the module docstring), and the search driver resolved up front so the
    blocked child never builds it."""
    from main.prober.core_trace import load_summary
    from main.prober.core_trace_integration_test import _play_core_games, build_core_lib
    from main.prober.engine import _has_state
    from utils.bridge.sim_bridge_bin import resolve_search_driver_bin

    lib = build_core_lib()
    run_dir = tmp_path_factory.mktemp("node_free_run")
    paths = _play_core_games(lib, str(run_dir), 2, turn_limit=250, seed=31, tag="nf")
    driver = resolve_search_driver_bin()
    for sp in paths:
        invs = load_summary(sp)["invocations"]
        with __import__("numpy").load(sp.replace("_summary.json", "_states.npz")) as z:
            npz = {k: z[k] for k in z.files}
        pick = [i for i, inv in enumerate(invs) if inv["phase"] == "move_selection" and inv["turn"] >= 2
                and i + 1 < len(invs) and _has_state(npz, i) and _has_state(npz, i + 1)
                and sum(a["valid"] for a in inv["actions"].values()) >= 2]
        if pick:
            # the battle's outcome bucket, so falsify-scan / calibration select THIS battle (limit 1)
            from agents.training.trace_result import OUTCOMES

            outcome = os.path.basename(sp).split("_")[0]
            assert outcome in OUTCOMES, (outcome, sp)
            return str(run_dir), sp, pick[len(pick) // 2], outcome, driver
    pytest.fail("no core trace with a usable move_selection decision")


@pytest.mark.slow  # ship gate (2026-10-19): 10 s beside a live run
@pytest.mark.sim
@pytest.mark.integration
def test_search_replay_and_the_prober_run_with_node_impossible(core_run, tmp_path):
    run_dir, battle, decide, outcome, driver = core_run
    r = _run(_CHILD, tmp_path, run_dir, battle, decide, outcome, timeout=900,
             extra={"POKESIM_SEARCH_DRIVER_BIN": driver})
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("NODE-FREE-RESULT ")), None)
    assert r.returncode == 0 and line, (r.stdout[-3000:], r.stderr[-4000:])
    res = json.loads(line[len("NODE-FREE-RESULT "):])
    assert not res["attempts"], res["report"]
    assert not _log(tmp_path), _log(tmp_path)
    failed = {k: v for k, v in res["steps"].items() if not v["ok"] or v["error"]}
    assert not failed, failed
    assert set(res["steps"]) == {"replay_battle", "reroll_turn", "reroll_many", "SearchSession", "team_details",
                                 "better_line_decision", "lookahead_decision", "query falsify",
                                 "query falsify-scan", "query calibration"}, res["steps"]
