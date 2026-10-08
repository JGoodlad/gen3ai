"""A Rust-eval CORE TRACE through the prober: expansion, refusal of a tampered record, the core's own
stream reading and the JSON CLI on a core-trace run (`main.prober.core_trace`).

The traces are REAL: games played by the Rust env core behind the FFI (p1 a seeded random-legal
scripted policy, p2 the in-core `heuristic` bot) and written by `write_core_trace` — the same
writer the Rust eval executor calls. One game runs the core at a short `turn_limit`, so the
trainee's STALL FORFEIT (`forcelose p1`) is exercised end to end.

CORE-ONLY since P6 slice 6d-1 (2026-10-08): this file used to hold the poke-env ORACLE the core's walk
was held to (P5) — the same game played LIVE on the rust bridge by a scripted `Gen3Player` recording with
a real `BattleRecorder` (the expanded summary equal field for field, the live encoder's obs row for row,
the stand-in protocol log equal to the live `_replay.html`), the poke-env materializer's rows and choice
maps, and the fork's `Player.MESSAGES_TO_IGNORE`. Those three tests went with the road they compared
(`RLPlayer`, `run_local_battles`, `obs_materializer`); what the oracle established is now held by the
banked evidence (`designs/rust_sim/`, the P6 6b engine-truth record) and by the checks below, each of
which reads the core and the STORED rows only: the expansion lines up with the stored rows, the stall
forfeit, the protocol-log stand-in, the walk's stream rows equal the STORED rows byte for byte, the
walk's choice maps equal the stream reading's and name exactly the stored row's legal actions, and every
tampered record is refused.

Needs the rust env cdylib (built here, selfcheck profile) and the rust `core_events` (resolved by
`main.prober.core_walk`) — `sim`. Seconds, not minutes.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from utils.paths import src_path

pytestmark = [pytest.mark.sim, pytest.mark.integration]

FEATURES = ("--profile", "selfcheck", "--features", "emission-selfcheck")
NAMES = ("coreone", "coretwo")
STEP = 1000
STALL_LIMIT = 6


def _cargo() -> str:
    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo):
        pytest.fail("cargo is not installed — the core-trace tests cannot build the env core")
    return cargo


@pytest.fixture(scope="module")
def lib():
    from utils.rust_env import ffi

    crate = src_path("rust_env")
    env = dict(os.environ, CARGO_TARGET_DIR=str(src_path("rust_env", "target")))
    r = subprocess.run([_cargo(), "build", "--lib", *FEATURES, "--manifest-path", str(crate / "Cargo.toml")],
                       env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, f"building the cdylib failed:\n{r.stderr[-4000:]}"
    return ffi.load(ffi.default_path("selfcheck"), nan_poison=True)


def _play_core_games(lib, run_dir: str, n: int, *, turn_limit: int, seed: int, tag: str) -> list:
    """Play ``n`` games on the core and write each as a core trace; returns the summary paths."""
    from agents.training.rust_eval.traces import write_core_trace
    from agents.training.trace_result import classify_result, outcome_prefix
    from utils.rust_env import ffi
    from utils.rust_env import protocol as P
    from utils.team_loader import TeamLoader
    from utils.teambuilder import Gen3Teambuilder

    teams = list(Gen3Teambuilder(TeamLoader().get_all_teams()).packed_teams)[:12]
    spec = P.spec_json(n=2, threads=1, teams=teams, names=NAMES, turn_limit=turn_limit, refusal_budget=4, bank_dir=None,
                       opponents=[{"kind": "bot", "bot": "heuristic", "seed": 3, "streams": "episode"}])
    core = ffi.FfiCore(spec, lib=lib)
    c = core.cols
    rng = np.random.default_rng(seed)

    def stage(envs):
        for e in envs:
            c["ep_team"][e] = rng.choice(len(teams), 2, replace=False)
            c["ep_seed"][e] = rng.integers(0, 65536, 4)
            c["ep_opp"][e] = 0

    stage(range(2))
    core.reset()
    stage(range(2))
    buf: dict = {e: [] for e in range(2)}
    out: list = []
    while len(out) < n:
        for e in range(2):
            # The executor's rule: a p1 decision at turn >= turn_limit is the stall forfeit — no
            # forward, no row (`rust_eval.executor`, `rust_env::episode::stall_forfeit_due`).
            if c["need"][e, 0] and int(c["turn"][e]) < turn_limit:
                m = c["mask"][e, 0].astype(bool)
                a = int(rng.choice(np.flatnonzero(m)))
                z = np.where(m, rng.normal(size=m.shape), -np.inf)
                lp = (z - np.log(np.exp(z[m] - z[m].max()).sum()) - z[m].max()).astype(np.float32)
                buf[e].append((c["obs"][e, 0].copy(), m.copy(), lp, float(rng.normal()), a))
                c["action"][e, 0] = a
        core.step()
        fin = core.finished()
        for f in fin:
            e = f["env"]
            rows, buf[e] = buf[e], []
            if len(out) >= n:
                continue
            k = len(out)
            t = ffi.trace(lib, f["script"], commit="test", label=f"{tag}{k}")
            res, kind = classify_result(won=f["winner"] == 1, lost=f["winner"] == 2, finished=True,
                                        turn=f["end_turn"], turn_cap=250)
            pre = os.path.join(run_dir, "eval_traces", f"step_{STEP}", "heuristic",
                               f"{outcome_prefix(res)}_{tag}_{k + 1:03d}")
            write_core_trace(pre, trace=t, step=STEP, battle_id=f"core-{STEP}-heuristic-{tag}{k}",
                             result=res, draw_kind=kind, turns=f["end_turn"],
                             trainee_username=NAMES[0], obs=np.stack([r[0] for r in rows]),
                             logp=np.stack([r[2] for r in rows]),
                             values=np.array([r[3] for r in rows], dtype=np.float32),
                             actions=np.array([r[4] for r in rows]), masks=np.stack([r[1] for r in rows]))
            if turn_limit != 250:
                # The writer does not record the core's `turn_limit`; the expansion reads it from
                # `trace_source.turn_limit` when present (else the production threshold).
                sp = f"{pre}_summary.json"
                with open(sp) as fh:
                    s = json.load(fh)
                s["meta"]["trace_source"]["turn_limit"] = turn_limit
                with open(sp, "w") as fh:
                    json.dump(s, fh, indent=2)
            out.append(f"{pre}_summary.json")
        stage([f["env"] for f in fin])
    return out


@pytest.fixture(scope="module")
def run(lib, tmp_path_factory):
    run_dir = str(tmp_path_factory.mktemp("core_run"))
    normal = _play_core_games(lib, run_dir, 2, turn_limit=250, seed=7, tag="s0")
    stall = _play_core_games(lib, run_dir, 1, turn_limit=STALL_LIMIT, seed=11, tag="s1")
    return run_dir, normal, stall


def _npz(summary_path: str) -> dict:
    with np.load(summary_path.replace("_summary.json", "_states.npz")) as z:
        return {k: z[k] for k in z.files}


def _stored(summary_path: str) -> dict:
    with open(summary_path) as f:
        return json.load(f)


def _recon(summary_path: str) -> dict:
    with open(summary_path.replace("_summary.json", "_reconstruction.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# (a) the expansion lines up with the stored rows
# ---------------------------------------------------------------------------

def test_the_expansion_matches_the_states_rows_and_keeps_the_stored_meta(run):
    from main.prober.core_trace import load_summary

    _run_dir, normal, stall = run
    for sp in normal + stall:
        s, npz, stored = load_summary(sp), _npz(sp), _stored(sp)
        assert s["meta"] == stored["meta"], sp
        invs = s["invocations"]
        assert len(invs) == stored["meta"]["invocations"] == len(npz["actions"]) > 0, sp
        assert {"ours", "opponent"} <= set(s["teams"]) and len(s["teams"]["ours"]) == 6
        turns = [inv["turn"] for inv in invs]
        assert turns == sorted(turns) and turns[-1] <= stored["meta"]["turns"]
        for i, inv in enumerate(invs):
            labels = list(inv["actions"])
            assert len(labels) == 11, (sp, i, labels)      # no two labels collide on these teams
            assert inv["chosen"] == labels[int(npz["actions"][i])], (sp, i)
            assert [v["valid"] for v in inv["actions"].values()] == npz["action_mask"][i].tolist()
        assert invs[-1]["outcome"]["events"][-1] == f"result:{stored['meta']['result'].lower()}"


def test_the_stall_game_ends_on_the_trainees_forfeit_with_no_row_for_it(run):
    from main.prober.core_trace import load_summary

    _run_dir, _normal, stall = run
    (sp,) = stall
    cmds = _recon(sp)["commands"]
    assert cmds[-1] == ["forcelose", "p1"], cmds[-3:]      # the core's stall forfeit
    s = load_summary(sp)
    assert s["meta"]["result"] == "LOSS"
    assert all(inv["turn"] < STALL_LIMIT for inv in s["invocations"])


def test_the_protocol_log_stands_in_for_the_missing_replay_html(run):
    from main.prober.core_trace import load_summary, protocol_log

    _run_dir, normal, _stall = run
    sp = normal[0]
    assert not os.path.exists(sp.replace("_summary.json", "_replay.html"))
    log = protocol_log(sp, load_summary(sp))
    assert any(ln.startswith("|turn|") for ln in log) and any(ln.startswith("|move|") for ln in log)
    assert log[-1].startswith("|win|") and not any(ln.startswith("|request|") for ln in log)


# ---------------------------------------------------------------------------
# (b) the core's own readings agree with each other and with the STORED rows
# ---------------------------------------------------------------------------

def test_the_core_stream_rows_equal_the_stored_rows_and_the_choice_maps_name_the_legal_actions(run):
    """`core_walk.read_streams` over the trainee's walked stream (our actions replayed by index) reproduces the
    STORED rows byte for byte, and `core_walk.decision_choices` (the walk's reading) equals the stream's reading
    of the same decision and names exactly the actions the stored row's mask marks legal."""
    from main.prober import core_walk
    from utils.bridge.reconstruction import ReconstructionRecord

    _run_dir, normal, stall = run
    n_checked = 0
    for sp in normal + stall:
        rec = ReconstructionRecord.from_dict(_recon(sp))
        npz = _npz(sp)
        side = rec.side_of(rec.trainee_username)
        w = core_walk.walk(rec, side)
        (decs,) = core_walk.read_streams([core_walk.StreamRequest(
            rec.username(side), rec.packed_team(side), side, [c for s, c in w.chunks if s == side],
            actions=npz["actions"].tolist())])
        n = len(npz["actions"])
        assert len(decs) >= n, (sp, len(decs), n)
        bad = [k for k in range(n) if not np.array_equal(decs[k].obs, npz["obs"][k])]
        assert not bad, f"{sp}: the core stream's rows differ from the stored rows at {bad[:10]}"
        for k in sorted({0, n // 2, n - 1}):
            choices, turn = core_walk.decision_choices(rec, side, k)
            assert choices == decs[k].choices, (sp, k, choices, decs[k].choices)
            assert sorted(choices) == np.flatnonzero(npz["action_mask"][k]).tolist(), (sp, k)
            assert turn == decs[k].turn, (sp, k)
            n_checked += 1
    assert n_checked >= 6


# ---------------------------------------------------------------------------
# (c) a record that disagrees with the replay is REFUSED
# ---------------------------------------------------------------------------

def _copy_trace(sp: str, dst: str) -> str:
    base = os.path.basename(sp)[: -len("_summary.json")]
    src_dir = os.path.dirname(sp)
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src_dir):
        if name.startswith(base):
            shutil.copy(os.path.join(src_dir, name), dst)
    return os.path.join(dst, os.path.basename(sp))


def test_a_tampered_record_text_is_refused_naming_the_line(run, tmp_path):
    from main.prober.core_trace import CoreTraceMismatch, expand

    sp = _copy_trace(run[1][0], str(tmp_path / "t"))
    rp = sp.replace("_summary.json", ".p1.jsonl.gz")
    with gzip.open(rp, "rt") as f:
        rows = f.read().splitlines()
    j = next(i for i, r in enumerate(rows) if '"text":"|move|' in r)
    rows[j] = rows[j].replace("|move|", "|move|X", 1)
    with gzip.open(rp, "wt") as f:
        f.write("\n".join(rows) + "\n")
    with pytest.raises(CoreTraceMismatch, match=r"protocol line \d+: record='\|move\|X"):
        expand(sp, _stored(sp))


def test_states_rows_that_disagree_with_the_replay_are_refused(run, tmp_path):
    from main.prober.core_trace import CoreTraceMismatch, expand

    sp = _copy_trace(run[1][0], str(tmp_path / "n"))
    npz_path = sp.replace("_summary.json", "_states.npz")
    npz = _npz(sp)
    np.savez_compressed(npz_path, **{k: v[:-1] for k, v in npz.items()})
    with pytest.raises(CoreTraceMismatch, match="trainee decisions"):
        expand(sp, _stored(sp))

    npz["action_mask"][0] = ~npz["action_mask"][0]
    np.savez_compressed(npz_path, **npz)
    with pytest.raises(CoreTraceMismatch, match="decision 0 .* legal mask differs"):
        expand(sp, _stored(sp))


# ---------------------------------------------------------------------------
# (d) the JSON CLI on a core-trace run
# ---------------------------------------------------------------------------

def test_the_json_cli_reads_a_core_trace_run(run):
    run_dir, normal, stall = run
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(src_path()), os.environ.get("PYTHONPATH", "")]))

    def q(*argv):
        r = subprocess.run([sys.executable, "-m", "main.prober.query", *argv],
                           capture_output=True, text=True, env=env, timeout=600)
        assert r.returncode == 0, (argv, r.stderr[-3000:])
        return json.loads(r.stdout)

    summ = q("summary", run_dir)
    assert summ["run_dir"] == run_dir and summ["n_steps"] == 1
    ids = {row["id"] for row in q("scan", run_dir)}
    assert ids == set(normal + stall)
    turns = q("turns", normal[0])
    assert turns["n_decisions"] == _stored(normal[0])["meta"]["invocations"]
    assert any(e for t in turns["turns"] for d in t["decisions"] for e in (d.get("timeline") or []))
