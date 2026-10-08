"""``main.ops.eval_trace_gen`` on the Rust eval core, on a real (tiny) cycle.

A run laid out as a live one — ``model_config.json`` recording the eval regime, the read step's
``eval_traces/step_<N>/snapshot.zip`` and one pool snapshot BELOW it — whose two checkpoints are seeded PERTURBED
fresh current-architecture policies (``rust_eval.offline.build_models``: a greedy decision is not the zero-init
pointer head's uniform tie). The cycle is generated TWICE at one seed and once at another (2 games x 3 opponents
each; a MODULE fixture: the tier budget is per test call), and the tests read the result:

(a) the READERS accept it — ``main.ops.critic_read``'s cycle pick and refusal gates, the prober's ``summary`` /
    ``scan``, ``cf_audit``'s identity half (the snapshot and the sentinel pins resolve against the shadow run) — and
    ``cf_audit``'s sampling FRAME refuses it, loudly, as it refuses every Rust-eval core trace (F-LH-4: no recorded
    win-prob head);
(b) a rerun at the same ``--seed`` is IDENTICAL, file for file and byte for byte; another seed is another cycle;
(c) the manifest's regime and sentinel blocks are the RUN's.
"""
from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from main.ops import eval_trace_gen as ETG

pytestmark = [pytest.mark.sim, pytest.mark.integration]

STEP = 8_192
SENTINEL_STEP = 4_096
BOTS = ("random", "heuristic")


def _a_run(root: Path) -> Path:
    from agents.training.rust_eval import offline as OFF

    trainee, sentinels, cfg = OFF.build_models(root / "build", n_sentinels=1)
    run = root / "archive" / "rb_etg_fixture"
    (run / "eval_traces" / f"step_{STEP}").mkdir(parents=True)
    (run / "snapshots").mkdir()
    rec = json.loads(Path(cfg).read_text())
    rec.update(eval_sentinel_greedy=True)
    (run / "model_config.json").write_text(json.dumps(rec))
    shutil.copy2(trainee, run / "eval_traces" / f"step_{STEP}" / "snapshot.zip")
    shutil.copy2(sentinels[0], run / "snapshots" / f"snapshot_{SENTINEL_STEP:012d}.zip")
    return run


def _gen(run: Path, out: Path, seed: int) -> dict:
    rc = ETG.main([f"{run}@{STEP}", "--games", "2", "--sentinels", "1", "--opponents", ",".join(BOTS),
                   "--out", str(out), "--seed", str(seed), "--shard-games", "2", "--n-envs", "2",
                   "--threads", "2", "--torch-threads", "2", "--front", "ffi", "--profile", "selfcheck",
                   "--nice", "0"])
    assert rc == 0
    return json.loads((out / "eval_traces" / f"step_{STEP}" / "eval_manifest.json").read_text())


@pytest.fixture(scope="module")
def cycles(tmp_path_factory):
    from agents.training.rust_rollout.testkit import build_selfcheck

    build_selfcheck()
    root = tmp_path_factory.mktemp("etg")
    run = _a_run(root)
    out = {"run": run}
    for name, seed in (("a", 13), ("b", 13), ("c", 14)):
        out[name] = root / f"cycle_{name}"
        out[f"man_{name}"] = _gen(run, out[name], seed)
    return out


def _step_dir(out: Path) -> Path:
    return out / "eval_traces" / f"step_{STEP}"


def _files(out: Path) -> list:
    sd = _step_dir(out)
    return sorted(str(p.relative_to(sd)) for p in sd.rglob("*") if p.is_file())


# ------------------------------------------------------------------------------------------------- (c) the regime

def test_the_manifest_records_the_RUNs_regime_and_the_sentinels_it_played(cycles):
    run, man = cycles["run"], cycles["man_a"]
    rec = json.loads((run / "model_config.json").read_text())
    gen = man[ETG.GENERATED_KEY]
    assert ETG.is_generated(man) and gen["tool"] == ETG.GENERATOR_TAG
    assert gen["eval_sentinel_greedy"] is rec["eval_sentinel_greedy"] is True
    assert gen["eval_sentinel_greedy_source"] == "recorded"
    assert gen["eval_mirrored_pairs"] is bool(rec.get("eval_mirrored_pairs", False)) is False
    assert gen["oracle_reveal"] == rec.get("oracle_reveal", "off") == "off"
    assert gen["transport"] == ETG.TRANSPORT and gen["eval_core"]["front"] == "ffi"
    assert (gen["seed"], gen["seed_source"], gen["reproducible"]) == (13, "given", True)
    assert gen["sentinel_steps"] == [SENTINEL_STEP] and gen["sentinels_used"] == 1 and gen["bots"] == list(BOTS)
    assert (gen["battles_expected"], gen["battles_played"], gen["complete"]) == (6, 6, True)
    # the shadow metadata pins the sentinel the cycle PLAYED (cf_audit reads it)
    meta = json.loads((cycles["a"] / "metadata.json").read_text())
    assert meta["latest_eval"]["pool"]["sentinels"] == [
        {"step": SENTINEL_STEP, "snapshot": f"snapshot_{SENTINEL_STEP:012d}.zip"}]
    assert meta["derived_from"]["tool"] == ETG.GENERATOR_TAG
    # rule 17: per-opponent counts and FULL capture
    per = man["selection"]["opponents"]
    assert set(per) == {*BOTS, "sentinel_0"}
    for key, r in per.items():
        assert r["battles_played"] == 2 and r["traces_written"] == 2, (key, r)
    assert (cycles["a"] / "snapshots").is_symlink() and not list(_step_dir(cycles["a"]).glob("shard__*.json"))


# ------------------------------------------------------------------------------------------------- (b) the seed

def test_a_rerun_at_the_same_seed_is_identical_file_for_file(cycles):
    a, b = cycles["a"], cycles["b"]
    fa, fb = _files(a), _files(b)
    assert fa == fb and sum(f.endswith("_states.npz") for f in fa) == 6
    for rel in fa:
        pa, pb = _step_dir(a) / rel, _step_dir(b) / rel
        if rel.endswith(".npz"):
            with np.load(pa) as x, np.load(pb) as y:
                assert sorted(x.files) == sorted(y.files)
                for k in x.files:
                    assert np.array_equal(x[k], y[k], equal_nan=True), (rel, k)
        elif rel.endswith(".gz"):
            assert gzip.decompress(pa.read_bytes()) == gzip.decompress(pb.read_bytes()), rel
        elif rel == "eval_manifest.json":
            ma, mb = json.loads(pa.read_text()), json.loads(pb.read_text())
            for m in (ma, mb):
                for k in ("generated_at", "wall_seconds", "games_per_sec", "source_run_dir"):
                    m[ETG.GENERATED_KEY].pop(k, None)
                m.pop("saved_at", None)
            assert ma == mb
        else:
            assert pa.read_bytes() == pb.read_bytes(), rel


def test_another_seed_is_another_cycle(cycles):
    """The seed reaches the games: another seed's reconstruction records (the teams, the battle seed) differ."""
    def recons(out):
        return sorted(p.read_text() for p in _step_dir(out).rglob("*_reconstruction.json"))
    assert len(recons(cycles["c"])) == 6 and recons(cycles["a"]) != recons(cycles["c"])
    assert cycles["man_c"][ETG.GENERATED_KEY]["seed"] == 14


# ------------------------------------------------------------------------------------------------- (a) the readers

def test_critic_read_picks_the_generated_cycle_and_its_gates_read_it(cycles):
    from main.ops import critic_read as CR

    out = cycles["a"]
    assert CR.resolve_read_root(cycles["run"], str(out), role="arm") == out.resolve()
    cyc = CR.pick_cycle(out, on_live="use", step=None)
    assert cyc["step"] == STEP and cyc["manifest"]["selection"]
    cov = CR.winprob_coverage(cyc["trace_dir"])
    assert cov["n_npz"] == 6 and cov["n_missing"] == 0
    assert CR.draw_share(cyc["manifest"]) is not None
    # two generated frames at one spec (different seeds) are comparable; the population says what it is
    CR.check_comparable({"manifest": cycles["man_a"]}, {"manifest": cycles["man_c"]})
    assert ETG.population_of(cycles["man_a"]).startswith("OFFLINE-GENERATED cycle on the Rust eval core")


def test_the_prober_summary_and_scan_read_the_generated_cycle(cycles):
    from main.prober import query

    p = query._build_parser()
    summ = query._run(p.parse_args(["summary", str(cycles["a"])]))
    assert summ["n_steps"] == 1
    rows = query._run(p.parse_args(["scan", str(cycles["a"])]))
    assert len(rows) == 6


def test_cf_audit_resolves_the_identity_against_the_shadow_run_and_its_frame_refuses_core_traces(cycles):
    """The identity half: the traces dir, the network that played it (``snapshot.zip`` in the step dir) and the
    pinned sentinel resolve inside the shadow run. The FRAME refuses: the sampling frame needs a recorded win-prob
    head, and a Rust-eval core trace records none (F-LH-4) — the same refusal every live Rust-core cycle meets."""
    from agents.training import cf_audit as CF
    from main.prober.core_trace import CoreTraceUnsupported

    out = cycles["a"]
    traces = CF._resolve_traces(str(out), None)
    assert Path(traces) == _step_dir(out) and (Path(traces) / "snapshot.zip").exists()
    snaps = CF.sentinel_snapshots(str(out))
    assert set(snaps) == {"sentinel_0"} and Path(snaps["sentinel_0"]).name == f"snapshot_{SENTINEL_STEP:012d}.zip"
    assert Path(snaps["sentinel_0"]).resolve() == (cycles["run"] / "snapshots" /
                                                    f"snapshot_{SENTINEL_STEP:012d}.zip").resolve()
    with pytest.raises(CoreTraceUnsupported, match="cf_audit.build_frame"):
        CF.build_frame(traces)
