"""TAU-ON-CUDA PARITY cell (REVIEW.md F2 of `../tau_fp32_alt_2026-10-20/`): do the PRODUCTION pair of programs land on ONE
τ on the card? The pair is T2's compiled CUDA rollout (inductor / Triton, ``dynamic=False``) against the CUDA learner
forward (`policy.evaluate_actions`, train mode, eager, as the K9(b) probe runs it). Every earlier "two programs land on
one τ" number compared CPU programs (ATen against inductor's C++ kernel, 32 candidate permutations).

THREE READS, in order, in one process tree (~9 min, the first compile cold):

1. **τ on synthetic saturated rows** (the review's literal fix): the CUDA case of
   `src/agents/model/tau_solve_test.py::test_fp32_alt_compiled_by_inductor_matches_eager_on_the_tau_class_row`, run as a
   pytest subprocess with ``GEN3AI_TEST_ALLOW_GPU=1 GEN3AI_SKIP_SLOW_STATUS_RECORD=1`` (it writes its record to
   ``$TAU_CUDA_PARITY_RECORD``). Compiled Triton `fixed_size_tau` against eager, leads 23 / 40 / 100 nats, k = 1 / 3.
2. **The network, taken-action log π** on (a) the three X26 τ-class VIOLATORS (A3 / A6 event dumps u32180, u32320,
   u35950: the violator + the scan's 32 worst, each at its event's NEAREST checkpoint, `taus.EVENTS`) and (b) the 915
   FIXTURE rows (`play_rows(24, seed=6201)`, T62's) at P* (X26 checkpoint_280000170): the compiled CUDA decision forward
   against the CUDA learner forward, in bucket-`BUCKET` chunks (the last chunk tiled; the compiled shape is the rollout's).
3. **τ on those same rows**: every row's species-presence τ inputs are CAPTURED from an eager CUDA decision forward
   (`taus.Tap`, call 0) and solved by the SAME shared helper as read 1 (`compiled_vs_eager_tau`): compiled Triton against
   eager, one batch.

Reported: τ's max |Δ| (absolute and in fp32 ulp of the row's largest |τ| / |score|) overall, p99, per set; the taken
action's |Δ log π| max / p99 / median per set, and the violators' own.

THE PASS BAR (preregistered 2026-10-20, before any CUDA datum; deterministic, on seeded inputs):
* read 1: every row's compiled-vs-eager τ within ``BAR_TAU_ULPS`` = 4 fp32 ulp (the CPU test's bar is 2; measured 0);
* read 3: every LIVE row (0 < k < n candidates) of every set within 4 ulp, the ulp taken at the row's scale
  ``max(|τ|, max_cand |score|)`` (the rounding of ``x = a + τ`` that the proof absorbs);
* read 2: the three violators' taken-action |Δ log π| <= ``BAR_LOGP_VIOLATORS`` = 1e-4 (K9(b)'s fatal bar), and the fixture
  rows' p99 <= 1e-4. The fixture MAX is reported, not judged (F4 of the FP32_ALT README: a non-τ fp32 tail at 4e-4 exists).
* the τ forcing record is clean (the compiled decision forward TRACED the wrapper; the learner and the capture reached
  it eagerly); every number is finite.
A FAIL is a HOLD (exit 9): the proof's worst case is 2c <= 4.8e-5 (F3), so a miss between the bar and that is a number to
read, not a refutation. WHAT IT DOES NOT SEE: the τ INSIDE the compiled decision forward (it is a graph-internal value;
read 3 solves the same inputs with the same function compiled standalone, which can fuse differently); the rollout's CUDA
graph (it replays the captured kernels, so it cannot change the numerics; this cell calls the compiled function directly).

    python3 tau_cuda_parity_cell.py --out cell.json        # on the GPU (the battery runs it under gpu_lock)
    python3 tau_cuda_parity_cell.py --check                # CPU: the API, the inputs, the pytest case exists
    python3 tau_cuda_parity_cell.py --check --prove        # CPU, ~3 min: the whole pipeline on the CPU (compile
                                                           #   backend "eager"), the verdict's teeth, and the clean SKIP

Without a CUDA device the cell SKIPS cleanly: result status ``skip``, exit 0 (the battery reads that as NOT ok -- a HOLD --
so a window with no card never counts as a pass).

Exit: 0 ok or skip · 3 a precondition (the FP32_ALT API is absent, an input file is missing) · 4 the forcing was violated
· 9 a pass bar was missed (status ``fail``).
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import tau_force as TF
from tau_parity_read import BAR_LOGP_FIXTURE_P99, BAR_LOGP_VIOLATORS, BAR_TAU_ULPS, judge

HERE = Path(__file__).resolve().parent
MEAS = HERE.parent
ROOT = HERE.parents[3]
SRC = ROOT / "src"

BUCKET = 256
FIXTURE_BATTLES, FIXTURE_SEED, FIXTURE_ROWS_EXPECTED = 24, 6201, 915
SOLVE = "fp32_alt"
PYTEST_FILE = "src/agents/model/tau_solve_test.py"
PYTEST_SELECT = "compiled_by_inductor and cuda"
RECORD_ENV = "TAU_CUDA_PARITY_RECORD"


def _taus() -> Any:
    """`tau_fp32_alt_2026-10-20/taus.py` (EVENTS, Tap, set_solve, the replay helpers), loaded as `equivalence.py` does."""
    spec = importlib.util.spec_from_file_location("fp32alt_taus", MEAS / "tau_fp32_alt_2026-10-20" / "taus.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _paths(TS: Any) -> dict[str, Path]:
    cache = Path(TS.CACHE)
    p280_dir = cache / "x35_refpop" / "pstar" / "rb_x26_s2601"
    out = {"p280_dir": p280_dir, "p280": p280_dir / "checkpoints" / "checkpoint_280000170_steps.zip"}
    for u, (d, ckpts) in TS.EVENTS.items():
        out[f"rows_u{u}"] = cache / d / f"behaviour_violation_u{u}_rows.npz"
        out[f"ckpt_u{u}"] = cache / d / f"checkpoint_{ckpts[0]}_steps.zip"
    return out


def preconditions(HS: Any) -> list[str]:
    bad = TF.check_api(HS)
    try:
        TS = _taus()
    except Exception as e:  # noqa: BLE001 -- reported as the precondition
        return bad + [f"cannot load taus.py: {e!r}"]
    bad += [f"missing input {k}: {p}" for k, p in _paths(TS).items() if not p.exists() and k != "p280_dir"]
    if not (SRC / "agents" / "model" / "tau_solve_test.py").is_file():
        bad.append("src/agents/model/tau_solve_test.py is missing")
    return bad


# ------------------------------------------------------------------------------------------------ read 1

def pytest_env(rec: Path) -> dict[str, str]:
    """The pytest subprocess's environment: the card un-hidden (``GEN3AI_TEST_ALLOW_GPU``; only under the battery's lease),
    no write into main's slow-tier status file, and the record path the CUDA case writes its numbers to."""
    return {**os.environ, "GEN3AI_TEST_ALLOW_GPU": "1", "GEN3AI_SKIP_SLOW_STATUS_RECORD": "1", RECORD_ENV: str(rec),
            "PYTHONPATH": str(SRC), "PYTHONDONTWRITEBYTECODE": "1"}


def run_pytest_tau(workdir: Path, timeout_s: int) -> dict[str, Any]:
    """Read 1: the CUDA case of the shared test, as a pytest subprocess (the root conftest hides the GPU unless
    ``GEN3AI_TEST_ALLOW_GPU=1``; ``GEN3AI_SKIP_SLOW_STATUS_RECORD=1`` keeps the slow-tier file untouched)."""
    rec = workdir / "tau_pytest_record.json"
    rec.unlink(missing_ok=True)
    env = pytest_env(rec)
    argv = [sys.executable, "-m", "pytest", PYTEST_FILE, "-k", PYTEST_SELECT, "-q", "-p", "no:cacheprovider", "-rs"]
    t0 = time.time()
    try:
        q = subprocess.run(argv, env=env, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout_s, check=False)
        rc, tail = q.returncode, (q.stdout + q.stderr).strip()[-1500:]
    except subprocess.TimeoutExpired:
        rc, tail = 124, f"pytest timed out after {timeout_s} s"
    out: dict[str, Any] = {"returncode": rc, "seconds": round(time.time() - t0, 1), "argv": argv[1:], "tail": tail}
    if rec.is_file():
        out.update(json.loads(rec.read_text()))
    else:
        out["why"] = ("pytest left no record: the CUDA case did not run (a SKIP, or a collection error); tail: "
                      + tail[-300:])
    return out


# ------------------------------------------------------------------------------------------------ reads 2 and 3

def _chunk_idx(start: int, stop: int, bucket: int) -> Any:
    import numpy as np
    base = np.arange(start, stop)
    return base[np.arange(bucket) % len(base)]          # the last chunk is tiled to the bucket's shape


def _stats(d: Any) -> dict[str, float]:
    import numpy as np
    return {"max": float(d.max()), "p99": float(np.quantile(d, 0.99)), "median": float(np.median(d)),
            "n": int(d.size)}


def _tau_ulps(comp: Any, eager: Any, scores: Any, cand: Any, k: Any) -> dict[str, Any]:
    """Per-row |comp - eager| in fp32 ulp of the row's scale max(|tau|, max_cand |score|), over LIVE rows (0 < k < n)."""
    import torch
    scale = torch.maximum(eager.abs(), (scores.abs() * cand).amax(-1)).clamp_min(1e-30)
    ulp = torch.nextafter(scale, torch.full_like(scale, float("inf"))) - scale
    d = (comp.double() - eager.double()).abs()
    live = (k > 0) & (k < cand.sum(-1))
    u = (d / ulp.double())[live]
    return {"live_rows": int(live.sum()), "rows": int(d.numel()), "max_abs": float(d[live].max()) if live.any() else 0.0,
            "p99_abs": float(torch.quantile(d[live].float(), 0.99)) if live.any() else 0.0,
            "max_ulps": float(u.max()) if live.any() else 0.0, "equal_frac": float((d[live] == 0).double().mean())
            if live.any() else 1.0, "finite": bool(torch.isfinite(d[live]).all()) if live.any() else True}


def network_reads(device: str, compile_backend: str | None, bucket: int, battles: int, seed: int,
                  save: Any = None) -> dict[str, Any]:
    """Reads 2 and 3 on ``device``. ``compile_backend=None`` is the rollout's inductor build; ``"eager"`` is the CPU
    smoke's (dynamo traces through the wrapper, nothing is built). Returns the record; raises on a broken pipeline."""
    import numpy as np
    import torch

    from agents.inference.service.decision import DecisionModule
    from agents.inference.service.engine import decide
    from agents.model import hypothesis_set as HS
    from agents.model.compile_cache import ensure_hermetic_cache
    from agents.model.extra_obs_keys import zero_extra_obs
    from agents.model.tau_solve_test import compiled_vs_eager_tau
    from main.capacity import load_policy
    from utils.rust_env.fixture_battles import play_rows

    TS = _taus()
    P = _paths(TS)
    dev = torch.device(device)
    TF.install(HS, SOLVE)                                  # BEFORE anything is compiled
    torch.set_float32_matmul_precision("highest")
    res: dict[str, Any] = {"device": device, "compile_backend": compile_backend or "inductor", "bucket": bucket,
                           "torch": torch.__version__, "sets": {}}
    t0 = time.time()
    model, _cfg, _ = load_policy(str(P["p280"]), str(P["p280_dir"]), "cpu")
    pol = model.policy
    TS.set_solve(pol, SOLVE)
    pol.to(dev)
    assert pol.features_extractor.hypothesis_builder.tau_solve == SOLVE
    dm = DecisionModule(pol.eval()).eval().to(dev)
    # --- the sets: the 915 fixture rows at P*, then each event's dump at its nearest checkpoint
    rows = play_rows(battles, seed=seed)
    fo = np.concatenate([o for o, _ in rows]).astype(np.float32)
    fm = np.concatenate([m for _, m in rows]).astype(bool)
    fobs = {"observation": fo, "action_mask": fm.astype(np.float32)}
    fobs.update({k: v.numpy() for k, v in zero_extra_obs(pol.features_extractor, batch=len(fo)).items()})
    sets: list[tuple[str, dict[str, Any] | None, Any, Any, Any]] = [("fixture", None, fobs, None, fm)]
    for u in TS.EVENTS:
        d = TS.RN.RD.rows_of(np.load(P[f"rows_u{u}"]))
        sets.append((f"u{u}", P[f"ckpt_u{u}"], d["obs"], d["acts"], d["obs"]["action_mask"].astype(bool)))
    res["fixture_rows"] = int(len(fo))
    res["fixture_rows_expected"] = FIXTURE_ROWS_EXPECTED if (battles, seed) == (FIXTURE_BATTLES, FIXTURE_SEED) else None
    assert sets[0][0] == "fixture"                       # the fixture runs at P*'s weights, the events then load theirs

    ensure_hermetic_cache("gpu window battery 2026-10-20 tau cuda parity cell")
    torch._dynamo.reset()
    compiled = torch.compile(decide, dynamic=False, **({"backend": compile_backend} if compile_backend else {}))
    captured: list[tuple[str, Any, Any, Any]] = []                       # (set, scores f32, cand, k) of call 0
    violators: dict[str, list[float]] = {"set": [], "dlogp": []}
    for name, ckpt, obs, acts, masks in sets:
        if ckpt is not None:
            pol.load_state_dict(TS.RN.policy_sd(str(ckpt)))              # in place: the compiled function sees it
        n = len(masks)
        lp_c = np.empty(n)
        lp_l = np.empty(n)
        taken = np.zeros(n, dtype=np.int64) if acts is None else np.asarray(acts, dtype=np.int64)
        s_scores, s_cand, s_k = [], [], []
        for start in range(0, n, bucket):
            stop = min(start + bucket, n)
            idx = _chunk_idx(start, stop, bucket)
            o = torch.as_tensor(obs["observation"][idx]).float().to(dev)
            m = torch.as_tensor(masks[idx]).bool().to(dev)
            with torch.no_grad():
                logp, _v, _g = compiled(dm, o, m)
                if acts is None:                                         # fixture: the compiled greedy legal action
                    taken[start:stop] = logp.argmax(-1)[: stop - start].cpu().numpy()
                with TS.Tap() as tap:                                    # read 3's inputs, from an eager forward
                    dm(o, m)
            a = torch.as_tensor(taken[idx]).long().to(dev)
            lp_c[start:stop] = logp.gather(-1, a[:, None])[: stop - start, 0].double().cpu().numpy()
            c0 = tap.calls[0]
            r = stop - start
            s_scores.append(c0["scores"].reshape(-1, c0["scores"].shape[-1])[:r].float())
            s_cand.append(c0["cand"].reshape(-1, c0["cand"].shape[-1])[:r])
            s_k.append(c0["k"].reshape(-1)[:r])
            ob = {k: torch.as_tensor(v[idx]).to(dev) for k, v in obs.items()}
            ob = {k: (v.float() if v.is_floating_point() else v) for k, v in ob.items()}
            pol.set_training_mode(True)
            with torch.enable_grad():
                _val, lp, _ent = pol.evaluate_actions(ob, a, action_masks=m)
            pol.set_training_mode(False)
            lp_l[start:stop] = lp.detach()[:r].double().cpu().numpy()
        d = np.abs(lp_l - lp_c)
        res["sets"][name] = {"rows": n, **{f"dlogp_{k}": v for k, v in _stats(d).items()},
                             "finite": bool(np.isfinite(d).all()), "over_1e-4": int((d > 1e-4).sum()),
                             "taken_greedy_from_compiled": acts is None}
        if name != "fixture":
            violators["set"].append(name)
            violators["dlogp"].append(float(d[0]))                       # row 0 of a dump is THE violator
            res["sets"][name]["violator_dlogp"] = float(d[0])
            res["sets"][name]["scan_rows_dlogp"] = _stats(d[1:])
        captured.append((name, torch.cat(s_scores), torch.cat(s_cand), torch.cat(s_k)))
        res["seconds"] = round(time.time() - t0, 1)
        if save:
            save(res)
    res["violators"] = violators
    res["forcing_midway"] = {"eager": dict(TF.EAGER), "traced": dict(TF.TRACED)}
    # --- read 3: ONE batch through the shared helper (it resets dynamo, so it runs after the decision reads)
    del compiled
    scores = torch.cat([c[1] for c in captured])
    cand = torch.cat([c[2] for c in captured])
    kk = torch.cat([c[3] for c in captured]).long()
    comp, eager = compiled_vs_eager_tau(scores.to(dev), cand.to(dev), kk.to(dev), device, solve=SOLVE)
    comp, eager = comp.float(), eager.float()
    sc, cn, kc = scores.cpu(), cand.cpu(), kk.cpu()
    per_set: dict[str, Any] = {}
    off = 0
    for name, s, _c, _k in captured:
        r = s.shape[0]
        sl = slice(off, off + r)
        per_set[name] = _tau_ulps(comp[sl], eager[sl], sc[sl], cn[sl], kc[sl])
        off += r
    allr = _tau_ulps(comp, eager, sc, cn, kc)
    worst = max(per_set, key=lambda s_: per_set[s_]["max_ulps"])
    res["tau_captured"] = {**allr, "worst_set": worst, "per_set": per_set, "batch_rows": int(scores.shape[0])}
    for name, v in per_set.items():
        res["sets"][name]["finite"] = bool(res["sets"][name]["finite"] and v["finite"])
        res["sets"][name]["tau_max_ulps"] = v["max_ulps"]
        res["sets"][name]["tau_max_abs"] = v["max_abs"]
    rep = TF.report(HS, [SOLVE])
    if TF.TRACED[SOLVE] <= 0:
        rep["violations"].append("the compiled decision forward never traced the wrapper (a vacuous compiled arm)")
    if TF.EAGER[SOLVE] <= 0:
        rep["violations"].append("the learner / capture forward never reached the wrapper eagerly")
    res["tau_force"] = rep
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ------------------------------------------------------------------------------------------------ the driver

def _save(path: Path) -> Any:
    def save(res: dict[str, Any]) -> None:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(res, indent=1, default=str))
        tmp.replace(path)
    return save


def prove(a: argparse.Namespace) -> str:
    """CPU, ~3 min: the verdict's teeth, the pytest CUDA case SKIPPING cleanly with no card, the whole network pipeline
    on the CPU (compile backend eager; the shared tau helper compiles the real CPU inductor kernel), the cell's own clean
    SKIP. Raises on any miss."""
    import tempfile
    notes = []
    # 1. the verdict has teeth: a clean synthetic result passes, each planted miss fails, for the named reason
    good = {"tau_pytest": {"max_ulps": 0.0, "returncode": 0}, "tau_captured": {"max_ulps": 1.0, "worst_set": "x"},
            "violators": {"dlogp": [1e-5, 2e-5, 3e-5]}, "sets": {"fixture": {"dlogp_p99": 2e-5, "finite": True}},
            "tau_force": {"violations": []}}
    assert judge(good) == [], judge(good)
    for label, mut in (
            ("synthetic tau", lambda r: r["tau_pytest"].update(max_ulps=BAR_TAU_ULPS + 0.5)),
            ("pytest exit", lambda r: r["tau_pytest"].update(returncode=1)),
            ("captured tau", lambda r: r["tau_captured"].update(max_ulps=BAR_TAU_ULPS + 0.5)),
            ("violator", lambda r: r["violators"].update(dlogp=[1e-5, 2e-4])),
            ("fixture p99", lambda r: r["sets"]["fixture"].update(dlogp_p99=2e-4)),
            ("non-finite", lambda r: r["sets"]["fixture"].update(finite=False)),
            ("forcing", lambda r: r["tau_force"].update(violations=["x"])),
            ("no pytest record", lambda r: r["tau_pytest"].update(max_ulps=None, why="skipped"))):
        bad = copy.deepcopy(good)
        mut(bad)
        assert judge(bad), f"the verdict missed a planted {label}"
    notes.append("the verdict passes a clean result and refuses 8 planted misses")
    env = pytest_env(Path("x"))
    assert env["GEN3AI_TEST_ALLOW_GPU"] == "1" and env["GEN3AI_SKIP_SLOW_STATUS_RECORD"] == "1" and env[RECORD_ENV] == "x"
    # 2. the pytest CUDA case is COLLECTED and SKIPS cleanly with no visible card (rc 0, no record, and the cell reads it
    #    as a missing record, never a pass)
    with tempfile.TemporaryDirectory(prefix="tau_cuda_parity_prove_") as td:
        env_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        assert env_cuda == "", f"--prove must run with CUDA_VISIBLE_DEVICES='' (got {env_cuda!r})"
        p = run_pytest_tau(Path(td), 600)
        assert p["returncode"] == 0 and "max_ulps" not in p, p
        assert "1 skipped" in p["tail"], p["tail"]
        assert judge({**good, "tau_pytest": p}), "a skipped pytest case must not pass"
        notes.append(f"the pytest CUDA case skips cleanly with no card ({p['tail'].splitlines()[-1].strip()})")
        # 3. the cell's own clean SKIP on the CPU
        out = Path(td) / "skip.json"
        q = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--out", str(out)], capture_output=True,
                           text=True, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(SRC)}, check=False)
        d = json.loads(out.read_text())
        assert q.returncode == 0 and d["status"] == "skip", (q.returncode, d, q.stderr[-400:])
        notes.append("the cell SKIPS cleanly with no CUDA (exit 0, status skip)")
    # 4. the whole network pipeline on the CPU, then the verdict on its record
    t0 = time.time()
    res = network_reads("cpu", "eager", a.smoke_bucket, a.smoke_battles, FIXTURE_SEED)
    assert set(res["sets"]) == {"fixture", "u32180", "u32320", "u35950"}, res["sets"].keys()
    assert not res["tau_force"]["violations"], res["tau_force"]
    assert len(res["violators"]["dlogp"]) == 3 and all(x == x for x in res["violators"]["dlogp"])
    assert res["tau_captured"]["batch_rows"] == sum(s["rows"] for s in res["sets"].values())
    assert res["tau_captured"]["live_rows"] > 0
    full = {**res, "tau_pytest": {"max_ulps": 0.0, "returncode": 0}}
    fails = judge(full)
    notes.append(f"the network pipeline ran on the CPU in {time.time() - t0:.0f} s (bucket {a.smoke_bucket}, "
                 f"{res['fixture_rows']} fixture rows from {a.smoke_battles} battle(s), 3 events); tau captured max "
                 f"{res['tau_captured']['max_ulps']:.2f} ulp over {res['tau_captured']['live_rows']} live rows, "
                 f"violators dlogp {['%.1e' % x for x in res['violators']['dlogp']]}, fixture p99 "
                 f"{res['sets']['fixture']['dlogp_p99']:.1e}; verdict on the CPU record: "
                 f"{'PASS' if not fails else 'FAIL ' + '; '.join(fails)}")
    return "; ".join(notes)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--prove", action="store_true", help="with --check: the heavy CPU proof (the dry run's)")
    ap.add_argument("--bucket", type=int, default=BUCKET)
    ap.add_argument("--battles", type=int, default=FIXTURE_BATTLES)
    ap.add_argument("--seed", type=int, default=FIXTURE_SEED)
    ap.add_argument("--smoke-bucket", type=int, default=32)
    ap.add_argument("--smoke-battles", type=int, default=2)
    ap.add_argument("--pytest-timeout-s", type=int, default=900)
    a = ap.parse_args(argv)

    import torch

    from agents.model import hypothesis_set as HS

    bad = preconditions(HS)
    if bad:
        print("PRECONDITION FAILED: " + "; ".join(bad), file=sys.stderr)
        return 3
    if a.check:
        print("ok: " + (prove(a) if a.prove else "API and inputs present; the CUDA pytest case is selected by "
                         f"`-k '{PYTEST_SELECT}'` in {PYTEST_FILE}"))
        return 0
    if not a.out:
        ap.error("--out is required (or --check)")
    out = Path(a.out)
    save = _save(out)
    res: dict[str, Any] = {"cell": "tau_cuda_parity", "status": "running", "solve": SOLVE,
                           "bars": {"tau_ulps": BAR_TAU_ULPS, "logp_violators": BAR_LOGP_VIOLATORS,
                                    "logp_fixture_p99": BAR_LOGP_FIXTURE_P99}, "torch": torch.__version__}
    if not torch.cuda.is_available():
        res.update(status="skip", reason="no visible CUDA device: this cell measures the card")
        save(res)
        print("SKIP: " + res["reason"])
        return 0
    save(res)
    with tempfile.TemporaryDirectory(prefix="tau_cuda_parity_") as td:     # BEFORE this process opens a CUDA context
        res["tau_pytest"] = run_pytest_tau(Path(td), a.pytest_timeout_s)
    res["gpu"] = torch.cuda.get_device_name(0)
    save(res)
    net = network_reads("cuda", None, a.bucket, a.battles, a.seed, save=lambda r: save({**res, **r}))
    res.update(net)
    res["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    fails = judge(res)
    res["failures"] = fails
    res["verdict"] = "PASS" if not fails else "FAIL"
    forcing = bool(res["tau_force"]["violations"])
    res["status"] = "ok" if not fails else "fail"
    if fails:
        res["reason"] = "; ".join(fails)
    save(res)
    brief = {"verdict": res["verdict"], "tau_pytest_max_ulps": res["tau_pytest"].get("max_ulps"),
             "tau_pytest_max_abs": res["tau_pytest"].get("max_abs_dtau"),
             "tau_captured": {k: res["tau_captured"][k] for k in ("max_abs", "p99_abs", "max_ulps", "live_rows")},
             "violators_dlogp": res["violators"], "fixture": {k: v for k, v in res["sets"]["fixture"].items()},
             "failures": fails}
    print(json.dumps(brief, indent=1, default=str))
    if forcing:
        print(res["reason"], file=sys.stderr)
        return 4
    return 9 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
