"""pytest plugin (the 2026-11-11 diagnosis): make a Triton compile failure name its kernel.

Load with `-p triton_diag_plugin` and this file's directory on PYTHONPATH. It changes NO test outcome:

  * every in-process `CachingAutotuner.precompile` is counted; one that RAISES is diagnosed and the SAME
    exception is re-raised. The diagnosis: each autotune config of the failing kernel is recompiled alone,
    in-process (so Triton's MLIR diagnostic lands on this process's stderr, bracketed by marker lines), the
    kernel's source is saved to `$TRITON_DIAG_OUT/failing_kernels/<name>.py` with its size hints and the
    per-config results, and `$TRITON_DIAG_OUT/triton_diag.json` is rewritten.
    Inductor compiles in-process only with `TORCHINDUCTOR_COMPILE_THREADS=1`; with the default worker pool
    the failing compile runs in a subprocess this patch cannot see (the plugin then only counts).
  * `TRITON_DIAG_BATCH=<B>` (a COUNTERFACTUAL, never the test as written): the test module's `_obs` helper
    takes batch B by default instead of the module's `_BATCH`. Printed loudly; refused if `_obs` has no
    `batch` parameter.

`install(out_dir, graph_label=None)` is also what the CPU fake-CUDA harness calls (keep_going=True there).
"""
import inspect
import json
import os

STATE = {"installed": False, "precompiles": 0, "failures": [], "out": None, "batch_override": None,
         "keep_going": False, "label": lambda: "?"}


def _say(msg):
    os.write(2, (msg + "\n").encode())


def _dump():
    out = STATE["out"]
    if not out:
        return
    with open(os.path.join(out, "triton_diag.json"), "w") as fh:
        json.dump({"precompiles_in_process": STATE["precompiles"], "failures": STATE["failures"],
                   "batch_override": STATE["batch_override"],
                   "compile_threads_env": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS")}, fh, indent=1)


def _diagnose(autotuner, exc, configs):
    name = getattr(autotuner.fn, "__name__", "?")
    label = STATE["label"]()
    _say(f"\n=== TRITON_DIAG: FAILING kernel {name} [{label}] size_hints={autotuner.size_hints!r}: {exc!r}"[:600])
    per_cfg = []
    for c in configs:   # the list as it was BEFORE the compile (a finished compile sets `.configs = None`)
        _say(f"--- TRITON_DIAG: recompiling config {c} alone ---")
        try:
            autotuner._precompile_config(c)
            per_cfg.append((str(c), "OK"))
        except Exception as e2:  # noqa: BLE001 - a diagnostic records whatever the compiler raised
            per_cfg.append((str(c), "FAIL " + repr(e2)[:160]))
        _say(f"--- TRITON_DIAG: config {c}: {per_cfg[-1][1]} ---")
    rec = {"graph": label, "kernel": name, "size_hints": repr(autotuner.size_hints), "error": repr(exc)[:400],
           "configs": per_cfg, "source": None}
    out = STATE["out"]
    if out:
        try:
            d = os.path.join(out, "failing_kernels")
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{label}__{name}.py" if label != "?" else f"{name}.py")
            with open(path, "w") as fh:
                fh.write(f"# size_hints={autotuner.size_hints!r}\n# configs={per_cfg!r}\n")
                fh.write(autotuner.fn.src)
            rec["source"] = path
            meta = autotuner.triton_meta
            rec["signature"] = {k: str(v) for k, v in meta.get("signature", {}).items()}
            rec["constants"] = {k: repr(v) for k, v in meta.get("constants", {}).items()}
            import pickle
            with open(path[:-3] + ".meta.pkl", "wb") as fh:   # replays the compile off-line (`ASTSource` inputs)
                pickle.dump({"signature": meta.get("signature"), "constants": meta.get("constants"),
                             "attrs": meta.get("configs"), "configs": [str(c) for c in configs],
                             "config_kwargs": [dict(c.kwargs, num_warps=c.num_warps, num_stages=c.num_stages)
                                               for c in configs]}, fh)
        except Exception as e3:  # noqa: BLE001
            _say(f"TRITON_DIAG: could not save the kernel source: {e3!r}")
    STATE["failures"].append(rec)
    _dump()
    print(f"KERNEL FAIL [{label}] {name}: {exc!r} configs={per_cfg} source={rec['source']}"[:1500], flush=True)


def install(out_dir, *, keep_going=False, label=None, compile_only=False):
    """Patch `CachingAutotuner.precompile` once. `keep_going` swallows the failure (the CPU harness wants
    every failing kernel); the pytest plugin always re-raises. `compile_only` (the CPU harness: no CUDA driver)
    runs only the compile half of `precompile`, never the launcher half."""
    if STATE["installed"]:
        return
    from torch._inductor.runtime import triton_heuristics as th
    STATE.update(installed=True, out=out_dir, keep_going=keep_going)
    if label is not None:
        STATE["label"] = label
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    orig = th.CachingAutotuner.precompile

    def precompile(self, *args, **kwargs):
        STATE["precompiles"] += 1
        configs = list(self.configs or [])
        try:
            return self._precompile_worker() if compile_only else orig(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            _diagnose(self, exc, configs)
            if STATE["keep_going"]:
                return None
            raise

    th.CachingAutotuner.precompile = precompile
    _say(f"TRITON_DIAG: precompile patch installed (out={out_dir}, keep_going={keep_going}, "
         f"TORCHINDUCTOR_COMPILE_THREADS={os.environ.get('TORCHINDUCTOR_COMPILE_THREADS')!r})")


# ---------------------------------------------------------------- pytest hooks

def pytest_runtest_setup(item):
    # (registered tryfirst below: ahead of pytest's own skip evaluation, so a SKIPPED run still shows the lines)
    install(os.environ.get("TRITON_DIAG_OUT"))
    want = os.environ.get("TRITON_DIAG_BATCH")
    if not want:
        return
    fn = getattr(item.module, "_obs", None)
    if fn is None or "batch" not in inspect.signature(fn).parameters:
        raise RuntimeError("TRITON_DIAG_BATCH set, but this checkout's test module has no `_obs(..., batch=)`")
    params = [p for p in inspect.signature(fn).parameters.values()
              if p.default is not inspect.Parameter.empty and p.kind is p.POSITIONAL_OR_KEYWORD]
    if "batch" not in {p.name for p in params}:
        raise RuntimeError("TRITON_DIAG_BATCH: `_obs`'s `batch` is not a positional default; edit the plugin")
    old = tuple(p.default for p in params)
    new = tuple(int(want) if p.name == "batch" else p.default for p in params)
    fn.__defaults__ = new
    STATE["batch_override"] = int(want)
    _say(f"TRITON_DIAG: COUNTERFACTUAL -- `_obs` defaults {old} -> {new}: this is NOT the test as written")


pytest_runtest_setup.tryfirst = True  # hookimpl marker without importing pytest at module load


def pytest_sessionfinish(session, exitstatus):
    if STATE["installed"]:
        _dump()
        _say(f"TRITON_DIAG: {STATE['precompiles']} in-process precompiles, {len(STATE['failures'])} failing "
             f"kernel(s), batch_override={STATE['batch_override']}")
