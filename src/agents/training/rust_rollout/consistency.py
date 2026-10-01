"""K9(b) BEHAVIOUR-POLICY CONSISTENCY and the STALENESS PROBE — one learner forward, before any
optimizer step of every update (M5 Lane G; ``program_rust_core.md`` Lane K9(b) + order constraint 6).

WHAT IT DOES. ``train()`` calls ``behaviour_probe(model)`` in its pre-loop setup, while the buffer is
still ``[n_steps, n_envs]`` and before the first ``optimizer.step()``. It picks ONE micro-batch of rows
(``batch_size`` of them — the learner's own compiled shape) and runs the LEARNER'S forward on them
(``policy.evaluate_actions`` in train mode, with autograd on exactly as a minibatch is, so the probe
reuses the training graph rather than declaring a new signature) and compares the recomputed
``log π(a|s)`` with the buffer's stored BEHAVIOUR log-prob ``μ(a|s)``:

* **K9(b)** — on the rows played at the CURRENT policy version (the weights the learner holds now),
  ``max |log π − log μ| < 1e-4`` or a typed `BehaviourMismatch` (``--behaviour-check fatal``, the
  default under ``--env-core rust``). It catches stale served weights (T2 not reloaded), an
  eval-vs-train-mode difference, and a rollout/learner observation mismatch. It would NOT catch a
  miscompile shared by both sides — K6's eager canary exists for that.
* **STALENESS** (owner, 2026-09-29: MEASURE it, never drop or down-weight a row for age) — on the rows
  of every older version, bucketed by AGE (updates since the version played the row): the PPO ratio
  ``π/μ`` at the start of the update (mean, mean |r − 1|), the share outside the clip band, and sb3's
  approx-KL ``mean((r − 1) − log r)``. With the buffer-wide age histogram (``FillReport``) these are the
  ``staleness/*`` tags the SIZING study and the version-pinning decision read.

Row choice: every current-version row first (up to half the micro-batch), then the older ages in equal
shares, deterministic (seeded by ``num_timesteps``). On today's Python path (no version record) every
row is current, so ``train()`` does not call this probe there: the in-loop gate
(``instrumented_ppo/learner_gates.py``) compares the first micro-batch's own forward instead (one host read,
no second forward) — ``fatal`` by default on both cores (M5 Lane K9).
"""
from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

#: K9(b)'s GATE, KEYED BY the float32 matmul precision the run actually uses
#: (``torch.get_float32_matmul_precision()`` — ``--matmul-precision``): one or more (STATISTIC of
#: |log π − log μ| over the rows played at the CURRENT version, bar) conditions, ALL of which must hold.
#: ONE table; both implementations (this probe and ``instrumented_ppo/learner_gates``) read it through
#: `judge_behaviour`. Measured on CUDA (``BAR_MEASUREMENT``: 4 seeds x 3 real updates x 6,144 real rows,
#: 2,048-row micro-batches, eager and compiled — n = 147,456 rows / 72 micro-batches per precision;
#: the localized faults in ``BAR_MEASUREMENT_LOCALIZED``):
#:
#: * ``highest`` (fp32) — ``max`` < 1e-4: healthy per-row max 1.9e-6 (52x headroom); the smallest
#:   one-step-stale fault micro-batch max 0.020 (200x over).
#: * ``high`` (TF32) — TWO conditions (orchestrator, 2026-09-30):
#:   - ``p99`` < ``TF32_P99_MULTIPLE`` x the healthy per-row p99.9 (1.17e-3), rounded up = 3.6e-3 — the
#:     GLOBAL faults (stale weights, a mode or sampling mismatch move EVERY row): healthy micro-batch
#:     p99 at most 8.5e-4 (4.2x under), one optimizer step stale at least 0.0112 (3.1x over). The max
#:     cannot do this alone: TF32's healthy tail (0.040, a few numerically sensitive rows) sits above
#:     the smallest one-step-stale micro-batch max (0.022).
#:   - ``max`` < ``TF32_MAX_MULTIPLE`` x the healthy per-row max (0.040), rounded up = 0.071 — the
#:     LOCALIZED gross faults the p99 cannot see when they touch < 1 % of rows (misaligned rows, a
#:     wrong action index, a mask mismatch: |Δ| of order 0.1–10). PERSISTENT: FATAL only when violated
#:     on ``TF32_MAX_PERSISTENCE`` consecutive updates (1.8x headroom over a limited sample — a long run
#:     will draw a rounding outlier above it; a real localized fault recurs every update).
#:   Residual blind spot: a fault on < 1 % of rows AND smaller than 0.071 (the TF32 noise floor's
#:   tail) — invisible under TF32 (the fp32 max sees it).
TF32_P99_MULTIPLE = 3.0
TF32_MAX_MULTIPLE = 1.75
#: The TF32 max condition's PERSISTENCE (orchestrator, 2026-09-30; k from data): a single violation is a
#: LOUD warning with a row dump; FATAL only when it RECURS on this many CONSECUTIVE updates. A real
#: localized fault (a mis-shuffle, a wrong action index, a mask bug) is systematic and recurs every
#: update; a rounding outlier is a fresh draw from fresh rows each update. k = 4, not the default 2: the
#: healthy TF32 max is HEAVY-tailed (``BAR_MEASUREMENT_TAIL``: 96 seeds, 589,824 rows; peaks-over-
#: threshold ξ = 0.40; a crossing of 0.071 on 1.1 % of updates [95 % 0.28–2.2 %]), so over 10,000 updates
#: 2 consecutive would false-FATAL with P = 0.72 (POT) and 3 with 0.014 [0.11 at the bootstrap's high
#: end]; 4 gives 1.6e-4 [2.5e-3; the GEV block-max fit 0.012]. The DECLARED criterion: the smallest k
#: whose 10k-update false-FATAL rate is < 1 % at the POT bootstrap's upper end.
TF32_MAX_PERSISTENCE = 4


class GateCondition(NamedTuple):
    statistic: str      # "max" | "p99" of |log π − log μ| over the current rows
    bar: float          # the condition holds iff statistic < bar
    persistence: int    # FATAL once violated on this many CONSECUTIVE updates (1 = single-shot)


BEHAVIOUR_GATES: Dict[str, Tuple[GateCondition, ...]] = {
    "highest": (GateCondition("max", 1e-4, 1),),
    "high": (GateCondition("p99", 3.6e-3, 1), GateCondition("max", 0.071, TF32_MAX_PERSISTENCE)),
}
BEHAVIOUR_BAR = BEHAVIOUR_GATES["highest"][0].bar
BAR_MEASUREMENT = "designs/research_state/measurements/k9_behaviour_bar_2026-09-30/result.json"
BAR_MEASUREMENT_LOCALIZED = "designs/research_state/measurements/k9_behaviour_bar_2026-09-30/corrupt_result.json"
BAR_MEASUREMENT_TAIL = "designs/research_state/measurements/k9_behaviour_bar_2026-09-30/tail_result.json"
#: The run-dir file a violation's row dump is appended to (one JSON line per violation).
VIOLATION_DUMP = "behaviour_violations.jsonl"
_DUMP_ROWS_LOG, _DUMP_ROWS_FILE = 10, 200


class UndeclaredPrecision(RuntimeError):
    """K9(b): no gate is declared for this float32 matmul precision (none has been measured)."""


def _precision() -> str:
    import torch as th

    return th.get_float32_matmul_precision()


def behaviour_gate(precision: Optional[str] = None) -> Tuple[GateCondition, ...]:
    """The conditions for ``precision`` (default: this process's matmul precision NOW)."""
    precision = _precision() if precision is None else str(precision)
    gate = BEHAVIOUR_GATES.get(precision)
    if gate is None:
        raise UndeclaredPrecision(
            f"[K9(b)] no behaviour gate is declared for float32 matmul precision {precision!r} (declared: "
            f"{sorted(BEHAVIOUR_GATES)}); measure its healthy |d log pi| ({BAR_MEASUREMENT}) and add a row")
    return gate


class Judged(NamedTuple):
    condition: GateCondition
    value: float
    ok: bool


def judge_behaviour(abs_d: np.ndarray, precision: Optional[str] = None) -> Tuple[Judged, ...]:
    """Every condition of the precision's gate on the per-row |Δ| of the current rows (NaN never passes)."""
    out = []
    for c in behaviour_gate(precision):
        v = behaviour_statistic(abs_d, c.statistic)
        out.append(Judged(c, v, bool(v < c.bar)))
    return tuple(out)


def _describe(judged: Tuple[Judged, ...], precision: str) -> str:
    return f"matmul precision {precision!r}: " + ", ".join(
        f"{j.condition.statistic} {j.value:.3g} {'<' if j.ok else 'NOT <'} {j.condition.bar:g}" for j in judged)


def _dump(model: Any, abs_d: np.ndarray, judged: Tuple[Judged, ...], precision: str, where: str,
          streaks: Dict[str, int], actions: Any, masks: Any,
          details: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """The offending rows — the largest |Δ| first — to the log (``_DUMP_ROWS_LOG``) and, when the model
    knows its run dir, appended to ``<run_dir>/behaviour_violations.jsonl`` (``_DUMP_ROWS_FILE``).
    ``details`` (the Rust probe's, `_violation_details`) adds per row both log-probs, both full masked
    distributions with their entropy and margin, and the row's collection provenance; plus the
    full-buffer scan, the violating rows' observations and the policy weights (`_dump_artifacts`)."""
    import json
    import os

    a = np.asarray(abs_d, dtype=np.float64).reshape(-1)
    order = np.argsort(-np.where(np.isnan(a), np.inf, a))[:_DUMP_ROWS_FILE]   # report ordering: NaN rows FIRST
    acts = None if actions is None else np.asarray(_host(actions)).reshape(-1)
    msk = None if masks is None else np.asarray(_host(masks)).reshape(a.size, -1)
    rows = [{"index": int(i), "abs_dlogp": float(a[i]),
             "action": None if acts is None else int(acts[i]),
             "mask": None if msk is None else "".join("1" if x > 0.5 else "0" for x in msk[i]),
             **({} if details is None else row_detail(details, int(i)))} for i in order]
    record = {"where": where, "num_timesteps": int(getattr(model, "num_timesteps", 0) or 0),
              "n_updates": int(getattr(model, "_n_updates", 0) or 0), "precision": precision,
              "conditions": [{"statistic": j.condition.statistic, "value": j.value, "bar": j.condition.bar,
                              "ok": j.ok, "persistence": j.condition.persistence,
                              "streak": streaks.get(f"{precision}:{j.condition.statistic}", 0)} for j in judged],
              "p99": behaviour_statistic(a, "p99"), "rows_judged": int(a.size), "rows": rows}
    if details is not None:
        if "route" in details:
            record["route"] = details["route"]
        if "scan" in details:              # the scan's arrays (underscored) go to the .npz, not the JSON
            record["scan"] = {k: v for k, v in details["scan"].items() if not k.startswith("_")}
    dump_dir = getattr(model, "behaviour_dump_dir", None)
    if details is not None and dump_dir:
        record["artifacts"] = _dump_artifacts(model, str(dump_dir), details, a, record["n_updates"])
    for r in rows[:_DUMP_ROWS_LOG]:
        extra = ("" if "logp_stored" not in r else
                 f"  log mu {r['logp_stored']:.6g} -> log pi {r['logp_recomputed']:.6g}  p {r['p_action_recomputed']:.3g}"
                 f"  env {r.get('env')} ep {r.get('episode')} dec {r.get('dec_n')} slot {r.get('slot')}")
        print(f"    row {r['index']:>5}  |d log pi| {r['abs_dlogp']:.4g}  action {r['action']}  mask {r['mask']}{extra}",
              flush=True)
    if "scan" in record:
        sc = record["scan"]
        print(f"    full-buffer scan: {sc['over_bar']} of {sc['rows']} current rows over {sc['bar']:g} "
              f"(max {sc['max']:.4g}, p99 {sc['p99']:.3g})", flush=True)
    if dump_dir:
        try:
            with open(os.path.join(str(dump_dir), VIOLATION_DUMP), "a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"    (full dump: {os.path.join(str(dump_dir), VIOLATION_DUMP)})", flush=True)
        except OSError as exc:      # the dump is diagnostics; the verdict below stands either way
            print(f"    (dump to {dump_dir} failed: {exc})", flush=True)
    return rows


#: Per violation: the observations of at most this many violating rows (an ``.npz``), and the policy
#: weights for at most `_DUMP_WEIGHTS_MAX` violations per process (``.pt``, ~40 MB each) — together they
#: make a dumped row re-evaluable offline on exactly the weights that played it.
_DUMP_OBS_ROWS, _DUMP_WEIGHTS_MAX = 32, 3


#: A masked log-prob at or below this is an ILLEGAL action (sb3's HUGE_NEG -1e8 on the learner side,
#: -inf on T2's): reported as null.
_ILLEGAL_BELOW = -1e7


def _legal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return np.where(x <= _ILLEGAL_BELOW, -np.inf, x)


def _finite_list(x: np.ndarray) -> List[Optional[float]]:
    return [float(v) if np.isfinite(v) else None for v in _legal(x)]


def _dist_stats(lp: np.ndarray, action: int) -> Dict[str, Any]:
    """Entropy, top-1 minus top-2 log-prob margin, argmax and p(action) of one masked log-prob row."""
    lp = _legal(lp)
    ok = np.isfinite(lp)
    if not ok.any():
        return {"entropy": None, "margin": None, "argmax": None, "p_action": None}
    p = np.exp(lp[ok])
    srt = np.sort(lp[ok])[::-1]
    return {"entropy": float(-(p * lp[ok]).sum()), "margin": float(srt[0] - srt[1]) if srt.size > 1 else None,
            "argmax": int(np.flatnonzero(ok)[int(np.argmax(lp[ok]))]),
            "p_action": float(np.exp(lp[action])) if 0 <= action < lp.size and np.isfinite(lp[action]) else 0.0}


def row_detail(details: Dict[str, Any], i: int) -> Dict[str, Any]:
    """One judged row's dump fields from `_violation_details`' arrays (row ``i`` of the judged rows)."""
    out: Dict[str, Any] = {"buffer_t": int(details["t"][i]), "buffer_e": int(details["e"][i]),
                           "logp_stored": float(details["logp_stored"][i]),
                           "logp_recomputed": float(details["logp_recomputed"][i])}
    action = int(details["action"][i])
    for side in ("stored", "recomputed"):
        dist = details.get(f"dist_{side}")
        if dist is None:
            continue
        out[f"dist_{side}"] = _finite_list(dist[i])
        for k, v in _dist_stats(dist[i], action).items():
            out[f"{k}_{side}"] = v
    for k, arr in details.get("provenance", {}).items():
        if k != "logp_all":
            out[k if k != "margin" else "draw_margin"] = arr[i].item()
    return out


def _dump_artifacts(model: Any, dump_dir: str, details: Dict[str, Any], abs_d: np.ndarray,
                    n_updates: int) -> Dict[str, Any]:
    """The violating rows' observations (``.npz``: every obs key, the action, both log-probs, the
    buffer coordinates) and — for the first `_DUMP_WEIGHTS_MAX` violations of this process — the policy
    weights that played them (``.pt``). Returns the file names written."""
    import os

    import torch as th

    out: Dict[str, Any] = {}
    try:
        bar = min(c.bar for c in behaviour_gate())
        over = np.flatnonzero(~(np.asarray(abs_d) < bar))
        over = over[np.argsort(-np.asarray(abs_d)[over])][:_DUMP_OBS_ROWS]
        sc = details.get("scan", {})
        rows = {f"obs__{k}": v[over] for k, v in details.get("obs", {}).items()}
        rows.update({"probe_index": over, "t": details["t"][over], "e": details["e"][over],
                     "action": details["action"][over], "logp_stored": details["logp_stored"][over],
                     "logp_recomputed": details["logp_recomputed"][over]})
        for k, v in sc.get("_obs", {}).items():
            rows[f"scan_obs__{k}"] = v
        for k in ("_t", "_e", "_action", "_logp_stored", "_logp_recomputed"):
            if k in sc:
                rows[f"scan{k}"] = sc[k]
        name = f"behaviour_violation_u{n_updates}_rows.npz"
        np.savez_compressed(os.path.join(dump_dir, name), **rows)
        out["rows_npz"] = name
        done = model.__dict__.setdefault("_behaviour_weight_dumps", 0)
        if done < _DUMP_WEIGHTS_MAX:
            name = f"behaviour_violation_u{n_updates}_policy.pt"
            th.save({k: v.detach().cpu() for k, v in model.policy.state_dict().items()},
                    os.path.join(dump_dir, name))
            model._behaviour_weight_dumps = done + 1
            out["policy_pt"] = name
    except OSError as exc:              # diagnostics: the verdict stands either way
        out["error"] = str(exc)
    return out


def _host(x: Any) -> Any:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else x


def enforce_behaviour(model: Any, abs_d: np.ndarray, *, where: str, actions: Any = None,
                      masks: Any = None, details: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Judge, track PERSISTENCE, dump, and act — the ONE enforcement both implementations call.

    Per condition a CONSECUTIVE-violation streak lives on the model (``_behaviour_streaks``, keyed by
    precision and statistic; a clean update resets it). A violation whose streak has reached the
    condition's ``persistence`` — or ANY non-finite |Δ| (a NaN is never rounding) — is FATAL under
    ``--behaviour-check fatal``; a violation still short of it is a LOUD warning; every violation dumps
    the offending rows. ``warn`` only ever warns. Returns the ``behaviour/*`` bar and streak tags."""
    precision = _precision()
    judged = judge_behaviour(abs_d, precision)
    streaks: Dict[str, int] = model.__dict__.setdefault("_behaviour_streaks", {})
    nonfinite = not np.isfinite(np.asarray(abs_d, dtype=np.float64)).all()
    metrics: Dict[str, float] = {}
    failed, fatal = [], []
    for j in judged:
        key = f"{precision}:{j.condition.statistic}"
        streaks[key] = streaks.get(key, 0) + 1 if not j.ok else 0
        metrics[f"behaviour/bar_{j.condition.statistic}"] = j.condition.bar
        metrics[f"behaviour/streak_{j.condition.statistic}"] = float(streaks[key])
        if not j.ok:
            failed.append(j)
            if nonfinite or streaks[key] >= j.condition.persistence:
                fatal.append(j)
    if not failed:
        return metrics
    worst = float(np.nanmax(np.asarray(abs_d, dtype=np.float64))) if np.size(abs_d) else 0.0
    msg = (f"[K9(b)] BEHAVIOUR-POLICY MISMATCH: on {where}, the learner's log pi(a|s) differs from the stored "
           f"behaviour log-prob by up to {worst:.3g} ({_describe(judged, precision)}) before any optimizer "
           "step — stale rollout weights, an eval-vs-train-mode difference, a rollout/learner observation "
           "mismatch, or (a localized fault) misaligned rows, a wrong action index or a mask mismatch")
    rule = "; ".join(f"{j.condition.statistic}: violated {streaks[f'{precision}:{j.condition.statistic}']} "
                     f"consecutive update(s), FATAL at {j.condition.persistence}" for j in failed)
    mode = str(getattr(model, "behaviour_check", "off") or "off")
    head = ("🛑 " if (fatal and mode == "fatal") else "🚨 ")
    print(f"{head}{msg} [{rule}]", flush=True)
    _dump(model, abs_d, judged, precision, where, streaks, actions, masks, details)
    if fatal and mode == "fatal":
        raise BehaviourMismatch(f"{msg} [{rule}]")
    if mode == "fatal":
        print("🚨 [K9(b)] NOT fatal YET: a single violation of a persistence>1 condition is a warning; "
              "the SAME condition violated on the next update(s) is FATAL (BEHAVIOUR_GATES).", flush=True)
    return metrics


def _stashed_logp(policy: Any) -> Optional[np.ndarray]:
    """The full masked log-probs of the policy's LAST ``evaluate_actions`` (its ``_last_pi_distribution``
    stash), as float64 numpy, or None."""
    pi = getattr(policy, "_last_pi_distribution", None)
    logits = getattr(getattr(pi, "distribution", pi), "logits", None)
    return None if logits is None else logits.detach().double().cpu().numpy()


def _route(model: Any) -> Dict[str, Any]:
    """Which forward produced the stored log-probs: the collector's T2 backend, buckets, lanes, the
    trainee slot(s), and the keyed draw's run seed (with env / episode / dec_n a row is replayable)."""
    col = getattr(model, "_rust_collector", None)
    svc = getattr(col, "svc", None)
    eng = getattr(svc, "engine", None)
    out: Dict[str, Any] = {"producer": "T2 trainee slot (rust collector)" if col is not None else "unknown"}
    for k, v in (("backend", getattr(eng, "backend", None)), ("buckets", getattr(eng, "buckets", None)),
                 ("lanes", getattr(eng, "n_lanes", None)), ("current_slot", getattr(col, "current_slot", None)),
                 ("run_seed", getattr(getattr(col, "cfg", None), "run_seed", None)),
                 ("policy_version", getattr(model, "_rust_version", None))):
        if v is not None:
            out[k] = [int(b) for b in v] if k == "buckets" else (str(v) if k == "backend" else int(v))
    return out


def _violation_details(model: Any, buf: Any, t: np.ndarray, e: np.ndarray, old: np.ndarray, new: np.ndarray,
                       full_new: Optional[np.ndarray], actions: np.ndarray,
                       obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Per judged row (aligned to the judged rows): the buffer coordinates, both log-probs, both full
    masked distributions (the stored one is the row the keyed draw used — `store.row_provenance`), the
    collection provenance, the observations, and the route that produced the stored side."""
    prov_all = getattr(model, "_rust_row_provenance", None) or {}
    prov = {k: np.asarray(v)[t, e] for k, v in prov_all.items()}
    return {"t": t, "e": e, "action": np.asarray(actions).astype(np.int64), "logp_stored": old,
            "logp_recomputed": new, "dist_recomputed": full_new, "dist_stored": prov.get("logp_all"),
            "provenance": prov, "obs": obs, "route": _route(model)}


#: The worst rows of the full-buffer scan the dump keeps (JSON fields + their observations in the .npz).
_SCAN_KEEP = 32


def scan_current(model: Any, ages: np.ndarray) -> Dict[str, Any]:
    """EVERY current-version row of the buffer through the learner's forward (train mode, chunks of
    ``batch_size`` — the probe's own shape): how many exceed the precision's smallest bar, the max and
    the p99, and the worst `_SCAN_KEEP` rows in full (`row_detail` fields; their observations under
    underscored keys for the ``.npz``). Run on a violation (one row, or many?) — or every update when
    ``model.behaviour_scan_all`` is set (a diagnostic driver's switch, never a flag)."""
    import torch as th
    from stable_baselines3.common.utils import obs_as_tensor

    buf = model.rollout_buffer
    n_envs = int(buf.n_envs)
    flat = np.flatnonzero(np.asarray(ages).reshape(-1) == 0)
    B = max(1, int(getattr(model, "batch_size", 0) or flat.size))
    bar = min(c.bar for c in behaviour_gate())
    absd = np.empty(flat.size)
    newv = np.empty(flat.size)
    full = np.full((flat.size, int(buf.action_masks.shape[-1])), np.nan)
    was_training = model.policy.training
    model.policy.set_training_mode(True)
    try:
        for s in range(0, flat.size, B):
            f = flat[s:s + B]
            t, e = f // n_envs, f % n_envs
            obs = {k: v[t, e] for k, v in buf.observations.items()}
            acts = th.as_tensor(buf.actions[t, e].reshape(-1)).long().to(model.device)
            masks = th.as_tensor(buf.action_masks[t, e]).to(model.device)
            with th.no_grad():
                _v, lp, _ent = model.policy.evaluate_actions(obs_as_tensor(obs, model.device), acts,
                                                             action_masks=masks)
            newv[s:s + f.size] = lp.detach().double().cpu().numpy()
            fl = _stashed_logp(model.policy)
            if fl is not None:
                full[s:s + f.size] = fl
            absd[s:s + f.size] = np.abs(newv[s:s + f.size] - buf.log_probs[t, e].astype(np.float64))
    finally:
        model.policy.set_training_mode(was_training)
    worst = np.argsort(-np.where(np.isnan(absd), np.inf, absd))[:_SCAN_KEEP]
    t, e = flat[worst] // n_envs, flat[worst] % n_envs
    d = _violation_details(model, buf, t, e, buf.log_probs[t, e].astype(np.float64), newv[worst], full[worst],
                           buf.actions[t, e].reshape(-1), {k: v[t, e] for k, v in buf.observations.items()})
    rows = []
    for j in range(worst.size):
        m = buf.action_masks[t[j], e[j]].reshape(-1)
        rows.append({"abs_dlogp": float(absd[worst[j]]), "action": int(d["action"][j]),
                     "mask": "".join("1" if x > 0.5 else "0" for x in m), **row_detail(d, j)})
    return {"rows": int(flat.size), "bar": float(bar), "over_bar": int((~(absd < bar)).sum()),
            "max": float(np.nanmax(absd)) if absd.size else 0.0,
            "p99": behaviour_statistic(absd, "p99"), "worst": rows,
            "_obs": d["obs"], "_t": t, "_e": e, "_action": d["action"], "_logp_stored": d["logp_stored"],
            "_logp_recomputed": d["logp_recomputed"]}


def behaviour_statistic(abs_d: np.ndarray, statistic: str) -> float:
    """The gate statistic of the per-row |Δ| (NaN in, NaN out — a NaN never passes a ``<`` bar)."""
    a = np.asarray(abs_d, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return 0.0
    if not np.isfinite(a).all():
        return float("nan")
    if statistic == "max":
        return float(a.max())
    if statistic == "p99":
        return float(np.quantile(a, 0.99))
    raise ValueError(f"unknown behaviour statistic {statistic!r}")


AGE_BUCKETS = ((0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 1 << 30))


class BehaviourMismatch(RuntimeError):
    """K9(b): the learner's log π(a|s) differs from the stored behaviour log-prob on current rows."""


def bucket_name(lo: int, hi: int) -> str:
    return f"{lo}" if lo == hi else (f"{lo}plus" if hi >= (1 << 29) else f"{lo}_{hi}")


def choose_rows(ages: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Flat indices of ``n`` rows (module docs): current (age 0) rows first, up to n // 2 (all of them
    when nothing older exists), then equal shares of each older bucket, topped up from what remains."""
    ages = np.asarray(ages).reshape(-1)
    n = int(min(n, ages.size))
    cur = np.flatnonzero(ages == 0)
    older = np.flatnonzero(ages > 0)
    take_cur = cur if older.size == 0 else cur[:max(n // 2, n - older.size)]
    take_cur = rng.permutation(take_cur)[:n] if take_cur.size > n else take_cur
    picked = [take_cur]
    left = n - take_cur.size
    if left > 0 and older.size:
        present = [(lo, hi) for lo, hi in AGE_BUCKETS[1:] if ((ages >= lo) & (ages <= hi) & (ages > 0)).any()]
        share = max(1, left // max(1, len(present)))
        chosen: List[np.ndarray] = []
        for lo, hi in present:
            pool = np.flatnonzero((ages >= lo) & (ages <= hi))
            chosen.append(rng.permutation(pool)[:share])
        got = np.concatenate(chosen) if chosen else np.zeros(0, np.int64)
        rest = np.setdiff1d(older, got)
        if got.size < left:
            got = np.concatenate([got, rng.permutation(rest)[:left - got.size]])
        picked.append(got[:left])
    return np.concatenate(picked).astype(np.int64)


def behaviour_probe(model: Any) -> Optional[Dict[str, float]]:
    """Run the probe on ``model.rollout_buffer`` (module docs); record ``behaviour/*`` and ``staleness/*``;
    raise `BehaviourMismatch` under ``fatal``. Returns the metrics (None when off)."""
    mode = str(getattr(model, "behaviour_check", "off") or "off")
    if mode == "off":
        return None
    import torch as th
    from stable_baselines3.common.utils import obs_as_tensor

    buf = model.rollout_buffer
    n_steps, n_envs = int(buf.buffer_size), int(buf.n_envs)
    versions = getattr(model, "_rust_row_versions", None)
    if versions is not None and np.asarray(versions).shape == (n_steps, n_envs):
        ages = int(getattr(model, "_rust_version", 0)) - np.asarray(versions, dtype=np.int64)
    else:
        ages = np.zeros((n_steps, n_envs), dtype=np.int64)
    rng = np.random.default_rng([int(getattr(model, "seed", 0) or 0), int(model.num_timesteps)])
    B = int(getattr(model, "batch_size", 0) or n_steps * n_envs)
    flat = choose_rows(ages, B, rng)
    t_idx, e_idx = flat // n_envs, flat % n_envs
    obs = {k: v[t_idx, e_idx] for k, v in buf.observations.items()}
    acts = th.as_tensor(buf.actions[t_idx, e_idx].reshape(-1)).long().to(model.device)
    masks = th.as_tensor(buf.action_masks[t_idx, e_idx]).to(model.device)
    old = buf.log_probs[t_idx, e_idx].astype(np.float64)
    age = ages[t_idx, e_idx]
    was_training = model.policy.training
    model.policy.set_training_mode(True)
    with th.enable_grad():
        _v, logp, _ent = model.policy.evaluate_actions(obs_as_tensor(obs, model.device), acts, action_masks=masks)
    new = logp.detach().float().cpu().numpy().astype(np.float64)
    full_new = _stashed_logp(model.policy)        # the SAME forward's full masked log-probs [B, A]
    model.policy.set_training_mode(was_training)
    d = new - old
    cur = age == 0
    out: Dict[str, float] = {"behaviour/rows_current": float(cur.sum()), "behaviour/rows_probed": float(flat.size)}
    worst = float(np.abs(d[cur]).max()) if cur.any() else float("nan")
    out["behaviour/max_abs_dlogp_current"] = worst
    clip = model.clip_range(model._current_progress_remaining) if callable(model.clip_range) else float(model.clip_range)
    ratio = np.exp(d)
    for lo, hi in AGE_BUCKETS:
        m = (age >= lo) & (age <= hi)
        if not m.any():
            continue
        r = ratio[m]
        name = bucket_name(lo, hi)
        out[f"staleness/probe_age_{name}_rows"] = float(m.sum())
        out[f"staleness/probe_age_{name}_ratio_mean"] = float(r.mean())
        out[f"staleness/probe_age_{name}_ratio_absdev"] = float(np.abs(r - 1.0).mean())
        out[f"staleness/probe_age_{name}_clip_frac"] = float((np.abs(r - 1.0) > float(clip)).mean())
        out[f"staleness/probe_age_{name}_approx_kl"] = float(((r - 1.0) - d[m]).mean())
    fill = getattr(model, "_rust_fill", None)
    if fill is not None:
        tot = max(1, sum(fill.age_hist.values()))
        out["staleness/age_mean"] = float(fill.mean_age)
        out["staleness/age_max"] = float(max(fill.age_hist) if fill.age_hist else 0)
        out["staleness/current_share"] = float(fill.current_share)
        for lo, hi in AGE_BUCKETS:
            k = sum(v for a, v in fill.age_hist.items() if lo <= a <= hi)
            if k:
                out[f"staleness/rows_age_{bucket_name(lo, hi)}_share"] = k / tot
        out["staleness/games_split"] = float(fill.games_split)
        out["staleness/carry_rows"] = float(fill.carry_rows)
        out["staleness/in_progress_rows"] = float(fill.in_progress_rows)
        out["staleness/rows_cut_total"] = float(fill.cut_rows_total)
    out["behaviour/p99_abs_dlogp_current"] = (behaviour_statistic(np.abs(d[cur]), "p99") if cur.any()
                                              else float("nan"))
    logger = getattr(model, "logger", None)
    try:
        # an update with no current rows judges nothing: every streak stands (neither a violation nor a pass)
        if cur.any():
            ci = np.flatnonzero(cur)
            details = None
            violated = not all(j.ok for j in judge_behaviour(np.abs(d[cur])))
            if violated or bool(getattr(model, "behaviour_scan_all", False)):
                details = _violation_details(model, buf, t_idx[ci], e_idx[ci], old[ci], new[ci],
                                             None if full_new is None else full_new[ci],
                                             buf.actions[t_idx[ci], e_idx[ci]].reshape(-1),
                                             {k: v[ci] for k, v in obs.items()})
                details["scan"] = scan_current(model, ages)
                out["behaviour/scan_rows_over_bar"] = float(details["scan"]["over_bar"])
                out["behaviour/scan_max_abs_dlogp"] = float(details["scan"]["max"])
            out.update(enforce_behaviour(
                model, np.abs(d[cur]), where=f"{int(cur.sum())} rows played at the CURRENT policy version",
                actions=acts[ci], masks=masks[ci], details=details))
    finally:
        if logger is not None:
            for k, v in out.items():
                logger.record(k, v)
        model._behaviour_probe_metrics = out
    return out
