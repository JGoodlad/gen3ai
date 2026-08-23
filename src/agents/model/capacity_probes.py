"""The CAPACITY-EVAL BATTERY's engine — saturation probes over a frozen checkpoint.

The flywheel era piles distilled skills into ONE fixed-capacity trunk (no FiLM, no LoRA —
conditioning was closed by two independent nulls), so the question "is the shared trunk filling
up?" stops being academic. This module is the instrument: three probes plus a parameter census,
all offline, all read-only, all designed so the same numbers can be taken again next generation
at matched step and DIFFERENCED.

  (a) `rank_probe`          — representation effective rank (participation ratio + srank@0.99)
                              on five taps: post-encoder role tokens, post-transformer team
                              tokens, `value_pooled`, `pi_features`, `vf_features`.
  (b) `trainability_probe`  — Lyle et al., *Understanding and Preventing Capacity Loss in RL*:
                              how well a LINEAR head fits K fixed random target functions from
                              the FROZEN features, against the same probe on a fresh
                              randomly-initialised extractor of the same config.
  (c) `decodability_probe`  — linear-probe r²/AUC to ground-truth facts read straight out of the
                              obs vector. Deliberately easy; the metric is the DRIFT.
  (d) `parameter_utilization` — per-phase weight-norm census from the `state_dict` alone.

🚨 **THE VALIDITY RULE, and it comes from a retraction in this repo.** A "conditioning headroom"
claim was once derived from a low participation ratio (`PR(K_ū)=17`); the reading was a noise
artifact and the lever built on it was refuted by two independent nulls. The minted lesson is
*gate a lever on "does this quantity PREDICT performance", never on "is it low"*. So: **no number
here licenses a kill or a build on its own.** This battery is an EARLY-WARNING TRIPWIRE whose
alarms get INVESTIGATED — every metric ships a `validity` note saying what movement would mean
and what paired behavioural evidence would have to confirm it. See
`designs/research_state/capacity_battery.md`.

Pure NumPy for every estimator (torch only to run the forward), so the math is unit-testable
without a checkpoint.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.training.rank_metrics import effective_rank

# Bump when the JSON schema or an estimator's DEFINITION changes — a generation-over-generation
# reader must be able to refuse to difference two artifacts that measured different things.
CAPACITY_BATTERY_VERSION = 1

#: The five taps, in pipeline order. Token taps are captured as [N, T, D]; the rest as [N, D].
FEATURE_TAPS: Tuple[str, ...] = (
    "role_tokens",      # post-encoder, pre-transformer  [N, 12, D_MODEL]
    "team_tokens",      # post-transformer               [N, 12, D_MODEL]
    "value_pooled",     # the critic's CLS readout       [N, D_MODEL]
    "pi_features",      # the actor's final rep          [N, PROJECTION_DIM]
    "vf_features",      # the critic's final rep         [N, PROJECTION_DIM]
)
_TOKEN_TAPS = frozenset({"role_tokens", "team_tokens"})

#: Ridge penalties searched by OOF score. Features are z-scored per fold, so the scale is
#: comparable across taps of different width (diag(XᵀX) ≈ n_train either way).
L2_GRID: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)

DEFAULT_N_TARGETS = 8
DEFAULT_FOLDS = 5


# --------------------------------------------------------------------------- pure math

def kfold_indices(n: int, folds: int, seed: int) -> List[np.ndarray]:
    """Deterministic contiguous k-fold test indices over a seeded permutation of ``range(n)``."""
    if folds < 2 or n < folds:
        raise ValueError(f"need folds >= 2 and n >= folds, got n={n}, folds={folds}")
    perm = np.random.default_rng(seed).permutation(n)
    return [np.sort(part) for part in np.array_split(perm, folds)]


def random_targets(obs: np.ndarray, n_targets: int = DEFAULT_N_TARGETS,
                   seed: int = 0) -> np.ndarray:
    """→ ``[N, n_targets]`` fixed random target functions of the raw observation.

    ``t_k(x) = tanh( (signed_log(x) @ w_k) / sqrt(D) )`` with ``w_k ~ N(0, 1)`` from ``seed``.

    Two properties are load-bearing and neither is cosmetic:

    * **No batch statistics.** The `signed_log` squash (``sign(x)·log1p(|x|)``) is applied
      ELEMENTWISE, so the target function depends only on ``(seed, n_targets)`` and the obs
      layout — never on which states this run happened to sample. A z-scored target would have
      been silently re-defined by every generation's own eval traces, which is exactly the kind
      of moving yardstick that makes a cross-generation difference unreadable.
    * **The squash is not optional.** Raw obs columns span embedded IDs up to ~600 alongside
      0-1 fractions, so an unsquashed projection is ~entirely the ID columns and `tanh`
      saturates to a near-binary target that no head can fit gradually.
    """
    x = np.asarray(obs, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"obs must be [N, D], got shape {x.shape}")
    w = np.random.default_rng(seed).standard_normal((x.shape[1], int(n_targets)))
    z = np.sign(x) * np.log1p(np.abs(x))
    out: np.ndarray = np.tanh(z @ w / np.sqrt(x.shape[1]))
    return out


def _standardize(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = train.mean(0, keepdims=True)
    sd = train.std(0, keepdims=True)
    dead = sd < 1e-12                      # a constant column carries no signal; zero it rather
    sd = np.where(dead, 1.0, sd)           # than divide by ~0 and manufacture a huge feature
    tr = (train - mu) / sd
    te = (test - mu) / sd
    tr[:, dead[0]] = 0.0
    te[:, dead[0]] = 0.0
    return tr, te


def ridge_oof(X: np.ndarray, Y: np.ndarray, lambdas: Sequence[float] = L2_GRID,
              folds: int = DEFAULT_FOLDS, seed: int = 0,
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Cross-validated multi-target ridge → ``(oof [N, K], chosen_l2 [K])``.

    ONE economy SVD per fold serves every target AND every penalty
    (``β(λ) = V diag(s/(s²+λ)) Uᵀ y``), which is why this exists next to the prober's
    `fit_probe`: that helper is single-target and re-solves per (fold, λ), and the battery needs
    ~14 targets × 5 taps × 2 arms inside a 5-minute budget. The estimator is the same ridge with
    the same OOF-selected penalty; only the factorisation is shared.

    The penalty is chosen PER TARGET by out-of-fold R², so an easy fact and a hard random
    projection are not forced onto one compromise λ.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    n, k, n_lam = len(X), Y.shape[1], len(lambdas)
    all_oof = np.zeros((n, k, n_lam))
    for test_idx in kfold_indices(n, folds, seed):
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        x_tr, x_te = _standardize(X[mask], X[test_idx])
        y_mu = Y[mask].mean(0)
        u, s, vt = np.linalg.svd(x_tr, full_matrices=False)
        uty = u.T @ (Y[mask] - y_mu)                       # [r, K]
        for li, lam in enumerate(lambdas):
            coef = vt.T @ ((s / (s * s + float(lam)))[:, None] * uty)     # [D, K]
            all_oof[test_idx, :, li] = x_te @ coef + y_mu
    scores = np.stack([r2_columns(Y, all_oof[:, :, li]) for li in range(n_lam)])   # [L, K]
    best = np.argmax(scores, axis=0)
    return all_oof[np.arange(n)[:, None], np.arange(k)[None, :], best[None, :]], \
        np.asarray([float(lambdas[b]) for b in best])


def r2_columns(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Per-column coefficient of determination ``1 − SS_res/SS_tot``; 0.0 for a constant column.

    NOT clipped at zero: a negative OOF R² is a real and informative reading (the head fit worse
    than the target's own mean), and clipping it would hide the one outcome that says the probe
    itself failed.
    """
    Y = np.asarray(Y, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    if Y.ndim == 1:
        Y, P = Y[:, None], P[:, None]
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    ss_res = ((Y - P) ** 2).sum(0)
    return np.where(ss_tot > 0, 1.0 - ss_res / np.maximum(ss_tot, 1e-300), 0.0)


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based ROC AUC (ties averaged). NaN when one class is absent."""
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    pos, neg = float((y > 0.5).sum()), float((y <= 0.5).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    sorted_scores = score[order]
    i = 0
    while i < len(score):                              # average ranks within a tie group
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y > 0.5].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def jsonable(value: Any) -> Any:
    """Recursively coerce a report into STRICT JSON — NumPy scalars to Python, non-finite to null.

    `json.dump` happily writes bare ``NaN``/``Infinity``, which round-trip in Python and are a
    parse error everywhere else. A capacity artifact is meant to be differenced by whatever reads
    it next, so the non-finite cases (a `capacity_ratio` over a zero-r² reference; an AUC with one
    class absent) become ``null`` here rather than a token that only Python accepts.
    """
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def rank_summary(Z: np.ndarray) -> Dict[str, float]:
    """Effective-rank descriptors of a centered ``[N, D]`` representation.

    Delegates the spectrum to `agents.training.rank_metrics.effective_rank` (the project's
    tested estimator, already logged live under `rank/*` — one definition, so the battery and
    the training-time metric are differenceable) and adds the two the battery reports plus the
    width they must be read against.

    ``srank99`` is the count of leading dims holding 99% of the VARIANCE (`n99`), which is the
    spelling `rank_metrics` already uses in production. It is not Kumar et al.'s
    singular-value-threshold srank; the two are named alike in the literature and differ.
    """
    Z = np.asarray(Z, dtype=np.float64)
    r = effective_rank(Z)
    return {"pr": float(r["pr"]), "srank99": float(r["n99"]), "effrank": float(r["effrank"]),
            "n90": float(r["n90"]), "n95": float(r["n95"]),
            "dim": int(Z.shape[1]) if Z.ndim == 2 else 0,
            "n_rows": int(Z.shape[0]) if Z.ndim == 2 else 0,
            "pr_frac": float(r["pr"]) / float(Z.shape[1]) if Z.ndim == 2 and Z.shape[1] else 0.0}


# --------------------------------------------------------------------------- feature capture

def _pool(Z: np.ndarray) -> np.ndarray:
    """Token tap → one state-level row (mean over the token axis)."""
    pooled: np.ndarray = Z.mean(axis=1) if Z.ndim == 3 else Z
    return pooled


def capture_features(fe: Any, obs: np.ndarray, masks: np.ndarray, batch: int = 256,
                     device: str = "cpu") -> Dict[str, np.ndarray]:
    """Run the extractor over ``obs`` and return the five taps as NumPy arrays.

    The role/team taps come from forward HOOKS on `pokemon_encoder` / `team_transformer` (the
    same two modules `rank_metrics.rank_probe` hooks live, so the offline and online readings
    are of the same tensors); `value_pooled` comes off the typed `last_value_pooled` stash; the
    two projections are the extractor's own return. Hooks are always removed.
    """
    import torch

    caught: Dict[str, Any] = {}

    def _enc_hook(_m: Any, _i: Any, out: Any) -> None:
        caught["role_tokens"] = out.detach()                       # [B, 12, D]

    def _tt_hook(_m: Any, _i: Any, out: Any) -> None:
        caught["team_tokens"] = torch.cat([out[0], out[1]], dim=1).detach()   # [B, 12, D]

    parts: Dict[str, List[np.ndarray]] = {name: [] for name in FEATURE_TAPS}
    h1 = fe.pokemon_encoder.register_forward_hook(_enc_hook)
    h2 = fe.team_transformer.register_forward_hook(_tt_hook)
    try:
        with torch.no_grad():
            for i in range(0, len(obs), batch):
                ob = {"observation": torch.as_tensor(obs[i:i + batch], device=device),
                      "action_mask": torch.as_tensor(masks[i:i + batch], device=device).float()}
                pi, vf = fe(ob)
                parts["pi_features"].append(pi.float().cpu().numpy())
                parts["vf_features"].append(vf.float().cpu().numpy())
                parts["value_pooled"].append(fe.last_value_pooled.float().cpu().numpy())
                for name in ("role_tokens", "team_tokens"):
                    if name not in caught:
                        raise RuntimeError(
                            f"the {name} hook did not fire — the extractor's phase layout has "
                            "drifted and this tap would silently go missing from the battery")
                    parts[name].append(caught.pop(name).float().cpu().numpy())
    finally:
        h1.remove()
        h2.remove()
    return {name: np.concatenate(chunks) for name, chunks in parts.items()}


def build_fresh_extractor(policy: Any, seed: int = 0) -> Any:
    """A freshly initialised extractor of the SAME config — the trainability probe's reference arm.

    Reproduces the PRODUCTION init path rather than a bare constructor, because this repo has
    already been bitten by the difference: SB3's `_build` orthogonally re-initialises every
    Linear in the extractor (clobbering all 13 deliberate zero-inits), and
    `Gen3DualHeadMaskablePolicy.__init__` then calls `restore_identity_init()` to put them back.
    A reference arm built without both steps is not the network training starts from — see
    `src/agents/model/CLAUDE.md` → *Identity-at-init is NOT free*.
    """
    import torch
    from stable_baselines3.common.policies import ActorCriticPolicy

    torch.manual_seed(seed)
    fe = type(policy.features_extractor)(policy.observation_space,
                                         **(policy.features_extractor_kwargs or {}))
    fe.apply(lambda m: ActorCriticPolicy.init_weights(m, gain=float(np.sqrt(2))))
    restore = getattr(fe, "restore_identity_init", None)
    if callable(restore):
        restore()
    return fe.eval()


# --------------------------------------------------------------------------- ground-truth facts

def ground_truth_facts(obs: np.ndarray, layout: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Facts recoverable from the obs vector itself → ``{name: {values, task, note}}``.

    Every index is derived from `Gen3ObservationEncoder.get_layout()` and the named constants —
    nothing here is a hardcoded offset, so a layout change moves the reads with it or raises.

    These are DELIBERATELY EASY (each is a coordinate of the input, or a count over six of
    them). A high r² is the expected reading and says nothing on its own; the signal is the
    DRIFT of an established fact's decodability across generations at matched step.
    """
    from agents.observation.constants import POKEMON_ACTIVE_OFFSET

    x = np.asarray(obs, dtype=np.float64)
    parts, pk = layout["parts"], layout["pokemon"]
    hp_off = int(pk["hp"]["offset"])
    act_off = int(POKEMON_ACTIVE_OFFSET)

    def team(side: str) -> np.ndarray:
        p = parts[side]
        return x[:, int(p["start"]):int(p["end"])].reshape(len(x), *p["reshape"])

    def active_hp(block: np.ndarray) -> np.ndarray:
        idx = np.argmax(block[:, :, act_off], axis=1)
        return np.asarray(block[np.arange(len(block)), idx, hp_off])

    ours, opps = team("our_team"), team("opp_team")
    g0 = int(parts["global"]["start"])
    gl = layout["global_layout"]
    hz, clk = g0 + int(gl["hazards"]["offset"]), g0 + int(gl["clock"]["offset"])
    r0 = int(parts["reactive"]["start"])
    react = layout["reactive_layout"]

    return {
        "our_active_hp": {"values": active_hp(ours), "task": "regression",
                          "note": "HP fraction of our active mon"},
        "opp_active_hp": {"values": active_hp(opps), "task": "regression",
                          "note": "HP fraction of the opponent's active mon"},
        "our_alive": {"values": (ours[:, :, hp_off] > 0).sum(1).astype(float),
                      "task": "regression", "note": "count of our slots with hp > 0"},
        "opp_alive": {"values": (opps[:, :, hp_off] > 0).sum(1).astype(float),
                      "task": "regression",
                      "note": "count of opp slots with hp > 0 (UNREVEALED slots included — this "
                              "is what the obs says, not the true remaining party)"},
        "our_spikes": {"values": (x[:, hz] > 0).astype(float), "task": "classification",
                       "note": "spikes on OUR side (global hazards[0] > 0)"},
        "opp_spikes": {"values": (x[:, hz + 1] > 0).astype(float), "task": "classification",
                       "note": "spikes on the OPPONENT's side (global hazards[1] > 0)"},
        "clock_elapsed": {"values": x[:, clk], "task": "regression",
                          "note": "the global CLOCK group's log-elapsed scalar"},
        "turns_since_progress": {
            "values": x[:, r0 + int(react["turns_since_progress"]["offset"])],
            "task": "regression", "note": "the board block's no-progress clock"},
    }


# --------------------------------------------------------------------------- the probes

def rank_probe(feats: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """(a) Effective rank at every tap. Token taps are ranked over ``[N·T, D]`` — the rank of the
    TOKEN population, matching `rank_metrics.rank_probe`'s live `rank/trunk_*` definition."""
    out: Dict[str, Dict[str, float]] = {}
    for name in FEATURE_TAPS:
        Z = feats.get(name)
        if Z is None:
            continue
        flat = Z.reshape(-1, Z.shape[-1]) if Z.ndim == 3 else Z
        out[name] = rank_summary(flat)
    return out


def _skip_reason(values: np.ndarray, task: str) -> Optional[str]:
    if float(np.std(values)) < 1e-9:
        return "degenerate target: zero variance on this state sample"
    if task == "classification":
        rate = float(np.mean(values > 0.5))
        if rate < 0.02 or rate > 0.98:
            return f"degenerate target: positive rate {rate:.3f} outside [0.02, 0.98]"
    return None


def _fit_block(feats: Dict[str, np.ndarray], Y: np.ndarray, folds: int, seed: int,
               ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """One `ridge_oof` per tap over the SHARED target matrix — the SVD is amortised over every
    column, which is why the random targets and the ground-truth facts are fitted together."""
    fits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in FEATURE_TAPS:
        Z = feats.get(name)
        if Z is None:
            continue
        fits[name] = ridge_oof(_pool(Z), Y, folds=folds, seed=seed)
    return fits


def trainability_probe(fits_trained: Dict[str, Tuple[np.ndarray, np.ndarray]],
                       fits_fresh: Dict[str, Tuple[np.ndarray, np.ndarray]],
                       targets: np.ndarray, cols: slice) -> Dict[str, Dict[str, Any]]:
    """(b) Lyle-style capacity loss — random-target fit quality, trained vs fresh.

    ``capacity_ratio = r2_trained / r2_fresh``. Below 1 means the TRAINED features fit fresh
    random targets WORSE than a randomly-initialised network of the same shape does: capacity
    consumed. `nmse` is ``1 − r2`` (normalized MSE against the target's own variance), so the
    target scale cancels and the reading is comparable across generations.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for name in FEATURE_TAPS:
        if name not in fits_trained or name not in fits_fresh:
            continue
        r2_t = float(np.mean(r2_columns(targets, fits_trained[name][0][:, cols])))
        r2_f = float(np.mean(r2_columns(targets, fits_fresh[name][0][:, cols])))
        out[name] = {
            "r2_trained": r2_t, "r2_fresh": r2_f,
            "nmse_trained": 1.0 - r2_t, "nmse_fresh": 1.0 - r2_f,
            "capacity_ratio": (r2_t / r2_f) if abs(r2_f) > 1e-9 else float("nan"),
            "per_target_r2_trained": [float(v) for v in
                                      r2_columns(targets, fits_trained[name][0][:, cols])],
        }
    return out


def decodability_probe(fits_trained: Dict[str, Tuple[np.ndarray, np.ndarray]],
                       fits_fresh: Dict[str, Tuple[np.ndarray, np.ndarray]],
                       facts: Dict[str, Dict[str, Any]], fact_cols: Dict[str, int],
                       skipped: Dict[str, str]) -> Dict[str, Any]:
    """(c) Linear decodability of established facts, trained vs a random-features baseline.

    A regression fact reports OOF r²; a classification fact reports AUC of the same OOF ridge
    score (one estimator for both, so a fact's task cannot silently change what "the probe" is).
    """
    out: Dict[str, Any] = {"skipped": dict(skipped), "facts": {}}
    for fact, col in fact_cols.items():
        spec = facts[fact]
        y = np.asarray(spec["values"], dtype=np.float64)
        row: Dict[str, Any] = {"task": spec["task"], "note": spec["note"],
                               "target_std": float(np.std(y)), "taps": {}}
        if spec["task"] == "classification":
            row["pos_rate"] = float(np.mean(y > 0.5))
        def score(fits: Dict[str, Tuple[np.ndarray, np.ndarray]], tap: str, task: str) -> float:
            pred = fits[tap][0][:, col]
            return (auc_score(y, pred) if task == "classification"
                    else float(r2_columns(y, pred)[0]))

        for name in FEATURE_TAPS:
            if name not in fits_trained:
                continue
            row["taps"][name] = {
                "trained": score(fits_trained, name, spec["task"]),
                "fresh": score(fits_fresh, name, spec["task"]) if name in fits_fresh else None,
                "l2": float(fits_trained[name][1][col])}
        out["facts"][fact] = row
    return out


def parameter_utilization(fe: Any) -> Dict[str, Any]:
    """(d) Per-phase parameter census from the parameters alone — count, L2 norm, RMS, dead frac.

    RMS (``‖W‖₂ / sqrt(numel)``) is the comparable column: a raw norm grows with width, so two
    phases of different size are not differenceable on it. `zero_frac` is the cheap dead-weight
    read; a zero-init route that has never learned still shows ~1.0 here, which is the single
    most common reason a phase looks "unused".
    """
    phases: Dict[str, Dict[str, float]] = {}
    for full_name, p in fe.named_parameters():
        phase = full_name.split(".")[0]
        arr = p.detach().float().cpu().numpy().ravel()
        acc = phases.setdefault(phase, {"n_params": 0.0, "sq": 0.0, "n_zero": 0.0})
        acc["n_params"] += arr.size
        acc["sq"] += float(np.dot(arr, arr))
        acc["n_zero"] += float((arr == 0.0).sum())
    total = sum(v["n_params"] for v in phases.values()) or 1.0
    return {
        "n_params_total": int(total),
        "phases": {
            name: {"n_params": int(v["n_params"]),
                   "param_share": v["n_params"] / total,
                   "l2_norm": float(np.sqrt(v["sq"])),
                   "rms": float(np.sqrt(v["sq"] / max(v["n_params"], 1.0))),
                   "zero_frac": v["n_zero"] / max(v["n_params"], 1.0)}
            for name, v in sorted(phases.items(), key=lambda kv: -kv[1]["n_params"])
        },
    }


# --------------------------------------------------------------------------- orchestration

def run_battery(fe_trained: Any, fe_fresh: Any, obs: np.ndarray, masks: np.ndarray,
                layout: Dict[str, Any], *, n_targets: int = DEFAULT_N_TARGETS,
                folds: int = DEFAULT_FOLDS, seed: int = 0, batch: int = 256,
                device: str = "cpu",
                progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Run (a)-(d) and return the JSON-shaped result body (no `meta` — the CLI owns provenance)."""
    say = progress or (lambda _m: None)

    say("capturing features (trained)")
    feats = capture_features(fe_trained, obs, masks, batch=batch, device=device)
    say("capturing features (fresh reference)")
    feats_fresh = capture_features(fe_fresh, obs, masks, batch=batch, device=device)

    say("effective rank")
    rank = {"trained": rank_probe(feats), "fresh": rank_probe(feats_fresh)}

    targets = random_targets(obs, n_targets, seed=seed)
    facts = ground_truth_facts(obs, layout)
    skipped = {name: r for name, spec in facts.items()
               if (r := _skip_reason(np.asarray(spec["values"]), spec["task"])) is not None}
    live = [name for name in facts if name not in skipped]
    fact_cols = {name: n_targets + i for i, name in enumerate(live)}
    Y = np.column_stack([targets] + [np.asarray(facts[n]["values"], dtype=np.float64)
                                     for n in live])

    say(f"ridge probes ({Y.shape[1]} targets x {len(FEATURE_TAPS)} taps x 2 arms)")
    fits_t = _fit_block(feats, Y, folds, seed)
    fits_f = _fit_block(feats_fresh, Y, folds, seed)

    return {
        "battery_version": CAPACITY_BATTERY_VERSION,
        "rank": rank,
        "trainability": {
            "n_targets": int(n_targets), "seed": int(seed), "folds": int(folds),
            "target_family": "tanh(signed_log(obs) @ N(0,1) / sqrt(D))",
            "taps": trainability_probe(fits_t, fits_f, targets, slice(0, n_targets)),
        },
        "decodability": decodability_probe(fits_t, fits_f, facts, fact_cols, skipped),
        "params": parameter_utilization(fe_trained),
    }
