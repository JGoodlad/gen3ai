"""DEPTH-USE and CAPACITY-USE diagnostics (`gen3_probe_battery_v1`, the owner's scope addition of 2026-10-09).

Per checkpoint (``pin_worker depth``, on a deterministic every-k-th-row subsample of the bank):

* **Depth use.** The LOGIT LENS (``lens_L2``: the heads read the representation after layer 1 — the last round's
  whole contribution removed; for this post-LN block it is also "skip the layer"), ``zero_update_L*`` (both residual
  updates of a round zeroed, its LayerNorms kept), ``no_attn_L*`` / ``no_ffn_L*`` (one update zeroed), each scored
  against the full forward as KL(policy_full ‖ policy_variant) over rows with >= 2 legal actions, top-1 agreement and
  the win-prob's mean absolute change; and ``‖Δx‖ / ‖x‖`` per token type per layer.
* **Capacity use.** The participation ratio / n90 / n99 of the token representations per token type and depth (of
  d = 128); the singular-value spectra of every trunk matrix (q / k / v, out, FFN in / out) and the input token
  projections; a LOW-RANK TRUNCATION of the trunk's matrices (and, separately, the input projections) to rank
  r ∈ {16, 32, 48, 64, 96}; every attention head ablated alone; dead / always-on FFN units.

Each arm is summarised over its seeds (mean, 95 % t-interval); a random-init build of each architecture is the
"architecture alone" reference (its spectra are the orthogonal init's, so a trained matrix is read against it).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from main.probe_battery.bank import refuse_models_output, run_worker
from main.probe_battery.capture import parse_spec
from main.probe_battery.report import mean_ci, welch

#: the verdict's thresholds (descriptive; ``designs/prober/probe_battery.md`` §6)
KL_SMALL = 0.02          # "outputs stay within a small KL" for the used-width read
KL_BIG_LAST = 0.10       # a logit-lens KL above this = the last round still changes the decision substantially


def run_all(*, checkout: Path, expect_commit: str, bank: Path, specs: Sequence[str], out: Path, n_rows: int = 4000,
            threads: int = 4) -> List[str]:
    refuse_models_output(out)
    out.mkdir(parents=True, exist_ok=True)
    done = []
    for spec in specs:
        label, zp, rnd = parse_spec(spec)
        dst = out / f"{label}.json"
        if not dst.exists():
            args = ["depth", "--bank", str(bank), "--ckpt", zp, "--label", label, "--out", str(dst),
                    "--n-rows", str(n_rows), "--threads", str(threads)]
            if rnd is not None:
                args += ["--random-init", str(rnd)]
            run_worker(checkout, expect_commit, args, name="probebat-depth", mem_gb=24)
        done.append(label)
    return done


def _flat(d: Dict[str, Any]) -> Dict[str, float]:
    """One checkpoint's numbers as ``{metric path: value}``."""
    out: Dict[str, float] = {}
    for k, v in d["variants"].items():
        for m, x in v.items():
            out[f"variant/{k}/{m}"] = x
    for k, v in d["rank"].items():
        for m in ("pr", "n90", "n99"):
            out[f"rank/{k}/{m}"] = v[m]
    for k, v in d["update_norm"].items():
        if v is not None:
            out[f"update/{k}"] = v
    for k, v in d["ffn"].items():
        for m in ("dead", "always_on", "mean_active"):
            out[f"ffn/{k}/{m}"] = v[m]
    for k, v in d["spectra"].items():
        for m in ("pr", "n90", "n99"):
            out[f"spectrum/{k}/{m}"] = v[m]
    return out


def summarise(res_dir: Path, arms: Dict[str, List[str]], randoms: Dict[str, List[str]]) -> Dict[str, Any]:
    flat = {}
    for labs in list(arms.values()) + list(randoms.values()):
        for lab in labs:
            flat[lab] = _flat(json.loads((Path(res_dir) / f"{lab}.json").read_text()))
    keys = sorted({k for v in flat.values() for k in v})
    S: Dict[str, Any] = {}
    names = list(arms)
    for k in keys:
        ent: Dict[str, Any] = {}
        for arm, labs in list(arms.items()) + [(f"{a}_random", v) for a, v in randoms.items()]:
            xs = [flat[lab][k] for lab in labs if k in flat[lab]]
            if xs:
                ent[arm] = mean_ci(xs)
                ent[arm]["per_seed"] = [round(float(x), 5) for x in xs]
        if len(names) == 2 and all(n in ent for n in names):
            ent["gap"] = welch(ent[names[0]]["per_seed"], ent[names[1]]["per_seed"])
        S[k] = ent
    return S


def used_rank(S: Dict[str, Any], arm: str, group: str, kl: float = KL_SMALL) -> Any:
    """The smallest truncation rank whose mean policy KL stays below ``kl`` (None: none does)."""
    for r in (16, 32, 48, 64, 96):
        e = S.get(f"variant/trunc_{group}_r{r}/kl", {}).get(arm)
        if e and e["mean"] is not None and e["mean"] < kl:
            return r
    return None


def verdict(S: Dict[str, Any], arms: Sequence[str]) -> Dict[str, Any]:
    """The plain-language read per arm: is width under-used, is depth saturated?"""
    out = {}
    for arm in arms:
        def m(k: str) -> Any:
            return (S.get(k, {}).get(arm) or {}).get("mean")

        rounds = sorted({int(k.split("/")[1][len("lens_L"):]) for k in S if k.startswith("variant/lens_L")
                         and k.endswith("/kl") and (S[k].get(arm) or {}).get("mean") is not None})
        last = f"L{rounds[-1]}" if rounds else "L2"
        heads = [m(k) for k in S if k.startswith("variant/head_off_") and k.endswith("/kl")]
        heads = [x for x in heads if x is not None]
        groups = ("our_mons", "their_mons", "board", "E3", "E4", "events")
        out[arm] = {
            "last_round": last, "lens_last_kl": m(f"variant/lens_{last}/kl"),
            "lens_last_top1": m(f"variant/lens_{last}/top1_agree"),
            "lens_last_value_mae": m(f"variant/lens_{last}/value_mae"), "lens_L1_kl": m("variant/lens_L1/kl"),
            "update_last_mean": _mean([m(f"update/{g}@{last}") for g in ("our_mons", "their_mons", "board", "E3")]),
            "update_L1_mean": _mean([m(f"update/{g}@L1") for g in ("our_mons", "their_mons", "board", "E3")]),
            "pr_last": {g: m(f"rank/{g}@{last}/pr") for g in groups},
            "n99_last": {g: m(f"rank/{g}@{last}/n99") for g in groups},
            "used_rank_trunk": used_rank(S, arm, "trunk"), "used_rank_input": used_rank(S, arm, "input"),
            "ffn_dead": {f"L{i}": m(f"ffn/L{i}/dead") for i in rounds},
            "max_head_kl": max(heads or [0.0]), "min_head_kl": min(heads or [0.0]),
        }
        v = out[arm]
        v["depth_saturated"] = bool(v["lens_last_kl"] is not None and v["lens_last_kl"] > KL_BIG_LAST)
        v["width_underused"] = bool(v["used_rank_trunk"] is not None and v["used_rank_trunk"] <= 64)
    return out


def _mean(xs: Sequence[Any]) -> Any:
    ys = [x for x in xs if x is not None]
    return sum(ys) / len(ys) if ys else None


def write_report(*, res_dir: Path, arms: Dict[str, List[str]], randoms: Dict[str, List[str]], out: Path) -> Dict:
    refuse_models_output(out)
    S = summarise(res_dir, arms, randoms)
    # the verdict is for TRAINED arms only: a fresh build's pointer head is zero-init (a uniform policy), so every
    # policy KL of a random arm is 0 by construction and would read as "nothing matters"
    V = verdict(S, list(arms))
    out.mkdir(parents=True, exist_ok=True)
    (out / "depth_summary.json").write_text(json.dumps({"verdict": V, "metrics": S}, indent=1, sort_keys=True))
    return V
