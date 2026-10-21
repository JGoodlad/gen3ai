"""X50 R1 — the REGISTERED read (design §5): X26's intent head on human opponents vs its own snapshots vs bots.

    python -m main.x50_r1.read --humans <archive>/humans --traces <archive>/traces \
        --checkpoint <main>/models/rb_x26_s2601/checkpoints/checkpoint_380000195_steps.zip --out <dir>

Reads the extractor's parts one at a time (never every row at once), runs ONE forward per row on the CPU
(``main.policy_spectrum.reader.load_checkpoint``; the flat pointer's ``last_flat_intent_logits`` /
``last_flat_intent``), resolves each label to the flat column with ``flat_intent_targets`` (the loss's function), and
keeps per-row numbers only. Populations, metrics, the bootstrap and the decision rule are the registration's,
verbatim; nothing here is a free parameter.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

#: The registration (design §5).
PRIMARY_RATING = 1500
BOOT_N = 2000
BOOT_SEED = 50
MARGIN = 0.05          # nats
BATCH = 512
BOTS = ("heuristic", "heuristic2", "staller", "staller_v2", "aggressive", "aggressive_v2", "setup_sweep",
        "setup_sweep_v2")


def forward_part(model: Any, npz: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Per-row numbers for one part: target, class, NLL, top-1, and the two floors."""
    import torch

    from agents.model.extra_obs_keys import zero_extra_obs
    from agents.model.flat_intent import flat_intent_targets

    pol = model.policy
    fe = pol.features_extractor
    cols: Dict[str, List[np.ndarray]] = {k: [] for k in ("tgt", "cls", "nll", "top1", "nll_uni", "top1_uni",
                                                          "nll_pres", "top1_pres", "n_live")}
    rows, masks, lab = npz["rows"], npz["mask"], npz["label"]
    for i in range(0, len(rows), BATCH):
        ob = {"observation": torch.from_numpy(np.ascontiguousarray(rows[i:i + BATCH], dtype=np.float32)),
              "action_mask": torch.from_numpy(masks[i:i + BATCH].astype(np.float32))}
        ob.update(zero_extra_obs(fe, batch=len(ob["action_mask"]), device="cpu"))
        with torch.no_grad():
            pol.extract_features(ob)
            logits = fe.last_flat_intent_logits.float()
            fi = fe.last_flat_intent
            lb = torch.from_numpy(lab[i:i + BATCH])
            tgt, cls = flat_intent_targets(fi, lb[:, 0], lb[:, 1], lb[:, 2], lb[:, 3])
            live = fi.live
            ok = tgt >= 0
            t = tgt.clamp(min=0)[:, None]
            lp = torch.log_softmax(logits, -1)
            pres = torch.where(live, fi.log_pi.float(), torch.full_like(logits, float("-inf")))
            lpp = torch.log_softmax(pres, -1)
            n_live = live.sum(-1).clamp(min=1)
            nan = torch.full_like(tgt, 0, dtype=torch.float32) * float("nan")
            cols["tgt"].append(tgt.numpy())
            cols["cls"].append(cls.numpy())
            cols["nll"].append(torch.where(ok, -lp.gather(1, t)[:, 0], nan).numpy())
            cols["top1"].append(torch.where(ok, (logits.argmax(-1) == tgt).float(), nan).numpy())
            cols["nll_uni"].append(torch.where(ok, torch.log(n_live.float()), nan).numpy())
            cols["top1_uni"].append(torch.where(ok, 1.0 / n_live.float(), nan).numpy())
            cols["nll_pres"].append(torch.where(ok, -lpp.gather(1, t)[:, 0], nan).numpy())
            cols["top1_pres"].append(torch.where(ok, (pres.argmax(-1) == tgt).float(), nan).numpy())
            cols["n_live"].append(n_live.numpy())
    return {k: np.concatenate(v) for k, v in cols.items()}


def rule(lo: float, hi: float) -> str:
    """The registered decision rule on the 95 % CI of Δ NLL(H − S) (design §5)."""
    if lo > MARGIN:
        return "GAP"
    if lo > -MARGIN and hi < MARGIN:
        return "NO GAP"
    return "INCONCLUSIVE"


def load_parts(d: Path) -> List[Path]:
    return sorted(d.glob("*.json"))


def collect(model: Any, d: Path, kind: str) -> Dict[str, np.ndarray]:
    """Every part of one extraction directory → per-row arrays (metadata + forward numbers)."""
    keep = ("label", "hp", "forced", "placeholder", "turn", "k", "battle", "seat", "mode", "rating_labelled")
    out: Dict[str, List[np.ndarray]] = {}
    opp_of: Dict[int, str] = {}
    counts: Dict[str, int] = {}
    for js in load_parts(d):
        meta = json.loads(js.read_text())
        for k, v in meta["counts"].items():
            counts[k] = counts.get(k, 0) + int(v)
        npz = dict(np.load(js.with_suffix(".npz")))
        if not npz:
            continue
        if kind == "traces":
            # battle ids are global: the part's index x its part size, plus the battle's place in the part
            part_first = int(js.stem.rsplit("_", 1)[1]) * int(meta["part_size"])
            for i, p in enumerate(meta["battles"]):
                opp_of[part_first + i] = Path(p).parent.name
        fw = forward_part(model, npz)
        for k in keep:
            out.setdefault(k, []).append(npz[k])
        for k, v in fw.items():
            out.setdefault(k, []).append(v)
    arr = {k: np.concatenate(v) for k, v in out.items()}
    if kind == "traces":
        arr["opp"] = np.array([opp_of[int(b)] for b in arr["battle"]])
    arr["_counts"] = counts  # type: ignore[assignment]
    return arr


def boot_means(x: np.ndarray, clusters: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """BOOT_N cluster-bootstrap means of ``x`` (resampling whole battles)."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=x, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    out = np.empty(BOOT_N)
    for b in range(BOOT_N):
        pick = rng.integers(0, len(uniq), len(uniq))
        out[b] = sums[pick].sum() / cnts[pick].sum()
    return out


def summarize(a: Dict[str, np.ndarray], sel: np.ndarray, rng: np.random.Generator) -> Dict[str, Any]:
    ok = sel & (a["tgt"] >= 0)
    cl = a["battle"][ok]
    res: Dict[str, Any] = {"rows": int(ok.sum()), "battles": int(len(np.unique(cl)))}
    if not ok.any():
        return res
    for m in ("nll", "top1", "nll_uni", "top1_uni", "nll_pres", "top1_pres"):
        x = a[m][ok].astype(float)
        bm = boot_means(x, cl, rng)
        res[m] = {"mean": float(x.mean()), "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                  "_boot": bm}
    res["share_switch"] = float((a["label"][ok, 0] == 1).mean())
    res["share_placeholder"] = float(a["placeholder"][ok].mean())
    return res


def strip(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not k.startswith("_")}
    return o


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x50_r1.read", description=__doc__.split("\n\n")[0])
    ap.add_argument("--humans", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(argv)
    from main.policy_spectrum.reader import inference_globals, load_checkpoint

    t0 = time.time()
    with inference_globals(a.threads):
        model = load_checkpoint(Path(a.checkpoint))
        H = collect(model, Path(a.humans), "humans")
        T = collect(model, Path(a.traces), "traces")
    rng = np.random.default_rng(BOOT_SEED)
    nohp_h = ~H["hp"]
    nohp_t = ~T["hp"]
    is_sent = np.char.startswith(T["opp"].astype(str), "sentinel_")
    is_bot = np.isin(T["opp"], BOTS)
    spec, full = T["mode"] == 0, T["mode"] == 1
    pops = {
        "H": summarize(H, nohp_h & (H["rating_labelled"] >= PRIMARY_RATING), rng),
        "S": summarize(T, nohp_t & spec & is_sent, rng),
        "B": summarize(T, nohp_t & spec & is_bot, rng),
        "S_full": summarize(T, nohp_t & full & is_sent, rng),
        "B_full": summarize(T, nohp_t & full & is_bot, rng),
        "H_all_rated": summarize(H, nohp_h & (H["rating_labelled"] >= 0), rng),
        "H_hidden_power": summarize(H, H["hp"] & (H["rating_labelled"] >= PRIMARY_RATING), rng),
    }
    for lo, hi in ((0, 1300), (1300, 1500), (1500, 1700), (1700, 10 ** 6)):
        pops[f"H_band_{lo}_{hi if hi < 10 ** 6 else 'up'}"] = summarize(
            H, nohp_h & (H["rating_labelled"] >= lo) & (H["rating_labelled"] < hi), rng)
    for kname, kv in (("move", 0), ("switch", 1)):
        pops[f"H_{kname}"] = summarize(H, nohp_h & (H["rating_labelled"] >= PRIMARY_RATING) & (H["label"][:, 0] == kv), rng)
        pops[f"S_{kname}"] = summarize(T, nohp_t & spec & is_sent & (T["label"][:, 0] == kv), rng)
        pops[f"B_{kname}"] = summarize(T, nohp_t & spec & is_bot & (T["label"][:, 0] == kv), rng)
    pops["H_no_placeholder"] = summarize(H, nohp_h & (H["rating_labelled"] >= PRIMARY_RATING) & ~H["placeholder"], rng)
    pops["S_no_placeholder"] = summarize(T, nohp_t & spec & is_sent & ~T["placeholder"], rng)

    def delta(x: str, y: str, m: str = "nll") -> Dict[str, Any]:
        d = pops[x][m]["_boot"] - pops[y][m]["_boot"]
        return {"mean": pops[x][m]["mean"] - pops[y][m]["mean"],
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}

    deltas = {"H-S nll": delta("H", "S"), "H-S top1": delta("H", "S", "top1"),
              "H-B nll": delta("H", "B"), "H-B top1": delta("H", "B", "top1"),
              "S-S_full nll": delta("S", "S_full"), "B-B_full nll": delta("B", "B_full"),
              "S-S_full top1": delta("S", "S_full", "top1"), "B-B_full top1": delta("B", "B_full", "top1")}
    verdict = rule(*deltas["H-S nll"]["ci95"])
    cls_share = {}
    for name, arr, sel in (("H", H, nohp_h & (H["rating_labelled"] >= PRIMARY_RATING)),
                           ("S", T, nohp_t & spec & is_sent), ("B", T, nohp_t & spec & is_bot)):
        ok = sel & (arr["tgt"] >= 0)
        lab_ok = sel & (arr["label"][:, 0] != 2)
        cls_share[name] = {"supervised_of_labelled": float(ok.sum() / max(lab_ok.sum(), 1)),
                           **{f"cls_{int(c)}": float((arr["cls"][ok] == c).mean()) for c in np.unique(arr["cls"][ok])}}
    out = {"schema": "x50_r1_read_v1", "registration": "designs/endstate/design_x50_rung1_intent_labels.md §5",
           "checkpoint": a.checkpoint, "verdict": verdict, "rule": f"Δ NLL(H − S) CI vs ±{MARGIN} nats",
           "deltas": deltas, "populations": strip(pops), "label_classes": cls_share,
           "class_codes": "flat_intent LABEL_* codes (agents.model.flat_intent)",
           "extraction_counts": {"humans": H["_counts"], "traces": T["_counts"]},
           "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "cluster": "battle"},
           "seconds": round(time.time() - t0, 1)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "r1_read.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"verdict": verdict, "deltas": deltas}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
