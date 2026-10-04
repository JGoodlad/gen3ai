"""The READER (M5 Lane S gate ③): any checkpoint on the whole bank, forward passes only, no games.

1. RE-ENCODE the bank once with this checkout's encoder (:mod:`main.policy_spectrum.replay`) and
   check each re-encoded row that has a recorded reference against its sha256 — a read states
   whether it ran on the recorded observations (``obs_as_recorded``) or on a re-encoding that has
   drifted from them (a new architecture, by design).
2. For each checkpoint: load it for inference on CPU (``load_foreign_opponent``: the
   ``arch_signature`` is verified, never trusted), eval mode, fixed thread count, batched forward
   → the policy's masked probabilities on every banked decision.
3. :func:`main.policy_spectrum.spectrum.read` → the JSON; the per-decision probabilities go beside
   it as ``<label>.probs.npz`` (float32, bank order) for paired comparisons and gate ④.

A checkpoint is always a named ``.zip`` — a bare run directory is REFUSED (it would resolve to the
run's LAST snapshot, which moves).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from main.policy_spectrum import spectrum as S
from main.policy_spectrum.bank import Bank

READ_SCHEMA = "gen3_policy_spectrum_read_v1"
BATCH = 256


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def reencode(bank: Bank, workers: int = 2) -> Tuple[np.ndarray, np.ndarray, dict]:
    """``(rows [N, obs_dim] float32, masks [N, 11] bool, gate)`` in bank decision order."""
    rows, masks, gate, _ = reencode_with_labels(bank, workers=workers)
    return rows, masks, gate


def reencode_with_labels(bank: Bank, workers: int = 2
                         ) -> Tuple[np.ndarray, np.ndarray, dict, List[Optional[dict]]]:
    """:func:`reencode` plus, per banked decision, the trackers' intent label of the viewer's NEXT entry
    (``ReplayDecision.next_label``: what the opponent did at THIS decision; None at the viewer's last
    entry) — the opponent-intent readers' ground truth (``main.belief_roles``)."""
    from main.policy_spectrum.replay import core_events_identity, replay

    results = replay([b.recorded() for b in bank.battles], workers=workers)
    by_key: Dict[Tuple[str, str, int], object] = {}
    for b, res in zip(bank.battles, results):
        if not res.ok:
            raise RuntimeError(f"{b.battle_id}: the core refused the replay: {res.error}")
        for side, ds in (("p1", res.decisions[0]), ("p2", res.decisions[1])):
            for d in ds:
                by_key[(b.battle_id, side, d.n)] = d
    n = len(bank.decisions)
    rows = None
    masks = np.zeros((n, 11), dtype=bool)
    labels: List[Optional[dict]] = []
    checked = equal = 0
    drift: List[str] = []
    for i, row in enumerate(bank.decisions):
        d = by_key.get((row["battle"], row["side"], row["n"]))
        if d is None:
            raise RuntimeError(f"{row['id']}: the re-encoding has no such decision")
        if d.choice != row["played"] or "".join(str(int(x)) for x in d.mask) != row["mask"]:
            raise RuntimeError(f"{row['id']}: the re-encoded decision is not the banked one "
                               f"(played {d.choice!r} vs {row['played']!r}, mask differs?)")
        if rows is None:
            rows = np.zeros((n, d.row.shape[0]), dtype=np.float32)
        rows[i] = d.row
        masks[i] = d.mask.astype(bool)
        labels.append(d.next_label)
        if row["rec_obs_sha256"]:
            checked += 1
            if hashlib.sha256(d.row.tobytes()).hexdigest() == row["rec_obs_sha256"]:
                equal += 1
            elif len(drift) < 5:
                drift.append(row["id"])
    gate = {"recorded_rows_checked": checked, "byte_equal": equal,
            "obs_as_recorded": checked == equal, "first_drifted": drift,
            "encoder": core_events_identity()}
    return rows, masks, gate, labels


def _find_config(zip_path: Path) -> Path:
    for d in list(zip_path.parents)[:4]:
        c = d / "model_config.json"
        if c.exists():
            return c
    raise FileNotFoundError(f"no model_config.json within 3 levels above {zip_path}")


def inference_globals(threads: int, device: str = "cpu"):
    """The torch globals a read runs under, SCOPED: ``threads`` intra-op threads and, on CUDA, fp32
    (TF32 matmuls OFF) so a GPU continuation is the same function as the CPU reads up to float
    reassociation. Everything is restored on exit (``torch_globals``) — wrap the load AND every
    forward that must see these settings; the loader itself sets nothing process-wide."""
    from utils.torch_state_guard import torch_globals

    return torch_globals(num_threads=int(threads),
                         allow_tf32=False if str(device).startswith("cuda") else None)


def load_checkpoint(zip_path: Path, device: str = "cpu"):
    """Inference load onto ``device``. Sets NO torch global: run it, and the forwards that use the
    model, inside :func:`inference_globals` (the thread count, and TF32 off on CUDA)."""
    from agents.model.snapshot import current_model_version, load_foreign_opponent
    from agents.observation.state_encoder import load_mappings

    zip_path = Path(zip_path)
    if zip_path.is_dir() or zip_path.suffix != ".zip":
        raise ValueError(f"{zip_path}: name the checkpoint .zip — a bare run directory resolves to "
                         "the run's LAST snapshot and moves")
    cv = current_model_version(load_mappings())
    model, _ = load_foreign_opponent(str(zip_path), current_version=cv, device=device,
                                     config_path=str(_find_config(zip_path)))
    model.policy.eval()
    return model


def policy_logits(model, rows: np.ndarray, masks: np.ndarray, batch: int = BATCH) -> np.ndarray:
    """Raw policy logits [N, 11] float32 (masking is applied by the caller)."""
    import torch as th

    from agents.model.extra_obs_keys import zero_extra_obs

    dev = next(model.policy.parameters()).device
    out = []
    for i in range(0, len(rows), batch):
        mb = th.tensor(masks[i:i + batch].astype(np.float32), device=dev)
        ob = {"observation": th.tensor(rows[i:i + batch], device=dev), "action_mask": mb}
        ob.update(zero_extra_obs(model.policy.features_extractor, batch=len(mb), device=dev))
        with th.no_grad():
            out.append(model.policy.get_distribution(ob).distribution.logits.cpu().numpy())
    return np.concatenate(out, 0).astype(np.float32)


def recording_agreement(bank: Bank, probs: np.ndarray, zip_path: Path,
                        models_root: Optional[Path]) -> Optional[dict]:
    """When the checkpoint IS the policy that recorded some banked decisions (an eval snapshot),
    the max |Δp| between this read and the recorded logits on them — the reader's teeth."""
    if models_root is None:
        return None
    try:
        rel = str(Path(zip_path).resolve().relative_to(Path(models_root).resolve()))
    except ValueError:
        return None
    ids = {b.battle_id for b in bank.battles
           if b.source.get("banked_policy") == rel and b.banked_side == b.traced_side}
    idx = [i for i, d in enumerate(bank.decisions) if d["battle"] in ids and d["rec_logits"]]
    if not idx:
        return None
    rec = S.masked_probs(np.array([bank.decisions[i]["rec_logits"] for i in idx]),
                         np.array([[c == "1" for c in bank.decisions[i]["mask"]] for i in idx]))
    read = probs[idx]
    flips = rec.argmax(1) != read.argmax(1)

    def _margin(p: np.ndarray) -> np.ndarray:        # top-1 minus top-2 probability, per row
        top2 = np.sort(p, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]

    # A flipped argmax is judged by Lane E's tie rule: its LARGER margin (recorded or read) against a
    # bar — so the caller can tell a near-tie flip from a real disagreement. None when nothing flipped.
    flip_margin = (float(np.maximum(_margin(rec), _margin(read))[flips].max()) if flips.any() else None)
    return {"decisions": len(idx), "max_abs_dp": float(np.abs(rec - read).max()),
            "argmax_agree": float((~flips).mean()), "argmax_flip_max_margin": flip_margin}


def read_checkpoint(bank: Bank, rows: np.ndarray, masks: np.ndarray, gate: dict, zip_path: Path,
                    label: str, out_dir: Path, threads: int = 4, commit: str = "unknown",
                    models_root: Optional[Path] = None, note: Optional[str] = None) -> dict:
    with inference_globals(threads):
        model = load_checkpoint(zip_path)
        logits = policy_logits(model, rows, masks)
    del model
    probs = S.masked_probs(logits, masks)
    body = S.read(probs, bank.decisions)
    result = {
        "schema": READ_SCHEMA,
        "spectrum_schema": S.SPECTRUM_SCHEMA,
        "label": label,
        "note": note,
        "checkpoint": {"path": str(zip_path), "sha256": file_sha256(Path(zip_path))},
        "bank": {"content_sha256": bank.manifest["content_sha256"],
                 "decisions": len(bank.decisions), "battles": len(bank.battles)},
        "reader_commit": commit,
        "threads": threads,
        "reencode": {k: v for k, v in gate.items() if k != "encoder"},
        "encoder_sha256": gate["encoder"]["core_events_sha256"],
        "recording_agreement": recording_agreement(bank, probs, zip_path, models_root),
        **body,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = label.replace("/", "_").replace(" ", "_")
    (out_dir / f"{safe}.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    np.savez_compressed(out_dir / f"{safe}.probs.npz", probs=probs.astype(np.float32),
                        logits=logits, ids_sha256=np.array(_ids_sha(bank)))
    return result


def _ids_sha(bank: Bank) -> str:
    return hashlib.sha256("\n".join(d["id"] for d in bank.decisions).encode()).hexdigest()


def load_probs(out_dir: Path, label: str, bank: Bank) -> np.ndarray:
    safe = label.replace("/", "_").replace(" ", "_")
    z = np.load(out_dir / f"{safe}.probs.npz")
    if str(z["ids_sha256"]) != _ids_sha(bank):
        raise ValueError(f"{label}: probabilities were read on a different bank")
    return z["probs"].astype(np.float64)


def refuse_under_models(path: Path) -> None:
    from utils.paths import main_models_dir

    md = main_models_dir()
    if md is not None:
        rp, mr = Path(path).resolve(), md.resolve()
        if rp == mr or mr in rp.parents:
            raise SystemExit(f"[policy_spectrum] REFUSED: {path} is under models/ (read-only)")


def default_threads() -> int:
    return max(1, min(4, (os.cpu_count() or 2) // 4))
