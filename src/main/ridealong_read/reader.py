"""The READER: one checkpoint's ride-along heads on the whole bank, forward passes only, CPU.

1. The bank is re-encoded ONCE per invocation (``main.policy_spectrum.reader.reencode``, its gate ①
   stamped on every read).
2. Per checkpoint: an inference load (``main.policy_spectrum.reader.load_checkpoint`` — a named
   ``.zip`` only; the critic must be a WIN-PROB V ∈ [0, 1], ``qhat.check_winprob``), then one batched
   extractor forward per minibatch that yields V, the policy's logits and the stashes the heads read
   (``value_pooled``, the pointer inputs, α's logits and seat moves). The heads' batch is built through
   ``RideAlongBatch.detached`` exactly as the learner builds it.
3. HEADS PROVENANCE: a checkpoint that CARRIES trained heads (``policy.ridealong`` is not None, loaded
   from its state_dict) is read through them (``heads: "trained"``). Otherwise FRESH heads are attached
   with the baseline spec (ensemble 5, RND + every declared RND VARIANT, A 5, B 5; ``build_ridealong``
   on a private RNG) and the read says ``heads: "fresh-untrained"``: the ensemble's disagreement and A
   are their randomized PRIORS only, and the RND observation statistics, the ``feat`` variant's
   feature statistics (fitted on the TRAIN rows' ``value_pooled`` after the forward) and every error
   z-score are FITTED ON THE BANK's TRAIN split — every such number is a plumbing check, not a
   measurement. A pre-v127 checkpoint carries no variants, so only fresh heads read them there.
4. The meters (:mod:`main.ridealong_read.meters`, which score every RND variant's z too), the RND
   variants' pre-registered comparisons (:mod:`main.ridealong_read.variants` /
   :mod:`main.ridealong_read.variants_probe`) and the per-checkpoint half of the RND input-choice
   measurement (:mod:`main.ridealong_read.rnd_choice`) → one JSON per checkpoint; the per-row arrays
   (each variant flattened as ``rndv_<name>_err`` / ``rndv_<name>_z``) go to the archive directory
   as ``<label>.rows.npz`` (never committed).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from agents.model.ridealong_heads import RND_VARIANTS
from main.ridealong_read.boot import METHOD as BOOT_METHOD

READ_SCHEMA = "gen3_ridealong_read_v1"
BATCH = 512
#: The heads attached to a checkpoint trained without them — the baseline GPU run's spec, with every
#: declared RND variant beside base (`agents.model.ridealong_heads.RND_VARIANTS`, canonical order).
BASELINE_SPEC_KW: Dict[str, Any] = {"ensemble": 5, "rnd": True, "adv": 5, "opp": 5,
                                    "rnd_variants": RND_VARIANTS}
#: The Lane S ground truth (truth_v2): continuation label -> (rows file, the checkpoint that IS that
#: continuation's greedy policy, relative to ``models/``).
TRUTH_CONTINUATIONS: Dict[str, Tuple[str, str]] = {
    "K2final": ("rows_K2final_S64.compact.jsonl.gz", "ai_v14_07_g0p_k2/final_model.zip"),
    "N0final75M": ("rows_N0final75M_S64.compact.jsonl.gz", "ai_v14_01_base/final_model.zip"),
    "C_fixfinal": ("rows_C_fixfinal_S64.compact.jsonl.gz", "ai_v14_06_lbat_ctrl_fix/final_model.zip"),
}


def bank_dir() -> Path:
    from utils.paths import repo_path

    return repo_path("designs", "research_state", "measurements", "m5_laneS", "bank_v1")


def truth_dir() -> Path:
    from utils.paths import repo_path

    return repo_path("designs", "research_state", "measurements", "m5_laneS", "truth_v2")


def check_checkpoint_arg(path: Path) -> None:
    """REFUSE a bare run directory (it resolves to the run's LAST snapshot, which moves)."""
    p = Path(path)
    if p.is_dir() or p.suffix != ".zip":
        raise SystemExit(f"[ridealong_read] REFUSED: {path} — name the checkpoint .zip; a bare run "
                         "directory resolves to the run's LAST snapshot and moves")
    if not p.exists():
        raise SystemExit(f"[ridealong_read] {path}: no such checkpoint")


def limit_bank(bank: Any, n_battles: Optional[int]) -> Any:
    """The first battles of each source cycle, round-robin across sources, until ``n_battles`` —
    so a small read still spans the train / held-out split. ``None`` = the whole bank."""
    from main.policy_spectrum.bank import Bank

    if n_battles is None or n_battles >= len(bank.battles):
        return bank
    by_src: Dict[str, List[str]] = {}
    for b in bank.battles:
        by_src.setdefault(b.source["label"], []).append(b.battle_id)
    keep: List[str] = []
    k = 0
    while len(keep) < n_battles:
        added = False
        for ids in by_src.values():
            if k < len(ids) and len(keep) < n_battles:
                keep.append(ids[k])
                added = True
        if not added:
            break
        k += 1
    ks = set(keep)
    return Bank(bank.manifest, [b for b in bank.battles if b.battle_id in ks],
                [d for d in bank.decisions if d["battle"] in ks])


# ---------------------------------------------------------------------------------------------
# the forward
# ---------------------------------------------------------------------------------------------

@dataclass
class Columns:
    """Per-row outputs, bank order."""
    v: np.ndarray            # [N] V = P(win)
    logits: np.ndarray       # [N, 11] raw policy logits
    pi: np.ndarray           # [N, 11] masked probabilities
    pooled: np.ndarray       # [N, D_MODEL] value_pooled (feature-RND's input)
    ens_p: np.ndarray        # [N]
    ens_std: np.ndarray      # [N] std of the members' PROBABILITIES (the heads' disagreement)
    ens_logit_std: np.ndarray  # [N] std of the members' LOGITS (no sigmoid compression)
    rnd_err: np.ndarray      # [N]
    rnd_z: np.ndarray        # [N]
    adv_mean: np.ndarray     # [N, 11]
    adv_std: np.ndarray      # [N, 11] member spread of A CENTRED under π (the heads' readout)
    adv_raw_std: np.ndarray  # [N, 11] member spread of the UNCENTRED A (no π-dependent term)
    opp_mean: np.ndarray     # [N, S+1] (blob: α's support) | [N, K+8] (fixed_mass: the flat pointer)
    #: Per RND VARIANT present: ``rndv_<name>_err`` / ``rndv_<name>_z`` [N] (empty: no variants).
    rndv: Dict[str, np.ndarray] = field(default_factory=dict)

    def variant_names(self) -> List[str]:
        return [k[len("rndv_"):-len("_err")] for k in self.rndv if k.endswith("_err")]

    def novelty_keys(self) -> List[str]:
        """``rnd`` (base, when the heads carry it) then ``rndv_<name>`` per variant."""
        base = ["rnd"] if np.isfinite(self.rnd_err).any() else []
        return base + [f"rndv_{n}" for n in self.variant_names()]

    def err(self, key: str) -> np.ndarray:
        return self.rnd_err if key == "rnd" else self.rndv[f"{key}_err"]

    def z(self, key: str) -> np.ndarray:
        return self.rnd_z if key == "rnd" else self.rndv[f"{key}_z"]

    def arrays(self) -> Dict[str, np.ndarray]:
        """Every column FLAT (the npz archive's names): the array fields plus ``rndv_<name>_err`` /
        ``rndv_<name>_z`` per variant."""
        return {**{k: getattr(self, k) for k in ARRAY_FIELDS}, **self.rndv}


#: The Columns fields that are one per-row array each (``rndv`` is the per-variant mapping).
ARRAY_FIELDS: Tuple[str, ...] = tuple(k for k in Columns.__dataclass_fields__ if k != "rndv")


def attach_heads(policy: Any, obs_dim: int, rows: np.ndarray,
                 fit_mask: Optional[np.ndarray] = None,
                 force_fresh: bool = False) -> Tuple[Any, dict]:
    """``(heads, provenance)`` — the checkpoint's own trained heads, or fresh baseline-spec heads with
    the RND observation statistics fitted on ``rows[fit_mask]`` (the TRAIN split; all rows when the
    mask is None or empty). ``force_fresh`` ignores a checkpoint's own heads: the pre-registered
    FLOOR is the same reader, on the same checkpoint, with fresh (prior-only) heads."""
    import torch

    from agents.model.ridealong_heads import RideAlongSpec, build_ridealong

    own = None if force_fresh else getattr(policy, "ridealong", None)
    if own is not None:
        spec = own.spec
        rnd = getattr(own, "rnd", None)
        prov = {"heads": "trained", "spec": dict(ensemble=spec.ensemble, rnd=spec.rnd, adv=spec.adv,
                                                 opp=spec.opp,
                                                 rnd_variants=list(getattr(spec, "rnd_variants", ()))),
                "rnd_variant_decls": _variant_decls(getattr(spec, "rnd_variants", ())),
                "rnd_obs_count": float(rnd.obs_count.item()) if rnd is not None else None,
                "rnd_stats": "the checkpoint's own running statistics"}
        own.eval()
        return own, prov
    # X5 U4: B is over the FLAT pointer's columns (the only B since the X5 version break).
    spec = RideAlongSpec(**BASELINE_SPEC_KW,
                         opp_flat_k=RideAlongSpec.from_extractor(policy.features_extractor).opp_flat_k)
    heads = build_ridealong(policy.features_extractor, obs_dim=obs_dim, spec=spec)
    if heads is None:
        raise RuntimeError("build_ridealong returned None for a non-empty spec")
    heads.eval()
    use_mask = fit_mask is not None and bool(np.asarray(fit_mask).any())
    fit_rows = rows[fit_mask] if use_mask else rows
    if heads.rnd is not None:
        heads.rnd.update_obs_stats(torch.from_numpy(np.ascontiguousarray(fit_rows, dtype=np.float32)))
    return heads, {"heads": "fresh-untrained",
                   "spec": {**BASELINE_SPEC_KW, "rnd_variants": list(spec.rnd_variants)},
                   "rnd_variant_decls": _variant_decls(spec.rnd_variants),
                   "rnd_obs_count": float(len(fit_rows)),
                   "rnd_stats": ("observation statistics and the error z-score FITTED ON THE TRAIN "
                                 "SPLIT's rows (rnd_choice.TRAIN_SOURCES), then every row scored; "
                                 "the feat variant's FEATURE statistics fitted on the TRAIN rows' "
                                 "value_pooled after the forward (feat re-scored), and every "
                                 "variant's error z-score on the TRAIN rows"
                                 if use_mask else
                                 "observation statistics, feat's feature statistics and every "
                                 "error z-score fitted on ALL read rows (the read has no "
                                 "TRAIN-split rows)"),
                   "caveat": "untrained heads: ensemble disagreement and A are randomized priors "
                             "only; RND novelty is an untrained predictor's error"}


def _variant_decls(names: Sequence[str]) -> Dict[str, dict]:
    """Each present variant's declaration (lr_mult, half-life, predictor_vs_target, why)."""
    from agents.model.ridealong_heads import RND_VARIANT_BY_NAME

    out = {}
    for n in names:
        d = RND_VARIANT_BY_NAME[n]
        out[n] = {"input": d.input, "lr_mult": d.lr_mult,
                  "decay_half_life_updates": d.decay_half_life_updates,
                  "predictor_vs_target": d.predictor_vs_target, "why": d.why}
    return out


def forward_columns(model: Any, heads: Any, rows: np.ndarray, masks: np.ndarray, fit_err_stats: bool,
                    batch: int = BATCH, fit_mask: Optional[np.ndarray] = None) -> Columns:
    import torch

    from agents.model.extra_obs_keys import zero_extra_obs
    from agents.model.ridealong_heads import RideAlongBatch
    from main.policy_spectrum.spectrum import masked_probs

    pol = model.policy
    fe = pol.features_extractor
    cols: Dict[str, List[np.ndarray]] = {k: [] for k in ARRAY_FIELDS}
    rndv: Dict[str, List[np.ndarray]] = {}
    for i in range(0, len(rows), batch):
        mk = masks[i:i + batch]
        mb = torch.tensor(mk.astype(np.float32))
        ob = {"observation": torch.from_numpy(np.ascontiguousarray(rows[i:i + batch])),
              "action_mask": mb}
        ob.update(zero_extra_obs(fe, batch=len(mb), device="cpu"))
        with torch.no_grad():
            pi_f, vf_f = pol.extract_features(ob)
            lat_pi = pol.mlp_extractor.forward_actor(pi_f)
            lat_vf = pol.mlp_extractor.forward_critic(vf_f)
            v = pol._critic_value(lat_vf).reshape(-1)
            logits = pol._get_action_dist_from_latent(lat_pi).distribution.logits
            pi = torch.from_numpy(masked_probs(logits.numpy(), mk).astype(np.float32))
            b = RideAlongBatch.detached(
                obs=ob["observation"], pooled=fe.last_value_pooled,
                pointer=tuple(fe.last_pointer_inputs), pi=pi, logits=logits,
                legal=torch.from_numpy(mk.astype(bool)), values=v,
                flat_logits=getattr(fe, "last_flat_intent_logits", None),
                flat_ids=(fe.last_flat_intent.cand_ids
                          if getattr(fe, "last_flat_intent", None) is not None else None))
            out = heads.readout(b)
            adv_mod = getattr(heads, "adv", None)
            raw_sd = (adv_mod(b.pooled, b.pointer).std(1, unbiased=False)
                      if adv_mod is not None else None)
        n = len(mk)
        cols["v"].append(v.numpy())
        cols["logits"].append(logits.numpy())
        cols["pi"].append(pi.numpy())
        cols["pooled"].append(b.pooled.numpy())
        nan = np.full(n, np.nan, dtype=np.float32)
        cols["ens_p"].append(out["ens_p"].numpy() if "ens_p" in out else nan)
        cols["ens_std"].append(out["ens_std"].numpy() if "ens_std" in out else nan)
        cols["ens_logit_std"].append(out["ens_logits"].std(-1, unbiased=False).numpy()
                                     if "ens_logits" in out else nan)
        cols["rnd_err"].append(out["rnd_err"].numpy() if "rnd_err" in out else nan)
        cols["rnd_z"].append(out["rnd_z"].numpy() if "rnd_z" in out else nan)
        nan11 = np.full((n, 11), np.nan, dtype=np.float32)
        cols["adv_mean"].append(out["adv_mean"].numpy() if "adv_mean" in out else nan11)
        cols["adv_std"].append(out["adv_std"].numpy() if "adv_std" in out else nan11)
        cols["adv_raw_std"].append(raw_sd.numpy() if raw_sd is not None else nan11)
        cols["opp_mean"].append(out["opp_mean"].numpy() if "opp_mean" in out
                                else np.full((n, 1), np.nan, dtype=np.float32))
        for k, t in out.items():
            if k.startswith("rndv_"):
                rndv.setdefault(k, []).append(t.numpy())
    c = Columns(**{k: np.concatenate(v, 0) for k, v in cols.items()},
                rndv={k: np.concatenate(v, 0) for k, v in rndv.items()})
    rnd = getattr(heads, "rnd", None)
    if fit_err_stats and rnd is not None:
        from main.ridealong_read.variants_probe import fit_fresh_stats

        fm = fit_mask if fit_mask is not None and bool(np.asarray(fit_mask).any()) else \
            np.ones(len(c.rnd_err), bool)
        rnd.update_err_stats(torch.from_numpy(c.rnd_err[fm]))
        c.rnd_z = rnd.zscore(torch.from_numpy(c.rnd_err)).numpy()
        fit_fresh_stats(heads, c, fm)
    return c


# ---------------------------------------------------------------------------------------------
# one checkpoint's read
# ---------------------------------------------------------------------------------------------

def own_continuation(zip_path: Path, models_root: Optional[Path]) -> Optional[str]:
    """The truth continuation this checkpoint IS (its greedy policy played the truth's playouts)."""
    if models_root is None:
        return None
    try:
        rel = str(Path(zip_path).resolve().relative_to(Path(models_root).resolve()))
    except ValueError:
        return None
    for label, (_, owner) in TRUTH_CONTINUATIONS.items():
        if rel == owner:
            return label
    return None


def load_truth(tdir: Path) -> Dict[str, List[dict]]:
    from main.policy_spectrum.truth import load_rows

    out = {}
    for label, (fname, _) in TRUTH_CONTINUATIONS.items():
        p = Path(tdir) / fname
        if p.exists():
            out[label] = load_rows(p)
    return out


def uncertainty_scores(c: Columns) -> Dict[str, np.ndarray]:
    from main.ridealong_read.meters import binary_entropy

    return {"ens_std": c.ens_std.astype(np.float64),
            "ens_logit_std": c.ens_logit_std.astype(np.float64),
            "rnd_z": c.rnd_z.astype(np.float64),
            **{f"rndv_{n}_z": c.rndv[f"rndv_{n}_z"].astype(np.float64) for n in c.variant_names()},
            "ref_v_entropy": binary_entropy(c.v)}


def bank_meters(bank: Any, c: Columns, masks: np.ndarray) -> dict:
    """Meter (i) + the bank-level half of (iii)."""
    from main.ridealong_read.meters import (
        DRAW_POLICY,
        bank_starved_spread,
        outcome_target,
        score_vs_error,
        stratified_uncertainty,
    )

    dec = bank.decisions
    battles = np.array([d["battle"] for d in dec])
    z, keep = outcome_target([d["outcome"] for d in dec], DRAW_POLICY)
    err = np.abs(c.v.astype(np.float64) - z)
    scores = uncertainty_scores(c)
    strata = {"opp_class": [d["opp_class"] for d in dec], "phase": [d["phase"] for d in dec]}
    sel = {k: np.asarray(v)[keep] for k, v in strata.items()}
    unc = stratified_uncertainty({k: s[keep] for k, s in scores.items()}, err[keep],
                                 battles[keep], sel, ref=scores["ref_v_entropy"][keep])
    zl, _ = outcome_target([d["outcome"] for d in dec], "loss")
    errl = np.abs(c.v.astype(np.float64) - zl)
    unc["draws_as_loss"] = {k: score_vs_error(s, errl, battles, boot=False) for k, s in scores.items()}
    unc["rows"] = {"total": len(dec), "used": int(keep.sum()), "draws_excluded": int((~keep).sum())}
    v = c.v.astype(np.float64)
    return {"i_uncertainty_vs_v_error": unc,
            "v_summary": {"mean_v": round(float(v[keep].mean()), 4),
                          "win_rate": round(float(z[keep].mean()), 4),
                          "brier": round(float(((v - z) ** 2)[keep].mean()), 4)},
            "ensemble_summary": {"mean_disagreement": round(float(np.nanmean(c.ens_std)), 6),
                                 "brier_member_mean": round(float(((c.ens_p - z) ** 2)[keep].mean()), 4)},
            "iii_bank_adv_std_starved_vs_fed": bank_starved_spread(c.adv_std, c.pi, masks),
            "iii_bank_adv_raw_std_starved_vs_fed": bank_starved_spread(c.adv_raw_std, c.pi, masks),
            "opp_plumbing": {"rows": int(np.isfinite(c.opp_mean).all(1).sum()),
                             "mean_abs_opp_mean": round(float(np.nanmean(np.abs(c.opp_mean))), 6)}}


def truth_meters(bank: Any, c: Columns, truth: Mapping[str, List[dict]], own: Optional[str]) -> dict:
    """Meters (ii), (iii) and (iv) against every available continuation's truth; (iv) only against
    the checkpoint's OWN continuation (the truth's state value is its greedy root action's value)."""
    from main.policy_spectrum.qhat_report import Boot
    from main.ridealong_read.meters import (
        adv_std_flags,
        paired_spearman_diff,
        truth_turns,
        truth_value_error,
        within_turn_reads,
    )

    id_to_row = {d["id"]: i for i, d in enumerate(bank.decisions)}
    battle_of = {d["id"]: d["battle"] for d in bank.decisions}
    out: dict = {}
    for label, rows in truth.items():
        turns = truth_turns(rows, id_to_row, battle_of)
        if not turns:
            out[label] = {"turns": 0}
            continue
        B = Boot([T.battle for T in turns])
        blk = {"own_continuation": label == own,
               "ii_A_mean": within_turn_reads(turns, c.adv_mean, c.pi, B),
               "ii_policy_logits_reference": within_turn_reads(turns, c.logits, c.pi, B),
               "ii_spearman_A_minus_logits": paired_spearman_diff(turns, c.adv_mean, c.logits, B),
               "iii_adv_std_on_starved_near_best": adv_std_flags(turns, c.adv_std, c.pi, B),
               "iii_adv_raw_std_on_starved_near_best": adv_std_flags(turns, c.adv_raw_std, c.pi, B)}
        if label == own:
            blk["iv_uncertainty_vs_truth_v_error"] = truth_value_error(
                turns, c.v, c.logits, uncertainty_scores(c))
        out[label] = blk
    if own is None:
        out["iv_note"] = ("no own-continuation truth for this checkpoint: the truth rows carry "
                          "per-action values only, and a state value needs the continuation policy's "
                          "greedy root action — (iv) is not read")
    return out


def file_sha256(path: Path) -> str:
    from main.policy_spectrum.reader import file_sha256 as _sha

    return _sha(Path(path))


def read_checkpoint(bank: Any, rows: np.ndarray, masks: np.ndarray, gate: dict, zip_path: Path,
                    label: str, *, threads: int, commit: str, truth: Mapping[str, List[dict]],
                    models_root: Optional[Path], rnd_obs: Optional[dict],
                    epochs_read: Sequence[int], archive: Optional[Path],
                    fresh_heads: bool = False) -> Tuple[dict, Columns, Any]:
    """One checkpoint → its JSON body, its columns, and the RND variants' ``variants.Carry`` (what
    the cross-checkpoint reads — saturation ratios, feat's drift, the identification series — need)."""
    from main.policy_spectrum.qhat import check_winprob
    from main.policy_spectrum.reader import inference_globals, load_checkpoint
    from main.ridealong_read import rnd_choice as R
    from main.ridealong_read import variants_probe as VP

    with inference_globals(threads):
        model = load_checkpoint(zip_path)
        critic = check_winprob(model)
        sp = R.split(bank.decisions)
        heads, prov = attach_heads(model.policy, rows.shape[1], rows, fit_mask=sp["train"],
                                   force_fresh=fresh_heads)
        cols = forward_columns(model, heads, rows, masks, fit_mask=sp["train"],
                               fit_err_stats=prov["heads"] == "fresh-untrained")
        own = own_continuation(zip_path, models_root)
        body = bank_meters(bank, cols, masks)
        body["truth"] = truth_meters(bank, cols, truth, own)
        battles = np.array([d["battle"] for d in bank.decisions])
        feat = None
        if sp["train"].any() and sp["heldout"].any():
            res = R.train_and_read(cols.pooled[sp["train"]],
                                   lambda rnd: R.novelty_reads(R.rnd_errors(rnd, cols.pooled), sp, battles),
                                   epochs_read=epochs_read)
            res.pop("_rnd")
            feat = res
        def pooled_fn(r: np.ndarray, m: np.ndarray, _mdl: Any = model) -> np.ndarray:
            return pooled_from_model(_mdl, r, m)

        rv, carry = VP.read_variants(
            heads, model.policy.features_extractor, bank.decisions, cols, rows, masks, label,
            str(zip_path), prov["heads"], pooled_fn)
    del model
    result = {
        "schema": READ_SCHEMA,
        "label": label,
        **prov,
        "critic": critic,
        "checkpoint": {"path": str(zip_path), "sha256": file_sha256(zip_path)},
        "own_truth_continuation": own,
        "bank": {"content_sha256": bank.manifest["content_sha256"], "decisions": len(bank.decisions),
                 "battles": len(bank.battles)},
        "reader_commit": commit,
        "threads": threads,
        "reencode": {k: v for k, v in gate.items() if k != "encoder"},
        "encoder_sha256": gate["encoder"]["core_events_sha256"],
        "bootstrap": BOOT_METHOD,
        "draw_policy": "draws EXCLUDED from |V - z| (win 1, loss 0); draws_as_loss is the sensitivity row",
        **body,
        "rnd_variants": rv,
        "v_rnd_input_choice": {"obs_rnd": rnd_obs, "feature_rnd_self": feat,
                               "note": "the drift half (feature-RND trained on checkpoint A, scored "
                                       "through a later B) is in rnd_input_choice.json"},
    }
    if archive is not None:
        archive.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(archive / f"{safe_label(label)}.rows.npz",
                            ids_sha256=np.array(ids_sha(bank)), **_npz_arrays(cols))
    return result, cols, carry


def _npz_arrays(cols: Columns) -> Dict[str, Any]:
    return dict(cols.arrays())


def safe_label(label: str) -> str:
    return label.replace("/", "_").replace(" ", "_").replace("'", "")


def ids_sha(bank: Any) -> str:
    return hashlib.sha256("\n".join(d["id"] for d in bank.decisions).encode()).hexdigest()


def pooled_only(zip_path: Path, rows: np.ndarray, masks: np.ndarray, threads: int,
                batch: int = BATCH) -> np.ndarray:
    """``value_pooled`` [N, D_MODEL] of a checkpoint (the drift read's B side)."""
    from main.policy_spectrum.reader import inference_globals, load_checkpoint

    with inference_globals(threads):
        model = load_checkpoint(zip_path)
        out = pooled_from_model(model, rows, masks, batch)
    del model
    return out


def pooled_from_model(model: Any, rows: np.ndarray, masks: np.ndarray,
                      batch: int = BATCH) -> np.ndarray:
    """``value_pooled`` [N, D_MODEL] of a LOADED model (run inside ``inference_globals``)."""
    import torch

    from agents.model.extra_obs_keys import zero_extra_obs

    pol = model.policy
    fe = pol.features_extractor
    out = []
    for i in range(0, len(rows), batch):
        mb = torch.tensor(np.asarray(masks[i:i + batch]).astype(np.float32))
        ob = {"observation": torch.from_numpy(np.ascontiguousarray(rows[i:i + batch], dtype=np.float32)),
              "action_mask": mb}
        ob.update(zero_extra_obs(fe, batch=len(mb), device="cpu"))
        with torch.no_grad():
            pol.extract_features(ob)
            out.append(fe.last_value_pooled.numpy().copy())
    return np.concatenate(out, 0)


def write_json(out_dir: Path, name: str, obj: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    return p
