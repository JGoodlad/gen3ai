"""The MON-TIED out_gain's identity proof (config v145, ``gen3_mon_tied_gain_v1``; 2026-10-07).

At the checkout it runs from (CPU, one thread, no GPU):

    python designs/research_state/measurements/mon_tied_gain_identity_2026-10-07/identity.py [--out result.json]

v145 changes ONE thing in the model: ``damage_op_layout.out_gain_channel_keys`` drops the MON index (our team
slot, their mon) from the op's gain keys, so the incoming per-mon rows' 72 gains become 12 and the Choice-Band
tail's 12 become 2 (production ``damage_op.out_gain`` 99 -> 29). So the v144 model is rebuilt IN THIS PROCESS by
patching ``agents.model.damage_op.out_gain_channel_keys`` with ``V144_KEYS`` (the v144 function, verbatim) — and
(0) that patched build's K9 init hash must equal ``V144_GOLDEN_INIT``, the init hash the v144 code committed to
``learner_golden.json`` (seeds only, no buffer): the patch IS v144, byte for byte, or the script refuses. The forward
and update checks use the K9 buffer committed at the checkout it runs from.

(1) MAP: each v145 gain = the AVERAGE of the v144 per-position gains it ties (over flat positions — every v144
    parameter in a group covers the same number of positions, so this is the per-parameter mean).
(2) FORWARD, "already equal": the v144 learner with each tied group set to its average vs the v145 learner with
    the mapped gains — ``evaluate_actions_functional`` on the K9 buffer's 64 rows, ``torch.equal`` on every output
    (the expanded per-position gain is the same value, so the forward is the same arithmetic).
(3) GRADIENT, same weights: one backward of a fixed loss over the same rows — every non-gain gradient
    ``torch.equal``; each tied gain's gradient vs the SUM of the v144 per-position gains' gradients it ties
    (a reduction in a different order: fp32 rounding).
(4) UPDATE, the gains FROZEN in both (``requires_grad=False``): one K9 ``train()`` seeded as
    ``learner_golden.compute`` seeds it; every parameter ``torch.equal`` and every pinned loss equal. This
    isolates the reparameterisation: everything that is not the gain updates identically.
(5) UPDATE, the gains TRAINABLE: the same, reported (not asserted). It CANNOT be an identity: Adam normalises per
    ELEMENT, so six per-slot gains with six different gradients take six different steps where one tied gain
    takes one, and the global grad-norm clip sees a different gain-gradient norm — the tie changes the update by
    design (that is its purpose).
(6) The K9 golden's own (perturbed, unequal) per-slot gains: how far tying by averaging moves the forward.
(7) TRAINED divergence: the per-team-slot spread of the incoming-row / CB-tail gains in every archived X5 A/B
    checkpoint (pre-break, 138 per-position gains; read RAW from each zip's ``policy.pth`` — they do not load at
    HEAD and do not need to). READ-ONLY on ``models/``.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
FORWARD_KEYS = ("values", "log_prob", "entropy", "masked_logp", "masks_bool")
GAIN = "features_extractor.damage_op.out_gain"
#: The K9 learner golden's init hash as committed by the v144 code (torch 2.8.0+cu126). The init is built from seeds
#: alone (no buffer), so it stays the v144 reference after v145 re-recorded the golden.
V144_GOLDEN_INIT = "b608d0cbdb2dea29104fabcc35e1162e1a546a50684dbbcd077ee39bd88b13dc"


def V144_KEYS(*, outgoing: bool, matrices_outgoing: bool, matrices_incoming_k: int) -> List[tuple]:
    """``damage_op_layout.out_gain_channel_keys`` as of config v144 (main ``25ea2cc6``), verbatim in effect: the
    request-slot / move-seat index dropped, the MON index (our team slot i, their mon d) kept."""
    from agents.model.damage_op_layout import (_DMG_IMX_CELL, _DMG_IMX_HEADER, _DMG_OMX_CELL, _DMG_OUT_N_MOVES,
                                               _DMG_OUT_PER_MOVE, _DMG_PER_MON, _DMG_STATUS_N_MOVES,
                                               _N_OUT_SECONDARY)
    from agents.observation.constants import TEAM_SIZE
    keys: List[tuple] = []
    for i in range(TEAM_SIZE):
        keys += [("incoming_row", i, f) for f in range(_DMG_PER_MON)]
    keys += [("cb_high", i) for i in range(TEAM_SIZE)]
    keys += [("cb_pko", i) for i in range(TEAM_SIZE)]
    keys.append(("p_cb",))
    if outgoing:
        for _k in range(_DMG_OUT_N_MOVES):
            keys += [("out_move", f) for f in range(_DMG_OUT_PER_MOVE)]
        keys.append(("out_p_outspeed",))
        for _k in range(_DMG_OUT_N_MOVES):
            keys += [("out_secondary", c) for c in range(_N_OUT_SECONDARY)]
        keys += [("status_p_land",)] * _DMG_STATUS_N_MOVES
        keys += [("status_known",)] * _DMG_STATUS_N_MOVES
    if matrices_outgoing:
        for _k in range(_DMG_OUT_N_MOVES):
            for d in range(TEAM_SIZE):
                keys += [("omx_cell", d, f) for f in range(_DMG_OMX_CELL)]
        keys += [("omx_revealed", d) for d in range(TEAM_SIZE)]
    if matrices_incoming_k > 0:
        for _k in range(matrices_incoming_k):
            keys += [("imx_header", j) for j in range(_DMG_IMX_HEADER)]
        for i in range(TEAM_SIZE):
            for _k in range(matrices_incoming_k):
                keys += [("imx_cell", i, f) for f in range(_DMG_IMX_CELL)]
    return keys


def _maxdiff(a: Any, b: Any) -> float:
    if a.dtype.is_floating_point:
        return float((a.double() - b.double()).abs().max()) if a.numel() else 0.0
    return float((a != b).sum())


def _build(LG: Any, v144: bool) -> Any:
    import agents.model.damage_op as DO
    real = DO.out_gain_channel_keys
    if v144:
        DO.out_gain_channel_keys = V144_KEYS
    try:
        return LG.build_learner()
    finally:
        DO.out_gain_channel_keys = real


def _rows(LG: Any, model: Any) -> Any:
    import torch as th
    LG.load_buffer_into(model, LG.BUFFER_PATH)
    rb = model.rollout_buffer
    obs = {k: th.as_tensor(v.reshape(-1, *v.shape[2:])) for k, v in rb.observations.items()}
    acts = th.as_tensor(rb.actions.reshape(-1, *rb.actions.shape[2:])).long().flatten()
    masks = th.as_tensor(rb.action_masks.reshape(-1, rb.action_masks.shape[-1]))
    return obs, acts, masks


def _forward(LG: Any, model: Any) -> Dict[str, Any]:
    import torch as th
    obs, acts, masks = _rows(LG, model)
    pol = model.policy
    pol.set_training_mode(True)
    with th.no_grad():
        return dict(zip(FORWARD_KEYS, pol.evaluate_actions_functional(obs, acts, masks)))


def _cmp_forward(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    import torch as th
    return {k: {"equal": bool(th.equal(a[k], b[k])), "max_abs_delta": _maxdiff(a[k], b[k])} for k in FORWARD_KEYS}


def _groups(op144: Any, op145: Any) -> Any:
    """``G [n145, n144]``: G[j, i] = 1 when v144 gain i ties into v145 gain j (read off the two tie buffers)."""
    import torch as th
    t144, t145 = op144._out_gain_tie, op145._out_gain_tie          # [n, out_dim] one-hot columns
    g = (t145 @ t144.t() > 0).float()                               # positions shared
    assert bool((g.sum(0) == 1).all()), "a v144 gain spans two v145 gains: the v145 tie is not a coarsening"
    return g, t144, t145, th


def _set_v144_to_group_mean(op144: Any, mean145: Any, g: Any) -> None:
    import torch as th
    with th.no_grad():
        op144.out_gain.copy_(g.t() @ mean145)


def _map_mean(op144: Any, t145: Any) -> Any:
    e = op144.expanded_out_gain().detach()                          # [out_dim]
    return (t145 @ e) / t145.sum(1)


def _trained_divergence() -> Dict[str, Any]:
    """Per-team-slot spread of the trained incoming-row (72) and CB-tail (12) gains, every archived X5 A/B
    checkpoint with a final_model.zip. READ-ONLY."""
    import torch as th
    from agents.model.damage_op_layout import _DMG_PER_MON
    from agents.observation.constants import TEAM_SIZE
    from utils.paths import main_models_dir
    root = main_models_dir()
    out: Dict[str, Any] = {"root": str(root) if root else None, "runs": {}}
    if root is None:
        out["skipped"] = "no run archive"
        return out
    names = ("phys_low", "phys_high", "phys_crit", "phys_pko", "phys_acc", "spec_low", "spec_high", "spec_crit",
             "spec_pko", "spec_acc", "p_outspeed", "provenance")
    agg_rel: List[float] = []
    for z in sorted(glob.glob(os.path.join(str(root), "rb_x5ab_*", "final_model.zip"))):
        run = os.path.basename(os.path.dirname(z))
        with zipfile.ZipFile(z) as zf:
            sd = th.load(io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=True)
        key = next((k for k in sd if k.endswith("damage_op.out_gain")), None)
        if key is None:
            out["runs"][run] = {"skipped": "no damage_op.out_gain"}
            continue
        gain = sd[key].double()
        if gain.numel() != 138:
            out["runs"][run] = {"skipped": f"out_gain has {gain.numel()} entries, not the 138-wide pre-break layout"}
            continue
        rows = gain[:TEAM_SIZE * _DMG_PER_MON].reshape(TEAM_SIZE, _DMG_PER_MON)       # [slot, channel]
        cb = gain[TEAM_SIZE * _DMG_PER_MON:TEAM_SIZE * _DMG_PER_MON + 2 * TEAM_SIZE].reshape(2, TEAM_SIZE)
        per = {}
        for f, n in enumerate(names):
            col = rows[:, f]
            per[n] = {"min": float(col.min()), "max": float(col.max()), "mean": float(col.mean()),
                      "max_over_min": float(col.max() / col.min()), "per_slot": [round(float(x), 4) for x in col]}
            agg_rel.append(float(col.max() / col.min()))
        for r, n in enumerate(("cb_high", "cb_pko")):
            col = cb[r]
            per[n] = {"min": float(col.min()), "max": float(col.max()), "mean": float(col.mean()),
                      "max_over_min": float(col.max() / col.min()), "per_slot": [round(float(x), 4) for x in col]}
            agg_rel.append(float(col.max() / col.min()))
        worst = max(per.items(), key=lambda kv: kv[1]["max_over_min"])
        out["runs"][run] = {"zip": z, "channels": per, "worst_channel": worst[0],
                            "worst_max_over_min": worst[1]["max_over_min"],
                            "median_max_over_min": float(sorted(v["max_over_min"] for v in per.values())[len(per) // 2])}
    if agg_rel:
        s = sorted(agg_rel)
        out["all_channels_all_runs"] = {"n": len(s), "median_max_over_min": s[len(s) // 2], "max": s[-1]}
    return out


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "result.json"))
    a = ap.parse_args(argv)

    import torch as th
    from agents.training import learner_golden as LG

    res: Dict[str, Any] = {
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
        "torch": th.__version__, "buffer_sha256": LG.buffer_sha256()}
    with LG._one_thread():
        m144 = _build(LG, v144=True)
        m145 = _build(LG, v144=False)
        op144 = m144.policy.features_extractor.damage_op
        op145 = m145.policy.features_extractor.damage_op
        # (0) the patched build IS v144
        res["v144_init_sha256"] = LG.params_sha256(m144)
        res["golden_v144_init_sha256"] = V144_GOLDEN_INIT
        res["v144_build_is_the_recorded_v144"] = res["v144_init_sha256"] == V144_GOLDEN_INIT
        if not res["v144_build_is_the_recorded_v144"]:
            print("REFUSED: the patched v144 build does not reproduce the committed K9 init hash", file=sys.stderr)
            print(json.dumps(res, indent=1))
            return 2
        res["gain_counts"] = {"v144": int(op144.out_gain.numel()), "v145": int(op145.out_gain.numel())}
        res["n_params"] = {"v144": int(sum(p.numel() for p in m144.policy.parameters())),
                           "v145": int(sum(p.numel() for p in m145.policy.parameters()))}
        g, t144, t145, _ = _groups(op144, op145)
        # every NON-gain parameter of the two builds is the same bytes (only the gain's shape moved)
        sd144, sd145 = m144.policy.state_dict(), m145.policy.state_dict()
        res["non_gain_init_equal"] = all(th.equal(sd144[k], sd145[k]) for k in sd145 if not k.endswith("out_gain"))

        # (6) the golden's own perturbed gains, tied by averaging
        fwd144_raw = _forward(LG, m144)
        mean = _map_mean(op144, t145)
        e144 = op144.expanded_out_gain().detach()
        res["golden_perturbed_gain_spread"] = {
            "max_abs_dev_from_group_mean": float((e144 - t145.t() @ mean).abs().max())}
        with th.no_grad():
            op145.out_gain.copy_(mean)
        res["forward_golden_unequal_gains_averaged"] = _cmp_forward(_forward(LG, m145), fwd144_raw)

        # (2) "already equal": set v144's per-slot gains to their group mean, then compare
        _set_v144_to_group_mean(op144, mean, g)
        assert th.equal(op144.expanded_out_gain(), op145.expanded_out_gain())
        f144, f145 = _forward(LG, m144), _forward(LG, m145)
        res["forward_equal_gains"] = _cmp_forward(f145, f144)
        res["forward_equal_gains_bitwise"] = all(v["equal"] for v in res["forward_equal_gains"].values())

        # (3) gradient on the same weights
        grads: Dict[str, Dict[str, Any]] = {}
        for tag, m in (("v144", m144), ("v145", m145)):
            obs, acts, masks = _rows(LG, m)
            m.policy.zero_grad(set_to_none=True)
            values, log_prob, entropy, _, _ = m.policy.evaluate_actions_functional(obs, acts, masks)
            (log_prob.sum() + 0.5 * values.sum() + 0.01 * entropy.sum()).backward()
            grads[tag] = {n: (p.grad.detach().clone() if p.grad is not None else None)
                          for n, p in m.policy.named_parameters()}
            m.policy.zero_grad(set_to_none=True)
        ng = [n for n in grads["v145"] if not n.endswith("out_gain")]
        res["grad_non_gain_bitwise"] = all(
            (grads["v144"][n] is None and grads["v145"][n] is None)
            or (grads["v144"][n] is not None and grads["v145"][n] is not None
                and th.equal(grads["v144"][n], grads["v145"][n])) for n in ng)
        gname = next(n for n in grads["v145"] if n.endswith("damage_op.out_gain"))
        summed = g @ grads["v144"][gname]
        res["grad_tied_vs_sum_of_per_slot"] = {
            "max_abs_delta": _maxdiff(grads["v145"][gname], summed),
            "max_abs": float(summed.abs().max()),
            "max_rel_delta": float(((grads["v145"][gname] - summed).abs() / summed.abs().clamp_min(1e-30)).max())}

        # (4) one K9 update with the gains frozen in both
        posts: Dict[str, Any] = {}
        for tag, m in (("v144", m144), ("v145", m145)):
            m.policy.features_extractor.damage_op.out_gain.requires_grad_(False)
            r = LG.compute(model=m, buffer=LG.BUFFER_PATH)
            posts[tag] = ({k: v.clone() for k, v in m.policy.state_dict().items()}, r["losses"])
        res["update_gain_frozen"] = {
            "params_bitwise": all(th.equal(posts["v144"][0][k], posts["v145"][0][k])
                                  for k in posts["v145"][0] if not k.endswith("out_gain")),
            "losses_equal": posts["v144"][1] == posts["v145"][1],
            "max_abs_delta": max(_maxdiff(posts["v144"][0][k], posts["v145"][0][k])
                                 for k in posts["v145"][0] if not k.endswith("out_gain"))}

    # (5) one K9 update, gains trainable (fresh builds, the same equal-gain mapping)
    with LG._one_thread():
        m144 = _build(LG, v144=True)
        m145 = _build(LG, v144=False)
        op144 = m144.policy.features_extractor.damage_op
        op145 = m145.policy.features_extractor.damage_op
        g, t144, t145, _ = _groups(op144, op145)
        mean = _map_mean(op144, t145)
        with th.no_grad():
            op145.out_gain.copy_(mean)
        _set_v144_to_group_mean(op144, mean, g)
        r144 = LG.compute(model=m144, buffer=LG.BUFFER_PATH)
        r145 = LG.compute(model=m145, buffer=LG.BUFFER_PATH)
        s144, s145 = m144.policy.state_dict(), m145.policy.state_dict()
        by_group: Dict[str, float] = {}
        for k in s145:
            if k.endswith("out_gain") or k.startswith(("pi_features_extractor.", "vf_features_extractor.")):
                continue
            grp = ".".join(k.split(".")[:2])
            by_group[grp] = max(by_group.get(grp, 0.0), _maxdiff(s144[k], s145[k]))
        gain_after_144 = op144.expanded_out_gain().detach()
        gain_after_145 = op145.expanded_out_gain().detach()
        res["update_gain_trainable"] = {
            "params_bitwise": all(v == 0.0 for v in by_group.values()),
            "max_abs_delta_non_gain": max(by_group.values()),
            "top_groups": dict(sorted(by_group.items(), key=lambda kv: -kv[1])[:6]),
            "expanded_gain_max_abs_delta": _maxdiff(gain_after_144, gain_after_145),
            "losses_max_abs_delta": max(abs(r144["losses"][k] - r145["losses"][k]) for k in r145["losses"]),
            "note": "not an identity by construction: Adam normalises per element and the clip sees a different "
                    "gain-gradient norm; reported, never asserted"}

    res["trained_divergence"] = _trained_divergence()
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    short = {k: res[k] for k in ("commit", "v144_build_is_the_recorded_v144", "gain_counts", "n_params",
                                  "non_gain_init_equal", "forward_equal_gains_bitwise", "grad_non_gain_bitwise",
                                  "grad_tied_vs_sum_of_per_slot", "update_gain_frozen")}
    print(json.dumps(short, indent=1))
    print("trainable update:", json.dumps(res["update_gain_trainable"], indent=1))
    print("golden unequal gains averaged:",
          {k: v["max_abs_delta"] for k, v in res["forward_golden_unequal_gains_averaged"].items()})
    td = res["trained_divergence"]
    print("trained divergence:", json.dumps(td.get("all_channels_all_runs"), indent=1))
    for run, v in td.get("runs", {}).items():
        print(f"  {run}: worst {v.get('worst_channel')} max/min {v.get('worst_max_over_min', float('nan')):.3f}, "
              f"median {v.get('median_max_over_min', float('nan')):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
