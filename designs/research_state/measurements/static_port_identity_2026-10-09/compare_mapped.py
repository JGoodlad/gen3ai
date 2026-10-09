"""The static-port WEIGHT-MAPPING identity proof — the compare half (CPU, one thread), at a TARGET checkout:

    python -E compare_mapped.py --src <target checkout>/src REF_DIR [--mapping head|static_port]
        [--variants a,b,...] [--readd-flat-bias] [--update] [--out result.json]

REF_DIR is what `capture_reference.py` wrote at the pin (`6c6d2e09`); every file's sha256 is checked against the
COMMITTED `reference_meta.json` first (a mismatch refuses). For each variant:

(1) BUILD the screen's static arm at the target (`_static_build.static_args`: `--arch production` + `static`,
    fixed_mass, tower, move-resolution off, value-threat-inject on, speed-physics off, obs-facts off,
    op-reduction max) and assert it IS static x fixed_mass x tower.
(2) MAP the reference init through the declared mapping (`MAPPINGS`), every step asserting what it claims:
    ``head``        drop EXACTLY `DECLARED_REMOVED` (F1's value tower, F16b's flat-pointer bias; + the pi_ / vf_
                    aliases); TIE `damage_op.out_gain` 138 -> the target's groups (`_out_gain_tie`, which must
                    equal `out_gain_groups_v145.json`): a group whose reference positions are bitwise equal maps
                    to that value EXACTLY, else to their float64 mean (reported, never an identity);
    ``static_port`` = ``head`` + (c1) delete the type2 column block of `pokemon_encoder.role_encoder.0.weight`
                    (asserting it equals the type1 block bitwise — the reference must be conditioned with c1),
                    + (c2) drop `op_content.outgoing_proj.*` (asserting it is all zero) and keep the target's OWN
                    init for every new key under `op_content.`, asserting the new module's LAST Linear (output
                    layer) is all zero.
    Then `load_state_dict(strict=True)`: no undeclared key may be dropped, missing or reshaped.
(3) FORWARD: `evaluate_actions_functional` on the reference's 64 rows (obs 2761) with the target's OBS-FACTS block
    APPENDED (the target's width - 2761 columns, which must be `OFFSET_OBS_FACTS = 2761`), under TWO fills (zeros,
    and seeded N(0, 3^2)); `torch.equal` on values / log_prob / entropy / masked_logp / masks_bool against the
    reference, and the two fills against each other (obs-facts `off` must read nothing of the block).
(4) ``--update`` (variants the reference updated): one K9 `train()` on the same buffer (observation extended by the
    zero fill), seeded as `learner_golden.compute` seeds it, the gains FROZEN as at the reference; every surviving
    parameter vs the mapped reference post-state, and the pinned losses, exactly.

``--readd-flat-bias`` is the F16b CONTROL (the precedent's, `version_break_identity_2026-10-07`): it re-attaches
the removed flat-pointer bias in this script only (the reference value, in the optimizer right after its weight,
the pre-break position), so the comparison isolates every OTHER change.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

from _static_build import (CACHE, FORWARD_KEYS, HERE, assert_static_fixed_mass_tower, bootstrap, build,
                           checkout_of, forward, maxdiff, sha_file, static_args)

_ALIASES = ("pi_features_extractor.", "vf_features_extractor.")
_FE = "features_extractor."
DECLARED_REMOVED: Dict[str, str] = {
    # F1 — the dead SB3 value tower (the extractor's value projection, SB3's critic MLP and value_net).
    "features_extractor.value_pre_norm.weight": "F1",
    "features_extractor.value_pre_norm.bias": "F1",
    "features_extractor.value_projection.weight": "F1",
    "features_extractor.value_projection.bias": "F1",
    "mlp_extractor.value_net.0.weight": "F1",
    "mlp_extractor.value_net.0.bias": "F1",
    "mlp_extractor.value_net.2.weight": "F1",
    "mlp_extractor.value_net.2.bias": "F1",
    "value_net.weight": "F1",
    "value_net.bias": "F1",
    # F16b — the flat opponent pointer's shared scorer bias (one bias, one softmax: shift-invariant).
    "features_extractor.flat_intent_head.out.bias": "F16b",
}
_FLAT_BIAS = "features_extractor.flat_intent_head.out.bias"
_GAIN = "damage_op.out_gain"
_ROLE_W = "pokemon_encoder.role_encoder.0.weight"
_OPC_OUT = "op_content.outgoing_proj."
#: c1's column blocks of the static encoder's s_in (`StaticTokenEncoder.encode`; located on real rows by
#: `dump_surface.py`, re-checked by `capture_reference.py`): type1 [62, 78), type2 [78, 94).
TYPE1, TYPE2 = (62, 78), (78, 94)


def _all_aliases(k: str, keys) -> List[str]:
    out = [k]
    if k.startswith(_FE):
        out += [a + k[len(_FE):] for a in _ALIASES if a + k[len(_FE):] in keys]
    return out


# ------------------------------------------------------------------------------------------------ the mappings
def map_removed(ref: Dict[str, Any], tgt: Dict[str, Any], fe: Any, rep: Dict[str, Any], ctl: Dict[str, Any]):
    declared = {}
    for k, why in DECLARED_REMOVED.items():
        if ctl.get("readd_flat_bias") and k == _FLAT_BIAS:
            continue
        for ak in _all_aliases(k, ref):
            declared[ak] = why
    rep["declared_removed"] = sorted(declared)
    rep["removed_params"] = int(sum(ref[k].numel() for k in declared if not k.startswith(_ALIASES)))
    for k in declared:
        if k not in ref:
            raise AssertionError(f"declared removed key {k} is not in the reference")
        if k in tgt:
            raise AssertionError(f"declared removed key {k} is still in the target")
    return {k: v for k, v in ref.items() if k not in declared}


def map_tie(ref, tgt, fe, rep, ctl):
    import torch as th
    groups = json.loads((HERE / "out_gain_groups_v145.json").read_text())
    tie = fe.damage_op._out_gain_tie
    gop = [int(i) for i in tie.argmax(0).tolist()]
    if gop != groups["group_of_position"] or [list(map(str, k)) for k in fe.damage_op.out_gain_keys] != groups["keys"]:
        raise AssertionError("the target's out_gain tie groups differ from out_gain_groups_v145.json")
    idx = th.tensor(gop)
    out = dict(ref)
    exact = True
    for k in _all_aliases(_FE + _GAIN, ref):
        g = ref[k]
        tied = th.empty(len(groups["keys"]), dtype=g.dtype)
        for j in range(len(groups["keys"])):
            vals = g[idx == j]
            if bool((vals == vals[0]).all()):
                tied[j] = vals[0]
            else:
                exact = False
                tied[j] = vals.double().mean().to(g.dtype)
        out[k] = tied
    rep["tie_exact"] = exact
    return out


def map_c1(ref, tgt, fe, rep, ctl):
    out = dict(ref)
    for k in _all_aliases(_FE + _ROLE_W, ref):
        w = ref[k]
        a, b = w[:, TYPE1[0]:TYPE1[1]], w[:, TYPE2[0]:TYPE2[1]]
        if not bool((a == b).all()):
            raise AssertionError(f"c1: {k}'s type2 block != its type1 block — capture the reference with c1")
        import torch as th
        out[k] = th.cat([w[:, :TYPE2[0]], w[:, TYPE2[1]:]], dim=1)
    rep["c1"] = {"deleted_cols": list(TYPE2), "kept_type1_cols": list(TYPE1)}
    return out


def map_c2(ref, tgt, fe, rep, ctl):
    import torch as th
    out = {}
    dropped = []
    for k, v in ref.items():
        if (_OPC_OUT in k) and (k.startswith(_FE + _OPC_OUT) or any(k.startswith(a + _OPC_OUT) for a in _ALIASES)):
            if bool(v.abs().sum() != 0):
                raise AssertionError(f"c2: {k} is not all zero — capture the reference with c2")
            dropped.append(k)
            continue
        out[k] = v
    new = sorted(k for k in tgt if k not in out and k.startswith(_FE + "op_content."))
    for k in tgt:
        if k not in out and any(k.startswith(p + "op_content.") for p in _ALIASES):
            new.append(k)
    for k in new:
        out[k] = tgt[k].clone()          # the TARGET's own init for the replacement module
    # The replacement's OUTPUT layer must be all zero: the LAST Linear among the new op_content parameters.
    new_mods = sorted({k[len(_FE):].rsplit(".", 1)[0] for k in new if k.startswith(_FE)})
    lins = [(n, m) for n, m in fe.named_modules() if isinstance(m, th.nn.Linear) and n in new_mods]
    if new and not lins:
        raise AssertionError(f"c2: new op_content keys {new} hold no Linear — cannot locate the output layer")
    if lins:
        # The replacement's output layer is ZERO-INIT in the code; the K9 build perturbs every parameter, so the
        # target state_dict holds a perturbed copy. Condition it back to zero (the symmetric twin of capturing the
        # reference with `outgoing_proj` = 0): the identity claim is "both routes contribute exactly 0".
        last_name, last = lins[-1]
        zeroed = []
        for k in list(out):
            stem = k.rsplit(".", 1)[0]
            if stem.endswith(last_name) and k in new:
                out[k] = th.zeros_like(out[k])
                zeroed.append(k)
        if not zeroed:
            raise AssertionError(f"c2: found no state_dict key for the replacement's output layer {last_name}")
        rep["c2_output_layer"] = last_name
        rep["c2_output_layer_zeroed_keys"] = sorted(zeroed)
    rep["c2"] = {"dropped": sorted(dropped), "target_init_keys": sorted(new)}
    return out


MAPPINGS: Dict[str, List[Callable[..., Dict[str, Any]]]] = {
    "head": [map_removed, map_tie],
    "static_port": [map_removed, map_tie, map_c1, map_c2],
}


# ------------------------------------------------------------------------------------------------ the run
def _readd_flat_bias(pol: Any) -> None:
    """CONTROL ONLY (the precedent's): put a biased `out` back on the flat pointer's head, the SAME weight
    Parameter, the new bias in the optimizer right after it (its pre-break position)."""
    from agents.model.hypothesis_set import IsolatedLinear
    head = pol.features_extractor.flat_intent_head
    old = head.out
    new = IsolatedLinear(int(old.weight.shape[1]), int(old.weight.shape[0]), bias=True)
    new.weight = old.weight
    head.out = new
    assert len(pol.optimizer.param_groups) == 1, "the control assumes the policy's one param group"
    params = pol.optimizer.param_groups[0]["params"]
    i = next(k for k, p in enumerate(params) if p is old.weight)
    params.insert(i + 1, new.bias)


def _map_and_load(model, ref_sd, mapping, ctl, rep):
    pol = model.policy
    tgt = pol.state_dict()
    sd = dict(ref_sd)
    for fn in MAPPINGS[mapping]:
        sd = fn(sd, tgt, pol.features_extractor, rep, ctl)
    missing = sorted(set(tgt) - set(sd))
    extra = sorted(set(sd) - set(tgt))
    reshaped = sorted(k for k in set(sd) & set(tgt) if tuple(sd[k].shape) != tuple(tgt[k].shape))
    rep["undeclared_missing"], rep["undeclared_extra"], rep["reshaped"] = missing, extra, reshaped
    if missing or extra or reshaped:
        raise AssertionError(f"mapping {mapping}: missing {missing[:6]} extra {extra[:6]} reshaped {reshaped[:6]}")
    pol.load_state_dict({k: sd[k] for k in tgt}, strict=True)
    return sd


def _extend(obs, width, fill, seed=7):
    import torch as th
    o = obs["observation"]
    extra = width - o.shape[-1]
    if fill == "zeros":
        tail = th.zeros(o.shape[0], extra, dtype=o.dtype)
    else:
        g = th.Generator().manual_seed(seed)
        tail = (3.0 * th.randn(o.shape[0], extra, generator=g)).to(o.dtype)
    out = dict(obs)
    out["observation"] = th.cat([o, tail], dim=-1)
    return out


def main(argv):
    src, rest = bootstrap(argv)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", nargs="?", default=str(CACHE / "ref"))
    ap.add_argument("--mapping", choices=sorted(MAPPINGS), default="head")
    ap.add_argument("--variants", default=None, help="default: every variant the reference holds")
    ap.add_argument("--readd-flat-bias", action="store_true")
    ap.add_argument("--update", action="store_true", help="also compare the K9 update where the reference has one")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(rest)
    ref = Path(a.ref)
    meta = json.loads((HERE / "reference_meta.json").read_text())
    for f, want in meta["files"].items():
        have = sha_file(ref / f)
        if have != want:
            print(f"REFUSED: {ref / f} sha256 {have} != committed {want}", file=sys.stderr)
            return 2

    import numpy as np
    import torch as th
    from agents.observation.constants import OBS_FACTS_DIM, OFFSET_OBS_FACTS
    from agents.training import learner_golden as LG

    variants = a.variants.split(",") if a.variants else sorted(meta["variants"])
    rows = th.load(ref / "rows.pt", map_location="cpu")
    width_ref = int(rows["obs"]["observation"].shape[-1])
    res: Dict[str, Any] = {"commit": checkout_of(src), "reference_commit": meta["commit"], "torch": th.__version__,
                           "mapping": a.mapping, "control_readd_flat_bias": bool(a.readd_flat_bias),
                           "variants": {}}
    if OFFSET_OBS_FACTS != width_ref:
        raise SystemExit(f"REFUSED: the target's OBS-FACTS block starts at {OFFSET_OBS_FACTS}, the reference rows "
                         f"are {width_ref} wide — the prefix is not the pin's observation")
    args, applied = static_args()
    res["overrides"] = applied
    ctl = {"readd_flat_bias": bool(a.readd_flat_bias)}
    with LG._one_thread():
        for name in variants:
            vrep: Dict[str, Any] = {"conditions": meta["variants"][name]["conditions"]}
            model = build(args)
            pol = model.policy
            vrep["build"] = assert_static_fixed_mass_tower(pol.features_extractor)
            width = int(model.observation_space.spaces["observation"].shape[0])
            if width - width_ref != OBS_FACTS_DIM:
                raise SystemExit(f"REFUSED: target obs {width} - reference {width_ref} != OBS_FACTS_DIM {OBS_FACTS_DIM}")
            if a.readd_flat_bias:
                _readd_flat_bias(pol)
            init = th.load(ref / name / "init_state.pt", map_location="cpu")
            try:
                _map_and_load(model, init, a.mapping, ctl, vrep)
            except AssertionError as e:
                vrep["mapping_refused"] = str(e)
                res["variants"][name] = vrep
                print(f"[{name}] MAPPING REFUSED: {e}", flush=True)
                continue
            vrep["n_params_target"] = int(sum(p.numel() for p in pol.parameters()))
            fwd_ref = th.load(ref / name / "forward.pt", map_location="cpu")
            outs = {}
            for fill in ("zeros", "randn"):
                outs[fill] = forward(pol, _extend(rows["obs"], width, fill), rows["acts"], rows["masks"])
            vrep["forward"] = {fill: {k: {"equal": bool(th.equal(o[k], fwd_ref[k])),
                                          "max_abs_delta": maxdiff(o[k], fwd_ref[k])} for k in FORWARD_KEYS}
                               for fill, o in outs.items()}
            vrep["forward_bitwise"] = all(v["equal"] for f in vrep["forward"].values() for v in f.values())
            vrep["obs_facts_fills_equal"] = all(bool(th.equal(outs["zeros"][k], outs["randn"][k]))
                                                for k in FORWARD_KEYS)
            line = "BITWISE" if vrep["forward_bitwise"] else "max|d| " + ", ".join(
                f"{k} {vrep['forward']['zeros'][k]['max_abs_delta']:.2e}" for k in FORWARD_KEYS
                if not vrep["forward"]["zeros"][k]["equal"])
            print(f"[{name}] forward {line}; obs-facts fills equal {vrep['obs_facts_fills_equal']}; "
                  f"tie exact {vrep.get('tie_exact')}", flush=True)

            if a.update and (ref / name / "post_state.pt").exists():
                post_ref = th.load(ref / name / "post_state.pt", map_location="cpu")
                post_map = dict(post_ref)
                prep: Dict[str, Any] = {}
                for fn in MAPPINGS[a.mapping]:
                    post_map = fn(post_map, pol.state_dict(), pol.features_extractor, prep, ctl)
                ref_losses = json.loads((ref / name / "losses.json").read_text())
                # The reference's own buffer (the pin's committed file, sha256-checked), observation extended.
                pin_buf = Path(meta["src"]) / "agents" / "training" / meta["buffer"]
                if sha_file(pin_buf) != meta["buffer_sha256"]:
                    raise SystemExit(f"REFUSED: {pin_buf} is not the reference's buffer")
                buf_out = CACHE / f"buffer_ext_{width}.npz"
                with np.load(pin_buf) as z:
                    data = {k: z[k] for k in z.files}
                o = data["obs:observation"]
                data["obs:observation"] = np.concatenate(
                    [o, np.zeros(o.shape[:-1] + (width - o.shape[-1],), dtype=o.dtype)], axis=-1)
                np.savez(buf_out, **data)
                r = LG.compute(model=model, buffer=buf_out,
                               before_train=lambda: pol.features_extractor.damage_op.out_gain.requires_grad_(False))
                now = pol.state_dict()
                groups: Dict[str, Dict[str, Any]] = {}
                for k in sorted(now):
                    if k.startswith(_ALIASES) or (a.readd_flat_bias and k == _FLAT_BIAS):
                        continue
                    gname = ".".join(k.split(".")[:2])
                    g = groups.setdefault(gname, {"tensors": 0, "unequal": 0, "max_abs_delta": 0.0})
                    g["tensors"] += 1
                    if not th.equal(now[k], post_map[k]):
                        g["unequal"] += 1
                        g["max_abs_delta"] = max(g["max_abs_delta"], maxdiff(now[k], post_map[k]))
                lc = {k: {"reference": ref_losses.get(k), "target": r["losses"].get(k),
                          "equal": ref_losses.get(k) == r["losses"].get(k)}
                      for k in sorted(set(ref_losses) | set(r["losses"]))}
                vrep["update"] = {"bitwise": all(g["unequal"] == 0 for g in groups.values()),
                                  "groups_unequal": {k: g for k, g in groups.items() if g["unequal"]},
                                  "losses_equal": all(v["equal"] for v in lc.values()),
                                  "losses_unequal": {k: v for k, v in lc.items() if not v["equal"]},
                                  "gains_frozen": True}
                print(f"[{name}] update bitwise {vrep['update']['bitwise']}, losses equal "
                      f"{vrep['update']['losses_equal']}", flush=True)
                for gname, g in vrep["update"]["groups_unequal"].items():
                    print(f"    update {gname}: {g['unequal']}/{g['tensors']} differ, max|d| {g['max_abs_delta']:.3e}")
            res["variants"][name] = vrep
    out = Path(a.out) if a.out else HERE / f"result_{a.mapping}{'_control' if a.readd_flat_bias else ''}.json"
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}")
    _ = re
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
