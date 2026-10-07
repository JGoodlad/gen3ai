"""The version break's WEIGHT-MAPPING IDENTITY proof (part 2, the EXACT-refactor bundle).

At the checkout it runs from (CPU, one thread):

    python designs/research_state/measurements/version_break_identity_2026-10-07/compare_mapped.py \
        [REF_DIR] [--out result.json]

REF_DIR (default ``~/.cache/gen3ai/vb/ref``) holds the PRE-BREAK reference ``capture_reference.py`` wrote at
``26131c0c`` (the K9 learner golden's ``fixed_mass`` arm = the production surface). Each file's sha256 is
checked against the COMMITTED ``reference_meta.json`` first; a mismatch refuses.

(a) builds the K9 learner at this checkout (``learner_golden.build_learner()``; its buffer's sha256 must be
    the reference's);
(b) MAPS the reference init weights: drops EXACTLY ``DECLARED_REMOVED`` (each key named, with the finding
    that removed it) — asserting the dropped set equals the declared set, nothing is missing, and every
    remaining key's shape matches — then ``load_state_dict(strict=True)``;
(c) FORWARD identity: ``evaluate_actions_functional`` on the buffer's 64 rows vs ``forward.pt`` with
    ``torch.equal`` on values, log_prob, entropy, masked_logp, masks_bool;
(d) UPDATE identity: one K9 ``train()`` seeded exactly as ``learner_golden.compute`` seeds it, then every
    SURVIVING parameter / buffer vs ``post_state.pt`` and the pinned losses vs ``reference_losses.json``,
    exactly.

Anything not bitwise equal is reported with its max |delta| per group, never rounded away.
``--readd-flat-bias`` is the CONTROL for F16b: it re-attaches the removed flat-pointer bias (the reference's
value, trained by the same optimizer) inside this script only, so the remaining comparison isolates every
OTHER removal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
DEFAULT_REF = Path.home() / ".cache" / "gen3ai" / "vb" / "ref"

#: Every state_dict key the break removes from the production policy, with the finding that removed it.
#: A key of the shared extractor is listed ONCE under ``features_extractor.``; sb3 also exposes the shared
#: module as ``pi_features_extractor.`` / ``vf_features_extractor.`` and those aliases are expanded below.
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
_ALIASES = ("pi_features_extractor.", "vf_features_extractor.")
_FLAT_BIAS = "features_extractor.flat_intent_head.out.bias"
FORWARD_KEYS = ("values", "log_prob", "entropy", "masked_logp", "masks_bool")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _expand(keys: Dict[str, str], ref_keys: List[str]) -> Dict[str, str]:
    out = dict(keys)
    for k, why in keys.items():
        if k.startswith("features_extractor."):
            for a in _ALIASES:
                ak = a + k[len("features_extractor."):]
                if ak in ref_keys:
                    out[ak] = why
    return out


def _maxdiff(a: Any, b: Any) -> float:
    import torch as th
    if a.dtype == th.bool:
        return float((a != b).sum())
    d = (a.double() - b.double()).abs()
    fin = th.isfinite(a.double()) & th.isfinite(b.double())
    same_inf = (a.double() == b.double()) & ~fin
    if bool((~fin & ~same_inf).any()):
        return float("inf")
    return float(d[fin].max()) if bool(fin.any()) else 0.0


def _group(k: str) -> str:
    return ".".join(k.split(".")[:2])


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", nargs="?", default=str(DEFAULT_REF))
    ap.add_argument("--out", default=str(HERE / "result.json"))
    ap.add_argument("--findings", default=None,
                    help="comma list (e.g. F1): declare only these findings' keys — for an INTERMEDIATE commit "
                         "of the bundle; default every finding")
    ap.add_argument("--baseline", action="store_true",
                    help="HARNESS CHECK at a pre-part-2 commit: declare NO removed key")
    ap.add_argument("--readd-flat-bias", action="store_true",
                    help="CONTROL: re-attach the removed F16b bias (reference value, trained) in this script")
    a = ap.parse_args(argv)
    ref = Path(a.ref)
    meta = json.loads((HERE / "reference_meta.json").read_text())
    for f, want in meta["files"].items():
        have = _sha(ref / f)
        if have != want:
            print(f"REFUSED: {ref / f} sha256 {have} != committed {want}", file=sys.stderr)
            return 2

    import subprocess

    import torch as th
    from agents.model.hypothesis_set import IsolatedLinear
    from agents.training import learner_golden as LG

    res: Dict[str, Any] = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                                    text=True, cwd=HERE).stdout.strip(),
                           "reference_commit": meta["commit"], "torch": th.__version__,
                           "control_readd_flat_bias": bool(a.readd_flat_bias), "baseline": bool(a.baseline),
                           "findings": a.findings or "all"}
    if LG.buffer_sha256() != meta["buffer_sha256"]:
        print("REFUSED: the K9 buffer differs from the reference's", file=sys.stderr)
        return 2
    init = th.load(ref / "init_state.pt", map_location="cpu")
    fwd_ref = th.load(ref / "forward.pt", map_location="cpu")
    post = th.load(ref / "post_state.pt", map_location="cpu")
    ref_losses = json.loads((HERE / "reference_losses.json").read_text())

    with LG._one_thread():
        model = LG.build_learner()
        pol = model.policy
        if a.readd_flat_bias:
            # CONTROL ONLY: the pre-break head's `out` carried a bias; put an identical Linear back so the
            # reference bias rides the forward AND the optimizer exactly as it did before the break.
            # The SAME weight Parameter is kept (the optimizer already holds it); the new bias is inserted
            # right after it in the optimizer's one param group — its pre-break position, so the global
            # grad-norm clip sums the per-parameter norms in the pre-break order.
            head = pol.features_extractor.flat_intent_head
            old = head.out
            new = IsolatedLinear(int(old.weight.shape[1]), int(old.weight.shape[0]), bias=True)
            new.weight = old.weight
            head.out = new
            assert len(pol.optimizer.param_groups) == 1, "the control assumes the policy's one param group"
            params = pol.optimizer.param_groups[0]["params"]
            i = next(k for k, p in enumerate(params) if p is old.weight)
            params.insert(i + 1, new.bias)
        head_sd = pol.state_dict()
        _want = None if a.findings is None else set(a.findings.split(","))
        _decl = {k: v for k, v in DECLARED_REMOVED.items() if _want is None or v in _want}
        declared = {} if a.baseline else _expand(_decl, list(init))
        if a.readd_flat_bias:
            declared = {k: v for k, v in declared.items() if not k.endswith("flat_intent_head.out.bias")}
        dropped = sorted(set(init) - set(head_sd))
        missing = sorted(set(head_sd) - set(init))
        reshaped = sorted(k for k in set(init) & set(head_sd) if tuple(init[k].shape) != tuple(head_sd[k].shape))
        res["declared_removed"] = sorted(declared)
        res["dropped"] = dropped
        res["dropped_params"] = int(sum(init[k].numel() for k in dropped
                                        if not k.startswith(_ALIASES)))
        res["missing_from_reference"] = missing
        res["reshaped"] = reshaped
        assert set(dropped) == set(declared), (
            f"dropped set != declared: undeclared {sorted(set(dropped) - set(declared))}, "
            f"declared-but-present {sorted(set(declared) - set(dropped))}")
        assert not missing, f"HEAD has keys the reference lacks: {missing}"
        assert not reshaped, f"reshaped keys: {reshaped}"
        pol.load_state_dict({k: init[k] for k in head_sd}, strict=True)

        # (c) forward identity — the capture's exact sequence.
        LG.load_buffer_into(model, LG.BUFFER_PATH)
        rb = model.rollout_buffer
        obs = {k: th.as_tensor(v.reshape(-1, *v.shape[2:])) for k, v in rb.observations.items()}
        acts = th.as_tensor(rb.actions.reshape(-1, *rb.actions.shape[2:])).long().flatten()
        masks = th.as_tensor(rb.action_masks.reshape(-1, rb.action_masks.shape[-1]))
        pol.set_training_mode(True)
        with th.no_grad():
            out = pol.evaluate_actions_functional(obs, acts, masks)
        fwd = dict(zip(FORWARD_KEYS, out))
        res["forward"] = {k: {"equal": bool(th.equal(fwd[k], fwd_ref[k])),
                              "max_abs_delta": _maxdiff(fwd[k], fwd_ref[k])} for k in FORWARD_KEYS}

        # (d) update identity — `learner_golden.compute`'s own seeding and train().
        r = LG.compute(model=model, buffer=LG.BUFFER_PATH)
        now = pol.state_dict()
    groups: Dict[str, Dict[str, Any]] = {}
    for k in sorted(now):
        if k.startswith(_ALIASES):
            continue
        if a.readd_flat_bias and k == _FLAT_BIAS:
            continue  # the control's re-attached bias is not a SURVIVING parameter
        g = groups.setdefault(_group(k), {"tensors": 0, "unequal": 0, "max_abs_delta": 0.0})
        g["tensors"] += 1
        if not th.equal(now[k], post[k]):
            g["unequal"] += 1
            g["max_abs_delta"] = max(g["max_abs_delta"], _maxdiff(now[k], post[k]))
    res["update_groups"] = groups
    res["update_bitwise_equal"] = all(g["unequal"] == 0 for g in groups.values())
    lc = {}
    for k in sorted(set(ref_losses) | set(r["losses"])):
        rv, hv = ref_losses.get(k), r["losses"].get(k)
        lc[k] = {"reference": rv, "head": hv, "equal": rv == hv}
    res["losses"] = lc
    res["losses_equal"] = all(v["equal"] for v in lc.values())
    res["forward_bitwise_equal"] = all(v["equal"] for v in res["forward"].values())
    res["n_params_head"] = int(sum(p.numel() for p in pol.parameters()))
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: res[k] for k in ("commit", "forward_bitwise_equal", "update_bitwise_equal",
                                          "losses_equal", "dropped_params", "n_params_head")}, indent=1))
    for k, v in res["forward"].items():
        if not v["equal"]:
            print(f"  forward {k}: max|d| {v['max_abs_delta']:.3e}")
    for gname, g in groups.items():
        if g["unequal"]:
            print(f"  update {gname}: {g['unequal']}/{g['tensors']} tensors differ, max|d| {g['max_abs_delta']:.3e}")
    for k, v in lc.items():
        if not v["equal"]:
            print(f"  loss {k}: ref {v['reference']!r} head {v['head']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
