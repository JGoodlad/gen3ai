"""Capture the PIN's static-arm reference for the static-port weight-mapping identity proof.

Run ONCE at the static-token screen's pin (P_st `6c6d2e09`), CPU, one thread, from any cwd, WITHOUT writing
into that checkout:

    python -E capture_reference.py --src <pin checkout>/src OUT_DIR

The learner is the K9 golden's seeded, NAME-KEYED-perturbed build (`learner_golden.build_learner(args,
perturb_keyed=True)`) of the screen's static arm (`_static_build.static_args`): `--arch production` +
`token_encoding static` + `belief_tokens fixed_mass`, readout `tower`, move-resolution off, value-threat-inject
on, speed-physics off. The rows are the pin's committed `learner_golden_buffer_fixed_mass.npz` (64 real rows,
obs 2761 wide).

Each VARIANT is the same init CONDITIONED into the subspace where a claimed identity can hold — one condition
per post-pin change, composable, each documented in `CONDITIONS`:

  tie   (a) the op's 138 per-position `out_gain`s set group-equal per the v145 tie groups
            (`out_gain_groups_v145.json`, read off HEAD's `_out_gain_tie`): each group = its float64 mean, cast.
  pre1  (b) the gains of the channels HEAD's part-5 consumers read PRE-gain (`intent_conditional`: the outgoing
            high roll `out_move[1]`, `out_p_outspeed`, the flinch column of `out_secondary`) set to exactly 1.0,
            so the pin's POST-gain read of them equals HEAD's pre-gain read bit for bit.
  one   (b') every gain = 1.0 (the blunt form of tie + pre1).
  fb0   the flat opponent pointer's shared scorer bias (`flat_intent_head.out.bias`, deleted by F16b) = 0.0, so
            `s + b` == `s` exactly (the forward half of F16b; the update half needs the control).
  c1    (the parent's static-port branch) `pokemon_encoder.role_encoder.0.weight`'s type2 column block set EQUAL
            to its type1 block, so `W1 e(t1) + W2 e(t2)` == `W1 (e(t1) + e(t2))` in real arithmetic.
  c2    (the parent's branch) `op_content.outgoing_proj` weight AND bias = 0, so its replacement (a zero-init
            Deep-Sets module) contributes the same exact 0.

For every variant: ``<variant>/init_state.pt`` (the conditioned `policy.state_dict()`), ``<variant>/forward.pt``
(`evaluate_actions_functional` on the 64 rows); and for ``--update-variants``: one K9 `train()` with the gains
FROZEN (``post_state.pt``, ``losses.json``). Shared: ``rows.pt`` (obs dict, actions, masks), ``meta.json`` (the
commit, torch, the build facts, every file's sha256). Weights stay under ~/.cache (never committed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from _static_build import (HERE, assert_static_fixed_mass_tower, bootstrap, build, checkout_of, forward, sd_sha,
                           sha_file, static_args)

CONDITIONS = ("tie", "pre1", "one", "fb0", "c1", "c2")
VARIANTS = {
    "raw": (),
    "tied": ("tie",),
    "tied_fb0": ("tie", "fb0"),
    "tied_pre1": ("tie", "pre1"),
    "tied_pre1_fb0": ("tie", "pre1", "fb0"),            # THE identity configuration vs HEAD
    "tied_pre1_fb0_c1c2": ("tie", "pre1", "fb0", "c1", "c2"),
    "one_fb0": ("one", "fb0"),
}
UPDATE_VARIANTS_DEFAULT = ("tied_pre1",)                 # compared with/without the F16b control


def _groups():
    g = json.loads((HERE / "out_gain_groups_v145.json").read_text())
    return g["keys"], g["group_of_position"]


def condition(pol, conds) -> dict:
    """Apply ``conds`` IN PLACE on the live parameters (the pi_/vf_ aliases share them). Returns facts."""
    import torch as th
    from agents.model.extractor_forward import _OUT_SEC_FLINCH_COL
    fe = pol.features_extractor
    facts = {}
    keys, gop = _groups()
    with th.no_grad():
        g = fe.damage_op.out_gain
        if len(gop) != g.numel():
            raise SystemExit(f"REFUSED: v145 groups name {len(gop)} positions, the pin's out_gain has {g.numel()}")
        idx = th.tensor(gop)
        if "tie" in conds:
            gd = g.detach().double()
            for j in range(len(keys)):
                sel = idx == j
                g[sel] = gd[sel].mean().float()
            facts["tie_groups"] = len(keys)
        if "pre1" in conds:
            want = {("out_move", "1"), ("out_p_outspeed",), ("out_secondary", str(_OUT_SEC_FLINCH_COL))}
            js = [j for j, k in enumerate(keys) if tuple(k) in want]
            if len(js) != 3:
                raise SystemExit(f"REFUSED: pre1 found {len(js)} of the 3 part-5 channel groups")
            for j in js:
                g[idx == j] = 1.0
            facts["pre1_groups"] = [keys[j] for j in js]
            facts["pre1_positions"] = int(sum(int((idx == j).sum()) for j in js))
        if "one" in conds:
            g.fill_(1.0)
        if "fb0" in conds:
            fe.flat_intent_head.out.bias.zero_()
        if "c1" in conds:
            from agents.model.static_tokens import STAT_FEATURE_DIM
            # `StaticTokenEncoder.encode`'s s_in order: species emb · actual stats (STAT_FEATURE_DIM) · item emb ·
            # item known (1) · type1 emb · type2 emb · ... — confirmed on real rows by `dump_surface.py` (the
            # two windows of s_in that are rows of the type table: [62, 78) and [78, 94)).
            t = int(fe.layout["type_embedding_dim"])
            t1 = (int(fe.layout["species_embedding_dim"]) + STAT_FEATURE_DIM
                  + int(fe.layout["item_embedding_dim"]) + 1)
            if (t1, t) != (62, 16):
                raise SystemExit(f"REFUSED: c1's type1 block is at {t1} (width {t}), dump_surface located 62 (16)")
            w = fe.pokemon_encoder.role_encoder[0].weight
            w[:, t1 + t:t1 + 2 * t] = w[:, t1:t1 + t]
            facts["c1_type1_cols"] = [t1, t1 + t]
            facts["c1_type2_cols"] = [t1 + t, t1 + 2 * t]
        if "c2" in conds:
            op = fe.op_content.outgoing_proj
            op.weight.zero_()
            op.bias.zero_()
            facts["c2_outgoing_proj"] = list(op.weight.shape)
    return facts


def _rows(model):
    import torch as th
    rb = model.rollout_buffer
    obs = {k: th.as_tensor(v.reshape(-1, *v.shape[2:])).clone() for k, v in rb.observations.items()}
    acts = th.as_tensor(rb.actions.reshape(-1, *rb.actions.shape[2:])).long().flatten().clone()
    masks = th.as_tensor(rb.action_masks.reshape(-1, rb.action_masks.shape[-1])).clone()
    return obs, acts, masks


def main(argv):
    src, rest = bootstrap(argv)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--update-variants", default=",".join(UPDATE_VARIANTS_DEFAULT))
    a = ap.parse_args(rest)
    import torch as th
    from agents.training import learner_golden as LG

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    buf = LG.arm_buffer("fixed_mass")
    args, applied = static_args()
    meta = {"commit": checkout_of(src), "src": src, "torch": th.__version__, "overrides": applied,
            "buffer": buf.name, "buffer_sha256": sha_file(buf), "variants": {}, "files": {}}
    with LG._one_thread():
        base = build(args)
        meta["build"] = assert_static_fixed_mass_tower(base.policy.features_extractor)
        init0 = {k: v.clone() for k, v in base.policy.state_dict().items()}
        LG.load_buffer_into(base, buf)
        obs, acts, masks = _rows(base)
        th.save({"obs": obs, "acts": acts, "masks": masks}, out / "rows.pt")
        for name in a.variants.split(","):
            conds = VARIANTS[name]
            base.policy.load_state_dict(init0, strict=True)
            facts = condition(base.policy, conds)
            vd = out / name
            vd.mkdir(exist_ok=True)
            sd = base.policy.state_dict()
            th.save(sd, vd / "init_state.pt")
            th.save(forward(base.policy, obs, acts, masks), vd / "forward.pt")
            meta["variants"][name] = {"conditions": list(conds), "facts": facts, "init_sd_sha256": sd_sha(sd)}
            print(f"[{name}] conditions {conds}  {facts}", flush=True)
        for name in [v for v in a.update_variants.split(",") if v]:
            m = build(args)
            sd = th.load(out / name / "init_state.pt", map_location="cpu")
            m.policy.load_state_dict(sd, strict=True)
            r = LG.compute(model=m, buffer=buf,
                           before_train=lambda m=m: m.policy.features_extractor.damage_op.out_gain.requires_grad_(False))
            th.save(m.policy.state_dict(), out / name / "post_state.pt")
            (out / name / "losses.json").write_text(json.dumps(r["losses"], indent=1, sort_keys=True))
            meta["variants"][name]["update"] = {"init_params_sha256": r["init_params_sha256"],
                                                "post_params_sha256": r["post_params_sha256"],
                                                "gains_frozen": True}
            print(f"[{name}] update done; {len(r['losses'])} pinned losses", flush=True)
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "meta.json":
            meta["files"][str(p.relative_to(out))] = sha_file(p)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    (HERE / "reference_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: meta[k] for k in ("commit", "torch", "buffer_sha256")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
