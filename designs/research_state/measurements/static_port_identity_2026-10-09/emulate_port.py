"""A SELF-TEST of the ``static_port`` mapping before the parent's branch exists: emulate its two changes IN THIS
PROCESS on the target checkout's static arm, then run the same map + forward compare `compare_mapped.py` runs.

    python -E emulate_port.py --src <target checkout>/src REF_DIR [--c1-spelling sum|concat] [--out result.json]

The emulation (built AFTER the seeded learner, so no init byte moves; the weights are the mapped reference's):

* (c1) `pokemon_encoder.role_encoder.0` becomes ``SumTypesLinear`` (162 -> 256): it reads the encoder's 178-wide
  ``s_in`` and replaces the two type blocks [62, 78) / [78, 94) by their SUM, `e(t1) + e(t2)` — the parent's
  ``emb(t1) + emb(t2)`` (input width - 16). ``--c1-spelling concat`` is the CONTROL: the same 162-wide weight
  applied to the CONCATENATED input with its type1 block duplicated (``[W_pre, W1, W1, W_post] @ s_in``), i.e. the
  pin's arithmetic exactly — so a difference between the two spellings is the summation order alone.
* (c2) `op_content.outgoing_proj` becomes ``DeepSetsOut``: per (their mon, our move) phi = Linear(6 -> 32) + ReLU on
  the move's six d1 cells, SUM over the four moves, rho = Linear(32 -> 128) ZERO-INIT (weight and bias) — the
  parent's per-move shared Deep-Sets module. phi keeps a nonzero (seeded) init so only rho's zero makes it 0.

Expected: c2 bitwise (x + 0 == x); c1 bitwise under ``concat``, identity up to fp32 summation order under ``sum``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv):
    from _static_build import bootstrap
    src, rest = bootstrap(argv)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--variant", default="tied_pre1_fb0_c1c2")
    ap.add_argument("--c1-spelling", choices=("sum", "concat"), default="sum")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(rest)

    import torch as th
    import torch.nn.functional as F
    import compare_mapped as CM
    from _static_build import (FORWARD_KEYS, HERE, assert_static_fixed_mass_tower, build, checkout_of, forward,
                               maxdiff, sha_file, static_args)
    from agents.training import learner_golden as LG

    T1, T2 = CM.TYPE1, CM.TYPE2

    class SumTypesLinear(th.nn.Module):
        def __init__(self, n_in: int, n_out: int, spelling: str) -> None:
            super().__init__()
            self.weight = th.nn.Parameter(th.zeros(n_out, n_in))
            self.bias = th.nn.Parameter(th.zeros(n_out))
            self.spelling = spelling

        def forward(self, x: th.Tensor) -> th.Tensor:           # x: the 178-wide s_in
            if self.spelling == "sum":
                x2 = th.cat([x[..., :T1[0]], x[..., T1[0]:T1[1]] + x[..., T2[0]:T2[1]], x[..., T2[1]:]], dim=-1)
                return F.linear(x2, self.weight, self.bias)
            w = self.weight
            wfull = th.cat([w[:, :T2[0]], w[:, T1[0]:T1[1]], w[:, T2[0]:]], dim=1)   # type1 block duplicated
            return F.linear(x, wfull, self.bias)

    class DeepSetsOut(th.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            with th.random.fork_rng(devices=[]):
                th.manual_seed(20261009)
                self.phi = th.nn.Linear(6, 32)
            self.rho = th.nn.Linear(32, 128)
            th.nn.init.zeros_(self.rho.weight)
            th.nn.init.zeros_(self.rho.bias)

        def forward(self, per_mon: th.Tensor) -> th.Tensor:     # [B,6,24] = their mon x (our move k, cell)
            B = per_mon.shape[0]
            m = per_mon.reshape(B, per_mon.shape[1], 4, 6)
            return self.rho(th.relu(self.phi(m)).sum(dim=2))

    ref = Path(a.ref)
    meta = json.loads((HERE / "reference_meta.json").read_text())
    for f, want in meta["files"].items():
        if sha_file(ref / f) != want:
            print(f"REFUSED: {ref / f} != committed sha256", file=sys.stderr)
            return 2
    rows = th.load(ref / "rows.pt", map_location="cpu")
    args, applied = static_args()
    rep = {"commit": checkout_of(src), "reference_commit": meta["commit"], "variant": a.variant,
           "c1_spelling": a.c1_spelling, "emulated": True, "torch": th.__version__}
    with LG._one_thread():
        model = build(args)
        pol = model.policy
        fe = pol.features_extractor
        rep["build"] = assert_static_fixed_mass_tower(fe)
        old = fe.pokemon_encoder.role_encoder[0]
        fe.pokemon_encoder.role_encoder[0] = SumTypesLinear(int(old.in_features) - (T2[1] - T2[0]),
                                                            int(old.out_features), a.c1_spelling)
        fe.op_content.outgoing_proj = DeepSetsOut()
        init = th.load(ref / a.variant / "init_state.pt", map_location="cpu")
        CM._map_and_load(model, init, "static_port", {}, rep)
        width = int(model.observation_space.spaces["observation"].shape[0])
        fwd_ref = th.load(ref / a.variant / "forward.pt", map_location="cpu")
        o = forward(pol, CM._extend(rows["obs"], width, "randn"), rows["acts"], rows["masks"])
    rep["forward"] = {k: {"equal": bool(th.equal(o[k], fwd_ref[k])), "max_abs_delta": maxdiff(o[k], fwd_ref[k])}
                      for k in FORWARD_KEYS}
    rep["forward_bitwise"] = all(v["equal"] for v in rep["forward"].values())
    out = Path(a.out) if a.out else HERE / f"result_emulated_port_c1{a.c1_spelling}.json"
    out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: rep[k] for k in ("c1_spelling", "forward_bitwise", "c2", "c1", "c2_output_layer")},
                     indent=1))
    for k, v in rep["forward"].items():
        if not v["equal"]:
            print(f"  forward {k}: max|d| {v['max_abs_delta']:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
