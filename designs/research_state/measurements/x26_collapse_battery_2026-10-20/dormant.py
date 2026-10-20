"""B10 (plasticity), the offline half that needs no training: the DORMANT-UNIT share and the weight norms of X26
checkpoints, on X26's own bank (forward passes only, CPU). NOT a registered instrument (the battery's B10 is "not
built", F-CC-6); a descriptive first read.

    run.sh dormant.py --bank <bank dir> --out <json> --ckpt <zip>=<label> [--ckpt ...]

Definition (Sokar et al. 2023's tau-dormant score, UNVERIFIED citation from memory): for each hidden unit i of a
layer, a_i = the mean over the bank's rows (and over every token position) of |post-activation output|; its score is
s_i = a_i / mean_j(a_j) over the layer's units. A unit is tau-dormant iff s_i <= tau. Reported at tau = 0 (exactly
dead over the whole bank), 0.025 and 0.1, per layer and pooled (unit-weighted).

Layers read: every ``nn.ReLU`` module's output, and every transformer layer's feed-forward ``linear1`` output after
the layer's own ``activation`` (ReLU where the layer names none: ``BiasedEncoderLayer`` applies ``F.relu`` inline), outside the detached ride-along heads (``ridealong.*``, readouts that cannot steer the
policy). Padding token positions are included in the means (the bank's empty seats): a unit that fires only on real
tokens reads LESS dormant than it is, never more.

Weight norms: the L2 norm of every parameter group (top two name levels), straight from the checkpoint, no forward.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TAUS = (0.0, 0.025, 0.1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args(argv)

    import torch as th
    import torch.nn.functional as F

    from agents.model.extra_obs_keys import zero_extra_obs
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals, load_checkpoint, reencode

    bank = load_bank(Path(a.bank))
    rows, masks, gate = reencode(bank, workers=a.workers)
    out = {"schema": "x26_collapse_battery_dormant_v1", "bank": str(a.bank), "n_rows": int(len(rows)),
           "gate": {k: gate[k] for k in ("recorded_rows_checked", "byte_equal", "obs_as_recorded")},
           "taus": list(TAUS), "reads": {}}
    print(f"bank {len(rows)} rows; obs as recorded {gate['byte_equal']}/{gate['recorded_rows_checked']}", flush=True)

    for spec in a.ckpt:
        path, label = spec.split("=", 1)
        with inference_globals(a.threads):
            model = load_checkpoint(Path(path))
            pol = model.policy
            sums: dict[str, th.Tensor] = {}
            counts: dict[str, int] = defaultdict(int)
            hooks = []

            def acc(name, y):
                y = y.detach().float()
                d = y.shape[-1]
                flat = y.reshape(-1, d).abs()
                s = flat.sum(0)
                sums[name] = sums[name] + s if name in sums else s
                counts[name] += flat.shape[0]

            for name, mod in pol.named_modules():
                if name.startswith("ridealong"):
                    continue
                if isinstance(mod, th.nn.ReLU):
                    hooks.append(mod.register_forward_hook(lambda m, i, o, n=name: acc(n, o)))
                elif isinstance(getattr(mod, "linear1", None), th.nn.Linear) and \
                        isinstance(getattr(mod, "linear2", None), th.nn.Linear):
                    act = getattr(mod, "activation", F.relu)
                    if isinstance(act, str):
                        act = getattr(F, act)
                    hooks.append(mod.linear1.register_forward_hook(
                        lambda m, i, o, n=name + ".ffn", f=act: acc(n, f(o))))
            dev = next(pol.parameters()).device
            for i in range(0, len(rows), a.batch):
                mb = th.tensor(masks[i:i + a.batch].astype(np.float32), device=dev)
                ob = {"observation": th.tensor(rows[i:i + a.batch], device=dev), "action_mask": mb}
                ob.update(zero_extra_obs(pol.features_extractor, batch=len(mb), device=dev))
                with th.no_grad():
                    pol.get_distribution(ob)
                    pol.predict_values(ob)
            for h in hooks:
                h.remove()
            layers = {}
            pooled = {t: [0, 0] for t in TAUS}
            for name, s in sums.items():
                m = (s / counts[name]).numpy()
                mu = float(m.mean())
                score = m / mu if mu > 0 else np.zeros_like(m)
                rec = {"units": int(m.size), "rows_x_positions": int(counts[name]), "layer_mean_abs": mu}
                for t in TAUS:
                    k = int((score <= t).sum())
                    rec[f"dormant_{t}"] = round(k / m.size, 5)
                    pooled[t][0] += k
                    pooled[t][1] += m.size
                layers[name] = rec
            wn = defaultdict(float)
            for name, p in pol.named_parameters():
                wn[".".join(name.split(".")[:2])] += float(p.detach().float().pow(2).sum())
            total_nr = sum(v for k, v in wn.items() if not k.startswith("ridealong"))
            out["reads"][label] = {
                "checkpoint": path,
                "pooled": {f"dormant_{t}": round(pooled[t][0] / max(pooled[t][1], 1), 5) for t in TAUS},
                "units": pooled[TAUS[0]][1], "layers": layers,
                "weight_norm_total_excl_ridealong": round(total_nr ** 0.5, 4),
                "weight_norm_by_group": {k: round(v ** 0.5, 4) for k, v in sorted(wn.items())},
            }
            print(label, out["reads"][label]["pooled"], "wnorm", out["reads"][label]["weight_norm_total_excl_ridealong"],
                  flush=True)
            del model
    Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
