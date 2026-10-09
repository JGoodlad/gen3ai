"""Dump the static arm's SURFACE at one checkout (CPU, one thread) — no weights, a small JSON:

    python -I dump_surface.py --src <checkout>/src OUT.json

* the build facts (static x fixed_mass x tower, read off the modules);
* every policy state_dict key with its shape;
* the observation LAYOUT (`Gen3ObservationEncoder.get_layout()`) and every int constant of
  `agents.observation.constants` — the prefix-identity check reads these;
* the observation space's keys / shapes;
* the static encoder's ``s_in`` column blocks (``role_encoder.0``'s input), located by a forward pre-hook on real
  golden rows and matched against the type-embedding table (condition c1's offsets);
* the op's ``out_gain`` tie groups where the checkout ties them (`_out_gain_tie`; v144+).
"""
from __future__ import annotations

import json
import sys

from _static_build import assert_static_fixed_mass_tower, bootstrap, build, checkout_of, static_args


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return repr(x)


def main(argv):
    src, rest = bootstrap(argv)
    out_path = rest[0]
    import torch as th
    from agents.observation import constants as C
    from agents.training import learner_golden as LG
    from utils.torch_state_guard import single_thread_build

    args, applied = static_args()
    model = build(args)
    pol = model.policy
    fe = pol.features_extractor
    res = {"src": src, "commit": checkout_of(src), "torch": th.__version__, "overrides": applied,
           "build": assert_static_fixed_mass_tower(fe)}
    sd = pol.state_dict()
    res["state_dict"] = {k: list(v.shape) for k, v in sd.items()}
    res["n_params"] = int(sum(p.numel() for p in pol.parameters()))
    # The layout the MODEL reads (the extractor's own copy, built by the trainer from
    # `Gen3ObservationEncoder.get_layout()` with the run's mappings).
    res["layout"] = _jsonable(fe.layout)
    res["obs_constants"] = {k: v for k, v in sorted(vars(C).items())
                            if k.isupper() and isinstance(v, int) and not isinstance(v, bool)}
    res["obs_space"] = {k: list(s.shape) for k, s in model.observation_space.spaces.items()}
    tie = getattr(fe.damage_op, "_out_gain_tie", None)
    res["out_gain"] = {"numel": int(fe.damage_op.out_gain.numel()), "out_dim": int(fe.damage_op.out_dim)}
    if tie is not None:
        res["out_gain"]["group_of_position"] = [int(i) for i in tie.argmax(0).tolist()]
        res["out_gain"]["keys"] = [list(map(str, k)) for k in fe.damage_op.out_gain_keys]

    # c1: locate the type1 / type2 column blocks of role_encoder.0's input on real rows.
    seen = {}

    def _hook(mod, inp):
        seen.setdefault("s_in", inp[0].detach().clone())
    pe = fe.pokemon_encoder
    h = pe.role_encoder[0].register_forward_pre_hook(_hook)
    buf = LG.arm_buffer("fixed_mass") if hasattr(LG, "arm_buffer") else LG.BUFFER_PATH
    with single_thread_build():
        LG.load_buffer_into(model, buf)
        rb = model.rollout_buffer
        obs = {k: th.as_tensor(v.reshape(-1, *v.shape[2:])) for k, v in rb.observations.items()}
        with th.no_grad():
            pol.extract_features(obs)
    h.remove()
    s_in = seen["s_in"]
    tdim = int(fe.layout["type_embedding_dim"])
    table = fe.embeddings.type_embedding.weight.detach() if hasattr(fe, "embeddings") else None
    res["role_in"] = {"width": int(s_in.shape[-1]), "role_encoder.0.weight": list(pe.role_encoder[0].weight.shape),
                      "type_embedding_dim": tdim,
                      "species_embedding_dim": int(fe.layout["species_embedding_dim"]),
                      "item_embedding_dim": int(fe.layout["item_embedding_dim"])}
    # Every column window of width tdim whose every row equals SOME row of the type table, exactly.
    if table is not None:
        flat = s_in.reshape(-1, s_in.shape[-1])
        hits = []
        for off in range(0, flat.shape[-1] - tdim + 1):
            win = flat[:, off:off + tdim]
            eq = (win[:, None, :] == table[None, :, :]).all(-1).any(-1)
            if bool(eq.all()) and bool((win != 0).any()):
                hits.append(off)
        res["role_in"]["type_table_windows"] = hits
    res["buffer"] = str(buf.name)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print(json.dumps({k: res[k] for k in ("commit", "build", "n_params", "out_gain", "role_in")}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
