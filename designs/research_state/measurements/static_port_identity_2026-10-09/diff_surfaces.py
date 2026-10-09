"""Diff two `dump_surface.py` outputs (no torch): the observation-layout PREFIX check, the state_dict key / shape
diff, and the build facts.

    python diff_surfaces.py surface_pin.json surface_target.json [--out surface_diff.json]

The prefix verdict: every layout entry and every `agents.observation.constants` int the two share must be EQUAL,
except the three entries that ARE the total width (`total_dim`, `base_dim`, `parts.reactive.end` — the reactive
part's `end` is `self.dimension`); the target may ADD only `obs_facts*` entries / `OFFSET_OBS_FACTS`, which must
start at the pin's total width.
"""
from __future__ import annotations

import json
import sys

TOTAL_KEYS = {"total_dim", "base_dim", "parts.reactive.end"}


def _flat(d, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flat(v, f"{pre}{k}."))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(_flat(v, f"{pre}{i}."))
    else:
        out[pre[:-1]] = d
    return out


def main(argv):
    p, t = json.load(open(argv[0])), json.load(open(argv[1]))
    out_path = argv[argv.index("--out") + 1] if "--out" in argv else None
    pl, tl = _flat(p["layout"]), _flat(t["layout"])
    pc, tc = p["obs_constants"], t["obs_constants"]
    lay_diff = {k: [pl[k], tl[k]] for k in sorted(set(pl) & set(tl)) if pl[k] != tl[k]}
    const_diff = {k: [pc[k], tc[k]] for k in sorted(set(pc) & set(tc)) if pc[k] != tc[k]}
    only_t = sorted(set(tl) - set(pl))
    only_p = sorted(set(pl) - set(tl))
    width = pl["total_dim"]
    bad = (set(lay_diff) - TOTAL_KEYS) or only_p or [k for k in only_t if not k.startswith("obs_facts")] \
        or const_diff or [k for k in set(tc) - set(pc) if k != "OFFSET_OBS_FACTS"] \
        or tc.get("OFFSET_OBS_FACTS", width) != width or tl.get("obs_facts_offset", width) != width
    ps, ts = p["state_dict"], t["state_dict"]
    res = {
        "pin": p["commit"], "target": t["commit"],
        "prefix_identical": not bad,
        "layout_differing_entries": lay_diff,
        "layout_entries_only_target": only_t, "layout_entries_only_pin": only_p,
        "obs_constants_differing": const_diff,
        "obs_constants_only_target": sorted(set(tc) - set(pc)),
        "obs_width": [width, tl["total_dim"]],
        "obs_facts_offset_target": tc.get("OFFSET_OBS_FACTS"),
        "obs_space_differing": {k: [p["obs_space"].get(k), t["obs_space"].get(k)]
                                for k in sorted(set(p["obs_space"]) | set(t["obs_space"]))
                                if p["obs_space"].get(k) != t["obs_space"].get(k)},
        "state_dict_only_pin": sorted(k for k in set(ps) - set(ts) if k.startswith(("features_extractor.", "mlp_", "value_net", "pointer_head", "action"))),
        "state_dict_only_target": sorted(set(ts) - set(ps)),
        "state_dict_reshaped": {k: [ps[k], ts[k]] for k in sorted(set(ps) & set(ts)) if ps[k] != ts[k]},
        "n_params": [p["n_params"], t["n_params"]],
        "build": {"pin": p["build"], "target": t["build"]},
        "role_in": {"pin": p["role_in"], "target": t["role_in"]},
    }
    s = json.dumps(res, indent=1, sort_keys=True)
    if out_path:
        open(out_path, "w").write(s + "\n")
    print(s)
    return 0 if res["prefix_identical"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
