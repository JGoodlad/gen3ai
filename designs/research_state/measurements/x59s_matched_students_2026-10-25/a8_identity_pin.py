"""Amendment 8 identity, PIN side (the a8 tree): each student through h2h's own host loader (`_load_host`), its logits
bit-equal to the reference; every registered pairing declared on one engine; one engine built and closed per pairing
(the slot loads, copy checks and parity gates, no game)."""
import json, sys
from pathlib import Path
import numpy as np
from main.policy_spectrum.reader import inference_globals, policy_logits
from main.h2h.play import resolve_player, check_core_flags, H2HEngine, Compute, panel_builder, engine_lock
from main.h2h.arch import _load_host, declare_engine

import agents.model.snapshot as _SN, main.h2h.arch as _AR
assert "/x59s_play_pin_a8/src/" in _SN.__file__ and "/x59s_play_pin_a8/src/" in _AR.__file__, (_SN.__file__, _AR.__file__)
assert hasattr(_SN, "sanitize_future_ffn_width")
out = Path(sys.argv[1]); root = Path(sys.argv[2])
st = json.loads((root / "read/students/students.json").read_text())["students"]
res = {"students": {}, "declare": {}, "engine": {}, "snapshot_module": _SN.__file__}
refs = {}
for lab, s in sorted(st.items()):
    z = np.load(out / f"{lab}.npz")
    ref = resolve_player(s["zip_path"]); check_core_flags(ref); refs[lab] = ref
    with inference_globals(2):
        lg = policy_logits(_load_host(ref), z["rows"], z["mask"])
    eq = bool(lg.shape == z["ref"].shape and np.array_equal(lg.view(np.uint32), z["ref"].view(np.uint32)))
    res["students"][lab] = {"bit_equal": eq, "zip_sha256": ref.sha256, "max_abs": float(np.abs(lg - z["ref"]).max())}
    print(lab, eq, flush=True)
for k in range(5901, 5906):
    for c, opp in (("rating", f"s_hmix_{k}"), ("data", f"s_scut_{k}")):
        p, o = refs[f"s_h1500_{k}"], refs[opp]
        d = declare_engine([(p, o)], _load_host, "off")
        res["declare"][f"{c}_{k}"] = {"groups": len(d.archs)}
for c, opp in (("rating", "s_hmix_5901"), ("data", "s_scut_5901")):
    comp = Compute(device="cpu", n_envs=4, threads=2, torch_threads=2)
    with engine_lock(comp):
        eng = H2HEngine(refs["s_h1500_5901"], refs[opp], comp, team_builder=panel_builder("p72"))
        try:
            res["engine"][c] = {"regime_id": eng.regime["regime_id"], "groups": len(eng.decl.archs)}
        finally:
            eng.close()
res["pass"] = all(v["bit_equal"] for v in res["students"].values()) and len(res["students"]) == 15
(out / "pin_side.json").write_text(json.dumps(res, indent=1, sort_keys=True))
print(json.dumps({"pass": res["pass"], "declare": res["declare"], "engine": res["engine"]}), flush=True)
