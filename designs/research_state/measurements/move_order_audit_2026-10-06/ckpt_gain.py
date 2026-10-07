"""Read a trained checkpoint's per-slot op out_gain and the move-network weight on the validity bit."""
import io
import sys
import zipfile

import torch as th

for path in sys.argv[1:]:
    with zipfile.ZipFile(path) as zf:
        sd = th.load(io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=True)
    keys = [k for k in sd if k.endswith("out_gain")]
    print(path)
    for k in keys:
        g = sd[k]
        inc = 6 * 12 + 13
        blk = g[inc:inc + 16].view(4, 4)
        print(" ", k, "outgoing [low,high,crit,pko] per request slot:")
        for r in blk.tolist():
            print("    ", [round(x, 3) for x in r])
        st0 = inc + 45
        print("    status p_land per slot", [round(x, 3) for x in g[st0:st0 + 4].tolist()])
    mk = [k for k in sd if "move_network" in k and k.endswith("weight")]
    if mk:
        w = sd[mk[0]]
        print(" ", mk[0], tuple(w.shape), "last-input-col (validity?) |w| mean", w[:, -1].abs().mean().item(),
              "all-col |w| mean", w.abs().mean().item())
