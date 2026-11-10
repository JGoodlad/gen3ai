"""Which tensors did X68's fine-tune move, and by how much on the channels that were raised?"""
import io, os, sys, zipfile, numpy as np, torch
D = os.path.expanduser("~/gen3ai_archive/move_leg_diag_2026-11-10/dump")
p0 = str(np.load(f"{D}/pstar__tables.npz")["ckpt_path"]); p1 = str(np.load(f"{D}/x68ft__tables.npz")["ckpt_path"])
def sd(p):
    with zipfile.ZipFile(p) as z:
        return torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=False)
a, b = sd(p0), sd(p1)
ch = [k for k in a if not torch.equal(a[k], b[k])]
print("changed tensors:", len(ch))
for k in ch:
    d = (b[k].float() - a[k].float())
    print(f"  {k:70s} shape {tuple(a[k].shape)}  |a| rms {a[k].float().pow(2).mean().sqrt():.4f}  |delta| rms {d.pow(2).mean().sqrt():.4f} max {d.abs().max():.4f}")
wk = [k for k in a if k.endswith("move_belief.move_head.weight")][0]; bk = wk.replace("weight", "bias")
W0, W1, B0, B1 = a[wk].float(), b[wk].float(), a[bk].float(), b[bk].float()
dn = (W1 - W0).norm(dim=1); n0 = W0.norm(dim=1)
print("move_head row norms: P* median", float(n0.median()), "delta median", float(dn.median()))
for m in (253, 325, 222, 214, 89, 157, 363, 355, 237):
    print(f"  ch {m}: |w0| {n0[m]:.3f} |dw| {dn[m]:.3f} cos(w0,w1) {torch.nn.functional.cosine_similarity(W0[m], W1[m], dim=0):.3f} b0 {B0[m]:+.3f} b1 {B1[m]:+.3f}")
top = torch.argsort(-dn)[:15]
print("largest row moves:", [(int(i), round(float(dn[i]), 3), round(float(n0[i]), 3)) for i in top])
t = np.load(f"{D}/pstar__tables.npz"); pl = t["prior_logits"]
print("prior logit of typed HP channels (355-370): min", pl[:, 355:371].min(), "max", pl[:, 355:371].max(), " of 237: max", pl[:, 237].max())
