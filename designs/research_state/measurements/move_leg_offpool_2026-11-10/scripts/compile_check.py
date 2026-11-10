"""One-off: does the CPU forward+BACKWARD of the production extractor lower under `move_label_only`, and does the
compiled backward equal eager's? (torch 2.8; the CUDA learner compile cannot be checked without a GPU lease.)"""
import torch
import agents.model.extractor_compiles_test as T

torch.set_num_threads(1)
torch._dynamo.reset()
torch._dynamo.config.suppress_errors = False
print("torch", torch.__version__, flush=True)
for mode in ("move_label_only", "shaping"):
    fe, layout = T._build_production_extractor(belief_grad_mode=mode)
    fe.train()
    obs = T._obs(layout, "cpu")
    fe.zero_grad(set_to_none=True)
    T._backward_through(fe, fe(obs))
    eager = {n: (None if p.grad is None else p.grad.clone()) for n, p in fe.named_parameters()}
    fe.zero_grad(set_to_none=True)
    torch._dynamo.reset()
    T._backward_through(fe, T.compile_on(fe.forward, device="cpu")(obs))
    comp = {n: (None if p.grad is None else p.grad.clone()) for n, p in fe.named_parameters()}
    worst, n_cmp, mism = 0.0, 0, []
    for n in eager:
        a, b = eager[n], comp[n]
        az = a is None or float(a.abs().sum()) == 0.0
        bz = b is None or float(b.abs().sum()) == 0.0
        if az != bz:
            mism.append(n); continue
        if az:
            continue
        n_cmp += 1
        worst = max(worst, float((a - b).abs().max() / (a.abs().max() + 1e-30)))
    mh = [n for n in eager if n.startswith("move_belief.move_head")]
    print(f"{mode}: compiled CPU forward+backward OK; {n_cmp} params compared, worst relative gradient error "
          f"{worst:.3e}; live/dead mismatches {mism[:5]}; move_head eager "
          f"{[None if eager[n] is None else float(eager[n].abs().sum()) for n in mh]} compiled "
          f"{[None if comp[n] is None else float(comp[n].abs().sum()) for n in mh]}", flush=True)
