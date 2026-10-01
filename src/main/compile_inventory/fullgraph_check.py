"""Does each candidate REGION compile with `fullgraph=True`? — the K8 verdict `explain` cannot give.

`torch._dynamo.explain` counts a break through the graph it ENDS, so a break at a frame's first
instruction (2.5.1's `forward_guard` lookup, 2026-09-30) reads as "0 breaks". K8 compiles every
region `fullgraph=True`, so THAT is the test: compile the callable with `fullgraph=True`, run it once
in each mode, and report dynamo's refusal (reason + innermost user frame) or OK.

The production policy surface (`main.fresh_checkpoint`'s kwargs), the committed real-obs fixture
rows, CPU, fresh weights. Candidates: the extractor's `forward`, the same through `Module.__call__`
and `extract_features`, and the rollout's `policy(obs, action_masks=…)`; modes train/grad,
train/no-grad, eval/no-grad.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Tuple

MODES = ("train_grad", "train_nograd", "eval_nograd")


def _refusal(e: BaseException) -> str:
    lines = [ln for ln in str(e).strip().splitlines() if ln.strip()]
    locs = re.findall(r'File "([^"]+)", line (\d+), in (\S+)', str(e))
    where = ""
    if locs:
        f, n, fn = locs[-1]
        where = f" @ {f.split('/src/')[-1].split('site-packages/')[-1]}:{n} {fn}"
    return "REFUSED: " + (lines[0][:200] if lines else repr(e)) + where


def run(batch: int = 16, backend: str = "eager") -> Dict[str, Any]:
    import numpy as np
    import torch
    from gymnasium import spaces

    from agents.action.constants import ACTION_SPACE_SIZE
    from agents.model.compile_parity_fixture import load_parity_rows
    from agents.model.extra_obs_keys import zero_extra_obs
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from main.fresh_checkpoint import _production_policy_kwargs

    with torch.random.fork_rng():
        torch.manual_seed(0)
        _args, layout, pk = _production_policy_kwargs()
        dim = layout["total_dim"]
        obs_space = spaces.Dict({
            "observation": spaces.Box(0.0, 1.0, (dim,), np.float32),
            "action_mask": spaces.Box(0, 1, (ACTION_SPACE_SIZE,), np.int8)})
        pol = Gen3DualHeadMaskablePolicy(obs_space, spaces.Discrete(ACTION_SPACE_SIZE),
                                         lambda _: 3e-4, **pk)
    from agents.model.compile_cache import ensure_hermetic_cache
    ensure_hermetic_cache("compile_inventory fullgraph")  # K3: never torch's shared default dir
    fe = pol.features_extractor
    rows, masks = load_parity_rows(dim)
    obs = {"observation": torch.as_tensor(rows[:batch], dtype=torch.float32)}
    obs.update(zero_extra_obs(fe, batch=batch, device=torch.device("cpu")))
    m = np.asarray(masks[:batch]).astype(bool)
    cands: List[Tuple[str, Callable[[Any], Any]]] = [
        ("extractor.forward", lambda o: fe.forward(o)),
        ("extractor via Module.__call__", lambda o: fe(o)),
        ("policy.extract_features", lambda o: pol.extract_features(o)),
        ("policy(obs, action_masks) [rollout]", lambda o: pol(o, action_masks=m)),
    ]
    out: Dict[str, Any] = {"torch": torch.__version__, "batch": batch, "backend": backend,
                           "results": {}}
    for name, fn in cands:
        for mode in MODES:
            if name.startswith("policy(") and mode != "eval_nograd":
                continue                       # the rollout call is an eval/no-grad call
            torch._dynamo.reset()
            pol.set_training_mode(mode.startswith("train"))
            try:
                f = torch.compile(fn, fullgraph=True, backend=backend)
                if mode.endswith("nograd"):
                    with torch.no_grad():
                        f(obs)
                else:
                    pi, vf = f(obs)
                    (pi.sum() + vf.sum()).backward()
                    pol.zero_grad(set_to_none=True)
                verdict = "OK"
            except Exception as e:  # noqa: BLE001 - the refusal IS the measurement
                verdict = _refusal(e)
            out["results"][f"{name} | {mode}"] = verdict
    torch._dynamo.reset()
    return out
