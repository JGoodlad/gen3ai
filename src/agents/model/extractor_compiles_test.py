"""Does the feature extractor still COMPILE? The device x grad matrix, pinned.

Two separate obligations live here, and they are easy to conflate:

1. **The `species_posterior` spelling.** `--compile-opponents` used to set
   `torch._dynamo.config.suppress_errors = True`. That looked like defensive hygiene but was
   actually working around a single failure: on the `BeliefHead.species_posterior` path, the
   softmax over species logits lowered to a `[B,6,n_species]` numerator plus a `[B,6,1]`
   denominator and the Inductor **CPU** scheduler asserted (`AssertionError: buf<N>`) trying to
   fuse the division. Suppression converted that into a per-frame eager fallback, so the
   production config compiled only PARTIALLY (3.6x instead of 6.5x) — and every unrelated backend
   failure in the process became silent too. `BeliefHead.species_posterior` fixes it by spelling
   the same math as `log_softmax(...).exp()`. `tmp/softmax_variant_probe.py` shows `.contiguous()`,
   `.clone()`, a 2-D reshape and a hand-rolled `exp / sum` all still FAIL, so the working spelling
   is not obvious and is very easy to "simplify" back into a broken one.

2. **The compile TARGETS themselves.** Four cells, and they are genuinely different code paths —
   Inductor's CPU backend emits C++, its CUDA backend emits **Triton**, and the backward is
   produced by AOTAutograd rather than by tracing the forward. A green CPU-forward test says
   nothing about the other three:

   | | forward | forward + backward |
   |---|---|---|
   | **CPU** | ✅ the frozen self-play OPPONENT (`--compile-opponents`, B=1 in every env worker) | ❌ **does not lower** — Inductor's C++ backend asserts on an `atomic_add` scatter |
   | **CUDA** | ✅ eval / inference on the card | ✅ the TRAINER's step, the ~1.75x lever |

   Each cell has its own test below, including the ❌ one — that is a LIMITATION PIN, and it fails
   if the limitation ever LIFTS, because three things in the codebase currently assume it holds
   (see `test_cpu_backward_still_does_not_compile`). The rest run BY DEFAULT: the point is to catch
   "the model no longer compiles" at code time rather than as a silent ~6.5x slower opponent
   forward, or as a compile lever that turns out to be unavailable when someone reaches for it a
   month from now.

**You cannot compile FOR cuda ON the cpu.** Measured, not assumed (2026-08-14, torch 2.5.1 +
triton 3.1.0): with `CUDA_VISIBLE_DEVICES=""` an Inductor cuda compile dies with
`RuntimeError: No CUDA GPUs are available` — the backend queries live device properties, so
codegen is not a pure AOT source->PTX step it could do blind. Tracing under `FakeTensorMode`
against a cuda device does not substitute either: it exercises **dynamo**, which is device-
agnostic anyway (a graph break is a Python-level event), and never reaches the backend where the
device-specific bugs live. So the CUDA cells need the real card — and because this box normally
carries a training run, they SKIP THEMSELVES when the GPU is busy rather than risk an OOM in it.
See `_cuda_skip_reason`.

Costs, measured on an idle RTX 3080 Ti at the batch these tests use: cuda forward ~5 s / 552 MiB
of process VRAM, cuda forward+backward ~57 s / ~1.1 GiB. The CPU cells are ~10 s each on a warm
Inductor cache.

Opt out of every compile cell when you need a fast loop:

    GEN3AI_SKIP_COMPILE_TESTS=1 pytest src/ -q

Run the CUDA cells at all — a normal `pytest` run SKIPS them, because the root `conftest.py` hides
the GPU from the whole suite so a stray `device="auto"` can never steal VRAM from a live run:

    GEN3AI_TEST_ALLOW_GPU=1 pytest src/agents/model/extractor_compiles_test.py -q
"""
import inspect
import os

import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model.features_extractor import BeliefHead, Gen3FeaturesExtractor
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

# Opt-OUT, not opt-in: a model that stops compiling is a ~6.5x regression that is invisible at
# runtime, so the default has to be "the test suite catches it".
_SKIP_COMPILE = os.environ.get("GEN3AI_SKIP_COMPILE_TESTS") == "1"
_skip_compile = pytest.mark.skipif(_SKIP_COMPILE, reason="GEN3AI_SKIP_COMPILE_TESTS=1")

# The CUDA cells need the REAL card — see the module docstring; cuda codegen cannot be driven from
# a CPU-only process. Two independent things stop them from touching a training run, and the first
# one is not ours:
#
#   1. The ROOT `conftest.py` hard-sets `CUDA_VISIBLE_DEVICES=""` for the whole suite, precisely so
#      a stray `device="auto"` can never steal VRAM from a live run. So under plain `pytest` these
#      cells see no GPU AT ALL and skip. `GEN3AI_TEST_ALLOW_GPU=1` is that conftest's documented
#      escape hatch, and it is the ONE knob here — a second env var of our own would just be a
#      knob the reader has to discover twice.
#   2. Even unhidden, we refuse to run when the card is BUSY. Measured footprint at this batch is
#      ~1.1 GiB for the forward+backward cell; a compile test must never be the thing that OOMs a
#      20-hour run.
_CUDA_MIN_FREE_MIB = 2500


def _smi(query, flag="--query-gpu"):
    """Ask the driver, WITHOUT creating a CUDA context.

    `torch.cuda.mem_get_info()` would answer the memory question, but it initialises a context
    (~250 MiB) merely to decide whether to skip — i.e. the check itself would do the interference
    it exists to avoid. `nvidia-smi` reads the driver instead. Returns None when it cannot tell.
    """
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", f"{flag}={query}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return None
        return [l for l in out.stdout.strip().splitlines() if l.strip()]
    except Exception:
        return None


def _other_gpu_processes():
    """PIDs of OTHER processes with a CUDA context on the card (ours excluded). None if unknown."""
    rows = _smi("pid", flag="--query-compute-apps")
    if rows is None:
        return None
    mine = {os.getpid(), os.getppid()}
    return [int(r.split(",")[0]) for r in rows if int(r.split(",")[0]) not in mine]


def _free_vram_mib():
    rows = _smi("memory.free")
    return int(rows[0]) if rows else None


def _cuda_skip_reason():
    """None to run, else the reason — which NAMES the cause and the knob.

    A bare "no CUDA device" would be actively misleading here: the box HAS a 12 GiB card and the
    suite hides it on purpose. And a silent skip on a box that is always training would turn this
    gate into a no-op that still reads as green — the project's own "a default branch nothing tests
    is untested no matter how green the suite looks" lesson. So every skip says what to do about it.
    """
    if not torch.cuda.is_available():
        if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
            return ("the root conftest.py hides the GPU from the whole suite (CUDA_VISIBLE_DEVICES="
                    "''), so the CUDA compile targets are NOT covered by a normal test run. Re-run "
                    "with GEN3AI_TEST_ALLOW_GPU=1 on an idle box to actually check them.")
        return "no CUDA device on this machine"
    # "Is anyone ELSE on the card" comes first, and free VRAM is only the backstop. Free memory is
    # the wrong question on its own: measured 2026-08-14 with a live training run holding 8.4 GB,
    # 3.3 GB was still free — comfortably over any floor sized for this test's ~1.1 GB — so a
    # memory-only guard would have run anyway and left that run ~2 GB of headroom for its next eval
    # cycle. The rule is "do not share the card", not "squeeze in beside them".
    others = _other_gpu_processes()
    if others is None:
        # Fail SAFE: unknown occupancy means we do not gamble a training run on it.
        return "cannot read GPU occupancy (no nvidia-smi?) — skipping rather than risk a live run"
    if others:
        return (f"another process is using the GPU (pid {others[0]}) — almost certainly a training "
                f"run. NOT a pass: the CUDA compile targets went unchecked. Re-run on an idle box.")
    free = _free_vram_mib()
    if free is not None and free < _CUDA_MIN_FREE_MIB:
        return (f"GPU has only {free} MiB free (< {_CUDA_MIN_FREE_MIB} needed) with no process "
                f"visible — something is holding VRAM. NOT a pass; re-run on an idle box.")
    return None


_skip_cuda = pytest.mark.skipif(_cuda_skip_reason() is not None,
                                reason=_cuda_skip_reason() or "")

# Small on purpose: these gates ask "does it lower", not "how fast is it". Batch 2 (not 1) keeps the
# batch dim from being a degenerate broadcast that could hide a shape bug.
_BATCH = 2


def test_species_posterior_matches_plain_softmax():
    """The rewrite must be MATH-neutral: `exp(log_softmax(x)) == softmax(x)`."""
    torch.manual_seed(0)
    head = BeliefHead.__new__(BeliefHead)          # bypass __init__; we only exercise the spelling
    logits = torch.randn(3, 6, 400) * 7.0          # wide scale: catches a non-stable factoring
    head.species_logits = lambda tokens, *a, **k: logits    # type: ignore[assignment]
    got = BeliefHead.species_posterior(head, torch.zeros(3, 6, 8))
    want = torch.softmax(logits, dim=-1)
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()
    assert torch.allclose(got.sum(-1), torch.ones(3, 6), atol=1e-5)


def test_species_posterior_is_stable_for_large_logits():
    """`log_softmax().exp()` keeps softmax's max-subtraction, so a naive `exp/sum` blowup must not
    reappear if someone re-spells this."""
    head = BeliefHead.__new__(BeliefHead)
    logits = torch.tensor([[[1e4, 1e4 + 1.0, -1e4]]])
    head.species_logits = lambda tokens, *a, **k: logits    # type: ignore[assignment]
    got = BeliefHead.species_posterior(head, torch.zeros(1, 1, 8))
    assert torch.isfinite(got).all()
    assert pytest.approx(1.0, abs=1e-5) == float(got.sum())


# ⚠️ THE PRODUCTION ARCH IS LOADED, NEVER HAND-PINNED. This test used to carry a literal
# `_PRODUCTION_ARCH = dict(...)` frozen at the ai_v8_03 shape (the between-layers refine loop, no
# prefuse, no edge families) — and it ROTTED silently: for three generations the default-on compile gate
# faithfully compiled a DEAD graph while the real production graph drifted, which is exactly how
# the gen-4 launch's CompilePrewarm failure (`gen3_unrevealed_outgoing_prior_v1`'s [B,6,S]
# expand mis-vectorizing on Inductor CPU) shipped past a green suite. The kwargs now come from
# `designs/production_config.json` — the SAME single source `delivery_graph`, the arch viewer
# and the prewarm-era eval workers key on — filtered by the live constructor signature exactly
# like `delivery_graph.build_graph` and `compile_prewarm` do. The `arch_signature` assertion
# makes a stale config FAIL LOUD here instead of quietly testing yesterday's model.
_PRODUCTION_CONFIG = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                  "designs", "production_config.json")


def _build_production_extractor(**overrides):
    """The literal production arch. `overrides` pin a field the live config happens to sit at —
    use it only where a test's SUBJECT is that field, and say why at the call site."""
    import json

    from agents.model.model_version import ARCH_SIGNATURE
    from agents.model.damage_tables import sanitize_historical_move_floor

    with open(_PRODUCTION_CONFIG) as fh:
        cfg = json.load(fh)
    assert cfg.get("arch_signature") == ARCH_SIGNATURE, (
        f"designs/production_config.json is STALE (arch_signature "
        f"{cfg.get('arch_signature')!r} != live {ARCH_SIGNATURE!r}) — this compile gate would "
        f"be testing a dead graph. Refresh the config from the production run's "
        f"model_config.json (the _PRODUCTION_ARCH-dict rot class this assertion exists for)."
    )
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    kw = {a: b for a, b in cfg.items() if a in sig}
    kw.update({a: b for a, b in overrides.items() if a in sig})
    # v65 (gen3_unconditional_move_legality_v1): the committed config is a VERBATIM pre-v65 run copy
    # and records move_candidate_floor 0.0, which the validated range now rejects. Reconcile at
    # construction — the same seam delivery_graph uses — rather than editing the historical record.
    sanitize_historical_move_floor(kw)
    torch.manual_seed(0)
    return Gen3FeaturesExtractor(space, layout=layout, mappings=mappings, **kw).eval(), layout


@_skip_compile
def test_production_arch_compiles_without_suppression():
    """THE regression. The `species_posterior` softmax spelling is what crashed Inductor; with
    suppression OFF a reintroduced bad spelling raises `BackendCompilerFailed` here instead of
    silently costing half the speedup in production."""
    torch.set_num_threads(1)
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = False
    fe, layout = _build_production_extractor()
    obs = {"observation": torch.rand(_BATCH, layout["total_dim"],
                                     generator=torch.Generator().manual_seed(7))}
    with torch.no_grad():
        ref = fe(obs)
        got = torch.compile(fe.forward)(obs)
    err = max(float((a - b).abs().max()) for a, b in zip(ref, got))
    assert err < 1e-5, f"compiled output diverged from eager: {err:.2e}"


@_skip_compile
def test_production_arch_compiles_to_one_graph():
    """Graph breaks are how a 6.5x quietly becomes a 1.2x — the win depends on ONE fused graph."""
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = False
    fe, layout = _build_production_extractor()
    obs = {"observation": torch.zeros(_BATCH, layout["total_dim"])}
    explained = torch._dynamo.explain(fe.forward)(obs)
    assert explained.graph_break_count == 0, explained.break_reasons
    assert explained.graph_count == 1


# ----------------------------------------------------------------- the other three matrix cells


def _obs(layout, device, seed=7):
    g = torch.Generator().manual_seed(seed)
    return {"observation": torch.rand(_BATCH, layout["total_dim"], generator=g).to(device)}


def _backward_through(fe, out):
    """One scalar off BOTH heads, so the backward covers the policy and value halves.

    Deliberately not `pi.sum()`: the root heads end in ReLU and a sum over a normed/rectified
    output can be constant or half-dead, which is how a fixture makes a route look untrained while
    testing nothing (three such degeneracies are catalogued in `belief_label_only_gate_test.py`).
    Squaring keeps every element's gradient live and sign-sensitive.
    """
    pi, vf = out
    (pi.square().mean() + vf.square().mean()).backward()
    return [(n, p) for n, p in fe.named_parameters() if p.grad is not None]


def _assert_real_gradient(fe, grads, where):
    assert len(grads) > 50, f"{where}: only {len(grads)} params got a gradient — the backward is a stub"
    live = [n for n, p in grads if float(p.grad.abs().sum()) > 0]
    assert len(live) > 30, (
        f"{where}: {len(live)} of {len(grads)} params have a NON-ZERO gradient. A compiled backward "
        f"that returns all-zeros satisfies every 'grad is not None' check while training nothing.")
    trunk = [n for n, _ in grads if n.startswith("team_transformer.")]
    assert trunk, f"{where}: no shared-trunk parameter received a gradient at all"


# gen3 test tiers (MEASURED 2026-08-14): 37-59 s, by far the slowest test in the routine gate and
# ~20% of it on its own. It is slow for a structural reason that will not improve — it drives a
# compile that is SUPPOSED to fail, so it pays a full Inductor lowering attempt and then the error
# path, warm cache or not. `slow` is the honest label; it still runs pre-ship and in CI, which is
# where a lifted limitation would be noticed. The sibling compile tests (3.6 s / 14.6 s) stay in
# the routine gate, so "the model stopped compiling" still fails fast.
@pytest.mark.slow
@_skip_compile
def test_cpu_backward_still_does_not_compile():
    """CPU cell 2 — a LIMITATION PIN, not a capability test: the backward does NOT lower on CPU
    and the codebase depends on that being known.

    `maybe_compile_extractor` wraps the compiled callable so that any GRAD-ENABLED call routes to
    eager. That looks like belt-and-braces until you know why it is there: AOTAutograd's CPU
    backward for this model contains a `scatter` with `scatter_mode='atomic_add'` — the backward of
    a gather is a scatter-ADD, since indices may repeat — and Inductor refuses it at
    `torch/_inductor/codegen/cpp.py: assert mode is None`. So the compiled artifact is
    inference-only by necessity, which is also why the prober can still backprop through this same
    extractor for gradient saliency.

    Two things about that refusal are worth knowing before acting on it. It is NOT a CPU-wide gap:
    of Inductor's three C++ store kernels, `CppKernel` and `CppVecKernel` both emit `atomic_add`
    and only `CppTile2DKernel` (the transposed variant, chosen by index LAYOUT) asserts. And WHICH
    of this model's gathers produces it is NOT pinned — detaching the shape-matching candidate
    (`damage_op`'s `w_all.gather(-1, topk_idx)`) left the refusal in place, so there are several.

    ⚠️ **The limitation is CONFIG-CONDITIONAL, and this test pins the config that reaches it.**
    The scatter only has a backward when a gradient flows into the belief heads, so
    `--belief-grad-mode label_only` — which publishes every belief output STOP-GRAD — deletes it
    from the graph and the CPU backward then compiles cleanly. That is not the backend limitation
    lifting; it is this config not containing the offending op. Measured 2026-08-15 when gen-11
    took over `designs/production_config.json` and turned `label_only` on: `shaping` REFUSED,
    `label_only` COMPILED, and `win_prob_mode` made no difference either way. So the override below
    is the SUBJECT of the test, not a convenience — without it this pin silently stops testing
    anything the moment production runs `label_only`, which is exactly what happened.

    THIS TEST FAILS IF THE LIMITATION GENUINELY LIFTS, and that is the point. If a torch upgrade
    fixes the atomic_add lowering, this raises no longer — at which point: the eager route for
    grad-enabled calls stops being load-bearing, CPU training could compile, and the docs in
    `src/agents/training/CLAUDE.md` -> Compiled CPU opponents need updating.

    It also asserts the REASON, not merely that something raised — an unrelated new backend failure
    must not silently satisfy this.
    """
    torch.set_num_threads(1)
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = False
    # `shaping` is the subject: it is the mode under which the atomic_add scatter HAS a backward.
    fe, layout = _build_production_extractor(belief_grad_mode="shaping")
    fe.train()
    try:
        _backward_through(fe, torch.compile(fe.forward)(_obs(layout, "cpu")))
    except Exception as exc:
        # The exception is a BARE `AssertionError` with an EMPTY message (`assert mode is None`,
        # unannotated in torch), so there is nothing to match on in `str(exc)` — the only
        # identifying evidence is the frame it was raised from. Match the TRACEBACK.
        import traceback
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        assert "_inductor/codegen/cpp.py" in tb and "assert mode is None" in tb, (
            "the CPU backward failed for an UNEXPECTED reason — this pin is about the C++ "
            "backend's atomic_add scatter refusal specifically, so investigate rather than "
            f"widening the match:\n{tb[-1500:]}")
        return
    raise AssertionError(
        "the CPU forward+BACKWARD now COMPILES under belief_grad_mode='shaping' — i.e. WITH the "
        "belief scatter's backward in the graph, so this is the real limitation lifting rather "
        "than the config dodging it. That is good news, and it invalidates three things that "
        "currently assume otherwise: (1) `maybe_compile_extractor` routes grad-enabled calls to "
        "eager because AOTAutograd's CPU backward could not lower — that route may now be "
        "unnecessary; (2) `src/agents/training/CLAUDE.md` states the compiled artifact is "
        "inference-only and gives this as the reason; (3) a CPU-side training compile becomes "
        "possible. Re-check all three, then delete this test.")


@_skip_compile
@_skip_cuda
def test_production_arch_compiles_on_cuda():
    """CUDA cell 1. Inductor emits TRITON here, not C++ — a different backend with different bugs,
    so the green CPU cell above is not evidence for this one. Numerics are checked against eager on
    the same device, which is what makes a silently-wrong kernel a failure rather than a speedup."""
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = False
    fe, layout = _build_production_extractor()
    fe = fe.cuda().eval()
    obs = _obs(layout, "cuda")
    with torch.no_grad():
        ref = fe(obs)
        got = torch.compile(fe.forward)(obs)
    err = max(float((a - b).abs().max()) for a, b in zip(ref, got))
    # Looser than the CPU cell's 1e-5: cuDNN/cuBLAS may pick TF32 or a different reduction order for
    # the compiled kernels than for eager. 1e-4 still catches a wrong kernel; it tolerates a
    # differently-ordered correct one.
    assert err < 1e-4, f"compiled CUDA output diverged from eager: {err:.2e}"


@_skip_compile
@_skip_cuda
def test_production_arch_compiles_to_one_graph_on_cuda():
    """Dynamo is device-agnostic in principle, which is exactly why this is worth asserting: a
    device-conditional branch (`if x.is_cuda:`) added anywhere in the forward would break the graph
    HERE and nowhere else, and the CPU cell would stay green."""
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = False
    fe, layout = _build_production_extractor()
    fe = fe.cuda().eval()
    explained = torch._dynamo.explain(fe.forward)(_obs(layout, "cuda"))
    assert explained.graph_break_count == 0, explained.break_reasons
    assert explained.graph_count == 1


@_skip_compile
@_skip_cuda
def test_production_arch_compiles_forward_and_backward_on_cuda():
    """CUDA cell 2 — the TRAINER's actual step, and the one this whole file exists to keep available.

    Measured on this arch at batch 4096: eager 150.85 ms -> compiled 86.21 ms fwd+bwd (1.75x), which
    at the ~89% train share of production wall is worth ~+60% end-to-end FPS. That lever is only
    real while this test passes, and nothing else in the suite would notice it going away — the
    trainer does not compile its forward today, so the loss would be silent until someone tried.
    """
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = False
    fe, layout = _build_production_extractor()
    fe = fe.cuda()
    fe.train()
    grads = _backward_through(fe, torch.compile(fe.forward)(_obs(layout, "cuda")))
    _assert_real_gradient(fe, grads, "cuda fwd+bwd")
